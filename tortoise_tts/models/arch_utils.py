# Adapted from https://github.com/neonbjb/tortoise-tts/tree/98a891e66e7a1f11a830f31bd1ce06cc1f6a88af/tortoise/models/arch_utils.py

import os
import functools
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

from pathlib import Path
from .xtransformers import ContinuousTransformerWrapper, RelativePositionBias

def zero_module(module):
	"""
	Zero out the parameters of a module and return it.
	"""
	for p in module.parameters():
		p.detach().zero_()
	return module


class GroupNorm32(nn.GroupNorm):
	def forward(self, x):
		return super().forward(x.float()).type(x.dtype)


def normalization(channels):
	"""
	Make a standard normalization layer.

	:param channels: number of input channels.
	:return: an nn.Module for normalization.
	"""
	groups = 32
	if channels <= 16:
		groups = 8
	elif channels <= 64:
		groups = 16
	while channels % groups != 0:
		groups = int(groups / 2)
	assert groups > 2
	return GroupNorm32(groups, channels)


AVAILABLE_ATTENTIONS = ["mem_efficient", "math", "sdpa"]

try:
	from xformers.ops import LowerTriangularMask
	from xformers.ops.fmha import memory_efficient_attention

	AVAILABLE_ATTENTIONS.append("xformers")
except Exception as e:
	print("Error while importing `xformers`", e)

# from diffusers.models.attention_processing import AttnProcessor2_0
# to-do: optimize this, as the diffuser *heavily* relies on this
class QKVAttentionLegacy(nn.Module):
	"""
	A module which performs QKV attention. Matches legacy QKVAttention + input/output heads shaping
	"""

	def __init__(self, n_heads):
		super().__init__()
		self.n_heads = n_heads

	def forward(self, qkv, mask=None, rel_pos=None):
		"""
		Apply QKV attention.

		:param qkv: an [N x (H * 3 * C) x T] tensor of Qs, Ks, and Vs.
		:return: an [N x (H * C) x T] tensor after attention.
		"""

		bs, width, length = qkv.shape
		assert width % (3 * self.n_heads) == 0
		ch = width // (3 * self.n_heads)
		q, k, v = qkv.reshape(bs * self.n_heads, ch * 3, length).split(ch, dim=1)
		scale = 1 / math.sqrt(math.sqrt(ch))
		weight = torch.einsum(
			"bct,bcs->bts", q * scale, k * scale
		)  # More stable with f16 than dividing afterwards
		if rel_pos is not None:
			weight = rel_pos(weight.reshape(bs, self.n_heads, weight.shape[-2], weight.shape[-1])).reshape(bs * self.n_heads, weight.shape[-2], weight.shape[-1])
		weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
		if mask is not None:
			# The proper way to do this is to mask before the softmax using -inf, but that doesn't work properly on CPUs.
			mask = mask.repeat(self.n_heads, 1).unsqueeze(1)
			weight = weight * mask
			
		a = torch.einsum("bts,bcs->bct", weight, v)

		return a.reshape(bs, -1, length)

class QKVAttention(nn.Module):
	"""
	A module which performs QKV attention and splits in a different order.
	"""

	def __init__(self, n_heads):
		super().__init__()
		self.n_heads = n_heads

	def forward(self, qkv, mask=None, rel_pos=None):
		"""
		Apply QKV attention.
		:param qkv: an [N x (3 * H * C) x T] tensor of Qs, Ks, and Vs.
		:return: an [N x (H * C) x T] tensor after attention.
		"""
		bs, width, length = qkv.shape
		assert width % (3 * self.n_heads) == 0
		ch = width // (3 * self.n_heads)
		q, k, v = qkv.chunk(3, dim=1)
		scale = 1 / math.sqrt(math.sqrt(ch))
		weight = torch.einsum(
			"bct,bcs->bts",
			(q * scale).view(bs * self.n_heads, ch, length),
			(k * scale).view(bs * self.n_heads, ch, length),
		)  # More stable with f16 than dividing afterwards
		if rel_pos is not None:
			weight = rel_pos(weight.reshape(bs, self.n_heads, weight.shape[-2], weight.shape[-1])).reshape(bs * self.n_heads, weight.shape[-2], weight.shape[-1])
		weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
		if mask is not None:
			# The proper way to do this is to mask before the softmax using -inf, but that doesn't work properly on CPUs.
			mask = mask.repeat(self.n_heads, 1).unsqueeze(1)
			weight = weight * mask
		a = torch.einsum("bts,bcs->bct", weight, v.reshape(bs * self.n_heads, ch, length))
		return a.reshape(bs, -1, length)

	@staticmethod
	def count_flops(model, _x, y):
		return count_flops_attn(model, _x, y)

# actually sourced from https://github.com/CompVis/stable-diffusion/blob/main/ldm/modules/diffusionmodules/openaimodel.py#L278
class AttentionBlock(nn.Module):
	"""
	An attention block that allows spatial positions to attend to each other.

	Originally ported from here, but adapted to the N-d case.
	https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.
	"""

	def __init__(
		self,
		channels,
		num_heads=1,
		num_head_channels=-1,
		use_checkpoint=False,
		use_new_attention_order=False,
		relative_pos_embeddings=False,
	):
		super().__init__()
		self.channels = channels
		if num_head_channels == -1:
			self.num_heads = num_heads
		else:
			assert (
				channels % num_head_channels == 0
			), f"q,k,v channels {channels} is not divisible by num_head_channels {num_head_channels}"
			self.num_heads = channels // num_head_channels
		self.use_checkpoint = use_checkpoint
		self.norm = normalization(channels)
		self.qkv = nn.Conv1d(channels, channels * 3, 1)
		if use_new_attention_order:
			# split qkv before split heads
			self.attention = QKVAttention(self.num_heads)
		else:
			# split heads before split qkv
			self.attention = QKVAttentionLegacy(self.num_heads)

		self.proj_out = zero_module(nn.Conv1d(channels, channels, 1))
		if relative_pos_embeddings:
			self.relative_pos_embeddings = RelativePositionBias(scale=(channels // self.num_heads) ** .5, causal=False, heads=num_heads, num_buckets=32, max_distance=64)
		else:
			self.relative_pos_embeddings = None

	def forward(self, x):
		if self.use_checkpoint:
			return checkpoint(self._forward, (x,), self.parameters(), True)
		return self._forward(x)

	def _forward(self, x, mask=None):
		b, c, *spatial = x.shape
		x = x.reshape(b, c, -1)
		qkv = self.qkv(self.norm(x))
		#h = self.attention(qkv)
		h = self.attention(qkv, mask, self.relative_pos_embeddings)
		h = self.proj_out(h)
		return (x + h).reshape(b, c, *spatial)


class Upsample(nn.Module):
	"""
	An upsampling layer with an optional convolution.

	:param channels: channels in the inputs and outputs.
	:param use_conv: a bool determining if a convolution is applied.
	"""

	def __init__(self, channels, use_conv, out_channels=None, factor=4):
		super().__init__()
		self.channels = channels
		self.out_channels = out_channels or channels
		self.use_conv = use_conv
		self.factor = factor
		if use_conv:
			ksize = 5
			pad = 2
			self.conv = nn.Conv1d(self.channels, self.out_channels, ksize, padding=pad)

	def forward(self, x):
		assert x.shape[1] == self.channels
		x = F.interpolate(x, scale_factor=self.factor, mode="nearest")
		if self.use_conv:
			x = self.conv(x)
		return x


class Downsample(nn.Module):
	"""
	A downsampling layer with an optional convolution.

	:param channels: channels in the inputs and outputs.
	:param use_conv: a bool determining if a convolution is applied.
	"""

	def __init__(self, channels, use_conv, out_channels=None, factor=4, ksize=5, pad=2):
		super().__init__()
		self.channels = channels
		self.out_channels = out_channels or channels
		self.use_conv = use_conv

		stride = factor
		if use_conv:
			self.op = nn.Conv1d(
				self.channels, self.out_channels, ksize, stride=stride, padding=pad
			)
		else:
			assert self.channels == self.out_channels
			self.op = nn.AvgPool1d(kernel_size=stride, stride=stride)

	def forward(self, x):
		assert x.shape[1] == self.channels
		return self.op(x)


class ResBlock(nn.Module):
	def __init__(
			self,
			channels,
			dropout,
			out_channels=None,
			use_conv=False,
			use_scale_shift_norm=False,
			up=False,
			down=False,
			kernel_size=3,
	):
		super().__init__()
		self.channels = channels
		self.dropout = dropout
		self.out_channels = out_channels or channels
		self.use_conv = use_conv
		self.use_scale_shift_norm = use_scale_shift_norm
		padding = 1 if kernel_size == 3 else 2

		self.in_layers = nn.Sequential(
			normalization(channels),
			nn.SiLU(),
			nn.Conv1d(channels, self.out_channels, kernel_size, padding=padding),
		)

		self.updown = up or down

		if up:
			self.h_upd = Upsample(channels, False)
			self.x_upd = Upsample(channels, False)
		elif down:
			self.h_upd = Downsample(channels, False)
			self.x_upd = Downsample(channels, False)
		else:
			self.h_upd = self.x_upd = nn.Identity()

		self.out_layers = nn.Sequential(
			normalization(self.out_channels),
			nn.SiLU(),
			nn.Dropout(p=dropout),
			zero_module(
				nn.Conv1d(self.out_channels, self.out_channels, kernel_size, padding=padding)
			),
		)

		if self.out_channels == channels:
			self.skip_connection = nn.Identity()
		elif use_conv:
			self.skip_connection = nn.Conv1d(
				channels, self.out_channels, kernel_size, padding=padding
			)
		else:
			self.skip_connection = nn.Conv1d(channels, self.out_channels, 1)

	def forward(self, x):
		if self.updown:
			in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
			h = in_rest(x)
			h = self.h_upd(h)
			x = self.x_upd(x)
			h = in_conv(h)
		else:
			h = self.in_layers(x)
		h = self.out_layers(h)
		return self.skip_connection(x) + h


class AudioMiniEncoder(nn.Module):
	def __init__(self,
				 spec_dim,
				 embedding_dim,
				 base_channels=128,
				 depth=2,
				 resnet_blocks=2,
				 attn_blocks=4,
				 num_attn_heads=4,
				 dropout=0,
				 downsample_factor=2,
				 kernel_size=3):
		super().__init__()
		self.init = nn.Sequential(
			nn.Conv1d(spec_dim, base_channels, 3, padding=1)
		)
		ch = base_channels
		res = []
		for l in range(depth):
			for r in range(resnet_blocks):
				res.append(ResBlock(ch, dropout, kernel_size=kernel_size))
			res.append(Downsample(ch, use_conv=True, out_channels=ch*2, factor=downsample_factor))
			ch *= 2
		self.res = nn.Sequential(*res)
		self.final = nn.Sequential(
			normalization(ch),
			nn.SiLU(),
			nn.Conv1d(ch, embedding_dim, 1)
		)
		attn = []
		for a in range(attn_blocks):
			attn.append(AttentionBlock(embedding_dim, num_attn_heads,))
		self.attn = nn.Sequential(*attn)
		self.dim = embedding_dim

	def forward(self, x):
		h = self.init(x)
		h = self.res(h)
		h = self.final(h)
		h = self.attn(h)
		return h[:, :, 0]


DEFAULT_MEL_NORM_FILE = Path(__file__).parent.parent.parent / 'data/models/mel_norms.pth'

class TorchMelSpectrogram(nn.Module):
	def __init__(self, filter_length=1024, hop_length=256, win_length=1024, n_mel_channels=80, mel_fmin=0, mel_fmax=8000,
				 sampling_rate=22050, normalize=False, mel_norm_file=DEFAULT_MEL_NORM_FILE):
		super().__init__()
		# These are the default tacotron values for the MEL spectrogram.
		self.filter_length = filter_length
		self.hop_length = hop_length
		self.win_length = win_length
		self.n_mel_channels = n_mel_channels
		self.mel_fmin = mel_fmin
		self.mel_fmax = mel_fmax
		self.sampling_rate = sampling_rate
		self.mel_stft = torchaudio.transforms.MelSpectrogram(n_fft=self.filter_length, hop_length=self.hop_length,
															 win_length=self.win_length, power=2, normalized=normalize,
															 sample_rate=self.sampling_rate, f_min=self.mel_fmin,
															 f_max=self.mel_fmax, n_mels=self.n_mel_channels,
															 norm="slaney")
		self.mel_norm_file = mel_norm_file
		if self.mel_norm_file is not None and self.mel_norm_file.exists():
			self.mel_norms = torch.load(self.mel_norm_file)
		else:
			self.mel_norms = None

	def forward(self, inp):
		if len(inp.shape) == 3:  # Automatically squeeze out the channels dimension if it is present (assuming mono-audio)
			inp = inp.squeeze(1)
		assert len(inp.shape) == 2
		self.mel_stft = self.mel_stft.to(inp.device)
		mel = self.mel_stft(inp)
		# Perform dynamic range compression
		mel = torch.log(torch.clamp(mel, min=1e-5))
		if self.mel_norms is not None:
			self.mel_norms = self.mel_norms.to(mel.device)
			mel = mel / self.mel_norms.unsqueeze(0).unsqueeze(-1)
		return mel


class CheckpointedLayer(nn.Module):
	"""
	Wraps a module. When forward() is called, passes kwargs that require_grad through torch.checkpoint() and bypasses
	checkpoint for all other args.
	"""
	def __init__(self, wrap):
		super().__init__()
		self.wrap = wrap

	def forward(self, x, *args, **kwargs):
		for k, v in kwargs.items():
			assert not (isinstance(v, torch.Tensor) and v.requires_grad)  # This would screw up checkpointing.
		partial = functools.partial(self.wrap, **kwargs)
		return partial(x, *args)


class CheckpointedXTransformerEncoder(nn.Module):
	"""
	Wraps a ContinuousTransformerWrapper and applies CheckpointedLayer to each layer and permutes from channels-mid
	to channels-last that XTransformer expects.
	"""
	def __init__(self, needs_permute=True, exit_permute=True, checkpoint=True, **xtransformer_kwargs):
		super().__init__()
		self.transformer = ContinuousTransformerWrapper(**xtransformer_kwargs)
		self.needs_permute = needs_permute
		self.exit_permute = exit_permute

		if not checkpoint:
			return
		for i in range(len(self.transformer.attn_layers.layers)):
			n, b, r = self.transformer.attn_layers.layers[i]
			self.transformer.attn_layers.layers[i] = nn.ModuleList([n, CheckpointedLayer(b), r])

	def forward(self, x, **kwargs):
		if self.needs_permute:
			x = x.permute(0,2,1)
		h = self.transformer(x, **kwargs)
		if self.exit_permute:
			h = h.permute(0,2,1)
		return h

"""
BSD 3-Clause License

Copyright (c) 2017, Prem Seetharaman
All rights reserved.

* Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice,
  this list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice, this
  list of conditions and the following disclaimer in the
  documentation and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from this
  software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from scipy.signal import get_window
from librosa.util import pad_center, tiny
import librosa.util as librosa_util

def window_sumsquare(window, n_frames, hop_length=200, win_length=800,
					 n_fft=800, dtype=np.float32, norm=None):
	"""
	# from librosa 0.6
	Compute the sum-square envelope of a window function at a given hop length.

	This is used to estimate modulation effects induced by windowing
	observations in short-time fourier transforms.

	Parameters
	----------
	window : string, tuple, number, callable, or list-like
		Window specification, as in `get_window`

	n_frames : int > 0
		The number of analysis frames

	hop_length : int > 0
		The number of samples to advance between frames

	win_length : [optional]
		The length of the window function.  By default, this matches `n_fft`.

	n_fft : int > 0
		The length of each analysis frame.

	dtype : np.dtype
		The data type of the output

	Returns
	-------
	wss : np.ndarray, shape=`(n_fft + hop_length * (n_frames - 1))`
		The sum-squared envelope of the window function
	"""
	if win_length is None:
		win_length = n_fft

	n = n_fft + hop_length * (n_frames - 1)
	x = np.zeros(n, dtype=dtype)

	# Compute the squared window at the desired length
	win_sq = get_window(window, win_length, fftbins=True)
	win_sq = librosa_util.normalize(win_sq, norm=norm)**2
	win_sq = librosa_util.pad_center(win_sq, n_fft)

	# Fill the envelope
	for i in range(n_frames):
		sample = i * hop_length
		x[sample:min(n, sample + n_fft)] += win_sq[:max(0, min(n_fft, n - sample))]
	return x

TACOTRON_MEL_MAX = 2.3143386840820312
TACOTRON_MEL_MIN = -11.512925148010254


def denormalize_tacotron_mel(norm_mel):
	return ((norm_mel+1)/2)*(TACOTRON_MEL_MAX-TACOTRON_MEL_MIN)+TACOTRON_MEL_MIN


def normalize_tacotron_mel(mel):
	return 2 * ((mel - TACOTRON_MEL_MIN) / (TACOTRON_MEL_MAX - TACOTRON_MEL_MIN)) - 1


def dynamic_range_compression(x, C=1, clip_val=1e-5):
	"""
	PARAMS
	------
	C: compression factor
	"""
	return torch.log(torch.clamp(x, min=clip_val) * C)


def dynamic_range_decompression(x, C=1):
	"""
	PARAMS
	------
	C: compression factor used to compress
	"""
	return torch.exp(x) / C

class STFT(torch.nn.Module):
	"""adapted from Prem Seetharaman's https://github.com/pseeth/pytorch-stft"""
	def __init__(self, filter_length=800, hop_length=200, win_length=800,
				 window='hann'):
		super(STFT, self).__init__()
		self.filter_length = filter_length
		self.hop_length = hop_length
		self.win_length = win_length
		self.window = window
		self.forward_transform = None
		scale = self.filter_length / self.hop_length
		fourier_basis = np.fft.fft(np.eye(self.filter_length))

		cutoff = int((self.filter_length / 2 + 1))
		fourier_basis = np.vstack([np.real(fourier_basis[:cutoff, :]),
								   np.imag(fourier_basis[:cutoff, :])])

		forward_basis = torch.FloatTensor(fourier_basis[:, None, :])
		inverse_basis = torch.FloatTensor(
			np.linalg.pinv(scale * fourier_basis).T[:, None, :])

		if window is not None:
			assert(filter_length >= win_length)
			# get window and zero center pad it to filter_length
			fft_window = get_window(window, win_length, fftbins=True)
			fft_window = pad_center(fft_window, size=filter_length)
			fft_window = torch.from_numpy(fft_window).float()

			# window the bases
			forward_basis *= fft_window
			inverse_basis *= fft_window

		self.register_buffer('forward_basis', forward_basis.float())
		self.register_buffer('inverse_basis', inverse_basis.float())

	def transform(self, input_data):
		num_batches = input_data.size(0)
		num_samples = input_data.size(1)

		self.num_samples = num_samples

		# similar to librosa, reflect-pad the input
		input_data = input_data.view(num_batches, 1, num_samples)
		input_data = F.pad(
			input_data.unsqueeze(1),
			(int(self.filter_length / 2), int(self.filter_length / 2), 0, 0),
			mode='reflect')
		input_data = input_data.squeeze(1)

		forward_transform = F.conv1d(
			input_data,
			Variable(self.forward_basis, requires_grad=False),
			stride=self.hop_length,
			padding=0)

		cutoff = int((self.filter_length / 2) + 1)
		real_part = forward_transform[:, :cutoff, :]
		imag_part = forward_transform[:, cutoff:, :]

		magnitude = torch.sqrt(real_part**2 + imag_part**2)
		phase = torch.autograd.Variable(
			torch.atan2(imag_part.data, real_part.data))

		return magnitude, phase

	def inverse(self, magnitude, phase):
		recombine_magnitude_phase = torch.cat(
			[magnitude*torch.cos(phase), magnitude*torch.sin(phase)], dim=1)

		inverse_transform = F.conv_transpose1d(
			recombine_magnitude_phase,
			Variable(self.inverse_basis, requires_grad=False),
			stride=self.hop_length,
			padding=0)

		if self.window is not None:
			window_sum = window_sumsquare(
				self.window, magnitude.size(-1), hop_length=self.hop_length,
				win_length=self.win_length, n_fft=self.filter_length,
				dtype=np.float32)
			# remove modulation effects
			approx_nonzero_indices = torch.from_numpy(
				np.where(window_sum > tiny(window_sum))[0])
			window_sum = torch.autograd.Variable(
				torch.from_numpy(window_sum), requires_grad=False)
			window_sum = window_sum.cuda() if magnitude.is_cuda else window_sum
			inverse_transform[:, :, approx_nonzero_indices] /= window_sum[approx_nonzero_indices]

			# scale by hop ratio
			inverse_transform *= float(self.filter_length) / self.hop_length

		inverse_transform = inverse_transform[:, :, int(self.filter_length/2):]
		inverse_transform = inverse_transform[:, :, :-int(self.filter_length/2):]

		return inverse_transform

	def forward(self, input_data):
		self.magnitude, self.phase = self.transform(input_data)
		reconstruction = self.inverse(self.magnitude, self.phase)
		return reconstruction

class TacotronSTFT(torch.nn.Module):
	def __init__(
		self,
		filter_length=1024,
		hop_length=256,
		win_length=1024,
		n_mel_channels=80,
		sampling_rate=22050,
		mel_fmin=0.0,
		mel_fmax=8000.0
	):
		super().__init__()
		self.n_mel_channels = n_mel_channels
		self.sampling_rate = sampling_rate
		self.stft_fn = STFT(filter_length, hop_length, win_length)
		from librosa.filters import mel as librosa_mel_fn
		mel_basis = librosa_mel_fn(
			sr=sampling_rate, n_fft=filter_length, n_mels=n_mel_channels, fmin=mel_fmin, fmax=mel_fmax)
		mel_basis = torch.from_numpy(mel_basis).float()
		self.register_buffer('mel_basis', mel_basis)

	def spectral_normalize(self, magnitudes):
		output = dynamic_range_compression(magnitudes)
		return output

	def spectral_de_normalize(self, magnitudes):
		output = dynamic_range_decompression(magnitudes)
		return output

	def mel_spectrogram(self, y):
		assert(torch.min(y.data) >= -10)
		assert(torch.max(y.data) <= 10)
		y = torch.clip(y, min=-1, max=1)

		magnitudes, phases = self.stft_fn.transform(y)
		magnitudes = magnitudes.data
		mel_output = torch.matmul(self.mel_basis, magnitudes)
		mel_output = self.spectral_normalize(mel_output)
		return mel_output