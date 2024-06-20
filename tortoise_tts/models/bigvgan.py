import torch
import torch.nn as nn
import torch.nn.functional as F

import math

import json
import os
import torch.utils.data

from torch import nn, sin, pow
from torch.nn import Conv1d, ConvTranspose1d, Conv2d, Parameter
from torch.nn.utils import weight_norm, remove_weight_norm, spectral_norm
from librosa.filters import mel as librosa_mel_fn


# filter.py
# Adapted from https://github.com/junjun3518/alias-free-torch under the Apache License 2.0
#   LICENSE is in incl_licenses directory.

if 'sinc' in dir(torch):
	sinc = torch.sinc
else:
	# This code is adopted from adefossez's julius.core.sinc under the MIT License
	# https://adefossez.github.io/julius/julius/core.html
	#   LICENSE is in incl_licenses directory.
	def sinc(x: torch.Tensor):
		"""
		Implementation of sinc, i.e. sin(pi * x) / (pi * x)
		__Warning__: Different to julius.sinc, the input is multiplied by `pi`!
		"""
		return torch.where(x == 0,
						   torch.tensor(1., device=x.device, dtype=x.dtype),
						   torch.sin(math.pi * x) / math.pi / x)


# This code is adopted from adefossez's julius.lowpass.LowPassFilters under the MIT License
# https://adefossez.github.io/julius/julius/lowpass.html
#   LICENSE is in incl_licenses directory.
def kaiser_sinc_filter1d(cutoff, half_width, kernel_size): # return filter [1,1,kernel_size]
	even = (kernel_size % 2 == 0)
	half_size = kernel_size // 2

	#For kaiser window
	delta_f = 4 * half_width
	A = 2.285 * (half_size - 1) * math.pi * delta_f + 7.95
	if A > 50.:
		beta = 0.1102 * (A - 8.7)
	elif A >= 21.:
		beta = 0.5842 * (A - 21)**0.4 + 0.07886 * (A - 21.)
	else:
		beta = 0.
	window = torch.kaiser_window(kernel_size, beta=beta, periodic=False)

	# ratio = 0.5/cutoff -> 2 * cutoff = 1 / ratio
	if even:
		time = (torch.arange(-half_size, half_size) + 0.5)
	else:
		time = torch.arange(kernel_size) - half_size
	if cutoff == 0:
		filter_ = torch.zeros_like(time)
	else:
		filter_ = 2 * cutoff * window * sinc(2 * cutoff * time)
		# Normalize filter to have sum = 1, otherwise we will have a small leakage
		# of the constant component in the input signal.
		filter_ /= filter_.sum()
		filter = filter_.view(1, 1, kernel_size)

	return filter


class LowPassFilter1d(nn.Module):
	def __init__(self,
				 cutoff=0.5,
				 half_width=0.6,
				 stride: int = 1,
				 padding: bool = True,
				 padding_mode: str = 'replicate',
				 kernel_size: int = 12):
		# kernel_size should be even number for stylegan3 setup,
		# in this implementation, odd number is also possible.
		super().__init__()
		if cutoff < -0.:
			raise ValueError("Minimum cutoff must be larger than zero.")
		if cutoff > 0.5:
			raise ValueError("A cutoff above 0.5 does not make sense.")
		self.kernel_size = kernel_size
		self.even = (kernel_size % 2 == 0)
		self.pad_left = kernel_size // 2 - int(self.even)
		self.pad_right = kernel_size // 2
		self.stride = stride
		self.padding = padding
		self.padding_mode = padding_mode
		filter = kaiser_sinc_filter1d(cutoff, half_width, kernel_size)
		self.register_buffer("filter", filter)

	#input [B, C, T]
	def forward(self, x):
		_, C, _ = x.shape

		if self.padding:
			x = F.pad(x, (self.pad_left, self.pad_right),
					  mode=self.padding_mode)
		out = F.conv1d(x, self.filter.expand(C, -1, -1),
					   stride=self.stride, groups=C)

		return out

# resample.py
# Adapted from https://github.com/junjun3518/alias-free-torch under the Apache License 2.0
#   LICENSE is in incl_licenses directory.

class UpSample1d(nn.Module):
	def __init__(self, ratio=2, kernel_size=None):
		super().__init__()
		self.ratio = ratio
		self.kernel_size = int(6 * ratio // 2) * 2 if kernel_size is None else kernel_size
		self.stride = ratio
		self.pad = self.kernel_size // ratio - 1
		self.pad_left = self.pad * self.stride + (self.kernel_size - self.stride) // 2
		self.pad_right = self.pad * self.stride + (self.kernel_size - self.stride + 1) // 2
		filter = kaiser_sinc_filter1d(cutoff=0.5 / ratio,
									  half_width=0.6 / ratio,
									  kernel_size=self.kernel_size)
		self.register_buffer("filter", filter)

	# x: [B, C, T]
	def forward(self, x):
		_, C, _ = x.shape

		x = F.pad(x, (self.pad, self.pad), mode='replicate')
		x = self.ratio * F.conv_transpose1d(
			x, self.filter.expand(C, -1, -1), stride=self.stride, groups=C)
		x = x[..., self.pad_left:-self.pad_right]

		return x


class DownSample1d(nn.Module):
	def __init__(self, ratio=2, kernel_size=None):
		super().__init__()
		self.ratio = ratio
		self.kernel_size = int(6 * ratio // 2) * 2 if kernel_size is None else kernel_size
		self.lowpass = LowPassFilter1d(cutoff=0.5 / ratio,
									   half_width=0.6 / ratio,
									   stride=ratio,
									   kernel_size=self.kernel_size)

	def forward(self, x):
		xx = self.lowpass(x)

		return xx

# act.py
# Adapted from https://github.com/junjun3518/alias-free-torch under the Apache License 2.0
#   LICENSE is in incl_licenses directory.

class Activation1d(nn.Module):
	def __init__(self,
				 activation,
				 up_ratio: int = 2,
				 down_ratio: int = 2,
				 up_kernel_size: int = 12,
				 down_kernel_size: int = 12):
		super().__init__()
		self.up_ratio = up_ratio
		self.down_ratio = down_ratio
		self.act = activation
		self.upsample = UpSample1d(up_ratio, up_kernel_size)
		self.downsample = DownSample1d(down_ratio, down_kernel_size)

	# x: [B,C,T]
	def forward(self, x):
		x = self.upsample(x)
		x = self.act(x)
		x = self.downsample(x)

		return x

# activations.py
# Implementation adapted from https://github.com/EdwardDixon/snake under the MIT license.
#   LICENSE is in incl_licenses directory.

class Snake(nn.Module):
	'''
	Implementation of a sine-based periodic activation function
	Shape:
		- Input: (B, C, T)
		- Output: (B, C, T), same shape as the input
	Parameters:
		- alpha - trainable parameter
	References:
		- This activation function is from this paper by Liu Ziyin, Tilman Hartwig, Masahito Ueda:
		https://arxiv.org/abs/2006.08195
	Examples:
		>>> a1 = snake(256)
		>>> x = torch.randn(256)
		>>> x = a1(x)
	'''
	def __init__(self, in_features, alpha=1.0, alpha_trainable=True, alpha_logscale=False):
		'''
		Initialization.
		INPUT:
			- in_features: shape of the input
			- alpha: trainable parameter
			alpha is initialized to 1 by default, higher values = higher-frequency.
			alpha will be trained along with the rest of your model.
		'''
		super(Snake, self).__init__()
		self.in_features = in_features

		# initialize alpha
		self.alpha_logscale = alpha_logscale
		if self.alpha_logscale: # log scale alphas initialized to zeros
			self.alpha = Parameter(torch.zeros(in_features) * alpha)
		else: # linear scale alphas initialized to ones
			self.alpha = Parameter(torch.ones(in_features) * alpha)

		self.alpha.requires_grad = alpha_trainable

		self.no_div_by_zero = 0.000000001

	def forward(self, x):
		'''
		Forward pass of the function.
		Applies the function to the input elementwise.
		Snake ∶= x + 1/a * sin^2 (xa)
		'''
		alpha = self.alpha.unsqueeze(0).unsqueeze(-1) # line up with x to [B, C, T]
		if self.alpha_logscale:
			alpha = torch.exp(alpha)
		x = x + (1.0 / (alpha + self.no_div_by_zero)) * pow(sin(x * alpha), 2)

		return x


class SnakeBeta(nn.Module):
	'''
	A modified Snake function which uses separate parameters for the magnitude of the periodic components
	Shape:
		- Input: (B, C, T)
		- Output: (B, C, T), same shape as the input
	Parameters:
		- alpha - trainable parameter that controls frequency
		- beta - trainable parameter that controls magnitude
	References:
		- This activation function is a modified version based on this paper by Liu Ziyin, Tilman Hartwig, Masahito Ueda:
		https://arxiv.org/abs/2006.08195
	Examples:
		>>> a1 = snakebeta(256)
		>>> x = torch.randn(256)
		>>> x = a1(x)
	'''
	def __init__(self, in_features, alpha=1.0, alpha_trainable=True, alpha_logscale=False):
		'''
		Initialization.
		INPUT:
			- in_features: shape of the input
			- alpha - trainable parameter that controls frequency
			- beta - trainable parameter that controls magnitude
			alpha is initialized to 1 by default, higher values = higher-frequency.
			beta is initialized to 1 by default, higher values = higher-magnitude.
			alpha will be trained along with the rest of your model.
		'''
		super(SnakeBeta, self).__init__()
		self.in_features = in_features

		# initialize alpha
		self.alpha_logscale = alpha_logscale
		if self.alpha_logscale: # log scale alphas initialized to zeros
			self.alpha = Parameter(torch.zeros(in_features) * alpha)
			self.beta = Parameter(torch.zeros(in_features) * alpha)
		else: # linear scale alphas initialized to ones
			self.alpha = Parameter(torch.ones(in_features) * alpha)
			self.beta = Parameter(torch.ones(in_features) * alpha)

		self.alpha.requires_grad = alpha_trainable
		self.beta.requires_grad = alpha_trainable

		self.no_div_by_zero = 0.000000001

	def forward(self, x):
		'''
		Forward pass of the function.
		Applies the function to the input elementwise.
		SnakeBeta ∶= x + 1/b * sin^2 (xa)
		'''
		alpha = self.alpha.unsqueeze(0).unsqueeze(-1) # line up with x to [B, C, T]
		beta = self.beta.unsqueeze(0).unsqueeze(-1)
		if self.alpha_logscale:
			alpha = torch.exp(alpha)
			beta = torch.exp(beta)
		x = x + (1.0 / (beta + self.no_div_by_zero)) * pow(sin(x * alpha), 2)

		return x

# bigvgan.py
# Copyright (c) 2022 NVIDIA CORPORATION. 
#   Licensed under the MIT license.

# Adapted from https://github.com/jik876/hifi-gan under the MIT license.
#   LICENSE is in incl_licenses directory.

LRELU_SLOPE = 0.1

class AMPBlock1(torch.nn.Module):
	def __init__(self, h, channels, kernel_size=3, dilation=(1, 3, 5), activation=None):
		super(AMPBlock1, self).__init__()
		self.h = h

		self.convs1 = nn.ModuleList([
			weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[0],
							   padding=get_padding(kernel_size, dilation[0]))),
			weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[1],
							   padding=get_padding(kernel_size, dilation[1]))),
			weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[2],
							   padding=get_padding(kernel_size, dilation[2])))
		])
		self.convs1.apply(init_weights)

		self.convs2 = nn.ModuleList([
			weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1,
							   padding=get_padding(kernel_size, 1))),
			weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1,
							   padding=get_padding(kernel_size, 1))),
			weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1,
							   padding=get_padding(kernel_size, 1)))
		])
		self.convs2.apply(init_weights)

		self.num_layers = len(self.convs1) + len(self.convs2)  # total number of conv layers

		if activation == 'snake':  # periodic nonlinearity with snake function and anti-aliasing
			self.activations = nn.ModuleList([
				Activation1d(
					activation=Snake(channels, alpha_logscale=h.snake_logscale))
				for _ in range(self.num_layers)
			])
		elif activation == 'snakebeta':  # periodic nonlinearity with snakebeta function and anti-aliasing
			self.activations = nn.ModuleList([
				Activation1d(
					activation=SnakeBeta(channels, alpha_logscale=h.snake_logscale))
				for _ in range(self.num_layers)
			])
		else:
			raise NotImplementedError(
				"activation incorrectly specified. check the config file and look for 'activation'.")

	def forward(self, x):
		acts1, acts2 = self.activations[::2], self.activations[1::2]
		for c1, c2, a1, a2 in zip(self.convs1, self.convs2, acts1, acts2):
			xt = a1(x)
			xt = c1(xt)
			xt = a2(xt)
			xt = c2(xt)
			x = xt + x

		return x

	def remove_weight_norm(self):
		for l in self.convs1:
			remove_weight_norm(l)
		for l in self.convs2:
			remove_weight_norm(l)


class AMPBlock2(torch.nn.Module):
	def __init__(self, h, channels, kernel_size=3, dilation=(1, 3), activation=None):
		super(AMPBlock2, self).__init__()
		self.h = h

		self.convs = nn.ModuleList([
			weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[0],
							   padding=get_padding(kernel_size, dilation[0]))),
			weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[1],
							   padding=get_padding(kernel_size, dilation[1])))
		])
		self.convs.apply(init_weights)

		self.num_layers = len(self.convs)  # total number of conv layers

		if activation == 'snake':  # periodic nonlinearity with snake function and anti-aliasing
			self.activations = nn.ModuleList([
				Activation1d(
					activation=Snake(channels, alpha_logscale=h.snake_logscale))
				for _ in range(self.num_layers)
			])
		elif activation == 'snakebeta':  # periodic nonlinearity with snakebeta function and anti-aliasing
			self.activations = nn.ModuleList([
				Activation1d(
					activation=SnakeBeta(channels, alpha_logscale=h.snake_logscale))
				for _ in range(self.num_layers)
			])
		else:
			raise NotImplementedError(
				"activation incorrectly specified. check the config file and look for 'activation'.")

	def forward(self, x):
		for c, a in zip(self.convs, self.activations):
			xt = a(x)
			xt = c(xt)
			x = xt + x

		return x

	def remove_weight_norm(self):
		for l in self.convs:
			remove_weight_norm(l)



class AttrDict(dict):
	def __init__(self, *args, **kwargs):
		super(AttrDict, self).__init__(*args, **kwargs)
		self.__dict__ = self

class BigVGAN(nn.Module):
	# this is our main BigVGAN model. Applies anti-aliased periodic activation for resblocks.
	def __init__(self, config=None, data=None):
		super(BigVGAN, self).__init__()

		"""
		with open(os.path.join(os.path.dirname(__file__), 'config.json'), 'r') as f:
			data = f.read()
		"""
		if config and data is None:
			with open(config, 'r') as f:
				data = f.read()
			jsonConfig = json.loads(data)
		elif data is not None:
			if isinstance(data, str):
				jsonConfig = json.loads(data)
			else:
				jsonConfig = data
		else:
			raise Exception("no config specified")


		global h
		h = AttrDict(jsonConfig)

		self.mel_channel = h.num_mels
		self.noise_dim = h.n_fft
		self.hop_length = h.hop_size
		self.num_kernels = len(h.resblock_kernel_sizes)
		self.num_upsamples = len(h.upsample_rates)

		# pre conv
		self.conv_pre = weight_norm(Conv1d(h.num_mels, h.upsample_initial_channel, 7, 1, padding=3))

		# define which AMPBlock to use. BigVGAN uses AMPBlock1 as default
		resblock = AMPBlock1 if h.resblock == '1' else AMPBlock2

		# transposed conv-based upsamplers. does not apply anti-aliasing
		self.ups = nn.ModuleList()
		for i, (u, k) in enumerate(zip(h.upsample_rates, h.upsample_kernel_sizes)):
			self.ups.append(nn.ModuleList([
				weight_norm(ConvTranspose1d(h.upsample_initial_channel // (2 ** i),
											h.upsample_initial_channel // (2 ** (i + 1)),
											k, u, padding=(k - u) // 2))
			]))

		# residual blocks using anti-aliased multi-periodicity composition modules (AMP)
		self.resblocks = nn.ModuleList()
		for i in range(len(self.ups)):
			ch = h.upsample_initial_channel // (2 ** (i + 1))
			for j, (k, d) in enumerate(zip(h.resblock_kernel_sizes, h.resblock_dilation_sizes)):
				self.resblocks.append(resblock(h, ch, k, d, activation=h.activation))

		# post conv
		if h.activation == "snake":  # periodic nonlinearity with snake function and anti-aliasing
			activation_post = Snake(ch, alpha_logscale=h.snake_logscale)
			self.activation_post = Activation1d(activation=activation_post)
		elif h.activation == "snakebeta":  # periodic nonlinearity with snakebeta function and anti-aliasing
			activation_post = SnakeBeta(ch, alpha_logscale=h.snake_logscale)
			self.activation_post = Activation1d(activation=activation_post)
		else:
			raise NotImplementedError(
				"activation incorrectly specified. check the config file and look for 'activation'.")

		self.conv_post = weight_norm(Conv1d(ch, 1, 7, 1, padding=3))

		# weight initialization
		for i in range(len(self.ups)):
			self.ups[i].apply(init_weights)
		self.conv_post.apply(init_weights)

	def forward(self,x, c):
		# pre conv
		x = self.conv_pre(x)

		for i in range(self.num_upsamples):
			# upsampling
			for i_up in range(len(self.ups[i])):
				x = self.ups[i][i_up](x)
			# AMP blocks
			xs = None
			for j in range(self.num_kernels):
				if xs is None:
					xs = self.resblocks[i * self.num_kernels + j](x)
				else:
					xs += self.resblocks[i * self.num_kernels + j](x)
			x = xs / self.num_kernels

		# post conv
		x = self.activation_post(x)
		x = self.conv_post(x)
		x = torch.tanh(x)

		return x

	def remove_weight_norm(self):
		print('Removing weight norm...')
		for l in self.ups:
			for l_i in l:
				remove_weight_norm(l_i)
		for l in self.resblocks:
			l.remove_weight_norm()
		remove_weight_norm(self.conv_pre)
		remove_weight_norm(self.conv_post)

	def inference(self, c, z=None):
		# pad input mel with zeros to cut artifact
		# see https://github.com/seungwonpark/melgan/issues/8
		zero = torch.full((c.shape[0], h.num_mels, 10), -11.5129).to(c.device)
		mel = torch.cat((c, zero), dim=2)

		if z is None:
			z = torch.randn(c.shape[0], self.noise_dim, mel.size(2)).to(mel.device)

		audio = self.forward(mel, z)
		audio = audio[:, :, :-(self.hop_length * 10)]
		audio = audio.clamp(min=-1, max=1)
		return audio

	def eval(self, inference=False):
		super(BigVGAN, self).eval()
		# don't remove weight norm while validation in training loop
		if inference:
			self.remove_weight_norm()


class DiscriminatorP(nn.Module):
	def __init__(self, h, period, kernel_size=5, stride=3, use_spectral_norm=False):
		super(DiscriminatorP, self).__init__()
		self.period = period
		self.d_mult = h.discriminator_channel_mult
		norm_f = weight_norm if use_spectral_norm == False else spectral_norm
		self.convs = nn.ModuleList([
			norm_f(Conv2d(1, int(32 * self.d_mult), (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
			norm_f(Conv2d(int(32 * self.d_mult), int(128 * self.d_mult), (kernel_size, 1), (stride, 1),
						  padding=(get_padding(5, 1), 0))),
			norm_f(Conv2d(int(128 * self.d_mult), int(512 * self.d_mult), (kernel_size, 1), (stride, 1),
						  padding=(get_padding(5, 1), 0))),
			norm_f(Conv2d(int(512 * self.d_mult), int(1024 * self.d_mult), (kernel_size, 1), (stride, 1),
						  padding=(get_padding(5, 1), 0))),
			norm_f(Conv2d(int(1024 * self.d_mult), int(1024 * self.d_mult), (kernel_size, 1), 1, padding=(2, 0))),
		])
		self.conv_post = norm_f(Conv2d(int(1024 * self.d_mult), 1, (3, 1), 1, padding=(1, 0)))

	def forward(self, x):
		fmap = []

		# 1d to 2d
		b, c, t = x.shape
		if t % self.period != 0:  # pad first
			n_pad = self.period - (t % self.period)
			x = F.pad(x, (0, n_pad), "reflect")
			t = t + n_pad
		x = x.view(b, c, t // self.period, self.period)

		for l in self.convs:
			x = l(x)
			x = F.leaky_relu(x, LRELU_SLOPE)
			fmap.append(x)
		x = self.conv_post(x)
		fmap.append(x)
		x = torch.flatten(x, 1, -1)

		return x, fmap


class MultiPeriodDiscriminator(nn.Module):
	def __init__(self, h):
		super(MultiPeriodDiscriminator, self).__init__()
		self.mpd_reshapes = h.mpd_reshapes
		print("mpd_reshapes: {}".format(self.mpd_reshapes))
		discriminators = [DiscriminatorP(h, rs, use_spectral_norm=h.use_spectral_norm) for rs in self.mpd_reshapes]
		self.discriminators = nn.ModuleList(discriminators)

	def forward(self, y, y_hat):
		y_d_rs = []
		y_d_gs = []
		fmap_rs = []
		fmap_gs = []
		for i, d in enumerate(self.discriminators):
			y_d_r, fmap_r = d(y)
			y_d_g, fmap_g = d(y_hat)
			y_d_rs.append(y_d_r)
			fmap_rs.append(fmap_r)
			y_d_gs.append(y_d_g)
			fmap_gs.append(fmap_g)

		return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class DiscriminatorR(nn.Module):
	def __init__(self, cfg, resolution):
		super().__init__()

		self.resolution = resolution
		assert len(self.resolution) == 3, \
			"MRD layer requires list with len=3, got {}".format(self.resolution)
		self.lrelu_slope = LRELU_SLOPE

		norm_f = weight_norm if cfg.use_spectral_norm == False else spectral_norm
		if hasattr(cfg, "mrd_use_spectral_norm"):
			print("INFO: overriding MRD use_spectral_norm as {}".format(cfg.mrd_use_spectral_norm))
			norm_f = weight_norm if cfg.mrd_use_spectral_norm == False else spectral_norm
		self.d_mult = cfg.discriminator_channel_mult
		if hasattr(cfg, "mrd_channel_mult"):
			print("INFO: overriding mrd channel multiplier as {}".format(cfg.mrd_channel_mult))
			self.d_mult = cfg.mrd_channel_mult

		self.convs = nn.ModuleList([
			norm_f(nn.Conv2d(1, int(32 * self.d_mult), (3, 9), padding=(1, 4))),
			norm_f(nn.Conv2d(int(32 * self.d_mult), int(32 * self.d_mult), (3, 9), stride=(1, 2), padding=(1, 4))),
			norm_f(nn.Conv2d(int(32 * self.d_mult), int(32 * self.d_mult), (3, 9), stride=(1, 2), padding=(1, 4))),
			norm_f(nn.Conv2d(int(32 * self.d_mult), int(32 * self.d_mult), (3, 9), stride=(1, 2), padding=(1, 4))),
			norm_f(nn.Conv2d(int(32 * self.d_mult), int(32 * self.d_mult), (3, 3), padding=(1, 1))),
		])
		self.conv_post = norm_f(nn.Conv2d(int(32 * self.d_mult), 1, (3, 3), padding=(1, 1)))

	def forward(self, x):
		fmap = []

		x = self.spectrogram(x)
		x = x.unsqueeze(1)
		for l in self.convs:
			x = l(x)
			x = F.leaky_relu(x, self.lrelu_slope)
			fmap.append(x)
		x = self.conv_post(x)
		fmap.append(x)
		x = torch.flatten(x, 1, -1)

		return x, fmap

	def spectrogram(self, x):
		n_fft, hop_length, win_length = self.resolution
		x = F.pad(x, (int((n_fft - hop_length) / 2), int((n_fft - hop_length) / 2)), mode='reflect')
		x = x.squeeze(1)
		x = torch.stft(x, n_fft=n_fft, hop_length=hop_length, win_length=win_length, center=False, return_complex=True)
		x = torch.view_as_real(x)  # [B, F, TT, 2]
		mag = torch.norm(x, p=2, dim=-1)  # [B, F, TT]

		return mag


class MultiResolutionDiscriminator(nn.Module):
	def __init__(self, cfg, debug=False):
		super().__init__()
		self.resolutions = cfg.resolutions
		assert len(self.resolutions) == 3, \
			"MRD requires list of list with len=3, each element having a list with len=3. got {}". \
				format(self.resolutions)
		self.discriminators = nn.ModuleList(
			[DiscriminatorR(cfg, resolution) for resolution in self.resolutions]
		)

	def forward(self, y, y_hat):
		y_d_rs = []
		y_d_gs = []
		fmap_rs = []
		fmap_gs = []

		for i, d in enumerate(self.discriminators):
			y_d_r, fmap_r = d(x=y)
			y_d_g, fmap_g = d(x=y_hat)
			y_d_rs.append(y_d_r)
			fmap_rs.append(fmap_r)
			y_d_gs.append(y_d_g)
			fmap_gs.append(fmap_g)

		return y_d_rs, y_d_gs, fmap_rs, fmap_gs

def get_mel(x):
	return mel_spectrogram(x, h.n_fft, h.num_mels, h.sampling_rate, h.hop_size, h.win_size, h.fmin, h.fmax)

def mel_spectrogram(y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=False):
	if torch.min(y) < -1.:
		print('min value is ', torch.min(y))
	if torch.max(y) > 1.:
		print('max value is ', torch.max(y))

	global mel_basis, hann_window
	if fmax not in mel_basis:
		mel = librosa_mel_fn(sampling_rate, n_fft, num_mels, fmin, fmax)
		mel_basis[str(fmax)+'_'+str(y.device)] = torch.from_numpy(mel).float().to(y.device)
		hann_window[str(y.device)] = torch.hann_window(win_size).to(y.device)

	y = torch.nn.functional.pad(y.unsqueeze(1), (int((n_fft-hop_size)/2), int((n_fft-hop_size)/2)), mode='reflect')
	y = y.squeeze(1)

	# complex tensor as default, then use view_as_real for future pytorch compatibility
	spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window[str(y.device)],
					  center=center, pad_mode='reflect', normalized=False, onesided=True, return_complex=True)
	spec = torch.view_as_real(spec)
	spec = torch.sqrt(spec.pow(2).sum(-1)+(1e-9))

	spec = torch.matmul(mel_basis[str(fmax)+'_'+str(y.device)], spec)
	spec = torch.nn.utils.spectral_normalize_torch(spec)

	return spec

def feature_loss(fmap_r, fmap_g):
	loss = 0
	for dr, dg in zip(fmap_r, fmap_g):
		for rl, gl in zip(dr, dg):
			loss += torch.mean(torch.abs(rl - gl))

	return loss * 2


def init_weights(m, mean=0.0, std=0.01):
	classname = m.__class__.__name__
	if classname.find("Conv") != -1:
		m.weight.data.normal_(mean, std)


def get_padding(kernel_size, dilation=1):
	return int((kernel_size * dilation - dilation) / 2)


def discriminator_loss(disc_real_outputs, disc_generated_outputs):
	loss = 0
	r_losses = []
	g_losses = []
	for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
		r_loss = torch.mean((1 - dr) ** 2)
		g_loss = torch.mean(dg ** 2)
		loss += (r_loss + g_loss)
		r_losses.append(r_loss.item())
		g_losses.append(g_loss.item())

	return loss, r_losses, g_losses


def generator_loss(disc_outputs):
	loss = 0
	gen_losses = []
	for dg in disc_outputs:
		l = torch.mean((1 - dg) ** 2)
		gen_losses.append(l)
		loss += l

	return loss, gen_losses


if __name__ == '__main__':
	model = BigVGAN()

	c = torch.randn(3, 100, 10)
	z = torch.randn(3, 64, 10)
	print(c.shape)

	y = model(c, z)
	print(y.shape)
	assert y.shape == torch.Size([3, 1, 2560])

	pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
	print(pytorch_total_params)