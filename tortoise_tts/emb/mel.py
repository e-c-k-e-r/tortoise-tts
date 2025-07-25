from ..config import cfg

import argparse
import random
import torch
import torchaudio

from functools import cache
from pathlib import Path
from typing import Union

from einops import rearrange
from torch import Tensor
from tqdm import tqdm

from ..models import load_model, unload_model

import torch.nn.functional as F

def pad_or_truncate(t, length):
	"""
	Utility function for forcing <t> to have the specified sequence length, whether by clipping it or padding it with 0s.
	"""
	if t.shape[-1] == length:
		return t
	elif t.shape[-1] < length:
		return F.pad(t, (0, length-t.shape[-1]))
	else:
		return t[..., :length]

# decodes mel spectrogram into a wav
@torch.inference_mode()
def decode(codes: Tensor, device="cuda"):
	model = load_model("vocoder", device)
	return vocoder.inference(codes)

# huh
def decode_to_wave(resps: Tensor, device="cuda"):
	return decode(resps, device=device, levels=levels)

def decode_to_file(resps: Tensor, path: Path, device="cuda"):
	wavs, sr = decode(resps, device=device)

	torchaudio.save(str(path), wavs.cpu(), sr)
	return wavs, sr

def _replace_file_extension(path, suffix):
	return (path.parent / path.name.split(".")[0]).with_suffix(suffix)

def format_autoregressive_conditioning( wav, cond_length=132300, device="cuda" ):
	"""
	Converts the given conditioning signal to a MEL spectrogram and clips it as expected by the models.
	"""
	model = load_model("tms", device=device)

	if cond_length > 0:
		gap = wav.shape[-1] - cond_length
		if gap < 0:
			wav = F.pad(wav, pad=(0, abs(gap)))
		elif gap > 0:
			rand_start = random.randint(0, gap)
			wav = wav[:, rand_start:rand_start + cond_length]

	mel_clip = model(wav.unsqueeze(0)).squeeze(0) # ???
	return mel_clip.unsqueeze(0).to(device) # ???

def format_diffusion_conditioning( sample, device, do_normalization=False ):
	model = load_model("stft", device=device, sr=24_000)

	sample = torchaudio.functional.resample(sample, 22050, 24000)
	sample = pad_or_truncate(sample, 102400)
	sample = sample.to(device)
	mel = model.mel_spectrogram(sample)
	"""
	if do_normalization:
		mel = normalize_tacotron_mel(mel)
	"""
	return mel

# encode a wav to conditioning latents + mel codes
@torch.inference_mode()
def encode(wav: Tensor, sr: int = cfg.sample_rate, device="cuda", dtype=None):
	wav_length = wav.shape[-1]
	duration = wav_length / sr
	wav = torchaudio.functional.resample(wav, sr, 22050)

	dvae = load_model("dvae", device=device)
	unified_voice = load_model("autoregressive", device=device)
	diffusion = load_model("diffusion", device=device)
	mel_inputs = format_autoregressive_conditioning( wav, 0, device )

	autoregressive_conds = torch.stack([ format_autoregressive_conditioning(wav.to(device), device=device) ], dim=1)
	diffusion_conds = torch.stack([ format_diffusion_conditioning(wav.to(device), device=device) ], dim=1)

	codes = dvae.get_codebook_indices( mel_inputs )

	autoregressive_latent = unified_voice.get_conditioning(autoregressive_conds)
	diffusion_latent = diffusion.get_conditioning(diffusion_conds)

	return {
		"codes": codes,
		"conds": (autoregressive_conds, diffusion_conds),
		"latent": (autoregressive_latent, diffusion_latent),
		"metadata": {
			"original_length": wav_length,
			"sample_rate": sr,
			"duration": duration
		}
	}

def encode_from_files(paths, device="cuda"):
	tuples = [ torchaudio.load(str(path)) for path in paths ]

	wavs = []
	main_sr = tuples[0][1]
	for wav, sr in tuples:
		assert sr == main_sr, "Mismatching sample rates"

		if wav.shape[0] == 2:
			wav = wav[:1]

		wavs.append(wav)

	wav = torch.cat(wavs, dim=-1)
	
	return encode(wav, sr, device)

def encode_from_file(path, device="cuda"):
	if isinstance( path, list ):
		return encode_from_files( path, device )
	else:
		path = str(path)
		wav, sr = torchaudio.load(path)

	if wav.shape[0] == 2:
		wav = wav[:1]
	
	qnt = encode(wav, sr, device)

	return qnt

"""
Helper Functions
"""
# trims from the start, up to `target`
def trim( qnt, target ):
	length = qnt.shape[0]
	if target > 0:
		start = 0
		end = start + target
		if end >= length:
			start = length - target
			end = length
	# negative length specified, trim from end
	else:
		start = length + target
		end = length
		if start < 0:
			start = 0

	return qnt[start:end]

# trims a random piece of audio, up to `target`
# to-do: try and align to EnCodec window
def trim_random( qnt, target ):
	length = qnt.shape[0]

	start = int(length * random.random())
	end = start + target
	if end >= length:
		start = length - target
		end = length				

	return qnt[start:end]

# repeats the audio to fit the target size
def repeat_extend_audio( qnt, target ):
	pieces = []
	length = 0
	while length < target:
		pieces.append(qnt)
		length += qnt.shape[0]

	return trim(torch.cat(pieces), target)

# merges two quantized audios together
# I don't know if this works
def merge_audio( *args, device="cpu", scale=[] ):
	qnts = [*args]
	decoded = [ decode(qnt, device=device, levels=levels)[0] for qnt in qnts ]

	if len(scale) == len(decoded):
		for i in range(len(scale)):
			decoded[i] = decoded[i] * scale[i]

	combined = sum(decoded) / len(decoded)
	return encode(combined, cfg.sample_rate, device="cpu", levels=levels)[0].t()