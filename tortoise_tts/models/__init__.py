# https://github.com/neonbjb/tortoise-tts/tree/98a891e66e7a1f11a830f31bd1ce06cc1f6a88af/tortoise/models
# All code under this folder is licensed as Apache License 2.0 per the original repo

from functools import cache

from .arch_utils import TorchMelSpectrogram, TacotronSTFT

from .unified_voice import UnifiedVoice
from .diffusion import DiffusionTTS
from .vocoder import UnivNetGenerator
from .clvp import CLVP
from .dvae import DiscreteVAE
from .random_latent_generator import RandomLatentConverter

import os
import torch
from pathlib import Path

DEFAULT_MODEL_PATH = Path(__file__).parent.parent.parent / 'data/models'
DEFAULT_MODEL_URLS = {
	'autoregressive.pth': 'https://huggingface.co/jbetker/tortoise-tts-v2/resolve/main/.models/autoregressive.pth',
    'classifier.pth': 'https://huggingface.co/jbetker/tortoise-tts-v2/resolve/main/.models/classifier.pth',
    'clvp2.pth': 'https://huggingface.co/jbetker/tortoise-tts-v2/resolve/main/.models/clvp2.pth',
    'cvvp.pth': 'https://huggingface.co/jbetker/tortoise-tts-v2/resolve/main/.models/cvvp.pth',
    'diffusion.pth': 'https://huggingface.co/jbetker/tortoise-tts-v2/resolve/main/.models/diffusion_decoder.pth',
    'vocoder.pth': 'https://huggingface.co/jbetker/tortoise-tts-v2/resolve/main/.models/vocoder.pth',
    'dvae.pth': 'https://huggingface.co/jbetker/tortoise-tts-v2/resolve/3704aea61678e7e468a06d8eea121dba368a798e/.models/dvae.pth',
    'rlg_auto.pth': 'https://huggingface.co/jbetker/tortoise-tts-v2/resolve/main/.models/rlg_auto.pth',
    'rlg_diffuser.pth': 'https://huggingface.co/jbetker/tortoise-tts-v2/resolve/main/.models/rlg_diffuser.pth',
    'mel_norms.pth': 'https://huggingface.co/jbetker/tortoise-tts-v2/resolve/main/data/mel_norms.pth',
}

import requests
from tqdm import tqdm

# kludge, probably better to use HF's model downloader function
# to-do: write to a temp file then copy so downloads can be interrupted
def download_model( save_path, chunkSize = 1024, unit = "MiB" ):
	scale = 1
	if unit == "KiB":
		scale = (1024)
	elif unit == "MiB":
		scale = (1024 * 1024)
	elif unit == "MiB":
		scale = (1024 * 1024 * 1024)
	elif unit == "KB":
		scale = (1000)
	elif unit == "MB":
		scale = (1000 * 1000)
	elif unit == "MB":
		scale = (1000 * 1000 * 1000)

	name = save_path.name
	url = DEFAULT_MODEL_URLS[name] if name in DEFAULT_MODEL_URLS else None
	if url is None:
		raise Exception(f'Model requested for download but not defined: {name}')

	if not save_path.parent.exists():
		save_path.parent.mkdir(parents=True, exist_ok=True)

	r = requests.get(url, stream=True)
	content_length = int(r.headers['Content-Length'] if 'Content-Length' in r.headers else r.headers['content-length']) // scale

	with open(save_path, 'wb') as f:
		bar = tqdm( unit=unit, total=content_length )
		for chunk in r.iter_content(chunk_size=chunkSize): 
			if not chunk:
				continue
			
			bar.update( len(chunk) / scale )
			f.write(chunk)

# semi-necessary as a way to provide a mechanism for other portions of the program to access models
@cache
def load_model(name, device="cuda", **kwargs):
	load_path = None
	state_dict_key = None
	strict = True

	if "rlg" in name:
		if "autoregressive" in name:
			model = RandomLatentConverter(1024, **kwargs)
			load_path = DEFAULT_MODEL_PATH / 'rlg_auto.pth'
		if "diffusion" in name:
			model = RandomLatentConverter(2048, **kwargs)
			load_path = DEFAULT_MODEL_PATH / 'rlg_diffuser.pth'
	elif "autoregressive" in name or "unified_voice" in name:
		strict = False
		model = UnifiedVoice(**kwargs)
		load_path = DEFAULT_MODEL_PATH / 'autoregressive.pth'
	elif "diffusion" in name:
		model = DiffusionTTS(**kwargs)
		load_path = DEFAULT_MODEL_PATH / 'diffusion.pth'		
	elif "clvp" in name:
		model = CLVP(**kwargs)
		load_path = DEFAULT_MODEL_PATH / 'clvp2.pth'
	elif "vocoder" in name:
		model = UnivNetGenerator(**kwargs)
		load_path = DEFAULT_MODEL_PATH / 'vocoder.pth'
		state_dict_key = 'model_g'
	elif "dvae" in name:
		load_path = DEFAULT_MODEL_PATH / 'dvae.pth'
		model = DiscreteVAE(**kwargs)
	# to-do: figure out of the below two give the exact same output
	elif "stft" in name:
		sr = kwargs.pop("sr")
		if sr == 24_000:
			model = TacotronSTFT(1024, 256, 1024, 100, 24000, 0, 12000, **kwargs)
		else:
			model = TacotronSTFT(**kwargs)
	elif "tms" in name:
		model = TorchMelSpectrogram(**kwargs)

	model = model.to(device=device)

	if load_path is not None:
		# download if does not exist
		if not load_path.exists():
			download_model( load_path )

		state_dict = torch.load(load_path, map_location=device)
		if state_dict_key:
			state_dict = state_dict[state_dict_key]
		model.load_state_dict(state_dict, strict=strict)

	model.eval()

	return model

def unload_model():
	load_model.cache_clear()

def get_model(config, training=True):
	name = config.name

	model = load_model(config.name)
	config.training = "autoregressive" in config.name
	model.config = config

	print(f"{name} ({next(model.parameters()).dtype}): {sum(p.numel() for p in model.parameters() if p.requires_grad)} parameters")

	return model

def get_models(models, training=True):
	return { model.full_name: get_model(model, training=training) for model in models }