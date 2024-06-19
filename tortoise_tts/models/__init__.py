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

DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../data/')

# semi-necessary as a way to provide a mechanism for other portions of the program to access models
@cache
def load_model(name, device="cuda", **kwargs):
	load_path = None
	state_dict_key = None
	strict = True

	if "rlg" in name:
		if "autoregressive" in name:
			model = RandomLatentConverter(1024, **kwargs)
			load_path = f'{DEFAULT_MODEL_PATH}/rlg_auto.pth'
		if "diffusion" in name:
			model = RandomLatentConverter(2048, **kwargs)
			load_path = f'{DEFAULT_MODEL_PATH}/rlg_diffuser.pth'
	elif "autoregressive" in name or "unified_voice" in name:
		strict = False
		model = UnifiedVoice(**kwargs)
		load_path = f'{DEFAULT_MODEL_PATH}/autoregressive.pth'
	elif "diffusion" in name:
		model = DiffusionTTS(**kwargs)
		load_path = f'{DEFAULT_MODEL_PATH}/diffusion.pth'		
	elif "clvp" in name:
		model = CLVP(**kwargs)
		load_path = f'{DEFAULT_MODEL_PATH}/clvp2.pth'
	elif "vocoder" in name:
		model = UnivNetGenerator(**kwargs)
		load_path = f'{DEFAULT_MODEL_PATH}/vocoder.pth'
		state_dict_key = 'model_g'
	elif "dvae" in name:
		load_path = f'{DEFAULT_MODEL_PATH}/dvae.pth'
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