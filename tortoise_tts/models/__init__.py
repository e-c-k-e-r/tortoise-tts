# https://github.com/neonbjb/tortoise-tts/tree/98a891e66e7a1f11a830f31bd1ce06cc1f6a88af/tortoise/models
# All code under this folder is licensed as Apache License 2.0 per the original repo

from functools import cache

from ..arch_utils import TorchMelSpectrogram, TacotronSTFT

from .unified_voice import UnifiedVoice
from .diffusion import DiffusionTTS
from .vocoder import UnivNetGenerator
from .clvp import CLVP
from .dvae import DiscreteVAE

# semi-necessary as a way to provide a mechanism for other portions of the program to access models
@cache
def load_model(name, device="cuda", **kwargs):
	if "autoregressive" in name or "unified_voice" in name:
		model = UnifiedVoice(**kwargs)
	elif "diffusion" in name:
		model = DiffusionTTS(**kwargs)
	elif "clvp" in name:
		model = CLVP(**kwargs)
	elif "vocoder" in name:
		model = UnivNetGenerator(**kwargs)
	elif "dvae" in name:
		model = DiscreteVAE(**kwargs)
	# to-do: figure out of the below two give the exact same output, since the AR uses #1, the Diffusion uses #2
	elif "stft" in name:
		model = TacotronSTFT(1024, 256, 1024, 100, 24000, 0, 12000, **kwargs)
	elif "tms" in name:
		model = TorchMelSpectrogram(**kwargs)

	model = model.to(device=device)
	return model

def unload_model():
	load_model.cache_clear()

def get_model(config, training=True):
	name = config.name

	model = load_model(config.name)

	config.training = False

	print(f"{name} ({next(model.parameters()).dtype}): {sum(p.numel() for p in model.parameters() if p.requires_grad)} parameters")

	return model

def get_models(models, training=True):
	return { model.full_name: get_model(model, training=training) for model in models }