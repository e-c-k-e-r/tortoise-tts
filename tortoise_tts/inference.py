import torch
import torchaudio
import soundfile

from torch import Tensor
from einops import rearrange
from pathlib import Path

from .emb.mel import encode_from_files as encode_mel, trim, trim_random
from .utils import to_device

from .config import cfg
from .models import get_models, load_model
from .engines import load_engines, deepspeed_available
from .data import get_phone_symmap, tokenize

from .models.arch_utils import denormalize_tacotron_mel
from .models.diffusion import get_diffuser

from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F

if deepspeed_available:
	import deepspeed

class TTS():
	def __init__( self, config=None, device=None, amp=None, dtype=None ):
		self.loading = True 
		
		self.input_sample_rate = 24000
		self.output_sample_rate = 24000

		if config:
			cfg.load_yaml( config )

		try:
			cfg.format( training=False )
			cfg.dataset.use_hdf5 = False # could use cfg.load_hdf5(), but why would it ever need to be loaded for inferencing
		except Exception as e:
			print("Error while parsing config YAML:")
			raise e # throw an error because I'm tired of silent errors messing things up for me

		if amp is None:
			amp = cfg.inference.amp
		if dtype is None or dtype == "auto":
			dtype = cfg.inference.weight_dtype
		if device is None:
			device = cfg.device

		cfg.device = device
		cfg.mode = "inferencing"
		cfg.trainer.backend = cfg.inference.backend
		cfg.trainer.weight_dtype = dtype
		cfg.inference.weight_dtype = dtype

		self.device = device
		self.dtype = cfg.inference.dtype
		self.amp = amp

		self.symmap = None

		self.engines = load_engines(training=False)
		for name, engine in self.engines.items():
			if self.dtype != torch.int8:
				engine.to(self.device, dtype=self.dtype if not self.amp else torch.float32)

		self.engines.eval()

		if self.symmap is None:
			self.symmap = get_phone_symmap()

		self.loading = False 

	def encode_text( self, text, language="en" ):
		# already a tensor, return it
		if isinstance( text, Tensor ):
			return text

		tokens = tokenize( text )

		return torch.tensor( tokens )

	def encode_audio( self, paths, trim_length=0.0 ):
		# already a tensor, return it
		if isinstance( paths, Tensor ):
			return paths

		# split string into paths
		if isinstance( paths, str ):
			paths = [ Path(p) for p in paths.split(";") ]

		# merge inputs
		return encode_mel( paths, device=self.device )

	@torch.inference_mode()
	def inference(
		self,
		text,
		references,
		#language="en",
		max_ar_steps=500,
		max_diffusion_steps=80,
		#max_ar_context=-1,
		#input_prompt_length=0.0,
		ar_temp=1.0,
		diffusion_temp=1.0,
		#min_ar_temp=0.95,
		#min_diffusion_temp=0.5,
		top_p=1.0,
		top_k=0,
		repetition_penalty=1.0,
		#repetition_penalty_decay=0.0,
		length_penalty=0.0,
		beam_width=1,
		#mirostat_tau=0,
		#mirostat_eta=0.1,
		out_path=None
	):
		lines = text.split("\n")

		wavs = []
		sr = 24_000

		autoregressive = None
		diffusion = None
		clvp = None
		vocoder = None
		diffuser = get_diffuser(steps=max_diffusion_steps, cond_free=False)

		autoregressive_latents, diffusion_latents = self.encode_audio( references )["latent"]
		
		for name, engine in self.engines.items():
			if "autoregressive" in name:
				autoregressive = engine.module
			elif "diffusion" in name:
				diffusion = engine.module
			elif "clvp" in name:
				clvp = engine.module
			elif "vocoder" in name:
				vocoder = engine.module

		if autoregressive is None:
			autoregressive = load_model("autoregressive", device=cfg.device)
		if diffusion is None:
			diffusion = load_model("diffusion", device=cfg.device)
		if clvp is None:
			clvp = load_model("clvp", device=cfg.device)
		if vocoder is None:
			vocoder = load_model("vocoder", device=cfg.device)

		wavs = []
		# other vars
		calm_token = 832

		for line in lines:
			if out_path is None:
				output_dir = Path("./data/results/")
				if not output_dir.exists():
					output_dir.mkdir(parents=True, exist_ok=True)
				out_path = output_dir / f"{cfg.start_time}.wav"

			text = self.encode_text( line ).to(device=cfg.device)

			text_tokens = pad_sequence([ text ], batch_first = True)
			text_lengths = torch.Tensor([ text.shape[0] ]).to(dtype=torch.int32)

			with torch.autocast("cuda", dtype=cfg.inference.dtype, enabled=cfg.inference.amp):
				# autoregressive pass
				codes = autoregressive.inference_speech(
					autoregressive_latents,
					text_tokens,
					do_sample=True,
					top_p=top_p,
					temperature=ar_temp,
					num_return_sequences=1,
					length_penalty=length_penalty,
					repetition_penalty=repetition_penalty,
					max_generate_length=max_ar_steps,
				)
				padding_needed = max_ar_steps - codes.shape[1]
				codes = F.pad(codes, (0, padding_needed), value=autoregressive.stop_mel_token)

				for i, code in enumerate( codes ):
					stop_token_indices = (codes[i] == autoregressive.stop_mel_token).nonzero()

					if len(stop_token_indices) == 0:
						continue

					codes[i][stop_token_indices] = 83
					stm = stop_token_indices.min().item()
					codes[i][stm:] = 83
					if stm - 3 < codes[i].shape[0]:
						codes[i][-3] = 45
						codes[i][-2] = 45
						codes[i][-1] = 248

				wav_lengths = torch.tensor([codes.shape[-1] * autoregressive.mel_length_compression], device=text_tokens.device)

				latents = autoregressive.forward(
					autoregressive_latents,
					text_tokens,
					text_lengths,
					codes,
					wav_lengths,
					return_latent=True,
					clip_inputs=False
				)

				calm_tokens = 0
				for k in range( codes.shape[-1] ):
					if codes[0, k] == calm_token:
						calm_tokens += 1
					else:
						calm_tokens = 0
					if calm_tokens > 8:  # 8 tokens gives the diffusion model some "breathing room" to terminate speech.
						latents = latents[:, :k]
						break

				# diffusion pass
				output_seq_len = latents.shape[1] * 4 * 24000 // 22050  # This diffusion model converts from 22kHz spectrogram codes to a 24kHz spectrogram signal.
				output_shape = (latents.shape[0], 100, output_seq_len)
				precomputed_embeddings = diffusion.timestep_independent(latents, diffusion_latents, output_seq_len, False)

				noise = torch.randn(output_shape, device=latents.device) * diffusion_temp
				mel = diffuser.p_sample_loop(
					diffusion,
					output_shape,
					noise=noise,
					model_kwargs={'precomputed_aligned_embeddings': precomputed_embeddings},
					progress=True
				)
				mels = denormalize_tacotron_mel(mel)[:,:,:output_seq_len]

				# vocoder pass
				waves = vocoder.inference(mels)

				for wav in waves:
					if out_path is not None:
						torchaudio.save( out_path, wav.cpu(), sr )
					wavs.append(wav)
		
		return (torch.concat(wavs, dim=-1), sr)
		
