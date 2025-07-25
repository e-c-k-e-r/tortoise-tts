import torch
import torchaudio
import soundfile
import time

from torch import Tensor
from einops import rearrange
from pathlib import Path
from tqdm import tqdm

from .emb.mel import encode_from_files as encode_mel, trim, trim_random
from .utils import to_device, set_seed, ml

from .config import cfg, DEFAULT_YAML
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

		if config is None:
			config = DEFAULT_YAML

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

	# taken from here https://github.com/coqui-ai/TTS/blob/d21f15cc850788f9cdf93dac0321395138665287/TTS/tts/models/xtts.py#L666
	def handle_chunks(self, wav_gen, wav_gen_prev, wav_overlap, overlap_len):
		"""Handle chunk formatting in streaming mode"""
		wav_chunk = wav_gen[:-overlap_len]
		if wav_gen_prev is not None:
			wav_chunk = wav_gen[(wav_gen_prev.shape[0] - overlap_len) : -overlap_len]
		if wav_overlap is not None:
			crossfade_wav = wav_chunk[:overlap_len]
			crossfade_wav = crossfade_wav * torch.linspace(0.0, 1.0, overlap_len).to(crossfade_wav.device)
			wav_chunk[:overlap_len] = wav_overlap * torch.linspace(1.0, 0.0, overlap_len).to(wav_overlap.device)
			wav_chunk[:overlap_len] += crossfade_wav
		wav_overlap = wav_gen[-overlap_len:]
		wav_gen_prev = wav_gen
		return wav_chunk, wav_gen_prev, wav_overlap

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
		ar_temp=0.8,
		diffusion_temp=1.0,
		#min_ar_temp=0.95,
		#min_diffusion_temp=0.5,
		top_p=1.0,
		top_k=0,
		repetition_penalty=1.0,
		#repetition_penalty_decay=0.0,
		length_penalty=1.0,
		beam_width=1,
		#mirostat_tau=0,
		#mirostat_eta=0.1,

		diffusion_sampler="ddim",
		cond_free=True,

		vocoder_type="bigvgan",

		seed=None,

		out_path=None,
	):
		lines = text.split("\n")

		wavs = []
		sr = 24_000

		autoregressive = None
		diffusion = None
		clvp = None
		vocoder = None
		diffuser = get_diffuser(steps=max_diffusion_steps, cond_free=cond_free)
		
		for name, engine in self.engines.items():
			if "autoregressive" in name:
				autoregressive = engine.module
			elif "diffusion" in name:
				diffusion = engine.module
			elif "clvp" in name:
				clvp = engine.module
			elif vocoder_type in name:
				vocoder = engine.module

		if autoregressive is None:
			autoregressive = load_model("autoregressive", device=cfg.device)
		if diffusion is None:
			diffusion = load_model("diffusion", device=cfg.device)
		if clvp is None:
			clvp = load_model("clvp", device=cfg.device)
		if vocoder is None:
			vocoder = load_model(vocoder_type, device=cfg.device)
		
		autoregressive = autoregressive.to(cfg.device)
		diffusion = diffusion.to(cfg.device)
		autoregressive_latents, diffusion_latents = self.encode_audio( references )["latent"]

		# shove everything to cpu
		if cfg.inference.auto_unload:
			autoregressive = autoregressive.to("cpu")
			diffusion = diffusion.to("cpu")
			clvp = clvp.to("cpu")
			vocoder = vocoder.to("cpu")

		wavs = []
		# other vars
		calm_token = 83

		candidates = 1

		set_seed(seed)

		for line in lines:
			if out_path is None:
				output_dir = Path("./data/results/")
				if not output_dir.exists():
					output_dir.mkdir(parents=True, exist_ok=True)
				out_path = output_dir / f"{time.time()}.wav"

			text = self.encode_text( line ).to(device=cfg.device)

			text_tokens = pad_sequence([ text ], batch_first = True)
			text_lengths = torch.Tensor([ text.shape[0] ]).to(dtype=torch.int32)

			# streaming interface spits out the final hidden state, which HiFiGAN seems to be trained against
			if vocoder_type == "hifigan":
				waves = []
				all_latents = []
				all_codes = []

				wav_gen_prev = None
				wav_overlap = None
				is_end = False
				first_buffer = 60

				stream_chunk_size = 40
				overlap_wav_len = 1024

				with torch.autocast("cuda", dtype=cfg.inference.dtype, enabled=cfg.inference.amp):
					with ml.auto_unload(autoregressive, enabled=cfg.inference.auto_unload):
						with ml.auto_unload(vocoder, enabled=cfg.inference.auto_unload):
							inputs = autoregressive.compute_embeddings( autoregressive_latents, text_tokens )

							gpt_generator = autoregressive.get_generator(
								inputs=inputs,
								top_k=top_k,
								top_p=top_p,
								temperature=ar_temp,
								do_sample=True,
								num_beams=max(1, beam_width),
								num_return_sequences=1,
								length_penalty=length_penalty,
								repetition_penalty=repetition_penalty,
								output_attentions=False,
								output_hidden_states=True,
							)

							bar = tqdm( unit="it", total=500 )
							while not is_end:
								try:
									codes, latent = next(gpt_generator)
									all_latents += [latent]
									all_codes += [codes]
								except StopIteration:
									is_end = True

								if is_end or (stream_chunk_size > 0 and len(all_codes) >= max(stream_chunk_size, first_buffer)):
									first_buffer = 0
									all_codes = []
									bar.update( stream_chunk_size )

									latents = torch.cat(all_latents, dim=0)[None, :].to(cfg.device)
									wav_gen = vocoder.inference(latents, autoregressive_latents)
									wav_gen = wav_gen.squeeze()

									wav_chunk = wav_gen[:-overlap_wav_len]
									if wav_gen_prev is not None:
										wav_chunk = wav_gen[(wav_gen_prev.shape[0] - overlap_wav_len) : -overlap_wav_len]
									if wav_overlap is not None:
										crossfade_wav = wav_chunk[:overlap_wav_len]
										crossfade_wav = crossfade_wav * torch.linspace(0.0, 1.0, overlap_wav_len).to(crossfade_wav.device)
										wav_chunk[:overlap_wav_len] = wav_overlap * torch.linspace(1.0, 0.0, overlap_wav_len).to(wav_overlap.device)
										wav_chunk[:overlap_wav_len] += crossfade_wav
									
									wav_overlap = wav_gen[-overlap_wav_len:]
									wav_gen_prev = wav_gen
									

									# yielding requires to do a bunch of pain to work around it turning into an async function
									"""
									yield wav_chunk
									"""

									waves.append( wav_chunk.unsqueeze(0) )

							bar.close()

				wav = torch.concat(waves, dim=-1)

				if out_path is not None:
					torchaudio.save( out_path, wav.cpu(), sr )

				wavs.append(wav)

				continue

			with torch.autocast("cuda", dtype=cfg.inference.dtype, enabled=cfg.inference.amp):
				with ml.auto_unload(autoregressive, enabled=cfg.inference.auto_unload):
					# autoregressive pass
					codes = autoregressive.inference_speech(
						autoregressive_latents,
						text_tokens,
						do_sample=True,
						top_k=top_k,
						top_p=top_p,
						temperature=ar_temp,
						num_return_sequences=candidates,
						num_beams=max(1,beam_width),
						length_penalty=length_penalty,
						repetition_penalty=repetition_penalty,
						max_generate_length=max_ar_steps,
					)

					"""
					padding_needed = max_ar_steps - codes.shape[1]
					codes = F.pad(codes, (0, padding_needed), value=autoregressive.stop_mel_token)
					"""

					for i, code in enumerate( codes ):
						stop_token_indices = (codes[i] == autoregressive.stop_mel_token).nonzero()
						stm = stop_token_indices.min().item()

						if len(stop_token_indices) == 0:
							continue

						codes[i][stop_token_indices] = 83
						codes[i][stm:] = 83

						if stm - 3 < codes[i].shape[0]:
							codes[i][-3] = 45
							codes[i][-2] = 45
							codes[i][-1] = 248

					wav_lengths = torch.tensor([codes.shape[-1] * autoregressive.mel_length_compression], device=text_tokens.device)

					# to-do: actually move this after the CLVP to get the best latents instead
					latents = autoregressive.forward(
						autoregressive_latents if candidates <= 1 else autoregressive_latents.repeat(candidates, 1),
						text_tokens if candidates <= 1 else text_tokens.repeat(candidates, 1),
						text_lengths if candidates <= 1 else text_lengths.repeat(candidates, 1),
						codes,
						wav_lengths if candidates <= 1 else wav_lengths.repeat(candidates, 1),
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

				# clvp pass
				if candidates > 1:
					with ml.auto_unload(clvp, enabled=cfg.inference.auto_unload):
						scores = clvp(text_tokens.repeat(codes.shape[0], 1), codes, return_loss=False)
						indices = torch.topk(scores, k=candidates).indices
						codes = codes[indices]

				# diffusion pass
				with ml.auto_unload(diffusion, enabled=cfg.inference.auto_unload):
					output_seq_len = latents.shape[1] * 4 * 24000 // 22050  # This diffusion model converts from 22kHz spectrogram codes to a 24kHz spectrogram signal.
					output_shape = (latents.shape[0], 100, output_seq_len)
					precomputed_embeddings = diffusion.timestep_independent(latents, diffusion_latents, output_seq_len, False)

					noise = torch.randn(output_shape, device=latents.device) * diffusion_temp
					mel = diffuser.sample_loop(
						diffusion,
						output_shape,
						sampler=diffusion_sampler,
						noise=noise,
						model_kwargs={'precomputed_aligned_embeddings': precomputed_embeddings},
						progress=True
					)
					mels = denormalize_tacotron_mel(mel)[:,:,:output_seq_len]

				# vocoder pass
				with ml.auto_unload(vocoder, enabled=cfg.inference.auto_unload):
					waves = vocoder.inference(mels)

				for wav in waves:
					if out_path is not None:
						torchaudio.save( out_path, wav.cpu(), sr )
					wavs.append(wav)

		return (torch.concat(wavs, dim=-1), sr)
		
