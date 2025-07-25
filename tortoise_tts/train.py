# todo: clean this mess up

from .config import cfg
from .data import create_train_val_dataloader, get_random_prompt, tokenize
from .emb import mel

from .utils import setup_logging, to_device, trainer, flatten_dict, do_gc, ml
from .utils.distributed import is_global_leader

import auraloss
import json
import logging
import random
import torch
import torchaudio
import torch.nn.functional as F
import traceback
import shutil

from collections import defaultdict

from tqdm import tqdm
import argparse

from torch.nn.utils.rnn import pad_sequence

from .models.arch_utils import denormalize_tacotron_mel
from .models.diffusion import get_diffuser
from .models import load_model

_logger = logging.getLogger(__name__)

mel_stft_loss = auraloss.freq.MelSTFTLoss(cfg.sample_rate, device="cpu")

def train_feeder(engine, batch, teacher=None):
	with torch.autocast("cuda", dtype=cfg.trainer.dtype, enabled=cfg.trainer.amp):
		device = batch["text"][0].device
		batch_size = len(batch["text"])

		autoregressive_latents = torch.stack([ latents for latents in batch["latents_0"] ])
		diffusion_latents = torch.stack([ latents for latents in batch["latents_1"] ])

		text_tokens = pad_sequence([ text for text in batch["text"] ], batch_first = True)
		text_lengths = torch.Tensor([ text.shape[0] for text in batch["text"] ]).to(dtype=torch.int32)
		mel_codes = pad_sequence([ codes[0] for codes in batch["mel"] ], batch_first = True, padding_value = engine.module.stop_mel_token )
		wav_lengths = torch.Tensor([ x for x in batch["wav_length"] ]).to(dtype=torch.int32)

		engine.forward(autoregressive_latents, text_tokens, text_lengths, mel_codes, wav_lengths)

		engine.current_batch_size = batch_size
		losses = engine.gather_attribute("loss")
		stat = engine.gather_attribute("stats")

		loss = torch.stack([*losses.values()]).sum()

	stats = {}
	stats |= {k: v.item() for k, v in losses.items()}
	stats |= {k: v.item() for k, v in stat.items()}

	engine.tokens_processed += sum([ text.shape[0] for text in batch["text"] ])
	engine.tokens_processed += sum([ mel.shape[-1] for mel in batch["mel"] ])

	return loss, stats

@torch.inference_mode()
def run_eval(engines, eval_name, dl, args=None):
	stats = defaultdict(list)
	stats['loss'] = []

	if cfg.evaluation.size == 0:
		return

	autoregressive = None
	diffusion = None
	clvp = None
	vocoder = None
	diffuser = get_diffuser(steps=30, cond_free=False)
	
	for name in engines:
		engine = engines[name]
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

	def generate( batch, generate_codes=True ):
		temperature = 1.0
		max_mel_tokens = 500 # * autoregressive.mel_length_compression
		stop_mel_token = autoregressive.stop_mel_token
		calm_token = 83
		verbose = False

		autoregressive_latents = torch.stack([ latents for latents in batch["latents_0"] ])
		diffusion_latents = torch.stack([ latents for latents in batch["latents_1"] ])

		text_tokens = pad_sequence([ text for text in batch["text"] ], batch_first = True)
		text_lengths = torch.Tensor([ text.shape[0] for text in batch["text"] ]).to(dtype=torch.int32)
		mel_codes = pad_sequence([ codes[0] for codes in batch["mel"] ], batch_first = True, padding_value = stop_mel_token )
		wav_lengths = torch.Tensor([ x for x in batch["wav_length"] ]).to(dtype=torch.int32)

		mel_codes = autoregressive.set_mel_padding(mel_codes, wav_lengths)

		with torch.autocast("cuda", dtype=cfg.trainer.dtype, enabled=cfg.trainer.amp):
			# autoregressive pass
			if generate_codes:
				codes = autoregressive.inference_speech(
					autoregressive_latents,
					text_tokens,
					do_sample=True,
					top_p=0.8,
					temperature=temperature,
					num_return_sequences=1,
					length_penalty=1.0,
					repetition_penalty=2.0,
					max_generate_length=max_mel_tokens,
				)
				padding_needed = max_mel_tokens - codes.shape[1]
				codes = F.pad(codes, (0, padding_needed), value=stop_mel_token)
			else:
				codes = mel_codes

			for i, code in enumerate( codes ):
				stop_token_indices = (codes[i] == stop_mel_token).nonzero()

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
			with ml.auto_unload(diffusion, enabled=True):
				output_seq_len = latents.shape[1] * 4 * 24000 // 22050  # This diffusion model converts from 22kHz spectrogram codes to a 24kHz spectrogram signal.
				output_shape = (latents.shape[0], 100, output_seq_len)
				precomputed_embeddings = diffusion.timestep_independent(latents, diffusion_latents, output_seq_len, False)

				noise = torch.randn(output_shape, device=latents.device) * temperature
				mel = diffuser.p_sample_loop(
					diffusion,
					output_shape,
					noise=noise,
					model_kwargs={'precomputed_aligned_embeddings': precomputed_embeddings},
					progress=True
				)
				mels = denormalize_tacotron_mel(mel)[:,:,:output_seq_len]

			# vocoder pass
			with ml.auto_unload(vocoder, enabled=True):
				wavs = vocoder.inference(mels)

			return wavs

	def process( name, batch, hyps, refs ):
		for speaker, path, ref_audio, hyp_audio in zip(batch["spkr_name"], batch["path"], refs, hyps):
			filename = f'{speaker}_{path.parts[-1]}'

			# to-do, refine the output dir to be sane-er
			ref_path = (cfg.log_dir / str(engines.global_step) / "ref" / filename).with_suffix(".wav")
			hyp_path = (cfg.log_dir / str(engines.global_step) / name / eval_name / filename).with_suffix(".wav")
			prom_path = (cfg.log_dir / str(engines.global_step) / name / "prom" / filename).with_suffix(".wav")

			hyp_path.parent.mkdir(parents=True, exist_ok=True)
			ref_path.parent.mkdir(parents=True, exist_ok=True)
			prom_path.parent.mkdir(parents=True, exist_ok=True)

			torchaudio.save( hyp_path, hyp_audio.cpu(), 24_000 )
			torchaudio.save( ref_path, ref_audio.cpu(), 24_000 )

			# pseudo loss calculation since we don't get the logits during eval
			min_length = min( ref_audio.shape[-1], hyp_audio.shape[-1] )
			ref_audio = ref_audio[..., 0:min_length]
			hyp_audio = hyp_audio[..., 0:min_length]
			stats['loss'].append(mel_stft_loss(hyp_audio[None, :, :], ref_audio[None, :, :]).item())
	
	processed = 0
	while processed < cfg.evaluation.size:
		# directly randomly sample
		if eval_name == "subtrain":
			# sample from dataset
			# to-do: derive from current iteration
			samples = [ to_device(dl.dataset[random.randint( 0, len( dl.dataset ) )], cfg.device) for sample in range( cfg.evaluation.batch_size ) ]
			# collate manually
			batch = {k: [s[k] for s in samples] for k in samples[0]}
		else:
			batch = to_device(next(iter(dl)), cfg.device)

		# limit to eval batch size in the event we somehow have a weird dataloader
		for key in batch.keys():
			batch[key] = batch[key][:cfg.evaluation.batch_size]

		batch_size = len(batch["text"])
		processed += batch_size

		hyp = generate( batch, generate_codes=True )
		ref = generate( batch, generate_codes=False )

		process( name, batch, hyp, ref )

	stats = {k: sum(v) / len(v) for k, v in stats.items() if v}
	engines_stats = {
		eval_name: stats,
		"it": engines.global_step,
	}

	try:
		for name, engine in engines.items():
			if engine.wandb is not None:
				engine.wandb.log({
					f'{eval_name}.loss.mstft': stats['loss'],
				}, step=engine.global_step)
	except Exception as e:
		print(e)

	#engines_stats['epoch'] = iteration * cfg.hyperparameters.gradient_accumulation_steps / len(dl)

	_logger.info(f"Validation Metrics: {json.dumps(engines_stats)}.")


def train():
	parser = argparse.ArgumentParser("TorToiSe TTS")
	parser.add_argument("--eval", action="store_true", default=None)
	parser.add_argument("--eval-random-text-prompts", action="store_true", default=None)
	#parser.add_argument("--eval-random-audio-prompts", action="store_true", default=None)
	args, unknown = parser.parse_known_args()

	# create log folder
	setup_logging(cfg.log_dir)
	# copy config yaml to backup
	if cfg.yaml_path is not None and is_global_leader():
		shutil.copy( cfg.yaml_path, cfg.log_dir / "config.yaml" )
	# create dataloaders
	train_dl, val_dl = create_train_val_dataloader()
	# evaluation lambda
	def eval_fn(engines):
		do_gc()
		engines.eval()
		# wrapped in a try block because it's sometimes prone to breaking
		try:
			run_eval(engines, "subtrain", train_dl, args)
			run_eval(engines, "val", val_dl, args)
		except Exception as e:
			_logger.warning(f"Error occurred while performing eval: {str(e)}")
			_logger.warning(traceback.format_exc())

		engines.train()
		#mel.unload_model()
		do_gc()
	# unload EnCodec if it's already loaded
	#mel.unload_model()
	# only eval is requested
	if args.eval:
		return eval_fn(engines=trainer.load_engines())

	"""
	# start web UI
	if cfg.trainer.load_webui:
		from .webui import start
		start(lock=False)
	"""

	# train
	trainer.train(
		train_dl=train_dl,
		train_feeder=train_feeder,
		eval_fn=eval_fn,
	)

if __name__ == "__main__":
	# to-do: for DDP, spawn multiprocess instead of requiring `torchrun --nnodes=1 --nproc-per-node=4 -m vall_e.train yaml="./data/config.yaml"`
	train()
