"""
# Handles processing audio provided through --input-audio of adequately annotated transcriptions provided through --input-metadata (through transcribe.py)
# Outputs NumPy objects containing quantized audio and adequate metadata for use of loading in the trainer through --output-dataset
"""

import os
import json
import argparse
import torch
import torchaudio
import numpy as np
import logging

_logger = logging.getLogger(__name__)

from tqdm.auto import tqdm
from pathlib import Path

from ..config import cfg

# need to validate if this is safe to import before modifying the config
from .mel import encode as quantize # , encode_batch as quantize_batch
from ..data import _load_artifact

def pad(num, zeroes):
	return str(num).zfill(zeroes+1)

def load_audio( path, device=None, dtype=None ):
	waveform, sr = torchaudio.load( path )
	if waveform.shape[0] > 1:
		# mix channels
		waveform = torch.mean(waveform, dim=0, keepdim=True)
	if dtype is not None:
		waveform = waveform.to(dtype)
	if device is not None:
		waveform = waveform.to(device)
	return waveform, sr

def process_items( items, stride=0, stride_offset=0 ):
	items = sorted( items )
	return items if stride == 0 else [ item for i, item in enumerate( items ) if (i+stride_offset) % stride == 0 ]

def process_job( outpath, waveform, sample_rate, text=None, language="en", device="cuda", dtype=None ):
	# encodec requires this to be on CPU for resampling
	state_dict = quantize(waveform, sr=sample_rate, device=device, dtype=dtype)


	if torch.count_nonzero(state_dict["codes"]) == 0:
		tqdm.write(f"Quantization returned zero'd tensor: {outpath}")
		return
				
	state_dict["codes"] = state_dict["codes"].cpu().numpy().astype(np.uint16)
	state_dict["conds"] = (
		state_dict["conds"][0].cpu().numpy().astype(np.float32),
		state_dict["conds"][1].cpu().numpy().astype(np.float32),
	)
	state_dict["latent"] = (
		state_dict["latent"][0].cpu().numpy().astype(np.float32),
		state_dict["latent"][1].cpu().numpy().astype(np.float32),
	)

	if text:
		text = text.strip()

		tokenized_text = cfg.tokenizer.encode(text)
		tokenized_text = np.array(tokenized_text).astype(np.uint8) 

		state_dict["text"] = tokenized_text
		state_dict['metadata'] |= {
			"text": text,
			"language": language,
		}

	np.save(open(outpath, "wb"), state_dict)

"""
def process_batched_jobs( jobs, speaker_id="", device=None, raise_exceptions=True, batch_size=1, dtype=None ):
	if not jobs:
		return

	# sort to avoid egregious padding
	jobs = sorted(jobs, key=lambda x: x[1].shape[-1], reverse=True)

	batches = []
	while jobs:
		batches.append(jobs[:batch_size])
		jobs = jobs[batch_size:]

	for batch in tqdm(batches, desc=f'Quantizing {speaker_id} (batch size: {batch_size})'):
		wavs = []
		srs = []
		
		for outpath, waveform, sample_rate, text, language in batch:
			wavs.append(waveform)
			srs.append(sample_rate)
		
		try:
			codes = quantize_batch(wavs, sr=srs, device=device, dtype=dtype)
		except Exception as e:
			_logger.error(f"Failed to quantize: {outpath}: {str(e)}")
			if raise_exceptions:
				raise e
			continue

		for (outpath, waveform, sample_rate, text, language), state_dict in zip( batch, codes ):
			if torch.count_nonzero(state_dict["codes"]) == 0:
				tqdm.write(f"Quantization returned zero'd tensor: {outpath}")
				continue

			if text:
				text = text.strip()
				state_dict['metadata'] |= {
					"text": text,
					"language": language,
				}
			
			np.save(open(outpath, "wb"), state_dict)
"""

def process_jobs( jobs, speaker_id="", device=None, raise_exceptions=True, batch_size=1, dtype=None ):
	if not jobs:
		return

	# batch things
	"""
	if batch_size > 1:
		return process_batched_jobs( jobs, speaker_id=speaker_id, device=device, raise_exceptions=raise_exceptions, batch_size=batch_size, dtype=dtype )
	"""
	for job in tqdm(jobs, desc=f"Quantizing: {speaker_id}"):
		outpath, waveform, sample_rate, text, language = job
		try:
			process_job( outpath, waveform, sample_rate, text, language, device, dtype=dtype )
		except Exception as e:
			_logger.error(f"Failed to quantize: {outpath}: {str(e)}")
			if raise_exceptions:
				raise e
			continue

def process(
	audio_backend="mel",
	input_audio="voices",
	input_voice=None,
	input_metadata="metadata",
	output_dataset="training",
	transcription_filename="whisper.json",
	raise_exceptions=False,
	verify_audio=False,
	stride=0,
	stride_offset=0,
	slice="auto",
	batch_size=1,
	max_duration=None,
	min_utterances=None,
	skip_existing_folders=False,
	low_memory=False,
	batch_threshold=0,
	strict_languages=False,

	device="cuda",
	dtype="float32",
	amp=False,
):
	# prepare from args
	cfg.device = device

	# to-do: cleanup
	# cfg.set_audio_backend(audio_backend)
	#audio_extension = cfg.audio_backend_extension
	audio_extension = ".mel"
	dtype = "float32"
	amp = False

	cfg.inference.weight_dtype = dtype # "bfloat16"
	cfg.inference.amp = amp # False

	dtype = cfg.inference.dtype if not amp else None

	output_dataset = f"{output_dataset}/24KHz-{cfg.audio_backend}" # "training"

	# to-do: make this also prepared from args
	language_map = {} # k = group, v = language

	ignore_groups = [] # skip these groups
	ignore_speakers = [] # skip these speakers

	only_groups = [] # only process these groups
	only_speakers = [] # only process these speakers

	always_slice_groups = ["Audiobooks", "LibriVox"] # always slice from this group
	audio_only = ["Noise"] # special pathway for processing audio only (without a transcription)

	missing = {
		"transcription": [],
		"audio": []
	}
	dataset = []
	jobs = []
	waveforms = {}

	def check_and_process_jobs(jobs, speaker_id=""):
		if len(jobs) < batch_threshold:
			return False
		
		process_jobs( jobs, device=device, speaker_id=speaker_id, raise_exceptions=raise_exceptions, batch_size=batch_size, dtype=dtype if not amp else None )
		return True

	if input_voice is not None:
		only_speakers = [input_voice]

	for group_name in sorted(os.listdir(f'./{input_audio}/')):
		if not os.path.isdir(f'./{input_audio}/{group_name}/'):
			_logger.warning(f'Is not dir:" /{input_audio}/{group_name}/')
			continue

		if group_name in ignore_groups:
			continue
		if only_groups and group_name not in only_groups:
			continue

		for speaker_id in tqdm(process_items(os.listdir(f'./{input_audio}/{group_name}/'), stride=stride, stride_offset=stride_offset), desc=f"Processing speaker in {group_name}"):
			if not os.path.isdir(f'./{input_audio}/{group_name}/{speaker_id}'):
				_logger.warning(f'Is not dir: ./{input_audio}/{group_name}/{speaker_id}')
				continue
			
			if speaker_id in ignore_speakers:
				continue
			if only_speakers and speaker_id not in only_speakers:
				continue
			
			outfolder = Path(f'./{output_dataset}/{group_name}/{speaker_id}/')

			if skip_existing_folders and outfolder.exists():
				continue
			
			outfolder.mkdir(parents=True, exist_ok=True)

			if speaker_id in audio_only:
				for filename in tqdm(sorted(os.listdir(f'./{input_audio}/{group_name}/{speaker_id}/')), desc=f"Processing {speaker_id}"):
					inpath = Path(f'./{input_audio}/{group_name}/{speaker_id}/{filename}')
					outpath = Path(f'./{output_dataset}/{group_name}/{speaker_id}/{filename}').with_suffix(audio_extension)

					if outpath.exists():
						continue

					waveform, sample_rate = load_audio( inpath, dtype=dtype )
					try:
						process_job( outpath, waveform, sample_rate, None, language="en", device=device, dtype=dtype if not amp else None)
					except Exception as e:
						_logger.error(f"Failed to quantize: {outpath}: {str(e)}")
						if raise_exceptions:
							raise e
						continue

				continue
			
			metadata_path = Path(f'./{input_metadata}/{group_name}/{speaker_id}/{transcription_filename}')
			if not metadata_path.exists():
				missing["transcription"].append(str(metadata_path))
				_logger.warning(f'Missing transcription metadata: ./{input_audio}/{group_name}/{speaker_id}/{transcription_filename}')
				continue

			try:
				metadata = json.loads(open(metadata_path, "r", encoding="utf-8").read())
			except Exception as e:
				missing["transcription"].append(str(metadata_path))
				_logger.warning(f'Failed to open transcription metadata: ./{input_audio}/{group_name}/{speaker_id}/{transcription_filename}: {e}')
				continue

			if f'{group_name}/{speaker_id}' not in dataset:
				dataset.append(f'{group_name}/{speaker_id}')


			use_slices = slice == True or (slice == "auto" and len(metadata.keys()) == 1) or group_name in always_slice_groups
			if min_utterances and len(metadata.keys()) < min_utterances:
				continue

			for filename in sorted(metadata.keys()):
				inpath = Path(f'./{input_audio}/{group_name}/{speaker_id}/{filename}')

				"""
				if not inpath.exists():
					missing["audio"].append(str(inpath))
					continue
				"""

				extension = os.path.splitext(filename)[-1][1:]
				fname = filename.replace(f'.{extension}', "")

				waveform, sample_rate = None, None
				language = language_map[group_name] if group_name in language_map else (metadata[filename]["language"] if "language" in metadata[filename] else "en")

				if language == "english":
					language = "en"
				elif language == "japanese":
					language = "ja"
				elif language == "french":
					language = "fr"
				elif language == "german":
					language = "de"
				elif language == "korean":
					language = "ko"
				elif language == "chinese":
					language = "zh"

				if strict_languages and language not in ["en", "ja", "fr", "de", "ko", "zh"]:
					language = "auto"

				if len(metadata[filename]["segments"]) == 0 or not use_slices:
					outpath = Path(f'./{output_dataset}/{group_name}/{speaker_id}/{fname}.{extension}').with_suffix(audio_extension)
					text = metadata[filename]["text"]

					if len(text) == 0 or outpath.exists():
						continue

					if max_duration:
						info = torchaudio.info( inpath )
						if info.num_frames / info.sample_rate > max_duration:
							continue

					waveform, sample_rate = load_audio( inpath, dtype=dtype )
					jobs.append(( outpath, waveform, sample_rate, text, language ))
				else:
					i = 0
					presliced = not inpath.exists()

					for segment in metadata[filename]["segments"]:
						id = pad(i, 4)
						i = i + 1

						if presliced:
							inpath = Path(f'./{input_audio}/{group_name}/{speaker_id}/{fname}_{id}.{extension}')

						if not inpath.exists():
							missing["audio"].append(str(inpath))
							continue

						outpath = Path(f'./{output_dataset}/{group_name}/{speaker_id}/{fname}_{id}.{extension}').with_suffix(audio_extension)
						text = segment["text"]

						if len(text) == 0 or outpath.exists():
							if not verify_audio:
								continue

							artifact = _load_artifact( outpath )
							if torch.count_nonzero(artifact) > 0:
								continue
							tqdm.write(f"Found zero'd quantized audio tensor: {outpath}")

						start = (segment['start']-0.05)
						end = (segment['end']+0.5)
						
						if max_duration and end - start > max_duration:
							continue

						# audio not already loaded, load it
						if waveform is None:
							waveform, sample_rate = load_audio( inpath, dtype=dtype )

						start = int(start * sample_rate)
						end = int(end * sample_rate)

						if not presliced:
							if start < 0:
								start = 0
							if end >= waveform.shape[-1]:
								end = waveform.shape[-1] - 1

							if end - start < 0:
								continue

						jobs.append(( outpath, waveform if presliced else waveform[:, start:end], sample_rate, text, language ))

			if check_and_process_jobs(jobs, speaker_id=speaker_id):
				jobs = []
	
	batch_threshold = 0
	check_and_process_jobs(jobs, speaker_id=speaker_id)

	open(f"./{output_dataset}/missing.json", 'w', encoding='utf-8').write(json.dumps(missing))
	open(f"./{output_dataset}/dataset.json", 'w', encoding='utf-8').write(json.dumps(dataset))

def main():
	parser = argparse.ArgumentParser()

	parser.add_argument("--audio-backend", type=str, default="mel")
	parser.add_argument("--input-audio", type=str, default="voices")
	parser.add_argument("--input-voice", type=str, default=None)
	parser.add_argument("--input-metadata", type=str, default="training/metadata")
	parser.add_argument("--output-dataset", type=str, default="training/dataset")
	parser.add_argument("--transcription-filename", type=str, default="whisper.json")
	parser.add_argument("--raise-exceptions", action="store_true")
	parser.add_argument("--verify-audio", action="store_true")
	#parser.add_argument("--low-memory", action="store_true")
	parser.add_argument("--skip-existing-folders", action="store_true")
	parser.add_argument("--strict-languages", action="store_true")
	parser.add_argument("--stride", type=int, default=0)
	parser.add_argument("--stride-offset", type=int, default=0)
	parser.add_argument("--slice", type=str, default="auto")
	parser.add_argument("--batch-size", type=int, default=0)
	parser.add_argument("--max-duration", type=int, default=0)
	parser.add_argument("--min-utterances", type=int, default=0)
	parser.add_argument("--batch-threshold", type=int, default=0)
	
	parser.add_argument("--device", type=str, default="cuda")
	parser.add_argument("--dtype", type=str, default="float32")
	parser.add_argument("--amp", action="store_true")
	
	args = parser.parse_args()

	# do some assumption magic
	# to-do: find a nice way to spawn multiple processes where tqdm plays nicely
	if args.device.isnumeric():
		args.stride = torch.cuda.device_count()
		args.stride_offset = int(args.device)
		args.device = f'cuda:{args.device}'

	if args.slice == "true":
		args.slice = True
	elif args.slice == "false":
		args.slice = False

	process(
		audio_backend=args.audio_backend,
		input_audio=args.input_audio,
		input_voice=args.input_voice,
		input_metadata=args.input_metadata,
		output_dataset=args.output_dataset,
		transcription_filename=args.transcription_filename,
		raise_exceptions=args.raise_exceptions,
		verify_audio=args.verify_audio,
		stride=args.stride,
		stride_offset=args.stride_offset,
		slice=args.slice,
		batch_size=args.batch_size,
		max_duration=args.max_duration,
		min_utterances=args.min_utterances,
		batch_threshold=args.batch_threshold,
		skip_existing_folders=args.skip_existing_folders,
		strict_languages=args.strict_languages,
		
		#low_memory=args.low_memory,

		device=args.device,
		dtype=args.dtype,
		amp=args.amp,
	)

if __name__ == "__main__":
	main()
