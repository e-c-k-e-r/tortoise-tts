import os
import json
import torch
import numpy as np

from tqdm.auto import tqdm
from pathlib import Path

from vall_e.config import cfg

# things that could be args
cfg.sample_rate = 24_000
cfg.inference.audio_backend = "encodec"
"""
cfg.inference.weight_dtype = "bfloat16"
cfg.inference.dtype = torch.bfloat16
cfg.inference.amp = True
"""

from vall_e.emb.g2p import encode as valle_phonemize
from tortoise_tts.emb.mel import encode as tortoise_mel_encode, _replace_file_extension

audio_extension = ".mel"

input_dataset = "LibriTTS_R"
output_dataset = f"LibriTTS-Train-{'2' if cfg.sample_rate == 24_000 else '4'}4KHz"
device = "cuda"

txts = []
wavs = []

for dataset_name in os.listdir(f'./{input_dataset}/'):
	if not os.path.isdir(f'./{input_dataset}/{dataset_name}/'):
		continue

	for speaker_id in tqdm(os.listdir(f'./{input_dataset}/{dataset_name}/'), desc="Processing speaker"):
		if not os.path.isdir(f'./{input_dataset}/{dataset_name}/{speaker_id}'):
			continue
		
		os.makedirs(f'./{output_dataset}/{speaker_id}/', exist_ok=True)
		for book_id in os.listdir(f'./{input_dataset}/{dataset_name}/{speaker_id}'):
			if not os.path.isdir(f'./{input_dataset}/{dataset_name}/{speaker_id}/{book_id}'):
				continue
			for filename in os.listdir(f'./{input_dataset}/{dataset_name}/{speaker_id}/{book_id}'):
				# os.rename(f'./{input_dataset}/{dataset_name}/{speaker_id}/{book_id}/{filename}', f'./{output_dataset}/{speaker_id}/{filename}')

				inpath = Path(f'./{input_dataset}/{dataset_name}/{speaker_id}/{book_id}/{filename}')
				outpath = Path(f'./{output_dataset}/{speaker_id}/{filename}')
				
				if ".wav" in filename: # and not _replace_file_extension(outpath, ".dac").exists():
					txts.append((
						inpath,
						outpath
					))

for paths in tqdm(txts, desc="Processing..."):
	inpath, outpath = paths
	try:
		text = open(_replace_file_extension(inpath, ".original.txt"), "r", encoding="utf-8").read()
		
		mel = valle_quantize(_replace_file_extension(inpath, ".wav"), device=device)
		mel["text"] = text
		np.save(open(_replace_file_extension(outpath, audio_extension), "wb"), mel)
	except Exception as e:
		tqdm.write(f"Failed to process: {paths}: {e}")
