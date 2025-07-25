# todo: clean this mess up

import copy
import h5py
import json
import re
import logging
import numpy as np
import os
import random
import torch
import itertools

from .config import cfg
#from .emb.mel import post_process, trim, trim_random, repeat_extend_audio, concat_audio, merge_audio, decode_to_file, decode as decode_qnt, encode as encode_qnt, pad_codes_with_silence
from .emb.mel import decode_to_file, decode as decode_qnt, encode as encode_qnt
from .utils.sampler import PoolSampler, OrderedSampler, BatchedOrderedSampler, RandomSampler
from .utils.distributed import global_rank, local_rank, world_size, is_global_leader
from .utils.io import torch_save, torch_load, json_read, json_write, json_stringify, json_parse
from .utils import setup_logging

from collections import defaultdict
from functools import cache, cached_property
from itertools import groupby, zip_longest
from pathlib import Path
from typing import Any

from torch import Tensor
from torch.utils.data import DataLoader, Dataset as _Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.nn.utils.rnn import pad_sequence

from tqdm.auto import tqdm
# torch.multiprocessing.set_sharing_strategy("file_system")

_logger = logging.getLogger(__name__)

# cringe
"""
try:
	import nltk
	nltk.data.path.append("./.nltk/")
	if not Path(".nltk").exists():
		nltk.download('punkt_tab', download_dir="./.nltk/")
except Exception as e:
	nltk = None
	_logger.warning(f"Error while querying for NTLK: {str(e)}")
"""

def sentence_split( s, split_by="sentences", quote_placeholder="<QUOTE>" ):
	if split_by is None:
		return [s]

	# NTLK is not available, fallback
	if nltk is None:
		split_by = "\n"

	# split by delimiter instead
	if split_by != "sentences":
		return s.split(split_by)

	# use NTLK to handle splitting by sentences, because I don't want to write my own parser to split by punctuation
	# nltk does not split quotations all that nicely, so we coerce them into placeholders, then replace afterwards
	s = s.replace('"', quote_placeholder)
	sentences = nltk.sent_tokenize(s)
	return [ sentence.replace(quote_placeholder, '"') for sentence in sentences if sentence ]

# normalization code borrowed from TorToiSe TTS
# (it's not perfect but it works)

try:
	from tokenizers.normalizers import Lowercase, NFD, StripAccents
	
	normalizer = tokenizers.normalizers.Sequence([Lowercase(), NFD(), StripAccents()])
except Exception as e:
	normalizer = None

# List of (regular expression, replacement) pairs for abbreviations:
_abbreviations = [(re.compile('\\b%s\\.' % x[0], re.IGNORECASE), x[1]) for x in [
	('mrs', 'misess'),
	('mr', 'mister'),
	('dr', 'doctor'),
	('st', 'saint'),
	('co', 'company'),
	('jr', 'junior'),
	('maj', 'major'),
	('gen', 'general'),
	('drs', 'doctors'),
	('rev', 'reverend'),
	('lt', 'lieutenant'),
	('hon', 'honorable'),
	('sgt', 'sergeant'),
	('capt', 'captain'),
	('esq', 'esquire'),
	('ltd', 'limited'),
	('col', 'colonel'),
	('ft', 'fort'),
]]
def normalize_abbreviations(text):
	for regex, replacement in _abbreviations:
		text = re.sub(regex, replacement, text)
	return text

def _remove_commas(m):
	return m.group(1).replace(',', '')

def _expand_decimal_point(m):
	return m.group(1).replace('.', ' point ')

def _expand_dollars(m):
	match = m.group(1)
	parts = match.split('.')
	if len(parts) > 2:
		return match + ' dollars' # Unexpected format
	dollars = int(parts[0]) if parts[0] else 0
	cents = int(parts[1]) if len(parts) > 1 and parts[1] else 0
	if dollars and cents:
		dollar_unit = 'dollar' if dollars == 1 else 'dollars'
		cent_unit = 'cent' if cents == 1 else 'cents'
		return '%s %s, %s %s' % (dollars, dollar_unit, cents, cent_unit)
	elif dollars:
		dollar_unit = 'dollar' if dollars == 1 else 'dollars'
		return '%s %s' % (dollars, dollar_unit)
	elif cents:
		cent_unit = 'cent' if cents == 1 else 'cents'
		return '%s %s' % (cents, cent_unit)
	else:
		return 'zero dollars'

# in case the current env does not have it installed, so I don't need it as a hard dependency
try:
	import inflect

	_inflect = inflect.engine()

	def _expand_ordinal(m):
		return _inflect.number_to_words(m.group(0))

	def _expand_number(m):
		num = int(m.group(0))
		if num > 1000 and num < 3000:
			if num == 2000:
				return 'two thousand'
			elif num > 2000 and num < 2010:
				return 'two thousand ' + _inflect.number_to_words(num % 100)
			elif num % 100 == 0:
				return _inflect.number_to_words(num // 100) + ' hundred'
			else:
				return _inflect.number_to_words(num, andword='', zero='oh', group=2).replace(', ', ' ')
		else:
			return _inflect.number_to_words(num, andword='')
except Exception as e:
	_inflect = None

_comma_number_re = re.compile(r'([0-9][0-9\,]+[0-9])')
_decimal_number_re = re.compile(r'([0-9]+\.[0-9]+)')
_pounds_re = re.compile(r'£([0-9\,]*[0-9]+)')
_dollars_re = re.compile(r'\$([0-9\.\,]*[0-9]+)')
_ordinal_re = re.compile(r'[0-9]+(st|nd|rd|th)')
_number_re = re.compile(r'[0-9]+')
_whitespace_re = re.compile(r'\s+')
_end_punct_re = re.compile(r'[\.\?\!]$')
_aux_punct_re = re.compile(r'[,;:\?\.\!-]')

def normalize_numbers(text):
	text = re.sub(_comma_number_re, _remove_commas, text)
	text = re.sub(_pounds_re, r'\1 pounds', text)
	text = re.sub(_dollars_re, _expand_dollars, text)
	text = re.sub(_decimal_number_re, _expand_decimal_point, text)
	if _inflect is not None:
		text = re.sub(_ordinal_re, _expand_ordinal, text)
		text = re.sub(_number_re, _expand_number, text)
	return text

# full will do aggressive normalization, perfect for WER/CER
# not full will do basic cleaning
def normalize_text(text, language="auto", full=True):
	if full:
		if normalizer is not None:
			text = normalizer.normalize_str( text )
		else:
			text = text.lower()
		text = normalize_numbers(text) # expand numbers
		text = normalize_abbreviations(text) # expand abbreviations
		#text = re.sub(_end_punct_re, '', text) # collapse whitespace
		text = re.sub(_aux_punct_re, '', text) # collapse whitespace
		text = text.replace('"', '') # remove quotation marks
	else:
		text = normalize_numbers(text) # expand numbers
		text = normalize_abbreviations(text) # expand abbreviations
		text = re.sub(_whitespace_re, ' ', text) # collapse whitespace

	# to-do: other languages
	return text

@cache
def get_random_prompts( validation=False, min_length=0, tokenized=False, source_path=Path("./data/harvard_sentences.txt") ):
	duration_range = [ 5.5, 12.0 ] # to-do: pull from cfg.dataset.duration_range
	sentences = [
		"The birch canoe slid on the smooth planks.",
		"Glue the sheet to the dark blue background.",
		"It's easy to tell the depth of a well.",
		"These days a chicken leg is a rare dish.",
		"Rice is often served in round bowls.",
		"The juice of lemons makes fine punch.",
		"The box was thrown beside the parked truck.",
		"The hogs were fed chopped corn and garbage.",
		"Four hours of steady work faced us.",
		"A large size in stockings is hard to sell.",
		"The boy was there when the sun rose.",
		"A rod is used to catch pink salmon.",
		"The source of the huge river is the clear spring.",
		"Kick the ball straight and follow through.",
		"Help the woman get back to her feet.",
		"A pot of tea helps to pass the evening.",
		"Smoky fires lack flame and heat.",
		"The soft cushion broke the man's fall.",
		"The salt breeze came across from the sea.",
		"The girl at the booth sold fifty bonds.",
		"The small pup gnawed a hole in the sock.",
		"The fish twisted and turned on the bent hook.",
		"Press the pants and sew a button on the vest.",
		"The swan dive was far short of perfect.",
		"The beauty of the view stunned the young boy.",
		"Two blue fish swam in the tank.",
		"Her purse was full of useless trash.",
		"The colt reared and threw the tall rider.",
		"It snowed, rained, and hailed the same morning.",
		"Read verse out loud for pleasure.",
		"Perfect. Please move quickly to the chamber lock, as the effect of prolonged exposure to the button are not part of this test.",
	]

	if source_path.exists():
		sentences = open( source_path, "r", encoding="utf-8" ).read().split("\n")

	# Pull from validation dataset if existing + requested
	if validation and cfg.dataset.validation:
		paths = _load_paths(cfg.dataset.validation, type="validation", silent=True)
		paths = list(itertools.chain.from_iterable(paths.values()))
		
		for path in paths:
			duration = 0
			text_string = ""
			if cfg.dataset.use_hdf5:
				key = _get_hdf5_path(path)

				metadata = { f'{k}': f'{v}' for k, v in cfg.hdf5[key].attrs.items() }
				metadata = process_artifact_metadata( { "metadata": metadata } )
				text_string = metadata["text"] if "text" in metadata else ""
				duration = metadata['duration'] if "duration" in metadata else 0
			else:
				_, metadata = _load_artifact(path, return_metadata=True)
				metadata = process_artifact_metadata( { "metadata": metadata } )
				text_string = metadata["text"] if "text" in metadata else ""
				duration = metadata['duration'] if "duration" in metadata else 0
			
			if len( text_string ) < min_length or not (duration_range[0] <= duration and duration <= duration_range[1]):
				continue

			sentences.append( text_string )

	# tokenize here because our harvard sentences need to be phonemized anyways
	"""
	if tokenized:
		return [ torch.tensor( tokenize( encode_phns( text ) ) ).to(dtype=torch.uint8) for text in sentences ]
	"""

	return sentences

# samples a random text prompt
def get_random_prompt( *args, **kwargs ):
	# Harvard sentences
	return random.choice(get_random_prompts( *args, **kwargs ))

# to-do: clean up this symmap mess
def get_phone_symmap():
	return cfg.tokenizer.get_vocab()

def tokenize( phones ):
	if isinstance( phones, list ):
		phones = "".join( phones )
	return cfg.tokenizer.encode( phones )

def text_tokenize( text ):
	if isinstance( text, list ):
		text = "".join( text )
	return cfg.text_tokenizer.encode( text )

def get_lang_symmap():
	return {
		"en": 0,
		"ja": 1,
		"de": 2,
		"fr": 3,
		"zh": 4, # mandarin I presume
		"ko": 5,
	}

def get_tone_symmap():
	return {
		"neutral": 0,
		# could use 4 instead of 8 basic emotions
		# "joy": 1,
		# "fear": 2,
		# "surprise": 3,
		# "anger": 4,
	}

def get_task_symmap():
	return {
		"tts": 0,
		"tts-c": 1,
		"ns": 2,
		"sr": 3,
		"tse": 4,
		"soe": 5,
		"mask": 6,
		"eoe": 7,
		"stt": 8,

		"len": 0, # fake
		"nse": 6, # fake
		"cse": 6, # fake
		
		"phn": 0, # fake
		"un-phn": 0, # fake
	}

def _replace_file_extension(path, suffix):
	if not isinstance( path, Path ):
		path = Path(path)
	return (path.parent / path.name.split(".")[0]).with_suffix(suffix)

def _get_artifact_extension():
	#return ".dac" if cfg.audio_backend == "dac" else ".enc"
	return cfg.audio_backend_extension

def _get_metadata_extension():
	return ".json"

def _get_artifact_path(path):
	return _replace_file_extension(path, _get_artifact_extension())

def _get_path_key( type, dir, id ):
	return f"/{type}/{_get_hdf5_path(dir)}/{id}" if cfg.dataset.use_hdf5 else str(dir / id)

def _load_dataset_metadata(dataset, type="training", silent=not is_global_leader(), dataset_hash_key=None):
	assert cfg.dataset.min_duration >= 1.0, "Minimum duration too low."
	
	# for now only ensure metadata-based path
	assert cfg.dataset.use_metadata, "Metadata required."

	if not dataset_hash_key:
		dataset_hash_key = cfg.dataset.hash_key(sorted(dataset))

	cached_dir = cfg.cache_dir / dataset_hash_key
	cached_path = cached_dir / f"dataset[{type}].json"

	if cached_path.exists() and cfg.dataset.cache:
		return json_read( cached_path )

	dataset_metadata = {}
	def validate_utterance( id, entry ):
		duration = entry.get('duration', 0)
		in_bounds = cfg.dataset.min_duration <= duration and duration <= cfg.dataset.max_duration
		
		if cfg.dataset.validate and type == "training" and not in_bounds:
			return False

		if cfg.dataset.strict_validate:
			if cfg.dataset.use_hdf5 and key(type, dir, id) not in cfg.hdf5:
				return False

			if not (cfg.data_dir / dir / id).with_suffix(_get_artifact_extension()).exists():
				return False

		return True

	def process_utterance( id, entry, metadata_keys=None ):
		duration = entry.get('duration', 0)
		similar = entry.get('similar', None)
		# store duration length, and similar key name (because indices might shift)
		return [duration, ([ metadata_keys[idx] for idx in similar ] if similar and metadata_keys else [])]

	for dir in tqdm(dataset, desc=f"Parsing dataset: {type}", disable=silent ):
		metadata_path = cfg.metadata_dir / f'{dir}.json'
		if not metadata_path.exists():
			continue

		# to-do: make json_read handle when it actually can't read the file......
		try:
			metadata = json_read( metadata_path )
		except Exception as e:
			continue

		speaker = str(dir)
		metadata_keys = list(metadata.keys())
		dataset_metadata[speaker] = { id: process_utterance( id, entry, metadata_keys ) for id, entry in metadata.items() if validate_utterance( id, entry ) }

		# remap strings to indices
		remapped_indices = { k: i for i, k in enumerate(dataset_metadata[speaker].keys()) }
		for id, (duration, similars) in dataset_metadata[speaker].items():
			dataset_metadata[speaker][id][1] = [ remapped_indices[k] for k in similars if k in remapped_indices ]

	# and write if global leader (to avoid other processes writing to the same file at once)
	if is_global_leader():
		if not cached_dir.exists():
			cached_dir.mkdir(parents=True, exist_ok=True)

		json_write( dataset_metadata, cached_path, truncate=True )

	return dataset_metadata

def _get_paths_of_extensions( path, extensions=_get_artifact_extension(), validate=False ):
	if isinstance(path, str):
		path = Path(path)
	
	return [ str(p) for p in list(path.iterdir()) ] if path.exists() and path.is_dir() else []

def _load_artifact(path, return_metadata=False, return_artifact=False, validate=True) -> Tensor:
	artifact = np.load(_get_artifact_path(path), allow_pickle=True)[()]
	codes = artifact["codes"]
	
	if validate and np.count_nonzero(codes) == 0:
		raise Exception(f"Artifact contains zero'd tensor: {path}")

	codes = torch.from_numpy(codes.astype(int)).to(torch.int16)
	# codes = post_process( codes )

	if return_artifact:
		return codes, artifact

	if return_metadata:
		return codes, artifact["metadata"]

	return codes

def _interleaved_reorder(l, fn):
	groups = defaultdict(list)
	for e in l:
		groups[fn(e)].append(e)
	groups = {k: groups[k] for k in sorted(groups)}
	for interleaved in zip_longest(*groups.values()):
		for value in interleaved:
			if value is not None:
				yield value

class Dataset(_Dataset):
	def __init__(
		self,
		phone_symmap=None,
		training=False,
		extra_paths_by_speaker_name: dict[str, list] = {},
	):
		super().__init__()

		self._head = None
		self.sampler = None

		self.paths = []

		self.training = training
		self.dataset_type = "training" if self.training else "validation"
		self.sampler_type = cfg.dataset.sample_type if self.dataset_type == "training" else "path"
		self.sampler_order = cfg.dataset.sample_order
		self.sampler_shuffle = cfg.dataset.sample_shuffle if self.dataset_type == "training" else True

		dataset = sorted(cfg.dataset.training if self.training else cfg.dataset.validation)
		self.dataset_hash_key = cfg.dataset.hash_key(dataset)
		
		self.duration = 0
		self.duration_buckets = {}
		self.current_index = 0
		self.batch_size = cfg.hyperparameters.batch_size if self.training else cfg.evaluation.batch_size

		# hard error because I kept getting tricked by this myself
		if self.sampler_order == "duration" and self.sampler_type != "path":
			raise Exception(f'Requesting sample_type={self.sampler_type} with sample_order={self.sampler_order}, yet combination will not give expected results.')
		
		# dict that maps [speaker][id] to (duration, similar utterances)
		self.metadata = _load_dataset_metadata(dataset, self.dataset_type, dataset_hash_key=self.dataset_hash_key)

		if len(self.metadata) == 0:
			raise Exception(f'Empty dataset for {self.dataset_type}')
		
		# cull speakers with too little utterances
		prune_keys = [ speaker for speaker in self.metadata.keys() if len(self.metadata[speaker]) < cfg.dataset.min_utterances ]
		for speaker in prune_keys:
			del self.metadata[speaker]

		self.paths = []
		self.speakers = list(self.metadata.keys())
		self.paths = [ ((speaker_id, utterance_id), self.metadata[speaker][utterance][0]) for speaker_id, speaker in enumerate(self.speakers) for utterance_id, utterance in enumerate(self.metadata[speaker].keys()) ]

		# split dataset accordingly per GPU
		if cfg.distributed and self.training:
			self.paths = [ path for i, path in enumerate(self.paths) if i % world_size() == 0 ]

		for ((speaker_id, utterance_id), duration) in self.paths:
			self.duration += duration
			
			# only calc duration if we're going to order by duration
			if self.sampler_order != "duration":
				continue

			bucket = int(round(duration))
			if bucket not in self.duration_buckets:
				self.duration_buckets[bucket] = []
			self.duration_buckets[bucket].append( ( (speaker_id, utterance_id), duration ) )

		# sort by duration
		if self.sampler_order == "duration":
			# ensure they're ordered
			self.duration_buckets = dict(sorted(self.duration_buckets.items()))

			flattened = {}
			# sort and interleave
			for bucket in self.duration_buckets:
				# sort by duration
				self.duration_buckets[bucket].sort( key=lambda x: x[-1] )
				# split to retain tuples
				flattened[bucket] = self.duration_buckets[bucket]
				"""
				# replace with path
				flattened[bucket] = [ x[0] for x in flattened[bucket] ]
				"""
				# flatten by paths
				flattened[bucket] = [*_interleaved_reorder(flattened[bucket], lambda x: x[0])]
			# flatten paths
			self.paths = list(itertools.chain.from_iterable(flattened.values()))
		elif self.sampler_order == "random":
			random.shuffle( self.paths )
		else:
			# just interleave
			self.paths = [*_interleaved_reorder(self.paths, lambda x: x[0])]

		self.noise_metadata = _load_dataset_metadata(cfg.dataset.noise, "noise", dataset_hash_key=self.dataset_hash_key)
		self.noise_speakers = list(self.noise_metadata.keys())
		self.noise_paths = [ (speaker_id, utterance_id) for speaker_id, speaker in enumerate(self.noise_speakers) for utterance_id, utterance in enumerate(self.noise_metadata[speaker].keys()) ]

		self.phone_symmap = phone_symmap or self._get_phone_symmap()
		self.speaker_symmap = self._get_speaker_symmap()
		self.lang_symmap = self._get_lang_symmap()
		self.tone_symmap = self._get_tone_symmap()
		self.task_symmap = self._get_task_symmap()

		# grab IDs for bos, space, and eos for easy input creation later
		try:
			self.empty_text = [ cfg.tokenizer._bos_token, cfg.tokenizer.get_vocab()[" "], cfg.tokenizer._eos_token ]
		except Exception as e:
			self.empty_text = [None, None, None]

		# have it fetch at training time if any is invalid, because the tokenizer obj might not have it easily fetchable ahead of itme
		# encoding before parallelizing things causes things to whine
		if self.empty_text[0] is None or self.empty_text[-1] is None:
			self.empty_text = None

		# assert len(self.phone_symmap) < 256, "Unique token count should be [0,255] to fit within uint8"
		self.text_dtype = torch.uint8 if len(self.phone_symmap) < 256 else torch.int16

		if len(self.paths) == 0:
			raise ValueError(f"No valid path is found for {self.dataset_type}")

		if self.training and self.sampler_order == "duration" and cfg.dataset.sample_max_duration_batch > 0:
			self.sampler = BatchedOrderedSampler(
				self.duration_buckets if not self.sampler_state_dict_path.exists() else {}, # pass nothing if we're just going to load from a state anyways
				max_duration=cfg.dataset.sample_max_duration_batch,
				max_batch_size=self.batch_size,
				shuffle=self.sampler_shuffle,
			)
			self.batch_size = 1
		else:
			self.sampler = OrderedSampler( len(self) ) if not self.sampler_shuffle else RandomSampler( len(self) )

		self.load_state_dict()

	@cached_property
	def sampler_state_dict_path(self):
		return cfg.ckpt_dir / (cfg.lora.full_name if cfg.lora is not None else cfg.model.full_name) / f"sampler.{self.sampler_type}.rank{global_rank()}.pt"
		
	def get_speaker(self, path):
		if isinstance(path, str):
			path = Path(path)
		res = cfg.get_speaker(path)
		return res

	def get_speaker_group(self, path):
		if isinstance(path, str):
			path = Path(path)
		res = cfg.get_speaker_group(path)
		return res

	# this isn't really necessary since our data/metadata contains markers for languages, but this is still in in-case it's needed to force a language setting (for example, whisperX's lang isn't that accurate at times)
	def get_language(self, speaker_group, lang="en"):
		for k, v in cfg.dataset.speaker_languages.items():
			if speaker_group in v:
				lang = k
				break

		return lang.lower()

	@cached_property
	def tasks(self):
		if not self.training:
			return ["tts"]
		return cfg.dataset.tasks_list # ["tts", "tts", "ns", "sr", "tse", "tts", "tts"] # , "cse", "nse"

	def save_state_dict(self, path = None):
		if path is None:
			path = self.sampler_state_dict_path

		if not path.parent.exists():
			path.parent.mkdir(parents=True, exist_ok=True)

		state_dict = self.sampler.get_state()
		"""
		if self.sampler_type == "path":
			state_dict = self.sampler.get_state()
		else:
			state_dict = {
				"samplers": { name: sampler.get_state() for name, sampler in self.samplers.items() },
				"speaker_samplers": { name: sampler.get_state() for name, sampler in self.speaker_samplers.items() },
			}
		"""

		if "dataset_hash_key" not in state_dict:
			 state_dict["dataset_hash_key"] = self.dataset_hash_key

		torch_save(state_dict, path)

	def load_state_dict(self, path = None):
		if not self.training:
			return

		if path is None:
			path = self.sampler_state_dict_path

		if not path.exists():
			return

		state_dict = torch_load(path)
		if "dataset_hash_key" in state_dict:
			if self.dataset_hash_key != state_dict["dataset_hash_key"]:
				_logger.warning(f'Mismatched dataset hash key for {self.dataset_type} dataloader, ignoring loading of state dict.')
				return

		state_dict = self.sampler.set_state(state_dict)
		"""
		if self.sampler_type == "path":
			state_dict = self.sampler.set_state(state_dict)
		else:
			for name, sampler in state_dict["samplers"].items():
				if name not in self.samplers:
					continue
				self.samplers[name].set_state( sampler )

			for name, sampler in state_dict["speaker_samplers"].items():
				if name not in self.speaker_samplers:
					continue
				self.speaker_samplers[name].set_state( sampler )
		"""

	def _get_phone_symmap(self):
		return get_phone_symmap()

	def _get_speaker_symmap(self):
		return {s: i for i, s in enumerate(self.speakers)}

	def _get_lang_symmap(self):
		return get_lang_symmap()

	def _get_tone_symmap(self):
		return get_tone_symmap()

	def _get_task_symmap(self):
		return get_task_symmap()

	def sample_noise(self):
		speaker_id, utterance_id = random.choice(self.noise_paths)
		
		speaker_name = self.noise_speakers[speaker_id]
		utterance_name = list(self.noise_metadata[speaker_name].keys())[utterance_id]

		path = cfg.data_dir / speaker_name / utterance_name

		if cfg.dataset.use_hdf5:
			key = _get_hdf5_path(path)
			qnt = torch.from_numpy(cfg.hdf5[key]["audio"][:, :]).to(torch.int16)
		else:
			qnt = _load_artifact(path, return_metadata=False, return_artifact=False)
		return qnt

	def sample_speakers(self, ignore=[]):
		choices = set(self.speakers) - set(ignore)
		return random.choice([*choices])

	def sample_utterance(self, speaker_name, ignore=[]):
		choices = [*(set(self.metadata[speaker_name].keys()) - set(ignore))]

		if len(choices) == 0:
			return None, None, None
		
		utterance_id = random.choice(choices)
		utterance_name = list(self.metadata[speaker_name].keys())[utterance_id]
			
		path = cfg.data_dir / speaker_name / utterance_name
		if cfg.dataset.use_hdf5:
			key = _get_hdf5_path(path)

			if key not in cfg.hdf5:
				raise RuntimeError(f'Key of Path ({path}) not in HDF5: {key}')

			#metadata = cfg.hdf5[key].attrs
			metadata = { f'{k}': f'{v}' for k, v in cfg.hdf5[key].attrs.items() }

			text = cfg.hdf5[key]["text"][:]
			resps = cfg.hdf5[key]["audio"][:, :]
			
			text = torch.from_numpy(text).to(self.text_dtype)
			resps = torch.from_numpy(resps).to(torch.int16)
			
			"""
			lang = metadata["language"] if "language" in metadata else None
			tone = metadata["tone"] if "tone" in metadata else None
			"""
		else:
			resps, metadata = _load_artifact(path, return_metadata=True)
			text = torch.tensor(tokenize( metadata["phonemes"] )).to(self.text_dtype)

			"""
			lang = metadata["language"] if "language" in metadata else None
			tone = metadata["tone"] if "tone" in metadata else None
			"""

		return path, text, resps

	# icky slop
	def get_similar_utterance(self, speaker_name, utterance_name, offset=None ):
		if offset is None:
			offset = cfg.dataset.prompt_similar_top_k_offset

		_, similars = self.metadata[speaker_name][utterance_name]

		if not similars:
			return None

		if len(similars) >= offset:
			offset = 0

		# cringe stopgap
		offset_end = offset + cfg.dataset.prompt_similar_top_k

		if offset >= len( similars ):
			return None
		if offset_end >= len( similars ):
			return None

		utterance_keys = list(self.metadata[speaker_name].keys())
		if cfg.dataset.prompt_similar_top_k > 1:
			index = random.choice( similars[offset:offset_end] )
		else:
			index = similars[offset]

		return utterance_keys[index]

	def __getitem__(self, index):
		self.current_index = index

		"""
		if self.empty_text is None:
			self.empty_text = tokenize(" ")
		
		bos_id, space_id, eos_id = self.empty_text
		"""

		if self.sampler_type == "speaker":
			speaker_id = index
			speaker_name = self.speakers[speaker_id]
			utterance_name = random.choice( list(self.metadata[speaker_name].keys()) ) # random.choice(self.metadata[speaker_name])
		else:
			(speaker_id, utterance_id), duration = self.paths[index]
			speaker_name = self.speakers[speaker_id]
			utterance_name = list(self.metadata[speaker_name].keys())[utterance_id]
		
		path = cfg.data_dir / speaker_name / utterance_name

		"""
		if cfg.dataset.use_hdf5:
			key = _get_hdf5_path(path)

			if key not in cfg.hdf5:
				_logger.warning(f'Key of Path ({path}) not in HDF5: {key}')
				return dict(path=None)

			# cringe stopgap
			if "text" not in cfg.hdf5[key] or "audio" not in cfg.hdf5[key]:
				_logger.warning(f"text/audio not in entry: {key}")
				return dict(path=None)

			# I need to do some weird coersion to a normal dict because it'll bitch about Hdf5 objects not being pickleable in worker processes
			metadata = { f'{k}': f'{v}' for k, v in cfg.hdf5[key].attrs.items() }

			text = cfg.hdf5[key]["text"][:]
			resps = cfg.hdf5[key]["audio"][:, :]
			
			text = torch.from_numpy(text).to(self.text_dtype)
			resps = torch.from_numpy(resps).to(torch.int16)
			
			lang = metadata["language"] if "language" in metadata else None
			tone = metadata["tone"] if "tone" in metadata else None
			text_string = metadata["text"] if "text" in metadata else None

			if cfg.dataset.retokenize_text and "text" in metadata:
				text = torch.tensor(tokenize( metadata["text"] )).to(self.text_dtype)
		else:
			resps, metadata = _load_artifact(path, return_metadata=True)
			text = torch.tensor(tokenize( metadata["text"] )).to(self.text_dtype)

			lang = metadata["language"] if "language" in metadata else None
			tone = metadata["tone"] if "tone" in metadata else None
			text_string = metadata["text"] if "text" in metadata else None
		"""

		mel, artifact = _load_artifact(path, return_artifact=True)
		text = torch.from_numpy( artifact["text"].astype(int) ).to(self.text_dtype)
		"""
		conds = (
			torch.from_numpy(artifact["conds"][0].astype(float)).to(torch.float32),
			torch.from_numpy(artifact["conds"][1].astype(float)).to(torch.float32),
		)
		"""
		latents = (
			torch.from_numpy(artifact["latent"][0].astype(float)).to(torch.float32),
			torch.from_numpy(artifact["latent"][1].astype(float)).to(torch.float32)
		)
		wav_length = artifact["metadata"]["original_length"]

		return dict(
			index=index,
			path=Path(path),
			spkr_name=speaker_name,
			spkr_id=speaker_id,

			latents_0=latents[0][0],
			latents_1=latents[1][0],
			
			#conds_0=conds[0][0, 0],
			#conds_1=conds[1][0, 0],

			text=text,
			mel=mel,
			wav_length=wav_length,
		)

	def head_(self, n):
		self._head = n

	def training_(self, value):
		self.training = value

	def index(self):
		return (self.sampler.index() if self.sampler is not None else -1) // self.batch_size
	
	def batches(self):
		if isinstance(self.sampler, BatchedOrderedSampler):
			return len(self.sampler)
		return len(self.sampler if self.sampler is not None else self) // self.batch_size

	def __len__(self):
		if self.sampler_type == "speaker":
			return min(len(self.speakers), self._head or len(self.speakers))
		return min(len(self.paths), self._head or len(self.paths))


def collate_fn(samples: list[dict]):
	samples = [ s for s in samples if s["path"] is not None ]
	batch: dict[str, Any] = {k: [s[k] for s in samples] for k in samples[0]}
	return batch


def _seed_worker(worker_id):
	worker_seed = torch.initial_seed() % 2**32
	np.random.seed(worker_seed)
	random.seed(worker_seed)


def _create_dataloader(dataset, training):
	kwargs = dict(
		shuffle=not training,
		batch_size=cfg.hyperparameters.batch_size if training else cfg.evaluation.batch_size,
		drop_last=training,
		sampler=dataset.sampler if training else None,
	) if not isinstance(dataset.sampler, BatchedOrderedSampler) else dict(
		batch_sampler=dataset.sampler,
	)

	return DataLoader(
		dataset=dataset,
		num_workers=cfg.dataset.workers,
		collate_fn=collate_fn,
		persistent_workers=cfg.dataset.workers > 1,
		pin_memory=True,
		worker_init_fn=_seed_worker,
		**kwargs,
	)

def create_datasets():
	train_dataset = Dataset( training=True )
	val_dataset = Dataset( phone_symmap=train_dataset.phone_symmap, training=False )

	return train_dataset, val_dataset

def create_train_dataloader():
	train_dataset = Dataset( training=True )
	train_dl = _create_dataloader(train_dataset, training=True)

	_logger.info(str(train_dataset.phone_symmap))
	_logger.info(str(train_dataset.speaker_symmap))
	
	_logger.info(f"#samples (train): {len(train_dataset)}.")
	_logger.info(f"#duration (train): {str(train_dataset.duration)}.")

	# remove duration map (it gets bloated)
	_durations_map = {}

	return train_dl

def create_val_dataloader():
	val_dataset = Dataset( training=False )
	val_dl = _create_dataloader(val_dataset, training=False)

	_logger.info(str(val_dataset.phone_symmap))
	_logger.info(str(val_dataset.speaker_symmap))
	
	_logger.info(f"#samples (val): {len(val_dataset)}.")
	_logger.info(f"#duration (val): {str(val_dataset.duration)}.")

	# remove duration map (it gets bloated)
	_durations_map = {}

	return val_dl

# to-do, use the above two, then create the subtrain dataset
def create_train_val_dataloader():
	train_dataset, val_dataset = create_datasets()
	train_dl = _create_dataloader(train_dataset, training=True)
	val_dl = _create_dataloader(val_dataset, training=False)

	_logger.info(str(train_dataset.phone_symmap))
	_logger.info(f'#speakers (train): {len(train_dataset.speaker_symmap)}')

	_logger.info(f"#samples (train): {len(train_dataset)}.")
	_logger.info(f"#samples (val): {len(val_dataset)}.")

	_logger.info(f"#duration (train): {str(train_dataset.duration)}.")
	_logger.info(f"#duration (val): {str(val_dataset.duration)}.")

	# remove duration map (it gets bloated)
	_durations_map = {}

	return train_dl, val_dl

# parse metadata from an numpy file (.enc/.dac) and validate it
def process_artifact_metadata( artifact ):
	metadata = {}

	# text transcription (just in case)
	if "text" in artifact["metadata"]:
		metadata["text"] = artifact["metadata"]["text"]
	# phonemization of text transcription (just in case)
	if "phonemes" in artifact["metadata"]:
		metadata["phonemes"] = artifact["metadata"]["phonemes"]
	# language for sampling / input creation
	if "language" in artifact["metadata"]:
		metadata["language"] = artifact["metadata"]["language"]
	# top-k similar utterances for this utternace
	if "similar" in artifact["metadata"]:
		metadata["similar"] = artifact["metadata"]["similar"]
	# duration for use of culling / sorting dataset
	if "duration" in artifact["metadata"]:
		metadata["duration"] = float(artifact["metadata"]["duration"])
	# derive duration from sample count / sample rate
	elif "original_length" in artifact["metadata"] and "sample_rate" in artifact["metadata"]:
		metadata["duration"] = artifact["metadata"]["original_length"] / artifact["metadata"]["sample_rate"]

	"""
	# rephonemize if required
	if "phonemes" not in metadata and "text" in metadata:
		metadata["phonemes"] = encode_phns( metadata["text"], language=metadata["language"] if "language" in metadata["language"] else "en" )

	# clean up phonemes from espeak
	#     for example: Sonnenküste Update => zˈɔnənkˌystə (en)ˈʌpdeɪt(de)
	# to-do: regex replace /([a-z]{2})/ to ""
	if "phonemes" in metadata:
		metadata["phonemes"] = metadata["phonemes"].replace("(en)", "")
		if "language" in metadata:
			metadata["phonemes"] = metadata["phonemes"].replace(f"({metadata['language']})", "")
		metadata["phonemes"] = re.sub(r'\([a-z]{2}\)', "", metadata["phonemes"])
	"""

	return metadata

# yucky, but I would like to have the LibriTTS-R utterances remapped to their LibriSpeech counterpart
# to-do: allow this to be adjusted without having to regenerate metadata / HDF5 by remapping name during dataloader creation
def remap_speaker_name( name ):
	# commented out because I don't want the LibriSpeech portion of the dataset to get added
	"""
	if "LibriTTS-R" in speaker_name:
		name = name.replace("LibriTTS-R", "LibriVox")
	"""
	return name

# parse dataset into better to sample metadata
def create_dataset_metadata( skip_existing=False ):
	symmap = get_phone_symmap()
	
	root = str(cfg.data_dir)
	metadata_root = str(cfg.metadata_dir)

	cfg.metadata_dir.mkdir(parents=True, exist_ok=True)

	def add( dir, type="training", audios=True, texts=True ):
		name = str(dir)
		name = name.replace(root, "")
		speaker_name = remap_speaker_name( name )

		metadata_path = Path(f"{metadata_root}/{speaker_name}.json")
		metadata_path.parents[0].mkdir(parents=True, exist_ok=True)

		metadata = json_read( metadata_path, default={} )

		if not os.path.isdir(f'{root}/{name}/'):
			return

		files = os.listdir(f'{root}/{name}/')

		# grab IDs for every file
		ids = { file.replace(_get_artifact_extension(), "").replace(_get_metadata_extension(), "") for file in files }

		wrote = False

		for id in tqdm(ids, desc=f"Processing {name}", disable=True):
			try:
				quant_path = Path(f'{root}/{name}/{id}{_get_artifact_extension()}')

				if audios and not quant_path.exists():
					continue

				key = f'{type}/{speaker_name}/{id}'

				if skip_existing and id in metadata:
					continue
				
				wrote = True

				if id not in metadata:
					metadata[id] = {}

				utterance_metadata = {}
				if audios:
					qnt, artifact = _load_artifact(quant_path, return_artifact=True)
					utterance_metadata = process_artifact_metadata( artifact )
					# to-do: derive duration from codes if duration is malformed because this happened to me with LibriTTS-R
					#utterance_metadata["duration"] = qnt.shape[0] / cfg.dataset.frames_per_second

				for k, v in utterance_metadata.items():
					metadata[id][k] = v

			except Exception as e:
				tqdm.write(f'Error while processing {id}: {e}')

		if wrote:
			json_write( metadata, metadata_path )

	# training
	for data_dir in tqdm(sorted(cfg.dataset.training), desc="Processing Training"):
		add( data_dir, type="training" )

	# validation
	for data_dir in tqdm(sorted(cfg.dataset.validation), desc='Processing Validation'):
		add( data_dir, type="validation" )

	# noise
	for data_dir in tqdm(sorted(cfg.dataset.noise), desc='Processing Noise'):
		add( data_dir, type="noise", texts=False )

# parse yaml to create an hdf5 file
def create_dataset_hdf5( skip_existing=True ):
	cfg.dataset.use_hdf5 = True
	cfg.load_hdf5(write=True)
	hf = cfg.hdf5

	symmap = get_phone_symmap()
	
	root = str(cfg.data_dir)
	metadata_root = str(cfg.metadata_dir)


	def add( dir, type="training", audios=True, texts=True, verbose=False ):
		name = str(dir)
		name = name.replace(root, "")
		speaker_name = remap_speaker_name( name )

		metadata_path = Path(f"{metadata_root}/{speaker_name}.json")
		metadata_path.parents[0].mkdir(parents=True, exist_ok=True)

		try:
			metadata = json_read(metadata_path, default={})
		except Exception as e:
			print(metadata_path, e)
			return

		if not os.path.isdir(f'{root}/{name}/'):
			return

		files = os.listdir(f'{root}/{name}/')

		# grab IDs for every file
		ids = { file.replace(_get_artifact_extension(), "").replace(_get_metadata_extension(), "") for file in files }

		"""
		# rephonemizes if you fuck up and use and old tokenizer...
		for id, entry in tqdm(metadata.items(), desc=f"Processing {name}"):
			key = f'{type}/{speaker_name}/{id}'

			if key not in hf:
				continue
			
			group = hf[key]

			if "phonemes" not in entry:
				continue
			if "text" not in group:
				continue

			txt = entry["phonemes"]
			phn = "".join(txt)
			phn = cfg.tokenizer.encode(phn)
			phn = np.array(phn).astype(np.uint8) 

			del group["text"]
			group.create_dataset('text', data=phn, compression='lzf')
		"""

		for id in tqdm(ids, desc=f"Processing {name}", disable=not verbose):
			try:
				quant_exists = os.path.exists(f'{root}/{name}/{id}{_get_artifact_extension()}') if audios else True
				text_exists = os.path.exists(f'{root}/{name}/{id}{_get_metadata_extension()}') if texts else True

				if not quant_exists:
					continue

				key = f'{type}/{speaker_name}/{id}'

				if skip_existing and key in hf:
					continue

				group = hf.create_group(key) if key not in hf else hf[key]

				if id not in metadata:
					metadata[id] = {}

				utterance_metadata = {}

				# audio
				if audios:
					qnt, artifact = _load_artifact(f'{root}/{name}/{id}{_get_artifact_extension()}', return_artifact=True)
					utterance_metadata = process_artifact_metadata( artifact )

					if "audio" not in group:
						group.create_dataset('audio', data=qnt.numpy().astype(np.int16), compression='lzf')

				# text
				# this is a relic from when I did have the quantized audio and phoneme transcription separate
				# to-do: ensure I can remove this block
				if texts:
					if not utterance_metadata and text_exists:
						utterance_metadata = json_read(f'{root}/{name}/{id}{_get_metadata_extension()}')

					phn = "".join(utterance_metadata["text"])
					phn = cfg.tokenizer.encode(phn)
					phn = np.array(phn).astype(np.uint8) 

					if "text" not in group:
						group.create_dataset('text', data=phn, compression='lzf')

				for k, v in utterance_metadata.items():
					group.attrs[k] = v
					metadata[id][k] = v

			except Exception as e:
				tqdm.write(f'Error while processing {id}: {e}')

		json_write( metadata, metadata_path )

	# training
	for data_dir in tqdm(cfg.dataset.training, desc="Processing Training"):
		add( data_dir, type="training" )

	# validation
	for data_dir in tqdm(cfg.dataset.validation, desc='Processing Validation'):
		add( data_dir, type="validation" )

	# noise
	for data_dir in tqdm(cfg.dataset.noise, desc='Processing Noise'):
		add( data_dir, type="noise", texts=False )

	# write symmap
	if "symmap" in hf:
		del hf['symmap']

	hf.create_dataset('symmap', data=json_stringify(symmap))
	hf.close()

if __name__ == "__main__":
	import argparse

	parser = argparse.ArgumentParser("Save trained model to path.")
	parser.add_argument("--action", type=str)
	parser.add_argument("--tasks", type=str)
	args, unknown = parser.parse_known_args()

	task = args.action

	setup_logging()
	cfg.dataset.workers = 1

	if args.action == "hdf5":
		create_dataset_hdf5()
	elif args.action == "list-dataset":
		dataset = []
		for group in os.listdir(cfg.data_dir):
			for name in os.listdir(cfg.data_dir / group):
				if len(os.listdir(cfg.data_dir / group / name)) == 0:
					continue
				dataset.append(f'{group}/{name}')

		_logger.info(json_stringify(dataset))
	elif args.action == "metadata":
		create_dataset_metadata()
	elif args.action == "sample":
		train_dl, val_dl = create_train_val_dataloader()

		samples = {
			"training": [ next(iter(train_dl)),  next(iter(train_dl)) ],
			"validation": [ next(iter(val_dl)),  next(iter(val_dl)) ],
		}

		Path("./data/sample-test/").mkdir(parents=True, exist_ok=True)

		for k, v in samples.items():
			for i in range(len(v)):
				for j in tqdm(range(len(v[i]['proms'])), desc="Decoding..."):
					"""
					"""
					try:
						decode_to_file( v[i]['proms'][j], f"./data/sample-test/{k}.{i}.{j}.proms.wav", device="cpu" )
					except Exception as e:
						_logger.info(f"Error while decoding prom {k}.{i}.{j}.wav: {str(e)}")
					try:
						decode_to_file( v[i]['resps'][j], f"./data/sample-test/{k}.{i}.{j}.resps.wav", device="cpu" )
					except Exception as e:
						_logger.info(f"Error while decoding resp {k}.{i}.{j}.wav: {str(e)}")
					#v[i]['proms'][j] = v[i]['proms'][j].shape
					#v[i]['resps'][j] = v[i]['resps'][j].shape
		
		for k, v in samples.items():
			for i in range(len(v)):
				_logger.info(f'{k}[{i}]: {v[i]}')
	elif args.action == "validate":
		train_dl, val_dl = create_train_val_dataloader()
		dataset = train_dl.dataset

		missing = []
		symmap = get_phone_symmap()

		for index in tqdm(range(len( dataset )), desc="Processing dataset..."):
			if dataset.sampler_type == "speaker":
				speaker_id = index
				speaker_name = dataset.speakers[speaker_id]
				utterance_name = random.choice( list(dataset.metadata[speaker_name].keys()) ) # random.choice(dataset.metadata[speaker_name])
			else:
				speaker_id, utterance_id = dataset.paths[index]
				speaker_name = dataset.speakers[speaker_id]
				speaker_keys = list(dataset.metadata[speaker_name].keys())
				utterance_name = speaker_keys[utterance_id]

			path = cfg.data_dir / speaker_name / utterance_name

			if cfg.dataset.use_hdf5:
				key = _get_hdf5_path(path)
				if key not in cfg.hdf5:
					continue
				metadata = { f'{k}': f'{v}' for k, v in cfg.hdf5[key].attrs.items() }
			else:
				_, metadata = _load_artifact(path, return_metadata=True)
			
			phonemes = metadata["phonemes"]

			for i, phone in enumerate( phonemes ):
				if phone in symmap:
					continue
				if phone in missing:
					continue

				_logger.info( f"{path} | {phonemes}[{i}] | {phone}" )
				missing.append( phone )

			"""
			text = tokenize( phonemes )[1:-1]
			unk_token = tokenize("<unk>")[1]

			if unk_token in text:
				print( unk_token, text, phonemes )

			for i, token in enumerate(text):
				if token != unk_token:
					continue
				
				phone = phonemes[i]
				if phone not in missing:
					_logger.info( f"{path} | {phonemes}[{i}] | {phone}" )
				missing |= set([phone])
			"""

		_logger.info( f"Missing tokens: {missing}" )


	elif args.action == "tasks":
		index = 0
		cfg.dataset.tasks_list = args.tasks.split(",")
		
		train_dl, val_dl = create_train_val_dataloader()
		batch = next(iter(train_dl))

		for text, resps, proms, task in zip(batch["text"], batch["resps"], batch["proms"], batch["task"]):
			if task not in cfg.dataset.tasks_list:
				continue

			_logger.info( f'{text} {task} {cfg.model.resp_levels}')
			_logger.info( f'{proms.shape} {resps.shape}' )

			tokens = 0
			tokens += sum([ text.shape[0] for text in batch["text"] ])
			tokens += sum([ resps.shape[0] for resps in batch["resps"] ])
			_logger.info( f'{tokens}' )

			decode_to_file( proms, f"./data/{task}.proms.wav", device="cpu" )
			decode_to_file( resps, f"./data/{task}.resps.wav", device="cpu" )
			break
