# todo: clean this mess up

import copy
import h5py
import json
import logging
import numpy as np
import os
import random
import torch
import itertools

from .config import cfg
from .emb.mel import trim, trim_random, repeat_extend_audio, merge_audio, decode_to_file
from .utils.sampler import PoolSampler, OrderedSampler, RandomSampler
from .utils.distributed import global_rank, local_rank, world_size

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

# to-do: clean up this symmap mess
def get_phone_symmap():
	return cfg.tokenizer.get_vocab()

def tokenize( phones ):
	return cfg.tokenizer.encode( "".join(phones) )

def get_lang_symmap():
	return {
		"en": 0,
		"ja": 1,
	}

def get_tone_symmap():
	return {
		"neutral": 0,
	}
	return symmap

def get_task_symmap():
	return {
		"<tts>": 0,
		"<tts-c>": 1,
		"<ns>": 2,
		"<sr>": 3,
		"<tse>": 4,
		"<soe>": 5,
		"<mask>": 6,
		"<eoe>": 7,
	}

def _replace_file_extension(path, suffix):
	return (path.parent / path.name.split(".")[0]).with_suffix(suffix)

def _get_mel_extension():
	return ".mel"

def _get_phone_extension():
	return ".json"

def _get_mel_path(path):
	return _replace_file_extension(path, _get_mel_extension())

def _get_phone_path(path):
	return _replace_file_extension(path, _get_phone_extension())

_durations_map = {}
# makeshift caching the above to disk
@cfg.diskcache()
def _get_duration_map( type="training" ):
	return _durations_map[type] if type in _durations_map else {}

@cfg.diskcache()
def _load_paths(dataset, type="training"):
	return { cfg.get_spkr( cfg.data_dir / data_dir / "dummy" ): _load_paths_from_metadata( data_dir, type=type, validate=cfg.dataset.validate and type == "training" ) for data_dir in tqdm(dataset, desc=f"Parsing dataset: {type}") }

def _load_paths_from_metadata(group_name, type="training", validate=False):
	data_dir = group_name if cfg.dataset.use_hdf5 else cfg.data_dir / group_name

	_fn = _get_hdf5_paths if cfg.dataset.use_hdf5 else _get_paths_of_extensions

	def key( id, entry=None ):
		return f"/{type}/{_get_hdf5_path(data_dir)}/{id}" if cfg.dataset.use_hdf5 else data_dir / id

	metadata_path = cfg.metadata_dir / f'{group_name}.json'
	metadata = {}

	if cfg.dataset.use_metadata and metadata_path.exists():
		metadata = json.loads(open( metadata_path, "r", encoding="utf-8" ).read())

	if len(metadata) == 0:
		return _fn( data_dir, type if cfg.dataset.use_hdf5 else _get_mel_extension(), validate )

	def _validate( id, entry ):
		phones = entry['phones'] if "phones" in entry else 0
		duration = entry['duration'] if "duration" in entry else 0

		# add to duration bucket
		k = key(id, entry)
		if type not in _durations_map:
			_durations_map[type] = {}
		_durations_map[type][k] = duration

		if not validate:
			return True

		return cfg.dataset.min_duration <= duration and duration <= cfg.dataset.max_duration

	return [ key(id, entry) for id, entry in metadata.items() if _validate(id, entry) ]


def _get_hdf5_path(path):
	# to-do: better validation
	#print(path)
	return str(path)

def _get_hdf5_paths( data_dir, type="training", validate=False ):
	data_dir = str(data_dir)
	
	key = f"/{type}/{_get_hdf5_path(data_dir)}"

	def _validate( id, entry ):
		phones = entry.attrs['phonemes']
		duration = entry.attrs['duration']

		if type not in _durations_map:
			_durations_map[type] = {}
		_durations_map[type][f"{key}/{id}"] = duration

		if not validate:
			return True
		
		return cfg.dataset.min_duration <= duration and duration <= cfg.dataset.max_duration

	return [ Path(f"{key}/{id}") for id, entry in cfg.hdf5[key].items() if _validate(id, entry) ] if key in cfg.hdf5 else []

def _get_paths_of_extensions( path, extensions=_get_mel_extension(), validate=False ):
	if isinstance(path, str):
		path = Path(path)

	def _validate(path):
		if "".join(path.suffixes) not in extensions:
			return False
		if not _get_phone_path(path).exists() or not _get_mel_path(path).exists():
			return False
		if not validate:
			return True
		# to-do: find an easy way to determine size from pickled mels without loading
		# to-do: find a consistent way to derive phoneme count from filesize (probably can't due to utf-8)
		phones = len(_get_phones(_get_phone_path(path))) # _get_phone_path(path).stat().st_size // 2 + 1
		return cfg.dataset.min_phones <= phones and phones <= cfg.dataset.max_phones


	return [ p for p in list(path.iterdir()) if _validate(p) ] if path.exists() and path.is_dir() else []

def _load_mels(path, return_metadata=False) -> Tensor:
	mel = np.load(_get_mel_path(path), allow_pickle=True)[()]
	if return_metadata:
		mel["metadata"]["text"] = mel["text"]

		return mel["codes"].to(torch.int16), mel["metadata"]
	return mel["codes"].to(torch.int16)

# prune consecutive spaces
def _cleanup_phones( phones, targets=[" "]):
	return [ p for i, p in enumerate(phones) if p not in targets or ( p in targets and p != phones[i-1] ) ]

@cache
def _get_phones(path):
	phone_path = _get_phone_path(path)
	mel_path = _get_mel_path(path)
	if phone_path.exists():
		metadata = json.loads(open(phone_path, "r", encoding="utf-8").read())
	elif mel_path.exists():
		_, metadata = _load_mels( path, return_metadata=True )
	else:
		raise Exception(f"Could not load phonemes: {path}")

	content = metadata["phonemes"]
	return "".join(content)

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
		extra_paths_by_spkr_name: dict[str, list] = {},
	):
		super().__init__()
		self._head = None
		self.shuffle = False
		self.sampler = None

		self.paths = []

		self.training = training
		self.dataset_type = "training" if self.training else "validation"
		self.dataset = cfg.dataset.training if self.training else cfg.dataset.validation
		self.sampler_type = cfg.dataset.sample_type # if self.dataset_type == "training" else "group"
		self.sampler_order = cfg.dataset.sample_order

		# to-do: do not do validation if there's nothing in the validation
		# this just makes it be happy
		if len(self.dataset) == 0:
			self.dataset = cfg.dataset.training
		
		# dict of paths keyed by speaker names
		self.paths_by_spkr_name = _load_paths(self.dataset, self.dataset_type)

		# cull speakers if they do not have enough utterances
		if cfg.dataset.min_utterances > 0:
			keys = list(self.paths_by_spkr_name.keys())
			for key in keys:
				if len(self.paths_by_spkr_name[key]) < cfg.dataset.min_utterances:
					del self.paths_by_spkr_name[key]

		# flatten paths
		self.paths = list(itertools.chain.from_iterable(self.paths_by_spkr_name.values()))
		
		# split dataset accordingly per GPU
		if cfg.distributed and self.training:
			batches = len(self.paths) // world_size()
			start = batches * global_rank()
			end = batches * (global_rank() + 1)

			self.paths = self.paths[start:end]

			# recreate paths_by_spkr_name
			self.paths_by_spkr_name = {}
			for path in self.paths:
				name = cfg.get_spkr( Path(path) )
				if name not in self.paths_by_spkr_name:
					self.paths_by_spkr_name[name] = []
				self.paths_by_spkr_name[name].append( path )

		# do it here due to the above
		self.duration = 0
		self.duration_map = _get_duration_map( self.dataset_type )
		self.duration_buckets = {}

		# store in corresponding bucket
		for path in self.paths:
			duration = self.duration_map[path]
			self.duration += duration
			
			# only calc duration if we're tot going to order by duration
			if self.sampler_order != "duration":
				continue

			bucket = str(int(round(duration)))
			if bucket not in self.duration_buckets:
				self.duration_buckets[bucket] = []
			self.duration_buckets[bucket].append( ( Path(path), duration ) )

		# sort by duration
		if self.sampler_order == "duration":
			# sort and interleave
			for bucket in self.duration_buckets:
				# sort by duration
				self.duration_buckets[bucket].sort( key=lambda x: x[1] )
				# replace with path
				self.duration_buckets[bucket] = [ x[0] for x in self.duration_buckets[bucket] ]
				# flatten by paths
				self.duration_buckets[bucket] = [*_interleaved_reorder(self.duration_buckets[bucket], self.get_speaker)]
			# flatten paths
			self.paths = list(itertools.chain.from_iterable(self.duration_buckets.values()))
		elif self.sampler_order == "shuffle":
			# just interleave
			self.paths = [*_interleaved_reorder(self.paths, self.get_speaker)]

		
		# dict of speakers keyed by speaker group
		self.spkrs_by_spkr_group = {}
		for data_dir in self.dataset:
			spkr = cfg.get_spkr( data_dir / "dummy" )
			spkr_group = cfg.get_spkr_group( data_dir / "dummy" )

			if spkr not in self.paths_by_spkr_name or len(self.paths_by_spkr_name[spkr]) < cfg.dataset.min_utterances:
				continue

			if spkr_group not in self.spkrs_by_spkr_group:
				self.spkrs_by_spkr_group[spkr_group] = []

			self.spkrs_by_spkr_group[spkr_group].append( spkr )

		self.spkr_groups = list(self.spkrs_by_spkr_group.keys())

		self.noise_paths = _load_paths(cfg.dataset.noise, "noise")
		self.noise_paths = list(itertools.chain.from_iterable(self.noise_paths.values()))

		self.phone_symmap = phone_symmap or self._get_phone_symmap()
		self.spkr_symmap = self._get_spkr_symmap()
		self.spkr_group_symmap = self._get_spkr_group_symmap()
		self.lang_symmap = self._get_lang_symmap()
		self.tone_symmap = self._get_tone_symmap()
		self.task_symmap = self._get_task_symmap()

		# assert len(self.phone_symmap) < 256, "Unique token count should be [0,255] to fit within uint8"
		self.text_dtype = torch.uint8 if len(self.phone_symmap) < 256 else torch.int16

		if len(self.paths) == 0:
			raise ValueError(f"No valid path is found for {self.dataset_type}")
		

		sampler_path = cfg.rel_path / f"sampler.{self.sampler_type}.rank{global_rank()}.pt"

		if self.sampler_type == "path":
			self.sampler = OrderedSampler( len(self) )
			self.samplers = {}
			self.spkr_samplers = {}
		else:
			self.sampler = RandomSampler( len(self) )
			self.samplers = { name: PoolSampler( paths, keep_all=True ) for name, paths in self.paths_by_spkr_name.items() }
			self.spkr_samplers = { name: PoolSampler( [*set(speakers)], keep_all=True ) for name, speakers in self.spkrs_by_spkr_group.items() }

		self.load_state_dict()
		
	def get_speaker(self, path):
		if isinstance(path, str):
			path = Path(path)
		res = cfg.get_spkr(path)
		return res

	def get_speaker_group(self, path):
		if isinstance(path, str):
			path = Path(path)
		res = cfg.get_spkr_group(path)
		return res

	def get_language(self, speaker_group):
		lang = "en"
		for k, v in cfg.dataset.speaker_languages.items():
			if speaker_group in v:
				lang = k
				break

		return lang

	@cached_property
	def spkrs(self):
		return sorted({self.get_speaker(path) for path in self.paths})

	@cached_property
	def tasks(self):
		return cfg.dataset.tasks_list # ["tts", "tts", "ns", "sr", "tse", "tts", "tts"] # , "cse", "nse"

	def save_state_dict(self, path = None):
		if path is None:
			path = cfg.rel_path / f"sampler.{self.sampler_type}.rank{global_rank()}.pt"

		if self.sampler_type == "path":
			state_dict = self.sampler.get_state()
		else:
			state_dict = {
				"samplers": { name: sampler.get_state() for name, sampler in self.samplers.items() },
				"spkr_samplers": { name: sampler.get_state() for name, sampler in self.spkr_samplers.items() },
			}
		torch.save(state_dict, path)

	def load_state_dict(self, path = None):
		if path is None:
			path = cfg.rel_path / f"sampler.{self.sampler_type}.rank{global_rank()}.pt"

		if not path.exists():
			return

		state_dict = torch.load(path)
		if self.sampler_type == "path":
			state_dict = self.sampler.set_state(state_dict)
		else:
			for name, sampler in state_dict["samplers"].items():
				if name not in self.samplers:
					continue
				self.samplers[name].set_state( sampler )

			for name, sampler in state_dict["spkr_samplers"].items():
				if name not in self.spkr_samplers:
					continue
				self.spkr_samplers[name].set_state( sampler )

	def _get_phone_symmap(self):
		return get_phone_symmap()

	def _get_spkr_symmap(self):
		return {s: i for i, s in enumerate(self.spkrs)}

	def _get_spkr_group_symmap(self):
		return {s: i for i, s in enumerate(self.spkr_groups)}

	def _get_lang_symmap(self):
		return get_lang_symmap()

	def _get_tone_symmap(self):
		return get_tone_symmap()

	def _get_task_symmap(self):
		return get_task_symmap()

	"""
	def get_task_token( self, token, levels=cfg.model.max_levels ):
		if not hasattr(self, "task_symmap"):
			self.task_symmap = self._get_task_symmap()
		return torch.Tensor([[ self.task_symmap[f'<{token}>'] for _ in range(levels) ]]).to(dtype=torch.int16)
	"""

	def sample_noise(self):
		path = random.choice(self.noise_paths)

		if cfg.dataset.use_hdf5:
			key = _get_hdf5_path(path)
			mel = torch.from_numpy(cfg.hdf5[key]["audio"]).to(torch.int16)
		else:
			mel = _load_mels(path, return_metadata=False)
		return mel

	def sample_speakers(self, ignore=[]):
		choices = set(self.spkrs) - set(ignore)
		return random.choice([*choices])

	def sample_prompts(self, spkr_name, ignore):
		prom_list = []

		choices = set(self.paths_by_spkr_name[spkr_name]) - {ignore}
		choices = [*choices]

		# no other utterances, it'd make more sense to prune speakers with only one utterance in the validation step
		if len(choices) == 0:
			choices = [*set(self.paths_by_spkr_name[spkr_name])]
			"""
			raise ValueError(
				f"Failed to find another different utterance for {spkr_name}."
			)
			"""

		path = random.choice(choices)
		if cfg.dataset.use_hdf5:
			key = _get_hdf5_path(path)

			if "audio" not in cfg.hdf5[key]:
				_logger.warning(f'MISSING AUDIO: {key}')
				return

			# audio / cond / latents
			# parameter names and documentation are weird
			prom = torch.from_numpy(cfg.hdf5[key]["cond"]).to(torch.int16)
		else:
			prom = _load_mels(path, return_metadata=False)

		return prom

	def __getitem__(self, index):
		if self.sampler_type == "group":
			spkr_group = self.spkr_groups[index]
			#spkr_group_id = self.spkr_group_symmap[spkr_group]
			spkr_name = self.spkr_samplers[spkr_group].sample()
			spkr_id = self.spkr_symmap[spkr_name]
			path = self.samplers[spkr_name].sample()
		elif self.sampler_type == "speaker":
			spkr_name = self.spkrs[index]
			spkr_id = self.spkr_symmap[spkr_name]
			path = self.samplers[spkr_name].sample()
			spkr_group = self.get_speaker_group(path)
			#spkr_group_id = self.spkr_group_symmap[spkr_group]
		else:
			path = self.paths[index]
			spkr_name = self.get_speaker(path)
			spkr_id = self.spkr_symmap[spkr_name]
			spkr_group = self.get_speaker_group(path)
			#spkr_group_id = self.spkr_group_symmap[spkr_group]

		if cfg.dataset.use_hdf5:
			key = _get_hdf5_path(path)

			if key not in cfg.hdf5:
				raise RuntimeError(f'Key of Path ({path}) not in HDF5: {key}')

			text = cfg.hdf5[key]["text"][:]
			mel = cfg.hdf5[key]["audio"][:]
			#conds = (cfg.hdf5[key]["conds_0"][:], cfg.hdf5[key]["conds_1"][:])
			latents = (cfg.hdf5[key]["latents_0"][:], cfg.hdf5[key]["latents_1"][:])
			
			text = torch.from_numpy(text).to(self.text_dtype)
			mel = torch.from_numpy(mel).to(torch.int16)
			#conds = (torch.from_numpy(conds[0]), torch.from_numpy(conds[1]))
			latents = (torch.from_numpy(latents[0]), torch.from_numpy(latents[1]))

			wav_length = cfg.hdf5[key].attrs["wav_length"]
		else:
			mel, metadata = _load_mels(path, return_metadata=True)
			text = torch.tensor(metadata["text"]).to(self.text_dtype)
			#conds = (torch.from_numpy(metadata["conds"][0]), torch.from_numpy(metadata["conds"][1]))
			latents = (torch.from_numpy(metadata["latent"][0]), torch.from_numpy(metadata["latent"][1]))
			wav_length = metadata["wav_length"]

		return dict(
			index=index,
			path=Path(path),
			spkr_name=spkr_name,
			spkr_id=spkr_id,

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

	def __len__(self):
		if self.sampler_type == "group":
			return min(len(self.spkr_groups), self._head or len(self.spkr_groups))
		if self.sampler_type == "speaker":
			return min(len(self.spkrs), self._head or len(self.spkrs))
		return min(len(self.paths), self._head or len(self.paths))


def collate_fn(samples: list[dict]):
	batch: dict[str, Any] = {k: [s[k] for s in samples] for k in samples[0]}
	return batch


def _seed_worker(worker_id):
	worker_seed = torch.initial_seed() % 2**32
	np.random.seed(worker_seed)
	random.seed(worker_seed)


def _create_dataloader(dataset, training):
	"""
	if cfg.distributed and training:
		sampler = DistributedSampler(dataset)
		shuffle = False
	"""		

	return DataLoader(
		dataset=dataset,
		batch_size=cfg.hyperparameters.batch_size if training else cfg.evaluation.batch_size,
		shuffle=dataset.shuffle,
		drop_last=training,
		num_workers=cfg.dataset.workers,
		collate_fn=collate_fn,
		persistent_workers=cfg.dataset.workers > 1,
		pin_memory=False, # True,
		worker_init_fn=_seed_worker,
		sampler=dataset.sampler,
	)

def create_datasets():
	train_dataset = Dataset( training=True )
	val_dataset = Dataset( phone_symmap=train_dataset.phone_symmap, training=False )

	return train_dataset, val_dataset


def create_train_val_dataloader():
	train_dataset, val_dataset = create_datasets()

	# it'll cry about trying to pickle a torch._C_generator or something
	try:
		subtrain_dataset = copy.deepcopy(train_dataset)
	except Exception as e:
		subtrain_dataset = Dataset( training=True )

	if subtrain_dataset.sampler_type == "path":
		subtrain_dataset.head_(cfg.evaluation.size)

	train_dl = _create_dataloader(train_dataset, training=True)
	val_dl = _create_dataloader(val_dataset, training=False)
	subtrain_dl = _create_dataloader(subtrain_dataset, training=False)

	_logger.info(str(train_dataset.phone_symmap))
	_logger.info(str(train_dataset.spkr_symmap))
	_logger.info(str(train_dataset.spkr_group_symmap))

	_logger.info(f"#samples (train): {len(train_dataset)}.")
	_logger.info(f"#samples (val): {len(val_dataset)}.")
	_logger.info(f"#samples (subtrain): {len(subtrain_dataset)}.")

	_logger.info(f"#duration (train): {str(train_dataset.duration)}.")
	_logger.info(f"#duration (val): {str(val_dataset.duration)}.")
	_logger.info(f"#duration (subtrain): {str(subtrain_dataset.duration)}.")

	assert isinstance(subtrain_dl.dataset, Dataset)

	return train_dl, subtrain_dl, val_dl

def unpack_audio( npz ):
	mel = npz["codes"].to(device="cpu")
	
	conds = npz["conds"][0].to(device="cpu"), npz["conds"][1].to(device="cpu")
	latent = npz["latent"][0].to(device="cpu"), npz["latent"][1].to(device="cpu")

	metadata = {}

	if "text" in npz:
		metadata["text"] = npz["text"]
	
	if "phonemes" in npz["metadata"]:
		metadata["phonemes"] = npz["metadata"]["phonemes"]
	
	if "language" in npz["metadata"]:
		metadata["language"] = npz["metadata"]["language"]
	
	if "original_length" in npz["metadata"]:
		metadata["wav_length"] = npz["metadata"]["original_length"]

	if "duration" in npz["metadata"]:
		metadata["duration"] = npz["metadata"]["duration"]
	elif "original_length" in npz["metadata"] and "sample_rate" in npz["metadata"]:
		metadata["duration"] = npz["metadata"]["original_length"] / npz["metadata"]["sample_rate"]

	return mel, conds, latent, metadata

# parse dataset into better to sample metadata
def create_dataset_metadata( skip_existing=True ):
	symmap = get_phone_symmap()
	
	root = str(cfg.data_dir)
	metadata_root = str(cfg.metadata_dir)

	cfg.metadata_dir.mkdir(parents=True, exist_ok=True)

	def add( dir, type="training", audios=True, texts=True ):
		name = str(dir)
		name = name.replace(root, "")

		speaker_name = name

		metadata_path = Path(f"{metadata_root}/{speaker_name}.json")
		metadata_path.parents[0].mkdir(parents=True, exist_ok=True)

		try:
			metadata = {} if not metadata_path.exists() else json.loads(open(str(metadata_path), "r", encoding="utf-8").read())
		except Exception as e:
			metadata = {}

		if not os.path.isdir(f'{root}/{name}/'):
			return
		# tqdm.write(f'{root}/{name}')
		files = os.listdir(f'{root}/{name}/')

		# grab IDs for every file
		ids = { file.replace(_get_mel_extension(), "").replace(_get_phone_extension(), "") for file in files }

		for id in tqdm(ids, desc=f"Processing {name}"):
			try:
				mel_exists = os.path.exists(f'{root}/{name}/{id}{_get_mel_extension()}') if audios else True
				text_exists = os.path.exists(f'{root}/{name}/{id}{_get_phone_extension()}') if texts else True

				if not mel_exists:
					continue

				key = f'{type}/{speaker_name}/{id}'

				if skip_existing and id in metadata:
					continue

				if id not in metadata:
					metadata[id] = {}

				utterance_metadata = {}
				if audios:
					# ideally we'll encode Encodec-based audio in a similar manner because np has smaller files than pt
					npz = np.load(f'{root}/{name}/{id}{_get_mel_extension()}', allow_pickle=True)[()]
					mel, conds, latents, utterance_metadata = unpack_audio( npz )
				# text
				if texts and text_exists and not utterance_metadata:
					utterance_metadata = json.loads(open(f'{root}/{name}/{id}{_get_phone_extension()}', "r", encoding="utf-8").read())

				for k, v in utterance_metadata.items():
					metadata[id][k] = v

			except Exception as e:
				tqdm.write(f'Error while processing {id}: {e}')

		with open(str(metadata_path), "w", encoding="utf-8") as f:
			f.write( json.dumps( metadata ) )

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


	def add( dir, type="training", audios=True, texts=True ):
		name = str(dir)
		name = name.replace(root, "")
		
		# yucky
		speaker_name = name
		if "LibriTTS-R" in speaker_name:
			speaker_name = speaker_name.replace("LibriTTS-R", "LibriVox")

		metadata_path = Path(f"{metadata_root}/{speaker_name}.json")
		metadata_path.parents[0].mkdir(parents=True, exist_ok=True)

		metadata = {} if not metadata_path.exists() else json.loads(open(str(metadata_path), "r", encoding="utf-8").read())

		if not os.path.isdir(f'{root}/{name}/'):
			return

		files = os.listdir(f'{root}/{name}/')

		# grab IDs for every file
		ids = { file.replace(_get_mel_extension(), "").replace(_get_phone_extension(), "") for file in files }

		for id in tqdm(ids, desc=f"Processing {name}"):
			try:
				mel_exists = os.path.exists(f'{root}/{name}/{id}{_get_mel_extension()}') if audios else True
				text_exists = os.path.exists(f'{root}/{name}/{id}{_get_phone_extension()}') if texts else True

				if not mel_exists:
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
					npz = np.load(f'{root}/{name}/{id}{_get_mel_extension()}', allow_pickle=True)[()]
					mel, conds, latents, utterance_metadata = unpack_audio( npz )

					if "audio" not in group:
						group.create_dataset('audio', data=mel.numpy(), compression='lzf')
					
					"""
					for i, cond in enumerate(conds):
						if f"conds_{i}" not in group:
							group.create_dataset(f'conds_{i}', data=cond.numpy(), compression='lzf')
					"""
						
					for i, latent in enumerate(latents):
						if f"latents_{i}" not in group:
							group.create_dataset(f'latents_{i}', data=latent.numpy(), compression='lzf')

				# text
				if texts:
					if not utterance_metadata and text_exists:
						utterance_metadata = json.loads(open(f'{root}/{name}/{id}{_get_phone_extension()}', "r", encoding="utf-8").read())

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
				raise e

		with open(str(metadata_path), "w", encoding="utf-8") as f:
			f.write( json.dumps( metadata ) )

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

	hf.create_dataset('symmap', data=json.dumps(symmap))
	hf.close()

if __name__ == "__main__":
	import argparse

	parser = argparse.ArgumentParser("Save trained model to path.")
	parser.add_argument("--action", type=str)
	parser.add_argument("--tasks", type=str)
	args, unknown = parser.parse_known_args()

	task = args.action

	cfg.dataset.workers = 1

	class LoggerOveride:
		def info(self, *args):
			print(*args)
	
	_logger = LoggerOveride()

	if args.action == "hdf5":
		create_dataset_hdf5()
	elif args.action == "list-dataset":
		dataset = []
		for group in os.listdir(cfg.data_dir):
			for name in os.listdir(cfg.data_dir / group):
				if len(os.listdir(cfg.data_dir / group / name)) == 0:
					continue
				dataset.append(f'{group}/{name}')

		print(json.dumps(dataset))
	elif args.action == "metadata":
		create_dataset_metadata()
	elif args.action == "sample":
		train_dl, subtrain_dl, val_dl = create_train_val_dataloader()

		samples = {
			"training": next(iter(train_dl)),
			#"evaluation": next(iter(subtrain_dl)),
			#"validation": next(iter(val_dl)),
		}

		for sample_name, sample_batch in samples.items():
			for name, batch in sample_batch.items():
				#print( name, [ x.shape if hasattr(x, "shape") else x for x in batch ] )
				print( name, [ x for x in batch ] )
		
		"""
		for k, v in samples.items():
			for i in range(len(v)):
				print(f'{k}[{i}]:', v[i])
		"""

	elif args.action == "tasks":
		index = 0
		cfg.dataset.tasks_list = args.tasks.split(",")
		
		train_dl, subtrain_dl, val_dl = create_train_val_dataloader()
		batch = next(iter(train_dl))

		for text, resps, proms, task in zip(batch["text"], batch["resps"], batch["proms"], batch["task"]):
			if task not in cfg.dataset.tasks_list:
				continue

			print(text, task, cfg.model.prom_levels)
			print( proms.shape, resps.shape )

			tokens = 0
			tokens += sum([ text.shape[0] for text in batch["text"] ])
			tokens += sum([ resps.shape[0] for resps in batch["resps"] ])
			print( tokens )

			decode_to_file( proms, f"./data/{task}.proms.wav", device="cpu" )
			decode_to_file( resps, f"./data/{task}.resps.wav", device="cpu" )
			break