import copy
#import diskcache
import h5py
import json
import os
import subprocess
import sys
import time
import argparse
import yaml
import random
import logging
import itertools

import torch
import numpy as np

from dataclasses import asdict, dataclass, field

from functools import cached_property
from pathlib import Path

from .utils.distributed import world_size
from .utils.io import torch_load
from .utils import set_seed, prune_missing, md5_hash, coerce_dtype

from .tokenizer import VoiceBpeTokenizer

# Yuck
from transformers import PreTrainedTokenizerFast

DEFAULT_YAML = Path(__file__).parent.parent / 'data/config.yaml'

@dataclass()
class BaseConfig:
	yaml_path: str | None = None # path passed in through --yaml

	@property
	def cfg_path(self):
		if self.yaml_path:
			return Path(self.yaml_path.parent)

		return Path(__file__).parent.parent / "data"

	@property
	def rel_path(self):
		return Path(self.cfg_path)

	@property
	def cache_dir(self):
		return self.rel_path / ".cache"

	@property
	def data_dir(self):
		return self.rel_path / "data"
	
	@property
	def metadata_dir(self):
		return self.rel_path / "metadata"

	@property
	def ckpt_dir(self):
		return self.rel_path / "ckpt"

	@property
	def log_dir(self):
		return self.rel_path / "logs" / str(self.start_time)

	@cached_property
	def start_time(self):
		return int(time.time())

	@cached_property
	def git_commit(self):
		try:
			cmd = "git rev-parse HEAD"
			return subprocess.check_output(cmd.split()).decode("utf8").strip()
		except:
			return ""

	@cached_property
	def git_status(self):
		try:
			cmd = "git status"
			return subprocess.check_output(cmd.split()).decode("utf8").strip()
		except:
			return ""

	def dumps(self):
		data = {k: getattr(self, k) for k in dir(self) if not k.startswith("__")}
		data = {k: v for k, v in data.items() if not callable(v)}
		return json.dumps(data, indent=2, default=str)

	def dump(self, path=None):
		if path is None:
			path = self.log_dir / "cfg.json"
		path.parent.mkdir(parents=True, exist_ok=True)
		with open(path, "w") as f:
			f.write(self.dumps())

	# ick
	@classmethod
	def prune_missing( cls, yaml ):
		default = cls(**{})
		default.format()
		yaml, missing = prune_missing( source=default, dest=yaml )
		if missing:
			_logger.warning(f'Missing keys in YAML: {missing}')
		return yaml

	@classmethod
	def from_yaml( cls, yaml_path ):
		state = {}
		state = yaml.safe_load(open(yaml_path, "r", encoding="utf-8"))
		state.setdefault("yaml_path", yaml_path)
		state = cls.prune_missing( state )
		return cls(**state)

	@classmethod
	def from_model( cls, model_path, lora_path=None ):
		if not model_path.exists():
			raise Exception(f'Model path does not exist: {model_path}')

		# load state dict and copy its stored model config
		model_kwargs = { "attention": "auto", "training": False, "teacher": False }

		model_state_dict = torch_load( model_path ) if model_path and model_path.exists() else None
		lora_state_dict = torch_load( lora_path ) if lora_path and lora_path.exists() else None

		models_config = [ model_state_dict["config"] | { "path": model_path } | model_kwargs ] if model_state_dict is not None else []
		loras_config = [ lora_state_dict["config"] | { "path": lora_path } ] if lora_state_dict is not None else []

		state = { "models": models_config, "loras": loras_config, "trainer": { "load_state_dict": True } }

		deduced_backend = None
		if model_state_dict is not None:
			# 9 audio levels, will always be DAC
			if "proms_emb.embs.8.weight" in model_state_dict["module"]:
				deduced_backend = "dac"
			# 8 audio levels, may be encodec/vocos (1024 tokens) or nemo (1000 tokens)
			elif "proms_emb.embs.7.weight" in model_state_dict["module"]:
				deduced_backend = "nemo" if model_state_dict["module"]["proms_emb.embs.7.weight"].shape[0] == 1000 else "vocos"
		
		if deduced_backend:
			_logger.info(f'Deduced audio backend: {deduced_backend}')
			state["audio_backend"] = deduced_backend

		return cls(**state)

	@classmethod
	def from_cli(cls, args=sys.argv):
		# legacy support for yaml=`` format
		for i, arg in enumerate(args):
			if arg.startswith("yaml"):
				args[i] = f'--{arg}'

		parser = argparse.ArgumentParser(allow_abbrev=False, add_help=False)
		parser.add_argument("--yaml", type=Path, default=os.environ.get('TORTOISE_YAML', None)) # os environ so it can be specified in a HuggingFace Space too
		parser.add_argument("--model", type=Path, default=os.environ.get('TORTOISE_MODEL', None)) # os environ so it can be specified in a HuggingFace Space too
		parser.add_argument("--lora", type=Path, default=os.environ.get('TORTOISE_LORA', None)) # os environ so it can be specified in a HuggingFace Space too
		args, unknown = parser.parse_known_args(args=args)

		if args.model:
			return cls.from_model( args.model, args.lora )

		if args.yaml:
			return cls.from_yaml( args.yaml )

		return cls(**{})

	def __repr__(self):
		return str(self)

	def __str__(self):
		return self.dumps()

@dataclass()
class Dataset:
	training: list[Path] = field(default_factory=lambda: []) # paths to load into the training dataset
	validation: list[Path] = field(default_factory=lambda: []) # paths to load into the validation dataset
	noise: list[Path] = field(default_factory=lambda: []) # paths to load into the noise dataset
	
	# to-do: replace these since I feel this can be a bottleneck
	speaker_name_getter: str = "lambda p: f'{p.parts[-3]}_{p.parts[-2]}'" # function eval'd to extract a speaker's name from an utternace path
	speaker_group_getter: str = "lambda p: f'{p.parts[-3]}'" # function eval'd to extract a speaker's group from an utternace path
	# to-do: validate if I can ignore this since this is an artifact from when I only saved phonemes and encoded audio, and no metadata
	speaker_languages: dict = field(default_factory=lambda: {}) # dict where keys are the language codes and values are the speaker groups
	
	use_hdf5: bool = False # whether to load from an HDF5 dataset
	hdf5_name: str = "data.h5" # file name to load the HDF5 dataset
	hdf5_flag: str = "a" # flag to load the HDF5 file, automatically adjusted anyways
	
	use_metadata: bool = False # use genretaed metadata to aid in dataset loading
	
	validate: bool = True # validate each utterance on wheter it can be included based on duration range caps
	strict_validate: bool = False # so far only governs if a path actually exists within the dataset, as this can be a bit slow (and shouldn't really happen normally)
	workers: int = 8 # number of dataloader workers to spawn
	cache: bool = True # use diskcache to cache the dataset

	min_utterances: int = 2 # minimum number of utterances a speaker can have
	max_utterances: int = 0 # max number of utterances a speaker can have (0 to disable)
	duration_range: list[float] = field(default_factory=lambda: [1.0, 12.0]) # the duration range an utterance can be to be included in the dataset
	
	sample_type: str = "path" # path | speaker
	sample_order: str = "interleaved" # duration
	sample_shuffle: bool = True # shuffles the indices in the sampler
	sample_max_duration_batch: float = 0.0 # total number of seconds of utterances per batched, 0 to disable
	# for a full sized model with 12GiB of VRAM for Encodec, 120 seconds is just enough
	# for a full sized model with 24GiB of VRAM for Encodec, 380 seconds is 80% VRAM consumed (but it might be limited by batch size)

	prompt_duration_range: list[float] = field(default_factory=lambda: [3.0, 6.0]) # the duration range the input prompts can be
	prompt_max_samples: int = 3 # maximum number of utterances that can be included in an input prompt for training
	prompt_continuous_utterance_p: float = 0.0 # probability to use the target utterance as an input prompt rather than using a different utterance
	prompt_similar_p: float = 0.75 # odds of sampling for a similar prompt instead of a random prompt
	prompt_similar_top_k: int = 1 # top-k similar candidates to sample from 
	prompt_similar_top_k_offset: int = 0 # offset from the top-k to sample from
	prompt_inject_noise_p: float = 0.0 # adds noise to the input prompt waveform to try and vary things
	
	resps_max_samples: int = 1 # number of samples to target for training
	resps_append_p: float = 1.0 # probability to append another sample to the training target
	resps_pad_silence_p: float = 0.0 # probability to pad resp with silence to fit within the next window

	tasks_list: list[str] = field(default_factory=lambda: ["tts"]) # list of tasks to train against
	reencode_on_concat: bool = False # whether to concat audio by decode => concat => encode, or naively concat codes
	reencode_device: str = "cpu" # "cpu" is slower but saves memory, cuda throws [rank0]: RuntimeError: Cannot re-initialize CUDA in forked subprocess. To use CUDA with multiprocessing, you must use the 'spawn' start method
	noise_scale: float = 0.25 # scaling noise value
	retokenize_text: bool = False

	_frames_per_second: int = 0 # allows setting your own hint

	def hash_key(self, *args):
		return md5_hash([ self.use_hdf5, self.min_duration, self.max_duration ] + [*args])

	@cached_property
	def frames_per_second(self):
		if self._frames_per_second > 0:
			return self._frames_per_second

		if cfg.audio_backend == "dac":
			if cfg.sample_rate == 44_100:
				return 87
			if cfg.sample_rate == 16_000:
				return 50
		
		if cfg.audio_backend == "nemo":
			return 86.1
		
		# 24Khz Encodec / Vocos and incidentally DAC are all at 75Hz
		return 75

	@property
	def min_phones(self):
		return self.phones_range[0]
	@property
	def max_phones(self):
		return self.phones_range[1]
	@property
	def min_duration(self):
		return self.duration_range[0]
	@property
	def max_duration(self):
		return self.duration_range[1]

# I really need to clean this up
@dataclass()
class Model:
	name: str = "" # vanity name for the model
	training: bool = False # I really need to attend to this
	teacher: bool = False # if this is to be treated as a teacher

	frozen_params: list[str] = field(default_factory=lambda: []) # frozen parameters that are not updated when training
	path: Path | None = None
	kwargs: dict = field(default_factory=lambda: {})

	def get(self, name=None):
		return [ self ] if not name or self.name == name else []
	
	def loss_factor(self, k):
		return self.loss_factors.get(k, 1.0)
	@property
	# required for fp8 as the lengths needs to be divisible by 8
	def input_alignment(self):
		return 8 if cfg.optimizations.fp8 else 0

	@property
	def full_name(self):
		name = [ self.name ]

		return "-".join(name)

	@property
	def activation_checkpointing(self):
		return cfg.trainer.activation_checkpointing
	
	@property
	def gradient_checkpointing(self):
		return cfg.trainer.gradient_checkpointing

	@property
	def lora_policy(self):
		include = ["model"] # by default only adapt the main model (not embeddings nor classifier/output projection/LM head/whatever)
		exclude = []

		if self.arch_type == "llama":
			include = ["self_attn", "mlp"] # target only the attention + mlp
			exclude = ["self_attn.k_proj"] # common literature says to ignore it
		if self.arch_type == "retnet":
			include = ["layers."] # target the core layers of the RetNet and ignore the auxiliary stuff
			exclude = ["retention.k_proj"] # attention-based transformers ignore the K, so might as well ignore it for the retnet

		return dict(include=include, exclude=exclude)

	# to-do: derive default arguments from here
	@property
	def get_kwargs(self, type):
		return self.kwargs

# should be renamed to Adapters
@dataclass()
class LoRA:
	name: str = "lora" # vanity name
	# to-do: find sane default values
	rank: int = 128 # rank for the LoRA
	alpha: int = 128 # rank for the LoRA
	training: bool = True # 
	embeddings: bool = False # train the embedding too
	parametrize: bool = False # whether to use the parameterized pathway for LoRAs or not
	path: Path | None = None

	@property
	def full_name(self):
		name = [ self.name, f"r{self.rank}", f"a{self.alpha}" ]
		return "-".join(name)

	# actually not needed anymore
	def active_level( self, level ):
		return True

@dataclass()
class Hyperparameters:
	batch_size: int = 8 # number of samples per training batch
	gradient_accumulation_steps: int = 32 # number of steps to accumulate gradients before updating
	gradient_clipping: int | float = 1.0 # largest size a gradient norm can be

	optimizer: str = "Adamw" # optimizer to use, should be 'Prodigyopt" now
	optimizer_params: dict = field(default_factory=lambda: {}) # to pass through deepspeed config
	
	learning_rate: float = 3.25e-4 # should be 1.0 for ProdigyOpt
	warmup_steps: int = 0 # number of steps to warm up the optimizer before performing updates, I think, this is just passed to deepspeed

	scheduler: str = "" # scheduler to use, currently don't ever use one so this doesn't really matter
	scheduler_params: dict = field(default_factory=lambda: {}) # to pass through deepspeed config

	autotune: bool = False # to do deepspeed's autotuning
	autotune_params: dict = field(default_factory=lambda: {}) # to pass through deepspeed config
	
	torch_optimizer: bool = False # if the requested optimizer is torch-derived rather than deepspeed supplied
	torch_scheduler: bool = False # if the requested scheduler is torch-derived rather than deepspeed-supplied

	teacher_alpha: float = 0.5 # mixing factor when performing knowledge distillation
	teacher_temperature: float = 1.0
	teacher_loss_fn: str = "mse" # kl | mse, use either kl_div or mse_loss (most implementations use kl, some literature says you can use mse)

@dataclass()
class Evaluation:
	batch_size: int = 64 # number of samples per batch during eval / val
	frequency: int = 250 # do eval / val every X iterations
	size: int = 64 # number of samples to generate during eval / val
	kwargs: dict = field(default_factory=lambda: {}) # inferencing kwargs

	# necessary in order to make it not confusing with requiring not-directyl exposed arguments passed to the model
	@cached_property
	def ar_kwargs( self ):
		return dict(
			max_steps=self.kwargs.get("max_ar_steps", 500),
			temperature=self.kwargs.get("ar_temperature", 1.0),
			min_temperature=self.kwargs.get("min_ar_temperature", -1),
			top_p=self.kwargs.get("top_p", 1.0), top_k=self.kwargs.get("top_k", 0), min_p=self.kwargs.get("min_p", 0.0),
			repetition_penalty=self.kwargs.get("repetition_penalty", 1.0), repetition_penalty_decay=self.kwargs.get("repetition_penalty_decay", 0),
			length_penalty=self.kwargs.get("length_penalty", 0),
			beam_width=self.kwargs.get("beam_width", 0),
			mirostat_tau=self.kwargs.get("mirostat_tau", 0),
			mirostat_eta=self.kwargs.get("mirostat_eta", 0),
			dry_multiplier=self.kwargs.get("dry_multiplier", 0),
			dry_base=self.kwargs.get("dry_base", 0),
			dry_allowed_length=self.kwargs.get("dry_allowed_length", 0),
			entropix=self.kwargs.get("entropix_sampling", False),
		)

	@cached_property
	def nar_kwargs( self ):
		return dict(
			max_levels=self.kwargs.get("max_nar_levels", 0),
			temperature=self.kwargs.get("nar_temperature", 0.0),
			min_temperature=self.kwargs.get("min_nar_temp", -1),
			top_p=self.kwargs.get("top_p", 1.0), top_k=self.kwargs.get("top_k", 0.0), min_p=self.kwargs.get("min_p", 0.0),
			repetition_penalty=self.kwargs.get("repetition_penalty", 1.0), repetition_penalty_decay=self.kwargs.get("repetition_penalty_decay", 0.0),
		)

@dataclass()
class DeepSpeed:
	zero_optimization_level: int = 0
	use_compression_training: bool = False # cope
	compression_bits: int = 8 # cope
	inferencing: bool = False # for using DeepSpeed's inferencing wrapper instead
	optimizer: bool = True # use DeepSpeed optimizer wrapper
	amp: bool = False # use DeepSpeed's AMP (requires some other package installed apparently)

	loss_scale_window: int = 1000
	min_loss_scale: float = 32768.0
	max_loss_scale: float = 1048576.0
	loss_scale = 0.0

	profile: bool = False
	config: dict = field(default_factory=lambda: {}) # to pass through deepspeed config

	@cached_property
	def ds_cfg(self):
		optimizer_params = cfg.hyperparameters.optimizer_params
		
		if 'lr' not in optimizer_params:
			optimizer_params["lr"] = cfg.hyperparameters.learning_rate,

		scheduler_params = cfg.hyperparameters.scheduler_params
		if 'warmup_num_steps' not in scheduler_params:
			scheduler_params['warmup_num_steps'] = cfg.hyperparameters.warmup_steps

		if 'total_num_steps' not in scheduler_params:
			scheduler_params['total_num_steps'] = cfg.trainer.iterations

		autotune_params = cfg.hyperparameters.autotune_params

		profiler_path = str( cfg.rel_path / "profiler.log" )
		
		ds_cfg_path = cfg.rel_path / "ds_config.json"
		if not ds_cfg_path.exists():
			ds_cfg_path = Path("./data/ds_config.json")

		if "enabled" not in autotune_params:
			autotune_params['enabled'] = True
		
		if "results_dir" not in autotune_params:
			autotune_params['results_dir'] = str( cfg.rel_path / "autotune" / "results" )
		
		if "exps_dir" not in autotune_params:
			autotune_params['exps_dir'] = str( cfg.rel_path / "autotune" / "exps_" )

		# DeepSpeed fp16 is incompatible with its AMP
		if cfg.trainer.weight_dtype.lower() == "float16":
			self.amp = False

		# disable local AMP
		if self.amp:
			cfg.trainer.amp = False

		ds_cfg = {
			"train_micro_batch_size_per_gpu": cfg.hyperparameters.batch_size,
			"gradient_accumulation_steps": cfg.hyperparameters.gradient_accumulation_steps,
			"optimizer": {
				"type": cfg.hyperparameters.optimizer,
				"params": optimizer_params,
			} if not cfg.hyperparameters.torch_optimizer else None,
			"scheduler": {
				"type": cfg.hyperparameters.scheduler,
				"params": scheduler_params,
			} if not cfg.hyperparameters.torch_scheduler else None,
			"gradient_clipping": cfg.hyperparameters.gradient_clipping,
			"fp16": {
				"enabled": cfg.trainer.weight_dtype.lower() == "float16",
				"auto_cast": True, # ???
				"loss_scale_window": self.loss_scale_window,
				"min_loss_scale": self.min_loss_scale,
				"loss_scale": self.loss_scale if cfg.trainer.scale_loss else 1.0, # use defined loss scale (defaults to 0, which is dynamic) if requested, or 1.0 (none) if not
			},
			"bf16": {
				"enabled": cfg.trainer.weight_dtype.lower() == "bfloat16",
			},
			"amp": {
				"enabled": self.amp,
			},
			"autotuning": autotune_params if cfg.hyperparameters.autotune else None,
			"compression_training": {
				"weight_quantization": {
					"shared_parameters":{
						"enabled": True,
						"quantizer_kernel": True,
						"schedule_offset": 0,
						"quantize_groups": 64,
						"quantize_verbose": True,
						"quantization_type": "symmetric",
						"rounding": "nearest",
						"quantize_weight_in_forward": cfg.trainer.weight_dtype.lower() != "float16", #  MoQ (quantize in optimization step) weight quantization is only supported for FP16
						"fp16_mixed_quantize":{
							"enabled": False,
							"quantize_change_ratio": 1
						}
					},
					"different_groups": {
						"wq1": {
							"params": {
								"start_bits": self.compression_bits,
								"target_bits": self.compression_bits,
								"quantization_period": 0
							},
							"modules": [ "self_attn", "mlp" ] # for LLaMA, need to find for other arches
						}
					}
				},
				"activation_quantization": {
					"shared_parameters":{
						"enabled": True,
						"quantizer_kernel": True,
						"schedule_offset": 0,
						"quantize_groups": 64,
						"quantize_verbose": True,
						"quantization_type": "symmetric",
						"rounding": "nearest",
						"quantize_weight_in_forward": cfg.trainer.weight_dtype.lower() != "float16", #  MoQ (quantize in optimization step) weight quantization is only supported for FP16
						"fp16_mixed_quantize":{
							"enabled": False,
							"quantize_change_ratio": 1
						}
					},
					"different_groups": {
						"aq1": {
							"params": {
								"bits": self.compression_bits,
							},
							"modules": [ "self_attn", "mlp" ] # for LLaMA, need to find for other arches
						}
					}
				},
			} if self.use_compression_training else None,
			"zero_optimization": {
				"stage": self.zero_optimization_level,
				"allgather_partitions": True,
				"contiguous_gradients": True,
				"overlap_comm": True,
				"reduce_scatter": True,
				#"reduce_bucket_size": 5e8,
				#"allgather_bucket_size": 5e8,
				#"sub_group_size": 5e8,
				#"zero_quantized_weights": self.use_compression_training,
				#"zero_hpz_partition_size": world_size(),
				#"zero_quantized_gradients": self.use_compression_training,
			} if self.zero_optimization_level > 0 else None,
			"comms_logger": {
				"enabled": False
			},
			"flops_profiler": {
				"enabled": self.profile,
				"profile_step": 1,
				"module_depth": -1,
				"top_modules": 1,
				"detailed": True,
				"output_file": profiler_path
			}
		}

		null_keys = [ k for k in ds_cfg if not ds_cfg[k] ]
		for k in null_keys:
			del ds_cfg[k]

		ds_cfg.update(self.config)
		if ds_cfg_path.exists():
			ds_cfg.update( json_read( ds_cfg_path ) )

		return ds_cfg

@dataclass()
class Trainer:
	iterations: int = 1_000_000 # maximum iterations to train

	save_tag: str = "step" # name to save checkpoints under, "step" will save as current step count
	load_tag: str | None = None # tag to load checkpoint from; if None: will check against contents of `./ckpt/{model-name}/latest` for the checkpoint name

	save_on_oom: bool = True # save if an OOM error is raised
	save_on_quit: bool = True # save when quitting training
	
	export_on_save: bool = False # export weights to local `fp32.pth` state_dict on saving a checkpoint
	export_on_quit: bool = False # export weights to local `fp32.pth` state_dict on quitting training
	
	save_frequency: int = 100 # frequency to save every X iterations

	keep_last_checkpoints: int = 0 # number of checkpoints to keep, prunes oldest ones

	load_state_dict: bool = False # loads `fp32.pth` state_dict, will automatically be done if a checkpoint is not found but `fp32.pth` exists
	load_states: bool = True #
	strict_loading: bool = False # sets strict_loading=True when loading the state dict
	load_module_only: bool = False # 
	restart_step_count: bool = False # clears the training stats when loading a checkpoint
	resize_modules: bool = True # automatically resizes 

	activation_checkpointing: bool | None = None # deprecated, should technically be used for only on activations and not the entire gradients, but HF only has gradient checkpointing
	gradient_checkpointing: bool = True # enables gradient checkpointing to save VRAM at the cost of slightly reduced performance when training
	detect_grad_anomaly: bool = False # torch.autograd.set_detect_anomaly

	check_for_oom: bool = True # checks for OOMs thrown during forward/backwards
	gc_mode: str | None = None # deprecated, but marks when to do GC

	wandb: bool = False # use wandb, if available
	wandb_params: dict = field(default_factory=lambda: dict)

	weight_dtype: str = "float16" # dtype to have the model under
	audio_device: str = "auto"
	decode_non_resp_audio: bool = True

	amp: bool = False # automatic mixed precision
	ddp: bool = False # torch's internal DDP, automatically set if local backend is used and multiple GPUs are requested
	#scale_loss: bool = False # whether to perform loss scaling (for FP16 training) (it actually seems more harmful than not for this specific workload)

	load_webui: bool = False # load the web UI to allow inferencing during training, to-do: actually make this work

	backend: str = "local" # training backend to use. currently supports "local" | "deepspeed"
	deepspeed: DeepSpeed = field(default_factory=lambda: DeepSpeed) # deepspeed settings

	@cached_property
	def dtype(self):
		return coerce_dtype(self.weight_dtype)

	@cached_property
	def scale_loss(self):
		# currently cannot feasibly apply loss scaling with DeepSpeed backend (it can handle it itself anyways)
		return self.dtype == torch.float16

@dataclass()
class Inference:
	backend: str = "local" # backend to use when inferencing
	weight_dtype: str = "float16" # dtype to load the model under
	amp: bool = True # automatic mixed precision during inferencing

	normalize: bool = False # to-do: actually normalize input / output audio, I believe this might cause issues though

	batch_size: int = 16 # I don't know what would be a good batch size

	audio_backends: dict = field(default_factory=lambda: {})

	auto_unload: bool = False

	@property
	def dtype(self):
		return coerce_dtype(self.weight_dtype)

@dataclass()
class Optimizations:
	injects: bool = False # overwrites default torch classes (not recommended)
	replace: bool = False # replaces modules in place with the optimized version (recommended)
	compile: bool | str = False # runs torch.compile on the model

	linear: bool = True # inject/replace linear for BnB
	embedding: bool = True # inject/replace embedding for BnB
	optimizers: bool = True # inject/replace optimizers (BnB, DAdaptation)
	
	bitsandbytes: bool = False # use bitsandbytes
	dadaptation: bool = False # use dadaptation optimizer
	bitnet: bool = False # use bitnet
	fp8: bool = False # use fp8

	# to-do: validate this madness works still, I don't remember what schizodemon told me to do this
	model_offloading: dict | None = None # automatically splits the model over a list of devices
	# example: {"include":["model"], "limits": [ (6 * 1024) * (1024 ** 2), -1 ]} will have the GPU capped to 6GiB, and offload the remaining layers to CPU
	# example: {"include":["model"], "device": ["cuda:0", "cuda:1"], "limits": [ 0.5, 0.5 ]} will have the GPU 1 try and use 50% of the model, and GPU 2 try and use the other 50%
	# | {"assign": [[ f'layers.{i}.' for i in range(0,6) ], [ f'layers.{i}.' for i in range(6,12) ]]} will assign layers 0-5 to device 1, and 6-12 to device 2

	tensorrt: bool = False
	unsloth: bool = False # unsloth gradient checkpointing (it just offloads tensors to the CPU during backwards, I don't think it's significant enough to bother with on small models)

@dataclass()
class Config(BaseConfig):
	device: str = "cuda" # target device
	mode: str = "training" # "inferencing"
	experimental: bool = False # debug flag
	silent_errors: bool = False # if False, raise exceptions on errors that could silently lead to problems, if True ignore them

	dataset: Dataset = field(default_factory=lambda: Dataset)
	models: dict | list | None = field(default_factory=lambda: [])
	loras: dict | list | None = field(default_factory=lambda: [])
	hyperparameters: Hyperparameters = field(default_factory=lambda: Hyperparameters)
	evaluation: Evaluation = field(default_factory=lambda: Evaluation)
	trainer: Trainer = field(default_factory=lambda: Trainer)
	inference: Inference = field(default_factory=lambda: Inference)
	optimizations: Optimizations = field(default_factory=lambda: Optimizations)
	
	tokenizer: str | None = None # tokenizer class
	tokenizer_path: str = "./tokenizer.json" # tokenizer path
	
	sample_rate: int = 24_000 # sample rate the model expects
	audio_backend: str = "mel" # audio backend to use "encodec" | "vocos" | "dac""
	vocoder: str = "bigvgan" # "vocoder" | "bigvgan" | "hifigan"

	weights_name: str = "fp32"
	weights_format: str = "sft" # "pth" | "sft"
	supported_weights_formats: list[str] = field(default_factory=lambda: ["sft", "safetensors", "pt", "pth"])

	def set_audio_backend(self, audio_backend):
		self.audio_backend = audio_backend

		audio_extension = ".mel"
		if audio_backend == "mel":
			audio_extension = ".mel"
			self.sample_rate = 24_000
		else:
			raise Exception(f"Unknown audio backend: {audio_backend}")

	@property
	def audio_backend_extension(self):
		return ".mel"

	@property
	def model(self):
		for i, model in enumerate(self.models):
			if model.training:
				return model

		return self.models[0] if len(self.models) > 0 else None

	# should be renamed to adapters
	@property
	def lora(self):
		for i, lora in enumerate(self.loras):
			if lora.training:
				return lora

		return self.loras[0] if len(self.loras) > 0 else None

	@property
	def distributed(self):
		return world_size() > 1

	@cached_property
	def get_spkr(self):
		return eval(self.dataset.speaker_name_getter)

	@cached_property
	def get_spkr_group(self):
		return eval(self.dataset.speaker_group_getter)

	"""
	@cached_property
	def diskcache(self):
		if self.yaml_path is not None and self.dataset.cache:
			return diskcache.Cache(self.cache_dir).memoize
		return lambda: lambda x: x
	"""

	# this gets called from vall_e.inference
	def load_yaml( self, config_path ):
		tmp = Config.from_yaml( config_path )
		self.__dict__.update(tmp.__dict__)
	
	def load_model( self, config_path, lora_path=None ):
		tmp = Config.from_model( config_path, lora_path )
		self.__dict__.update(tmp.__dict__)

	def load_hdf5( self, write=False ):
		if hasattr(self, 'hdf5'):
			self.hdf5.close()

		if self.distributed:
			self.dataset.hdf5_flag = "r"
		try:
			self.hdf5 = h5py.File(f'{self.rel_path}/{self.dataset.hdf5_name}', 'a' if write else self.dataset.hdf5_flag) # to-do, have an easy to set flag that determines if training or creating the dataset
		except Exception as e:
			_logger.warning(f"Error while opening HDF5 file: {self.rel_path}/{self.dataset.hdf5_name}: {str(e)}")
			self.dataset.use_hdf5 = False

	# a very icky way to handle wildcard expansions
	def expand( self, path ):
		if not isinstance( path, Path ):
			path = Path(path)

		# do not glob if no wildcard to glob
		if "*" not in str(path):
			return [ path ]

		dir = path.parent
		name = path.name
		
		metadata_parent = cfg.metadata_dir / dir
		data_parent = cfg.data_dir / dir
		
		res = []
		# grab any paths from metadata folder (since this is for HDF5)
		if metadata_parent.exists():
			res = [ path.parent / child.stem for child in Path(metadata_parent).glob(name) ]
			# return if found anything
			if res:
				return res
		# grab anything from the data folder (if no metadata exists)
		if data_parent.exists():
			res = [ path.parent / child.name for child in Path(data_parent).glob(name) ]
			# return if found anything
			if res:
				return res
		
		# return an empty list
		if self.silent_errors:
			return []

		# raise an error to avoid headaches
		raise Exception(f'Cannot unglob requested path: {path}')


	def format( self, training=True ):
		if isinstance(self.dataset, type):
			self.dataset = dict()

		if isinstance(self.models, type):
			self.models = dict()

		if isinstance(self.loras, type):
			self.loras = dict()
		
		if isinstance(self.hyperparameters, type):
			self.hyperparameters = dict()
		
		if isinstance(self.evaluation, type):
			self.evaluation = dict()
		
		if isinstance(self.trainer, type):
			self.trainer = dict()
		
		if isinstance(self.inference, type):
			self.inference = dict()
		
		if isinstance(self.optimizations, type):
			self.optimizations = dict()

		if isinstance( self.dataset, dict ):
			self.dataset = Dataset(**self.dataset)

		if isinstance( self.hyperparameters, dict ):
			self.hyperparameters = Hyperparameters(**self.hyperparameters)

		if isinstance( self.evaluation, dict ):
			self.evaluation = Evaluation(**self.evaluation)

		if isinstance( self.trainer, dict ):
			self.trainer = Trainer(**self.trainer)

		if isinstance( self.trainer.deepspeed, dict ):
			self.trainer.deepspeed = DeepSpeed(**self.trainer.deepspeed)

		if isinstance( self.inference, dict ):
			self.inference = Inference(**self.inference)
		
		if isinstance( self.optimizations, dict ):
			self.optimizations = Optimizations(**self.optimizations)

		# crunge
		if len(self.dataset.validation) == 0:
			self.dataset.validation = self.dataset.training

		# convert to expanded paths
		self.dataset.training = [ self.expand(dir) for dir in self.dataset.training ]
		self.dataset.validation = [ self.expand(dir) for dir in self.dataset.validation ]
		self.dataset.noise = [ self.expand(dir) for dir in self.dataset.noise ]
		# flatten
		self.dataset.training = list(itertools.chain.from_iterable(self.dataset.training))
		self.dataset.validation = list(itertools.chain.from_iterable(self.dataset.validation))
		self.dataset.noise = list(itertools.chain.from_iterable(self.dataset.noise))

		# do cleanup
		for model in self.models:
			if not isinstance( model, dict ):
				continue

		self.models = [ Model(**model) if isinstance(model, dict) else model for model in self.models ]
		self.loras = [ LoRA(**lora)  if isinstance(lora, dict) else lora for lora in self.loras ]

		if not self.models:
			self.models = [ Model() ]

		for model in self.models:
			if model.teacher:
				model.training = False
			if model.training:
				model.teacher = False

		# do not combine the two
		if self.hyperparameters.scheduler == "schedulefree" and self.optimizations.dadaptation:
			self.hyperparameters.scheduler = ""

		if self.hyperparameters.scheduler == "":
			self.hyperparameters.torch_scheduler = True

		if self.trainer.backend == "local" and self.distributed:
			self.trainer.ddp = True
		
		if self.trainer.activation_checkpointing is not None:
			self.trainer.gradient_checkpointing = self.trainer.activation_checkpointing

		if not training:
			self.dataset.use_hdf5 = False

		# load our HDF5 file if requested here
		if self.dataset.use_hdf5:
			self.load_hdf5()

		try:
			from transformers import PreTrainedTokenizerFast
			#self.tokenizer = (self.rel_path if self.yaml_path is not None else Path("./data/")) / self.tokenizer
			tokenizer_path = self.rel_path / self.tokenizer_path
			if not tokenizer_path.exists():
				tokenizer_path = Path("./data/") / self.tokenizer_path

			#self.tokenizer = PreTrainedTokenizerFast(tokenizer_file=str(tokenizer_path))
			self.tokenizer = VoiceBpeTokenizer(tokenizer_file=str(tokenizer_path))
		except Exception as e:
			print("Error while parsing tokenizer:", e)
			raise e

_logger = logging.getLogger(__name__)

cfg = Config.from_cli()

# some safety for remapping deprecated formats and re-coercing uninitialized properties into actual types
try:
	cfg.format()
except Exception as e:
	if not cfg.silent_errors:
		raise e # throw an error because I'm tired of silent errors messing things up for me
	_logger.error(f"Error while parsing config YAML: {str(e)}")

if __name__ == "__main__":
	print(cfg)
