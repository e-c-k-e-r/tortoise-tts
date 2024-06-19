import copy
import diskcache
import h5py
import json
import os
import subprocess
import sys
import time
import argparse
import yaml

import torch

from dataclasses import asdict, dataclass, field

from functools import cached_property
from pathlib import Path

from .utils.distributed import world_size
from .tokenizer import VoiceBpeTokenizer

# Yuck
from transformers import PreTrainedTokenizerFast
from tokenizers import Tokenizer


@dataclass()
class BaseConfig:
	yaml_path: str | None = None

	@property
	def cfg_path(self):
		return Path(self.yaml_path.parent) if self.yaml_path is not None else None

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

	@classmethod
	def from_yaml( cls, yaml_path ):
		state = {}
		state = yaml.safe_load(open(yaml_path, "r", encoding="utf-8"))
		state.setdefault("yaml_path", yaml_path)
		return cls(**state)

	@classmethod
	def from_cli(cls, args=sys.argv):
		# legacy support for yaml=`` format
		for i, arg in enumerate(args):
			if arg.startswith("yaml"):
				args[i] = f'--{arg}'

		parser = argparse.ArgumentParser(allow_abbrev=False)
		parser.add_argument("--yaml", type=Path, default=os.environ.get('VALLE_YAML', None)) # os environ so it can be specified in a HuggingFace Space too
		args, unknown = parser.parse_known_args(args=args)

		if args.yaml:
			return cls.from_yaml( args.yaml )			

		return cls(**{})

	def __repr__(self):
		return str(self)

	def __str__(self):
		return self.dumps()

@dataclass()
class Dataset:
	training: list[Path] = field(default_factory=lambda: [])
	validation: list[Path] = field(default_factory=lambda: [])
	noise: list[Path] = field(default_factory=lambda: [])
	
	temp: list[Path] = field(default_factory=lambda: [])

	speaker_name_getter: str = "lambda p: f'{p.parts[-3]}_{p.parts[-2]}'"
	speaker_group_getter: str = "lambda p: f'{p.parts[-3]}'"

	speaker_languages: dict = field(default_factory=lambda: {}) # dict where keys are the language codes and values are the speaker groups
	
	hdf5_name: str = "data.h5"
	use_hdf5: bool = False
	use_metadata: bool = False
	hdf5_flag: str = "a"
	validate: bool = True
	workers: int = 8
	cache: bool = True

	phones_range: list[int] = field(default_factory=lambda: [4, 256])
	duration_range: list[float] = field(default_factory=lambda: [1.0, 12.0])
	prompt_duration_range: list[float] = field(default_factory=lambda: [3.0, 6.0])
	min_utterances: int = 2

	random_utterance: float = 1.0
	max_prompts: int = 3
	
	prompt_duration: float = 0.0 # legacy
	
	max_resps: int = 1
	p_resp_append: float = 1.0

	sample_type: str = "path" # path | speaker
	sample_order: str = "shuffle" # duration

	tasks_list: list[str] = field(default_factory=lambda: ["tts"])
	
	@cached_property
	def frames_per_second(self):
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
	training: bool = False
	frozen_params: list[str] = field(default_factory=lambda: []) # frozen parameters that are not updated when training

	def get(self, name=None):
		return [ self ] if not name or self.name == name else []
	
	@property
	def full_name(self):
		name = [ self.name ]
		return "-".join(name)

	@property
	def gradient_checkpointing(self):
		return cfg.trainer.gradient_checkpointing

	@property
	def lora_policy(self):
		include = ["gpt"] # by default only adapt the main model (not embeddings nor classifier/output projection/LM head/whatever)
		exclude = []

		return dict(include=include, exclude=exclude)

@dataclass()
class LoRA:
	name: str = "lora" # vanity name
	# to-do: find sane default values
	rank: int = 8 # rank for the LoRA
	alpha: int = 16 # rank for the LoRA
	training: bool = True # 
	parametrize: bool = False # 
	module: str = "linear" # linear | conv1d

	@property
	def full_name(self):
		name = [ self.name, f"r{self.rank}", f"a{self.alpha}" ]
		return "-".join(name)

	
@dataclass()
class Hyperparameters:
	batch_size: int = 8
	gradient_accumulation_steps: int = 32
	gradient_clipping: int | float = 100

	optimizer: str = "Adamw"
	optimizer_params: dict = field(default_factory=lambda: {}) # to pass through deepspeed config
	
	learning_rate: float = 3.25e-4
	warmup_steps: int = 0

	scheduler: str = ""
	scheduler_type: str = "" # deprecated
	scheduler_params: dict = field(default_factory=lambda: {}) # to pass through deepspeed config

	autotune: bool = False
	autotune_params: dict = field(default_factory=lambda: {}) # to pass through deepspeed config
	
	torch_optimizer: bool = False
	torch_scheduler: bool = False
	
@dataclass()
class Evaluation:
	batch_size: int = 64
	frequency: int = 250
	size: int = 64
  
	steps: int = 500
	ar_temperature: float = 1.0
	nar_temperature: float = 0.2

	load_disabled_engines: bool = True

@dataclass()
class DeepSpeed:
	zero_optimization_level: int = 0
	use_compression_training: bool = False
	compression_bits: int = 8
	inferencing: bool = False
	
	amp: bool = False

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
				"contiguous_gradients": True,
				"overlap_comm": True,
				"reduce_scatter": True,
				"reduce_bucket_size": 5e8,
				"allgather_bucket_size": 5e8,
				"sub_group_size": 5e8,
				"round_robin_gradients": True,
				"offload_optimizer": {
					"device": "cpu",
					"pin_memory": True
				},
				"offload_param": {
					"device": "cpu",
					"pin_memory": True
				},
				"zero_quantized_weights": self.use_compression_training,
				"zero_hpz_partition_size": world_size(),
				"zero_quantized_gradients": self.use_compression_training,
			} if self.zero_optimization_level > 0 else None,
			"comms_logger": {
				"enabled": False
			}
		}

		null_keys = [ k for k in ds_cfg if not ds_cfg[k] ]
		for k in null_keys:
			del ds_cfg[k]

		if os.path.exists("./data/ds_config.json"):
			ds_cfg.update(json.load(open("./data/ds_config.json", "r", encoding="utf-8")))
		else:
			ds_cfg.update(self.config)

		return ds_cfg

@dataclass()
class Trainer:
	iterations: int = 100_000

	save_tag: str = "step"
	load_tag: str | None = None

	save_on_oom: bool = True
	save_on_quit: bool = True
	
	export_on_save: bool = False
	export_on_quit: bool = False
	
	save_frequency: int = 100

	keep_last_checkpoints: int = 0

	load_state_dict: bool = False
	load_states: bool = True
	strict_loading: bool = True
	load_module_only: bool = False
	restart_step_count: bool = False

	activation_checkpointing: bool | None = None # deprecated
	gradient_checkpointing: bool = True

	aggressive_optimizations: bool = False
	check_for_oom: bool = True
	gc_mode: str | None = None
	load_disabled_engines: bool = False

	weight_dtype: str = "float16"
	amp: bool = False
	ddp: bool = False

	load_webui: bool = False
	no_logger: bool = False

	backend: str = "local"
	deepspeed: DeepSpeed = field(default_factory=lambda: DeepSpeed)

	@cached_property
	def dtype(self):
		if self.weight_dtype == "float16":
			return torch.float16
		if self.weight_dtype == "bfloat16":
			return torch.bfloat16
		if self.weight_dtype == "float8_e5m2":
			return torch.float8_e5m2
		if self.weight_dtype == "float8_e4m3fn":
			return torch.float8_e4m3fn
		return torch.float32

	@cached_property
	def scale_loss(self):
		# currently cannot feasibly apply loss scaling with DeepSpeed backend (it can handle it itself anyways)
		if self.backend != "local":
			return False
		return self.dtype == torch.float16


@dataclass()
class Inference:
	backend: str = "local"
	weight_dtype: str = "float32"
	amp: bool = False

	normalize: bool = False # do NOT enable this unless you know exactly what you're doing

	# legacy / backwards compat
	use_vocos: bool = True
	use_encodec: bool = True
	use_dac: bool = True

	# shit that doesn't work
	recurrent_chunk_size: int = 0
	recurrent_forward: bool = False

	@cached_property
	def dtype(self):
		if self.weight_dtype == "float16":
			return torch.float16
		if self.weight_dtype == "bfloat16":
			return torch.bfloat16
		if self.weight_dtype == "int8":
			return torch.int8
		if self.weight_dtype == "float8_e5m2":
			return torch.float8_e5m2
		if self.weight_dtype == "float8_e4m3fn":
			return torch.float8_e4m3fn
		return torch.float32

# should be renamed to optimizations
@dataclass()
class Optimizations:
	injects: bool = False # overwrites default torch classes (not recommended)
	replace: bool = False # replaces modules in place with the optimized version (recommended)

	linear: bool = True # inject/replace linear for BnB
	embedding: bool = True # inject/replace embedding for BnB
	optimizers: bool = True # inject/replace optimizers (BnB, DAdaptation)
	
	bitsandbytes: bool = False # use bitsandbytes
	dadaptation: bool = False # use dadaptation optimizer
	bitnet: bool = False # use bitnet
	fp8: bool = False # use fp8

@dataclass()
class Config(BaseConfig):
	device: str = "cuda"
	mode: str = "training" # "inferencing"
	experimental: bool = False # So I can stop commenting out things when committing

	dataset: Dataset = field(default_factory=lambda: Dataset)
	models: dict | list | None = field(default_factory=lambda: [])
	loras: dict | list | None = field(default_factory=lambda: [])
	hyperparameters: Hyperparameters = field(default_factory=lambda: Hyperparameters)
	evaluation: Evaluation = field(default_factory=lambda: Evaluation)
	trainer: Trainer = field(default_factory=lambda: Trainer)
	inference: Inference = field(default_factory=lambda: Inference)
	bitsandbytes: dict | list | None = None # deprecated
	optimizations: Optimizations = field(default_factory=lambda: Optimizations)
	
	tokenizer: str = "./tokenizer.json"

	sample_rate: int = 24_000
	audio_backend: str = "mel"

	@property
	def model(self):
		for i, model in enumerate(self.models):
			if model.training:
				return model

		return self.models[0] if len(self.models) > 0 else None

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

	@cached_property
	def diskcache(self):
		if self.yaml_path is not None and self.dataset.cache:
			return diskcache.Cache(self.cache_dir).memoize
		return lambda: lambda x: x

	# I don't remember why this is needed
	def load_yaml( self, config_path ):
		tmp = Config.from_yaml( config_path )
		self.__dict__.update(tmp.__dict__)

	def load_hdf5( self, write=False ):
		if hasattr(self, 'hdf5'):
			self.hdf5.close()

		if self.distributed:
			self.dataset.hdf5_flag = "r"
		try:
			self.hdf5 = h5py.File(f'{self.rel_path}/{self.dataset.hdf5_name}', 'a' if write else self.dataset.hdf5_flag) # to-do, have an easy to set flag that determines if training or creating the dataset
		except Exception as e:
			print("Error while opening HDF5 file:", f'{self.rel_path}/{self.dataset.hdf5_name}', str(e))
			self.dataset.use_hdf5 = False

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

		self.dataset = Dataset(**self.dataset)
		self.dataset.training = [ Path(dir) for dir in self.dataset.training ]
		self.dataset.validation = [ Path(dir) for dir in self.dataset.validation ]
		self.dataset.noise = [ Path(dir) for dir in self.dataset.noise ]

		"""
		if self.models is not None:
			self.model = Model(**next(iter(self.models)))
		else:
			self.model = Model(**self.model)
		"""

		self.models = [ Model(**model) for model in self.models ]
		self.loras = [ LoRA(**lora) for lora in self.loras ]

		self.hyperparameters = Hyperparameters(**self.hyperparameters)

		self.evaluation = Evaluation(**self.evaluation)

		self.trainer = Trainer(**self.trainer)

		if not isinstance(self.trainer.deepspeed, type):
			self.trainer.deepspeed = DeepSpeed(**self.trainer.deepspeed)

		self.inference = Inference(**self.inference)

		if self.bitsandbytes is not None:
			self.optimizations = Optimizations(**self.bitsandbytes)
		else:
			self.optimizations = Optimizations(**self.optimizations)

		if self.hyperparameters.scheduler_type and not self.hyperparameters.scheduler:
			self.hyperparameters.scheduler = self.hyperparameters.scheduler_type
			self.hyperparameters.scheduler_type = ""

		# do not combine the two
		if self.hyperparameters.scheduler == "schedulefree" and self.optimizations.dadaptation:
			self.hyperparameters.scheduler = ""

		if self.hyperparameters.scheduler == "":
			self.hyperparameters.torch_scheduler = True

		if self.dataset.prompt_duration != 0:
			self.dataset.prompt_duration_range = [self.dataset.prompt_duration, self.dataset.prompt_duration]

		if self.trainer.backend == "local" and self.distributed:
			self.trainer.ddp = True
		
		if self.trainer.activation_checkpointing is not None:
			self.trainer.gradient_checkpointing = self.trainer.activation_checkpointing

		if not training:
			self.dataset.use_hdf5 = False

		# load our HDF5 file if requested here
		if self.dataset.use_hdf5:
			self.load_hdf5()

		# load tokenizer
		try:
			from transformers import PreTrainedTokenizerFast
			#cfg.tokenizer = (cfg.rel_path if cfg.yaml_path is not None else Path("./data/")) / cfg.tokenizer
			tokenizer_path = cfg.rel_path / cfg.tokenizer
			if not tokenizer_path.exists():
				tokenizer_path = Path("./data/") / cfg.tokenizer

			#cfg.tokenizer = PreTrainedTokenizerFast(tokenizer_file=str(tokenizer_path))
			cfg.tokenizer = VoiceBpeTokenizer(tokenizer_file=str(tokenizer_path))
		except Exception as e:
			print("Error while parsing tokenizer:", e)
			raise e

cfg = Config.from_cli()

# some safety for remapping deprecated formats and re-coercing uninitialized properties into actual types
try:
	cfg.format()
except Exception as e:
	print("Error while parsing config YAML:")
	raise e # throw an error because I'm tired of silent errors messing things up for me

if __name__ == "__main__":
	print(cfg)
