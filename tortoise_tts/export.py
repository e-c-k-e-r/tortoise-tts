import argparse

import torch
import torch.nn

from .data import get_phone_symmap
from .engines import load_engines
from .config import cfg
from .models.lora import lora_get_state_dict
from .utils.io import torch_save, torch_load, json_read, json_write, Path

def extract_lora( state_dict, config = None, save_path = None, dtype = None ):
	if dtype is None:
		dtype = cfg.inference.dtype

	format = save_path.suffix[1:]

	lora = state_dict["lora"] if "lora" in state_dict else None
	# should always be included, but just in case
	if lora is None and "module" in state_dict:
		lora, module = lora_get_state_dict( state_dict["module"], split = True )
		state_dict["module"] = module
	
	if "lora" in state_dict:
		state_dict["lora"] = None

	# should raise an exception since there's nothing to extract, or at least a warning
	if not lora:
		return state_dict

	# save lora specifically
	# should probably export other attributes, similar to what SD LoRAs do
	save_path = save_path.parent / f"lora.{format}"
	torch_save( {
		"module": lora,
		"config": cfg.lora.__dict__ if cfg.lora is not None else None,
	}, save_path )

	return state_dict

def main():
	parser = argparse.ArgumentParser("Save trained model to path.")
	parser.add_argument("--module-only", action='store_true')
	parser.add_argument("--export-lora", action='store_true', default=None) # exports LoRA
	parser.add_argument("--dtype", type=str, default="auto") # set target dtype to export to
	parser.add_argument("--format", type=str, default=cfg.weights_format) # set target format to export weights under
	args, unknown = parser.parse_known_args()

	if args.format.lower() not in ["sft", "safetensors", "pt", "pth"]:
		raise Exception(f"Unknown requested format: {args.format}")

	if args.module_only:
		cfg.trainer.load_module_only = True

	if args.dtype != "auto":
		cfg.trainer.weight_dtype = args.dtype
		
	# necessary to ensure we are actually exporting the weights right
	cfg.inference.backend = cfg.trainer.backend

	engines = load_engines(training=False) # to ignore loading optimizer state

	callback = None
	if args.export_lora:
		callback = extract_lora

	engines.export(userdata={"symmap": get_phone_symmap()}, callback=callback, format=args.format)

if __name__ == "__main__":
	main()