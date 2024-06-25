import argparse

import torch
import torch.nn

from .data import get_phone_symmap
from .engines import load_engines
from .config import cfg
from .models.lora import lora_get_state_dict

def extract_lora( state_dict, config = None, save_path = None ):
	lora = state_dict["lora"] if "lora" in state_dict else None
	# should always be included, but just in case
	if lora is None and "module" in state_dict:
		lora, module = lora_get_state_dict( state_dict["module"], split = True )
		state_dict["module"] = module
		state_dict["lora"] = lora

	# should raise an exception since there's nothing to extract, or at least a warning
	if not lora:
		return state_dict

	# save lora specifically
	# should probably export other attributes, similar to what SD LoRAs do
	save_path = save_path.parent / "lora.pth"
	torch.save( { "module": lora }, save_path )

	return state_dict


def main():
	parser = argparse.ArgumentParser("Save trained model to path.")
	parser.add_argument("--module-only", action='store_true')
	parser.add_argument("--lora", action='store_true', default=None) # exports LoRA
	args, unknown = parser.parse_known_args()

	if args.module_only:
		cfg.trainer.load_module_only = True

	callback = None
	if args.lora:
		callback = extract_lora

	engines = load_engines()
	engines.export(userdata={"symmap": get_phone_symmap()}, callback=callback)

if __name__ == "__main__":
	main()