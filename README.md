# TorToiSe TTS

An unofficial PyTorch re-implementation of [TorToise TTS](https://github.com/neonbjb/tortoise-tts/tree/98a891e66e7a1f11a830f31bd1ce06cc1f6a88af).

Almost all of the documentation and usage are carried over from my [VALL-E](https://github.com/e-c-k-e-r/vall-e) implementation, as documentation is lacking for this implementation, as I whipped it up over the course of two days using knowledge I haven't touched in a year.

## Requirements

A working PyTorch environment.
+ `python3 -m venv venv && source ./venv/bin/activate` is sufficient.

## Install

Simply run `pip install git+https://git.ecker.tech/mrq/tortoise-tts@new` or `pip install git+https://github.com/e-c-k-e-r/tortoise-tts`.

## Usage

### Inferencing

Using the default settings: `python3 -m tortoise_tts --yaml="./data/config.yaml" "Read verse out loud for pleasure." "./path/to/a.wav"`

To inference using the included Web UI: `python3 -m tortoise_tts.webui --yaml="./data/config.yaml"`
+ Pass `--listen 0.0.0.0:7860` if you're accessing the web UI from outside of `localhost` (or pass the host machine's local IP instead)

### Training / Finetuning

Training is as simple as copying the reference YAML from `./data/config.yaml` to any training directory of your choice (for examples: `./training/` or `./training/lora-finetune/`).

A pre-processed dataset is required. Refer to [the VALL-E implementation](https://github.com/e-c-k-e-r/vall-e#leverage-your-own-dataset) for more details.

To start the trainer, run `python3 -m tortoise_tts.train --yaml="./path/to/your/training/config.yaml`.
+ Type `save` to save whenever. Type `quit` to quit and save whenever. Type `eval` to run evaluation / validation of the model.

For training a LoRA, uncomment the `loras` block in your training YAML.

For loading an existing finetuned model, create a folder with this structure, and load its accompanying YAML:
```
./some/arbitrary/path/:
    ckpt:
        autoregressive:
            fp32.pth # finetuned weights
    config.yaml
```

For LoRAs, replace the above `fp32.pth` with `lora.pth`.

## To-Do

- [X] Reimplement original inferencing through TorToiSe (as done with `api.py`)
  - [ ] Reimplement candidate selection with the CLVP
  - [ ] Reimplement redaction with the Wav2Vec2
- [X] Implement training support (without DLAS)
  - [X] Feature parity with the VALL-E training setup with preparing a dataset ahead of time
- [X] Automagic offloading to CPU for unused models (for training and inferencing)
- [X] Automagic handling of the original weights into compatible weights
- [ ] Reimplement added features from my original fork:
  - [ ] "Better" conditioning latents calculating
  - [x] Use of KV-cache for the AR
  - [x] Re-enable DDIM sampler
- [ ] Extend the original inference routine with additional features:
  - [ ] non-float32 / mixed precision for the entire stack
  - [x] BitsAndBytes support
    - Provided Linears technically aren't used because GPT2 uses Conv1D instead...
  - [x] LoRAs
  - [x] Web UI
    - [ ] Feature parity with [ai-voice-cloning](https://git.ecker.tech/mrq/ai-voice-cloning)
      - Although I feel a lot of its features are the wrong way to go about it.
  - [ ] Additional samplers for the autoregressive model (such as mirostat / dynamic temperature)
  - [ ] Additional samplers for the diffusion model (beyond the already included DDIM)
  - [X] BigVGAN in place of the original vocoder
    - [X] HiFiGAN integration as well
  - [ ] XFormers / flash_attention_2 for the autoregressive model
    - Beyond HF's internal implementation of handling alternative attention
    - Both the AR and diffusion models also do their own attention...
  - [ ] Saner way of loading finetuned models / LoRAs
  - [ ] Some vector embedding store to find the "best" utterance to pick
- [ ] Documentation

## Why?

To correct the mess I've made with forking TorToiSe TTS originally with a bunch of slopcode, and the nightmare that ai-voice-cloning turned out.

Additional features can be applied to the program through a framework of my own that I'm very familiar with.

## License

Unless otherwise credited/noted in this README or within the designated Python file, this repository is [licensed](LICENSE) under AGPLv3.