# TorToiSe TTS

An unofficial PyTorch re-implementation of [TorToise TTS](https://github.com/neonbjb/tortoise-tts/tree/98a891e66e7a1f11a830f31bd1ce06cc1f6a88af).

## Requirements

A working PyTorch environment.

## Install

Simply run `pip install git+https://git.ecker.tech/mrq/tortoise-tts` or `pip install git+https://github.com/e-c-k-e-r/tortoise-tts`.

## To-Do

- [ ] Reimplement original inferencing through TorToiSe (as done with `api.py`)
- [ ] Implement training support (without DLAS)
  - [ ] Feature parity with the VALL-E training setup with preparing a dataset ahead of time
- [ ] Automagic handling of the original weights into compatible weights
- [ ] Extend the original inference routine with additional features:
  - [x] non-float32 / mixed precision
  - [x] BitsAndBytes support
  - [x] LoRAs
  - [x] Web UI
    - [ ] Feature parity with [ai-voice-cloning](https://git.ecker.tech/mrq/ai-voice-cloning)
  - [ ] Additional samplers for the autoregressive model
  - [ ] Additional samplers for the diffusion model
  - [ ] BigVGAN in place of the original vocoder
  - [ ] XFormers / flash_attention_2 for the autoregressive model
  - [ ] Some vector embedding store to find the "best" utterance to pick

## Why?

To correct the mess I've made with forking TorToiSe TTS originally with a bunch of slopcode, and the nightmare that ai-voice-cloning turned out.

Additional features can be applied to the program through a framework of my own that I'm very familiar with.

## License

Unless otherwise credited/noted in this README or within the designated Python file, this repository is [licensed](LICENSE) under AGPLv3.