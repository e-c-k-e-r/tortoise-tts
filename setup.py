import subprocess
import sys
from pathlib import Path
from datetime import datetime
from setuptools import setup, find_packages

def shell(*args):
    out = subprocess.check_output(args)
    return out.decode("ascii").strip()

def write_version(version_core, pre_release=True):
    if pre_release:
        time = shell("git", "log", "-1", "--format=%cd", "--date=iso")
        time = datetime.strptime(time, "%Y-%m-%d %H:%M:%S %z")
        time = time.strftime("%Y%m%d%H%M%S")
        version = f"{version_core}-dev{time}"
    else:
        version = version_core

    with open(Path("tortoise_tts", "version.py"), "w") as f:
        f.write('__version__ = "{}"\n'.format(version))

    return version

with open("README.md", "r") as f:
    long_description = f.read()

setup(
    name="tortoise-tts",
    python_requires=">=3.10.0",
    version=write_version("0.0.1"),
    description="An unofficial re-implementation of the audio LM TorToiSe",
    author="ecker",
    author_email="mrq@ecker.tech",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    install_requires=(
        # training backends
        ["deepspeed>=0.7.7"] if not sys.platform.startswith("win") else [])
        + [
        # logging niceties
        "coloredlogs>=15.0.1",
        "humanize>=4.4.0",
        "matplotlib>=3.6.0",
        "pandas>=1.5.0",

        # boiler plate niceties
        "diskcache>=5.4.0",
        "einops>=0.6.0",
        "tqdm",

        # HF bloat
        "tokenizers",
        "transformers",
        "inflect",
        "unidecode",
        "vector_quantize_pytorch",

        #
        "rotary_embedding_torch",

        # training bloat
        "auraloss[all]", # [all] is needed for MelSTFTLoss
        "h5py",
        "prodigyopt @ git+https://github.com/konstmish/prodigy",

        # practically the reason to use python
        "numpy",
        "torch>=1.13.0",
        "torchaudio>=0.13.0",
        "torchmetrics",

        # gradio web UI
        "gradio"
    ],
    extras_require = {
        "all": [
            # attention helpers
            "xformers",
            # "flash-attn" --no-build-isolation # commented out right now because I want to query this for Volta freaks like me who can't use it
        ]
    },
    url="https://git.ecker.tech/mrq/tortoise",
)
