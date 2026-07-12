<img src="docs/assets/audex-mascot.png" width="220" alt="Audex, a fuzzy A-shaped cyclops with circuit whiskers and a waveform antenna">

# Audex-Mac

Use NVIDIA's Nemotron-Labs-Audex speech and sound models locally on
Apple Silicon. Converse with your Mac and ask it to identify or create noises. 

## Disclaimer

My initial question was: "Can a coding agent 
take a vllm-metal checkpoint and support an otherwise totally-unsupported
model on Mac?" The answer was "yes, poorly".

I was impressed enough that I kept going providing steering and goals at
night and on weekends, and it's kinda' ...usable now?

## Why does this exist?

The typical reasons to run LLMs locally are things like, cost, privacy, repeatable scale,
classifier-free queries, stable model choice, and cost. Once a model is on your
disk, it is yours. While those apply, I mostly just want to talk to my Mac, and have it make cool noises.

## Build

Remember those signs at amusement parks? "You must be this tall to ride"?  That applies here.

Audex-Mac is for Apple Silicon Macs. Enough disk for the model and runtime, and sufficient RAM:

- Audex 2B BF16: **24 GB recommended**
- Audex 30B-A3B NVFP4: **48 GB recommended**
- Audex 30B-A3B BF16: **96 GB recommended**

"Why not a NVFP4 2B squeezed into 16 GB?"" My initial analysis says
it would be tight, swap-happy, and not worth the quality trade. It should fit
comfortably at 24 GB, where the existing BF16 model already fits.

```sh
git clone https://github.com/mbarnson/Audex-Mac.git
cd Audex-Mac
```

There is no ceremonial build dance. The first run creates the local environments,
installs the pinned runtime, finds cached models, and asks before downloading one.

## Run

```sh
./start.sh web
```

A browser opens to http://127.0.0.1:8765.

You can also interact at the terminal for a typed of push-to-talk conversation. Example:

```sh
./start.sh
./start.sh web --no-open --model 2b
./start.sh web --no-open --model 30b-nvfp4
```


For the currently-experimental, time-sucking, weird sound-making workbench:

```sh
./sound.sh
```

![Request a sound, Audex-Mac splits it into five candidates, and audition them blind.](docs/assets/sound-lab-flow.png)

Describe a sound. Audex makes five candidates and opens a local blind audition
board. Pick with your ears before revealing what prompt prompted those vectors.

## Test

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -e '.[dev]'
scripts/lint.sh
.venv/bin/python -m pytest -m fast
```

## Documentation

Benchmarks, licenses, constraints, architecture, patch history, quality evidence,
diagnostics, and the rest of the fiddly bits live in the
[documentation map](docs/README.md).

GLHF
