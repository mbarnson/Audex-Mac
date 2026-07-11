# Audex-Mac

<p align="center">
  <img src="docs/assets/audex-mascot.png" width="220" alt="Audex, a fuzzy A-shaped cyclops with circuit whiskers and a waveform antenna">
</p>

Audex-Mac runs NVIDIA's Nemotron-Labs-Audex speech and sound models locally on
Apple Silicon. Talk to your Mac, let it talk back, or ask it to manufacture a
small rack of weird noises. No cloud understudy waits behind the curtain.

## Why does this exist?

The respectable reasons to run AI locally are privacy, repeatable scale,
classifier-free queries, stable model choice, and cost. Once a model is on your
disk, it is yours. Nobody can replace it with a "safer" beige cardigan overnight.

Cost, admittedly, is funny. A similarly equipped 128 GB / 4 TB M5 Mac was
$8,149 from Apple in mid-2026. Got-*damn*. The tokens are free once you exclude
the machine from the accounting, a technique economists call "wanting a Mac."

None of that is why I built this. I wanted to talk to my Mac and see whether it
could make cool noises. I write songs; an absurdly over-engineered sampler for
Logic Pro or Bitwig sounded more interesting than another obedient chatbot.

Audex is therefore a Rube Goldberg synthesizer with a language model where the
reasonable design should be. This model is okay, but definitely not "I have
Mythos at home." More parameters will not fix your prompt. They may heat the room.

Windows support can wait. My 5800X3D, RTX 4080 with 16 GB VRAM, and 64 GB DDR4
machine is currently a Baldur's Gate 3 and flight-sim appliance. In AI terms, I
am apparently GPU-poor. What a stupid timeline.

Adult activities involving NVIDIA's voice model—including gooning and adjacent
research protocols—are *not yet tested*. Please do not send a benchmark corpus.

## Build (you must be this tall to ride)

This is for Apple Silicon Macs. You need native arm64 Python 3.12 or 3.13,
enough disk for the model and runtime, and approximately this much unified RAM:

- Audex 2B BF16: **24 GB recommended**
- Audex 30B-A3B NVFP4: **48 GB recommended**
- Audex 30B-A3B BF16: **96 GB recommended**

Could an eventual NVFP4 2B squeeze into 16 GB? Perhaps. The first analysis says
it would be tight, swap-happy, and not worth the quality trade. It should fit
comfortably at 24 GB, where the existing BF16 model already fits. So: maybe.

```sh
git clone https://github.com/mbarnson/Audex-Mac.git
cd Audex-Mac
```

There is no ceremonial build dance. The first run creates the local environments,
installs the pinned runtime, finds cached models, and asks before downloading one.

## Run

For a typed or push-to-talk conversation:

```sh
./start.sh
```

![A user types or talks, Audex operates a ridiculous speech machine inside the Mac, and spoken audio comes back](docs/assets/start-flow.png)

At `You:`, type and press Enter. Submit an empty prompt to start recording, then
press Enter again to stop. Shift+Enter inserts a newline in most sensible
terminals; Option+Enter is the fallback. Type `q` by itself to quit.

For the experimental sound-making workbench:

```sh
./sound.sh
```

![A songwriter requests a sound, Audex splits it into five candidates, and the songwriter auditions them blind](docs/assets/sound-lab-flow.png)

Describe a sound. Audex makes five candidates and opens a local blind audition
board. Pick with your ears before revealing which tiny jar of math made what.

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
[documentation map](docs/README.md), where they can be fiddled with safely.
