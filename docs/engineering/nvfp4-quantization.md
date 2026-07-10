# Audex-30B NVFP4 Conversion

Audex-Mac's first quality-focused four-bit recipe quantizes only the routed MoE
expert projections. The model remains an MLX-native W4A16 artifact: weights in
the routed experts use NVFP4 with group size 16, while activations and every
non-expert component retain their source precision.

This policy captures most of the memory and bandwidth reduction because the
routed experts contain about 29.4B of the model's 32.6B parameters. It avoids
combining several sources of error during the first conversational quality
trial.

## Precision policy

| Component | Stored precision |
|---|---|
| Routed `switch_mlp.fc1` and `switch_mlp.fc2` | NVFP4, group 16 |
| Router and shared experts | Source precision |
| Mamba projections and state parameters | Source precision |
| Attention Q/K/V/O | Source precision |
| Embeddings, norms, and LM head | Source precision |
| Audio encoder and projector | Source precision |
| Causal speech decoder | Source precision |

MLX-LM sanitization fuses the 128 routed experts into one `SwitchLinear` per
projection. The conversion therefore quantizes 46 fused projections: `fc1` and
`fc2` for each of the 23 MoE layers. The recipe validates that no other module
has gained a quantization scale tensor before publishing the snapshot.

## Reproduce

Cache the complete NVIDIA 30B speech snapshot first (this command downloads it
when it is absent), then run the conversion:

```sh
./start.sh --model audex-30b-a3b --yes-download --preflight-audio-runtime
./scripts/quantize-30b-nvfp4.sh
```

The output is a Hugging Face-shaped local cache entry for
`txgsync/Nemotron-Labs-Audex-30B-A3B-NVFP4-mlx`, including `refs/main` and a
deterministic snapshot revision derived from the source revision and recipe ID.
Large weights remain outside Git. Unchanged source assets are APFS hard-linked
when possible and copied otherwise; the output can be uploaded as normal files.

Use `--replace` to rebuild the same deterministic local revision after changing
the implementation. A recipe change intended to produce a distinct artifact
must also change `RECIPE_ID`.

## Why this is not called oQe

The current oMLX enhanced/imatrix path applies its weighted rounding only to
affine quantization. This recipe uses MLX's native NVFP4 quantizer and therefore
does not claim oQe calibration. A later revision can add target-format-aware
oracle scoring and promote sensitive whole MoE projections or layers to source
precision without changing the protected audio boundaries.

## Quality gate

The artifact is experimental until it has been compared with BF16 on:

- natural multi-turn conversation and reasoning;
- ASR and general audio understanding;
- conditional and classifier-free-guided TTS codec generation;
- speech intelligibility and voice stability; and
- long-context retrieval.

The machine-readable policy is also written into each generated snapshot as
`quantization_recipe.json`.

## Published artifact

The reproducible conversion is published as
[`txgsync/Nemotron-Labs-Audex-30B-A3B-NVFP4-mlx`](https://huggingface.co/txgsync/Nemotron-Labs-Audex-30B-A3B-NVFP4-mlx).
Its generated model card keeps the introduction concise, links back to this
demonstration repository, and preserves NVIDIA's complete upstream card. The
selective-precision policy is recorded above and in the artifact's
`quantization_recipe.json`; runtime and measured latency evidence remain in the
Audex-Mac repository. Hugging Face metadata relates the artifact to the NVIDIA
base as a quantization so it appears in the base model's model tree.
