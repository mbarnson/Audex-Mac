# Audex-Mac

Audex-Mac is a local Apple Silicon speech-to-speech demo for NVIDIA's
Nemotron-Labs-Audex models. It captures your voice, sends audio directly through
Audex, generates a text response, synthesizes Audex speech tokens, and plays the
decoded response—all on the Mac, without cloud APIs or separate STT/TTS models.

```sh
./start.sh
```

At the `You:` prompt, type a message and press Enter to send it directly to
Audex. Press Shift+Enter to insert a newline in Ghostty, Kitty, WezTerm, or
iTerm; Option+Enter is the fallback when a terminal does not report Shift+Enter
distinctly. Submit an empty message to start recording, then press Enter again
to stop. Typed and spoken turns share one conversation and both receive spoken
Audex responses.

## Status

The primary validated path is
[`nvidia/Nemotron-Labs-Audex-30B-A3B`](https://huggingface.co/nvidia/Nemotron-Labs-Audex-30B-A3B)
on a 128 GB Apple Silicon Mac.
[`nvidia/Nemotron-Labs-Audex-2B`](https://huggingface.co/nvidia/Nemotron-Labs-Audex-2B)
is supported by the loader and test scaffolding, but local end-to-end testing
has been minimal; do not assume the 30B quality or performance results transfer
to 2B.

The NVIDIA model card advertises up to a one-million-token context. This Mac
demo deliberately configures a 262,144-token active context to keep the
full-precision 30B model and vLLM Metal cache within unified-memory limits. The
complete transcript is persisted and re-prefilled when a conversation resumes.
Audex-Mac does not compact, summarize, or silently discard old turns.
On resume, Audex generates a one- or two-sentence greeting grounded in the most
recent exchange, then asks what to discuss next; that synthetic greeting prompt
is not added to the transcript.

Speech uses the faster no-CFG recipe by default. A controlled blind test
preferred CFG3 in four of six passages and nearly five of six after a close-call
reconsideration, but CFG roughly halves useful speech-codec throughput on the
current Metal path. Opt into the higher-quality CFG3 recipe with:

```sh
AUDEX_VLLM_TTS_CFG=1 ./start.sh
```

Spoken turns use Audex's direct audio-to-answer capability on the latency path,
then run Audex ASR after speech generation to persist the exact transcript. Set
`AUDEX_VLLM_DIRECT_AUDIO_RESPONSE=0` to diagnose the older serial
ASR-to-text-to-TTS pipeline. For short utterances, the direct path drops only
the audio encoder's padded tail before 30B prompt prefill; deferred ASR still
uses Audex's full 750-embedding audio representation. Set
`AUDEX_VLLM_DIRECT_AUDIO_TRIM_PADDED_EMBEDDINGS=0` to disable that optimization.

## Requirements

- Apple Silicon Mac
- native arm64 Python 3.12 or 3.13
- enough disk for NVIDIA's model snapshot and the generated vLLM Metal runtime
- substantial unified memory for full-precision 30B; 128 GB is the tested target
- microphone and speakers for the interactive demo

On first run, `start.sh` creates the local environment, installs the pinned
vLLM Metal runtime, detects cached Audex models, and asks before downloading a
missing checkpoint. If 30B is fully cached it is preferred; otherwise the
first-run download choice is 2B.

Build the quality-first MLX NVFP4 variant locally with:

```sh
./scripts/quantize-30b-nvfp4.sh
```

Or download the published conversion from
[`txgsync/Nemotron-Labs-Audex-30B-A3B-NVFP4-mlx`](https://huggingface.co/txgsync/Nemotron-Labs-Audex-30B-A3B-NVFP4-mlx).

The conversion quantizes only the routed MoE experts and preserves the audio
encoder/projector, speech decoder, routers, attention, Mamba, embeddings, and
LM head at source precision. Once its Hugging Face-shaped local snapshot is
complete, plain `./start.sh` prefers it over the BF16 30B checkpoint. See the
[NVFP4 recipe](docs/engineering/nvfp4-quantization.md) for the exact policy and
reproduction contract.

## What Runs Locally

One persistent vLLM Metal engine serves the Audex speech pipeline:

1. Audex's own audio encoder/projector maps the recorded utterance into the
   Audex text backbone, which generates a direct non-thinking answer.
2. The answer streams into no-CFG Audex speech-codec generation by default;
   CFG3 remains the opt-in quality recipe.
3. NVIDIA's bundled Audex causal speech decoder produces 16 kHz audio.
4. The low-latency playback path incrementally resamples that PCM to 192 kHz
   for the selected Core Audio device; deferred Audex ASR persists the exact
   transcript.

On the tested Mac, the local NVFP4 30B snapshot produced coherent model audio
at an estimated DAC onset of 0.959 seconds after turn submission in two
consecutive fixture runs. That result uses no-CFG, a one-frame causal decoder
primer, 80 ms application prebuffering, and a 192 kHz/512-frame output stream.
The full-precision BF16 30B checkpoint remains substantially slower.
The BF16 Audex-2B checkpoint ran the same fixture twice at estimated DAC onsets
of 0.578 and 0.553 seconds, while retaining Audex's full semantic audio path.
Its checkpoint declares a 131,072-token context limit, so Audex-Mac clamps the
conversation and engine budgets to 128K when 2B is selected.

Those short-fixture latency results are historical compute checkpoints. Current
interactive playback prioritizes glitch-free long-form speech: it requests a
50 ms device latency, gives the active TTS request exclusive engine steps, and
synthesizes future speech chunks sequentially. A 67.5-second NVFP4 30B essay
played with zero initial, device, or application-queue underruns using the 80 ms
application prebuffer.

Audex-Mac does not load Whisper, Kokoro, Silero, a cloud LLM, or another
semantic fallback. Host-side PCM preparation and device I/O are deterministic
plumbing; the semantic path remains Audex end to end.

## Useful Commands

Start a new conversation instead of resuming the current one:

```sh
./start.sh --new-conversation
```

Select the validated model explicitly:

```sh
./start.sh --model audex-30b-a3b
```

Select the local NVFP4 quality trial explicitly:

```sh
./start.sh --model audex-30b-a3b-nvfp4
```

Replay a fixture without microphone input or playback:

```sh
./start.sh --model audex-30b-a3b \
  --input-wav .audex/fixtures/example.wav --no-play
```

Allow thinking mode:

```sh
./start.sh --thinking
```

Use the older direct MLX implementation for diagnostics:

```sh
./start.sh --sts-backend mlx
```

See the [runbook](docs/operations/runbook.md) for preflights, dependency refresh,
fixture generation, benchmarks, and the blind TTS quality workflow.

## Quality Evidence

The checked-in quality harness contains six 46-to-80-word passages covering
conversation, proper names, numbers, technical vocabulary, expressive dialogue,
and long-form stability. It compares three code-enforced recipes with identical
text boundaries, fixed seed, compact codec-window decode, and the production
waveform decoder.

| Recipe | Human wins | Mean ASR WER | Required-term recall |
|---|---:|---:|---:|
| Plain, no CFG | 1 | 7.0% | 70.0% |
| NVIDIA TTS CFG2 | 1 | 8.3% | 75.6% |
| Audex CFG3 | 4 | 12.3% | 70.0% |

The result is intentionally modest: one listener, one seed, and six passages.
Its useful finding is that transcription accuracy did not predict perceived
prosody. CFG3 was preferred for natural cadence and voice stability, while the
no-CFG samples produced most of the robotic, harsh-onset, and multi-voice
complaints.

The corpus and evaluation tools are reproducible from a fresh checkout. Install
the optional MLX-Audio evaluator with `pip install -e '.[oracle]'`; raw WAVs,
model outputs, and private blind keys remain local artifacts under `.audex/`
and are never committed.

## Runtime Notes

Audex support currently depends on Audex-Mac-owned patches over a pinned vLLM
Metal commit. Patch installation is guarded and fails loudly if the upstream
API shape changes. vLLM may log a CPU compatibility facade even while inference
runs through MLX on `Device(gpu, 0)`; use the device diagnostics rather than the
facade string when verifying Metal execution.

Known limitations:

- 30B is the substantially tested model; 2B needs broader live validation.
- Long resumed conversations are re-prefilled because vLLM conversation-state
  cache persistence is not implemented yet.
- CFG3 favors speech quality over the faster no-CFG path and disables current
  text-to-TTS interleaving.
- This is a terminal typed/push-to-talk demo: no VAD, barge-in, GUI, or server
  API.
- Audex pronunciation remains imperfect on adversarial names and identifiers.

## Documentation

- [Documentation map](docs/README.md)
- [Demo scope](docs/project/scope.md)
- [Runbook](docs/operations/runbook.md)
- [Current evidence and limitations](docs/engineering/viability.md)
- [vLLM Metal engineering history](docs/engineering/vllm-metal.md)
- [Patch ledger](docs/engineering/patches.md)
- [Autonomous audio-capability evaluation](docs/engineering/autonomous-audio-evaluation.md)
- [Stable patch contract](PATCH.md)

## License and Repository Hygiene

Audex-Mac source code is MIT licensed. NVIDIA's model weights and inference
components are not covered by this repository's MIT license; the model cards
currently apply NVIDIA's Oneway Noncommercial License.

Downloaded weights, local environments, cache state, generated conversations,
run logs, and all audio files are ignored. Large binary artifacts—including
WAV samples—do not belong in Git.

Install the local pre-commit hook with:

```sh
scripts/install-hooks.sh
```
