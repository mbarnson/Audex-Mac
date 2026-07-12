# Audex Browser Interface

`./start.sh web` launches a loopback-only browser application over the same
Audex/vLLM Metal runtime used by the CLI. The UI is plain HTML, CSS, and
JavaScript served by Python's standard-library HTTP server. A focused
`websockets` loopback server carries live spoken responses.

The implemented design that replaces completed-WAV playback with automatic
incremental speech is documented in
[Browser Low-Latency Speech Streaming](browser-low-latency-streaming.md).

## Modes

| Mode | Model path | Conversation KV identity |
|---|---|---|
| Text in, Text out | text generation | preserved |
| Text in, Speech out | text generation + Audex TTS | preserved |
| Speech in, Text out | Audex ASR + text generation | preserved |
| Speech in, Speech out | Audex ASR/direct response + Audex speech decoder | preserved |
| Audio in, Text out | free-form Audex audio understanding | separate request |
| Text in, Audio out | Sound Lab planning + CFG3 TTA + XCodec | separate request |
| Audio in, Audio out | audio understanding to literal caption, then CFG3 TTA | separate request |

The four conversational modes share one append-only model conversation. Mode
selection changes only the input/output adapters. It never creates a new
`VllmSpeechToSpeechSession`, changes the conversation state key, or clears the
prefix cache. Text-only modes stop before TTS so written responses do not pay
for discarded speech generation.

All browser chats share one heavy vLLM engine and speech decoder. Each chat has
its own persistent conversation ID, prompt history, and vLLM cache key. Moving
between chats activates the selected history on the shared engine; it does not
load another model. Requests are serialized around that shared mutable session.

## Audio Transport

The browser captures microphone samples with Web Audio, downmixes and resamples
them to 16 kHz mono, and encodes PCM16 WAV. Selected audio files go through the
same local conversion. Input audio uses base64 JSON because the server has no
multipart dependency. The server rejects non-RIFF/WAVE uploads and limits
request size to 64 MB.

Spoken model output takes a different low-latency path. Ordered JSON lifecycle
events and binary mono PCM16 frames travel over a per-turn WebSocket. A
long-lived browser `AudioContext` and `AudioWorklet` resample and buffer those
frames, then start playback automatically after an 80 ms prebuffer. Text deltas
update the visible assistant transcript while that audio is still being
generated. The completed WAV is persisted and exposed through the existing
chat-scoped media route for later replay; it is not the live transport.

Input recordings, spoken replies, and generated sounds are exposed only through
chat-scoped media URLs. The server resolves those URLs against persisted message
metadata instead of accepting filesystem paths from the browser.

## Persistence

- Browser presentation metadata: `.audex/web/chats/`
- Model conversation history/cache identity: `.audex/web/model-conversations/`
- Browser input audio: `.audex/web/uploads/`
- Speech run logs and WAVs: `.audex/web/runs/`
- Sound Lab catalog and generated assets: `.audex/web/sound-lab/`

Chat titles begin with an Audex name and can be edited inline. Browser messages
record the mode, visible transcript, media URL, and generated asset captions.
Sound Lab still retains its durable generation diagnostics, but the browser asks
for an unblinded view and never requires a preference vote or reveal action.

## Launch Options

```sh
./start.sh web --help
./start.sh web --no-open --port 8765
./start.sh web --model 2b
./start.sh web --model 30b
./start.sh web --model 30b-nvfp4
```

The server intentionally refuses non-loopback `--host` values. Use a deliberate
SSH tunnel if remote access is required; network hosting and authentication are
outside this product contract.
