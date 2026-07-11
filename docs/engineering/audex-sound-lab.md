# Audex Sound Lab

This document proposes a conversational workbench for exploring Audex's
non-speech audio understanding and generation capabilities. The intended user
experience is as simple as the existing speech demo:

```console
./sound.sh
```

The user can speak or type requests such as "make five different explosions,"
audition blind variants as they finish, ask for refinements, or say "listen, what
do you hear?" before recording a short environmental-audio sample. The system
keeps a searchable local catalog of every attempt and what was learned from it.

This is a product and implementation direction, not a change to `start.sh`.
The working near-real-time conversation path remains isolated.

## Implementation Status

Phase 1 is implemented as a typed vertical slice through `./sound.sh`. It uses a
single persistent Audex/vLLM Metal runtime for strict tool planning, designed
variant captions, and continuously batched CFG3 text-to-audio generation. One
five-way audition is submitted in waves of at most two lockstep CFG pairs,
matching NVIDIA's documented `cfg-pairs-per-batch=2`. `sound.sh` gives this bounded
workload an 8K engine context and four non-paged KV slots; those settings are
scoped to Sound Lab and do not change `start.sh`. The local blind board opens
automatically, publishes complete XCodec WAVs, records a winner and note, and
persists recipes and artifacts under `.audex/sound-lab/`.

Generation failures use a separate exploratory policy from the autonomous
benchmark. Sound Lab now follows NVIDIA's released inference script: any
nonempty phase-valid codec stream is decoded, truncated or padded with `1e-3`
near-silence to ten seconds, then enhanced. Phase-invalid candidates retry once
in waves of at most two pairs with deterministic alternate seeds. A second
failure remains in the catalog with failure classes, frame count, duration, and
end-token status.

Every initial and retry attempt retains its actual seed, timing, frame count,
duration, end-token status, and failure classes; a recovered candidate reveals
the retry seed that actually produced its WAV. First-pass successes are yielded
to the catalog before the retry batch begins, so a retry-engine failure cannot
turn an already-ready candidate into a failure. Board WAVs pass through
NVIDIA's released 48 kHz enhancement VAE; raw 16 kHz XCodec1 WAVs remain beside
them for diagnosis.

This is deliberately not described as the complete v1 product. The terminal is
render-blocking in Phase 1. Explicit scheduler work classes, concurrent parent
conversation, spoken status, live `listen` capture, blind self-audit, and FTS5
catalog memory remain Phases 2 through 4 below. `start.sh` behavior remains
unchanged.

Local validation on 2026-07-10: `scripts/lint.sh` passed and
`.venv/bin/python -m pytest -m fast` passed 758 tests with 4 intentionally
deselected. The loopback board integration used a real ephemeral HTTP server.
An owner-run Apple Silicon audition produced recognizable dog barks through the
full CFG3/XCodec path. The continuous-batch and bounded-retry revision still
requires an owner-run timing and listening pass.

## Why Build It

Audex is unusual because one model can participate in all of these roles:

- converse through speech or text;
- understand speech and non-speech audio;
- generate speech and non-speech audio;
- plan work and emit structured tool calls;
- review and describe its own generated artifacts.

The existing autonomous evaluation harness answers whether those capabilities
score well on pinned datasets and metrics. Sound Lab answers a different
question: what can a person make and discover with them?

The useful loop is not a benchmark form. It is a conversation in which sound
requests become background jobs, candidates arrive while the conversation can
continue, and each audition makes the local catalog more informative.

## Product Principles

1. **One command.** `./sound.sh` is the only entry point the user must learn.
2. **Audex remains the product.** Do not replace audio input, reasoning, review,
   speech, or sound generation with unrelated cloud models.
3. **Local first.** Audio, conversation state, job state, reviews, and catalog
   data remain on the Mac under `.audex/`.
4. **Conversation stays responsive.** Rendering is asynchronous. A tool call
   acknowledges queued work instead of blocking the parent conversation until
   every WAV is complete.
5. **Audition before explanation.** Candidate identities are blind by default;
   production settings must not bias the listener's preference.
6. **Failures are evidence.** Rejected, malformed, or unpleasant outputs remain
   cataloged locally with their outcome instead of silently disappearing.
7. **Claims retain provenance.** Human preference, deterministic measurement,
   and model self-review are separate evidence classes.
8. **The speech demo is protected.** Sound Lab may reuse implementation
   components, but it must not alter `start.sh` behavior as a side effect.

## Product Shapes Considered

These shapes are alternatives or stages, not eight separate products.

### 1. Conversational Foley Console

A terminal-only session closely resembling `start.sh`. The user speaks or types,
jobs appear as compact status rows, and keys select blind candidates A-E.

This is the smallest interface and the fastest route to exercising the model.
It becomes cramped when several jobs, waveforms, lineage, and catalog search are
visible at once.

### 2. Terminal Conversation With Audition Board

The terminal remains the conversational home while a local browser page opens
automatically as a live contact sheet. The browser shows blind candidates as
they complete, provides playback and preference controls, and exposes lineage
without turning the experience into a full DAW.

This is the recommended first product. It preserves the simplicity and
observability of the current CLI while giving audio artifacts an interface that
suits them.

### 3. Native SwiftUI Soundroom

A native Mac app presents a voice orb, conversation, background tool activity,
variant rack, waveform inspection, and catalog browsing. The interaction maps
well onto the background-injection and tool-row patterns already explored in
ClatterClaw.

This is an attractive later shell, especially for microphone permissions,
Core Audio, media keys, and polished playback. Building it first would make UI
work the critical path before the orchestration behavior is understood.

### 4. Headless Sound Foundry

A local daemon owns the engine, job queue, event stream, and catalog. Terminal,
browser, and future native clients attach to the same service.

This is a useful architecture if multiple clients or long-lived background
rendering become important. It is unnecessary process topology for the first
working version; the v1 interface can preserve this seam without immediately
deploying a daemon.

### 5. Self-Dispatching Foley Crew

One parent Audex conversation delegates to logical child roles using the same
loaded model: sound designer, renderer, blind audio auditor, and librarian. The
parent stays responsive while children produce and review artifacts.

This is not a claim that every child has a reusable conversation KV prefix.
Planner and reviewer requests can share an exact text prefix when their prompt
hashes match. Text-to-audio requests use the `<audiogen_start>` structure and do
not share the parent's conversation prefix, although they still use the same
engine and continuous batching.

### 6. Live Acoustic Notebook

The user says "listen" and the system arms an exclusive recording window after
current playback drains. Audex then answers a question about the captured sound,
and the clip plus answer can become a catalog item.

This makes audio understanding tangible without first locating datasets or
files. It also creates useful paired material for later capability experiments.

### 7. Sound Atlas

The catalog becomes a map of what Audex has attempted: prompts, acoustic
families, failures, preferences, self-observations, and reproducible settings.
The model can answer questions about its history and deliberately explore weak
or uncertain regions when the user asks.

The atlas is valuable only if it distinguishes evidence from narrative. Audex's
description of its own output is a hypothesis, not ground truth.

### 8. Conversational DAW

The user asks for layers, timing changes, composites, and revisions on a visual
timeline. This could eventually turn generated assets into designed scenes.

It is intentionally deferred. Waveform editing, mixing, and nondestructive
timeline semantics would overwhelm the first question: what can Audex itself
understand and generate?

## Recommended Experience

The first implementation combines the terminal-and-board interface, the
self-dispatching Foley crew, the acoustic notebook, and the beginnings of the
Sound Atlas.

On startup:

1. `./sound.sh` loads one persistent Audex/vLLM Metal engine.
2. It resumes or creates a Sound Lab conversation and catalog session.
3. It starts a local-only board server on an available loopback port.
4. It opens the board automatically in the default browser.
5. The terminal accepts push-to-talk input and typed prompts.

When the user asks for five explosions:

1. The parent conversation calls `render_sounds` and immediately receives an
   acknowledgment containing an opaque job ID.
2. A sound-designer child converts the request into five intentionally distinct
   briefs and seeds. "Five" means designed variation, not five accidental
   samples from an identical prompt.
3. Render work is admitted in bounded waves so it does not monopolize the
   engine or memory.
4. Each completed WAV is persisted and published to the board immediately.
5. A blind audio-auditor child receives the waveform and review question but
   not the requested caption or production recipe.
6. The parent can keep conversing and speaking short, no-CFG status messages
   while work proceeds.
7. The user auditions candidates labeled A-E and records a preference, rejection,
   or note by voice, terminal key, or board control.
8. A librarian records the outcome, observations, lineage, and evidence class.

The label-to-asset mapping remains stable for the session. It is stored so later
comments such as "B had a better tail" remain meaningful. Recipe details may be
revealed after the user records a preference.

"More like B" in v1 means derive a new brief from B's prompt, settings, human
notes, and model observations. It must not imply waveform-conditioned generation
until the model path actually supports that operation.

## Example Sessions

### Generate And Audition

```text
You: I'd like to audition five explosions. Make them genuinely different.
Audex: I'll work up five distinct explosions. We can keep talking while they render.

  [working] job 7KQ2  designing 5 variants
  [ready]   A ready   [board]
  [working] B rendering
  [queued]  C queued

Audex: The first one is ready to hear.
You: A is too cinematic. I want something closer and uglier.
Audex: Got it. I'll keep that as evidence and steer the next pass toward a close,
       physical blast rather than a trailer impact.
You: B is the winner. Give me three more like B, but vary the space around it.
```

### Listen To The Room

```text
You: Listen. What do you hear?
Audex: Ready. I'll listen for ten seconds after the tone.
  [playback drains] [earcon] [10-second capture]
Audex: I hear a steady ventilation hum, several keyboard impacts nearby, and a
       short metallic click near the end. The hum masks some low-level detail.
You: Save that and make two stylized versions of the metallic click.
```

## Agent And Tool Contract

The parent model gets a deliberately small tool surface. Tool calls use the
Nemotron chat template's structured XML form and are validated strictly before
dispatch. The stock "no tools" system instruction must be replaced only for the
Sound Lab session, not globally.

Every operation that can wait on the model, audio hardware, or catalog scan
returns a small acknowledgment immediately. Its progress and final result arrive
later through typed events and compact parent-context injections. `soundboard`
is the exception only in the sense that it performs a fast local UI action; it
still returns immediately and never waits on generation or playback completion.

### `render_sounds`

```text
render_sounds(
    brief: str,
    count: int,
    constraints: object | null,
    parent_asset_ids: list[str]
) -> {job_id: str, accepted_count: int}
```

Queues designed text-to-audio variants. The synchronous result is only an
acknowledgment. Progress and results arrive later as events and compact context
injections.

### `listen`

```text
listen(question: str, seconds: float = 10.0)
    -> {capture_id: str, state: "arming"}
```

Coordinates exclusive capture. It drains current speech playback, emits a
start cue, records the requested window, saves the WAV, and submits the audio
with the question through Audex's audio-understanding request path. The tool
returns in the `arming` state; capture progress and the eventual answer arrive
as asynchronous events rather than inside the tool result.

### `soundboard`

```text
soundboard(
    action: "show" | "play" | "reveal" | "focus",
    labels: list[str],
    job_id: str | null,
    note: str | null
) -> {accepted: bool}
```

Controls board state without exposing filesystem paths or mutable catalog
internals to the model.

### `inspect_catalog`

```text
inspect_catalog(
    query: str,
    question: str | null,
    asset_ids: list[str]
) -> {job_id: str}
```

Runs an asynchronous, potentially exhaustive catalog query. Normal turns receive
a small automatically constructed set of relevant catalog memories without a
tool call; this tool is for deliberate investigation.

The dispatcher returns errors explicitly. It must not silently translate an
invalid tool call, invent defaults that change the requested work, or fall back
to a remote service.

Model-authored sound-design JSON uses a separate bounded normalization policy.
The harness may extract one unambiguous JSON object from prose or a Markdown
fence, accept the documented `variants`/`sounds`/`candidates`,
`caption`/`prompt`/`description`, and
`difference`/`rationale`/`variation` aliases, ignore unrelated metadata, and
derive generation seeds deterministically on the host. Semantic invariants
remain strict: the candidate count must match, captions must be nonempty, and
conflicting aliases are rejected. If normalization fails, the model gets one
repair turn. A second failure stops the job. Every literal attempt and whether
repair was used are persisted on the job for diagnosis.

## Logical Roles

Roles are request policies over one model and engine, not separately hosted
models.

### Parent

Maintains the human conversation, decides when a request requires a tool, speaks
short acknowledgments and completion notices, and incorporates compact job
events. It never waits synchronously for a render batch.

### Sound Designer

Turns a user goal into distinct, reproducible candidate briefs. It should vary
meaningful acoustic dimensions such as source, distance, environment, duration,
temporal envelope, material, recording character, and intensity. It records why
each variant differs.

### Renderer

Builds CFG3 text-to-audio pairs using the existing TTA request builder and
decodes XCodec output through the full audio path. The current quality evidence
favors CFG3, so Sound Lab begins there. Performance experiments may offer a
separate no-CFG recipe, but must label it and never silently change the default.

### Blind Audio Auditor

Uses direct Audex audio understanding to describe the resulting waveform,
technical defects, and audible events. It does not see the requested caption,
candidate recipe, or sibling reviews before producing its first report. That
prevents the review from merely restating intent.

### Librarian

Indexes artifacts, queries prior work, summarizes capability evidence, and
injects only compact relevant findings into the parent conversation. It cannot
promote a model self-review into a human preference or measured fact.

## Runtime Architecture

V1 should remain one Python process unless measurements justify separation:

```text
terminal input / microphone
           |
           v
  parent Audex conversation <---- compact task and catalog events
           |
           v
    validated tool dispatcher
      |          |          |
      v          v          v
   job queue   capture   catalog/FTS
      |
      v
 one persistent AudexAsyncVllmRuntime
      |          |          |
   planning   TTA render   audio review
      |
      v
 XCodec WAV artifacts ----> local board event stream
```

Existing implementation seams already support much of this design:

- `AudexAsyncVllmRuntime.stream_many(...)` and
  `generate_many_final(...)` submit concurrent requests;
- `build_tta_requests(...)` creates the CFG3 conditional/unconditional
  text-to-audio pair;
- `AudexVllmTtaGenerationAdapter` performs the existing full XCodec decode;
- `build_audio_messages_response_request(...)` constructs direct audio
  understanding requests.

The board should initially be a small local HTTP application with vanilla
JavaScript and a server-sent event or WebSocket feed. It needs transport
controls, job state, blind labels, notes, reveal, and lineage - not a frontend
framework or waveform editor. A future SwiftUI Soundroom can consume the same
event and catalog contracts.

## Scheduling And Responsiveness

Continuous batching makes concurrent work possible but does not make every
mixture of work fair. The current vLLM Metal patch identifies a request as TTS
when it belongs to a CFG `cond`/`uncond` pair. Text-to-audio renders use the same
pair naming, so the scheduler can misclassify a long sound render as priority
interactive speech and hold ordinary text or audio-understanding work behind it.

Before Sound Lab renders and conversation share an engine, introduce explicit
request work classes:

- `interactive_text`
- `interactive_speech`
- `sound_render`
- `sound_review`
- `background_catalog`

CFG pair members must remain lockstep-compatible, but CFG membership must not
imply interactive priority. Admission should favor interactive text and speech,
then short review work, while allowing bounded sound-render progress. Record
queue wait, prefill, decode, audio duration, and end-to-end completion per class.

This scheduling fix is a Sound Lab prerequisite. It should be implemented as a
general request-metadata seam, not a special case that recognizes job names.

Spoken progress should remain concise and use the fast no-CFG speech recipe.
Do not synthesize every low-level event. Terminal and board status can update
frequently; speech should announce the first candidate, meaningful completion,
and errors.

## Capture Discipline

Sound playback and microphone capture cannot overlap accidentally. `listen`
uses an explicit state sequence:

1. announce readiness;
2. stop accepting new speech playback;
3. drain queued playback and close the output stream;
4. play a short earcon or countdown;
5. capture exclusively for the requested duration;
6. persist the original WAV and capture diagnostics;
7. reopen playback only after capture closes;
8. submit the saved audio to Audex understanding.

V1 supports live microphone capture. Arbitrary folder import is deferred so the
first experience tests the conversational loop rather than becoming an asset
manager.

## Catalog And Evidence Model

Store the catalog in SQLite with FTS5 under `.audex/sound-lab/`. WAV files and
large diagnostics remain adjacent local artifacts and stay out of Git.

Minimum entities:

### Session

- stable session and conversation IDs
- model and runtime identity
- start/end timestamps
- system prompt hash and recipe versions

### Job

- opaque ID and parent conversation turn
- requested brief, count, constraints, and state
- timestamps and failure details
- parent job or source assets for refinement lineage

### Candidate

- opaque asset ID and session-blind label
- generated brief, seed, recipe, model, and source job
- WAV path, hashes, duration, sample format, and structural diagnostics
- states including ready, preferred, rejected, failed, and archived

### Observation

- asset or job target
- source: `human`, `metric`, or `model`
- observer identity and prompt/config hash where applicable
- structured tags plus free-form note
- timestamp and supersession relationship

### Preference

- audition set and selected/rejected candidates
- ranking or pairwise relationship
- human explanation, if supplied
- whether recipes were still blind at decision time

Rejected WAVs remain present as negative evidence. Retention can become a
user-invoked maintenance policy later, but rejection must not delete them.

The parent receives a small relevant-memory block constructed from FTS matches,
lineage, recent preferences, and capability tags. Exact prompt-prefix hashes
must be recorded whenever KV reuse is claimed. Sharing one engine is not itself
evidence of KV-prefix reuse.

## Blindness And Review Policy

Blind labels are random session-local letters, not ordered recipe names.
Filename, UI ordering, timing text, and spoken announcements must not reveal
CFG, seed, prompt variant, or production order before preference capture.

The first model audit is also blind to intent. After that report is committed,
the librarian may compare it with the requested brief and add a separate
alignment observation. Never overwrite the blind report with the post-reveal
interpretation.

Human judgment is authoritative for "sounds best" and other preferences.
Metrics are authoritative only for what they measure. Model observations are
searchable hypotheses. A catalog summary must retain those distinctions.

## Autonomous Behavior

Sound Lab may review and organize user-requested work while idle. It may:

- finish queued candidates;
- run blind audits;
- compute local diagnostics;
- extract tags and candidate relationships;
- update catalog summaries;
- notify the parent that meaningful results are ready.

It may not invent and launch new render jobs merely to fill idle time. Broader
autonomous exploration can be considered later behind an explicit mode, budget,
and stop control.

## Implementation Roadmap

Phases 1 through 4 together constitute v1: conversational generation, live
listening, blind audition, and the durable local catalog. Phase 1 is a vertical
tracer bullet for validating the riskiest generation and board seams; it is not
the agreed product release by itself. Phase 5 is optional follow-on work.

### Phase 1: Vertical Slice

- add `sound.sh` and a Sound Lab CLI separate from `start.sh`;
- load one persistent runtime;
- accept typed prompt input before adding microphone interaction;
- validate one `render_sounds` tool call;
- generate a small designed CFG3 batch through the existing full TTA adapter,
  with at most two CFG pairs continuously scheduled per NVIDIA-shaped wave;
- decode phase-valid streams, pad/trim to the fixed ten-second target, retry
  phase-invalid candidates once, and retain actionable failure diagnostics;
- preserve raw XCodec1 audio and audition NVIDIA enhancement-VAE output;
- save artifacts and a minimal SQLite catalog;
- display blind candidates on an automatically opened local board;
- record a winner, rejection, and note;
- prove `start.sh` behavior and tests are unchanged.

### Phase 2: Responsive Conversation

- add explicit request work classes and fairness diagnostics;
- keep the parent text conversation live during rendering;
- add short no-CFG spoken acknowledgments and completion notices;
- stream candidate-ready and job-state events to the terminal and board;
- inject compact task results into the parent context.

### Phase 3: Audio Understanding

- add exclusive `listen` capture with drain, earcon, and diagnostics;
- send the capture through direct Audex audio understanding;
- add blind audits for generated candidates;
- preserve blind and post-reveal observations separately.

### Phase 4: Sound Atlas

- add FTS5 search, tags, lineage, and relevant-memory construction;
- support "more like B" as metadata/prompt lineage;
- add catalog inspection and capability summaries;
- run idle review only for user-requested artifacts.

### Phase 5: Optional Native Shell

- stabilize an event/catalog API independent of the browser board;
- evaluate a SwiftUI Soundroom using the same daemon or process interface;
- retain the terminal client as a first-class diagnostic surface.

## Acceptance Criteria For The First Useful Release

- `./sound.sh` starts the engine, terminal conversation, and local board without
  additional setup beyond the existing Audex installation.
- A typed or spoken request can queue two to five meaningfully distinct sounds.
- Candidate WAVs use the complete XCodec audio pipeline and are playable as soon
  as each finishes.
- The user can continue a text conversation while rendering proceeds.
- The board uses stable blind labels and reveals recipes only after preference
  capture.
- Preferences, rejections, blind model reviews, metrics, recipes, and lineage
  survive restart in the local catalog.
- A `listen` request captures an exclusive ten-second window and Audex answers a
  question about the saved audio.
- Rejected and failed attempts remain inspectable local evidence.
- No generated WAV or dataset blob is added to Git.
- `start.sh` remains behaviorally unchanged and its existing tests continue to
  pass.
- Scheduler diagnostics demonstrate that background sound rendering cannot
  indefinitely starve interactive text, speech, or audio review.

## Relationship To Autonomous Evaluation

[Autonomous audio-capability evaluation](autonomous-audio-evaluation.md) owns
pinned datasets, qualification gates, structural checks, and reproducible
metrics. Sound Lab should call those components where useful, but it must not
blur exploratory human preference into benchmark evidence.

Conversely, Sound Lab's catalog can reveal recurring failure families worth
turning into future fixed evaluation fixtures. Promotion into a gate is a
deliberate engineering action: pin the artifact or generation recipe, define
the expected observation, and record its provenance.
