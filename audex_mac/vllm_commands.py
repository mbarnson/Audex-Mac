"""CLI and probe adapters around the reusable vLLM speech session."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

from .audio_pcm import SAMPLE_RATE, load_wav_pcm
from .conversations import Conversation, ConversationStore
from .interactive_input import InputKind, read_turn_input
from .patches.vllm_metal_cfg import CFG_TTS_WINDOW_DECODE_ENV
from .personas import Persona
from .speech_output import RUNS_DIR, SpeechOutputSmokeResult, write_pcm16_wav
from .sts_cli import (
    DEFAULT_RESPONSE_MAX_TOKENS,
    SpeechToSpeechTurnResult,
    _Recording,
    _start_recording,
)
from .tts_quality import TtsQualityRecipe, load_tts_quality_corpus
from .vllm_sts_cli import (
    VllmSpeechToSpeechSession,
    _preemptive_generation_enabled,
    _vllm_tts_cfg_enabled,
    _wait_for_turn_submission,
)


def run_vllm_fixture_turn(
    *,
    full_model_path: Path,
    decoder_path: Path,
    input_wav_path: Path,
    selected_model_repo: str | None = None,
    output_dir: Path = RUNS_DIR,
    play: bool = True,
    response_max_tokens: int = DEFAULT_RESPONSE_MAX_TOKENS,
    speech_max_tokens: int | None = None,
    thinking_enabled: bool = False,
    conversation: Conversation | None = None,
    conversation_store: ConversationStore | None = None,
    persona: Persona | None = None,
    session: VllmSpeechToSpeechSession | None = None,
) -> SpeechToSpeechTurnResult:
    created_session = session is None
    if session is None:
        session = VllmSpeechToSpeechSession(
            full_model_path=full_model_path,
            decoder_path=decoder_path,
            selected_model_repo=selected_model_repo,
            output_dir=output_dir,
            thinking_enabled=thinking_enabled,
            response_max_tokens=response_max_tokens,
            speech_max_tokens=speech_max_tokens,
            conversation=conversation,
            conversation_store=conversation_store,
            persona=persona,
        )
    try:
        return session.run_turn_from_wav(input_wav_path=input_wav_path, play=play)
    finally:
        if created_session:
            session.shutdown()


def run_vllm_preemptive_replay(
    *,
    full_model_path: Path,
    decoder_path: Path,
    input_wav_path: Path,
    submission_delay_seconds: float,
    replay_realtime: bool = True,
    replay_chunk_seconds: float = 0.02,
    selected_model_repo: str | None = None,
    output_dir: Path = RUNS_DIR,
    play: bool = True,
    response_max_tokens: int = DEFAULT_RESPONSE_MAX_TOKENS,
    speech_max_tokens: int | None = None,
    thinking_enabled: bool = False,
    session: VllmSpeechToSpeechSession | None = None,
) -> SpeechToSpeechTurnResult:
    """Replay a completed utterance and submit after a controlled staging window."""

    class ReplayStream:
        def stop(self) -> None:
            pass

        def close(self) -> None:
            pass

    loaded = load_wav_pcm(input_wav_path)
    if loaded.sample_rate != SAMPLE_RATE:
        raise ValueError(
            f"Audex replay input must be {SAMPLE_RATE} Hz, got {loaded.sample_rate} Hz."
        )
    recording = _Recording(stream=ReplayStream())
    output_dir.mkdir(parents=True, exist_ok=True)
    replay_wav_path = (
        output_dir / f"preemptive-replay-{time.strftime('%Y%m%d-%H%M%S')}.wav"
    )

    def wait_for_submission() -> float:
        if replay_realtime:
            chunk_samples = max(1, int(SAMPLE_RATE * replay_chunk_seconds))
            replay_started = time.monotonic()
            for offset in range(0, len(loaded.samples), chunk_samples):
                recording.append_chunk(loaded.samples[offset : offset + chunk_samples])
                replay_target = (
                    replay_started
                    + min(
                        len(loaded.samples),
                        offset + chunk_samples,
                    )
                    / SAMPLE_RATE
                )
                time.sleep(max(0.0, replay_target - time.monotonic()))
        else:
            recording.append_chunk(loaded.samples)
        time.sleep(max(0.0, float(submission_delay_seconds)))
        return time.time()

    created_session = session is None
    if session is None:
        session = VllmSpeechToSpeechSession(
            full_model_path=full_model_path,
            decoder_path=decoder_path,
            selected_model_repo=selected_model_repo,
            output_dir=output_dir,
            thinking_enabled=thinking_enabled,
            response_max_tokens=response_max_tokens,
            speech_max_tokens=speech_max_tokens,
        )
    try:
        return session.run_preemptive_recorded_turn(
            recording=recording,
            input_wav_path=replay_wav_path,
            play=play,
            wait_for_submission=wait_for_submission,
        )
    finally:
        if created_session:
            session.shutdown()


def run_vllm_tts_text_probe(
    *,
    full_model_path: Path,
    decoder_path: Path,
    text: str,
    selected_model_repo: str | None = None,
    output_dir: Path = RUNS_DIR,
    play: bool = False,
    speech_max_tokens: int | None = None,
) -> SpeechOutputSmokeResult:
    session = VllmSpeechToSpeechSession(
        full_model_path=full_model_path,
        decoder_path=decoder_path,
        selected_model_repo=selected_model_repo,
        output_dir=output_dir,
        speech_max_tokens=speech_max_tokens,
    )
    try:
        return session.generate_speech_output(
            text=text,
            max_tokens=session._speech_max_tokens_for_text(text),
            play=play,
            artifact_prefix="tts-text-probe-vllm",
        )
    finally:
        session.shutdown()


def run_vllm_tts_quality_probe(
    *,
    full_model_path: Path,
    decoder_path: Path,
    corpus_path: Path,
    recipe: TtsQualityRecipe,
    selected_model_repo: str | None = None,
    output_dir: Path = RUNS_DIR,
    speech_max_tokens: int | None = None,
) -> Path:
    """Generate one long-form WAV per corpus case through one warm engine."""

    corpus = load_tts_quality_corpus(corpus_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    previous_window_decode = os.environ.get(CFG_TTS_WINDOW_DECODE_ENV)
    if recipe.sampling.cfg_enabled and recipe.compact_window_decode:
        os.environ[CFG_TTS_WINDOW_DECODE_ENV] = "1"
    session: VllmSpeechToSpeechSession | None = None
    samples: list[dict[str, object]] = []
    try:
        session = VllmSpeechToSpeechSession(
            full_model_path=full_model_path,
            decoder_path=decoder_path,
            selected_model_repo=selected_model_repo,
            output_dir=output_dir,
            speech_max_tokens=speech_max_tokens,
            tts_sampling_config=recipe.sampling,
        )
        for case in corpus.cases:
            artifact_prefix = f"tts-quality-{recipe.recipe_id}-{case.case_id}"
            speech = session.generate_speech_output(
                text=case.text,
                max_tokens=session._speech_max_tokens_for_text(case.text),
                play=False,
                artifact_prefix=artifact_prefix,
                tts_segments=(case.text,),
            )
            samples.append(
                {
                    "case_id": case.case_id,
                    "category": case.category,
                    "text": case.text,
                    "word_count": case.word_count,
                    "required_terms": list(case.required_terms),
                    "wav_path": str(speech.wav_path),
                    "run_log_path": str(speech.run_log_path),
                }
            )
    finally:
        if session is not None:
            session.shutdown()
        if previous_window_decode is None:
            os.environ.pop(CFG_TTS_WINDOW_DECODE_ENV, None)
        else:
            os.environ[CFG_TTS_WINDOW_DECODE_ENV] = previous_window_decode

    manifest = {
        "schema_version": 1,
        "recipe_id": recipe.recipe_id,
        "cfg_enabled": recipe.sampling.cfg_enabled,
        "seed": recipe.sampling.seed,
        "corpus_path": str(corpus_path),
        "sampling": {
            "temperature": recipe.sampling.temperature,
            "top_p": recipe.sampling.top_p,
            "top_k": recipe.sampling.top_k,
            "cfg_scale": recipe.sampling.cfg_scale,
        },
        "controlled_segments_per_case": 1,
        "compact_window_decode": recipe.compact_window_decode,
        "compact_window_decode_required": (
            recipe.sampling.require_compact_window_decode
        ),
        "samples": samples,
    }
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    manifest_path = (
        output_dir / f"tts-quality-{recipe.recipe_id}-{timestamp}.manifest.json"
    )
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return manifest_path


def run_vllm_interactive_ptt(
    *,
    full_model_path: Path,
    decoder_path: Path,
    selected_model_repo: str | None = None,
    output_dir: Path = RUNS_DIR,
    play: bool = True,
    response_max_tokens: int = DEFAULT_RESPONSE_MAX_TOKENS,
    speech_max_tokens: int | None = None,
    thinking_enabled: bool = False,
    conversation: Conversation | None = None,
    conversation_store: ConversationStore | None = None,
    persona: Persona | None = None,
    conversation_resumed: bool = False,
) -> SpeechToSpeechTurnResult | None:
    output_dir.mkdir(parents=True, exist_ok=True)
    print("Audex STS: loading persistent vLLM Metal session...", flush=True)
    if _vllm_tts_cfg_enabled():
        print(
            "Audex STS: CFG TTS enabled with text-to-TTS interleaving.",
            flush=True,
        )
    session = VllmSpeechToSpeechSession(
        full_model_path=full_model_path,
        decoder_path=decoder_path,
        selected_model_repo=selected_model_repo,
        output_dir=output_dir,
        thinking_enabled=thinking_enabled,
        response_max_tokens=response_max_tokens,
        speech_max_tokens=speech_max_tokens,
        conversation=conversation,
        conversation_store=conversation_store,
        persona=persona,
    )
    try:
        stats = session.stats
        print(
            "Audex STS: vLLM session ready "
            f"(model={stats.model_load_seconds:.3f}s, "
            f"audio={stats.audio_component_load_seconds:.3f}s, "
            f"decoder={stats.decoder_load_seconds:.3f}s).",
            flush=True,
        )
        session.speak_startup_greeting(
            conversation_resumed=conversation_resumed,
            play=play,
        )

        last_turn: SpeechToSpeechTurnResult | None = None
        while True:
            turn_input = read_turn_input()
            turn_submitted_at = time.time()
            if turn_input.kind is InputKind.QUIT:
                break
            if turn_input.kind is InputKind.TEXT:
                last_turn = session.run_turn_from_text(
                    user_text=turn_input.text,
                    play=play,
                    turn_submitted_at=turn_submitted_at,
                )
                print(f"Response: {last_turn.response_text}")
                print(f"Speech output WAV: {last_turn.output_wav_path}")
                print(f"Typed-turn run log: {last_turn.run_log_path}")
                continue

            input_wav_path = (
                output_dir / f"ptt-input-{time.strftime('%Y%m%d-%H%M%S')}.wav"
            )
            print("Recording. Press Enter to stop.")
            recording = _start_recording()
            if _preemptive_generation_enabled():
                last_turn = session.run_preemptive_recorded_turn(
                    recording=recording,
                    input_wav_path=input_wav_path,
                    play=play,
                )
            else:
                turn_submitted_at = _wait_for_turn_submission()
                samples = recording.stop()
                write_pcm16_wav(input_wav_path, samples, sample_rate=SAMPLE_RATE)
                last_turn = session.run_turn_from_wav(
                    input_wav_path=input_wav_path,
                    play=play,
                    turn_submitted_at=turn_submitted_at,
                )
            print(f"Captured input WAV: {input_wav_path}")
            print(f"Transcript: {last_turn.transcript}")
            print(f"Response: {last_turn.response_text}")
            print(f"Speech output WAV: {last_turn.output_wav_path}")
            print(f"Speech-to-speech run log: {last_turn.run_log_path}")

        if last_turn is None:
            print("Audex STS: no turns completed.")
        return last_turn
    finally:
        session.shutdown()
