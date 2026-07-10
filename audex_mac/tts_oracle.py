"""Optional MLX-Audio transcription oracle for TTS quality evaluation."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any


class MlxAudioTranscriber:
    """Load one MLX-Audio STT model and reuse it across WAV evaluations."""

    def __init__(self, model_name: str) -> None:
        self._model_name = model_name
        self._model: Any | None = None

    def load(self) -> float:
        """Load the configured model and return the elapsed time in seconds."""

        from mlx_audio.stt.utils import load

        started = time.perf_counter()
        self._model = load(self._model_name)
        return time.perf_counter() - started

    def transcribe_file(self, wav_path: Path) -> dict[str, object]:
        """Transcribe a WAV and normalize MLX-Audio's result shape."""

        if self._model is None:
            raise RuntimeError("load the MLX-Audio STT model before transcribing")
        started = time.perf_counter()
        result = self._model.generate(str(wav_path))
        elapsed = time.perf_counter() - started
        segments: list[dict[str, object]] = []
        text_parts: list[str] = []
        sentences = getattr(result, "sentences", None)
        if sentences:
            for sentence in sentences:
                text = str(sentence.text).strip()
                segments.append(
                    {
                        "start": getattr(sentence, "start", 0.0),
                        "end": getattr(sentence, "end", 0.0),
                        "text": text,
                    }
                )
                text_parts.append(text)
        elif hasattr(result, "text"):
            text = str(result.text).strip()
            segments.append({"start": 0.0, "end": 0.0, "text": text})
            text_parts.append(text)
        return {
            "text": " ".join(text_parts),
            "segments": segments,
            "elapsed": elapsed,
        }
