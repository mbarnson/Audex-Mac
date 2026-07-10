#!/usr/bin/env python3
"""Decode Audex speech codec frames with NVIDIA's bundled PyTorch decoder."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from audex_mac.speech_output import write_pcm16_wav


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Decode Audex speech frames with torch."
    )
    parser.add_argument("run_json", type=Path)
    parser.add_argument("--output-wav", type=Path, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--chunk-frames", type=int, default=5)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = json.loads(args.run_json.read_text(encoding="utf-8"))
    frames = payload.get("generated_codec_frames") or []
    if not frames:
        raise RuntimeError(f"No generated_codec_frames in {args.run_json}")
    decoder_path = Path(payload["decoder_path"])

    from transformers import AutoModel

    decoder = (
        AutoModel.from_pretrained(
            str(decoder_path),
            trust_remote_code=True,
        )
        .to(args.device)
        .eval()
    )
    session = decoder.create_session(chunk_frames=args.chunk_frames)
    waveform: list[float] = []
    output_sample_rate = None
    for pushed_sample_rate, chunk in session.push([[int(frame)] for frame in frames]):
        output_sample_rate = pushed_sample_rate
        waveform.extend(_flatten_floats(chunk))
    for flushed_sample_rate, chunk in session.flush():
        output_sample_rate = flushed_sample_rate
        waveform.extend(_flatten_floats(chunk))
    if output_sample_rate is None:
        raise RuntimeError("Audex torch decoder emitted no sample rate.")

    output_wav = args.output_wav
    if output_wav is None:
        output_wav = args.run_json.with_suffix(".torch.wav")
    write_pcm16_wav(output_wav, waveform, sample_rate=int(output_sample_rate))
    print(
        json.dumps(
            {
                "run_json": str(args.run_json),
                "output_wav": str(output_wav),
                "sample_rate": int(output_sample_rate),
                "sample_count": len(waveform),
                "peak_abs": max((abs(sample) for sample in waveform), default=0.0),
            },
            indent=2,
        )
    )
    return 0


def _flatten_floats(value) -> list[float]:
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "reshape"):
        value = value.reshape(-1)
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, list):
        return [float(item) for item in _flatten_list(value)]
    return [float(value)]


def _flatten_list(values: list) -> list:
    flattened = []
    for value in values:
        if isinstance(value, list):
            flattened.extend(_flatten_list(value))
        else:
            flattened.append(value)
    return flattened


if __name__ == "__main__":
    raise SystemExit(main())
