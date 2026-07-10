#!/usr/bin/env python3
"""Package TTS quality manifests into an opaque human-listening set."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path, nargs="+")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--key-out", type=Path, required=True)
    parser.add_argument("--random-seed", type=int, default=8675309)
    return parser.parse_args()


def main() -> int:
    from audex_mac.tts_quality import create_blind_listening_set

    args = parse_args()
    listening_set = create_blind_listening_set(
        manifest_paths=tuple(args.manifest),
        output_dir=args.output_dir,
        key_path=args.key_out,
        random_seed=args.random_seed,
    )
    print(f"Blind listening sheet: {listening_set.listening_path}")
    print(f"Blind sample count: {len(listening_set.sample_paths)}")
    print(f"Private decoding key: {listening_set.key_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
