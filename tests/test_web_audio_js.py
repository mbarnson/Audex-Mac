from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.fast


def test_browser_audio_module_downsamples_and_writes_pcm16_wav() -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is unavailable for the browser audio contract test")
    module = Path("audex_mac/web/static/audio.js").resolve()
    script = f"""
const audio = require({json.dumps(str(module))});
(async () => {{
  const merged = audio.mergeSamples([
    new Float32Array([1, 0]),
    new Float32Array([-1, 0]),
  ]);
  const downsampled = audio.downsample(merged, 32000, 16000);
  const wav = audio.encodeWav(downsampled, 16000);
  const bytes = new Uint8Array(await wav.arrayBuffer());
  console.log(JSON.stringify({{
    merged: Array.from(merged),
    downsampled: Array.from(downsampled),
    size: bytes.length,
    riff: String.fromCharCode(...bytes.slice(0, 4)),
    wave: String.fromCharCode(...bytes.slice(8, 12)),
    sampleRate: new DataView(bytes.buffer).getUint32(24, true),
    channels: new DataView(bytes.buffer).getUint16(22, true),
    base64: await audio.blobToBase64(wav),
  }}));
}})();
"""

    result = subprocess.run(
        [node, "-e", script],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)

    assert payload["merged"] == [1, 0, -1, 0]
    assert payload["downsampled"] == [0.5, -0.5]
    assert payload["size"] == 48
    assert payload["riff"] == "RIFF"
    assert payload["wave"] == "WAVE"
    assert payload["sampleRate"] == 16_000
    assert payload["channels"] == 1
    assert payload["base64"].startswith("UklGR")
