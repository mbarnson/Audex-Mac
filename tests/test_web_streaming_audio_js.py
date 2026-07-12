from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.fast


def test_browser_streaming_audio_decodes_frames_and_resamples_continuously() -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is unavailable for the browser streaming audio test")
    module = Path("audex_mac/web/static/streaming-audio.js").resolve()
    script = f"""
const audio = require({json.dumps(str(module))});
const turnId = "00112233445566778899aabbccddeeff";
const header = new ArrayBuffer(28);
const view = new DataView(header);
[..."APCM"].forEach((char, index) => view.setUint8(index, char.charCodeAt(0)));
for (let index = 0; index < 16; index += 1) view.setUint8(4 + index, parseInt(turnId.slice(index * 2, index * 2 + 2), 16));
view.setUint32(20, 7, true);
view.setInt16(24, 32767, true);
view.setInt16(26, -32768, true);
const decoded = audio.decodePcmFrame(header);
const resampler = new audio.StreamingResampler(2, 4);
const first = resampler.push(new Float32Array([0, 1]));
const second = resampler.push(new Float32Array([0, -1]));
const tail = resampler.finish();
console.log(JSON.stringify({{
  turnId: decoded.turnId,
  sequence: decoded.sequence,
  samples: Array.from(audio.pcm16ToFloat32(decoded.pcm)),
  resampled: Array.from(audio.concatFloat32([first, second, tail])),
}}));
"""

    result = subprocess.run(
        [node, "-e", script],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)

    assert payload["turnId"] == "00112233445566778899aabbccddeeff"
    assert payload["sequence"] == 7
    assert payload["samples"] == [32767 / 32768, -1]
    assert payload["resampled"] == pytest.approx(
        [0, 0.5, 1, 0.5, 0, -0.5, -1],
        abs=1e-6,
    )


def test_live_turn_client_autoplays_pcm_and_resolves_completed_turn() -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is unavailable for the browser live-turn client test")
    module = Path("audex_mac/web/static/streaming-audio.js").resolve()
    script = f"""
const audio = require({json.dumps(str(module))});
class FakeSocket {{
  constructor(url) {{ this.url = url; this.sent = []; FakeSocket.instance = this; }}
  send(value) {{ this.sent.push(value); }}
  open() {{ this.onopen(); }}
  message(data) {{ this.onmessage({{ data }}); }}
}}
const calls = [];
const player = {{
  prime: async () => calls.push(["prime"]),
  begin: async (value) => calls.push(["begin", value.turnId, value.sampleRate]),
  pushFrame: (value) => calls.push(["pcm", value.byteLength]),
  finish: () => calls.push(["finish"]),
  stop: () => calls.push(["stop"]),
}};
(async () => {{
  const client = new audio.AudexLiveTurnClient({{
    url: "ws://localhost/live",
    player,
    socketFactory: (url) => new FakeSocket(url),
    onEvent: (event) => calls.push(["event", event.type]),
  }});
  const pending = client.run({{ chat_id: "chat-1", mode: "text-speech", text: "Hi" }});
  await Promise.resolve();
  const socket = FakeSocket.instance;
  socket.open();
  socket.message(JSON.stringify({{ type: "turn.started", turn_id: "00112233445566778899aabbccddeeff" }}));
  socket.message(JSON.stringify({{ type: "assistant.audio.started", turn_id: "00112233445566778899aabbccddeeff", sample_rate: 24000 }}));
  await Promise.resolve();
  socket.message(new ArrayBuffer(28));
  socket.message(JSON.stringify({{ type: "assistant.audio.finished", turn_id: "00112233445566778899aabbccddeeff" }}));
  socket.message(JSON.stringify({{ type: "turn.finished", turn_id: "00112233445566778899aabbccddeeff", turn: {{ chat: {{ id: "chat-1" }} }} }}));
  const turn = await pending;
  console.log(JSON.stringify({{ sent: JSON.parse(socket.sent[0]), calls, turn }}));
}})();
"""

    result = subprocess.run(
        [node, "-e", script],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)

    assert payload["sent"] == {
        "chat_id": "chat-1",
        "mode": "text-speech",
        "text": "Hi",
    }
    assert payload["calls"] == [
        ["prime"],
        ["event", "turn.started"],
        ["event", "assistant.audio.started"],
        ["begin", "00112233445566778899aabbccddeeff", 24_000],
        ["pcm", 28],
        ["event", "assistant.audio.finished"],
        ["finish"],
        ["event", "turn.finished"],
    ]
    assert payload["turn"] == {"chat": {"id": "chat-1"}}


def test_stream_player_remains_active_until_the_worklet_drains() -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is unavailable for the browser stream-player test")
    module = Path("audex_mac/web/static/streaming-audio.js").resolve()
    script = f"""
const audio = require({json.dumps(str(module))});
const states = [];
const player = Object.create(audio.AudexStreamPlayer.prototype);
player.onState = (state) => states.push(state);
player.onDiagnostic = () => {{}};
player.turnId = "00112233445566778899aabbccddeeff";
player.resampler = {{}};
player.queuedFrames = 1200;
player.started = true;
player.drainTimer = null;
const activeBeforeDrain = player.isActive();
player._workletMessage({{ type: "drained" }});
console.log(JSON.stringify({{
  activeBeforeDrain,
  activeAfterDrain: player.isActive(),
  turnId: player.turnId,
  states,
}}));
"""

    result = subprocess.run(
        [node, "-e", script],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {
        "activeBeforeDrain": True,
        "activeAfterDrain": False,
        "turnId": None,
        "states": ["drained"],
    }
