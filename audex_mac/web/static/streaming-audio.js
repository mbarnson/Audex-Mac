"use strict";

const AUDEX_PCM_HEADER_BYTES = 24;
const AUDEX_PLAYBACK_PREBUFFER_SECONDS = 0.08;

function concatFloat32(chunks) {
  const total = chunks.reduce((sum, chunk) => sum + chunk.length, 0);
  const output = new Float32Array(total);
  let offset = 0;
  chunks.forEach((chunk) => {
    output.set(chunk, offset);
    offset += chunk.length;
  });
  return output;
}

function decodePcmFrame(buffer) {
  const bytes = buffer instanceof Uint8Array ? buffer : new Uint8Array(buffer);
  if (bytes.byteLength < AUDEX_PCM_HEADER_BYTES) throw new Error("Audex PCM frame is too short");
  if (String.fromCharCode(...bytes.subarray(0, 4)) !== "APCM") {
    throw new Error("Audex PCM frame marker is invalid");
  }
  const turnId = [...bytes.subarray(4, 20)]
    .map((value) => value.toString(16).padStart(2, "0"))
    .join("");
  const sequence = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength).getUint32(20, true);
  return { turnId, sequence, pcm: bytes.slice(AUDEX_PCM_HEADER_BYTES) };
}

function pcm16ToFloat32(pcm) {
  const bytes = pcm instanceof Uint8Array ? pcm : new Uint8Array(pcm);
  if (bytes.byteLength % 2 !== 0) throw new Error("Audex PCM16 frame has an odd byte count");
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const samples = new Float32Array(bytes.byteLength / 2);
  for (let index = 0; index < samples.length; index += 1) {
    samples[index] = view.getInt16(index * 2, true) / 32768;
  }
  return samples;
}

class StreamingResampler {
  constructor(inputRate, outputRate) {
    if (inputRate <= 0 || outputRate <= 0) throw new Error("Audio sample rates must be positive");
    this.ratio = inputRate / outputRate;
    this.pending = new Float32Array(0);
    this.position = 0;
  }

  push(samples) {
    if (samples.length) this.pending = concatFloat32([this.pending, samples]);
    const output = [];
    while (this.position + 1 < this.pending.length) {
      const left = Math.floor(this.position);
      const fraction = this.position - left;
      output.push(this.pending[left] + (this.pending[left + 1] - this.pending[left]) * fraction);
      this.position += this.ratio;
    }
    const consumed = Math.floor(this.position);
    if (consumed > 0) {
      this.pending = this.pending.slice(consumed);
      this.position -= consumed;
    }
    return Float32Array.from(output);
  }

  finish() {
    if (!this.pending.length) return new Float32Array(0);
    const output = Float32Array.of(this.pending.at(-1));
    this.pending = new Float32Array(0);
    this.position = 0;
    return output;
  }
}

class AudexStreamPlayer {
  constructor({ onState = () => {}, onDiagnostic = () => {} } = {}) {
    this.onState = onState;
    this.onDiagnostic = onDiagnostic;
    this.context = null;
    this.node = null;
    this.ready = null;
    this.turnId = null;
    this.expectedSequence = 0;
    this.resampler = null;
    this.queuedFrames = 0;
    this.started = false;
    this.drainTimer = null;
  }

  prime() {
    if (!this.context) {
      const AudioContextClass = window.AudioContext || window.webkitAudioContext;
      this.context = new AudioContextClass({ latencyHint: "interactive" });
      void this.context.resume().catch(() => {});
      this.ready = this.context.audioWorklet.addModule("/assets/pcm-player-worklet.js").then(() => {
        this.node = new AudioWorkletNode(this.context, "audex-pcm-player");
        this.node.port.onmessage = (event) => this._workletMessage(event.data);
        this.node.connect(this.context.destination);
      });
    } else if (this.context.state === "suspended") {
      void this.context.resume();
    }
    return this.ready;
  }

  async begin({ turnId, sampleRate }) {
    await this.prime();
    clearTimeout(this.drainTimer);
    this.drainTimer = null;
    this.turnId = turnId;
    this.expectedSequence = 0;
    this.resampler = new StreamingResampler(sampleRate, this.context.sampleRate);
    this.queuedFrames = 0;
    this.started = false;
    this.node.port.postMessage({ type: "reset" });
    this.onState("buffering");
  }

  pushFrame(buffer) {
    if (!this.turnId || !this.resampler) throw new Error("Audex audio arrived before stream start");
    const frame = decodePcmFrame(buffer);
    if (frame.turnId !== this.turnId) throw new Error("Audex audio belongs to an obsolete turn");
    if (frame.sequence !== this.expectedSequence) {
      throw new Error(`Audex audio sequence gap: expected ${this.expectedSequence}, received ${frame.sequence}`);
    }
    this.expectedSequence += 1;
    this._enqueue(this.resampler.push(pcm16ToFloat32(frame.pcm)));
  }

  finish() {
    if (!this.node || !this.resampler) return;
    this._enqueue(this.resampler.finish());
    if (!this.started) this._start();
    this.node.port.postMessage({ type: "finish" });
    const maximumPlaybackMs = Math.ceil((this.queuedFrames / this.context.sampleRate) * 1000) + 1000;
    this.drainTimer = setTimeout(() => {
      this.onDiagnostic("drain-timeout");
      this._markDrained();
    }, maximumPlaybackMs);
  }

  isActive() {
    return this.turnId !== null && this.started;
  }

  stop() {
    clearTimeout(this.drainTimer);
    this.drainTimer = null;
    if (this.node) this.node.port.postMessage({ type: "reset" });
    this.turnId = null;
    this.resampler = null;
    this.queuedFrames = 0;
    this.started = false;
    this.onState("idle");
  }

  _enqueue(samples) {
    if (!samples.length) return;
    this.queuedFrames += samples.length;
    this.node.port.postMessage({ type: "enqueue", samples }, [samples.buffer]);
    if (!this.started && this.queuedFrames >= this.context.sampleRate * AUDEX_PLAYBACK_PREBUFFER_SECONDS) {
      this._start();
    }
  }

  _start() {
    this.started = true;
    this.node.port.postMessage({ type: "start" });
  }

  _workletMessage(message) {
    if (message.type === "started") this.onState("playing");
    if (message.type === "drained") this._markDrained();
    if (message.type === "underrun") this.onDiagnostic("underrun");
  }

  _markDrained() {
    if (this.turnId === null) return;
    clearTimeout(this.drainTimer);
    this.drainTimer = null;
    this.turnId = null;
    this.resampler = null;
    this.queuedFrames = 0;
    this.started = false;
    this.onState("drained");
  }
}

class AudexLiveTurnClient {
  constructor({ url, player, onEvent = () => {}, socketFactory = null }) {
    this.url = url;
    this.player = player;
    this.onEvent = onEvent;
    this.socketFactory = socketFactory || ((socketUrl) => new WebSocket(socketUrl));
    this.socket = null;
    this.turnId = null;
    this.settle = null;
  }

  async run(request) {
    if (this.socket) throw new Error("An Audex live turn is already running");
    await this.player.prime();
    return new Promise((resolve, reject) => {
      const socket = this.socketFactory(this.url);
      this.socket = socket;
      this.settle = { resolve, reject };
      socket.binaryType = "arraybuffer";
      socket.onopen = () => socket.send(JSON.stringify(request));
      socket.onmessage = (event) => {
        void this._message(event.data).catch((error) => this._fail(error));
      };
      socket.onerror = () => this._fail(new Error("Audex live speech connection failed"));
      socket.onclose = () => {
        if (this.settle) this._fail(new Error("Audex live speech connection closed before completion"));
      };
    });
  }

  stop() {
    this.player.stop();
    if (this.socket) this.socket.close(1000, "Stopped by user");
    this._clear();
  }

  async _message(data) {
    if (data instanceof ArrayBuffer) {
      this.player.pushFrame(data);
      return;
    }
    const event = JSON.parse(data);
    this.onEvent(event);
    if (event.type === "turn.started") this.turnId = event.turn_id;
    if (this.turnId && event.turn_id !== this.turnId) {
      throw new Error("Audex live event belongs to an obsolete turn");
    }
    if (event.type === "assistant.audio.started") {
      await this.player.begin({ turnId: event.turn_id, sampleRate: event.sample_rate });
    } else if (event.type === "assistant.audio.finished") {
      this.player.finish();
    } else if (event.type === "turn.failed") {
      throw new Error(event.error || "Audex live turn failed");
    } else if (event.type === "turn.finished") {
      const resolve = this.settle.resolve;
      this._clear();
      resolve(event.turn);
    }
  }

  _fail(error) {
    if (!this.settle) return;
    const reject = this.settle.reject;
    this.player.stop();
    this._clear();
    reject(error);
  }

  _clear() {
    this.socket = null;
    this.turnId = null;
    this.settle = null;
  }
}

if (typeof window !== "undefined") {
  window.AudexLiveTurnClient = AudexLiveTurnClient;
  window.AudexStreamPlayer = AudexStreamPlayer;
}
if (typeof module !== "undefined") {
  module.exports = {
    AudexLiveTurnClient,
    AudexStreamPlayer,
    StreamingResampler,
    concatFloat32,
    decodePcmFrame,
    pcm16ToFloat32,
  };
}
