"use strict";

class AudexPcmPlayerProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.queue = [];
    this.offset = 0;
    this.playing = false;
    this.finished = false;
    this.underrunReported = false;
    this.port.onmessage = (event) => this.handle(event.data);
  }

  handle(message) {
    if (message.type === "reset") {
      this.queue = [];
      this.offset = 0;
      this.playing = false;
      this.finished = false;
      this.underrunReported = false;
    } else if (message.type === "enqueue") {
      this.queue.push(message.samples);
      this.underrunReported = false;
    } else if (message.type === "start") {
      this.playing = true;
      this.port.postMessage({ type: "started" });
    } else if (message.type === "finish") {
      this.finished = true;
    }
  }

  process(_inputs, outputs) {
    const channel = outputs[0][0];
    channel.fill(0);
    if (!this.playing) return true;
    let outputOffset = 0;
    while (outputOffset < channel.length && this.queue.length) {
      const current = this.queue[0];
      const count = Math.min(channel.length - outputOffset, current.length - this.offset);
      channel.set(current.subarray(this.offset, this.offset + count), outputOffset);
      outputOffset += count;
      this.offset += count;
      if (this.offset >= current.length) {
        this.queue.shift();
        this.offset = 0;
      }
    }
    if (!this.queue.length) {
      if (this.finished) {
        this.playing = false;
        this.port.postMessage({ type: "drained" });
      } else if (!this.underrunReported) {
        this.underrunReported = true;
        this.port.postMessage({ type: "underrun" });
      }
    }
    return true;
  }
}

registerProcessor("audex-pcm-player", AudexPcmPlayerProcessor);

