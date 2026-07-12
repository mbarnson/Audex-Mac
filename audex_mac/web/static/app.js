"use strict";

const state = {
  modes: [],
  chats: [],
  chat: null,
  mode: "text-text",
  audio: null,
  busy: false,
  recorder: null,
  recordTimer: null,
  recordingEpoch: 0,
  liveTurnUrl: null,
  liveClient: null,
  streamPlayer: null,
  liveUserMessageId: null,
  liveAssistantMessageId: null,
};

const $ = (selector) => document.querySelector(selector);
const elements = {
  sidebar: $("#sidebar"),
  conversationPanel: document.querySelector(".conversation-panel"),
  chatList: $("#chat-list"),
  chatCount: $("#chat-count"),
  newChat: $("#new-chat"),
  title: $("#chat-title"),
  modeSubtitle: $("#mode-subtitle"),
  modeGlyph: $("#active-mode-glyph"),
  cache: $("#cache-indicator"),
  dock: $("#mode-dock"),
  messages: $("#messages"),
  scroll: $("#message-scroll"),
  thinking: $("#thinking"),
  thinkingLabel: $("#thinking-label"),
  composer: $("#composer"),
  input: $("#message-input"),
  file: $("#audio-file"),
  attach: $("#attach-button"),
  record: $("#record-button"),
  recordTime: $("#record-time"),
  send: $("#send-button"),
  hint: $("#composer-hint"),
  audioPreview: $("#audio-preview"),
  audioName: $("#audio-name"),
  audioDuration: $("#audio-duration"),
  clearAudio: $("#clear-audio"),
  mobileMenu: $("#mobile-menu"),
  sidebarClose: $("#sidebar-close"),
  sidebarBackdrop: $("#sidebar-backdrop"),
  toasts: $("#toast-stack"),
};

const narrowLayout = window.matchMedia("(max-width: 720px)");

const modeGlyphs = {
  "text-text": "Aa",
  "text-speech": "◖))",
  "speech-text": "⌁A",
  "speech-speech": "≋",
  "audio-text": "⌁?",
  "text-audio": "✦",
  "audio-audio": "∞",
};

const modeShortLabels = {
  "text-text": "T→T",
  "text-speech": "T→S",
  "speech-text": "S→T",
  "speech-speech": "S→S",
  "audio-text": "A→T",
  "text-audio": "T→A",
  "audio-audio": "A→A",
};

async function api(path, options = {}) {
  const response = await fetch(path, {
    ...options,
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
  });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(payload.error || `Audex request failed (${response.status})`);
  return payload;
}

async function bootstrap() {
  try {
    const payload = await api("/api/bootstrap");
    state.modes = payload.modes;
    state.chats = payload.chats;
    configureLiveTurns(payload.live_turn_url);
    renderModeDock();
    if (state.chats.length === 0) {
      await createChat();
    } else {
      await openChat([...state.chats].sort(sortUpdated)[0].id);
    }
  } catch (error) {
    toast(error.message);
  }
}

function sortUpdated(a, b) {
  return b.updated_at.localeCompare(a.updated_at);
}

async function createChat() {
  if (state.busy) return;
  cancelRecording();
  try {
    const payload = await api("/api/chats", {
      method: "POST",
      body: JSON.stringify({ mode: state.mode }),
    });
    state.chats.push(payload.chat);
    await openChat(payload.chat.id, payload.chat);
    elements.input.focus();
  } catch (error) {
    toast(error.message);
  }
}

async function openChat(chatId, provided = null) {
  if (state.busy) return;
  cancelRecording();
  clearAudio();
  const drawerWasOpen = elements.sidebar.classList.contains("open");
  try {
    state.chat = provided || (await api(`/api/chats/${chatId}`)).chat;
    state.mode = state.chat.current_mode;
    renderAll();
    setSidebarOpen(false, { restoreFocus: drawerWasOpen });
    requestAnimationFrame(scrollToBottom);
  } catch (error) {
    toast(error.message);
  }
}

function renderAll() {
  renderChatList();
  renderHeader();
  renderModeDock();
  renderMessages();
  updateComposer();
}

function renderChatList() {
  elements.chatCount.textContent = state.chats.length;
  elements.chatList.replaceChildren(
    ...[...state.chats].sort(sortUpdated).map((chat) => {
      const button = document.createElement("button");
      button.className = `chat-list-item${state.chat?.id === chat.id ? " active" : ""}`;
      button.innerHTML = `
        <span class="history-mode-icon">${escapeHtml(modeGlyphs[chat.current_mode] || "✦")}</span>
        <span class="chat-copy"><strong>${escapeHtml(chat.title)}</strong><small>${escapeHtml(chatPreview(chat))}</small></span>
        <span class="chat-time">${relativeTime(chat.updated_at)}</span>`;
      button.addEventListener("click", () => openChat(chat.id));
      return button;
    }),
  );
}

function chatPreview(chat) {
  const last = chat.messages?.at(-1);
  return last?.transcript || modeById(chat.current_mode)?.label || "New conversation";
}

function renderHeader() {
  if (!state.chat) return;
  const mode = currentMode();
  elements.title.value = state.chat.title;
  elements.modeSubtitle.textContent = mode.label;
  elements.modeGlyph.textContent = modeGlyphs[mode.id] || "✦";
  elements.cache.classList.toggle("cold", !mode.preserves_conversation_cache);
  elements.cache.querySelector("span").textContent = mode.preserves_conversation_cache
    ? "Cache warm"
    : "Generation mode";
  elements.cache.title = mode.preserves_conversation_cache
    ? "This mode keeps appending to the conversation KV cache"
    : "Sound generation runs outside the conversational KV cache";
}

function renderModeDock() {
  elements.dock.replaceChildren(
    ...state.modes.map((mode) => {
      const button = document.createElement("button");
      button.className = `mode-button${state.mode === mode.id ? " active" : ""}`;
      button.dataset.tooltip = mode.description;
      button.title = mode.description;
      button.setAttribute("aria-pressed", state.mode === mode.id ? "true" : "false");
      button.setAttribute("aria-label", `${mode.label}. ${mode.description}`);
      button.innerHTML = `<span class="mode-icon">${escapeHtml(modeGlyphs[mode.id] || "✦")}</span><span class="mode-label">${escapeHtml(mode.label)}</span><span class="mode-short">${escapeHtml(modeShortLabels[mode.id])}</span>`;
      button.addEventListener("click", () => selectMode(mode.id));
      return button;
    }),
  );
}

function selectMode(modeId) {
  if (state.busy || state.mode === modeId) return;
  cancelRecording();
  state.mode = modeId;
  clearAudio();
  renderHeader();
  renderModeDock();
  updateComposer();
  if (currentMode().input_kind === "text") elements.input.focus();
}

function renderMessages() {
  const messages = state.chat?.messages || [];
  if (messages.length === 0) {
    elements.messages.innerHTML = `
      <div class="empty-state"><div class="empty-inner">
        <span class="audex-avatar" aria-hidden="true">A</span>
        <h1>Speak, listen, imagine.</h1>
        <p>One continuous Audex conversation across text and speech — plus a studio for understanding and generating sound.</p>
        <div class="starter-grid">
          <button class="starter" data-mode="speech-speech">Have a natural spoken conversation</button>
          <button class="starter" data-mode="text-speech">Read my typed prompt aloud</button>
          <button class="starter" data-mode="audio-text">Understand an audio recording</button>
          <button class="starter" data-mode="text-audio">Generate a sound from a description</button>
        </div>
      </div></div>`;
    elements.messages.querySelectorAll(".starter").forEach((button) => {
      button.addEventListener("click", () => selectMode(button.dataset.mode));
    });
    return;
  }
  elements.messages.replaceChildren(...messages.map(messageElement));
}

function messageElement(message) {
  const group = document.createElement("article");
  group.className = `message-group ${message.role}${(message.assets || []).length ? " has-assets" : ""}`;
  const mode = modeById(message.mode);
  const speaker = message.role === "user" ? "You" : "Audex";
  const transcriptLabel = message.role === "user" && mode.input_kind === "speech"
    ? "Transcript"
    : message.role === "user" && mode.input_kind === "audio"
      ? "Audio caption & prompt"
      : message.role === "assistant" && mode.output_kind === "speech"
        ? "Spoken transcript"
        : message.role === "assistant" && mode.output_kind === "audio"
          ? "Generation summary"
          : "";
  const audioLabel = `${speaker} ${mode.label} audio`;
  const audio = message.audio_url
    ? `<audio class="message-audio" controls preload="metadata" aria-label="${escapeHtml(audioLabel)}" src="${escapeHtml(message.audio_url)}"></audio>`
    : "";
  const assets = (message.assets || []).length
    ? `<div class="sound-assets">${message.assets.map(soundCard).join("")}</div>`
    : "";
  group.innerHTML = `
    <div class="message-meta"><span>${speaker}</span><span>·</span><span>${escapeHtml(mode.label)}</span><span>·</span><time>${formatTime(message.created_at)}</time></div>
    <div class="message-bubble">
      ${transcriptLabel ? `<span class="transcript-label">${transcriptLabel}</span>` : ""}
      <p>${escapeHtml(message.transcript)}</p>${audio}${assets}
    </div>`;
  return group;
}

function soundCard(asset, index) {
  const label = asset.label || `Sound ${index + 1}`;
  return `<article class="sound-card">
    <strong>${escapeHtml(label)}</strong>
    ${asset.caption ? `<p>${escapeHtml(asset.caption)}</p>` : ""}
    ${asset.audio_url ? `<audio controls preload="metadata" aria-label="Play ${escapeHtml(label)}" src="${escapeHtml(asset.audio_url)}"></audio>` : ""}
  </article>`;
}

function updateComposer() {
  const mode = currentMode();
  const showText = mode.input_kind !== "speech";
  elements.input.classList.toggle("hidden", !showText);
  elements.record.classList.toggle("hidden", mode.input_kind === "text");
  elements.attach.classList.toggle("hidden", mode.input_kind !== "audio");
  elements.input.placeholder = mode.input_kind === "audio"
    ? mode.output_kind === "text"
      ? "Ask about this audio (optional)…"
      : "Add generation direction (optional)…"
    : mode.output_kind === "audio"
      ? "Describe the sound you want to create…"
      : "Message Audex…";
  elements.hint.textContent = mode.input_kind === "text"
    ? "Enter to send · Shift Enter for a new line"
    : mode.input_kind === "speech"
      ? "Tap the microphone or press Enter, speak, then press Enter to send"
      : "Choose a file or record audio; the written prompt is optional";
  updateSendState();
}

function updateSendState() {
  if (!state.modes.length) return;
  const mode = currentMode();
  const ready = mode.input_kind === "text" ? elements.input.value.trim() : state.audio;
  elements.send.disabled = state.busy || !ready;
}

async function submitTurn(event) {
  event.preventDefault();
  if (state.busy || !state.chat) return;
  const mode = currentMode();
  const text = elements.input.value.trim();
  if (mode.input_kind === "text" && !text) return;
  if (mode.input_kind !== "text" && !state.audio) {
    toast("Record or choose an audio file first.");
    return;
  }
  const liveSpeech = mode.output_kind === "speech" && state.liveClient;
  if (liveSpeech) void state.streamPlayer.prime();
  setBusy(true);
  const optimistic = {
    message_id: `pending-${Date.now()}`,
    role: "user",
    transcript: text || (mode.input_kind === "speech" ? "Transcribing your speech…" : "Understanding your audio…"),
    mode: state.mode,
    created_at: new Date().toISOString(),
    audio_url: state.audio?.url || null,
    assets: [],
  };
  state.chat.messages.push(optimistic);
  const optimisticAssistant = liveSpeech ? {
    message_id: `pending-assistant-${Date.now()}`,
    role: "assistant",
    transcript: "Audex is preparing to speak…",
    mode: state.mode,
    created_at: new Date().toISOString(),
    audio_url: null,
    assets: [],
  } : null;
  if (optimisticAssistant) {
    state.liveUserMessageId = optimistic.message_id;
    state.liveAssistantMessageId = optimisticAssistant.message_id;
    state.chat.messages.push(optimisticAssistant);
  }
  renderMessages();
  elements.thinkingLabel.textContent = mode.output_kind === "audio" ? "Audex is shaping sound" : "Audex is thinking";
  elements.thinking.classList.remove("hidden");
  scrollToBottom();
  try {
    const request = { mode: state.mode };
    if (text) request.text = text;
    if (mode.input_kind !== "text") {
      request.audio = { name: state.audio.name, base64: await blobToBase64(state.audio.blob) };
    }
    const payload = liveSpeech
      ? await state.liveClient.run({ chat_id: state.chat.id, ...request })
      : await api(`/api/chats/${state.chat.id}/turns`, {
        method: "POST",
        body: JSON.stringify(request),
      });
    state.chat = payload.chat;
    upsertChat(payload.chat);
    elements.input.value = "";
    autoSizeInput();
    clearAudio();
    renderAll();
  } catch (error) {
    state.chat.messages = state.chat.messages.filter((message) => ![
      optimistic.message_id,
      optimisticAssistant?.message_id,
    ].includes(message.message_id));
    renderMessages();
    toast(error.message);
  } finally {
    state.liveUserMessageId = null;
    state.liveAssistantMessageId = null;
    if (!liveSpeech || !state.streamPlayer.isActive()) {
      elements.thinking.classList.add("hidden");
      setBusy(false);
    }
    scrollToBottom();
  }
}

function configureLiveTurns(url) {
  state.liveTurnUrl = url || null;
  if (!state.liveTurnUrl || typeof AudexStreamPlayer === "undefined") return;
  state.streamPlayer = new AudexStreamPlayer({
    onState: (playbackState) => {
      if (playbackState === "playing") {
        elements.thinkingLabel.textContent = "Audex is speaking";
        elements.thinking.classList.remove("hidden");
      } else if (playbackState === "drained") {
        elements.thinking.classList.add("hidden");
        setBusy(false);
      }
    },
    onDiagnostic: (diagnostic) => console.warn(`Audex playback: ${diagnostic}`),
  });
  state.liveClient = new AudexLiveTurnClient({
    url: state.liveTurnUrl,
    player: state.streamPlayer,
    onEvent: handleLiveTurnEvent,
  });
}

function handleLiveTurnEvent(event) {
  if (event.type === "assistant.text.delta") {
    const message = state.chat?.messages.find((item) => item.message_id === state.liveAssistantMessageId);
    if (message) {
      message.transcript = event.text || "Audex is speaking…";
      renderMessages();
      scrollToBottom();
    }
  } else if (event.type === "user.transcript.final") {
    const message = state.chat?.messages.find((item) => item.message_id === state.liveUserMessageId);
    if (message) {
      message.transcript = event.text;
      renderMessages();
      scrollToBottom();
    }
  } else if (event.type === "assistant.audio.started") {
    elements.thinkingLabel.textContent = "Audex is buffering speech";
  }
}

function setBusy(busy) {
  state.busy = busy;
  elements.input.disabled = busy;
  elements.title.disabled = busy;
  elements.record.disabled = busy;
  elements.file.disabled = busy;
  updateSendState();
}

function upsertChat(chat) {
  const index = state.chats.findIndex((item) => item.id === chat.id);
  if (index === -1) state.chats.push(chat);
  else state.chats[index] = chat;
}

async function renameChat() {
  if (!state.chat) return;
  const title = elements.title.value.trim();
  if (!title || title === state.chat.title) {
    elements.title.value = state.chat.title;
    return;
  }
  try {
    const payload = await api(`/api/chats/${state.chat.id}`, {
      method: "PATCH",
      body: JSON.stringify({ title }),
    });
    state.chat = payload.chat;
    upsertChat(payload.chat);
    renderChatList();
  } catch (error) {
    elements.title.value = state.chat.title;
    toast(error.message);
  }
}

async function chooseAudio(event) {
  const file = event.target.files?.[0];
  if (!file) return;
  try {
    const converted = await audioFileToWav(file);
    setAudio(converted, file.name.replace(/\.[^.]+$/, "") + ".wav");
  } catch (error) {
    toast(`Could not prepare that audio: ${error.message}`);
  } finally {
    event.target.value = "";
  }
}

function setAudio(blob, name) {
  clearAudio();
  const url = URL.createObjectURL(blob);
  state.audio = { blob, name, url };
  elements.audioName.textContent = name;
  elements.audioDuration.textContent = "16 kHz WAV · ready to send";
  elements.audioPreview.classList.remove("hidden");
  updateSendState();
}

function clearAudio() {
  if (state.audio?.url) URL.revokeObjectURL(state.audio.url);
  state.audio = null;
  elements.audioPreview.classList.add("hidden");
  updateSendState();
}

async function toggleRecording() {
  if (state.recorder) {
    const recording = state.recorder;
    const chatId = state.chat?.id;
    const modeId = state.mode;
    state.recorder = null;
    const completionEpoch = ++state.recordingEpoch;
    resetRecordingUi();
    const blob = await recording.stop();
    if (
      completionEpoch !== state.recordingEpoch
      || state.chat?.id !== chatId
      || state.mode !== modeId
    ) return;
    setAudio(blob, `audex-recording-${Date.now()}.wav`);
    if (currentMode().input_kind === "speech") {
      elements.composer.requestSubmit();
    }
    return;
  }
  const requestEpoch = ++state.recordingEpoch;
  if (currentMode().output_kind === "speech") void state.streamPlayer?.prime();
  try {
    const recorder = await createWavRecorder();
    if (requestEpoch !== state.recordingEpoch) {
      await recorder.cancel();
      return;
    }
    state.recorder = recorder;
    recorder.start();
    elements.record.classList.add("recording");
    elements.record.setAttribute("aria-label", "Stop recording");
    elements.record.setAttribute("aria-pressed", "true");
    elements.record.dataset.tooltip = "Stop recording";
    const started = Date.now();
    state.recordTimer = setInterval(() => {
      const elapsed = Math.floor((Date.now() - started) / 1000);
      elements.recordTime.textContent = `${Math.floor(elapsed / 60)}:${String(elapsed % 60).padStart(2, "0")}`;
    }, 250);
  } catch (error) {
    if (requestEpoch !== state.recordingEpoch) return;
    state.recorder = null;
    resetRecordingUi();
    toast(`Microphone unavailable: ${error.message}`);
  }
}

function resetRecordingUi() {
  clearInterval(state.recordTimer);
  state.recordTimer = null;
  elements.record.classList.remove("recording");
  elements.record.setAttribute("aria-label", "Record speech");
  elements.record.setAttribute("aria-pressed", "false");
  elements.record.dataset.tooltip = "Record speech";
  elements.recordTime.textContent = "0:00";
}

function cancelRecording() {
  state.recordingEpoch += 1;
  const recording = state.recorder;
  state.recorder = null;
  resetRecordingUi();
  if (recording) void recording.cancel().catch(() => {});
}

async function createWavRecorder() {
  const stream = await navigator.mediaDevices.getUserMedia({
    audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true },
  });
  const context = new AudioContext();
  const source = context.createMediaStreamSource(stream);
  const processor = context.createScriptProcessor(4096, 1, 1);
  const chunks = [];
  let closed = false;
  processor.onaudioprocess = (event) => chunks.push(new Float32Array(event.inputBuffer.getChannelData(0)));
  async function close() {
    if (closed) return;
    closed = true;
    processor.onaudioprocess = null;
    try { processor.disconnect(); } catch (_error) { /* already disconnected */ }
    try { source.disconnect(); } catch (_error) { /* already disconnected */ }
    stream.getTracks().forEach((track) => track.stop());
    if (context.state !== "closed") await context.close();
  }
  return {
    start() {
      source.connect(processor);
      processor.connect(context.destination);
    },
    async stop() {
      const samples = mergeSamples(chunks);
      const downsampled = downsample(samples, context.sampleRate, 16000);
      await close();
      return encodeWav(downsampled, 16000);
    },
    cancel: close,
  };
}

async function audioFileToWav(file) {
  const context = new AudioContext();
  try {
    const decoded = await context.decodeAudioData(await file.arrayBuffer());
    const mono = new Float32Array(decoded.length);
    for (let channel = 0; channel < decoded.numberOfChannels; channel += 1) {
      const input = decoded.getChannelData(channel);
      for (let index = 0; index < input.length; index += 1) mono[index] += input[index] / decoded.numberOfChannels;
    }
    return encodeWav(downsample(mono, decoded.sampleRate, 16000), 16000);
  } finally {
    await context.close();
  }
}

function currentMode() {
  return modeById(state.mode) || state.modes[0];
}

function modeById(id) {
  return state.modes.find((mode) => mode.id === id);
}

function autoSizeInput() {
  elements.input.style.height = "auto";
  elements.input.style.height = `${Math.min(elements.input.scrollHeight, 130)}px`;
  updateSendState();
}

function scrollToBottom() {
  elements.scroll.scrollTop = elements.scroll.scrollHeight;
}

function toast(message) {
  const item = document.createElement("div");
  item.className = "toast";
  item.setAttribute("role", "alert");
  item.textContent = message;
  elements.toasts.append(item);
  setTimeout(() => item.remove(), 6000);
}

function relativeTime(value) {
  const seconds = Math.max(0, (Date.now() - new Date(value).getTime()) / 1000);
  if (seconds < 60) return "now";
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m`;
  if (seconds < 86400) return `${Math.floor(seconds / 3600)}h`;
  return `${Math.floor(seconds / 86400)}d`;
}

function formatTime(value) {
  return new Intl.DateTimeFormat([], { hour: "numeric", minute: "2-digit" }).format(new Date(value));
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

elements.newChat.addEventListener("click", createChat);
elements.composer.addEventListener("submit", submitTurn);
elements.input.addEventListener("input", autoSizeInput);
elements.input.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    elements.composer.requestSubmit();
  }
});
elements.title.addEventListener("change", renameChat);
elements.title.addEventListener("keydown", (event) => {
  if (event.key === "Enter") { event.preventDefault(); elements.title.blur(); }
  if (event.key === "Escape") { elements.title.value = state.chat?.title || "Audex"; elements.title.blur(); }
});
elements.attach.addEventListener("click", () => elements.file.click());
elements.file.addEventListener("change", chooseAudio);
elements.record.addEventListener("click", toggleRecording);
elements.clearAudio.addEventListener("click", clearAudio);
elements.mobileMenu.addEventListener("click", () => {
  setSidebarOpen(!elements.sidebar.classList.contains("open"));
});
elements.sidebarClose.addEventListener("click", () => setSidebarOpen(false, { restoreFocus: true }));
elements.sidebarBackdrop.addEventListener("click", () => setSidebarOpen(false, { restoreFocus: true }));
document.addEventListener("keydown", (event) => {
  if (event.key === "Escape" && elements.sidebar.classList.contains("open")) {
    event.preventDefault();
    setSidebarOpen(false, { restoreFocus: true });
    return;
  }
  if (
    event.key === "Enter"
    && !event.metaKey
    && !event.ctrlKey
    && !event.altKey
    && !state.busy
    && currentMode()?.input_kind === "speech"
    && !["INPUT", "TEXTAREA", "BUTTON"].includes(document.activeElement?.tagName)
  ) {
    event.preventDefault();
    void toggleRecording();
    return;
  }
  if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "n") {
    event.preventDefault();
    createChat();
  }
});
narrowLayout.addEventListener("change", () => setSidebarOpen(false));
window.addEventListener("pagehide", cancelRecording);

function setSidebarOpen(open, { restoreFocus = false } = {}) {
  const isNarrow = narrowLayout.matches;
  const nextOpen = isNarrow && open;
  elements.sidebar.classList.toggle("open", nextOpen);
  elements.sidebarBackdrop.classList.toggle("hidden", !nextOpen);
  elements.mobileMenu.setAttribute("aria-expanded", String(nextOpen));
  elements.mobileMenu.setAttribute("aria-label", nextOpen ? "Hide conversations" : "Show conversations");
  if (isNarrow) {
    elements.sidebar.toggleAttribute("inert", !nextOpen);
    elements.sidebar.setAttribute("aria-hidden", String(!nextOpen));
    if (nextOpen) {
      elements.sidebarClose.focus();
      elements.conversationPanel.setAttribute("inert", "");
      elements.conversationPanel.setAttribute("aria-hidden", "true");
    } else {
      elements.conversationPanel.removeAttribute("inert");
      elements.conversationPanel.removeAttribute("aria-hidden");
    }
  } else {
    elements.sidebar.removeAttribute("inert");
    elements.sidebar.removeAttribute("aria-hidden");
    elements.conversationPanel.removeAttribute("inert");
    elements.conversationPanel.removeAttribute("aria-hidden");
  }
  if (!nextOpen && restoreFocus && isNarrow) elements.mobileMenu.focus();
}

setSidebarOpen(false);
bootstrap();
