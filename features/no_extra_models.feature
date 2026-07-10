Feature: No extra semantic models
  Audex-Mac should prove Audex native speech-to-speech rather than building a
  cascaded pipeline out of other models.

  @fast
  Scenario: Push-to-talk does not use external STT, TTS, or VAD
    Given the CLI is running in speech-to-speech mode
    When the user records an utterance with push-to-talk
    Then the input audio is passed to Audex audio input processing
    And no Whisper model is loaded
    And no Kokoro model is loaded
    And no Silero VAD model is loaded
    And the spoken response is decoded with the Audex causal speech decoder

  @fast
  Scenario: Deterministic audio plumbing is allowed
    Given the CLI captured audio from the microphone
    When Audex-Mac prepares the audio for Audex
    Then it may resample PCM
    And it may normalize audio samples
    And it may use codec tools for deterministic conversion
    But it must not infer speech text with a separate model

