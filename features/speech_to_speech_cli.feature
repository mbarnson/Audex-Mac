Feature: Speech-to-speech CLI
  The demo should provide one local conversation that accepts typed text or
  push-to-talk speech and always answers with Audex speech.

  @fast
  Scenario: User types to Audex and hears the answer
    Given the interactive Audex CLI is ready for a user turn
    When the user types a multiline message and presses Enter
    Then ASR is skipped
    And the typed message is sent directly to the conversation model
    And Audex generates and plays a spoken response

  @fast
  Scenario: Empty input keeps push-to-talk available
    Given the interactive Audex CLI is ready for a user turn
    When the user presses Enter without typing text
    Then Audex starts push-to-talk recording

  @slow @local_model @audio_device
  Scenario: Recorded speech completes a native Audex CLI turn
    Given a supported Audex model is cached
    And the Audex causal speech decoder is available
    And a recorded speech fixture is available
    When the CLI processes one recorded speech turn
    Then Audex receives the user's speech as native audio input
    And Audex generates a spoken response
    And the response is played locally on the Mac
    And the run log records the selected model and timing metrics

  @fast
  Scenario: Speech-to-speech defaults to non-thinking mode
    Given the CLI is started without a thinking flag
    When Audex-Mac builds the assistant response prefix
    Then it prepends <think></think>
    And it records thinking_enabled=false in the run log

  @fast
  Scenario: Interactive speech-to-speech keeps one Audex session alive
    Given the CLI is running in speech-to-speech mode
    When the user completes multiple push-to-talk turns
    Then the same Audex full model session handles every turn
    And the conversation history is retained until the context limit

  @fast
  Scenario: Speech-to-speech resumes persistent conversations by default
    Given a previous speech-to-speech conversation exists
    When the user starts the CLI without a conversation flag
    Then Audex-Mac resumes the previous conversation
    And the conversation text transcript is persisted to disk

  @fast
  Scenario: Speech-to-speech greets a first startup with Audex voice
    Given no previous speech-to-speech conversation is active
    When the interactive speech-to-speech CLI starts
    Then Audex says the first-startup greeting

  @fast
  Scenario: Speech-to-speech greets a returning named user with Audex voice
    Given a previous speech-to-speech conversation exists
    And the user identified themselves as Pat
    And the previous speech-to-speech conversation is resumed
    When the interactive speech-to-speech CLI starts
    Then Audex greets Pat as a returning user

  @fast
  Scenario: Speech-to-speech resumes without a name using a short greeting
    Given a previous speech-to-speech conversation exists
    And the previous speech-to-speech conversation is resumed
    When the interactive speech-to-speech CLI starts
    Then Audex welcomes the returning user

  @fast
  Scenario: Speech-to-speech can start and resume named conversations
    Given a previous speech-to-speech conversation exists
    When the user starts a new speech-to-speech conversation
    Then the new conversation becomes the default resume target
    When the user resumes the previous conversation by id
    Then Audex-Mac resumes the previous conversation

  @fast
  Scenario: Speech-to-speech persists binary MLX KV cache for conversation resume
    Given a previous speech-to-speech conversation exists
    When Audex-Mac saves the conversation state
    Then the conversation has a binary safetensors KV cache
    And the KV cache is matched to the conversation token hash

  @fast
  Scenario: Speech-to-speech uses markdown personas
    Given a markdown persona named assistant
    When Audex-Mac loads the speech-to-speech persona
    Then the persona body is added to the system prompt
    And the persona encourages concise empathetic spoken replies

  @fast
  Scenario: Speech-to-speech output is not capped for short smoke tests
    Given the voice agent may answer at conversational length
    When Audex-Mac resolves default speech-to-speech generation limits
    Then text generation allows at least 4096 tokens
    And speech generation uses a scaled audio-token budget

  @fast
  Scenario: Incremental ASR display hides Audex wrapper text
    Given Audex has emitted ASR wrapper text without transcript content
    When Audex-Mac cleans incremental ASR text
    Then the CLI suppresses wrapper-only ASR text
    And the CLI displays transcript text once it arrives

  @fast
  Scenario: Native audio input prompt uses Audex sound embeddings
    Given one 16 kHz utterance shorter than 30 seconds
    When Audex-Mac builds the native audio input prompt
    Then the prompt contains exactly 750 <so_embedding> tokens
    And the tokens are bracketed by <so_start> and <so_end>
    And no Whisper model is loaded

  @fast
  Scenario: vLLM speech-to-speech requests mirror NVIDIA's cascade
    Given a 16 kHz utterance for the vLLM speech-to-speech path
    When Audex-Mac builds the vLLM speech-to-speech request plan
    Then ASR is a vLLM multimodal audio request
    And text response generation is a non-thinking vLLM request by default
    And TTS uses paired vLLM CFG requests ending at <speechgen_start>

  @fast
  Scenario: vLLM speech-to-speech uses one persistent engine
    Given a persistent Audex vLLM runtime
    When Audex-Mac runs ASR text and TTS through the vLLM runtime
    Then the same vLLM engine receives every request
    And the TTS CFG pair is submitted as one paired engine call

  @fast
  Scenario: vLLM Metal is the default speech-to-speech backend
    Given the speech-to-speech CLI is configured with default options
    When Audex-Mac resolves the speech-to-speech backend
    Then speech-to-speech uses vLLM Metal by default
    And direct MLX speech-to-speech requires an explicit diagnostic backend selection

  @fast
  Scenario: vLLM speech-to-speech submits Audex projected audio embeddings
    Given projected Audex audio embeddings for the vLLM speech-to-speech path
    When Audex-Mac builds the projected vLLM ASR request
    Then the vLLM ASR request carries audex_projected_embeddings
    And the vLLM ASR request does not carry raw PCM audio

  @fast
  Scenario: PCM audio is prepared as fixed Audex clips
    Given a short stereo PCM utterance
    When Audex-Mac prepares PCM clips for Audex
    Then it produces one 480000-sample Audex clip
    And it preserves normalized mono samples before padding

  @fast
  Scenario: Audex audio preprocessing produces NV-Whisper features
    Given one prepared Audex PCM clip
    When Audex-Mac extracts Audex input features
    Then the feature tensor shape is 1 by 128 by 3000
    And the feature extractor is not a speech-to-text model

  @fast
  Scenario: Audex audio projector maps encoder frames to text embeddings
    Given Audex audio projector metadata
    When Audex-Mac resolves the audio projector tensors
    Then the projector expects 750 encoder frames per clip
    And the projector output hidden size is 2048

  @fast
  Scenario: Audex audio encoder maps Whisper features to encoder frames
    Given Audex audio encoder metadata
    When Audex-Mac resolves the audio encoder tensors
    Then the encoder expects 1 by 128 by 3000 input features
    And the encoder emits 750 frames with hidden size 1280

  @fast
  Scenario: Audex audio embeddings replace sound placeholder token embeddings
    Given an Audex prompt with two sound placeholder tokens
    And two projected Audex audio embeddings
    When Audex-Mac plans the audio embedding splice
    Then every sound placeholder has one projected audio embedding
    And mismatched audio embedding counts fail loudly

  @fast
  Scenario: Speech output uses the full Audex speech-token vocabulary
    Given the text-only Audex head has 131072 tokens
    And the full Audex head has 205312 tokens
    When Audex-Mac validates speech-token generation readiness
    Then the full head can address <speechgen_start> and speech codec tokens
    And the text-only head is rejected for speech output

  @fast
  Scenario: Audex causal speech decoder emits waveform samples
    Given eight Audex speech codec frames
    When Audex-Mac validates speech decoder output readiness
    Then the decoder output is finite 16 kHz waveform audio
    And the decoder output has 320 samples per codec frame

  @fast
  Scenario: Audex speech output smoke writes local audio artifacts
    Given generated Audex speech codec frames
    And a decoded Audex waveform
    When Audex-Mac validates speech output artifact readiness
    Then a local WAV artifact is present
    And a speech output run log is present
