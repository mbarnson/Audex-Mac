Feature: Text runtime compatibility and model quality evidence
  Text-only Audex generation should prove that the runtime can sustain the
  checked-in conversation while recording model quality separately.

  @slow @local_model
  Scenario: Ten-turn coding benchmark completes with runtime evidence
    Given a supported Audex model is selected
    And NVIDIA-recommended sampler settings are configured for text
    And max_tokens is at least 4096
    When the text benchmark conversation is executed
    Then ten-turn text runtime compatibility passes
    And model-quality observations are recorded without controlling compatibility
    And the run log records selected model, sampler params, timings, and transcript
    And the selected backend run log records token throughput and runtime evidence

  @fast
  Scenario: Model reasoning quality is non-blocking
    Given the text benchmark has completed with a reasoning error
    When Audex-Mac assesses text runtime compatibility and model quality
    Then it does not require exact token parity
    And it does not require logit parity
    And it does require runtime-compatible output
    And it records the reasoning error as non-blocking model quality

  @fast
  Scenario: Thinking-mode benchmark history remains valid for ten turns
    Given ten generated thinking-mode benchmark replies
    When Audex-Mac assembles the ten-turn benchmark conversation
    Then every assistant history entry contains a complete reasoning section
    And every turn is rendered through the selected model chat template
    And benchmark evaluation sees public answers rather than private reasoning

  @fast
  Scenario: vLLM Metal is the default text benchmark backend
    Given the text benchmark CLI is configured with default options
    When Audex-Mac resolves the text benchmark backend
    Then it uses vLLM Metal by default
    And direct MLX requires an explicit diagnostic backend selection
