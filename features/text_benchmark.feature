Feature: Text benchmark gate
  Text-only Audex generation should pass a behavioral coherence check before
  the demo can rely on the model's text behavior before exercising audio.

  @slow @local_model
  Scenario: Ten-turn coding benchmark is coherent enough
    Given a supported Audex model is selected
    And NVIDIA-recommended sampler settings are configured for text
    And max_tokens is at least 4096
    When the text benchmark conversation is executed
    Then the deterministic text acceptance gate passes
    And the transcript does not show excessive repetition
    And the transcript retains context across the benchmark turns
    And the run log records selected model, sampler params, timings, and transcript
    And the vLLM run log records token throughput and Audex patch evidence

  @fast
  Scenario: Exact token parity is not required
    Given the text benchmark has completed
    When Audex-Mac evaluates the text gate
    Then it does not require exact token parity
    And it does not require logit parity
    And it does require coherent viable output

  @fast
  Scenario: vLLM Metal is the default text benchmark backend
    Given the text benchmark CLI is configured with default options
    When Audex-Mac resolves the text benchmark backend
    Then it uses vLLM Metal by default
    And direct MLX requires an explicit diagnostic backend selection
