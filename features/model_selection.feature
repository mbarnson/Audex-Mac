Feature: Model selection
  Audex-Mac should choose the most useful supported cached model without making
  first-run setup unnecessarily expensive.

  @fast
  Scenario: Prefers cached NVFP4 30B over cached BF16 30B
    Given the Audex 30B-A3B NVFP4 snapshot is fully present in the Hugging Face cache
    And the Audex 30B-A3B snapshot is fully present in the Hugging Face cache
    When start.sh resolves the model to launch
    Then it selects txgsync/Nemotron-Labs-Audex-30B-A3B-NVFP4-mlx

  @fast
  Scenario: Uses cached 30B model when present
    Given the Audex 30B-A3B snapshot is fully present in the Hugging Face cache
    And the Audex 2B snapshot is fully present in the Hugging Face cache
    When start.sh resolves the model to launch
    Then it selects nvidia/Nemotron-Labs-Audex-30B-A3B
    And it logs that 30B-A3B was selected because it was already cached

  @fast
  Scenario: Defaults to cached 2B when 30B is absent
    Given the Audex 30B-A3B snapshot is not fully present in the Hugging Face cache
    And the Audex 2B snapshot is fully present in the Hugging Face cache
    When start.sh resolves the model to launch
    Then it selects nvidia/Nemotron-Labs-Audex-2B

  @fast
  Scenario: Text commands skip cached 30B when its text checkpoint is incomplete
    Given the Audex 30B-A3B speech snapshot is fully present in the Hugging Face cache
    And the Audex 30B-A3B text checkpoint is not fully present in the Hugging Face cache
    And the Audex 2B text checkpoint is fully present in the Hugging Face cache
    When start.sh resolves the model for a text command
    Then it selects nvidia/Nemotron-Labs-Audex-2B

  @fast
  Scenario: Defaults to 2B when no supported model is cached
    Given no supported Audex snapshot is fully present in the Hugging Face cache
    When start.sh resolves the model to launch
    Then it selects nvidia/Nemotron-Labs-Audex-2B
    And it tells the user that 2B is the default first-run model
    And it mentions nvidia/Nemotron-Labs-Audex-30B-A3B as the higher-reasoning option
