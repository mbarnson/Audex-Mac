Feature: Licensing
  Audex-Mac should make clear that MIT applies to this repository's code, not
  to NVIDIA model weights or NVIDIA-provided model artifacts.

  @fast
  Scenario: README distinguishes project license from model license
    Given the user reads README.md
    Then the README says Audex-Mac source code is MIT licensed
    And the README says NVIDIA model weights are governed by NVIDIA's license
    And the README links to the Audex model cards

  @fast
  Scenario: Download prompt mentions NVIDIA license
    Given no supported Audex model is cached
    When start.sh asks before downloading a model
    Then the prompt mentions NVIDIA's model license
    And it does not imply the model weights are MIT licensed

