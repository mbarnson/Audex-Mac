Feature: Startup bootstrap
  A user should be able to clone the repository, run one command, and reach the
  Audex-Mac CLI without manual dependency setup.

  @fast
  Scenario: Missing vLLM Metal runtime is bootstrapped
    Given no pinned vLLM Metal runtime exists
    When the user runs ./start.sh
    Then start.sh clones and checks out the pinned vLLM Metal commit
    And installs Audex-Mac into that runtime
    And enforces the runtime patch guards

  @fast
  Scenario: Existing valid bootstrap state avoids dependency churn
    Given the pinned vLLM Metal runtime imports successfully
    When the user runs ./start.sh
    Then start.sh does not reinstall dependencies by default
    And proceeds to model selection

  @fast
  Scenario: Missing model requires explicit download approval
    Given no supported Audex model is cached
    When the user runs ./start.sh
    Then start.sh explains the selected model size and NVIDIA license
    And asks for confirmation before downloading
