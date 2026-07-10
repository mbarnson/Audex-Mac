Feature: Startup bootstrap
  A user should be able to clone the repository, run one command, and reach the
  Audex-Mac CLI without manual dependency setup.

  @fast
  Scenario: Missing virtual environment is bootstrapped
    Given no local virtual environment exists
    When the user runs ./start.sh
    Then start.sh creates a local virtual environment
    And installs huggingface_hub
    And installs pinned project dependencies

  @fast
  Scenario: Existing valid bootstrap state avoids dependency churn
    Given the local virtual environment matches the pinned dependency state
    When the user runs ./start.sh
    Then start.sh does not reinstall dependencies by default
    And proceeds to model selection

  @fast
  Scenario: Missing model requires explicit download approval
    Given no supported Audex model is cached
    When the user runs ./start.sh
    Then start.sh explains the selected model size and NVIDIA license
    And asks for confirmation before downloading

