Feature: vLLM Metal patch guards
  Audex-Mac uses pinned vLLM Metal monkey patches and should fail loudly only
  when the pinned target is not safe to patch.

  @fast
  Scenario: Upstream vLLM Metal HEAD moved but pinned commit is intact
    Given the installed vLLM Metal package matches the pinned commit
    And upstream vLLM Metal HEAD differs from the pinned commit
    When start.sh runs patch guards
    Then startup continues
    And a loud advisory warning is shown
    And the generated coding-agent update prompt is written to the log

  @fast
  Scenario: Pinned vLLM Metal API shape changed
    Given the installed vLLM Metal package does not expose a required patched symbol
    When start.sh runs patch guards
    Then startup stops before model launch
    And the error names the missing symbol
    And the error points to docs/engineering/patches.md

  @fast
  Scenario: Pinned vLLM Metal callable signature changed
    Given a required vLLM Metal patch target has an incompatible signature
    When start.sh runs patch guards
    Then startup stops before model launch
    And the error names the missing required parameter
    And the error points to docs/engineering/patches.md

  @fast
  Scenario: vLLM Metal diagnostics distinguish the CPU facade from CPU fallback
    Given vLLM Metal reports its compatibility CPU facade
    And MLX reports a GPU default device
    When Audex-Mac evaluates the vLLM Metal diagnostic report
    Then the report treats the CPU facade as expected
    And the report treats MLX GPU evidence as required
