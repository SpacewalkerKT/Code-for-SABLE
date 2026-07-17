# Release Notes

## revision-release-20260717

This version updates the SABLE implementation and documentation to align with the revised manuscript. It corrects trainable-parameter initialization, clarifies the documented workflow, and refreshes the release package. No new experimental results are introduced.

Changes:

- Corrected trainable-parameter initialization from `nn.parameter(...)` to `nn.Parameter(...)`.
- Updated documentation to match the revised manuscript.
- Clarified task construction and episodic workflow.
- Clarified that quick-test execution is for environment and pipeline validation, not a new paper experiment.
- Added citation metadata and data documentation.

Notes:

- Quick-test outputs should not be interpreted as paper results.
- Retained archived outputs should not be described as newly repeated experiments.
