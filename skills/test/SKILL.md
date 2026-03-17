---
name: test
description: Generate and run tests for Python modules. Automatically creates unit tests, integration tests, and validates test coverage meets the 80% threshold.
---

# Testing

## When to Use
- Generating tests for Python modules
- Validating test coverage meets thresholds
- Running unit and integration test suites

## Workflow

1. **Analyze target** — Read source files, identify functions/classes/signatures
2. **Generate unit tests** — One test file per source module in `tests/`
3. **Generate integration tests** — Test cross-module workflows
4. **Run with pytest** — `pytest tests/ -v --tb=short --cov=src --cov-report=term-missing`
5. **Validate coverage** — Must meet 80% threshold; report uncovered lines
6. **Quality review** — Invoke `brutal-code-reviewer` to review generated tests

## Key Test Patterns

- **Preprocessing tests**: Verify pipeline handles NaN, unseen categories, empty DataFrames
- **Model tests**: Check predict output shape, reproducibility with random_state, serialization round-trip
- **API tests**: Verify endpoints return correct status codes, handle invalid payloads gracefully

## Flags

| Flag | Description |
|------|-------------|
| `--generate-only` | Generate test files without running them |
| `--coverage` | Show detailed coverage report with uncovered lines |

## Report Bus Integration (v1.2.0)

Save test results to the report bus:
```python
from ml_utils import save_agent_report
save_agent_report("test-runner", {
    "status": "completed",
    "findings": {"coverage": 85, "tests_passed": 42, "tests_failed": 0},
    "recommendations": [],
    "artifacts": ["tests/test_preprocessing.py", "tests/test_model.py"]
})
```

## Full Specification

See `commands/test.md` for complete test templates, pytest commands, and CI integration details.
