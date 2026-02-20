---
name: test
description: "Generate and run tests for Python modules. Automatically creates unit tests, integration tests, and validates test coverage meets the 80% threshold."
user_invocable: true
---

# Test Generation and Execution Skill

You are generating and running tests for the data science project.

## What This Skill Does

1. **Analyze Target**: Identify functions and classes that need tests
2. **Generate Tests**: Create comprehensive unit and integration tests
3. **Run Tests**: Execute tests with pytest
4. **Report Coverage**: Show coverage metrics and uncovered lines
5. **Enforce Threshold**: Ensure 80% minimum coverage

## Usage Patterns

### Test Specific Module
```
/test src/preprocessing.py
```

### Test All Source Files
```
/test src/
```

### Test with Coverage Report
```
/test --coverage
```

### Generate Tests Only (No Run)
```
/test --generate-only src/model.py
```

## Your Workflow

### Step 1: Analyze the Target

Read the target file(s) and identify:
- All functions and their signatures
- All classes and their methods
- Input/output types
- Edge cases to consider
- Dependencies to mock

### Step 2: Generate Test Structure

For each module, create tests following this structure:

```python
# tests/unit/test_{module_name}.py

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
from src.{module_name} import *


class Test{FunctionName}:
    """Tests for {function_name} function."""

    @pytest.fixture
    def sample_input(self):
        """Provide sample input for testing."""
        return ...

    # === Happy Path ===

    def test_{function_name}_basic(self, sample_input):
        """Test basic expected behavior."""
        result = {function_name}(sample_input)
        assert result is not None
        # Add specific assertions

    def test_{function_name}_with_valid_data(self):
        """Test with typical valid input."""
        ...

    # === Edge Cases ===

    def test_{function_name}_empty_input(self):
        """Test with empty input."""
        ...

    def test_{function_name}_single_element(self):
        """Test with single element."""
        ...

    def test_{function_name}_large_input(self):
        """Test with large dataset (performance)."""
        ...

    # === Error Handling ===

    def test_{function_name}_invalid_type(self):
        """Test that invalid types raise errors."""
        with pytest.raises(TypeError):
            {function_name}(invalid_input)

    def test_{function_name}_missing_required(self):
        """Test missing required parameters."""
        with pytest.raises(ValueError):
            {function_name}()

    # === Data-Specific (for ML/Data code) ===

    def test_{function_name}_handles_nan(self):
        """Test handling of NaN values."""
        ...

    def test_{function_name}_handles_outliers(self):
        """Test handling of extreme values."""
        ...
```

### Step 3: Run Tests

Execute with pytest:

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=src --cov-report=term-missing --cov-report=html

# Run specific test file
pytest tests/unit/test_preprocessing.py -v

# Run with coverage threshold enforcement
pytest tests/ --cov=src --cov-fail-under=80
```

### Step 4: Report Results

Provide a summary in this format:

```markdown
## Test Results

### Summary
- **Tests Run**: {count}
- **Passed**: {passed}
- **Failed**: {failed}
- **Coverage**: {percentage}%
- **Status**: {PASS/FAIL}

### Coverage by File
| File | Coverage | Uncovered Lines |
|------|----------|-----------------|
| src/preprocessing.py | 92% | 45-48, 72 |
| src/model.py | 85% | 120-125 |

### Failed Tests
(if any)
- test_name: reason for failure

### Generated Test Files
- tests/unit/test_preprocessing.py
- tests/unit/test_model.py

### Recommendations
1. Add tests for uncovered lines in src/model.py
2. Consider edge case for negative values
```

## Test Templates by Module Type

### Preprocessing Module Tests

```python
class TestPreprocessingPipeline:
    """Tests for preprocessing functions."""

    @pytest.fixture
    def sample_df(self):
        return pd.DataFrame({
            'numeric': [1.0, 2.0, np.nan, 4.0],
            'categorical': ['A', 'B', 'A', None],
            'target': [0, 1, 1, 0]
        })

    def test_missing_values_handled(self, sample_df):
        """Ensure no missing values after preprocessing."""
        result = preprocess(sample_df)
        assert result.isnull().sum().sum() == 0

    def test_output_shape_preserved(self, sample_df):
        """Ensure row count is preserved."""
        result = preprocess(sample_df)
        assert len(result) == len(sample_df)

    def test_no_target_leakage(self, sample_df):
        """Ensure target is not used in features."""
        X = sample_df.drop('target', axis=1)
        result = preprocess(X)
        assert 'target' not in result.columns

    def test_deterministic(self, sample_df):
        """Ensure preprocessing is reproducible."""
        r1 = preprocess(sample_df.copy())
        r2 = preprocess(sample_df.copy())
        pd.testing.assert_frame_equal(r1, r2)
```

### Model Module Tests

```python
class TestModelTraining:
    """Tests for model training functions."""

    @pytest.fixture
    def training_data(self):
        X = pd.DataFrame({
            'feature1': np.random.randn(100),
            'feature2': np.random.randn(100)
        })
        y = np.random.randint(0, 2, 100)
        return X, y

    def test_model_trains_without_error(self, training_data):
        """Test model training completes."""
        X, y = training_data
        model = train_model(X, y)
        assert model is not None

    def test_predictions_valid_range(self, training_data):
        """Test predictions are valid probabilities."""
        X, y = training_data
        model = train_model(X, y)
        probs = model.predict_proba(X)[:, 1]
        assert all(0 <= p <= 1 for p in probs)

    def test_model_serialization(self, training_data, tmp_path):
        """Test model can be saved and loaded."""
        import joblib
        X, y = training_data
        model = train_model(X, y)

        path = tmp_path / "model.joblib"
        joblib.dump(model, path)
        loaded = joblib.load(path)

        np.testing.assert_array_equal(
            model.predict(X),
            loaded.predict(X)
        )
```

### API Module Tests

```python
from fastapi.testclient import TestClient
from api.app import app


class TestAPIEndpoints:
    """Tests for API endpoints."""

    @pytest.fixture
    def client(self):
        return TestClient(app)

    def test_health_check(self, client):
        """Test health endpoint returns 200."""
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"

    def test_predict_valid_input(self, client):
        """Test prediction with valid input."""
        response = client.post("/predict", json={
            "feature1": 1.0,
            "feature2": "value"
        })
        assert response.status_code == 200
        assert "prediction" in response.json()

    def test_predict_invalid_input(self, client):
        """Test prediction with invalid input returns 422."""
        response = client.post("/predict", json={
            "invalid": "data"
        })
        assert response.status_code == 422
```

## Integration with CI

Tests are automatically run in CI via GitHub Actions:

```yaml
# .github/workflows/ci.yml
- name: Run tests
  run: pytest tests/ -v --cov=src --cov-fail-under=80
```

## Coverage Requirements

- **Minimum threshold**: 80%
- **Critical paths**: 95%
- **New code**: Must have tests

## After Test Generation

After generating tests, the `qa-test-agent` will be invoked to:
1. Review test quality
2. Suggest improvements
3. Verify coverage meets threshold
