---
name: team-review
description: "Coordinate a multi-agent code review with specialized reviewers for different aspects of the code."
user_invocable: true
aliases: ["team review", "review"]
---

# Team Review - Multi-Agent Code Review

You are coordinating a comprehensive code review with multiple specialized agents. Each agent reviews from their area of expertise, providing thorough coverage.

## Overview

The `/team review` command coordinates reviews from:
- **pr-reviewer** - Overall code quality and approval authority
- **qa-test-agent** - Test coverage and quality
- **brutal-code-reviewer** - Maintainability and AI-friendliness
- **ml-theory-advisor** - ML methodology (if applicable)
- **mlops-engineer** - Deployment readiness (if applicable)
- **snowflake-engineer** - Snowflake code (if applicable)

## Usage

```bash
# Review current changes (uncommitted)
/team review

# Review specific files
/team review src/preprocessing.py src/model.py

# Review a PR
/team review --pr 123

# Review with specific focus
/team review --focus ml
/team review --focus deployment
/team review --focus tests
```

## Your Review Workflow

### Step 1: Identify Changes

Determine what needs review:

```bash
# Uncommitted changes
git diff --name-only

# Staged changes
git diff --cached --name-only

# PR changes
gh pr diff {pr_number} --name-only
```

**Categorize files:**
- `src/*.py` - Source code (needs code review, possibly ML review)
- `tests/*.py` - Test files (needs test quality review)
- `api/*.py` - API code (needs deployment review)
- `deploy/*` - Deployment configs (needs mlops review)
- `*snowflake*` - Snowflake code (needs snowflake review)

### Step 2: Coordinate Reviews (Parallel)

**Always invoke:**

1. **pr-reviewer** - Mandatory for all reviews
   ```
   Review the following code changes for overall quality, security,
   and adherence to best practices. Files: {file_list}

   Provide:
   - Overall assessment
   - Blocking issues
   - Suggestions for improvement
   - Approval recommendation
   ```

2. **qa-test-agent** - For any code changes
   ```
   Review test coverage and quality for: {file_list}

   Check:
   - Test coverage percentage
   - Test quality (meaningful vs padding)
   - Edge cases covered
   - Missing test scenarios

   Provide coverage report and recommendations.
   ```

3. **brutal-code-reviewer** - For source code
   ```
   Review the following code for maintainability, clarity, and
   AI-friendliness: {file_list}

   Evaluate:
   - Code readability
   - Function/variable naming
   - Documentation quality
   - Complexity and refactoring opportunities
   - Potential bugs or issues
   ```

**Conditionally invoke:**

4. **ml-theory-advisor** - If ML code detected
   ```
   Review the following ML code for methodology issues: {file_list}

   Check:
   - Data leakage potential
   - Train/test split validity
   - Evaluation methodology
   - Overfitting risks
   ```

5. **mlops-engineer** - If deployment code detected
   ```
   Review the following deployment code: {file_list}

   Verify:
   - Production readiness
   - Security configurations
   - Health checks present
   - Logging and monitoring
   ```

6. **snowflake-engineer** - If Snowflake code detected
   ```
   Review the following Snowflake code: {file_list}

   Check:
   - Best practices compliance
   - Performance considerations
   - Security configurations
   - Cost implications
   ```

### Step 3: Collect and Consolidate Feedback

Gather results from all reviewers:

```markdown
## Review Summary

### Files Reviewed
- src/preprocessing.py
- src/model.py
- tests/unit/test_preprocessing.py

### Reviewers
| Reviewer | Status | Blocking Issues |
|----------|--------|-----------------|
| pr-reviewer | ✅ Approved | 0 |
| qa-test-agent | ⚠️ Changes Requested | 1 |
| brutal-code-reviewer | ✅ Approved | 0 |
| ml-theory-advisor | ✅ Approved | 0 |

### Overall Status: CHANGES REQUESTED
```

### Step 4: Present Detailed Findings

**Blocking Issues:**
```markdown
### 🚫 Blocking Issues

#### Issue 1: Insufficient Test Coverage
- **Reviewer**: qa-test-agent
- **File**: src/model.py
- **Lines**: 45-60
- **Issue**: New function `train_model` has no tests
- **Required Action**: Add unit tests for train_model function

```python
# Suggested test
def test_train_model_returns_fitted_model():
    X, y = create_sample_data()
    model = train_model(X, y)
    assert hasattr(model, 'predict')
```
```

**Non-Blocking Suggestions:**
```markdown
### 💡 Suggestions

#### Suggestion 1: Consider Type Hints
- **Reviewer**: brutal-code-reviewer
- **File**: src/preprocessing.py
- **Lines**: 12-15
- **Suggestion**: Add type hints for better documentation

```python
# Current
def impute_missing(df, strategy='median'):
    ...

# Suggested
def impute_missing(df: pd.DataFrame, strategy: str = 'median') -> pd.DataFrame:
    ...
```

#### Suggestion 2: Extract Magic Number
- **Reviewer**: brutal-code-reviewer
- **File**: src/model.py
- **Line**: 28
- **Suggestion**: Define 0.2 as a constant

```python
TEST_SIZE = 0.2
```
```

### Step 5: Approval Decision

Based on all reviews:

**If all reviewers approve and no blocking issues:**
```markdown
## ✅ APPROVED

All reviewers have approved. Ready to merge.

### Merge Instructions
```bash
git checkout main
git merge --squash feature-branch
git commit -m "feat: Add preprocessing pipeline

Reviewed-by: pr-reviewer
Reviewed-by: qa-test-agent
Reviewed-by: brutal-code-reviewer"
```
```

**If blocking issues exist:**
```markdown
## ❌ CHANGES REQUESTED

Please address the following before re-review:

1. [ ] Add tests for train_model function
2. [ ] Fix security issue in API endpoint

After fixing, run `/team review` again.
```

**If escalation needed:**
```markdown
## ⚠️ ESCALATION REQUIRED

Security concern detected. Human review required.

@maintainer Please review the API authentication changes.
```

## Review Report Format

```markdown
# Code Review Report

## Overview
- **Date**: {timestamp}
- **Commit**: {sha}
- **Files Changed**: {count}
- **Lines Changed**: +{added} / -{removed}

## Review Team
| Agent | Focus | Verdict |
|-------|-------|---------|
| pr-reviewer | Overall | ✅/❌/⚠️ |
| qa-test-agent | Testing | ✅/❌/⚠️ |
| brutal-code-reviewer | Quality | ✅/❌/⚠️ |
| ml-theory-advisor | ML | ✅/❌/⚠️ |

## Blocking Issues
{list of blocking issues}

## Suggestions
{list of suggestions}

## Test Coverage
- Current: {x}%
- Required: 80%
- Status: {PASS/FAIL}

## Final Verdict
{APPROVED / CHANGES_REQUESTED / ESCALATE}

## Next Steps
{actions to take}
```

## Configuration Options

| Option | Default | Description |
|--------|---------|-------------|
| `--pr` | none | Review specific PR by number |
| `--focus` | all | Focus area (ml, deployment, tests, quality) |
| `--strict` | false | Block on any suggestion (not just issues) |
| `--auto-fix` | false | Attempt automatic fixes for simple issues |

## Agent Access Control

Refer to `.claude/permissions.json` for agent approval authorities:

| Agent | Can Approve | Can Block | Can Merge |
|-------|-------------|-----------|-----------|
| pr-reviewer | Yes | Yes | Yes |
| qa-test-agent | Yes (tests) | Yes | No |
| brutal-code-reviewer | Yes | Yes | No |
| ml-theory-advisor | Yes (ML) | Yes | No |
| mlops-engineer | Yes (deploy) | Yes | No |
| snowflake-engineer | Yes (Snowflake) | Yes | No |
