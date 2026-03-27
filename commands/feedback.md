---
name: feedback
description: "Collect structured feedback from test users. Records feedback with severity, category, and context for system improvement."
user_invocable: true
aliases: ["report-feedback", "user-feedback"]
---

# Feedback Collection

You are collecting structured feedback from test users to improve the ml-automation system.

## How to Collect Feedback

### Step 1: Ask for Feedback

Present the user with a structured feedback form:

```
## Feedback Collection

What would you like to report?

1. **Bug** — Something isn't working as expected
2. **Improvement** — Something works but could be better
3. **Feature Request** — Something new you'd like to see
4. **Usability** — Confusing, unclear, or hard to use
5. **Performance** — Too slow, too many tokens, inefficient
6. **Quality** — Output quality issues (wrong results, bad code, poor analysis)
```

### Step 2: Gather Details

For each feedback type, ask:

**All types:**
- Which command/skill was involved? (e.g., `/eda`, `/team-coldstart`)
- What were you trying to do?
- What actually happened?

**Bug:** Steps to reproduce, error messages if any
**Improvement:** What would the ideal behavior look like?
**Feature Request:** What problem would this solve?
**Usability:** What was confusing and what would be clearer?
**Performance:** Which stage was slow? How long did it take?
**Quality:** What was wrong with the output? What would be correct?

### Step 3: Rate Severity

```
How impactful is this?
- **Critical** — Blocks work entirely
- **High** — Significant workaround needed
- **Medium** — Annoying but manageable
- **Low** — Minor polish item
```

### Step 4: Save Feedback

Save structured feedback to the report bus:

```python
from ml_utils import save_agent_report, log_trace_event
from datetime import datetime, timezone

feedback = {
    "status": "completed",
    "findings": {
        "summary": f"User feedback: {category} — {title}",
        "details": {
            "category": category,          # bug|improvement|feature|usability|performance|quality
            "severity": severity,          # critical|high|medium|low
            "title": title,
            "description": description,
            "command_involved": command,
            "steps_to_reproduce": steps,   # if bug
            "expected_behavior": expected,
            "actual_behavior": actual,
            "submitted_by": user_name,
            "submitted_at": datetime.now(timezone.utc).isoformat(),
        },
    },
    "recommendations": [{
        "action": f"Address {category}: {title}",
        "target_agent": "orchestrator",
        "priority": severity,
    }],
    "artifacts": [],
}

save_agent_report(f"feedback-{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}", feedback)

log_trace_event({
    "event_type": "user_action",
    "action": f"Submitted feedback: {category} — {title}",
    "details": {"severity": severity, "command": command},
    "command": "feedback",
})
```

### Step 5: Confirm

Display confirmation:
```
Feedback recorded. Thank you!

Summary:
- Type: {category}
- Severity: {severity}
- Title: {title}

Your feedback has been saved to the report bus and will be reviewed
for system improvement.
```

## Viewing Past Feedback

To view all collected feedback:
```python
from ml_utils import load_agent_reports

reports = load_agent_reports()
feedback_reports = {k: v for k, v in reports.items() if k.startswith("feedback-")}

for name, report in sorted(feedback_reports.items()):
    details = report.get("findings", {}).get("details", {})
    print(f"[{details.get('severity', '?')}] {details.get('category', '?')}: {details.get('title', '?')}")
```
