#!/usr/bin/env python3
"""
Eval Runner for ML Automation Plugin

Loads eval definitions from evals/*.json,
records results per iteration, and produces a summary report.

Usage:
    python evals/eval_runner.py --init-iteration 2
    python evals/eval_runner.py --record SKILL EVAL ASSERTION PASS|FAIL [--iteration N] [--notes "..."]
    python evals/eval_runner.py --record-eval SKILL EVAL PASS|FAIL [--iteration N]
    python evals/eval_runner.py --summary [--iteration N]
    python evals/eval_runner.py --compare ITER_A ITER_B
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

EVALS_DIR = Path(__file__).parent
ITERATIONS_DIR = Path(__file__).parent / "iterations"


def load_all_evals():
    """Load all eval definitions from evals/ directory."""
    all_evals = {}
    for f in sorted(EVALS_DIR.glob("*_evals.json")):
        with open(f) as fh:
            data = json.load(fh)
            skill = data.get("skill_name", f.stem.replace("_evals", ""))
            all_evals[skill] = data
    return all_evals


def init_iteration(iteration_num):
    """Initialize a new iteration directory with eval scaffolding."""
    iter_dir = ITERATIONS_DIR / f"iteration-{iteration_num}"
    iter_dir.mkdir(exist_ok=True)

    all_evals = load_all_evals()
    results = {
        "iteration": iteration_num,
        "created": datetime.now().isoformat(),
        "status": "in_progress",
        "results": {}
    }

    for skill_name, eval_data in all_evals.items():
        evals = eval_data.get("evals", [])
        results["results"][skill_name] = {}
        for ev in evals:
            eval_id = ev["name"]
            results["results"][skill_name][eval_id] = {
                "status": "pending",
                "assertions": {},
                "notes": "",
                "timestamp": None
            }
            # Initialize assertion results
            for assertion in ev.get("assertions", []):
                results["results"][skill_name][eval_id]["assertions"][assertion["id"]] = {
                    "status": "pending",
                    "text": assertion["text"]
                }

        # Create eval output directory
        for ev in evals:
            eval_dir = iter_dir / f"eval-{ev['name']}"
            eval_dir.mkdir(exist_ok=True)
            (eval_dir / "with_skill").mkdir(exist_ok=True)
            (eval_dir / "with_skill" / "outputs").mkdir(exist_ok=True)
            (eval_dir / "without_skill").mkdir(exist_ok=True)
            (eval_dir / "without_skill" / "outputs").mkdir(exist_ok=True)

            # Write eval metadata
            meta = {
                "eval_id": ev["id"],
                "eval_name": ev["name"],
                "skill": skill_name,
                "prompt": ev.get("prompt", ""),
                "assertions": ev.get("assertions", [])
            }
            with open(eval_dir / "eval_metadata.json", "w") as fh:
                json.dump(meta, fh, indent=2)

    # Save iteration results tracker
    with open(iter_dir / "results.json", "w") as fh:
        json.dump(results, fh, indent=2)

    total_evals = sum(len(d.get("evals", [])) for d in all_evals.values())
    total_assertions = sum(
        len(a) for d in all_evals.values()
        for ev in d.get("evals", [])
        for a in [ev.get("assertions", [])]
    )
    print(f"Initialized iteration-{iteration_num}")
    print(f"  Skills: {len(all_evals)}")
    print(f"  Evals: {total_evals}")
    print(f"  Assertions: {total_assertions}")
    print(f"  Directory: {iter_dir}")


def record_result(iteration_num, skill, eval_name, assertion_id, passed, notes=""):
    """Record a single assertion result."""
    iter_dir = ITERATIONS_DIR / f"iteration-{iteration_num}"
    results_file = iter_dir / "results.json"

    if not results_file.exists():
        print(f"Error: iteration-{iteration_num} not initialized. Run --init-iteration first.")
        sys.exit(1)

    with open(results_file) as fh:
        results = json.load(fh)

    if skill not in results["results"]:
        print(f"Error: skill '{skill}' not found in eval definitions.")
        sys.exit(1)

    if eval_name not in results["results"][skill]:
        print(f"Error: eval '{eval_name}' not found for skill '{skill}'.")
        sys.exit(1)

    eval_result = results["results"][skill][eval_name]

    if assertion_id:
        if assertion_id not in eval_result["assertions"]:
            print(f"Error: assertion '{assertion_id}' not found.")
            sys.exit(1)
        eval_result["assertions"][assertion_id]["status"] = "pass" if passed else "fail"
    else:
        # Record all assertions for this eval at once
        for aid in eval_result["assertions"]:
            eval_result["assertions"][aid]["status"] = "pass" if passed else "fail"

    eval_result["timestamp"] = datetime.now().isoformat()
    if notes:
        eval_result["notes"] = notes

    # Update eval status based on assertions
    statuses = [a["status"] for a in eval_result["assertions"].values()]
    if all(s == "pass" for s in statuses):
        eval_result["status"] = "pass"
    elif any(s == "fail" for s in statuses):
        eval_result["status"] = "fail"
    elif any(s == "pass" for s in statuses):
        eval_result["status"] = "partial"

    with open(results_file, "w") as fh:
        json.dump(results, fh, indent=2)

    status_str = "PASS" if passed else "FAIL"
    target = f"{assertion_id}" if assertion_id else "all assertions"
    print(f"[{status_str}] {skill}/{eval_name}/{target}")


def print_summary(iteration_num):
    """Print a summary of eval results for an iteration."""
    iter_dir = ITERATIONS_DIR / f"iteration-{iteration_num}"
    results_file = iter_dir / "results.json"

    if not results_file.exists():
        print(f"Error: iteration-{iteration_num} not found.")
        sys.exit(1)

    with open(results_file) as fh:
        results = json.load(fh)

    print(f"\n{'='*60}")
    print(f" Iteration {iteration_num} -- Eval Summary")
    print(f" Created: {results['created']}")
    print(f"{'='*60}\n")

    total_pass = 0
    total_fail = 0
    total_pending = 0

    for skill_name, evals in sorted(results["results"].items()):
        skill_pass = 0
        skill_fail = 0
        skill_pending = 0

        for eval_name, eval_data in evals.items():
            for aid, adata in eval_data["assertions"].items():
                if adata["status"] == "pass":
                    skill_pass += 1
                    total_pass += 1
                elif adata["status"] == "fail":
                    skill_fail += 1
                    total_fail += 1
                else:
                    skill_pending += 1
                    total_pending += 1

        total = skill_pass + skill_fail + skill_pending
        pct = (skill_pass / (skill_pass + skill_fail) * 100) if (skill_pass + skill_fail) > 0 else 0
        status_icon = "PASS" if skill_fail == 0 and skill_pending == 0 else ("PARTIAL" if skill_pass > 0 else "PENDING")

        print(f"  {skill_name:<20} {status_icon:<8} {skill_pass}/{total} assertions passed ({pct:.0f}%)")

        # Show failing assertions
        for eval_name, eval_data in evals.items():
            for aid, adata in eval_data["assertions"].items():
                if adata["status"] == "fail":
                    print(f"    FAIL: {eval_name}/{aid} -- {adata['text']}")

    total_all = total_pass + total_fail + total_pending
    overall_pct = (total_pass / (total_pass + total_fail) * 100) if (total_pass + total_fail) > 0 else 0

    print(f"\n{'_'*60}")
    print(f"  TOTAL: {total_pass}/{total_all} assertions passed ({overall_pct:.0f}%)")
    print(f"  Pass: {total_pass}  Fail: {total_fail}  Pending: {total_pending}")
    print(f"{'_'*60}\n")


def compare_iterations(iter_a, iter_b):
    """Compare results between two iterations."""
    results_a_file = ITERATIONS_DIR / f"iteration-{iter_a}" / "results.json"
    results_b_file = ITERATIONS_DIR / f"iteration-{iter_b}" / "results.json"

    if not results_a_file.exists() or not results_b_file.exists():
        print("Error: both iterations must exist.")
        sys.exit(1)

    with open(results_a_file) as fh:
        results_a = json.load(fh)
    with open(results_b_file) as fh:
        results_b = json.load(fh)

    print(f"\n{'='*60}")
    print(f" Comparison: Iteration {iter_a} vs Iteration {iter_b}")
    print(f"{'='*60}\n")

    for skill_name in sorted(set(list(results_a["results"].keys()) + list(results_b["results"].keys()))):
        evals_a = results_a["results"].get(skill_name, {})
        evals_b = results_b["results"].get(skill_name, {})

        pass_a = sum(1 for e in evals_a.values() for a in e["assertions"].values() if a["status"] == "pass")
        total_a = sum(len(e["assertions"]) for e in evals_a.values())
        pass_b = sum(1 for e in evals_b.values() for a in e["assertions"].values() if a["status"] == "pass")
        total_b = sum(len(e["assertions"]) for e in evals_b.values())

        delta = pass_b - pass_a
        arrow = f"+{delta}" if delta > 0 else str(delta)
        print(f"  {skill_name:<20} {pass_a}/{total_a} -> {pass_b}/{total_b}  ({arrow})")

        # Show regressions and fixes
        for eval_name in set(list(evals_a.keys()) + list(evals_b.keys())):
            ea = evals_a.get(eval_name, {}).get("assertions", {})
            eb = evals_b.get(eval_name, {}).get("assertions", {})
            for aid in set(list(ea.keys()) + list(eb.keys())):
                sa = ea.get(aid, {}).get("status", "n/a")
                sb = eb.get(aid, {}).get("status", "n/a")
                if sa == "pass" and sb == "fail":
                    print(f"    REGRESSION: {eval_name}/{aid}")
                elif sa == "fail" and sb == "pass":
                    print(f"    FIXED: {eval_name}/{aid}")


def _latest_iteration():
    """Find the latest iteration number."""
    iterations = [d.name for d in ITERATIONS_DIR.iterdir()
                  if d.is_dir() and d.name.startswith("iteration-")]
    if not iterations:
        print("Error: no iterations found. Run --init-iteration first.")
        sys.exit(1)
    nums = [int(d.split("-")[1]) for d in iterations]
    return max(nums)


def main():
    parser = argparse.ArgumentParser(description="ML Plugin Eval Runner")
    parser.add_argument("--init-iteration", type=int, help="Initialize a new iteration")
    parser.add_argument("--record", nargs=4, metavar=("SKILL", "EVAL", "ASSERTION", "RESULT"),
                        help="Record an assertion result (RESULT: pass or fail)")
    parser.add_argument("--record-eval", nargs=3, metavar=("SKILL", "EVAL", "RESULT"),
                        help="Record all assertions for an eval (RESULT: pass or fail)")
    parser.add_argument("--iteration", type=int, default=None, help="Iteration number")
    parser.add_argument("--summary", action="store_true", help="Print summary for an iteration")
    parser.add_argument("--compare", nargs=2, type=int, metavar=("ITER_A", "ITER_B"),
                        help="Compare two iterations")
    parser.add_argument("--notes", default="", help="Notes for --record")

    args = parser.parse_args()

    if args.init_iteration:
        init_iteration(args.init_iteration)
    elif args.record:
        skill, eval_name, assertion_id, result = args.record
        iteration = args.iteration or _latest_iteration()
        record_result(iteration, skill, eval_name, assertion_id, result.lower() == "pass", args.notes)
    elif args.record_eval:
        skill, eval_name, result = args.record_eval
        iteration = args.iteration or _latest_iteration()
        record_result(iteration, skill, eval_name, None, result.lower() == "pass", args.notes)
    elif args.summary:
        iteration = args.iteration or _latest_iteration()
        print_summary(iteration)
    elif args.compare:
        compare_iterations(args.compare[0], args.compare[1])
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
