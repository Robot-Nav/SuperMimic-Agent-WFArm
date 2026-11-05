"""Bounded CPU frontend pilot, isolated from every existing policy and run."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from supermimic.mimicagent.pipeline import export_library, run, write_json

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--terminal-feedback", action="store_true")
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    examples = ROOT / "examples/mimicagent"
    requests = [
        ("arm_lift_small", "站立四轮支撑，机械臂从折叠零位小幅展开肩肘，随后保持"),
        ("arm_reach_small", "站立四轮支撑，机械臂从折叠零位稍大幅展开肩肘，随后保持"),
        ("drive_and_lift", "四轮前进至0.5m/s并小幅展开肩肘，随后停车保持"),
    ]
    rows = []
    for name, prompt in requests:
        output = a.output / name
        result = run(prompt, (examples / "plan.txt").read_text(),
                     [examples / (name + ".py")], output, physical=True,
                     terminal_feedback=a.terminal_feedback)
        last = result["attempts"][-1]
        rows.append(dict(skill=name, kinematic=last.get("kinematic_accepted", False),
                         physical=result["training_eligible"], report=str(output / "report.json")))
        print(json.dumps(rows[-1], ensure_ascii=False), flush=True)
    eligible = [a.output / r["skill"] for r in rows if r["physical"]]
    if len(eligible) >= 2:
        export_library(eligible, a.output / "physical_library")
    write_json(a.output / "summary.json", dict(requests=rows,
        terminal_feedback=a.terminal_feedback,
        kinematic_passed=sum(r["kinematic"] for r in rows),
        physical_passed=sum(r["physical"] for r in rows),
        policy_improvement_measured=False,
        scope="3 coding-agent authored small motions; pipeline smoke, not original benchmark or full-workspace qualification"))


if __name__ == "__main__":
    main()
