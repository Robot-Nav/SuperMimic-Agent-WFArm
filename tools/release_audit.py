"""Audit a source-only release, including the exact files tracked by Git."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import subprocess

ROOT_FILES = {
    "README.md", "README_CN.md", "LICENSE", "NOTICE.md", "pyproject.toml",
    "requirements-wf.txt", "requirements-wf-media.txt", "requirements-wf-teacher.txt",
    ".gitignore", ".gitattributes", ".python-version",
}
DOC_FILES = {
    "docs/README.md", "docs/PROJECT_ANALYSIS_CN.md", "docs/PRIVATE_ASSETS.md",
    "docs/REPOSITORY_DESCRIPTION.md", "third_party/README.md", "scripts/README.md",
    "configs/README.md", "frameworks/fixed_base_arm/README.md",
}
SCRIPT_FILES = {
    "scripts/wf_mimicagent_pilot.py", "scripts/wf_mimicagent_smoke.py",
    "scripts/eval_benchmark.py", "scripts/eval_push.py", "scripts/diag_drift.py",
    "scripts/merge_benchmark_shards.py",
    "scripts/data/build_library.py", "scripts/data/filter_bones_seed.py",
    "scripts/data/csv_to_npz_batch.py",
}
TOOL_FILES = {"tools/check_assets.py", "tools/prepare_release.py", "tools/release_audit.py"}
FORBIDDEN_PARTS = {
    "wf-arm-tbot", "assets", "private_assets", "meshes", "cad", "calibration",
    "artifacts", "logs", "media", "runs", "exports", "supervision", "history",
    "maintenance", ".aws", ".codex", ".agents", ".release",
}
SECRET_PATTERNS = [
    re.compile(r"gh[pousr]_[A-Za-z0-9]{20,}"),
    re.compile(r"github_pat_[A-Za-z0-9_]{20,}"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
]
PERSONAL_PATH = re.compile(r"/(?:home|Users)/[A-Za-z0-9_.-]+/")


def allowed(name):
    path = Path(name)
    if any(part in FORBIDDEN_PARTS for part in path.parts):
        return False
    if name in ROOT_FILES | DOC_FILES | SCRIPT_FILES | TOOL_FILES:
        return True
    if name.startswith("src/supermimic/") and path.suffix == ".py":
        return True
    if name.startswith("frameworks/fixed_base_arm/") and "arm_smc" in path.parts and path.suffix == ".py":
        return True
    if name in {"configs/examples/wf_train.json", "configs/examples/fixed_arm_smoke.json",
                "tests/test_release_core.py", ".github/workflows/tests.yml"}:
        return True
    return name.startswith("examples/mimicagent/") and path.suffix in {".py", ".txt"}


def scan(root, names):
    root = Path(root).resolve()
    problems = []
    total = 0
    for name in sorted(names):
        path = root / name
        if not allowed(name):
            problems.append(dict(path=name, issue="outside source-release allowlist"))
            continue
        if path.is_symlink() or not path.resolve().is_relative_to(root):
            problems.append(dict(path=name, issue="symlink or escaped source root"))
            continue
        payload = path.read_bytes(); total += len(payload)
        if len(payload) > 512 * 1024 or b"\x00" in payload:
            problems.append(dict(path=name, issue="binary or oversized payload"))
            continue
        try:
            content = payload.decode("utf-8")
        except UnicodeDecodeError:
            problems.append(dict(path=name, issue="not UTF-8 source text"))
            continue
        if any(pattern.search(content) for pattern in SECRET_PATTERNS):
            problems.append(dict(path=name, issue="possible credential; value not printed"))
        if PERSONAL_PATH.search(content):
            problems.append(dict(path=name, issue="personal absolute path"))
    return dict(files=len(names), bytes=total, passed=not problems, problems=problems)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--tracked", action="store_true")
    args = parser.parse_args()
    if args.tracked:
        result = subprocess.run(["git", "ls-files", "-z"], cwd=args.root,
                                check=True, capture_output=True, text=True)
        names = [name for name in result.stdout.split("\x00") if name]
    else:
        ignored = {".git", ".venv", "__pycache__", ".pytest_cache", "build", "dist"}
        names = [str(p.relative_to(args.root)) for p in args.root.rglob("*")
                 if p.is_file() and not any(part in ignored or part.endswith(".egg-info")
                                           for part in p.relative_to(args.root).parts)]
    report = scan(args.root, names)
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
