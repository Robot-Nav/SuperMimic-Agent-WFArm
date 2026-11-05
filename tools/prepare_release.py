"""Build an isolated, audited source release from the research workspace."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import shutil

from release_audit import DOC_FILES, ROOT_FILES, SCRIPT_FILES, TOOL_FILES, scan

ROOT = Path(__file__).resolve().parents[1]


def dependency_closure(base, entry_modules, package):
    """Follow Python imports; do not follow model assets or runtime data paths."""
    selected, pending = set(), list(entry_modules)
    while pending:
        module = pending.pop()
        if not module.startswith(package):
            continue
        relative = module[len(package):].lstrip(".").replace(".", "/")
        path = base / (relative + ".py") if relative else base / "__init__.py"
        if not path.is_file():
            path = base / relative / "__init__.py"
        if not path.is_file() or path in selected:
            continue
        selected.add(path)
        parts = module.split(".")
        parent = parts[:-1] if path.name != "__init__.py" else parts
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                pending.extend(a.name for a in node.names)
            elif isinstance(node, ast.ImportFrom):
                prefix = parent[:len(parent)-node.level+1] if node.level else []
                target = ".".join(prefix + ([node.module] if node.module else []))
                pending.append(target)
                pending.extend(target + "." + a.name for a in node.names)
        for ancestor in path.parents:
            if ancestor == base.parent:
                break
            init = ancestor / "__init__.py"
            if init.is_file():
                selected.add(init)
    return selected


def release_files():
    names = set(ROOT_FILES | DOC_FILES | SCRIPT_FILES | TOOL_FILES)
    names.update({"tests/test_release_core.py",
                  "configs/examples/wf_train.json", "configs/examples/fixed_arm_smoke.json"})
    entries = ["wf.train", "wf.evaluate", "wf.export", "wf.reference", "wf.agent_references",
               "wf.merge_references", "wf.dynamic_wbc", "wf.kinematic_wbc",
               "mimicagent.pipeline", "mimicagent.worker", "scripts.train", "eval.benchmark",
               "data.motion_convert", "data.recorder", "assets"]
    files = dependency_closure(ROOT / "src/supermimic", ["supermimic." + n for n in entries], "supermimic")
    for directory in ("frameworks/fixed_base_arm/arm_smc",
                      "frameworks/fixed_base_arm/variants/supermimic_robust_control_v7/arm_smc"):
        files |= dependency_closure(ROOT / directory,
            ["arm_smc.train", "arm_smc.inference", "arm_smc.export_policy"], "arm_smc")
    names.update(str(p.relative_to(ROOT)) for p in files)
    names.update(str(p.relative_to(ROOT)) for p in (ROOT / "examples/mimicagent").glob("*")
                 if p.suffix in {".py", ".txt"})
    return sorted(names)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--manifest", type=Path)
    args = p.parse_args()
    output = args.output.resolve()
    if output == ROOT or ROOT.is_relative_to(output):
        raise ValueError("release output must be separate from the source root")
    files = release_files()
    # Preflight both presence and content before copying the first file.
    missing = [name for name in files if not (ROOT / name).is_file()]
    if missing:
        raise FileNotFoundError("release files missing: " + ", ".join(missing))
    report = scan(ROOT, files)
    if not report["passed"]:
        raise ValueError(json.dumps(report, indent=2))
    output.mkdir(parents=True, exist_ok=False)
    manifest = []
    for name in files:
        source, target = ROOT / name, output / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        manifest.append(dict(path=name, bytes=target.stat().st_size,
                             sha256=hashlib.sha256(target.read_bytes()).hexdigest()))
    result = dict(files=len(files), bytes=sum(x["bytes"] for x in manifest),
                  source_only=True, audit=report, entries=manifest)
    if args.manifest:
        args.manifest.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "entries"}, indent=2))


if __name__ == "__main__":
    main()
