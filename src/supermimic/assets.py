"""Locate privately supplied robot resources without embedding model data."""
from __future__ import annotations

import os
from pathlib import Path


def resolve_wfarm_assets(asset_dir: str | Path | None = None) -> Path:
    configured = asset_dir or os.environ.get("WFARM_ASSET_DIR")
    project = Path(__file__).resolve().parents[2]
    if configured:
        root = Path(os.path.expandvars(str(configured))).expanduser().resolve()
    else:
        local = project / "wf-arm-tbot"
        root = local if local.is_dir() else project / "private_assets/wf-arm-tbot"
    required = ("urdf/wf-arm-tbot.urdf", "config/robot.json", "config/robot_calibration.json")
    missing = [name for name in required if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError(
            "WF-Arm assets are supplied separately. Set WFARM_ASSET_DIR or asset_dir "
            f"to an authorized asset directory. Missing files: {', '.join(missing)}"
        )
    return root
