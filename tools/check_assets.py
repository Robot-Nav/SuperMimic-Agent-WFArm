"""Validate the presence of privately supplied WF-Arm resources."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from supermimic.assets import resolve_wfarm_assets


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--asset-dir")
    p.add_argument("--fixed-arm", action="store_true")
    args = p.parse_args()
    root = resolve_wfarm_assets(args.asset_dir)
    report = dict(required_resource_files_present=True,
                  fixed_arm_limits_present=(root / "config/operating_limits.json").is_file(),
                  geometry_or_calibration_printed=False)
    if args.fixed_arm:
        limits = json.loads((root / "config/operating_limits.json").read_text())
        if float(limits["arm_joint5_lower_rad"]) >= float(limits["arm_joint5_upper_rad"]):
            raise ValueError("invalid private operating-limit interval")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
