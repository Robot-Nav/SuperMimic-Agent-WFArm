"""Process boundary for an authored/generated trajectory program."""
from __future__ import annotations

import json
from pathlib import Path
import runpy
import sys

import numpy as np


def main():
    program, context, output = map(Path, sys.argv[1:])
    generate = runpy.run_path(str(program))["generate"]
    arrays = generate(json.loads(context.read_text()))
    required = {"base_position", "base_quaternion", "joint_position"}
    if not isinstance(arrays, dict) or set(arrays) != required:
        raise ValueError(f"generate(context) must return exactly {sorted(required)}")
    values = {k: np.asarray(v, dtype=np.float64) for k, v in arrays.items()}
    np.savez_compressed(output, **values)


if __name__ == "__main__":
    main()
