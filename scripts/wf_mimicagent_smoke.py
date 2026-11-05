"""Exercise real frontend data through existing SuperMimic BC, WM and BPTT."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

from supermimic.mimicagent.pipeline import write_json
from supermimic.wf.config import smoke_config
from supermimic.wf.train import Trainer


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--library", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    cfg = smoke_config()
    cfg.seed = 20261105
    cfg.output = str(a.output)
    cfg.iterations = 32
    cfg.cpu_threads = 2
    cfg.sim.arm_action_scale = None
    cfg.sim.arm_initial_position = (0.,) * 6
    cfg.sim.reset_settle_steps = 100
    cfg.demonstrations.directory = str(a.library.resolve())
    cfg.demonstrations.pretrain_updates = 2
    cfg.demonstrations.model_pretrain_updates = 2
    cfg.demonstrations.batch_size = 16
    write_json(a.output / "requested_config.json", asdict(cfg))
    trainer = Trainer(cfg)
    try:
        result = trainer.run()
        result.update(reference_clips=trainer.demonstrations.count,
            train_clips=len(trainer.demonstrations.training),
            validation_clips=len(trainer.demonstrations.validation),
            bc_updates=trainer.reference_pretrain_updates,
            model_pretrain_updates=trainer.reference_model_pretrain_updates,
            policy_improvement_measured=False,
            scope="data-loader/BC/WM/BPTT integration check; existing weights not loaded or changed")
        write_json(a.output / "integration_report.json", result)
    finally:
        trainer.sim.close()


if __name__ == "__main__":
    main()
