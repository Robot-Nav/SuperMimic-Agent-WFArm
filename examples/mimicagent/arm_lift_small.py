"""Coding-agent proposal: unfold slightly while maintaining four-wheel support."""
import numpy as np


def generate(context):
    n = context["frames"]
    t = np.arange(n) * context["dt"]
    phase = np.clip(t / (context["duration_s"] * .65), 0., 1.)
    smooth = phase**3 * (10 - 15 * phase + 6 * phase**2)
    q = np.tile(context["initial_joint_position"], (n, 1))
    q[:, 17] += .18 * smooth
    q[:, 18] -= .15 * smooth
    return dict(base_position=np.tile(context["initial_base_pose"][:3], (n, 1)),
                base_quaternion=np.tile(context["initial_base_pose"][3:], (n, 1)),
                joint_position=q)
