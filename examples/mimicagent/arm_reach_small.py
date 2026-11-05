"""Separate request: a larger shoulder/elbow motion and a stationary final hold."""
import numpy as np


def generate(context):
    n = context["frames"]
    phase = np.clip(np.arange(n) * context["dt"] / (context["duration_s"] * .65), 0., 1.)
    smooth = phase**3 * (10 - 15 * phase + 6 * phase**2)
    q = np.tile(context["initial_joint_position"], (n, 1))
    q[:, 17] += .28 * smooth
    q[:, 18] -= .22 * smooth
    return dict(base_position=np.tile(context["initial_base_pose"][:3], (n, 1)),
                base_quaternion=np.tile(context["initial_base_pose"][3:], (n, 1)),
                joint_position=q)
