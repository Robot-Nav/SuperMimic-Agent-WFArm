"""Requested .5 m/s wheeled travel plus small arm unfolding, followed by a stop.

This is a kinematic proposal. Actual motor replay decides whether it is usable.
"""
import numpy as np


def generate(context):
    n, dt = context["frames"], context["dt"]
    t = np.arange(n) * dt
    u = np.clip(t / (context["duration_s"] * .65), 0., 1.)
    smooth = u**3 * (10 - 15 * u + 6 * u**2)
    v = .5 * np.minimum(np.clip(t / .6, 0, 1), np.clip((context["duration_s"] * .65 - t) / .6, 0, 1))
    distance = np.r_[0., np.cumsum((v[1:] + v[:-1]) * dt / 2)]
    q = np.tile(context["initial_joint_position"], (n, 1))
    q[:, 17] += .18 * smooth; q[:, 18] -= .15 * smooth
    q[:, 12:16] += distance[:, None] * np.array([-1, 1, -1, 1]) / context["wheel_radius_m"]
    pos = np.tile(context["initial_base_pose"][:3], (n, 1)); pos[:, 0] += distance
    return dict(base_position=pos,
                base_quaternion=np.tile(context["initial_base_pose"][3:], (n, 1)), joint_position=q)
