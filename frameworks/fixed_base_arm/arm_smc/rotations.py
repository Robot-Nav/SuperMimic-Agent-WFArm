"""Small differentiable quaternion operations (MuJoCo's scalar-first convention).

All operations accept arbitrary batch dimensions and remain finite at identity.
No simulator or mjlab dependency is needed by the learned controller.
"""

from __future__ import annotations

import math

import torch


def normalize(v: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return v / torch.linalg.vector_norm(v, dim=-1, keepdim=True).clamp_min(eps)


def quat_normalize(q: torch.Tensor) -> torch.Tensor:
    identity = torch.zeros_like(q)
    identity[..., 0] = 1
    norm = torch.linalg.vector_norm(q, dim=-1, keepdim=True)
    return torch.where(norm > 1e-8, q / norm.clamp_min(1e-8), identity)


def quat_conjugate(q: torch.Tensor) -> torch.Tensor:
    return torch.cat((q[..., :1], -q[..., 1:]), -1)


quat_inv = quat_conjugate


def quat_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    aw, av, bw, bv = a[..., :1], a[..., 1:], b[..., :1], b[..., 1:]
    return torch.cat((aw * bw - (av * bv).sum(-1, keepdim=True),
                      aw * bv + bw * av + torch.cross(av, bv, dim=-1)), -1)


def quat_apply(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    q, v = torch.broadcast_tensors(q, torch.cat((v[..., :1], v), -1))
    v = v[..., 1:]
    xyz = q[..., 1:]
    t = 2 * torch.cross(xyz, v, dim=-1)
    return v + q[..., :1] * t + torch.cross(xyz, t, dim=-1)


def quat_apply_inverse(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    return quat_apply(quat_conjugate(q), v)


def quat_from_rotvec(v: torch.Tensor) -> torch.Tensor:
    # sqrt has an undefined derivative at zero; the tiny interior offset avoids it.
    theta = torch.sqrt(v.square().sum(-1, keepdim=True) + 1e-16)
    return torch.cat((torch.cos(theta / 2), 0.5 * torch.sinc(theta / (2 * math.pi)) * v), -1)


def quat_from_angle_axis(angle: torch.Tensor, axis: torch.Tensor) -> torch.Tensor:
    return quat_from_rotvec(angle[..., None] * axis)


def quat_error(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Shortest orientation difference from a to b, represented as a quaternion."""
    q = quat_normalize(quat_mul(quat_conjugate(quat_normalize(a)), quat_normalize(b)))
    return torch.where(q[..., :1] < 0, -q, q)


def quat_angle(q: torch.Tensor) -> torch.Tensor:
    """Geodesic angle in [0, pi], with a finite derivative at identity."""
    q = quat_normalize(q)
    n = torch.sqrt(q[..., 1:].square().sum(-1) + 1e-16) - 1e-8
    return 2 * torch.atan2(n.clamp_min(0), q[..., 0].abs().clamp_min(1e-8))


def rotvec_from_quat(q: torch.Tensor) -> torch.Tensor:
    q = quat_normalize(q)
    q = torch.where(q[..., :1] < 0, -q, q)
    n = torch.sqrt(q[..., 1:].square().sum(-1, keepdim=True) + 1e-16)
    scale = 2 * torch.atan2(n, q[..., :1].clamp_min(1e-8)) / n
    return scale*q[..., 1:]


def heading_quat(q: torch.Tensor) -> torch.Tensor:
    q = quat_normalize(q)
    w, x, y, z = q.unbind(-1)
    yaw = torch.atan2(2*(w*z+x*y), 1-2*(y*y+z*z))
    zero = torch.zeros_like(yaw)
    return torch.stack((torch.cos(yaw/2), zero, zero, torch.sin(yaw/2)), -1)


def following_frame(state: torch.Tensor, frame: str = "body"):
    """Pose carrying a moving task: legacy body SE(3), or planar XY/Yaw.

    Planar following leaves world height and task Roll/Pitch available for the
    leg/arm policy to achieve, instead of moving the target with body posture.
    """
    q = quat_normalize(state[..., 3:7])
    if frame == "body":
        return state[..., :3], q
    if frame != "heading":
        raise ValueError("following frame must be body or heading")
    return torch.cat((state[..., :2], torch.zeros_like(state[..., 2:3])), -1), heading_quat(q)


def following_velocity(state: torch.Tensor, frame: str = "body"):
    """World linear/angular velocity of the selected task frame."""
    q = quat_normalize(state[..., 3:7])
    if frame == "body":
        return state[..., 29:32], quat_apply(q, state[..., 32:35])
    if frame != "heading":
        raise ValueError("following frame must be body or heading")
    w, x, y, z = q.unbind(-1)
    # ZYX yaw rate is not generally the body's z angular-velocity component.
    sin_pitch = 2 * (w*y-z*x)
    yaw_rate = (2*(y*z+w*x)*state[..., 33] +
                (1-2*(x*x+y*y))*state[..., 34]) / (1-sin_pitch.square()).clamp_min(1e-6)
    zero = torch.zeros_like(yaw_rate)
    return (torch.cat((state[..., 29:31], zero[..., None]), -1),
            torch.stack((zero, zero, yaw_rate), -1))


def quat_to_matrix(q: torch.Tensor) -> torch.Tensor:
    q = quat_normalize(q)
    w, x, y, z = q.unbind(-1)
    return torch.stack((1 - 2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y),
                        2*(x*y+w*z), 1 - 2*(x*x+z*z), 2*(y*z-w*x),
                        2*(x*z-w*y), 2*(y*z+w*x), 1 - 2*(x*x+y*y)), -1).reshape(*q.shape[:-1], 3, 3)


def rot6d(q: torch.Tensor) -> torch.Tensor:
    """First two matrix columns, flattened column by column."""
    return quat_to_matrix(q)[..., :, :2].transpose(-1, -2).reshape(*q.shape[:-1], 6)


def projected_gravity(q: torch.Tensor) -> torch.Tensor:
    g = torch.zeros_like(q[..., 1:])
    g[..., 2] = -1
    return quat_apply_inverse(q, g)


# Familiar aliases for callers that share the original SuperMimic terminology.
rot6d_from_quat = rot6d
matrix_from_quat = quat_to_matrix
