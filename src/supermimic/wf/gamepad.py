"""Read a Linux gamepad and generate task commands, never motor commands.

Layout: GameSir Nova 2 Lite / Xbox joydev mapping. The packet contains a
body-velocity request and a world TCP goal transported from a body-frame goal.
It must be consumed by one command-conditioned whole-body Actor.
"""
from __future__ import annotations

import argparse
import array
from dataclasses import dataclass
import fcntl
import json
import os
import struct
import time

import numpy as np


class LinuxGamepad:
    def __init__(self, path="/dev/input/js0"):
        self.path = path
        self.fd = None
        self.name = "disconnected"
        self.axes = np.zeros(8)
        self.axes[[2, 5]] = -1.
        self.buttons = np.zeros(11, dtype=bool)
        self.pressed = set()
        self.connected = False
        self.last_error = None
        self._next_reconnect = 0.
        self.axis_codes = []
        self.button_codes = []
        self._open()

    def _open(self):
        try:
            fd = os.open(self.path, os.O_RDONLY | os.O_NONBLOCK | os.O_CLOEXEC)
            self.fd = fd
            name = bytearray(128)
            fcntl.ioctl(fd, 0x80806A13, name)
            axes, buttons = array.array("B", [0]), array.array("B", [0])
            fcntl.ioctl(fd, 0x80016A11, axes, True)
            fcntl.ioctl(fd, 0x80016A12, buttons, True)
            amap, bmap = array.array("B", [0]*64), array.array("H", [0]*512)
            fcntl.ioctl(fd, 0x80406A32, amap, True)
            fcntl.ioctl(fd, 0x84006A34, bmap, True)
            self.axis_codes = list(amap[:axes[0]])
            self.button_codes = list(bmap[:buttons[0]])
            if self.axis_codes[:6] != [0, 1, 2, 3, 4, 5] or self.button_codes[:2] != [304, 305]:
                raise ValueError("unsupported gamepad axis/button map; inspect --describe before use")
            self.name = name.split(b"\0", 1)[0].decode(errors="replace")
            self.axes = np.zeros(max(8, axes[0]))
            self.axes[[2, 5]] = -1.
            self.buttons = np.zeros(max(11, buttons[0]), dtype=bool)
            self.connected = True
            self.last_error = None
        except (OSError, ValueError) as error:
            self.last_error = str(error)
            self.close()

    def feed(self, data):
        if len(data) % 8:
            raise ValueError("joydev events must be whole eight-byte packets")
        for offset in range(0, len(data), 8):
            _, value, kind, index = struct.unpack("IhBB", data[offset:offset+8])
            initial = bool(kind & 0x80)
            kind &= 0x7f
            if kind == 1 and index < len(self.buttons):
                if value and not self.buttons[index] and not initial:
                    self.pressed.add(index)
                self.buttons[index] = bool(value)
            elif kind == 2 and index < len(self.axes):
                self.axes[index] = np.clip(value / 32767., -1., 1.)

    def poll(self):
        self.pressed.clear()
        if self.fd is None:
            if time.monotonic() >= self._next_reconnect:
                self._next_reconnect = time.monotonic() + 1.
                self._open()
            if self.fd is None:
                return self
        try:
            while True:
                try:
                    data = os.read(self.fd, 4096)
                except BlockingIOError:
                    break
                if not data:
                    raise OSError("gamepad disconnected")
                self.feed(data)
        except OSError as error:
            self.last_error = str(error)
            self.close()
        return self

    def describe(self):
        return dict(path=self.path, name=self.name, connected=self.connected,
                    axis_codes=self.axis_codes, button_codes=self.button_codes,
                    axes=self.axes.tolist(), buttons=self.buttons.tolist(), error=self.last_error)

    def close(self):
        if self.fd is not None:
            os.close(self.fd)
        self.fd = None
        self.connected = False
        self.buttons[:] = False
        self.axes[:] = 0.
        self.axes[[2, 5]] = -1.


def quat_mul(a, b):
    return np.r_[a[0]*b[0]-np.dot(a[1:], b[1:]),
                 a[0]*b[1:]+b[0]*a[1:]+np.cross(a[1:], b[1:])]


def quat_rotate(q, v):
    return v + 2*np.cross(q[1:], np.cross(q[1:], v)+q[0]*v)


def rotvec_quat(v):
    angle = np.linalg.norm(v)
    return np.r_[np.cos(angle/2), v*(np.sin(angle/2)/angle if angle > 1e-9 else .5)]


def relative_pose(base, pose):
    inverse = base[3:].copy()
    inverse[1:] *= -1
    return np.r_[quat_rotate(inverse, pose[:3]-base[:3]), quat_mul(inverse, pose[3:])]


def world_pose(base, relative):
    return np.r_[base[:3]+quat_rotate(base[3:], relative[:3]), quat_mul(base[3:], relative[3:])]


def task_reference_base(base, frame="body"):
    base=np.asarray(base,dtype=float).copy()
    if frame=="body":return base
    if frame!="heading":raise ValueError("goal following frame must be body or heading")
    w,x,y,z=base[3:]/np.linalg.norm(base[3:])
    yaw=np.arctan2(2*(w*z+x*y),1-2*(y*y+z*z))
    return np.r_[base[:2],0.,np.cos(yaw/2),0.,0.,np.sin(yaw/2)]


@dataclass
class CommandPacket:
    world_goal: np.ndarray
    base_velocity: np.ndarray
    gripper_opening: float
    reset_robot: bool = False
    quit: bool = False
    mode: str = "base_and_arm"
    enabled: bool = False
    goal_mode: int = 1


class GoalTeleop:
    def __init__(self, base_pose, tcp_pose, *, home_pose=None, work_pose=None,
                 translation_speed=.08, rotation_speed=.35, base_speed=.08,
                 lateral_speed=.02, yaw_speed=.10, goal_radius=1.15,
                 follow_frame="body", stop_world_fixed=False):
        self.follow_frame=follow_frame
        self.stop_world_fixed=stop_world_fixed
        self.reference=task_reference_base(base_pose,follow_frame)
        self._base_moving=False
        self.initial = relative_pose(self.reference, np.asarray(tcp_pose))
        self.home = self.initial.copy() if home_pose is None else np.asarray(home_pose).copy()
        self.work = self.initial.copy() if work_pose is None else np.asarray(work_pose).copy()
        self.target = self.initial.copy()
        self.translation_speed, self.rotation_speed = translation_speed, rotation_speed
        self.base_speed, self.lateral_speed, self.yaw_speed = base_speed, lateral_speed, yaw_speed
        self.goal_radius = goal_radius
        self.arm_only = False
        self.opening = .04
        self._preset = None
        self._home_requested = False

    @staticmethod
    def axis(value):
        return float(np.sign(value)*max(0., (abs(value)-.12)/.88))

    def update(self, pad, base_pose, tcp_pose, dt):
        if not 0 < dt <= .1:
            raise ValueError("teleop requires a control timestep in (0, 0.1]")
        b, axes = pad.buttons, pad.axes
        if 3 in pad.pressed:  # Y toggles the two requested layouts.
            self.arm_only = not self.arm_only
        reset = 7 in pad.pressed
        if reset:
            self.target = self.initial.copy()
            self.opening = .04
            self._preset = None
            self._home_requested = False
        if 0 in pad.pressed:  # A: initial TCP preset; never writes joint state.
            self._preset = [self.target.copy(), self.home.copy(), 0.]
            self._home_requested = True
            self.opening = .04
        if pad.connected and b[4] and 10 in pad.pressed:  # LB + R3.
            self._preset = [self.target.copy(), self.work.copy(), 0.]
            self._home_requested = False
        enabled = bool(pad.connected and b[4] and not b[1])
        velocity = np.zeros(3)
        position_rate = np.zeros(3)
        rotation_rate = np.zeros(3)
        x, y, rx, ry = (self.axis(axes[i]) for i in (0, 1, 3, 4))
        hx, hy = (self.axis(axes[i]) for i in (6, 7))
        if enabled:
            if self.arm_only:
                speed = .25 if b[5] else 1.
                position_rate[:] = [-y, -x, -hy]
                rotation_rate[:] = [rx, -ry, -hx]
                position_rate *= speed
                rotation_rate *= speed
            else:
                velocity[0] = -y*self.base_speed
                if b[2]:  # X + horizontal left stick requests lateral movement.
                    velocity[1] = -x*self.lateral_speed
                else:
                    velocity[2] = -x*self.yaw_speed
                position_rate[2] = -hy
                if b[5]:
                    rotation_rate[:] = [rx, -ry, -hx]
                else:
                    position_rate[:2] = [-ry, -rx]
            self.opening = float(np.clip(self.opening + .04*dt*(axes[5]-axes[2])/2, 0., .1))
        if b[1] or not pad.connected:
            velocity[:] = 0.
        if self._preset is not None and pad.connected and not b[1] and (enabled or self._home_requested):
            start, end, elapsed = self._preset
            elapsed = min(1.5, elapsed+dt)
            phase = elapsed/1.5
            weight = phase**3*(10-15*phase+6*phase**2)
            qend = end[3:] if np.dot(start[3:], end[3:]) >= 0 else -end[3:]
            quaternion = (1-weight)*start[3:]+weight*qend
            quaternion /= np.linalg.norm(quaternion)
            self.target = np.r_[(1-weight)*start[:3]+weight*end[:3], quaternion]
            self._preset = None if elapsed >= 1.5 else [start, end, elapsed]
            velocity[:] = 0.
        elif self._preset is not None:
            velocity[:] = 0.
        elif enabled:
            self.target[:3] += dt*self.translation_speed*position_rate
            quaternion = quat_mul(rotvec_quat(dt*self.rotation_speed*rotation_rate), self.target[3:])
            self.target[3:] = quaternion/np.linalg.norm(quaternion)
        delta = self.target[:3]-self.initial[:3]
        if np.linalg.norm(delta) > self.goal_radius:
            self.target[:3] = self.initial[:3]+delta*self.goal_radius/np.linalg.norm(delta)
        moving=bool(np.any(np.abs(velocity)>1e-8))
        if self.stop_world_fixed:
            if moving:
                reference=task_reference_base(base_pose,self.follow_frame)
                if not self._base_moving:
                    self.target=relative_pose(reference,world_pose(self.reference,self.target))
                self.reference=reference
            elif self._base_moving:
                # Include the last executed moving step, then retain this world
                # target while body posture or autonomous base correction varies.
                self.reference=task_reference_base(base_pose,self.follow_frame)
            self._base_moving=moving
        else:
            self.reference=task_reference_base(base_pose,self.follow_frame)
        return CommandPacket(world_goal=world_pose(self.reference, self.target),
            base_velocity=velocity, gripper_opening=self.opening, reset_robot=reset,
            quit=6 in pad.pressed, mode="arm_only" if self.arm_only else "base_and_arm", enabled=enabled,
            goal_mode=int(moving) if self.stop_world_fixed else 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="/dev/input/js0")
    parser.add_argument("--describe", action="store_true")
    parser.add_argument("--seconds", type=float, default=10.)
    args = parser.parse_args()
    pad = LinuxGamepad(args.device)
    try:
        pad.poll()
        print(json.dumps(pad.describe()), flush=True)
        if args.describe:
            return
        until = time.monotonic()+args.seconds
        while time.monotonic() < until:
            pad.poll()
            if pad.pressed or np.abs(pad.axes[[0, 1, 3, 4]]).max() > .12:
                print(json.dumps(pad.describe()), flush=True)
            time.sleep(.02)
    finally:
        pad.close()


if __name__ == "__main__":
    main()
