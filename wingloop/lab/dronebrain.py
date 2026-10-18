"""The fly's brain flying a drone: the same eye, mushroom body and navigator.

The drone is the Crazyflie of :mod:`.drone`, on the fly's control laws, in the
fly's arena made :data:`SCALE` times bigger -- a room about eight metres
across, a ball 0.8 m wide. Scaled evenly, a scene looks exactly the same from
a camera scaled with it, so what the drone's camera sees is, pixel for pixel,
what the fly's eye sees from the matching place. That is why nothing the fly
learned has to be learned again: the tethered views, the mushroom body, the
navigator and its thresholds are the fly's, unchanged.

:class:`DroneFlight` gives the drone the interface of the physical fly
(:class:`~.watch.Flight`), in the arena's units, so
:class:`~.autopilot.Autopilot` flies either. What differs is only the body:

- the navigator asks the fly for a pitch, which is how a fly sets its speed;
  the drone gets the speed that pitch would give, through its own speed loop
  (``cruise`` at the fly's cruising 12 degrees, nothing when it brakes);
- the drone turns more slowly than a fly -- its yaw comes from motor drag
  torque -- so it looks around at :data:`SCAN_RATE` rather than 240 deg/s;
- where the fly's flight state carries its flight neurons, the drone's
  carries its motors.

    flylab cel kula --dron
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET

import numpy as np

from .arena import FLOOR
from .drone import FlyPilot, Quadrotor, wrap
from .scene import rotation

#: Real millimetres per arena millimetre. The fly's arena, 50-80 mm to each
#: object, becomes 2.5-4 m; its 17 mm of flight height above the floor, 0.85 m.
SCALE = 50.0
#: How fast the drone looks around, degrees per second. Its yaw loop runs at 8
#: rad/s, so at the fly's 240 the heading lags the setpoint by about a radian
#: and the last part of every turn is never seen.
SCAN_RATE = 90.0
#: The navigator settings that differ for the drone: it has to stop, and to
#: stop turning, before its close look is worth anything; and once its close
#: look is sure, it can afford to go back and look again.
NAVIGATOR = {
    "scan_rate": SCAN_RATE,
    "settle_speed": 6.0,
    "centred": 5.0,
    "still": 20.0,
    "face_time": 3.0,
    "go_back": True,
    "explore": 40.0,
    "unsure": 0.06,
}


class DroneFlight:
    """A Crazyflie with the interface of :class:`~.watch.Flight`, in arena units."""

    who = "Dron"

    def __init__(
        self,
        height_mm: float = 5.0,
        scale: float = SCALE,
        cruise: float = 1.5,
        timestep: float = 0.001,
    ):
        self.scale = scale
        self.cruise = cruise  # m/s at the fly's cruising pitch of 12 degrees
        self.timestep = timestep
        self.drone = Quadrotor()
        self.drone.state.position = np.array([0.0, 0.0, self._metres(height_mm)])
        self.pilot = FlyPilot(altitude=self._metres(height_mm), cruise=cruise)
        self.speed = 0.0
        self._t = 0.0
        self._heading = 0.0  # unwrapped, radians
        self._yaw = 0.0
        self.knobs: dict = {}

    def _metres(self, z_arena: float) -> float:
        return (z_arena - FLOOR) * self.scale / 1000.0

    @property
    def t(self) -> float:
        return self._t

    def command(self, pitch_deg: float, heading_deg: float, height_mm: float) -> None:
        """The fly's setpoints: a pitch (read as a speed), a heading and a height."""
        self.speed = self.cruise * max(pitch_deg, 0.0) / 12.0
        self.pilot.heading = wrap(math.radians(heading_deg))
        self.pilot.altitude = self._metres(height_mm)

    def step(self, n: int = 1) -> None:
        d = self.drone
        for _ in range(n):
            motors = self.pilot.control(d, self.speed, 0.0, self.timestep)
            d.step(motors, self.timestep)
            self._t += self.timestep
            yaw = float(d.state.attitude[2])
            self._heading += wrap(yaw - self._yaw)
            self._yaw = yaw
        f = d.frame
        self.knobs = {"motors": (d.state.motors / f.max_thrust).tolist()}

    def pose(self) -> tuple[np.ndarray, np.ndarray]:
        """Where the camera is, in arena units, and which way it faces."""
        s = self.drone.state
        k = 1000.0 / self.scale
        position = np.array([s.position[0] * k, s.position[1] * k, s.position[2] * k + FLOOR])
        return position, rotation(*s.attitude)

    def state(self) -> dict:
        s = self.drone.state
        k = 1000.0 / self.scale
        roll, pitch, _ = s.attitude
        motors = s.motors / self.drone.frame.max_thrust
        return dict(
            t=self._t,
            x=float(s.position[0] * k),
            y=float(s.position[1] * k),
            z=float(s.position[2] * k + FLOOR),
            vx=float(s.velocity[0] * k),
            vy=float(s.velocity[1] * k),
            # Real millimetres per second, for people; the rest is arena units.
            speed=float(np.hypot(s.velocity[0], s.velocity[1]) * 1000.0),
            climb=float(s.velocity[2] * 1000.0),
            pitch=float(np.degrees(pitch)),
            roll=float(np.degrees(roll)),
            heading=float(np.degrees(self._heading)),
            motors=[round(float(m), 3) for m in motors],
            altitude_m=float(s.position[2]),
        )

    def rates(self) -> np.ndarray:
        """Body rates, rad/s: the drone's gyroscope, its haltere."""
        return np.asarray(self.drone.state.rate, dtype=float)

    def upright(self) -> bool:
        s = self.drone.state
        roll, pitch, _ = np.degrees(s.attitude)
        return abs(roll) < 60 and abs(pitch) < 60 and s.position[2] > 0.05


# ------------------------------------------------------------------ drawing


def _scene_xml(pretty: bool, items, scale: float) -> str:
    """The fly's scenery and a Crazyflie drawn to scale, as one MJCF document."""
    from .watch import add_scenery

    root = ET.Element("mujoco", model="flylab_drone")
    ET.SubElement(root, "option", gravity="0 0 0")
    ET.SubElement(root, "worldbody")
    add_scenery(root, pretty, items)
    world = root.find("worldbody")
    # The airframe in arena units: 92 mm across the motors becomes 1.8.
    k = 1000.0 / scale
    arm = 0.0397 * k
    body = ET.SubElement(world, "body", name="drone", mocap="true", pos=f"0 0 {5.0}")
    ET.SubElement(
        body,
        "geom",
        type="box",
        size=f"{0.35 * arm} {0.35 * arm} {0.1 * arm}",
        rgba="0.15 0.15 0.17 1",
        contype="0",
        conaffinity="0",
    )
    for i, (dx, dy) in enumerate(((1, 0), (0, 1), (-1, 0), (0, -1))):
        # The '+' frame of :class:`~.drone.Airframe`: motors front, left, back, right.
        ET.SubElement(
            body,
            "geom",
            type="capsule",
            fromto=f"0 0 0 {dx * arm} {dy * arm} 0",
            size=f"{0.06 * arm}",
            rgba="0.2 0.2 0.22 1",
            contype="0",
            conaffinity="0",
        )
        ET.SubElement(
            body,
            "geom",
            type="cylinder",
            pos=f"{dx * arm} {dy * arm} {0.12 * arm}",
            size=f"{0.55 * arm} {0.015 * arm}",
            rgba=("0.9 0.3 0.2 0.55" if i == 0 else "0.75 0.8 0.85 0.45"),
            contype="0",
            conaffinity="0",
        )
    return ET.tostring(root, encoding="unicode")


class DroneView:
    """The drone in the arena, drawn where the simulation has it."""

    def __init__(self, pretty: bool = False, items: tuple = (), scale: float = SCALE):
        import mujoco

        self._mj = mujoco
        self.model = mujoco.MjModel.from_xml_string(_scene_xml(pretty, tuple(items), scale))
        self.data = mujoco.MjData(self.model)
        self.root_body = self.model.body("drone").id
        mujoco.mj_forward(self.model, self.data)

    def show(self, flight: DroneFlight) -> None:
        position, R = flight.pose()
        quat = np.zeros(4)
        self._mj.mju_mat2Quat(quat, R.ravel())
        self.data.mocap_pos[0] = position
        self.data.mocap_quat[0] = quat
        self._mj.mj_forward(self.model, self.data)

    def chase(self, cam, heading_deg: float, distance: float = 14.0) -> None:
        """A camera behind and above the drone, turning with it."""
        cam.type = self._mj.mjtCamera.mjCAMERA_FREE
        cam.lookat[:] = self.data.xpos[self.root_body]
        cam.distance = distance
        cam.azimuth = heading_deg
        cam.elevation = -18.0


__all__ = ["NAVIGATOR", "SCALE", "SCAN_RATE", "DroneFlight", "DroneView"]
