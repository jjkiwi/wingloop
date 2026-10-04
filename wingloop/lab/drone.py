"""A quadrotor that flies on the fly's control laws and sees through its eye.

The airframe is a Crazyflie 2.x -- the parameters gym-pybullet-drones ships for
it -- simulated as a rigid body with four motors, first-order motor lag and
linear drag, in plain numpy so a mission runs anywhere.

What is the fly's is the controller, :class:`FlyPilot`:

- **attitude** is the haltere loop of :class:`wingloop.body.control.
  HaltereController` -- proportional on angle, derivative on body rate,
  critically damped at a stated bandwidth, with the gain derived as inertia
  times bandwidth squared over control authority rather than tuned;
- **altitude** is :class:`~wingloop.body.control.Throttle`'s law, the same
  critically damped second-order loop on height error, around hover thrust;
- **heading** is steered by vision, as the fly's is: the bearing of the most
  salient thing on the eye moves the heading setpoint, the way
  :class:`~wingloop.body.control.SteeringController` moves it;
- **what to do about an object** is the mushroom body's: an object it has
  learned to value is approached, one it has learned to fear is avoided, and
  a looming dark object it fears triggers an avoidance saccade.

The eye is the pretrained optic lobe of :mod:`.vision`. Speed over ground is
read from the airframe's state rather than from optic flow -- a stated
shortcut, the one a real drone takes with a flow sensor or GPS.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .scene import Camera, World, render, rotation

G = 9.81


@dataclass
class Airframe:
    """Crazyflie 2.x, '+' configuration: motors front, left, back, right."""

    mass: float = 0.027
    arm: float = 0.0397
    inertia: tuple = (1.4e-5, 1.4e-5, 2.17e-5)
    max_thrust: float = 0.149  # per motor, N (kf * max_rpm^2)
    torque_ratio: float = 0.0251  # yaw torque per newton of thrust, m (km / kf)
    motor_tau: float = 0.02
    drag: float = 0.01  # N per m/s, isotropic


@dataclass
class DroneState:
    position: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 1.0]))
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    attitude: np.ndarray = field(default_factory=lambda: np.zeros(3))  # roll, pitch, yaw
    rate: np.ndarray = field(default_factory=lambda: np.zeros(3))  # body rates
    motors: np.ndarray = field(default_factory=lambda: np.zeros(4))


def _euler_rates(attitude, rate):
    """Body rates to Euler-angle rates (ZYX)."""
    r, p, _ = attitude
    cr, sr, cp, tp = np.cos(r), np.sin(r), np.cos(p), np.tan(p)
    w = np.asarray(rate)
    return np.array(
        [
            w[0] + sr * tp * w[1] + cr * tp * w[2],
            cr * w[1] - sr * w[2],
            (sr * w[1] + cr * w[2]) / cp,
        ]
    )


class Quadrotor:
    def __init__(self, frame: Airframe | None = None, state: DroneState | None = None):
        self.frame = frame or Airframe()
        self.state = state or DroneState()
        hover = self.frame.mass * G / 4
        self.state.motors = np.full(4, hover)

    def mix(self, thrust: float, torque) -> np.ndarray:
        f = self.frame
        tx, ty, tz = torque
        c = f.torque_ratio
        cmd = np.array(
            [
                thrust / 4 - ty / (2 * f.arm) + tz / (4 * c),  # front
                thrust / 4 + tx / (2 * f.arm) - tz / (4 * c),  # left
                thrust / 4 + ty / (2 * f.arm) + tz / (4 * c),  # back
                thrust / 4 - tx / (2 * f.arm) - tz / (4 * c),  # right
            ]
        )
        return np.clip(cmd, 0.0, f.max_thrust)

    def step(self, command: np.ndarray, dt: float) -> None:
        f, s = self.frame, self.state
        s.motors = s.motors + (command - s.motors) * min(1.0, dt / f.motor_tau)
        front, left, back, right = s.motors
        thrust = s.motors.sum()
        torque = np.array(
            [
                f.arm * (left - right),
                f.arm * (back - front),
                f.torque_ratio * (front + back - left - right),
            ]
        )
        R = rotation(*s.attitude)
        accel = (R @ np.array([0.0, 0.0, thrust]) - f.drag * s.velocity) / f.mass
        accel[2] -= G
        inertia = np.asarray(f.inertia)
        w = s.rate
        wdot = (torque - np.cross(w, inertia * w)) / inertia
        s.velocity = s.velocity + accel * dt
        s.position = s.position + s.velocity * dt
        s.attitude = s.attitude + _euler_rates(s.attitude, w) * dt
        s.rate = w + wdot * dt
        if s.position[2] < 0.0:  # the ground
            s.position[2] = 0.0
            s.velocity[:] = 0.0


def wrap(angle: float) -> float:
    return float((angle + np.pi) % (2 * np.pi) - np.pi)


@dataclass
class FlyPilot:
    """The fly's three loops, on a quadrotor, plus vision and a mushroom body."""

    frame: Airframe = field(default_factory=Airframe)
    attitude_bandwidth: float = 25.0
    yaw_bandwidth: float = 8.0
    altitude_bandwidth: float = 3.0
    altitude: float = 1.5
    max_tilt: float = np.deg2rad(20.0)
    speed_gain: float = 0.25  # rad of tilt per m/s of speed error
    turn_rate_gain: float = 2.5  # rad/s of heading setpoint per unit bearing
    cruise: float = 1.0  # m/s

    def __post_init__(self):
        self.heading = 0.0

    def gains(self):
        """Inertia times bandwidth squared, and twice the bandwidth: the fly's formula.

        Authority is one here -- the controller commands torque and the mixer
        delivers it -- so the gains are the inertia's.
        """
        inertia = np.asarray(self.frame.inertia)
        bw = np.array([self.attitude_bandwidth, self.attitude_bandwidth, self.yaw_bandwidth])
        return inertia * bw**2, inertia * 2 * bw

    def control(self, drone: Quadrotor, forward_speed: float, turn: float, dt: float):
        s, f = drone.state, self.frame
        self.heading = wrap(self.heading + turn * dt)
        roll, pitch, yaw = s.attitude
        # Speed over ground in the heading frame.
        c, sn = np.cos(yaw), np.sin(yaw)
        forward = c * s.velocity[0] + sn * s.velocity[1]
        lateral = -sn * s.velocity[0] + c * s.velocity[1]
        pitch_cmd = np.clip(
            self.speed_gain * (forward_speed - forward), -self.max_tilt, self.max_tilt
        )
        roll_cmd = np.clip(self.speed_gain * lateral, -self.max_tilt, self.max_tilt)
        kp, kd = self.gains()
        error = np.array([roll_cmd - roll, pitch_cmd - pitch, wrap(self.heading - yaw)])
        torque = kp * error - kd * s.rate
        # Throttle: the fly's altitude law around hover, tilt-compensated.
        w = self.altitude_bandwidth
        accel = w**2 * (self.altitude - s.position[2]) - 2 * w * s.velocity[2]
        thrust = f.mass * (G + accel) / max(0.5, np.cos(roll) * np.cos(pitch))
        return drone.mix(thrust, torque)


@dataclass
class Memory:
    heading: float
    valence: float
    size: float
    features: np.ndarray = None  # the foveal features the mushroom body reads


@dataclass
class Mission:
    """Look around, judge what is there, go to the best thing, avoid the worst.

    Phases: ``scan`` (rotate in place, fixate each object that comes into the
    middle of the eye and ask the mushroom body what it is worth), ``approach``
    (fly to the most valued object, steering on its bearing), and ``arrived``.
    A disliked object that looms during the approach triggers a saccade away
    from it.
    """

    world: World
    eye: object
    mushroom_body: object = None
    pilot: FlyPilot = field(default_factory=FlyPilot)
    camera: Camera = field(default_factory=Camera)
    physics_dt: float = 0.002
    vision_dt: float = 0.04
    scan_rate: float = np.deg2rad(45.0)
    arrive_size: float = 0.12
    fixate_time: float = 0.4
    scan_only: bool = False

    def __post_init__(self):
        self.drone = Quadrotor(
            state=DroneState(position=np.array([0.0, 0.0, self.pilot.altitude]))
        )
        self.phase = "scan"
        self.memories: list[Memory] = []
        self.log: list[dict] = []
        self.t = 0.0
        self._scanned = 0.0
        self._fixating, self._valences, self._features = 0.0, [], []
        self._refractory = 0.0
        self._last_size = 0.0
        self._saccade = 0.0
        self.target_heading = None
        self.eye.reset()

    def _valence(self, activity) -> float:
        if self.mushroom_body is None:
            return 0.0
        return self.mushroom_body.valence(self.eye.fovea(activity))

    def _decide(self, activity, dt):
        """One visual step: returns (forward speed, turn rate)."""
        bearing, _ = self.eye.bearing(activity)
        size, dark_bearing = self.eye.dark()
        expansion = (size - self._last_size) / dt
        self._last_size = size
        yaw = self.drone.state.attitude[2]

        if self.phase == "scan":
            if self._scanned >= 2 * np.pi:
                if self.scan_only:
                    self.phase = "scanned"
                    return 0.0, 0.0
                liked = [m for m in self.memories if m.valence > 0]
                if not liked:
                    self.phase = "nothing worth approaching"
                    return 0.0, 0.0
                self.target_heading = max(liked, key=lambda m: m.valence).heading
                self.phase = "approach"
                return 0.0, 0.0
            if self._refractory > 0:
                # Just looked at something: keep turning without looking, the
                # way a fly's saccade carries its gaze past what it fixated.
                self._refractory -= self.scan_rate * dt
                self._scanned += self.scan_rate * dt
                return 0.0, self.scan_rate
            if size > 0.005 and abs(dark_bearing) < 0.15:
                self._fixating += dt
                self._valences.append(self._valence(activity))
                self._features.append(self.eye.fovea(activity))
                if self._fixating >= self.fixate_time:
                    half = len(self._valences) // 2
                    v = float(np.mean(self._valences[half:]))
                    f = np.mean(self._features[half:], axis=0)
                    if not any(
                        abs(wrap(m.heading - yaw)) < np.deg2rad(25) for m in self.memories
                    ):
                        self.memories.append(Memory(float(yaw), v, size, f))
                    self._fixating, self._valences, self._features = 0.0, [], []
                    self._refractory = np.deg2rad(40)
                    return 0.0, self.scan_rate
                return 0.0, self.pilot.turn_rate_gain * dark_bearing
            self._fixating, self._valences, self._features = 0.0, [], []
            self._scanned += self.scan_rate * dt
            return 0.0, self.scan_rate

        if self.phase == "approach":
            if self._saccade > 0:
                self._saccade -= dt
                return 0.6 * self.pilot.cruise, self._saccade_turn
            off = wrap(self.target_heading - yaw)
            if abs(off) > np.deg2rad(30):  # target not in the middle of the eye yet
                return 0.0, np.clip(3.0 * off, -1.5, 1.5)
            here = self._valence(activity)
            if size > 0.02 and expansion > 0.15 and here < 0 and abs(dark_bearing) > 0.05:
                # Looming and disliked: saccade away from it, as a fly does.
                self._saccade = 0.5
                self._saccade_turn = -np.sign(dark_bearing) * 2.0
                return 0.6 * self.pilot.cruise, self._saccade_turn
            if size >= self.arrive_size and abs(dark_bearing) < 0.3:
                self.phase = "arrived"
                return 0.0, 0.0
            self.target_heading = wrap(self.target_heading + 0.5 * dark_bearing * dt)
            return self.pilot.cruise, self.pilot.turn_rate_gain * dark_bearing

        return 0.0, 0.0

    def run(self, seconds: float = 20.0, stop_on_arrival: bool = True) -> list[dict]:
        per_vision = int(round(self.vision_dt / self.physics_dt))
        speed, turn = 0.0, 0.0
        steps = int(round(seconds / self.physics_dt))
        for i in range(steps):
            s = self.drone.state
            if i % per_vision == 0:
                image = render(self.world, s.position, s.attitude, self.camera)
                activity = self.eye.step(image, frames=int(round(self.vision_dt / self.eye.dt)))
                speed, turn = self._decide(activity, self.vision_dt)
                size, db = self.eye.dark()
                self.log.append(
                    dict(
                        t=self.t,
                        x=float(s.position[0]),
                        y=float(s.position[1]),
                        z=float(s.position[2]),
                        roll=float(s.attitude[0]),
                        pitch=float(s.attitude[1]),
                        yaw=float(s.attitude[2]),
                        phase=self.phase,
                        size=size,
                        bearing=db,
                        speed=speed,
                        turn=turn,
                        valence=self._valence(activity) if self.mushroom_body else 0.0,
                    )
                )
                if stop_on_arrival and self.phase in (
                    "arrived",
                    "nothing worth approaching",
                    "scanned",
                ):
                    break
            command = self.pilot.control(self.drone, speed, turn, self.physics_dt)
            self.drone.step(command, self.physics_dt)
            self.world.step(self.physics_dt)
            self.t += self.physics_dt
        return self.log

    def distances(self) -> dict:
        p = self.drone.state.position
        return {t.label: float(np.linalg.norm((t.position - p)[:2])) for t in self.world.things}


def place(kind: str, heading_deg: float, distance: float, label: str = "", **kw):
    """An object at a compass bearing and distance from the origin, at flight height."""
    from .scene import Thing

    h = np.deg2rad(heading_deg)
    xy = [distance * np.cos(h), distance * np.sin(h)]
    if kind == "bar":
        return Thing("bar", xy + [0.0], kw.pop("size", 0.3), label=label or kind, **kw)
    return Thing(kind, xy + [1.5], kw.pop("size", 0.6), label=label or kind, **kw)


def which(world: World, position, heading: float, tolerance: float = np.deg2rad(20)):
    """The object at this heading from here, as the experimenter sees it, or None."""
    best, err = None, tolerance
    for thing in world.things:
        d = thing.position[:2] - np.asarray(position)[:2]
        e = abs(wrap(np.arctan2(d[1], d[0]) - heading))
        if e < err:
            best, err = thing, e
    return best


def training_flights(eye, kinds, sessions: int = 12, seed: int = 0, distance=(3.5, 6.0)):
    """Fly the drone past one object at a time and record what it looked at.

    Each session puts a single object of a random kind at a random heading and
    distance, runs the scan, and keeps the features of every fixation together
    with the kind of object the experimenter knows was there. Fixations on
    nothing the experimenter can name are dropped. Returns (features, kinds).
    Rewarding them is the trainer's job: see :func:`teach`.
    """
    rng = np.random.default_rng(seed)
    feats, labels = [], []
    for _ in range(sessions):
        kind = kinds[rng.integers(len(kinds))]
        world = World([place(kind, rng.uniform(-180, 180), rng.uniform(*distance))])
        mission = Mission(world, eye, scan_only=True)
        mission.run(30.0)
        for m in mission.memories:
            thing = which(world, mission.drone.state.position, m.heading)
            if thing is not None:
                feats.append(m.features)
                labels.append(thing.kind)
    return np.asarray(feats), np.asarray(labels)


def teach(mushroom_body, features, kinds, reward: dict, epochs: int = 1, seed: int = 0):
    """Calibrate on the flights, then pair each fixation with its dopamine.

    Calibration is input normalisation from the same unrewarded experience;
    the dopamine is ``reward[kind]`` -- positive for the PAM (reward) cluster,
    negative for PPL1 (punishment).
    """
    mushroom_body.calibrate(features)
    rng = np.random.default_rng(seed)
    for _ in range(epochs):
        for i in rng.permutation(len(kinds)):
            mushroom_body.learn(features[i], reward.get(str(kinds[i]), 0.0))
