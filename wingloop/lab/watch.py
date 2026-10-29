"""Watch the fly fly: a 3-D view of the physical simulation, and a pilot's seat.

Two models, on purpose. The **physics** runs on the same rigid NeuroMechFly
the rest of ``wingloop.body`` has been validated on -- blade-element wings,
haltere loop, throttle through the connectome's power channel. The **view**
is a second copy of the fly with its wing joints left in, posed every frame
from the physics: the body where the physics put it, each wing at the stroke,
deviation and rotation angle the stroke generator commanded. The hinged model
is never stepped. Stepping it diverges on the first step (massless wing
joints at a 20 microsecond timestep), and nothing about what you see should
depend on a model that has not been checked.

Controls in the window (the console prints the flight state and the flight
neurons as it goes):

    Up / Down        fly faster / slower (pitch the body nose-down / up)
    Left / Right     turn left / right by 15 degrees
    W / S            climb / descend 3 mm
    Space            pause
    [ / ]            slower / faster slow motion
    H                back to hovering
    C                camera: follow the fly / free (drag with the mouse)
    B                optic lobe on / off (off flies faster)

A page in the browser shows the brain as it flies: what the fly sees, its
optic lobe (45,669 neurons of the pretrained flyvis model, one dot per
column), the haltere signals, the descending and motor neurons, and the wing
commands. It is served from this computer only (127.0.0.1).

The fly is a few millimetres long and beats its wings 218 times a second, so
what you see is slow motion: physics this detailed runs 30-100 times slower
than real time, and the console says how much.
"""

from __future__ import annotations

import tempfile
import threading
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np

from .flysim import FIXTURES, Brain, _models

#: The two flight neurons shown on screen, as :class:`~.flysim.Brain` names them.
DRIVE = "DNa08/DNp31 drive"
POWER = "power MNs (DLMn, DVMn)"

#: Where the scenery sits, mm. The fly starts at the origin.
FLOOR = -12.0
#: Columns standing around the arena, for something to fly past: (x, y, rgb).
COLUMNS = [
    (60, 25, "0.85 0.35 0.25"),
    (120, -30, "0.25 0.55 0.85"),
    (180, 40, "0.95 0.75 0.2"),
    (240, -10, "0.4 0.75 0.35"),
    (90, 90, "0.7 0.3 0.7"),
    (30, -70, "0.9 0.5 0.1"),
    (-40, 40, "0.3 0.6 0.6"),
    (200, 120, "0.85 0.35 0.25"),
    (300, 60, "0.25 0.55 0.85"),
    (150, -110, "0.95 0.75 0.2"),
    (-60, -40, "0.4 0.75 0.35"),
    (330, -60, "0.7 0.3 0.7"),
]


def _decorate(source: Path, destination: Path, pretty: bool = False, items=()) -> Path:
    """Add a sky, a chequered floor, light and columns to the hinged fly.

    Shadows are the expensive part of drawing this scene, so they are off
    unless ``pretty`` is asked for; nothing else about the picture changes.
    """
    tree = ET.parse(source)
    add_scenery(tree.getroot(), pretty, items)
    tree.write(destination)
    return destination


def add_scenery(root: ET.Element, pretty: bool = False, items=()) -> None:
    """The sky, the chequered floor, the light, the columns and the mission objects."""
    asset = root.find("asset")
    if asset is None:
        asset = ET.SubElement(root, "asset")
    ET.SubElement(
        asset,
        "texture",
        name="flylab_sky",
        type="skybox",
        builtin="gradient",
        rgb1="0.55 0.75 0.95",
        rgb2="0.95 0.97 1",
        width="256",
        height="256",
    )
    ET.SubElement(
        asset,
        "texture",
        name="flylab_floor",
        type="2d",
        builtin="checker",
        rgb1="0.42 0.55 0.32",
        rgb2="0.55 0.68 0.42",
        width="256",
        height="256",
    )
    ET.SubElement(
        asset,
        "material",
        name="flylab_floor",
        texture="flylab_floor",
        texrepeat="60 60",
        reflectance="0",
    )
    world = root.find("worldbody")
    ET.SubElement(
        world,
        "light",
        name="flylab_sun",
        directional="true",
        pos="0 0 300",
        dir="-0.3 0.2 -1",
        diffuse="0.9 0.9 0.85",
        castshadow="true" if pretty else "false",
    )
    ET.SubElement(
        world,
        "geom",
        name="flylab_floor",
        type="plane",
        size="600 600 1",
        pos=f"0 0 {FLOOR}",
        material="flylab_floor",
        contype="0",
        conaffinity="0",
    )
    for i, (x, y, rgb) in enumerate(COLUMNS):
        ET.SubElement(
            world,
            "geom",
            name=f"flylab_column{i}",
            type="cylinder",
            size="3 30",
            pos=f"{x} {y} {FLOOR + 30}",
            rgba=f"{rgb} 1",
            contype="0",
            conaffinity="0",
        )
    visual = root.find("visual")
    if visual is None:
        visual = ET.SubElement(root, "visual")
    for tag, attrs in (
        ("map", {"znear": "0.01", "zfar": "200"}),
        ("global", {"offwidth": "1280", "offheight": "720"}),
        ("quality", {"shadowsize": "4096" if pretty else "0"}),
    ):
        element = visual.find(tag)
        if element is None:
            element = ET.SubElement(visual, tag)
        element.attrib.update(attrs)
    from .arena import add_to_mjcf

    add_to_mjcf(world, items)


@lru_cache(maxsize=8)
def _visual_model_path(pretty: bool = False, items: tuple = ()) -> Path:
    from ..body.hinge import add_free_base, add_wing_hinges, neuromechfly_model, tuck_legs

    d = Path(tempfile.mkdtemp(prefix="flylab-view-"))
    rigid, _ = tuck_legs(neuromechfly_model(), d / "rigid.xml", keep="__none__")
    hinged = add_wing_hinges(rigid, d / "hinged.xml")
    free = add_free_base(getattr(hinged, "path", hinged), d / "free.xml", dofs="free")
    return _decorate(Path(free), d / "scene.xml", pretty, items)


class FlyView:
    """The posed copy of the fly that gets drawn."""

    def __init__(self, pretty: bool = False, items: tuple = ()):
        import mujoco

        self._mj = mujoco
        self.model = mujoco.MjModel.from_xml_path(str(_visual_model_path(pretty, tuple(items))))
        self.data = mujoco.MjData(self.model)
        root = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "free_FlyBody")
        self.root_qpos = int(self.model.jnt_qposadr[root])
        self.root_body = int(self.model.jnt_bodyid[root])
        self.wing_qpos = {}
        for j in range(self.model.njnt):
            name = self.model.joint(j).name
            if "Wing" in name:
                self.wing_qpos[name] = int(self.model.jnt_qposadr[j])

    def pose(self, body) -> None:
        """Put the drawn fly where the physics fly is, wings and all."""
        i = body.root_dof
        q = body.data.qpos
        self.data.qpos[self.root_qpos : self.root_qpos + 7] = q[i : i + 7]
        for name, angle in body.wing_angles.items():
            if name in self.wing_qpos:
                self.data.qpos[self.wing_qpos[name]] = float(angle)
        self._mj.mj_forward(self.model, self.data)

    def show(self, flight) -> None:
        """Pose the drawn fly from a :class:`Flight`, or from the fast fly."""
        if hasattr(flight, "body"):
            self.pose(flight.body)
            return
        position, R = flight.pose()
        quat = np.zeros(4)
        self._mj.mju_mat2Quat(quat, R.ravel())
        self.data.qpos[self.root_qpos : self.root_qpos + 3] = position
        self.data.qpos[self.root_qpos + 3 : self.root_qpos + 7] = quat
        for name, angle in flight.wing_angles().items():
            if name in self.wing_qpos:
                self.data.qpos[self.wing_qpos[name]] = float(angle)
        self._mj.mj_forward(self.model, self.data)

    def chase(self, cam, heading_deg: float, distance: float = 14.0) -> None:
        """A camera behind and above the fly, turning with it."""
        cam.type = self._mj.mjtCamera.mjCAMERA_FREE
        cam.lookat[:] = self.data.xpos[self.root_body]
        cam.distance = distance
        cam.azimuth = heading_deg
        cam.elevation = -18.0


@dataclass
class Pilot:
    """What the person at the controls is asking for."""

    pitch_deg: float = 0.0
    heading_deg: float = 0.0
    height_mm: float = 5.0
    paused: bool = False
    follow: bool = True
    brain: bool = True
    frame_every: int = 50  # physics steps (of 20 us) between drawn frames
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def key(self, keycode: int) -> None:
        with self.lock:
            if keycode == 265:  # up
                self.pitch_deg = min(self.pitch_deg + 3.0, 20.0)
            elif keycode == 264:  # down
                self.pitch_deg = max(self.pitch_deg - 3.0, -6.0)
            elif keycode == 263:  # left
                self.heading_deg += 15.0
            elif keycode == 262:  # right
                self.heading_deg -= 15.0
            elif keycode == 87:  # W
                self.height_mm += 3.0
            elif keycode == 83:  # S
                self.height_mm -= 3.0
            elif keycode == 32:  # space
                self.paused = not self.paused
            elif keycode == 91:  # [
                self.frame_every = max(5, self.frame_every // 2)
            elif keycode == 93:  # ]
                self.frame_every = min(800, self.frame_every * 2)
            elif keycode == 72:  # H
                self.pitch_deg = 0.0
            elif keycode == 67:  # C
                self.follow = not self.follow
            elif keycode == 66:  # B
                self.brain = not self.brain


def tour(t: float) -> tuple[float, float, float]:
    """A scripted flight for the video: (pitch deg, heading deg, height mm) at time t."""
    if t < 0.25:
        return 0.0, 0.0, 5.0
    if t < 0.8:
        return 12.0, 0.0, 5.0
    if t < 1.4:
        return 12.0, 60.0, 10.0
    if t < 2.0:
        return 12.0, -20.0, 10.0
    return 0.0, -20.0, 8.0


class Flight:
    """The physics: the fly, its muscle, and its three loops."""

    def __init__(self, height_mm: float = 5.0):
        from ..body.control import HaltereController, Throttle
        from ..body.flight import FlightBody
        from ..body.power import PowerOscillator, PowerStroke, aerodynamic_load
        from ..brain.readout import PowerReadout

        free, wing = _models()
        self.brain = Brain.stored()
        readout = PowerReadout(command=self.brain.command, activation=self.brain.activation)
        oscillator = PowerOscillator(drive=0.0, load=aerodynamic_load(wing))
        oscillator.angle = np.deg2rad(1.0)
        self.body = FlightBody(free, wing, timestep=2e-5)
        self.controller = HaltereController(
            stroke=PowerStroke(oscillator),
            throttle=Throttle(readout=readout, oscillator=oscillator, target=height_mm),
        )
        self.steps = 0
        self.knobs: dict = {}
        original = self.controller.knobs

        def recorded(body):
            self.knobs = original(body)
            return self.knobs

        self.controller.knobs = recorded

    @property
    def t(self) -> float:
        return float(self.body.t)

    @property
    def timestep(self) -> float:
        return float(self.body.model.opt.timestep)

    def pose(self) -> tuple[np.ndarray, np.ndarray]:
        """Where the head is and which way it faces: (position, rotation matrix)."""
        b = self.body
        return b.data.xpos[b.root_body].copy(), b.data.xmat[b.root_body].reshape(3, 3).copy()

    def command(self, pitch_deg: float, heading_deg: float, height_mm: float) -> None:
        c = self.controller
        c.target_pitch = np.deg2rad(pitch_deg)
        c.target_heading = np.deg2rad(heading_deg)
        c.throttle.target = height_mm

    def step(self, n: int = 1) -> None:
        body, c = self.body, self.controller
        for _ in range(n):
            angles, rates = c.command(body, body.t)
            body.set_wings(angles, rates)
            body.apply_aerodynamics()
            body._mj.mj_step(body.model, body.data)
            self.steps += 1

    def state(self) -> dict:
        from ..body.control import attitude

        b = self.body
        i = b.root_dof
        pos = b.data.qpos[i : i + 3]
        vel = b.data.qvel[i : i + 3]
        pitch, roll = attitude(b)
        heading = np.degrees(self.controller._turn.get("total", 0.0))
        throttle = self.controller.throttle.last or 0.0
        return dict(
            t=self.t,
            x=float(pos[0]),
            y=float(pos[1]),
            z=float(pos[2]),
            speed=float(np.hypot(vel[0], vel[1])),
            vx=float(vel[0]),
            vy=float(vel[1]),
            climb=float(vel[2]),
            pitch=float(np.degrees(pitch)),
            roll=float(np.degrees(roll)),
            heading=float(heading),
            neurons=self.brain.read(throttle, None),
        )

    def rates(self) -> np.ndarray:
        """Body rotation rates, rad/s, as the haltere loop reads them.

        Averaged over one wingbeat, the way the controller's sensing averages
        them: the raw rate swings with every stroke and reads as noise.
        """
        from ..body.control import angular_rate

        st = self.controller._state
        if st and st.get("filled"):
            return np.asarray(st["sum"][3:6] / st["filled"], dtype=float)
        return angular_rate(self.body)

    def upright(self) -> bool:
        s = self.state()
        return abs(s["pitch"]) < 60 and abs(s["roll"]) < 60 and np.isfinite(s["z"])


#: The drone's active-infrared depth camera (time of flight): rays over 70
#: degrees, a reach of 80 mm -- 4 m at the drone's scale, as small ToF depth
#: cameras have -- and 1% of noise on each distance.
TOF_FOV, TOF_SIZE, TOF_RANGE, TOF_NOISE = 70.0, 24, 80.0, 0.01
#: Its laser rangefinder: one beam along the heading, 2 degrees wide, a reach
#: of 240 mm -- 12 m at the drone's scale, as small lidar rangefinders have
#: -- and 1% of noise.
LRF_BEAM, LRF_RANGE, LRF_NOISE = 2.0, 240.0, 0.01
#: And its thermal camera: 96 x 96 pixels over 60 degrees (0.6 degrees a
#: pixel, as small uncooled cameras have), 0.05 K of noise. What it sees is
#: an assumption the simulation cannot check, so it is a setting: every
#: solid -- object or landscape column -- is ``IR_HEAT`` kelvin warmer than
#: the floor on average, each its own 0.6-1.4 times that (material, sun);
#: the floor is uneven by ``IR_CLUTTER``. 0 heat: thermally invisible.
IR_FOV, IR_SIZE, IR_NETD = 60.0, 96, 0.05
IR_HEAT, IR_CLUTTER = 1.5, 0.3


def _warm(things) -> list:
    """Give each solid its warmth, the same every time for the same solid."""
    for t in things:
        x, y = float(t.position[0]), float(t.position[1])
        u = (np.sin(12.9898 * x + 78.233 * y) * 43758.5453) % 1.0
        t.heat = IR_HEAT * (0.6 + 0.8 * u)
    return things


def _level(R) -> np.ndarray:
    """The heading of a body-to-world rotation, without its roll and pitch.

    The rangefinder and the thermal camera sit on a gimbal, as a drone's
    payload does: they look along the heading, level, however the drone leans.
    """
    h = np.arctan2(R[1][0], R[0][0])
    c, s = np.cos(h), np.sin(h)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


class FlysEye:
    """The arena as the fly sees it, and its optic lobe looking at it."""

    def __init__(self, every_ms: float = 5.0, items: tuple = ()):
        from .scene import Camera, Thing, World
        from .vision import FlyEye

        things = []
        for x, y, rgb in COLUMNS:
            r, g, b = (float(v) for v in rgb.split())
            luminance = 0.3 * r + 0.55 * g + 0.15 * b
            things.append(Thing("bar", [x, y, FLOOR], 3.0, height=60.0, brightness=luminance))
        from .arena import eye_things

        self.background = list(things)
        self.world = World(_warm(things + eye_things(items)), ground_height=FLOOR)
        self.camera = Camera(size=64, fov=120.0)
        self.eye = FlyEye(dt=every_ms / 1000.0)
        self.every_ms = every_ms
        self.image = np.zeros((64, 64), dtype=np.float32)
        self.activity = None
        self.pose = None  # where it looked from last: (position, body-to-world R)
        # Each photoreceptor's line of sight in the body frame, and its
        # elevation above the horizon as of the last look.
        half = np.deg2rad(self.camera.fov) / 2
        az, el = self.eye.column_x * half, self.eye.column_y * half
        self._sight = np.stack([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
        self.elevation = np.degrees(el)

    def set_items(self, items) -> None:
        """A new arena for the same eye: the objects change, the optic lobe stays."""
        from .arena import eye_things
        from .scene import World

        self.world = World(_warm(self.background + eye_things(items)), ground_height=FLOOR)
        self.eye.reset()

    def look(self, body) -> None:
        """Render the fly's view from where the physics has it, and step the optic lobe."""
        R = body.data.xmat[body.root_body].reshape(3, 3)
        self.look_from(body.data.xpos[body.root_body], R)

    def look_from(self, position, R) -> None:
        """Render the view from a position and orientation, and step the optic lobe."""
        from .scene import render

        self.image = render(
            self.world, position, None, self.camera, rotation_matrix=R, checker=20.0
        )
        self.pose = (np.asarray(position, dtype=float), np.asarray(R, dtype=float))
        self.activity = self.eye.step(self.image, frames=1)
        up = (np.asarray(R) @ self._sight)[2]
        self.elevation = np.degrees(np.arcsin(np.clip(up, -1.0, 1.0)))

    def _solid(self) -> np.ndarray:
        """Which photoreceptors look at something standing, by the depth camera."""
        from .scene import _bar_hits, _box_hits, _sphere_hits

        out = np.zeros(len(self.eye.column_x), dtype=bool)
        if self.pose is None:
            return out
        o, R = self.pose
        half = TOF_FOV / 2
        az = np.degrees(self.eye.column_x * np.deg2rad(self.camera.fov) / 2)
        el = np.degrees(self.eye.column_y * np.deg2rad(self.camera.fov) / 2)
        seen = (np.abs(az) < half) & (np.abs(el) < half)
        d = (np.asarray(R) @ self._sight).T  # world directions
        t = np.full(len(d), np.inf)
        for thing in self.world.things:
            if thing.kind == "sphere":
                hit = _sphere_hits(o, d, thing.position, thing.size)
            elif thing.kind == "bar":
                hit = _bar_hits(o, d, thing.position, thing.size, thing.height)
            else:
                hit = _box_hits(o, d, thing.position, thing.size)
            t = np.minimum(t, hit)
        return seen & (t <= TOF_RANGE)

    def _warm_sight(self, rng=None) -> np.ndarray:
        """Which photoreceptors look at something warmer than the floor, by thermal camera."""
        from .scene import thermal_along

        out = np.zeros(len(self.eye.column_x), dtype=bool)
        if self.pose is None:
            return out
        o, R = self.pose
        half = IR_FOV / 2
        az = np.degrees(self.eye.column_x * np.deg2rad(self.camera.fov) / 2)
        el = np.degrees(self.eye.column_y * np.deg2rad(self.camera.fov) / 2)
        seen = (np.abs(az) < half) & (np.abs(el) < half)
        d = (np.asarray(R) @ self._sight).T
        t = thermal_along(self.world, o, d, clutter=IR_CLUTTER)
        rng = rng if rng is not None else np.random.default_rng()
        t = t + IR_NETD * rng.standard_normal(t.shape)
        return seen & (t > max(3 * IR_CLUTTER, 5 * IR_NETD))

    def measure(self, rng=None) -> dict | None:
        """The thing straight ahead, by the depth camera: how far, how tall, how wide.

        Distances, mm; ``height`` above the floor, and ``top`` if it reaches
        the top of the depth camera's view -- taller than it can tell. The
        thing is what is solid -- above the floor -- nearest the middle of the
        view, with whatever joins it at about the same distance. None if
        nothing solid is within reach.
        """
        from scipy import ndimage

        from .scene import Camera, depth

        if self.pose is None:
            return None
        o, R = self.pose
        tof = Camera(size=TOF_SIZE, fov=TOF_FOV)
        t = depth(self.world, o, tof, rotation_matrix=R)
        rng = rng if rng is not None else np.random.default_rng()
        t = t * (1.0 + TOF_NOISE * rng.standard_normal(t.shape))
        t = np.where(t <= TOF_RANGE, t, np.inf)
        d = tof.rays() @ R.T
        reach = np.where(np.isfinite(t), t, 0.0)
        p = o + reach[..., None] * d
        floor = self.world.ground_height
        solid = np.isfinite(t) & (p[..., 2] > floor + 1.5)
        if not solid.any():
            return None
        n = TOF_SIZE
        rows, cols = np.nonzero(solid)
        near = np.hypot(rows - (n - 1) / 2, cols - (n - 1) / 2)
        if near.min() > n / 6:
            return None  # nothing solid near the middle
        seed = (rows[near.argmin()], cols[near.argmin()])
        band = solid & (np.abs(t - t[seed]) < 15.0)
        labels, _ = ndimage.label(band)
        mine = labels == labels[seed]
        pts = p[mine]
        ahead = np.array([R[0, 0], R[1, 0]])
        ahead /= np.linalg.norm(ahead) + 1e-9
        side = pts[:, 0] * -ahead[1] + pts[:, 1] * ahead[0]
        return dict(
            distance=float(t[seed]),
            height=float(pts[:, 2].max() - floor),
            width=float(side.max() - side.min()),
            top=bool(mine[0].any()),
        )

    def laser_range(self, rng=None) -> float:
        """How far the rangefinder's beam goes along the heading, mm; inf beyond its reach."""
        from .scene import Camera, depth

        if self.pose is None:
            return np.inf
        o, R = self.pose
        rng = rng if rng is not None else np.random.default_rng()
        t = depth(self.world, o, Camera(size=3, fov=LRF_BEAM), rotation_matrix=_level(R))
        t = float(t.min()) * (1.0 + LRF_NOISE * rng.standard_normal())
        return t if t <= LRF_RANGE else np.inf

    def measure_far(self, rng=None) -> dict | None:
        """The thing ahead, by the thermal camera and the rangefinder together.

        The rangefinder gives how far it is; the thermal camera, how many
        degrees tall and wide it is -- paint is not in a thermal picture; and
        the two make millimetres, from further than the depth camera reaches.
        The thing is what is warm where the beam points, with whatever joins
        it at about the same warmth. Like :meth:`measure`, with ``top`` if it
        reaches the top of the thermal camera's view. None if the beam finds
        nothing, or the thermal camera sees nothing warmer than the floor there.
        """
        from scipy import ndimage

        from .scene import Camera, thermal

        d = self.laser_range(rng)
        if not np.isfinite(d):
            return None
        o, R = self.pose
        rng = rng if rng is not None else np.random.default_rng()
        level = _level(R)
        cam = Camera(size=IR_SIZE, fov=IR_FOV)
        image = thermal(self.world, o, cam, rotation_matrix=level, clutter=IR_CLUTTER)
        image = image + IR_NETD * rng.standard_normal(image.shape)
        n = IR_SIZE
        mid = (n - 1) // 2
        warm = image > max(3 * IR_CLUTTER, 5 * IR_NETD)
        block = warm[mid - 1 : mid + 2, mid - 1 : mid + 2]
        if not block.any():
            return None
        rows, cols = np.nonzero(block)
        seed = (mid - 1 + rows[0], mid - 1 + cols[0])
        like = warm & (np.abs(image - image[seed]) < max(0.15 * image[seed], 3 * IR_NETD))
        labels, _ = ndimage.label(like)
        mine = labels == labels[seed]
        step = np.deg2rad(IR_FOV) / (n - 1)
        r, c = np.nonzero(mine)
        top = np.deg2rad(IR_FOV) / 2 - r.min() * step  # elevation of its top edge
        across = (c.max() - c.min() + 1) * step
        return dict(
            distance=d,
            height=float(o[2] - FLOOR + d * np.tan(top)),
            width=float(2 * d * np.tan(across / 2)),
            top=bool(r.min() == 0),
        )

    def standing(
        self,
        threshold: float = 0.25,
        within: float = 1.0,
        centre: float = 0.0,
        sky: float = 0.78,
        margin: float = 3.0,
        depth: bool = False,
        warm: bool = False,
    ) -> tuple[float, float]:
        """What stands up out of the world: (fraction of the eye, bearing).

        :meth:`~.vision.FlyEye.dark` sees what is dark; this also sees what
        breaks the skyline -- anything darker than the sky that is more than
        ``margin`` degrees above the horizon, which no floor ever is. The
        horizon comes from the body's own sense of its attitude, as a fly's
        does from its halteres and ocelli, not from the image. Painted in the
        floor's checker, an object is no longer dark, but what of it stands
        above the horizon still blots out the sky -- as does every landmark
        column.

        ``depth``: and what the depth camera finds standing on the floor, in
        its narrower view and within its reach -- below the horizon too, where
        a camouflaged object is floor against floor (:data:`TOF_RANGE`).
        ``warm``: and what the thermal camera sees warmer than the floor, in
        its view, however far (:data:`IR_HEAT`).
        """
        eye = self.eye
        pr = eye.photoreceptors
        hot = (pr < threshold) | ((pr < sky) & (self.elevation > margin))
        if depth:
            hot |= self._solid()
        if warm:
            hot |= self._warm_sight()
        if within < 1.0:
            hot &= np.abs(eye.column_x - centre) < within
        if not hot.any():
            return 0.0, 0.0
        return float(hot.mean()), float(eye.column_x[hot].mean())


def _status(s: dict, slowdown: float) -> str:
    n = s["neurons"]
    return (
        f"t={s['t'] * 1000:6.0f} ms  speed {s['speed']:5.0f} mm/s  height {s['z']:5.1f} mm  "
        f"heading {s['heading']:+6.1f}  pitch {s['pitch']:+5.1f}  roll {s['roll']:+5.1f}  |  "
        f"DNa08/DNp31 {n[DRIVE]:.2f}  power MNs {n[POWER]:.2f}  "
        f"| {slowdown:4.0f}x slow motion"
    )


def interactive(
    brain: bool = True, pretty: bool = False, browser: bool = True, fast: bool = False
) -> None:
    """Open the 3-D window and the brain page in the browser, and fly.

    ``fast`` flies the reduced-order fly of :mod:`.fastfly` with the optic lobe
    at its trained 20 ms, kept to real time: the fly as fast as a fly.
    """
    from .livebrain import LiveBrain

    print(__doc__.split("Controls in the window")[1].split("The fly is")[0])
    eye = None
    if brain:
        # The optic lobe (PyTorch) is loaded before MuJoCo: with MuJoCo's
        # software renderer loaded first, importing torchvision segfaults.
        print("Loading the fly's optic lobe (flyvis)... this takes a few seconds.")
        eye = FlysEye(every_ms=20.0 if fast else 5.0)
    import mujoco.viewer

    if fast:
        from .fastfly import FastFly

        flight = FastFly()
    else:
        flight = Flight()
    view, pilot = FlyView(pretty=pretty), Pilot()
    live = LiveBrain(eye.eye if eye else None)
    print(f"The brain page: {live.url}  (it opens in your browser)")
    if browser:
        live.open()
    print("The 3-D window opens in a moment.")
    try:
        with mujoco.viewer.launch_passive(
            view.model, view.data, key_callback=pilot.key
        ) as viewer:
            last_print = last_snap = last_frame = 0.0
            wall0, sim0, next_look = time.time(), flight.t, 0.0
            while viewer.is_running():
                with pilot.lock:
                    paused, every = pilot.paused, pilot.frame_every
                    flight.command(pilot.pitch_deg, pilot.heading_deg, pilot.height_mm)
                    looking = pilot.brain and eye is not None
                if paused:
                    time.sleep(0.03)
                    wall0 += 0.03  # a pause is not flight time
                    viewer.sync()
                    continue
                # The fast fly steps in ms, the full one in 20 us: the same
                # share of a second per frame either way at the default.
                flight.step(max(1, every * 2 // 5) if fast else every)
                if not flight.upright():
                    print("\nThe fly lost control. Close the window and start again.")
                    break
                if looking and flight.t >= next_look:
                    eye.look_from(*flight.pose())
                    next_look = flight.t + eye.every_ms / 1000.0
                now = time.time()
                if not fast or now - last_frame > 1.0 / 30.0:
                    # In the fast mode drawing, not flying, is the cost:
                    # at most 30 frames a second.
                    view.show(flight)
                    if pilot.follow:
                        with viewer.lock():
                            view.chase(viewer.cam, flight.state()["heading"])
                    viewer.sync()
                    last_frame = now
                wall, sim = now - wall0, flight.t - sim0
                if fast and sim > wall:  # no faster than a fly
                    time.sleep(min(0.05, sim - wall))
                    now = time.time()
                    wall = now - wall0
                slow = wall / sim if sim > 0 else 0.0
                if now - last_snap > 0.25:
                    live.update(
                        flight.state(),
                        flight.knobs,
                        flight.rates(),
                        slow,
                        eye.image if looking else None,
                        eye.activity if looking else None,
                    )
                    last_snap = now
                if now - last_print > 0.5:
                    print("\r" + _status(flight.state(), slow), end="", flush=True)
                    last_print = now
    finally:
        live.close()


def video(
    path,
    seconds: float = 2.4,
    frame_ms: float = 2.0,
    width: int = 960,
    height: int = 540,
    fps: int = 30,
    script=tour,
    brain: bool = True,
) -> Path:
    """Fly the scripted tour and write it as an MP4, the brain panel underneath."""
    import imageio.v2 as imageio
    from PIL import Image

    from .brainpanel import BrainPanel

    eye = FlysEye() if brain else None  # before MuJoCo; see interactive()
    import mujoco

    flight, view = Flight(), FlyView(pretty=True)
    panel = BrainPanel(eye.eye, live=False) if brain else None
    renderer = mujoco.Renderer(view.model, height, width)
    cam = mujoco.MjvCamera()
    every = max(1, int(round(frame_ms * 1e-3 / float(flight.body.model.opt.timestep))))
    frames = int(seconds * 1000 / frame_ms)
    path = Path(path)
    writer = imageio.get_writer(path, fps=fps, codec="libx264", quality=7, macro_block_size=1)
    next_look = 0.0
    try:
        for _ in range(frames):
            flight.command(*script(flight.t))
            flight.step(every)
            if not flight.upright():
                break
            view.pose(flight.body)
            view.chase(cam, flight.state()["heading"])
            renderer.update_scene(view.data, camera=cam)
            frame = _caption(renderer.render(), flight.state(), frame_ms, fps)
            if panel is not None:
                if flight.t >= next_look:
                    eye.look(flight.body)
                    next_look = flight.t + eye.every_ms / 1000.0
                panel.update(
                    flight.state(), flight.knobs, flight.rates(), eye.image, eye.activity
                )
                below = Image.fromarray(panel.image())
                below = below.resize((width, int(below.height * width / below.width)))
                frame = np.concatenate([frame, np.asarray(below)], axis=0)
                frame = frame[: frame.shape[0] // 2 * 2]
            writer.append_data(frame)
    finally:
        writer.close()
    return path


def _caption(frame: np.ndarray, s: dict, frame_ms: float, fps: int) -> np.ndarray:
    """Burn the flight state into the frame, top left."""
    from PIL import Image, ImageDraw

    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    slow = 1000.0 / (frame_ms * fps)
    n = s["neurons"]
    lines = [
        f"t = {s['t'] * 1000:5.0f} ms   ({slow:.0f}x slow motion)",
        f"speed {s['speed']:4.0f} mm/s   height {s['z']:5.1f} mm   "
        f"heading {s['heading']:+5.0f} deg",
        f"DNa08/DNp31 drive {n[DRIVE]:.2f}   power motor neurons {n[POWER]:.2f}",
    ]
    try:
        from PIL import ImageFont

        font = ImageFont.load_default(size=max(14, frame.shape[0] // 30))
    except (TypeError, ImportError):  # Pillow older than 10.1
        font = None
    step = max(16, frame.shape[0] // 26)
    draw.rectangle(
        [8, 8, int(frame.shape[1] * 0.72), 16 + step * len(lines)], fill=(255, 255, 255)
    )
    for k, line in enumerate(lines):
        draw.text((16, 12 + step * k), line, fill=(20, 20, 20), font=font)
    return np.asarray(img)


__all__ = ["FIXTURES", "Flight", "FlyView", "Pilot", "interactive", "tour", "video"]
