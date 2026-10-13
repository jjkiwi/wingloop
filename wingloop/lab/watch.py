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

A second window shows the brain as it flies: what the fly sees, its optic
lobe (45,669 neurons of the pretrained flyvis model, one dot per column), the
haltere signals, the descending and motor neurons, and the wing commands.

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


def _decorate(source: Path, destination: Path, pretty: bool = False) -> Path:
    """Add a sky, a chequered floor, light and columns to the hinged fly.

    Shadows are the expensive part of drawing this scene, so they are off
    unless ``pretty`` is asked for; nothing else about the picture changes.
    """
    tree = ET.parse(source)
    root = tree.getroot()
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
    tree.write(destination)
    return destination


@lru_cache(maxsize=2)
def _visual_model_path(pretty: bool = False) -> Path:
    from ..body.hinge import add_free_base, add_wing_hinges, neuromechfly_model, tuck_legs

    d = Path(tempfile.mkdtemp(prefix="flylab-view-"))
    rigid, _ = tuck_legs(neuromechfly_model(), d / "rigid.xml", keep="__none__")
    hinged = add_wing_hinges(rigid, d / "hinged.xml")
    free = add_free_base(getattr(hinged, "path", hinged), d / "free.xml", dofs="free")
    return _decorate(Path(free), d / "scene.xml", pretty)


class FlyView:
    """The posed copy of the fly that gets drawn."""

    def __init__(self, pretty: bool = False):
        import mujoco

        self._mj = mujoco
        self.model = mujoco.MjModel.from_xml_path(str(_visual_model_path(pretty)))
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


class FlysEye:
    """The arena as the fly sees it, and its optic lobe looking at it."""

    def __init__(self, every_ms: float = 5.0):
        from .scene import Camera, Thing, World
        from .vision import FlyEye

        things = []
        for x, y, rgb in COLUMNS:
            r, g, b = (float(v) for v in rgb.split())
            luminance = 0.3 * r + 0.55 * g + 0.15 * b
            things.append(Thing("bar", [x, y, FLOOR], 3.0, height=60.0, brightness=luminance))
        self.world = World(things, ground_height=FLOOR)
        self.camera = Camera(size=64, fov=120.0)
        self.eye = FlyEye(dt=every_ms / 1000.0)
        self.every_ms = every_ms
        self.image = np.zeros((64, 64), dtype=np.float32)
        self.activity = None

    def look(self, body) -> None:
        """Render the fly's view from where the physics has it, and step the optic lobe."""
        from .scene import render

        R = body.data.xmat[body.root_body].reshape(3, 3)
        position = body.data.xpos[body.root_body]
        self.image = render(
            self.world, position, None, self.camera, rotation_matrix=R, checker=20.0
        )
        self.activity = self.eye.step(self.image, frames=1)


def _status(s: dict, slowdown: float) -> str:
    n = s["neurons"]
    return (
        f"t={s['t'] * 1000:6.0f} ms  speed {s['speed']:5.0f} mm/s  height {s['z']:5.1f} mm  "
        f"heading {s['heading']:+6.1f}  pitch {s['pitch']:+5.1f}  roll {s['roll']:+5.1f}  |  "
        f"DNa08/DNp31 {n[DRIVE]:.2f}  power MNs {n[POWER]:.2f}  "
        f"| {slowdown:4.0f}x slow motion"
    )


def interactive(brain: bool = True, pretty: bool = False) -> None:
    """Open the 3-D window and the brain panel, and fly."""
    from .brainpanel import BrainPanel

    print(__doc__.split("Controls in the window")[1].split("The fly is")[0])
    eye = None
    if brain:
        # The optic lobe (PyTorch) is loaded before MuJoCo: with MuJoCo's
        # software renderer loaded first, importing torchvision segfaults.
        print("Loading the fly's optic lobe (flyvis)... this takes a few seconds.")
        eye = FlysEye()
    import mujoco.viewer

    flight, view, pilot = Flight(), FlyView(pretty=pretty), Pilot()
    panel = BrainPanel(eye.eye if eye else None, live=True)
    print("The window opens in a moment.")
    with mujoco.viewer.launch_passive(view.model, view.data, key_callback=pilot.key) as viewer:
        last_print = last_panel = 0.0
        wall0, sim0, next_look = time.time(), flight.t, 0.0
        while viewer.is_running() and panel.is_open():
            with pilot.lock:
                paused, every = pilot.paused, pilot.frame_every
                flight.command(pilot.pitch_deg, pilot.heading_deg, pilot.height_mm)
                looking = pilot.brain and eye is not None
            if paused:
                time.sleep(0.03)
                viewer.sync()
                panel.plt.pause(0.001)
                continue
            flight.step(every)
            if not flight.upright():
                print("\nThe fly lost control. Close the window and start again.")
                break
            if looking and flight.t >= next_look:
                eye.look(flight.body)
                next_look = flight.t + eye.every_ms / 1000.0
            view.pose(flight.body)
            if pilot.follow:
                with viewer.lock():
                    view.chase(viewer.cam, flight.state()["heading"])
            viewer.sync()
            now = time.time()
            if now - last_panel > 0.4:
                panel.update(
                    flight.state(),
                    flight.knobs,
                    flight.rates(),
                    eye.image if looking else None,
                    eye.activity if looking else None,
                )
                last_panel = time.time()
            if now - last_print > 0.5:
                wall, sim = now - wall0, flight.t - sim0
                slow = wall / sim if sim > 0 else 0.0
                print("\r" + _status(flight.state(), slow), end="", flush=True)
                last_print = now


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
