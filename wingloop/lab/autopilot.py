"""The fly flies to its targets on its own: training, the mission, and watching it.

Three steps, the way the experiment would be run with a real fly:

1. **Teaching on the tether** (:func:`learn`). The fly's eye -- the pretrained
   optic lobe -- is shown each kind of object from many distances and angles,
   and its mushroom body gets dopamine for the kinds that are targets and
   punishment for the rest. What it learned is saved, so the next run skips it.
2. **The mission** (:class:`Autopilot`). The physical fly in an arena with
   objects. Nobody steers: the :class:`~.navigate.Navigator` looks around,
   fixates every dark object and asks the mushroom body what it is worth, flies
   to the best one, brakes as it looms, and looks around again.
   Where it lands, the world rewards it (a target) or punishes it (anything
   else), and the mushroom body learns from that too, in flight.
3. **Watching it** (:func:`window`, :func:`video`): the 3-D window and the
   brain page in the browser, which here also shows the autopilot -- what the
   fly has judged, where it is going, and a map of the arena.

    flylab cel kula             teach "fly to the balls", then watch it do it
    flylab cel słupek --seed 3  another target, another arena
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from . import arena
from .navigate import Navigator, describe, nearest, teach, tethered_training

#: Polish names of the kinds, as the person types them, and back.
NAMES = {"kula": "sphere", "płytka": "slab", "plytka": "slab", "słupek": "bar", "slupek": "bar"}
POLISH = {k: v["pl"] for k, v in arena.KINDS.items()}


def kind_of(name: str) -> str:
    """``kula`` / ``sphere`` -> ``sphere``; anything else is an error worth reading."""
    k = NAMES.get(name.lower(), name.lower())
    if k not in arena.KINDS:
        known = ", ".join(sorted({*POLISH.values(), *arena.KINDS}))
        raise SystemExit(f"Nie znam obiektu {name!r}. Znane: {known}")
    return k


def learn(fly_eye, targets, samples: int = 90, seed: int = 0, progress=print):
    """Look on the tether and learn the targets, nothing saved: (mushroom body, valence)."""
    F, K = tethered_training(
        fly_eye, tuple(arena.KINDS), samples=samples, seed=seed, progress=progress
    )
    return teach_targets(F, K, targets, seed=seed)


#: The tethered fly's views of every kind, kept between runs: looking takes
#: several minutes, teaching from what it saw takes a fraction of a second, so
#: changing the target needs no new look.
VIEWS = "flylab_widoki_na_uwiezi.npz"
#: The same views, computed once and shipped, so the first run needs no look
#: either. ``--od-nowa`` looks again on this computer.
SHIPPED = Path(__file__).with_name("data") / "tethered_views.npz"


def _signature(samples: int, seed: int) -> str:
    """What the views depend on; a saved file with another signature is stale."""
    sizes = {k: v["size"] for k, v in arena.KINDS.items()}
    return f"v3 {samples} {seed} lead 120-180 {sizes}"


def views(fly_eye, folder=".", samples: int = 90, seed: int = 0, fresh=False, progress=print):
    """The tethered fly's look at every kind: (features, kinds), from disk if seen before."""
    path = Path(folder) / VIEWS
    sig = _signature(samples, seed)
    for saved in (path, SHIPPED):
        if saved.exists() and not fresh:
            with np.load(saved) as d:
                if str(d["signature"]) == sig:
                    if progress:
                        progress(f"Muszka już oglądała te obiekty na uwięzi: wczytuję {saved}")
                    return d["F"], d["K"]
    if progress:
        progress(
            "Muszka na uwięzi ogląda każdy rodzaj obiektu z różnych stron i odległości"
            " (kilka minut, tylko za pierwszym razem)..."
        )
    t0 = time.time()
    F, K = tethered_training(
        fly_eye, tuple(arena.KINDS), samples=samples, seed=seed, progress=progress
    )
    if progress:
        progress(f"  {len(K)} widoków w {time.time() - t0:.0f} s -> {path}")
    np.savez_compressed(path, F=F, K=K, signature=np.array(sig))
    return F, K


def teach_targets(F, K, targets, seed: int = 0):
    """Dopamine for the targets, punishment for the rest: (mushroom body, valence per kind)."""
    from .learning import MushroomBody

    kinds = tuple(arena.KINDS)
    mb = MushroomBody(F.shape[1], seed=seed, on_off=True)
    teach(mb, F, K, {k: (1.0 if k in targets else -1.0) for k in kinds})
    valence = {k: float(np.mean([mb.valence(f) for f in F[K == k]])) for k in kinds}
    return mb, valence


def learned(fly_eye, targets, folder=".", retrain: bool = False, progress=print):
    """Look on the tether (or remember having looked), then learn the targets."""
    F, K = views(fly_eye, folder, fresh=retrain, progress=progress)
    return teach_targets(F, K, targets)


class Autopilot:
    """The physical fly, its eye and its navigator, one look at a time.

    ``eye`` is a :class:`~.watch.FlysEye` built with the same ``items``; it is
    passed in because it has to be created before MuJoCo is imported (see
    :func:`~.watch.interactive`).
    """

    def __init__(
        self,
        items,
        mushroom_body,
        eye,
        valence=None,
        targets=None,
        height: float = 5.0,
        learn_in_flight: bool = True,
        vehicle=None,
        **navigator,
    ):
        from .watch import Flight

        self.items = tuple(items)
        self.eye = eye
        # The fly, or anything with its interface -- the drone of
        # :mod:`.dronebrain` is one.
        self.flight = vehicle if vehicle is not None else Flight(height_mm=height)
        self.who = "Muszka" if vehicle is None else "Dron"
        self.valence = valence or {}
        liked = {k for k, v in self.valence.items() if v > 0}
        self.targets = set(targets) if targets else liked
        self.nav = Navigator(
            mushroom_body,
            height=height,
            reward=self.reward if learn_in_flight else None,
            **navigator,
        )
        self.every = max(1, int(round(eye.every_ms / 1000.0 / self.flight.timestep)))
        self.trail: list = []
        self.started = False

    @property
    def done(self) -> bool:
        return self.nav.phase == "done"

    def step(self) -> None:
        """One look (5 ms of flight): see, decide, fly."""
        f = self.flight
        if not self.started:
            # Let the wings spin up and the height settle before deciding anything.
            f.command(0.0, 0.0, self.nav.height)
            f.step(self.every * 20)
            self.nav.start(f.state()["heading"])
            self.started = True
        self.eye.look_from(*f.pose())
        s = f.state()
        f.command(*self.nav.decide(self.eye, s, self.eye.every_ms / 1000.0))
        f.step(self.every)
        if (
            not self.trail
            or np.hypot(s["x"] - self.trail[-1][0], s["y"] - self.trail[-1][1]) > 1
        ):
            self.trail.append((round(s["x"], 1), round(s["y"], 1)))

    def reward(self, x: float, y: float) -> float:
        """The world's answer where the fly lands: +1 on a target, -1 on anything else."""
        item, d = nearest(self.items, x, y)
        if d > 30.0:
            return 0.0
        return 1.0 if item.kind in self.targets else -1.0

    def reached(self) -> list:
        """The items the fly has flown to, in order."""
        out = []
        for vx, vy in self.nav.visited:
            item, d = nearest(self.items, vx, vy)
            if d < 30.0:
                out.append(item)
        return out

    def seen_as(self, sighting):
        """Which item a sighting was (for the person; the fly does not know)."""
        x, y = sighting.where
        best, off = None, 15.0
        for it in self.items:
            o = abs((it.bearing_from(x, y) - sighting.heading + 180) % 360 - 180)
            if o < off:
                best, off = it, o
        return best

    def status(self) -> dict:
        """Everything the brain page shows about the mission."""
        s = self.flight.state()
        nav = self.nav
        reached = self.reached()
        now = None
        if self.eye.activity is not None:
            now = float(nav.mushroom_body.valence(self.eye.eye.fovea_shape(self.eye.activity)))
        target = nav.target
        return {
            "phase": nav.phase,
            "phase_pl": describe(nav.phase),
            "valence_now": None if now is None else round(now, 3),
            "learned": {POLISH[k]: round(v, 3) for k, v in self.valence.items()},
            "items": [
                dict(
                    kind=it.kind,
                    name=POLISH[it.kind],
                    x=it.x,
                    y=it.y,
                    reached=it in reached,
                    target=it.kind in self.targets,
                )
                for it in self.items
            ],
            "sightings": [
                dict(
                    heading=round(float(g.heading) % 360, 0),
                    valence=round(g.valence, 3),
                    name=POLISH[it.kind] if (it := self.seen_as(g)) else "?",
                    chosen=target is g,
                )
                for g in nav.sightings
            ],
            "fly": dict(x=round(s["x"], 1), y=round(s["y"], 1), heading=round(s["heading"], 1)),
            "trail": self.trail[-600:],
            "visited": [[round(float(a), 1), round(float(b), 1)] for a, b in nav.visited],
            "reached": [POLISH[it.kind] for it in reached],
            "lessons": [
                dict(name=POLISH[nearest(self.items, *place)[0].kind], dopamine=d)
                for _, d, place in nav.lessons
            ],
            "targets_left": sum(
                1 for it in self.items if it.kind in self.targets and it not in reached
            ),
        }

    def summary(self) -> str:
        reached = self.reached()
        wanted = [it for it in self.items if it.kind in self.targets]
        got = [it for it in wanted if it in reached]
        wrong = [it for it in reached if it not in wanted]
        line = (
            f"{self.who} doleciał{'a' if self.who == 'Muszka' else ''} do {len(got)}"
            f" z {len(wanted)} celów w {self.flight.t:.1f} s lotu"
        )
        if wrong:
            ending = "a" if self.who == "Muszka" else ""
            line += f"; pomylił{ending} się {len(wrong)} raz(y): " + ", ".join(
                POLISH[it.kind] for it in wrong
            )
        return line + "."


def run(
    items,
    mushroom_body,
    eye,
    valence=None,
    targets=None,
    seconds: float = 20.0,
    report=print,
    drone: bool = False,
):
    """The mission without a window: returns the :class:`Autopilot` when it ends."""
    pilot = _pilot(items, mushroom_body, eye, valence, targets, drone)
    last = None
    while pilot.flight.t < seconds and not pilot.done:
        pilot.step()
        if not pilot.flight.upright():
            report(f"{pilot.who}: utrata panowania nad lotem.")
            break
        if report and pilot.nav.phase != last:
            s = pilot.flight.state()
            report(
                f"  {s['t']:5.2f} s  {describe(pilot.nav.phase)}"
                f"  ({s['x']:.0f}, {s['y']:.0f}) mm"
            )
            last = pilot.nav.phase
    return pilot


def window(
    targets,
    seed: int = 0,
    per_kind: int = 2,
    pretty: bool = False,
    browser: bool = True,
    retrain: bool = False,
    folder=".",
    drone: bool = False,
) -> None:
    """Teach, then open the 3-D window and the brain page, and let it fly."""
    from .livebrain import LiveBrain
    from .watch import FlysEye, Pilot

    items = arena.mission(seed, per_kind=per_kind)
    print("Wczytuję oko muchy (flyvis)...")
    eye = FlysEye(items=items)  # before MuJoCo; see watch.interactive
    mb, valence = learned(eye, targets, folder=folder, retrain=retrain)
    _print_valence(valence, targets)
    import mujoco.viewer

    pilot = _pilot(items, mb, eye, valence, targets, drone)
    view, keys = _view(drone, pretty, items), Pilot()
    live = LiveBrain(eye.eye)
    print(f"Strona z mózgiem i autopilotem: {live.url}")
    if browser:
        live.open()
    print(
        "W oknie 3D: spacja = pauza, C = kamera za nim / swobodna. "
        f"Strzałki nic nie robią: {pilot.who.lower()} leci sam{'a' if not drone else ''}."
    )
    try:
        with mujoco.viewer.launch_passive(view.model, view.data, key_callback=keys.key) as v:
            wall0, last_snap, ended = time.time(), 0.0, False
            while v.is_running():
                with keys.lock:
                    paused, follow = keys.paused, keys.follow
                if paused or ended:
                    time.sleep(0.03)
                    v.sync()
                    continue
                pilot.step()
                if not pilot.flight.upright():
                    print(
                        f"\n{pilot.who}: utrata panowania nad lotem."
                        " Zamknij okno i spróbuj ponownie."
                    )
                    ended = True
                view.show(pilot.flight)
                if follow:
                    heading = pilot.flight.state()["heading"]
                    with v.lock():
                        view.chase(v.cam, heading, distance=CHASE[drone])
                v.sync()
                now = time.time()
                if now - last_snap > 0.25:
                    sim = pilot.flight.t
                    live.update(
                        pilot.flight.state(),
                        pilot.flight.knobs,
                        pilot.flight.rates(),
                        (now - wall0) / sim if sim > 0 else 0.0,
                        eye.image,
                        eye.activity,
                        autopilot=pilot.status(),
                    )
                    last_snap = now
                    print(f"\r{pilot.flight.t:5.2f} s  {describe(pilot.nav.phase):40s}", end="")
                if pilot.done and not ended:
                    print("\n" + pilot.summary())
                    ended = True
    finally:
        live.close()


#: How far behind the camera follows: the fly, and the drone.
CHASE = {False: 22.0, True: 12.0}


def _pilot(items, mb, eye, valence, targets, drone: bool) -> Autopilot:
    """The autopilot on the fly, or on the drone with the fly's brain."""
    if not drone:
        return Autopilot(items, mb, eye, valence, targets)
    from .dronebrain import NAVIGATOR, DroneFlight

    return Autopilot(items, mb, eye, valence, targets, vehicle=DroneFlight(), **NAVIGATOR)


def _view(drone: bool, pretty: bool, items):
    if drone:
        from .dronebrain import DroneView

        return DroneView(pretty=pretty, items=items)
    from .watch import FlyView

    return FlyView(pretty=pretty, items=items)


def _print_valence(valence: dict, targets) -> None:
    print("Czego się nauczyła (ocena: + przyciąga, - odpycha):")
    for k, v in valence.items():
        mark = "  <- cel" if k in targets else ""
        print(f"  {POLISH[k]:8s} {v:+.2f}{mark}")


def video(
    path,
    targets,
    seed: int = 0,
    per_kind: int = 2,
    seconds: float = 20.0,
    frame_ms: float | None = None,
    width: int = 960,
    height: int = 540,
    fps: int = 30,
    folder=".",
    retrain: bool = False,
    drone: bool = False,
) -> Path:
    """Record a whole mission: the chase view, with the map and the fly's eye beside it.

    ``frame_ms`` of flight per frame: 10 for the fly (slow motion, it is
    quick), 33 for the drone (about real time).
    """
    import imageio.v2 as imageio

    from .watch import FlysEye

    if frame_ms is None:
        frame_ms = 33.0 if drone else 10.0

    items = arena.mission(seed, per_kind=per_kind)
    eye = FlysEye(items=items)
    mb, valence = learned(eye, targets, folder=folder, retrain=retrain)
    import mujoco

    pilot = _pilot(items, mb, eye, valence, targets, drone)
    view = _view(drone, True, items)
    renderer = mujoco.Renderer(view.model, height, width)
    cam = mujoco.MjvCamera()
    looks = max(1, int(round(frame_ms / eye.every_ms)))
    path = Path(path)
    writer = imageio.get_writer(path, fps=fps, codec="libx264", quality=7, macro_block_size=1)
    try:
        while pilot.flight.t < seconds:
            for _ in range(looks):
                pilot.step()
            if not pilot.flight.upright():
                break
            view.show(pilot.flight)
            view.chase(cam, pilot.flight.state()["heading"], distance=CHASE[drone])
            renderer.update_scene(view.data, camera=cam)
            writer.append_data(_frame(renderer.render(), pilot, frame_ms, fps))
            if pilot.done:
                for _ in range(fps):  # hold the last frame a second
                    writer.append_data(_frame(renderer.render(), pilot, frame_ms, fps))
                break
    finally:
        writer.close()
    print(pilot.summary())
    return path


def _font(size: int):
    """A font with Polish letters: Pillow's built-in one has no ą, ę or ł."""
    from PIL import ImageFont

    for name in ("DejaVuSans.ttf", "arial.ttf", "segoeui.ttf", "LiberationSans-Regular.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default(size=size)


def _frame(scene: np.ndarray, pilot: Autopilot, frame_ms: float, fps: int) -> np.ndarray:
    """The 3-D view with the map, the fly's eye and the mission state on top."""
    from PIL import Image, ImageDraw

    img = Image.fromarray(scene)
    draw = ImageDraw.Draw(img)
    h = img.height
    font = _font(max(14, h // 30))
    side = h // 3
    # The map, top right: arena from above, the fly's track, what it reached.
    m = Image.fromarray(map_image(pilot.status(), side))
    img.paste(m, (img.width - side - 10, 10))
    # The fly's own view, under it.
    eye = (np.clip(pilot.eye.image, 0, 1) * 255).astype(np.uint8)
    e = Image.fromarray(eye).resize((side, side), Image.NEAREST)
    img.paste(e, (img.width - side - 10, 20 + side))
    st = pilot.status()
    slow = 1000.0 / (frame_ms * fps)
    lines = [
        f"t = {pilot.flight.t:4.1f} s   ({slow:.0f}x zwolnione)",
        f"muszka {st['phase_pl']}",
        "doleciała do: " + (", ".join(st["reached"]) or "-"),
    ]
    if st["sightings"]:
        lines.append(
            "ocena: "
            + "  ".join(f"{g['name']} {g['valence']:+.2f}" for g in st["sightings"][-4:])
        )
    step = int(font.size * 1.45)
    width = max(draw.textlength(line, font=font) for line in lines)
    draw.rectangle((8, 6, 20 + width, 12 + step * len(lines)), fill=(255, 255, 255))
    for k, line in enumerate(lines):
        draw.text((14, 10 + step * k), line, fill=(20, 20, 20), font=font)
    draw.text(
        (img.width - side - 10, 24 + 2 * side), "oko muchy", fill=(255, 255, 255), font=font
    )
    return np.asarray(img)


def map_image(status: dict, size: int = 200, extent: float = 110.0) -> np.ndarray:
    """A top-down map of the mission, as an RGB array."""
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (size, size), (250, 250, 247))
    d = ImageDraw.Draw(img)

    def px(x, y):
        return (size / 2 + x / extent * size / 2, size / 2 - y / extent * size / 2)

    d.rectangle((0, 0, size - 1, size - 1), outline=(195, 194, 183))
    colours = {k: tuple(int(255 * c * 1.6) for c in v["rgb"]) for k, v in arena.KINDS.items()}
    for it in status["items"]:
        cx, cy = px(it["x"], it["y"])
        r = 6
        d.ellipse((cx - r, cy - r, cx + r, cy + r), fill=colours[it["kind"]])
        if it["reached"]:
            d.ellipse(
                (cx - r - 3, cy - r - 3, cx + r + 3, cy + r + 3), outline=(12, 163, 12), width=2
            )
        elif it["target"]:
            d.ellipse(
                (cx - r - 3, cy - r - 3, cx + r + 3, cy + r + 3),
                outline=(250, 178, 25),
                width=2,
            )
    if len(status["trail"]) > 1:
        d.line([px(*p) for p in status["trail"]], fill=(42, 120, 214), width=2)
    f = status["fly"]
    fx, fy = px(f["x"], f["y"])
    a = np.deg2rad(f["heading"])
    tip = (fx + 9 * np.cos(a), fy - 9 * np.sin(a))
    d.ellipse((fx - 3, fy - 3, fx + 3, fy + 3), fill=(11, 11, 11))
    d.line((fx, fy, *tip), fill=(11, 11, 11), width=2)
    hx, hy = px(0, 0)
    d.rectangle((hx - 2, hy - 2, hx + 2, hy + 2), outline=(137, 135, 129))
    return np.asarray(img)


__all__ = [
    "Autopilot",
    "kind_of",
    "learn",
    "learned",
    "map_image",
    "run",
    "teach_targets",
    "video",
    "views",
    "window",
]
