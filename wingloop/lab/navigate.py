"""Teaching the fly what to fly to, and letting it fly there on its own.

**Teaching** is done the way it is done with real flies: tethered, in a flight
simulator. In the classic experiments (Wolf and Heisenberg) a fly held in
place sees shapes around it and is rewarded or punished depending on which
one it faces; afterwards it prefers the rewarded shape. Here the fly's eye --
the pretrained optic lobe -- looks at each kind of object from many distances
and angles, and the mushroom body is given dopamine for the kinds that are
targets and punishment for the rest. No physics is needed for that, so it
takes seconds rather than minutes.

**Flying** is the physical fly with its own three loops, steered by a
navigator that only uses what the fly has:

- its **eye**: where the darkest object is (bearing) and how big it is (an
  object growing fast is close);
- its **mushroom body**: what the object it is fixating is worth, from the
  middle of the eye, where a fixated object sits;
- its **path integration**: where it has been. Insects integrate their own
  path to know where home and visited places are; here the integrated path
  is read from the body's state, the one stand-in this module takes.

The mission: look around once (a full turn, fixating each object and
judging it), fly to the most valued target, brake as it looms, then look
around again for the next target it has not visited yet. Seen from one
target the others may be too far to make out, so when a look finds nothing
it flies home -- the way a bee or an ant returns to its nest by path
integration -- and looks from there before it calls the mission done.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .arena import KINDS, Item

#: Darker than this is an object (the floor's darkest tile is 0.35).
DARK = 0.25
#: Degrees either side of straight ahead in which an object is fixated while
#: scanning.
FIXATE = 18.0
#: Degrees either side of where the target should be in which the fly looks
#: for it while approaching. Narrow on purpose: turning toward a target sweeps
#: other objects across the middle of the eye, and a wide window latched onto
#: the first of them -- measured, a slab chosen at 126 degrees and a pole at 46
#: flown to instead.
TRACK = 15.0


@dataclass
class Sighting:
    heading: float  # compass heading of the object from where it was seen, degrees
    valence: float
    size: float
    where: tuple  # the fly's own position when it saw it, mm
    features: np.ndarray = field(default=None, repr=False)  # what the mushroom body was shown


# --------------------------------------------------------------- teaching


def tethered_training(
    fly_eye,
    kinds,
    samples: int = 24,
    seed: int = 0,
    distance=(35.0, 100.0),
    progress=None,
    scan_rate: float = 240.0,
    hold: float = 0.12,
    lead=(120.0, 180.0),
):
    """Show the tethered fly each kind of object; return (features, kinds).

    ``fly_eye`` is a :class:`~wingloop.lab.watch.FlysEye`; its world is swapped
    for a small arena for each view. Each object is shown **the way the flying
    fly meets it**: the panorama turns at the navigator's scan rate -- the fly
    turning on the spot -- until the object is near the middle of the eye,
    then the fly holds it there for the navigator's fixation time, with the
    settling and wobble of a real fixation, and what the eye says over the
    second half of the hold is the view. Two other objects stand off to the
    sides, as they do in an arena.

    The first version showed a still object to a still eye, and it taught the
    mushroom body well (91-96% on new still views) and the flying fly badly:
    in flight every object was valued below zero. The optic lobe's motion
    cells are still ringing from the turn when the fly fixates, and that was
    nothing like anything it had been taught on.

    How far the panorama turns before the object arrives (``lead``, degrees)
    matters too, because the eye is reset to rest for each view and a flying
    fly's eye never is. Judged on 53 fixations the physical fly made in nine
    arenas, 45 views turned in from 35-50 degrees taught the ball at 81%
    correct and 120-180 degrees at 89%; the slab and the pole were 85-96%
    either way.
    """
    from .arena import eye_things
    from .scene import World, render, rotation

    rng = np.random.default_rng(seed)
    background = list(fly_eye.background)
    dt = fly_eye.every_ms / 1000.0
    feats, labels = [], []
    for i in range(samples):
        kind = kinds[i % len(kinds)]
        d = rng.uniform(*distance)
        h = rng.uniform(-np.pi, np.pi)
        eye_pos = np.array([rng.uniform(-30, 30), rng.uniform(-30, 30), rng.uniform(4.5, 5.5)])
        shown = [Item(kind, eye_pos[0] + d * np.cos(h), eye_pos[1] + d * np.sin(h), 5.0)]
        for other in rng.choice(list(KINDS), 2):
            b = h + np.deg2rad(rng.uniform(90, 150)) * rng.choice([-1, 1])
            r = rng.uniform(40, 90)
            x, y = eye_pos[0] + r * np.cos(b), eye_pos[1] + r * np.sin(b)
            shown.append(Item(str(other), x, y))
        world = World(background + eye_things(shown), ground_height=fly_eye.world.ground_height)

        def look(yaw, eye_pos=eye_pos, world=world):
            image = render(
                world,
                eye_pos,
                None,
                fly_eye.camera,
                rotation_matrix=rotation(0.0, 0.0, yaw),
                checker=20.0,
            )
            return fly_eye.eye.step(image, frames=1)

        fly_eye.eye.reset()
        # Turning on the spot, the object coming in from the left.
        yaw = h - np.deg2rad(rng.uniform(*lead))
        while h - yaw > np.deg2rad(8.0):
            look(yaw)
            yaw += np.deg2rad(scan_rate) * dt
        # Fixating: the remaining error settles, with a small wobble on top.
        error, phase = h - yaw, rng.uniform(0, 2 * np.pi)
        n = max(2, int(round(hold / dt)))
        fovea = []
        for k in range(n):
            wobble = np.deg2rad(2.0) * np.sin(2 * np.pi * 8.0 * k * dt + phase)
            activity = look(h - error * np.exp(-k * dt / 0.02) + wobble)
            if k >= n // 2:
                fovea.append(fly_eye.eye.fovea_shape(activity))
        feats.append(np.mean(fovea, axis=0))
        labels.append(kind)
        if progress and (i + 1) % 10 == 0:
            progress(f"  obejrzała {i + 1} z {samples}")
    return np.asarray(feats), np.asarray(labels)


def teach(mushroom_body, features, kinds, reward: dict) -> None:
    """Dopamine for each view: ``reward[kind]``, positive for a target.

    Each side is balanced by how often it was seen: with one target kind and
    two punished ones, unbalanced punishment depresses the approach output
    twice as often as reward depresses avoidance, and every object ends up
    disliked -- measured, a target valued at -0.01 next to distractors at
    -0.1 to -0.3.
    """
    kinds = np.asarray([str(k) for k in kinds])
    n_good = max(1, sum(reward.get(k, 0.0) > 0 for k in kinds))
    n_bad = max(1, sum(reward.get(k, 0.0) < 0 for k in kinds))
    mushroom_body.calibrate(features)
    for f, k in zip(features, kinds, strict=True):
        r = reward.get(k, 0.0)
        scale = (len(kinds) / (2 * n_good)) if r > 0 else (len(kinds) / (2 * n_bad))
        # Capped only where depression would overshoot zero: lr * dopamine < 1.
        cap = 0.95 / mushroom_body.learning_rate
        mushroom_body.learn(f, float(np.clip(r * scale, -cap, cap)))


# ---------------------------------------------------------------- flying


@dataclass
class Navigator:
    """Turns what the eye and the mushroom body say into setpoints for the fly.

    Phases: ``scan`` (turn on the spot, fixate and judge each dark object),
    ``approach`` (fly at the best one, fixating it, brake as it looms),
    ``home`` (nothing liked in sight: fly back to the start by path
    integration and look again from there) and ``done``.
    """

    mushroom_body: object
    cruise_pitch: float = 12.0  # degrees nose-down while travelling
    height: float = 5.0
    scan_rate: float = 240.0  # degrees per second while looking around
    fixate_time: float = 0.12  # seconds of looking at each object
    refractory: float = 45.0  # degrees turned past an object before the next
    arrive_size: float = 0.07  # dark fraction of the eye at which it has arrived
    slow_size: float = 0.025  # ... at which it stops pushing forward
    station_gain: float = 0.05  # degrees of pitch per mm/s of drift, while hovering
    give_up: float = 4.0  # seconds of approach without arriving
    lost_after: float = 0.5  # seconds with nothing ahead before it gives the target up
    #: Arena mm/s below which a brake is over; None brakes for a fixed 0.12 s,
    #: which is what stops the fly. A drone has more momentum: it coasts on
    #: after 0.12 s, the place it records falls short of the object, and it
    #: came back to objects it had already reached.
    settle_speed: float | None = None
    #: Looks from home that find nothing before it calls the mission done. A
    #: single look is noisy: one measured mission ended at once, both its
    #: targets valued just below zero on the first turn.
    patience: int = 2
    #: With nothing liked and nowhere landed yet, try the most promising
    #: object anyway, and learn from what is there. Practice makes a fly
    #: choosier, and measured, a practised fly sometimes liked nothing at all
    #: and ended its mission on the spot with both targets in view.
    curious: bool = True
    #: What the world gives the fly where it lands, ``reward(x, y)`` ->
    #: dopamine (+ reward, - punishment, 0 nothing), or None for no learning
    #: in flight. The sugar on the target in the real experiment.
    reward: object = None
    phase: str = "scan"
    sightings: list = field(default_factory=list)
    visited: list = field(default_factory=list)
    lessons: list = field(default_factory=list)
    log: list = field(default_factory=list)

    def __post_init__(self):
        self._scanned = 0.0
        self._fix = 0.0
        self._fovea = []
        self._refr = 0.0
        self._target = None
        self._brake = 0.0
        self._braking = False
        self._moving = 0.0
        self._since = 0.0
        self._lost = 0.0
        self._empty_looks = 0
        self._home = (0.0, 0.0)
        self.heading = 0.0

    def start(self, heading: float, home=(0.0, 0.0)) -> None:
        """Begin where the fly is: its heading, and the place it calls home."""
        self.heading = heading
        self._home = tuple(home)

    @property
    def target(self):
        """The sighting being flown to, or None."""
        return self._target if self.phase == "approach" else None

    def _already_visited(self, here, heading_deg: float) -> bool:
        """Does this direction point at a place already visited?

        "At" within about 30 mm of it: from far away that is a narrow cone,
        from close by a wide one -- standing next to a visited object, most
        of what is around is that object.
        """
        for v in self.visited:
            dx, dy = v[0] - here[0], v[1] - here[1]
            to = np.degrees(np.arctan2(dy, dx))
            cone = max(25.0, np.degrees(np.arctan2(30.0, np.hypot(dx, dy))))
            if abs((to - heading_deg + 180) % 360 - 180) < cone:
                return True
        return False

    def decide(self, fly_eye, state: dict, dt: float) -> tuple[float, float, float]:
        """One look: returns (pitch deg, heading deg, height mm)."""
        half = fly_eye.camera.fov / 2
        here = (state["x"], state["y"])
        heading_now = state["heading"]
        self._moving = float(np.hypot(state.get("vx", 0.0), state.get("vy", 0.0)))
        if self.phase == "approach":
            # Only what is where the target should be counts as the target.
            off = (self._target.heading - heading_now + 180) % 360 - 180
            centre = float(np.clip(off, -half, half)) / half
            size, bearing = fly_eye.eye.dark(DARK, within=TRACK / half, centre=centre)
        else:
            # Only what is near the middle can be fixated and judged.
            size, bearing = fly_eye.eye.dark(DARK, within=FIXATE / half)
        bearing_deg = bearing * half  # +left
        self.log.append(
            dict(
                t=state["t"],
                phase=self.phase,
                size=size,
                bearing=bearing_deg,
                x=here[0],
                y=here[1],
                heading=heading_now,
            )
        )
        if self.phase == "scan":
            out = self._scan(fly_eye, size, bearing_deg, here, heading_now, dt)
        elif self.phase == "approach":
            out = self._approach(size, bearing_deg, here, heading_now, dt)
        elif self.phase == "home":
            out = self._go_home(here, heading_now)
        else:
            out = 0.0, self.heading, self.height
        if out[0] == 0.0 and "vx" in state:
            # Hovering: lean against drift along the heading, as a fly does on
            # the optic flow below it (read here from the body state). Worth
            # little and kept: braked from cruise and then turning on the spot
            # for a second, it drifts 13.5 mm with this and 16.4 without --
            # turning, the drift is along the heading only a fraction of the time.
            h = np.deg2rad(heading_now)
            forward = state["vx"] * np.cos(h) + state["vy"] * np.sin(h)
            out = (float(np.clip(-self.station_gain * forward, -5.0, 5.0)), *out[1:])
        return out

    def _turn(self, dt: float) -> tuple[float, float, float]:
        step = self.scan_rate * dt
        self._scanned += step
        self.heading += step
        return 0.0, self.heading, self.height

    def _scan(self, fly_eye, size, bearing_deg, here, heading_now, dt):
        if self._scanned >= 360.0:
            self._choose(here)
            return 0.0, self.heading, self.height
        if self._refr > 0:
            self._refr -= self.scan_rate * dt
            return self._turn(dt)
        centred = size > 0.003 and abs(bearing_deg) < 12.0
        if centred and not self._already_visited(here, heading_now + bearing_deg):
            # Fixate: hold the object in the middle of the eye and look at it.
            self._fix += dt
            self._fovea.append(fly_eye.eye.fovea_shape(fly_eye.activity))
            self.heading = heading_now + bearing_deg
            if self._fix >= self.fixate_time:
                f = np.mean(self._fovea[len(self._fovea) // 2 :], axis=0)
                v = float(self.mushroom_body.valence(f))
                self.sightings.append(Sighting(heading_now + bearing_deg, v, size, here, f))
                self._fix, self._fovea = 0.0, []
                self._refr = self.refractory
            return 0.0, self.heading, self.height
        self._fix, self._fovea = 0.0, []
        return self._turn(dt)

    def _approach(self, size, bearing_deg, here, heading_now, dt):
        if self._brake > 0 or self._braking:
            self._brake -= dt
            settled = self.settle_speed is None or self._moving < self.settle_speed
            self._braking = not (self._brake <= 0 and (settled or self._brake < -1.5))
            if not self._braking:
                # Path integration: remember where the target is -- a little
                # ahead of where the fly stopped.
                h = np.deg2rad(self.heading)
                place = (here[0] + 15 * np.cos(h), here[1] + 15 * np.sin(h))
                self.visited.append(place)
                self._learn_from(place)
                self._rescan()
            return -6.0, self.heading, self.height
        self._since += dt
        off = (self._target.heading - heading_now + 180) % 360 - 180
        if abs(off) > 25 and size < self.slow_size:
            # Not facing it yet: turn on the spot toward where it was.
            self.heading = heading_now + np.clip(off, -30, 30)
            return 0.0, self.heading, self.height
        if size > 0.002:
            # Fixate it: the heading follows the object across the eye.
            self.heading = heading_now + 0.8 * bearing_deg
            self._target.heading = self.heading
            self._lost = 0.0
        else:
            self._lost += dt
        if self._lost > self.lost_after or self._since > self.give_up:
            # Lost it, or flown past it: stop, and look again.
            self._rescan()
            return -6.0, self.heading, self.height
        if size >= self.arrive_size:
            self._brake, self._braking = 0.12, True
            return -6.0, self.heading, self.height
        push = self.cruise_pitch if size < self.slow_size else self.cruise_pitch * 0.35
        return push, self.heading, self.height

    def _learn_from(self, place) -> None:
        """Dopamine on arrival, onto the Kenyon cells the target's view drove.

        The view is the one the decision was made on, kept with the sighting,
        so the synapses that are depressed are the ones that made the choice:
        a mistake makes the next object that looks like it less attractive,
        and a find makes it more.
        """
        if self.reward is None or self._target is None or self._target.features is None:
            return
        dopamine = float(self.reward(*place))
        if dopamine:
            self.mushroom_body.learn(self._target.features, dopamine)
            self.lessons.append((self._target.heading, dopamine, place))

    def _go_home(self, here, heading_now):
        dx, dy = self._home[0] - here[0], self._home[1] - here[1]
        if np.hypot(dx, dy) < 12.0:
            self._rescan()
            return -6.0, self.heading, self.height
        want = np.degrees(np.arctan2(dy, dx))
        off = (want - heading_now + 180) % 360 - 180
        self.heading = heading_now + np.clip(off, -30, 30)
        if abs(off) > 25:
            return 0.0, self.heading, self.height
        near = np.hypot(dx, dy) < 30
        return self.cruise_pitch * (0.5 if near else 1.0), self.heading, self.height

    def _choose(self, here) -> None:
        liked = [
            s
            for s in self.sightings
            if s.valence > 0 and not self._already_visited(here, s.heading)
        ]
        if liked:
            self._target = max(liked, key=lambda s: s.valence)
            self._since = 0.0
            self._empty_looks = 0
            self.phase = "approach"
        elif np.hypot(here[0] - self._home[0], here[1] - self._home[1]) > 20.0:
            self.phase = "home"
        else:
            self._empty_looks += 1
            unvisited = [
                s for s in self.sightings if not self._already_visited(here, s.heading)
            ]
            if self._empty_looks < self.patience:
                self._rescan()
            elif self.curious and not self.visited and unvisited:
                self._target = max(unvisited, key=lambda s: s.valence)
                self._since = 0.0
                self.phase = "approach"
            else:
                self.phase = "done"

    def _rescan(self) -> None:
        self.phase = "scan"
        self._scanned = 0.0
        self._refr = 0.0
        self._fix, self._fovea = 0.0, []
        self._lost = 0.0
        self.sightings = []


def describe(phase: str) -> str:
    return {
        "scan": "rozgląda się",
        "approach": "leci do celu",
        "home": "wraca na start, by rozejrzeć się stamtąd",
        "done": "zwiedziła wszystkie cele",
    }.get(phase, phase)


def nearest(items, x: float, y: float) -> tuple[Item, float]:
    best = min(items, key=lambda it: it.distance_from(x, y))
    return best, best.distance_from(x, y)


__all__ = [
    "KINDS",
    "Navigator",
    "Sighting",
    "describe",
    "nearest",
    "teach",
    "tethered_training",
]
