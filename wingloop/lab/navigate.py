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


#: The landscape's columns, as a kind of object in the camouflage lessons:
#: seen by what breaks the skyline, a camouflaged target and a column look
#: alike until the mushroom body has learned which is which.
LANDMARK = "column"


def painted(rng, camo) -> float:
    """How much an object in a camouflage lesson wears the floor's checker, 0-1:
    a random amount for ``camo=True``, that amount for a number, none for False."""
    if camo is True:
        return float(rng.uniform(0.0, 1.0))
    return float(camo or 0.0)


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
    camo: bool = False,
):
    """Show the tethered fly each kind of object; return (features, kinds).

    ``camo``: every object painted in the floor's checker by a random amount
    (:attr:`~.arena.Item.camo`); and ``kinds`` may include :data:`LANDMARK`,
    one of the landscape's columns, shown the same way.

    For the drone, pass its :data:`~.dronebrain.SCAN_RATE` as ``scan_rate``:
    it turns more slowly than a fly. The close looks taken before landing
    are taught in flight instead (:func:`~.autopilot.flown_close_looks`).

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
        if kind == LANDMARK:
            column = background[rng.integers(len(background))].position
            eye_pos[:2] = column[:2] - d * np.array([np.cos(h), np.sin(h)])
            shown = []
        else:
            x, y = eye_pos[0] + d * np.cos(h), eye_pos[1] + d * np.sin(h)
            shown = [Item(kind, x, y, 5.0, camo=painted(rng, camo))]
        for other in rng.choice(list(KINDS), 2):
            b = h + np.deg2rad(rng.uniform(90, 150)) * rng.choice([-1, 1])
            r = rng.uniform(40, 90)
            x, y = eye_pos[0] + r * np.cos(b), eye_pos[1] + r * np.sin(b)
            shown.append(Item(str(other), x, y, camo=painted(rng, camo)))
        world = World(background + eye_things(shown), ground_height=fly_eye.world.ground_height)

        def look(yaw, eye_pos=eye_pos, world=world, at=None, pitch=0.0):
            image = render(
                world,
                eye_pos if at is None else at,
                None,
                fly_eye.camera,
                rotation_matrix=rotation(0.0, pitch, yaw),
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


def teach(
    mushroom_body,
    features,
    kinds,
    reward: dict,
    dose: float = 1.0,
    calibrate: bool = True,
    aim: float | None = None,
    epochs: int = 1,
    seed: int = 0,
) -> None:
    """Dopamine for each view: ``reward[kind]``, positive for a target.

    ``dose`` scales the dopamine of this lesson; ``calibrate=False`` keeps the
    input normalisation an earlier lesson set, for a second lesson on top.

    ``aim``: dopamine as a prediction error (Rescorla-Wagner) instead -- the
    valence aimed at, ``aim`` for a target and ``-aim`` for the rest, minus
    the valence the view already has; a view already there teaches nothing.
    The lesson is shown ``epochs`` times, in a new random order each time.
    Learning only from what it still gets wrong, the brain does not spend
    its synapses on what it knows, and spends them on the views near the
    line, the hard ones -- as a model of the fly's mushroom body in which
    dopamine signals a prediction error does (Bennett, Philippides &
    Nowotny 2021, Nat. Commun. 12:2569).

    Each side is balanced by how often it was seen: with one target kind and
    two punished ones, unbalanced punishment depresses the approach output
    twice as often as reward depresses avoidance, and every object ends up
    disliked -- measured, a target valued at -0.01 next to distractors at
    -0.1 to -0.3.
    """
    kinds = np.asarray([str(k) for k in kinds])
    n_good = max(1, sum(reward.get(k, 0.0) > 0 for k in kinds))
    n_bad = max(1, sum(reward.get(k, 0.0) < 0 for k in kinds))
    if calibrate:
        mushroom_body.calibrate(features)
    # Capped only where depression would overshoot zero: lr * dopamine < 1.
    cap = 0.95 / mushroom_body.learning_rate
    if aim is not None:
        rng = np.random.default_rng(seed)
        for _ in range(epochs):
            for i in rng.permutation(len(kinds)):
                r = reward.get(kinds[i], 0.0)
                if r == 0:
                    continue
                error = np.sign(r) * aim - mushroom_body.valence(features[i])
                if error * r <= 0:
                    continue  # already where it should be
                scale = len(kinds) / (2 * (n_good if r > 0 else n_bad))
                dopamine = dose * abs(r) * scale * error
                mushroom_body.learn(features[i], float(np.clip(dopamine, -cap, cap)))
        return
    for f, k in zip(features, kinds, strict=True):
        r = reward.get(k, 0.0)
        scale = (len(kinds) / (2 * n_good)) if r > 0 else (len(kinds) / (2 * n_bad))
        mushroom_body.learn(f, float(np.clip(dose * r * scale, -cap, cap)))


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
    #: and ended its mission on the spot with both targets in view. A brain
    #: that takes close looks takes one here too, and lands only if it is
    #: clearly liked (``sure``); without, it lands to find out.
    curious: bool = True
    #: And before calling a mission done with objects it has not been to, go
    #: and take a close look at the most promising of them -- the close look
    #: decides. Only for a brain that takes close looks. Measured, missions
    #: ended with half their time left and a target valued just below zero
    #: from afar, never looked at from where the eye is reliable.
    look_before_leaving: bool = False
    #: How much a close look must like what it was taken back to look at: it
    #: was not liked from afar, and the close look has to overrule that. On
    #: the drone, close looks at what it had turned down from afar came out
    #: 0.000-0.018 on what it should not land on and 0.14 on a ball it should;
    #: the fly's and the drone's targets found this way, 0.023-0.21.
    sure: float = 0.02
    #: Look again from close before landing -- what a practised brain does
    #: (:func:`~.autopilot._pilot`). Judged from flight, the eye is
    #: right about 90% of the time at 40-60 mm and about 50% beyond 80, so
    #: the fly stops where the object fills ``inspect_size`` of its eye --
    #: about 45 mm for a ball, 60-70 for a pole or a slab -- looks for
    #: ``fixate_time``, and lands only if it still likes what it sees.
    inspect: bool = False
    inspect_size: float = 0.02
    #: The close look waits until the object is near the middle of the eye --
    #: within ``centred`` degrees -- and the body has stopped turning, below
    #: ``still`` deg/s, for at most ``face_time`` seconds. Measured on the
    #: drone, every close look taken within 3 degrees and below 8 deg/s was
    #: judged right, and every wrong one had been taken still turning,
    #: 22-116 deg/s with the object 5-42 degrees off the middle -- it had
    #: started its approach close by, already near enough to look (the
    #: drone's values are in :data:`~.dronebrain.NAVIGATOR`). A hovering fly
    #: keeps turning 25-35 deg/s after its drifting target, judges well doing
    #: so, and drifts into the object if it waits.
    centred: float = 15.0
    #: See what breaks the skyline as well as what is dark
    #: (:meth:`~.watch.FlysEye.standing`): for camouflaged targets.
    skyline: bool = False
    still: float = 60.0
    face_time: float = 1.5
    #: A close look that turns the object down by less than ``unsure``, taken
    #: before it fills ``closer_size`` of the eye, is taken again from there:
    #: the drone's practised close looks were never wrong from 40-55 mm, and
    #: the targets it still missed had been turned down by a hair -- -0.005,
    #: -0.04 -- from 62-74 mm. 0: never.
    unsure: float = 0.0
    closer_size: float = 0.04
    #: Close looks from different sides before deciding, on the mean of their
    #: valences: after each but the last it flies ``around`` degrees round
    #: the object, sideways at ``side_pitch`` (read as a speed, like the
    #: pitch) -- a body that can (:meth:`~.dronebrain.DroneFlight.strafe`).
    #: Meant for camouflaged targets, where one close look is right 70-90%
    #: of the time and the floor's checker behind and on the object changes
    #: from side to side. Measured, it did not pay (see the README): looks
    #: from the side were judged worse unless taught, the mean of three told
    #: targets apart little better than one, and three looks took three
    #: times as long -- fewer objects looked at in a mission.
    looks: int = 1
    around: float = 45.0
    side_pitch: float = 10.0
    #: About how far off it looks from, mm: how fast the object wheels round
    #: in the eye as it flies sideways, to turn ahead of it.
    around_from: float = 60.0
    #: Peer: take the close look flying sideways, to the left at ``peer``
    #: (read as a speed, like the pitch), turning to keep the object in the
    #: middle of the eye, for ``peer_time`` seconds -- as a locust sways
    #: before it jumps. What is near moves across the eye unlike what is far
    #: behind it, and the optic lobe's motion cells see that: an object
    #: painted like the floor still does not move like the floor. Measured,
    #: taught on peering looks, the pole was told apart better (AUC 0.97
    #: against 0.93 at full camouflage), the ball and slab no better, and
    #: peering harder made the ball worse: kept off. 0: look hovering still.
    peer: float = 0.0
    peer_time: float = 0.4
    #: Judge a close look against the others of the mission rather than
    #: against zero: land when it is liked by more than ``relative`` above
    #: the middle (median) of the close looks it has turned down so far --
    #: at least ``compared`` of them -- and before calling the mission done,
    #: go back to the best one it turned down if that stands out so now. A
    #: brain taught on camouflaged objects values its targets below zero up
    #: close, but above the rest. Replayed over every close look of 18
    #: practice missions it was a trade, not a gain -- each target more
    #: cost one or two wrong landings: in a mission the target is often not
    #: the best looking thing, with columns among the poles and slabs among
    #: the balls. 0: never.
    relative: float = 0.0
    compared: int = 2
    #: Practice: land after a close look even when it is not liked, to find
    #: out. A rejected close look teaches nothing -- no landing, no dopamine
    #: -- so without this, practice could never correct a close look that
    #: wrongly turns a target down.
    land_to_learn: bool = False
    rejected: list = field(default_factory=list)
    #: What the world gives the fly where it lands, ``reward(x, y)`` ->
    #: dopamine (+ reward, - punishment, 0 nothing), or None for no learning
    #: in flight. The sugar on the target in the real experiment.
    reward: object = None
    phase: str = "scan"
    sightings: list = field(default_factory=list)
    #: Before calling a mission done, go back to where it saw something it has
    #: not looked at, and look from there: an object at the edge of what the
    #: eye makes out is seen on one look around and not the next. Worth it
    #: only to a brain whose close looks are sure -- the practised drone made
    #: no wrong landing in 212 close looks, while for the fly the extra looks
    #: cost more wrong landings than they found targets.
    go_back: bool = False
    #: Every sighting of the mission, for ``go_back``.
    seen: list = field(default_factory=list)
    #: And with nothing left to look at, fly ``explore`` mm out into the
    #: widest stretch of the panorama with nothing seen in it -- more than 90
    #: degrees -- and look around there. A ball 78 mm away is too small for
    #: the eye from home; measured, the drone ended a mission with 37 s to go
    #: and a ball in a 106-degree gap it had never seen. 0: never.
    explore: float = 0.0
    visited: list = field(default_factory=list)
    lessons: list = field(default_factory=list)
    inspections: list = field(default_factory=list)  # (valence close up, landed on it)
    #: Every landing's views -- the one chosen on and those seen on the way
    #: in -- with the dopamine it brought: the material for replay.
    experiences: list = field(default_factory=list)
    #: Dopamine on the views seen on the way in, as a fraction of the one on
    #: the view chosen on; 0 learns from the chosen view only.
    trace_lessons: float = 0.0
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
        self._lateral = 0.0  # mm/s to the left
        self._turning = 0.0  # deg/s, from the heading look to look
        self._faced = False  # has this approach turned to face its target yet
        self._closer = False  # going closer for a second look
        self._last_heading = None
        self._since = 0.0
        self._lost = 0.0
        self._trace = []
        self._scan_view = None
        self._radii = []
        self._next_trace = 0.0
        self._inspected = False
        self._stopped = False
        self._looking = 0.0
        self._empty_looks = 0
        self._needed = 0.0  # what a close look must like to land
        self._home = (0.0, 0.0)
        self._spot = None  # where it looks around from, if not home
        self._spots = []  # the places it has gone back to, to look again
        self._views = []  # valences of the close looks at this target so far
        self._judged = []  # close looks turned down: [where from, heading, valence, gone back]
        self._pick = None  # the one it is going back to
        self._round_from = 0.0  # its heading when it set off round the object
        self._size = self._round_size = 0.0  # how big the target is in the eye: now, and then
        self.sideways = 0.0  # flying sideways, to the left: a roll read as a speed
        self.heading = 0.0

    def start(self, heading: float, home=(0.0, 0.0)) -> None:
        """Begin where the fly is: its heading, and the place it calls home."""
        self.heading = heading
        self._home = tuple(home)

    @property
    def inspecting(self) -> bool:
        """Stopped near the target for a close look."""
        return self.phase == "approach" and (self._stopped or self._looking > 0)

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
        radii = [*self._radii, *([30.0] * len(self.rejected))]
        for v, r in zip((*self.visited, *self.rejected), radii, strict=False):
            dx, dy = v[0] - here[0], v[1] - here[1]
            to = np.degrees(np.arctan2(dy, dx))
            cone = max(25.0, np.degrees(np.arctan2(r, np.hypot(dx, dy))))
            if abs((to - heading_deg + 180) % 360 - 180) < cone:
                return True
        return False

    def decide(self, fly_eye, state: dict, dt: float) -> tuple[float, float, float]:
        """One look: returns (pitch deg, heading deg, height mm)."""
        half = fly_eye.camera.fov / 2
        here = (state["x"], state["y"])
        heading_now = state["heading"]
        self._moving = float(np.hypot(state.get("vx", 0.0), state.get("vy", 0.0)))
        h = np.deg2rad(heading_now)
        self._lateral = -state.get("vx", 0.0) * np.sin(h) + state.get("vy", 0.0) * np.cos(h)
        if self._last_heading is not None and dt > 0:
            self._turning = (heading_now - self._last_heading) / dt
        self._last_heading = heading_now
        see = fly_eye.standing if self.skyline else fly_eye.eye.dark
        if self.phase in ("approach", "around"):
            # Only what is where the target should be counts as the target --
            # going round it, nearer still: the landscape wheels past behind.
            off = (self._target.heading - heading_now + 180) % 360 - 180
            centre = float(np.clip(off, -half, half)) / half
            track = TRACK if self.phase == "approach" else TRACK / 2
            size, bearing = see(DARK, within=track / half, centre=centre)
        else:
            # Only what is near the middle can be fixated and judged.
            size, bearing = see(DARK, within=FIXATE / half)
        bearing_deg = bearing * half  # +left
        if (
            self.phase == "approach"
            and not self._braking
            and size > 0.002
            and abs(bearing_deg) < 12.0
            and state["t"] >= self._next_trace
            and fly_eye.activity is not None
        ):
            # The way in: the target in the middle of the eye, nearer each time.
            self._trace.append(fly_eye.eye.fovea_shape(fly_eye.activity))
            self._next_trace = state["t"] + 0.05
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
            out = self._approach(size, bearing_deg, here, heading_now, dt, fly_eye)
        elif self.phase == "home":
            out = self._go_home(here, heading_now)
        elif self.phase == "around":
            out = self._go_around(size, bearing_deg, heading_now, dt)
        elif self.phase == "back":
            out = self._go_back(here, heading_now)
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
                self.seen.append(self.sightings[-1])
                self._fix, self._fovea = 0.0, []
                self._refr = self.refractory
            return 0.0, self.heading, self.height
        self._fix, self._fovea = 0.0, []
        return self._turn(dt)

    def _approach(self, size, bearing_deg, here, heading_now, dt, fly_eye=None):
        if self._brake > 0 or self._braking:
            self._brake -= dt
            settled = self.settle_speed is None or self._moving < self.settle_speed
            self._braking = not (self._brake <= 0 and (settled or self._brake < -1.5))
            if not self._braking:
                # Path integration: remember where the target is -- a little
                # ahead of where the fly stopped.
                h = np.deg2rad(self.heading)
                place = (here[0] + 15 * np.cos(h), here[1] + 15 * np.sin(h))
                visited = enumerate(self.visited)
                again = [i for i, v in visited if self._same_place(v, place, here)]
                if again:
                    # Back where it has been: no new lesson, and a wider berth
                    # from now on. Measured before this, the drone could land on
                    # one object a dozen times in a mission.
                    for i in again:
                        self._radii[i] = 60.0
                else:
                    self.visited.append(place)
                    self._radii.append(30.0)
                    self._learn_from(place)
                self._rescan()
            return -6.0, self.heading, self.height
        self._since += dt
        self._size = size
        off = (self._target.heading - heading_now + 180) % 360 - 180
        if abs(off) > 25 and (size < self.slow_size or not self._faced):
            # Not facing it yet: turn on the spot toward where it was. Until it
            # has faced it once, whatever is big at the edge of the eye is not
            # it -- measured, a drone standing by one object locked onto it
            # instead of the one it had chosen, and went back to it a dozen
            # times in a mission.
            self.heading = heading_now + np.clip(off, -30, 30)
            return 0.0, self.heading, self.height
        self._faced = True
        if size > 0.002:
            # Fixate it: the heading follows the object across the eye.
            self.heading = heading_now + 0.8 * bearing_deg
            self._target.heading = self.heading
            self._lost = 0.0
        else:
            self._lost += dt
        if self._lost > self.lost_after or self._since > self.give_up:
            # Lost it, or flown past it: stop, and look again. A look before
            # leaving that finds nothing there is not tried again.
            if self._needed > 0:
                h = np.deg2rad(self._target.heading)
                self.rejected.append((here[0] + 50 * np.cos(h), here[1] + 50 * np.sin(h)))
            self._rescan()
            return -6.0, self.heading, self.height
        look_at = self.closer_size if self._closer else self.inspect_size
        if self.inspect and not self._inspected and size >= look_at:
            return self._inspect_target(here, fly_eye, dt, bearing_deg, heading_now)
        if size >= self.arrive_size:
            self._brake, self._braking = 0.12, True
            return -6.0, self.heading, self.height
        push = self.cruise_pitch if size < self.slow_size else self.cruise_pitch * 0.35
        return push, self.heading, self.height

    def _same_place(self, visited, place, here) -> bool:
        """Is the place it has just reached one it had reached before?

        Within 40 mm, and ahead of it rather than behind: a place is
        remembered short of the object -- by 20-35 mm for a slab, where the
        drone stops further off -- so two slabs 50 mm apart were remembered
        as one, the second never counted, and the drone went back to it a
        dozen times in a mission. What it came from is behind it.
        """
        if np.hypot(visited[0] - place[0], visited[1] - place[1]) >= 40.0:
            return False
        dx, dy = visited[0] - here[0], visited[1] - here[1]
        if np.hypot(dx, dy) < 10.0:
            return True
        to = np.degrees(np.arctan2(dy, dx))
        return abs((to - self.heading + 180) % 360 - 180) < 60.0

    def _inspect_target(self, here, fly_eye, dt, bearing_deg=0.0, heading_now=None):
        """Stop at a good distance, face it, look at it again, and decide."""
        self._looking += dt
        if not self._stopped:
            # Stop first -- 0.1 s for the fly, and until it has really stopped
            # for a body with more momentum -- turned to face it squarely.
            if heading_now is not None:
                self.heading = heading_now + bearing_deg
            settled = self.settle_speed is None or self._moving < self.settle_speed
            facing = abs(bearing_deg) < self.centred and abs(self._turning) < self.still
            if (self._looking >= 0.1 and settled and facing) or self._looking > self.face_time:
                self._stopped, self._looking = True, 0.0
            return (-6.0 if self._looking < 0.1 else 0.0), self.heading, self.height
        if fly_eye is not None and fly_eye.activity is not None:
            self._fovea.append(fly_eye.eye.fovea_shape(fly_eye.activity))
        if self.peer:
            spin = -np.degrees(self._lateral / self.around_from)
            if heading_now is not None:
                self.heading = heading_now + bearing_deg + spin * 0.3
            self.sideways = self.peer
        look_for = self.peer_time if self.peer else self.fixate_time
        if self._looking < look_for or not self._fovea:
            return 0.0, self.heading, self.height
        f = np.mean(self._fovea[len(self._fovea) // 2 :], axis=0)
        self._fovea, self._looking, self._inspected = [], 0.0, True
        self._stopped = False
        self.sideways = 0.0
        v = float(self.mushroom_body.valence(f))
        # The closer look is the one the decision -- and any lesson -- rests on.
        if not self._closer and not self._views:
            self._scan_view = self._target.features
        self._target.features, self._target.valence = f, v
        if self.looks > 1 and not self.land_to_learn:
            self._views.append(v)
            if len(self._views) < self.looks:
                self._round_to(heading_now)
                return 0.0, self.heading, self.height
            v = float(np.mean(self._views))
            self._target.valence = v
        unsure = self._needed - self.unsure < v <= self._needed
        if unsure and not self._closer and not self.land_to_learn and self.looks == 1:
            # Not sure, and still far: go closer and look again.
            self._closer, self._inspected, self._since = True, False, 0.0
            return 0.0, self.heading, self.height
        if v <= self._needed and not self.land_to_learn and not self._stands_out(v):
            h = np.deg2rad(self.heading)
            self.rejected.append((here[0] + 50 * np.cos(h), here[1] + 50 * np.sin(h)))
            if self.relative:
                self._judged.append([tuple(here), float(self.heading), v, False])
            self.inspections.append((v, False))
            self._rescan()
            return 0.0, self.heading, self.height
        self.inspections.append((v, True))
        # The rest of the way in has its own time: the stop and the look took
        # some, and a drone giving up here, a few centimetres short, came back
        # and judged the object again from too close -- and turned it down.
        self._since = 0.0
        return 0.0, self.heading, self.height

    def _stands_out(self, v, others=None) -> bool:
        """Is this close look liked well above the ones it has turned down?"""
        if not self.relative:
            return False
        others = [j[2] for j in self._judged] if others is None else others
        return len(others) >= self.compared and v - float(np.median(others)) > self.relative

    def _judged_to_try(self):
        """The best close look turned down, if it stands out from the rest now."""
        left = [j for j in self._judged if not j[3]]
        if not left:
            return None
        best = max(left, key=lambda j: j[2])
        others = [j[2] for j in self._judged if j is not best]
        return best if self._stands_out(best[2], others) else None

    def _go_back(self, here, heading_now):
        """Back to where it looked from, then in to land: the decision is made."""
        (x, y), heading, v, _ = self._pick
        dx, dy = x - here[0], y - here[1]
        if np.hypot(dx, dy) < 10.0:
            self._target = Sighting(heading, v, 0.01, tuple(here))
            self._since, self._lost = 0.0, 0.0
            self._faced, self._inspected = False, True
            self.phase = "approach"
            return -6.0, self.heading, self.height
        want = np.degrees(np.arctan2(dy, dx))
        off = (want - heading_now + 180) % 360 - 180
        self.heading = heading_now + np.clip(off, -30, 30)
        if abs(off) > 25:
            return 0.0, self.heading, self.height
        near = np.hypot(dx, dy) < 30
        return self.cruise_pitch * (0.5 if near else 1.0), self.heading, self.height

    def _round_to(self, heading_now) -> None:
        """Set off round the object it is facing, to look at it from another side."""
        self._round_from, self._round_size = heading_now, self._size
        self._since = self._lost = 0.0
        self.phase = "around"

    def _go_around(self, size, bearing_deg, heading_now, dt):
        """Fly sideways, turning to keep the object in the middle of the eye.

        Facing it the whole way, it goes round it, at the distance it is: how
        far round is how far it has turned. Going round to a place worked out
        from where the object should be did not work: the eye gives no
        distance, and from a guessed one the drone came round to look at a
        column behind a camouflaged object, or at nothing, one look in two.
        """
        self._since += dt
        # Flying left, what it is going round wheels right in the eye: turn
        # ahead of it. Only following it, the drone lagged 5 degrees behind,
        # and a column wheeling past behind a camouflaged object took it over.
        spin = -np.degrees(self._lateral / self.around_from)
        if size > 0.002:
            self._target.heading = heading_now + bearing_deg + spin * dt
            self._lost = 0.0
        else:
            self._target.heading += spin * dt
            self._lost += dt
        self.heading = self._target.heading + spin * 0.3
        turned = abs(heading_now - self._round_from)
        going = self._since < 2 * self.give_up and self._lost < self.lost_after
        if turned < self.around and going:
            self.sideways = self.side_pitch
            # Lagging behind it, it drifts outward: closer again when it has
            # shrunk, but not so close that it cannot come in to look. Backing
            # off further, to come in as the first time, other objects and
            # columns came into view and it went to them instead.
            closer = size < 0.7 * self._round_size
            return (self.cruise_pitch * 0.35 if closer else 0.0), self.heading, self.height
        # Round: in to it, to stop and look as the first time. Looking from
        # where it came round to, still leaning out of the sideways flight
        # or hardly moving in, the drone judged everything worse than from
        # the front.
        self.sideways = 0.0
        self.phase = "approach"
        self._faced, self._inspected, self._stopped = True, False, False
        self._looking, self._since, self._lost = 0.0, 0.0, 0.0
        return 0.0, self.heading, self.height

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
            if self._scan_view is not None and self._scan_view is not self._target.features:
                # And onto the far view it was chosen on, before the close
                # look. Without it a mistake was punished close up only, the far
                # view still looked good on the next turn, and the drone went
                # back: up to sixteen landings on one object in a practice mission.
                self.mushroom_body.learn(self._scan_view, dopamine)
            if self.trace_lessons and self._trace:
                # Dopamine reaches every Kenyon cell still carrying a trace of
                # recent activity -- the way in, seconds back, not only the
                # moment of choice.
                share = self.trace_lessons * dopamine / len(self._trace)
                for f in self._trace:
                    self.mushroom_body.learn(f, share)
            self.lessons.append((self._target.heading, dopamine, place))
            self.experiences.append(
                dict(chosen=self._target.features, way_in=list(self._trace), dopamine=dopamine)
            )

    def _base(self) -> tuple:
        """Where it looks around from: home, or a place it went back to."""
        return self._spot if self._spot is not None else self._home

    def _go_home(self, here, heading_now):
        base = self._base()
        dx, dy = base[0] - here[0], base[1] - here[1]
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
        elif np.hypot(here[0] - self._base()[0], here[1] - self._base()[1]) > 20.0:
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
                if self.inspect:
                    self._needed = self.sure
                else:
                    self._inspected = True  # it goes to find out, whatever it looks like
            elif self.look_before_leaving and self.inspect and unvisited:
                self._target = max(unvisited, key=lambda s: s.valence)
                self._since = 0.0
                self.phase = "approach"
                self._needed = self.sure
            elif self.go_back and self.inspect and (spot := self._spot_to_try(here)):
                # Something seen from elsewhere and never looked at: go back
                # to where it was seen from, and look from there.
                self._spot = spot
                self._spots.append(spot)
                self.phase = "home"
            elif self.relative and (pick := self._judged_to_try()):
                self._pick = pick
                pick[3] = True
                self.phase = "back"
            elif self.explore and (spot := self._gap_to_explore()):
                self._spot = spot
                self._spots.append(spot)
                self.phase = "home"
            else:
                self.phase = "done"

    def _gap_to_explore(self):
        """A place out in the widest stretch of the panorama with nothing in it."""
        hx, hy = self._home
        headings = [
            s.heading for s in self.seen if np.hypot(s.where[0] - hx, s.where[1] - hy) < 20
        ]
        for x, y in (*self.visited, *self.rejected, *self._spots):
            headings.append(np.degrees(np.arctan2(y - hy, x - hx)))
        if not headings:
            return None
        h = np.sort(np.mod(headings, 360.0))
        gaps = np.diff(np.append(h, h[0] + 360.0))
        i = int(np.argmax(gaps))
        if gaps[i] <= 90.0:
            return None
        mid = np.deg2rad(h[i] + gaps[i] / 2)
        return (hx + self.explore * np.cos(mid), hy + self.explore * np.sin(mid))

    def _spot_to_try(self, here):
        """Where it saw something it has not looked at, if it has not been back there."""
        best = None
        for s in self.seen:
            w = s.where
            if np.hypot(w[0] - here[0], w[1] - here[1]) < 20.0:
                continue
            if any(np.hypot(w[0] - p[0], w[1] - p[1]) < 20.0 for p in self._spots):
                continue
            if self._already_visited(w, s.heading):
                continue
            if best is None or s.valence > best.valence:
                best = s
        return None if best is None else tuple(best.where)

    def _rescan(self) -> None:
        self.phase = "scan"
        self._scanned = 0.0
        self._refr = 0.0
        self._fix, self._fovea = 0.0, []
        self._lost = 0.0
        self._trace = []
        self._scan_view = None
        self._inspected, self._looking, self._stopped = False, 0.0, False
        self._needed = 0.0
        self._faced = False
        self._closer = False
        self._views = []
        self.sideways = 0.0
        self.sightings = []


def describe(phase: str, who: str = "Muszka") -> str:
    """The phase in Polish, for the fly (she) or the drone (he)."""
    return {
        "scan": "rozgląda się",
        "approach": "leci do celu",
        "home": "wraca na start, by rozejrzeć się stamtąd",
        "around": "obchodzi cel, by obejrzeć go z innej strony",
        "back": "wraca do najlepszego z obejrzanych",
        "done": "zwiedziła wszystkie cele" if who == "Muszka" else "zwiedził wszystkie cele",
    }.get(phase, phase)


def nearest(items, x: float, y: float) -> tuple[Item, float]:
    best = min(items, key=lambda it: it.distance_from(x, y))
    return best, best.distance_from(x, y)


__all__ = [
    "KINDS",
    "LANDMARK",
    "Navigator",
    "Sighting",
    "describe",
    "nearest",
    "teach",
    "tethered_training",
]
