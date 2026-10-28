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
from .navigate import LANDMARK, Navigator, describe, nearest, painted, teach, tethered_training

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


def shipped(
    every_ms: float = 5.0, close: bool = False, drone: bool = False, camo: bool = False
) -> Path:
    """The shipped views for an eye stepped every ``every_ms`` (5, or 20 in the fast
    mode): the scanning ones, or with ``close`` the close looks before landing,
    flown (:func:`flown_close_looks`); with ``drone``, the drone's."""
    name = "tethered_views" + ("" if every_ms == 5.0 else f"_{every_ms:g}ms")
    tail = (
        ("_close" if close else "") + ("_dron" if drone else "") + ("_kamuflaz" if camo else "")
    )
    return SHIPPED.with_name(name + tail + ".npz")


def _signature(
    samples: int,
    seed: int,
    every_ms: float = 5.0,
    close: bool = False,
    drone: bool = False,
    camo: bool = False,
) -> str:
    """What the views depend on; a saved file with another signature is stale."""
    sizes = {k: v["size"] for k, v in arena.KINDS.items()}
    eye = "" if every_ms == 5.0 else f" eye {every_ms:g} ms"
    body = " drone 90 deg/s" if drone else ""
    if camo:
        body += " camouflage 0-1, columns, skyline"
        if close and drone and CAMO_LOOKS > 1:
            body += f", {CAMO_LOOKS} looks {AROUND:g} deg round"
        if close and drone and CAMO_PEER:
            body += f", peering {CAMO_PEER:g} for {CAMO_PEER_TIME:g} s"
    if close:
        return f"v7 flown close looks {samples} {seed} {sizes}{eye}{body}"
    return f"v3 {samples} {seed} lead 120-180 {sizes}{eye}{body}"


def views(
    fly_eye,
    folder=".",
    samples: int = 90,
    seed: int = 0,
    fresh=False,
    progress=print,
    close: bool = False,
    drone: bool = False,
    camo: bool = False,
):
    """The tethered fly's look at every kind: (features, kinds), from disk if seen before.

    ``camo``: the camouflage lesson -- every object painted in the floor's
    checker by a random amount, the landscape's columns shown as a kind of
    their own (:data:`~.navigate.LANDMARK`), and the close looks flown seeing
    what breaks the skyline.

    ``close``: the close looks it takes before landing, flown
    (:func:`flown_close_looks`, ``samples`` of them), rather than the views
    it judges while scanning. ``drone``: the drone's lesson -- turning at its
    90 deg/s, and its own close looks. Taught the fly's way, the drone's brain
    called 42% of the balls it saw in flight balls; taught its own way, 68%.
    """
    every_ms = getattr(fly_eye, "every_ms", 5.0)
    path = Path(folder) / VIEWS
    if every_ms != 5.0:
        path = path.with_name(path.stem + f"_{every_ms:g}ms.npz")
    if close:
        path = path.with_name(path.stem + "_close.npz")
    if drone:
        path = path.with_name(path.stem + "_dron.npz")
    if camo:
        path = path.with_name(path.stem + "_kamuflaz.npz")
    sig = _signature(samples, seed, every_ms, close, drone, camo)
    for saved in (path, shipped(every_ms, close, drone, camo)):
        if saved.exists() and not fresh:
            with np.load(saved) as d:
                if str(d["signature"]) == sig:
                    if progress:
                        progress(f"Muszka już oglądała te obiekty na uwięzi: wczytuję {saved}")
                    return d["F"], d["K"]
    if progress:
        progress(
            (
                "Muszka podlatuje do każdego rodzaju obiektu i ogląda go z bliska"
                if close
                else "Muszka na uwięzi ogląda każdy rodzaj obiektu z różnych stron i odległości"
            )
            + " (kilka minut, tylko za pierwszym razem)..."
        )
    t0 = time.time()
    kinds = tuple(arena.KINDS) + ((LANDMARK,) if camo else ())
    if close:
        # In parts of CLOSE_LOOKS, each from its own seed: a bigger lesson
        # keeps the smaller one as its first part, and the parts can be flown
        # side by side (:func:`close_part`).
        parts = [
            close_part(
                fly_eye, kinds, seed, j, samples, drone=drone, progress=progress, camo=camo
            )
            for j in range(-(-samples // CLOSE_LOOKS))
        ]
        F, K = np.concatenate([f for f, _ in parts]), np.concatenate([k for _, k in parts])
    else:
        extra = {"camo": camo}
        if drone:
            from .dronebrain import SCAN_RATE

            extra["scan_rate"] = SCAN_RATE
        F, K = tethered_training(
            fly_eye, kinds, samples=samples, seed=seed, progress=progress, **extra
        )
    if progress:
        progress(f"  {len(K)} widoków w {time.time() - t0:.0f} s -> {path}")
    np.savez_compressed(path, F=F, K=K, signature=np.array(sig))
    return F, K


#: The close-look lesson: how many close looks are flown for it, a third of
#: each kind, and the share of the tether's dopamine each one gets. Taught at
#: the full dose they pulled the ball's far view down -- the fly liked 23% of
#: the balls it saw while looking around, against 84% -- and at a tenth they
#: judge new close looks as well (97% of the targets liked, 0-2% of the rest).
CLOSE_LOOKS = 300
CLOSE_DOSE = 0.1
#: The camouflage close-look lesson is three times as big, flown in parts
#: of :data:`CLOSE_LOOKS` (seeds 0, 1, 2), and taught at twice the dose.
#: What tells camouflaged objects apart is subtle, and more of it is
#: learned from more looks: on 120 new flown looks at full camouflage, the
#: ball was told apart at AUC 0.86 after 300 and 0.93 after 900, the slab
#: 0.94 and 0.96, the pole 0.94 and 0.96; and more targets were liked
#: outright -- 83% of the balls right at zero against 68%. 1800 told them
#: apart a little better again, and in missions found no more.
CAMO_CLOSE_LOOKS = 900
#: And it is taught the way that, of the machine-learning techniques tried
#: (docs/uczenie_maszynowe.md), told camouflaged objects apart best: dopamine
#: as a prediction error, three passes (``aim``, ``epochs`` of
#: :func:`~.navigate.teach`), a sparser Kenyon-cell code -- 3% active
#: rather than 5% -- and the close looks at twice the dose. Chosen on 1500
#: flown looks kept for the purpose, and checked once on 240 others: at full
#: camouflage the slab was told apart at AUC 0.98 instead of 0.96, the pole
#: 0.99 instead of 0.96, the ball 0.94 as before; at half, all three better.
CAMO_CLOSE_DOSE = 0.4
CAMO_TEACHING = {"sparseness": 0.03, "aim": 0.3, "epochs": 3}
#: Its third lesson: the close looks the drone takes in missions, in 100
#: practice arenas (:func:`mission_looks`). The lessons above are flown one
#: object at a time; in a mission the drone looks from wherever it is, at
#: whatever it chose, and half of what it looks at closely is a landscape
#: column. Judged on looks from other missions, the brains taught on the
#: lessons alone told targets apart worse there than on the lessons' kind of
#: look (AUC 0.90-0.96 against 0.95-0.99) and liked 14-29% of the columns --
#: dataset shift, in machine-learning terms. Taught these looks as well
#: (dataset aggregation), AUC 0.93 for the ball, 0.92 for the slab and 0.98
#: for the pole, and 9-22% of the columns liked.
MISSION_ARENAS = (4000, 4100)
MISSION_PART = 25
#: In the camouflage world a punishment gets this share of a reward's
#: dopamine: practising lands on about three non-targets -- columns
#: included -- for every target, camouflaged objects look alike, and at full
#: dose the punishments spread onto the targets: after 25 practice missions
#: the brain turned nearly everything down, targets with the rest.
CAMO_PUNISH = 0.35
#: In the camouflage world the drone can look at an object from several
#: sides before it decides, going :data:`AROUND` degrees round it between
#: looks (:attr:`~.navigate.Navigator.looks`); its close-look lesson is
#: then flown the same way, every look of it, ``CAMO_LOOKS`` times as long.
#: Measured, it did not pay, and it looks once: see the README.
CAMO_LOOKS = 1
AROUND = 45.0
#: And it peers (:attr:`~.navigate.Navigator.peer`) for ``CAMO_PEER_TIME``
#: seconds: 0 looks hovering still.
CAMO_PEER = 0.0
CAMO_PEER_TIME = 0.4


def close_looks(camo: bool = False) -> int:
    """How many close looks the close-look lesson is made of."""
    return CAMO_CLOSE_LOOKS * CAMO_LOOKS if camo else CLOSE_LOOKS


class _Looker:
    """A mushroom body that likes everything, and keeps the views of its close looks."""

    def __init__(self):
        self.nav, self.views = None, []

    def valence(self, f) -> float:
        if self.nav.phase == "approach":
            self.views.append(np.asarray(f, dtype=np.float32))
        return 0.5

    def learn(self, f, dopamine) -> None:
        pass


def close_part(fly_eye, kinds, seed, j, samples, **kw):
    """Part ``j`` of a close-look lesson of ``samples`` looks: (features, kinds)."""
    n = min(CLOSE_LOOKS, samples - j * CLOSE_LOOKS)
    return flown_close_looks(fly_eye, kinds, n, seed=seed + j, **kw)


def flown_close_looks(
    fly_eye, kinds, samples, seed=0, drone=False, progress=None, tries=20.0, camo=False
):
    """The close-look lesson flown, not replayed: (features, kinds).

    The body itself -- the fast fly, or the drone -- with its own navigator,
    in a small arena: one object 55-95 mm away, 60-150 degrees off its
    heading, two others off to the sides. It turns to the object, flies at
    it, stops and takes its close look exactly as it will in a mission, and
    that view, with what the object really is, is the lesson.

    The tether's replay of the same approach -- turn, nose-down run-in, brake,
    hover -- taught a brain that judged the replay perfectly and real close
    looks badly: on 98 the drone flew, it liked 27% of the slabs and 48% of
    the poles it should have, and the fly 15% of the balls. What the replay
    could not reproduce was the body: the drone looks while it is still
    settling, leaning against its drift, sometimes still turning. Taught on
    flown close looks, new flown ones are judged right 94-100% of the time.

    ``camo``: the objects painted in the floor's checker by a random amount,
    the landscape's columns flown to as a kind of their own, the body
    seeing what breaks the skyline, and every one of the drone's
    :data:`CAMO_LOOKS` looks from different sides a lesson -- taught on its
    first looks only, its later ones were judged far worse (a ball's second,
    AUC 0.46 against 0.81).
    """
    from .navigate import Sighting

    rng = np.random.default_rng(seed)
    world = fly_eye.world
    F, K = [], []
    i = 0
    while len(F) < samples:
        kind = kinds[i % len(kinds)]
        i += 1
        d = rng.uniform(55.0, 95.0)
        h = np.deg2rad(rng.uniform(60.0, 150.0) * rng.choice([-1, 1]))
        if kind == LANDMARK:
            # A column of the landscape within reach of the start instead.
            near = [
                b.position for b in fly_eye.background if 55 < np.hypot(*b.position[:2]) < 95
            ]
            x, y = near[rng.integers(len(near))][:2]
            h, items = np.arctan2(y, x), []
        else:
            x, y = round(d * np.cos(h), 1), round(d * np.sin(h), 1)
            items = [arena.Item(kind, x, y, camo=painted(rng, camo))]
        for other in rng.choice(tuple(arena.KINDS), 2):
            b = h + np.deg2rad(rng.uniform(90, 150)) * rng.choice([-1, 1])
            r = rng.uniform(40, 90)
            ox, oy = round(r * np.cos(b), 1), round(r * np.sin(b), 1)
            items.append(arena.Item(str(other), ox, oy, camo=painted(rng, camo)))
        fly_eye.set_items(tuple(items))
        looker = _Looker()
        pilot = _pilot(
            items, looker, fly_eye, {}, {kind}, drone, fast=True, camo=camo, pace=False
        )
        looker.nav = nav = pilot.nav
        nav.reward, nav.inspect, nav.curious = None, True, False
        pilot.step()  # spun up, and facing away from it
        s = pilot.flight.state()
        bearing = np.degrees(np.arctan2(y - s["y"], x - s["x"]))
        nav._target = Sighting(float(bearing), 0.5, 0.01, (s["x"], s["y"]))
        nav._since, nav.phase = 0.0, "approach"
        looks = nav.looks
        while (
            len(looker.views) < looks
            and nav.phase in ("approach", "around")
            and pilot.flight.t < tries * looks
        ):
            pilot.step()
        # Lost it on the way in: no lesson.
        new = looker.views[:looks]
        F += new
        K += [kind] * len(new)
        if progress and new and len(F) // 30 > (len(F) - len(new)) // 30:
            progress(f"  obejrzała z bliska {len(F)} z {samples}")
    fly_eye.world = world
    fly_eye.eye.reset()
    return np.asarray(F)[:samples], np.asarray(K)[:samples]


def mission_looks(fly_eye, arenas, progress=None):
    """Close looks the camouflage drone takes in missions, labelled: (features, kinds).

    It flies a mission in each arena -- camouflage drawn 0.3-1 -- turning
    every close look down, so it goes on to look at everything it would go
    to. Each look is labelled with what is really there, an object or a
    landscape column, when that lies within 12 degrees of where it faces.
    The brain flying is the one taught on the other two lessons.
    """
    from .watch import COLUMNS

    F, K = views(fly_eye, "/nonexistent", progress=None, drone=True, camo=True)
    close = views(
        fly_eye,
        "/nonexistent",
        close_looks(True),
        close=True,
        progress=None,
        drone=True,
        camo=True,
    )
    mb, valence = teach_targets(
        F, K, {"bar"}, close=close, dose=CAMO_CLOSE_DOSE, **CAMO_TEACHING
    )
    columns = [arena.Item(LANDMARK, float(x), float(y)) for x, y, _ in COLUMNS]
    world = fly_eye.world
    looks, kinds = [], []

    class Looker:
        def valence(self, f):
            v = mb.valence(f)
            if pilot.nav.phase != "approach":
                return v
            st = pilot.flight.state()
            things = list(pilot.items) + columns
            near = [q for q in things if q.distance_from(st["x"], st["y"]) < 120] or things

            def off(q):
                return abs((q.bearing_from(st["x"], st["y"]) - st["heading"] + 180) % 360 - 180)

            q = min(near, key=off)
            if off(q) < 12.0:
                looks.append(np.asarray(f, dtype=np.float32))
                kinds.append(q.kind)
            return v - 10.0  # turned down, whatever it is

        def learn(self, f, dopamine):
            pass

    for seed in range(*arenas):
        level = float(np.random.default_rng(seed).uniform(0.3, 1.0))
        items = arena.mission(seed, per_kind=2, camo=level)
        fly_eye.set_items(items)
        pilot = _pilot(
            items, Looker(), fly_eye, valence, {"bar"}, True, fast=True, camo=True, pace=False
        )
        pilot.nav.reward = None
        while pilot.flight.t < MISSION_S[True] and not pilot.done:
            pilot.step()
        if progress:
            progress(f"  misja {seed}: {len(looks)} spojrzeń z bliska")
    fly_eye.world = world
    fly_eye.eye.reset()
    return np.asarray(looks), np.asarray(kinds)


def mission_views(fly_eye, folder=".", fresh=False, progress=print):
    """The mission close looks (:func:`mission_looks`), from disk if taken before."""
    every_ms = getattr(fly_eye, "every_ms", 5.0)
    name = f"_{every_ms:g}ms_dron_kamuflaz.npz"
    path = Path(folder) / ("flylab_spojrzenia_z_misji" + name)
    sizes = {k: v["size"] for k, v in arena.KINDS.items()}
    lesson = _signature(close_looks(True), 0, every_ms, True, True, True)
    sig = (
        f"v1 mission looks, arenas {MISSION_ARENAS[0]}-{MISSION_ARENAS[1]} in parts of "
        f"{MISSION_PART}, camouflage 0.3-1, labelled within 12 deg {sizes}; brain: {lesson}, "
        f"{CAMO_TEACHING}, dose {CAMO_CLOSE_DOSE}"
    )
    for saved in (path, SHIPPED.with_name("mission_looks" + name)):
        if saved.exists() and not fresh:
            with np.load(saved) as d:
                if str(d["signature"]) == sig:
                    return d["F"], d["K"]
    if progress:
        progress("Dron lata misje treningowe i zbiera spojrzenia z bliska (długo, raz)...")
    parts = [
        mission_looks(fly_eye, (a, min(a + MISSION_PART, MISSION_ARENAS[1])), progress)
        for a in range(*MISSION_ARENAS, MISSION_PART)
    ]
    F, K = np.concatenate([f for f, _ in parts]), np.concatenate([k for _, k in parts])
    np.savez_compressed(path, F=F, K=K, signature=np.array(sig))
    return F, K


def teach_targets(
    F,
    K,
    targets,
    seed: int = 0,
    close=None,
    dose: float = CLOSE_DOSE,
    sparseness: float = 0.05,
    aim: float | None = None,
    epochs: int = 1,
):
    """Dopamine for the targets, punishment for the rest: (mushroom body, valence per kind).

    ``close``: the flown close looks, (features, kinds), taught on top at
    ``dose`` of the dopamine (:data:`CLOSE_DOSE`). ``sparseness``: the share
    of Kenyon cells active; ``aim``, ``epochs``: dopamine as a prediction
    error (:func:`~.navigate.teach`).
    """
    from .learning import MushroomBody

    kinds = tuple(arena.KINDS)
    seen = {str(k) for k in K} | (set(map(str, close[1])) if close is not None else set())
    reward = {k: (1.0 if k in targets else -1.0) for k in (*kinds, *sorted(seen - set(kinds)))}
    mb = MushroomBody(F.shape[1], seed=seed, on_off=True, sparseness=sparseness)
    how = {"aim": aim, "epochs": epochs, "seed": seed}
    if close is None:
        teach(mb, F, K, reward, **how)
    else:
        mb.calibrate(np.concatenate([F, close[0]]))
        teach(mb, F, K, reward, calibrate=False, **how)
        teach(mb, *close, reward, dose=dose, calibrate=False, **how)
    valence = {k: float(np.mean([mb.valence(f) for f in F[K == k]])) for k in kinds}
    return mb, valence


def learned(
    fly_eye,
    targets,
    folder=".",
    retrain: bool = False,
    progress=print,
    drone: bool = False,
    camo: bool = False,
):
    """What the brain knows about these targets.

    The memory left by practice flights (:func:`practise`) if there is one for
    these targets, this body and this eye; otherwise a brain trained here and
    shipped (:func:`shipped_memory`), if there is one; otherwise the tethered
    lesson. ``retrain`` ignores them all and starts again from the tether.
    ``camo``: the brain for camouflaged targets, its own lessons and memory.
    """
    look = dict(fresh=retrain, progress=progress, drone=drone, camo=camo)
    F, K = views(fly_eye, folder, **look)
    close = views(fly_eye, folder, close_looks(camo and drone), close=True, **look)
    every_ms = getattr(fly_eye, "every_ms", 5.0)
    path = memory_path(targets, drone, every_ms, folder, camo)
    if not path.exists():
        path = shipped_memory(targets, drone, every_ms, camo)
    if path.exists() and not retrain:
        from .learning import load

        mb = load(path)
        with np.load(path) as d:
            flown = int(d["missions"]) if "missions" in d else 0
        mb.practised = flown
        if progress:
            progress(f"Wczytuję pamięć z treningu: {path} ({flown} misji treningowych)")
        kinds = tuple(arena.KINDS)
        valence = {k: float(np.mean([mb.valence(f) for f in F[K == k]])) for k in kinds}
        return mb, valence
    if camo and drone:
        M = mission_views(fly_eye, folder, fresh=retrain, progress=progress)
        close = (np.concatenate([close[0], M[0]]), np.concatenate([close[1], M[1]]))
        return teach_targets(F, K, targets, close=close, dose=CAMO_CLOSE_DOSE, **CAMO_TEACHING)
    return teach_targets(F, K, targets, close=close)


def shipped_memory(
    targets, drone: bool = False, every_ms: float = 5.0, camo: bool = False
) -> Path:
    """The brain practised here and shipped, for these targets, body and eye rate.

    The drone's three, one target each, in the fast mode: 50 practice
    missions, and in 24 exam arenas -- twelve of them never used while the
    navigator was being put right -- 144 targets of 144, no wrong landing.
    """
    return SHIPPED.with_name(memory_path(targets, drone, every_ms, camo=camo).name)


def memory_path(
    targets, drone: bool = False, every_ms: float = 5.0, folder=".", camo: bool = False
) -> Path:
    """Where practice flights leave what was learned, per targets, body and eye rate."""
    names = "+".join(POLISH[t] for t in sorted(targets))
    names = names.replace("ł", "l").replace("ó", "o")
    body = ("_dron" if drone else "") + ("_kamuflaz" if camo else "")
    return Path(folder) / f"pamiec_{names}{body}_{every_ms:g}ms.npz"


def _remember(mb, path: Path, missions: int, curve) -> None:
    """Save a practised brain, its practice count and its record, all at once."""
    from .learning import save

    part = path.with_name(path.stem + ".part.npz")
    save(mb, part)
    with np.load(part) as d:
        everything = dict(d)
    np.savez_compressed(part, **everything, missions=np.array(missions), curve=np.array(curve))
    part.replace(path)  # never half a memory on disk


#: Dopamine of a lesson in flight, as a fraction of the tether's full dose.
LESSON = 1.0
#: Dopamine on the views seen on the way in, as a fraction of the lesson's.
WAY_IN = 0.5


def practise(
    targets,
    missions: int = 15,
    drone: bool = False,
    fast: bool = True,
    folder=".",
    progress=print,
    first_arena: int = 1000,
    camo: bool = False,
) -> list:
    """Practice flights: missions in new arenas, learning from every landing.

    ``camo``: the brain for camouflaged targets, practising in arenas painted
    0-1 in the floor's checker, a different amount each mission, where
    landing on a landscape column is a mistake like any other.

    The brain is the same from one mission to the next, so what it learns
    where it lands -- reward on a target, punishment on anything else --
    accumulates, on the views it actually has in flight rather than on the
    tether's. Picks up from earlier practice if there was some. Returns, per
    mission, (targets found, mistakes, seconds flown).
    """
    from .watch import FlysEye

    eye = FlysEye(items=(), every_ms=EYE_MS[fast])
    mb, valence = learned(
        eye, targets, folder=folder, progress=progress, drone=drone, camo=camo
    )
    path = memory_path(targets, drone, eye.every_ms, folder, camo)
    done, curve = 0, []
    if path.exists():
        with np.load(path) as d:
            done = int(d["missions"]) if "missions" in d else 0
            curve = [tuple(r) for r in d["curve"].tolist()] if "curve" in d else []
    who = "Dron" if drone else "Muszka"
    for i in range(missions):
        # Practice arenas are numbered from 1000 up, apart from the ones
        # missions are flown in (0, 1, 2 ...), so the test is on new ground.
        seed = first_arena + done + i
        level = float(np.random.default_rng(seed).uniform()) if camo else 0.0
        items = arena.mission(seed, per_kind=2, camo=level)
        eye.set_items(items)
        pilot = run(
            items,
            mb,
            eye,
            valence,
            targets,
            seconds=MISSION_S[drone],
            report=None,
            drone=drone,
            fast=fast,
            practice=True,
            camo=camo,
        )
        wanted = [it for it in items if it.kind in targets]
        reached = pilot.reached()
        row = (
            sum(it in reached for it in wanted),
            sum(it not in wanted for it in reached) + pilot.on_landmarks(),
            round(pilot.flight.t, 1),
        )
        curve.append(row)
        mb.practised = done + i + 1
        # Saved after every mission: a practice interrupted loses one
        # mission, not all of them, and the next one picks up from there.
        _remember(mb, path, done + i + 1, curve)
        if progress:
            progress(
                f"  misja treningowa {done + i + 1}: {row[0]} z {len(wanted)} celów,"
                f" pomyłki: {row[1]}  ({row[2]} s lotu)"
            )
    if progress:
        progress(f"{who}: pamięć zapisana w {path}")
    return curve


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
        landmarks=(),
        punish: float = 1.0,
        **navigator,
    ):
        from .watch import Flight

        self.items = tuple(items)
        #: (x, y) of the landscape's columns, where landing counts as a
        #: mistake: in the camouflage world, where the eye sees them too.
        self.landmarks = tuple(landmarks)
        #: The punishment's share of a reward's dopamine.
        self.punish = punish
        self.eye = eye
        # The fly, or anything with its interface -- the drone of
        # :mod:`.dronebrain` is one.
        self.flight = vehicle if vehicle is not None else Flight(height_mm=height)
        self.who = getattr(self.flight, "who", "Muszka")
        self.valence = valence or {}
        liked = {k for k, v in self.valence.items() if v > 0}
        self.targets = set(targets) if targets else liked
        navigator.setdefault("trace_lessons", WAY_IN)
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
        if hasattr(f, "strafe"):
            f.strafe(self.nav.sideways)
        f.step(self.every)
        if (
            not self.trail
            or np.hypot(s["x"] - self.trail[-1][0], s["y"] - self.trail[-1][1]) > 1
        ):
            self.trail.append((round(s["x"], 1), round(s["y"], 1)))

    def reward(self, x: float, y: float) -> float:
        """The world's answer where the fly lands: reward on a target, punishment elsewhere."""
        if any(np.hypot(x - lx, y - ly) < 30.0 for lx, ly in self.landmarks):
            return -LESSON * self.punish
        item, d = nearest(self.items, x, y)
        if d > 30.0:
            return 0.0
        return LESSON if item.kind in self.targets else -LESSON * self.punish

    def reached(self) -> list:
        """The items the fly has flown to, in order."""
        out = []
        for vx, vy in self.nav.visited:
            item, d = nearest(self.items, vx, vy)
            if d < 30.0:
                out.append(item)
        return out

    def on_landmarks(self) -> int:
        """Landings on a landscape column rather than near any object: mistakes."""
        return sum(
            1
            for vx, vy in self.nav.visited
            if nearest(self.items, vx, vy)[1] >= 30.0
            and any(np.hypot(vx - lx, vy - ly) < 30.0 for lx, ly in self.landmarks)
        )

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
            "who": self.who,
            "phase": nav.phase,
            "phase_pl": "ogląda cel z bliska"
            if nav.inspecting
            else describe(nav.phase, self.who),
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
    fast: bool = False,
    practice: bool = False,
    camo: bool = False,
):
    """The mission without a window: returns the :class:`Autopilot` when it ends.

    ``practice``: it lands after every close look, liked or not, to learn.
    ``camo``: the brain for camouflaged targets, seeing what breaks the skyline.
    """
    pilot = _pilot(items, mushroom_body, eye, valence, targets, drone, fast, camo)
    if practice:
        pilot.nav.inspect = pilot.nav.land_to_learn = True
        pilot.nav.look_before_leaving = True
    last = None
    while pilot.flight.t < seconds and not pilot.done:
        pilot.step()
        if not pilot.flight.upright():
            report(f"{pilot.who}: utrata panowania nad lotem.")
            break
        if report and pilot.nav.phase != last:
            s = pilot.flight.state()
            report(
                f"  {s['t']:5.2f} s  {describe(pilot.nav.phase, pilot.who)}"
                f"  ({s['x']:.0f}, {s['y']:.0f}) mm"
            )
            last = pilot.nav.phase
    return pilot


#: Longest mission, seconds of flight: the fly, and the drone -- slower to
#: turn and to stop, a third of its exam missions ran out of a minute (25 of 72).
MISSION_S = {False: 30.0, True: 90.0}
#: Grades: (name, least share of targets found, most mistakes per mission).
GRADES = (("ekspert", 0.9, 0.2), ("zaawansowany", 0.75, 0.5), ("uczeń", 0.0, 99.0))


def exam(
    targets,
    arenas: int = 6,
    drone: bool = False,
    fast: bool = True,
    folder=".",
    progress=print,
    camo: float | None = None,
    first: int = 0,
) -> dict:
    """Missions in arenas it never practised in, with no learning: what it knows now.

    Practice arenas are numbered from 1000 up and the exam's from 0, so the
    exam is always on new ground; the brain is loaded fresh for each mission
    and learns nothing during the exam. ``camo``: the brain for camouflaged
    targets, in arenas painted that much (0-1) in the floor's checker.
    ``first``: the first arena.
    """
    from .watch import FlysEye

    eye = FlysEye(items=(), every_ms=EYE_MS[fast])
    rows = []
    hidden = camo is not None
    for seed in range(first, first + arenas):
        mb, valence = learned(
            eye, targets, folder=folder, progress=None, drone=drone, camo=hidden
        )
        items = arena.mission(seed, per_kind=2, camo=camo or 0.0)
        eye.set_items(items)
        pilot = _pilot(items, mb, eye, valence, targets, drone, fast, hidden)
        pilot.nav.reward = None  # an exam: no dopamine
        while pilot.flight.t < MISSION_S[drone] and not pilot.done:
            pilot.step()
        wanted = [it for it in items if it.kind in targets]
        reached = pilot.reached()
        rows.append(
            (
                sum(it in reached for it in wanted),
                sum(it not in wanted for it in reached) + pilot.on_landmarks(),
                len(wanted),
            )
        )
        if progress:
            progress(
                f"  arena {seed}: {rows[-1][0]} z {rows[-1][2]} celów, pomyłki: {rows[-1][1]}"
            )
    found = sum(r[0] for r in rows) / max(1, sum(r[2] for r in rows))
    mistakes = sum(r[1] for r in rows) / max(1, len(rows))
    grade = next(g for g, f, m in GRADES if found >= f and mistakes <= m)
    return {"found": found, "mistakes_per_mission": mistakes, "grade": grade, "rows": rows}


def window(
    targets,
    seed: int = 0,
    per_kind: int = 2,
    pretty: bool = False,
    browser: bool = True,
    retrain: bool = False,
    folder=".",
    drone: bool = False,
    fast: bool = False,
    camo: float | None = None,
) -> None:
    """Teach, then open the 3-D window and the brain page, and let it fly.

    ``fast``: the fast fly of :mod:`.fastfly` (or, for the drone, only the
    faster eye), the optic lobe at its trained 20 ms, kept to real time.
    ``camo``: the objects painted that much (0-1) in the floor's checker,
    and the brain for camouflaged targets flying.
    """
    from .livebrain import LiveBrain
    from .watch import FlysEye, Pilot

    hidden = camo is not None
    items = arena.mission(seed, per_kind=per_kind, camo=camo or 0.0)
    print("Wczytuję oko muchy (flyvis)...")
    eye = FlysEye(items=items, every_ms=EYE_MS[fast])  # before MuJoCo; see watch.interactive
    mb, valence = learned(
        eye, targets, folder=folder, retrain=retrain, drone=drone, camo=hidden
    )
    _print_valence(valence, targets)
    import mujoco.viewer

    pilot = _pilot(items, mb, eye, valence, targets, drone, fast, hidden)
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
            wall0, last_snap, last_frame, ended = time.time(), 0.0, 0.0, False
            while v.is_running():
                with keys.lock:
                    paused, follow = keys.paused, keys.follow
                if paused or ended:
                    time.sleep(0.03)
                    wall0 += 0.03  # a pause is not flight time
                    v.sync()
                    continue
                pilot.step()
                ahead = pilot.flight.t - (time.time() - wall0)
                if fast and ahead > 0:  # no faster than the animal
                    time.sleep(min(0.05, ahead))
                if not pilot.flight.upright():
                    print(
                        f"\n{pilot.who}: utrata panowania nad lotem."
                        " Zamknij okno i spróbuj ponownie."
                    )
                    ended = True
                now = time.time()
                if now - last_frame > FRAME_S or not fast:
                    # Drawing is the expensive part once the fly itself is
                    # cheap, so the fast mode draws at most 30 frames a second.
                    view.show(pilot.flight)
                    if follow:
                        heading = pilot.flight.state()["heading"]
                        with v.lock():
                            view.chase(v.cam, heading, distance=CHASE[drone])
                    v.sync()
                    last_frame = now
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
                    phase = describe(pilot.nav.phase, pilot.who)
                    print(f"\r{pilot.flight.t:5.2f} s  {phase:40s}", end="")
                if pilot.done and not ended:
                    print("\n" + pilot.summary())
                    ended = True
    finally:
        live.close()


#: How often the eye looks, ms: every 5 for the full simulation, every 20 --
#: the step the optic lobe was trained at -- in the fast mode.
EYE_MS = {False: 5.0, True: 20.0}
#: Seconds between drawn frames in the fast mode.
FRAME_S = 1.0 / 30.0
#: How far behind the camera follows: the fly, and the drone.
CHASE = {False: 22.0, True: 12.0}


#: The fly's pace in a mission: it looks around at 240 deg/s already, and
#: calls a mission done after one empty look rather than two. In 12 new
#: arenas per target its missions took 22.8 s instead of 25.7, with as many
#: targets found (69 of 72) and 4 wrong landings against 5. The drone's
#: pace is :data:`~.dronebrain.PACE`.
FLY_PACE = {"patience": 1}


def _pilot(
    items,
    mb,
    eye,
    valence,
    targets,
    drone: bool,
    fast: bool = False,
    camo: bool = False,
    pace: bool = True,
) -> Autopilot:
    """The autopilot on the fly (the full physics, or the fast fly), or on the drone.

    It takes a close look before it lands, and before it calls a mission done.
    Taught the close look on the tether, a novice's close looks turned down
    most targets -- 7 of 24 balls found -- and only practice made them
    useful; taught it in flight (:func:`flown_close_looks`), a novice's close
    look is as good as a practised one's.

    ``camo``: it also sees what breaks the skyline, and the landscape's
    columns are things it can wrongly land on. ``pace``: the drone at its
    mission pace (:data:`~.dronebrain.PACE`, :data:`FLY_PACE`); False for the lessons, flown at
    the pace they were first flown at.
    """
    expert = {"inspect": True, "look_before_leaving": True}
    if camo:
        from .watch import COLUMNS

        expert.update(
            skyline=True, landmarks=[(x, y) for x, y, _ in COLUMNS], punish=CAMO_PUNISH
        )
        if drone:
            expert.update(
                looks=CAMO_LOOKS, around=AROUND, peer=CAMO_PEER, peer_time=CAMO_PEER_TIME
            )
    if not drone:
        if pace:
            expert.update(FLY_PACE)
        if fast:
            from .fastfly import FastFly

            return Autopilot(items, mb, eye, valence, targets, vehicle=FastFly(), **expert)
        return Autopilot(items, mb, eye, valence, targets, **expert)
    from .dronebrain import NAVIGATOR, PACE, DroneFlight

    how = {**NAVIGATOR, **(PACE if pace else {}), **expert}
    return Autopilot(items, mb, eye, valence, targets, vehicle=DroneFlight(), **how)


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
    fast: bool = False,
    camo: float | None = None,
) -> Path:
    """Record a whole mission: the chase view, with the map and the fly's eye beside it.

    ``frame_ms`` of flight per frame: 10 for the fly (slow motion, it is
    quick), 33 for the drone (about real time).
    """
    import imageio.v2 as imageio

    from .watch import FlysEye

    if frame_ms is None:
        frame_ms = 33.0 if drone else 10.0

    hidden = camo is not None
    items = arena.mission(seed, per_kind=per_kind, camo=camo or 0.0)
    eye = FlysEye(items=items, every_ms=EYE_MS[fast])
    mb, valence = learned(
        eye, targets, folder=folder, retrain=retrain, drone=drone, camo=hidden
    )
    import mujoco

    pilot = _pilot(items, mb, eye, valence, targets, drone, fast, hidden)
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
        f"t = {pilot.flight.t:4.1f} s   "
        + (f"({slow:.0f}x zwolnione)" if slow > 1.5 else "(tempo rzeczywiste)"),
        f"{pilot.who.lower()} {st['phase_pl']}",
        f"doleciał{'a' if pilot.who == 'Muszka' else ''} do: "
        + (", ".join(st["reached"]) or "-"),
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
    "flown_close_looks",
    "shipped_memory",
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
