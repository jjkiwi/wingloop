"""The brain for a task: its lessons, kept on disk, and the brain taught on them."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from wingloop.lab import autopilot

from .swiat import swiat
from .zadanie import Zadanie

#: Where the tasks' lessons are kept.
DATA = Path(__file__).with_name("data")


def lekcje(fly_eye, z: Zadanie, fresh: bool = False, progress=print, folder=DATA, zmysly=False):
    """A task's lessons: ((features, kinds), (close-look features, kinds)).

    The tethered look at each kind, turning as the drone turns, and close
    looks flown, one object at a time -- the lab's two lessons
    (:func:`~wingloop.lab.navigate.tethered_training`,
    :func:`~wingloop.lab.autopilot.flown_close_looks`), in the task's world.
    Kept in ``folder`` and read back after; ``fresh`` flies them again.

    ``zmysly``: with what the other sensors read at each view
    (:mod:`.zmysly`): ((features, kinds, readings), (the same, close)) -- the
    same views, flown again with the sensors on, kept in a file of their own.
    """
    from wingloop.lab.dronebrain import SCAN_RATE
    from wingloop.lab.navigate import tethered_training

    from .zmysly import Zmysly

    path = Path(folder) / f"{z.name}_lekcje{'_zmysly' if zmysly else ''}.npz"
    tethered, close = z.lessons
    senses = dict(senses=Zmysly()) if zmysly else {}
    if path.exists() and not fresh:
        with np.load(path) as d:
            if len(d["K"]) == tethered and len(d["CK"]) == close:
                if zmysly:
                    return (d["F"], d["K"], d["S"]), (d["CF"], d["CK"], d["CS"])
                return (d["F"], d["K"]), (d["CF"], d["CK"])
    with swiat(z):
        kinds = tuple(z.kinds)
        t0 = time.time()
        if progress:
            progress(f"Dron ogląda na uwięzi każdy rodzaj obiektu ({z.pl})...")
        far = tethered_training(
            fly_eye,
            kinds,
            samples=tethered,
            seed=0,
            progress=progress,
            scan_rate=SCAN_RATE,
            **senses,
        )
        if progress:
            progress("... i podlatuje do każdego, by obejrzeć go z bliska...")
        near = autopilot.flown_close_looks(
            fly_eye, kinds, close, seed=0, drone=True, progress=progress, **senses
        )
        if progress:
            n = len(far[1]) + len(near[1])
            progress(f"  {n} widoków w {time.time() - t0:.0f} s -> {path}")
    keep = dict(F=far[0], K=far[1], CF=near[0], CK=near[1])
    if zmysly:
        keep.update(S=far[2], CS=near[2])
    np.savez_compressed(path, **keep)
    return far, near


def mozg(fly_eye, z: Zadanie, fresh: bool = False, progress=print, dodatki=None):
    """The brain taught to look for the task's target: (mushroom body, valence).

    ``dodatki`` (:class:`~.dodatki.Dodatki`) with senses or a cortex: the
    hybrid brain instead (:func:`~.hybryda.naucz`), its senses read from
    ``fly_eye``.
    """
    if dodatki is not None and dodatki.mozg:
        from .hybryda import naucz

        far, near = lekcje(fly_eye, z, fresh=fresh, progress=progress, zmysly=True)
        with swiat(z):
            brain, valence = naucz(far, near, z.target, dodatki.zmysly, dodatki.kora)
        brain.eye = fly_eye
        return brain, valence
    (F, K), close = lekcje(fly_eye, z, fresh=fresh, progress=progress)
    with swiat(z):
        return autopilot.teach_targets(F, K, {z.target}, close=close)
