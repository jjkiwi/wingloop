"""The drone's other senses, read where the eye looks, and coded for the brain.

The fly's eye on a rescue drone looks through a thermal camera whose gain
follows the warmest thing in view (:data:`~wingloop.lab.watch.THERMAL_SPAN`):
it sees what is warmer than what, never how warm. Nor does an eye tell how
big something is -- a near goat and a far person fill it alike. The drone
has sensors that do: the thermal camera itself, read radiometrically, in
kelvin above the ground, and the laser rangefinder, which with the thermal
camera measures how far, how tall and how wide
(:meth:`~wingloop.lab.watch.FlysEye.measure_far`).

:func:`odczyt` reads them where the eye looks; :func:`kod` turns a reading
into what a brain takes in -- each quantity a row of lines, each line tuned
to its own value, as a sensory neuron is (a population code), and every
line silent when a sensor finds nothing. A fly's mushroom body takes smell,
taste, temperature and sight into one calyx; :mod:`.hybryda` gives the
fly's mushroom body these lines next to the eye's.
"""

from __future__ import annotations

import numpy as np

#: What a reading holds, in this order: kelvin above the ground where the eye
#: looks; how far, how tall and how wide what it looks at is, mm; and whether
#: it reaches the top of the thermal camera's view. NaN: nothing measured.
POLA = ("temp", "distance", "height", "width", "top")

#: How far either side of the heading the gimbal looks for what the eye
#: fixates, degrees, and how much warmer than the ground it must be to be
#: measured, K.
PASMO, CIEPLE = 4.0, 1.5

#: Each quantity's lines: (least, most, lines, on a log scale).
STROJENIE = {
    "temp": (0.5, 40.0, 10, True),
    "distance": (15.0, 240.0, 10, True),
    "height": (1.0, 40.0, 10, True),
    "width": (2.0, 60.0, 10, True),
}


class Zmysly:
    """What the drone's other sensors read where the eye last looked: a function of the eye.

    Its own noise, so reading does not change what else the drone draws.
    """

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)

    def __call__(self, fly_eye) -> np.ndarray:
        return odczyt(fly_eye, self.rng)


def odczyt(fly_eye, rng=None) -> np.ndarray:
    """One reading, :data:`POLA`, of what the eye looks at.

    The eye fixates by turning, so what it looks at is ahead, but not level:
    the drone flies under a metre up, and a person lying 2-4 m off is 15-25
    degrees below the horizon, where a level rangefinder's beam passes over
    them. So the thermal camera and the rangefinder sit on a gimbal, as a
    rescue drone's do: the warmest thing in a band :data:`PASMO` degrees
    either side of the heading is what is read -- its kelvin, and, with the
    beam turned onto it, how far, how tall and how wide it is (the warm
    pixels joined to it). Less than :data:`CIEPLE` K above the ground: only
    its kelvin.
    """
    from scipy import ndimage

    from wingloop.lab.arena import FLOOR
    from wingloop.lab.watch import IR_FOV

    out = np.full(len(POLA), np.nan)
    aim = celownik(fly_eye, rng)
    if aim is None:
        return out
    image, (r, c), ray, d = aim
    peak = float(image[r, c])
    out[0] = peak
    if peak < CIEPLE or not np.isfinite(d):
        return out
    step = IR_FOV / (image.shape[0] - 1)
    labels, _ = ndimage.label(image > max(CIEPLE, 0.5 * peak))
    rows, across = np.nonzero(labels == labels[r, c])
    top = np.deg2rad(IR_FOV / 2 - rows.min() * step)
    flat = d * np.cos(np.deg2rad(IR_FOV / 2 - r * step))
    wide = np.deg2rad((across.max() - across.min() + 1) * step)
    o = fly_eye.pose[0]
    out[1:] = d, o[2] - FLOOR + flat * np.tan(top), 2 * flat * np.tan(wide / 2), rows.min() == 0
    return out


def celownik(fly_eye, rng=None):
    """The gimbal on what the eye looks at: (thermal picture, its warmest pixel in the
    band, the ray to it, how far the rangefinder's beam goes along it -- inf if
    it is not :data:`CIEPLE` K warm or out of reach), or None if the eye has not looked."""
    from wingloop.lab.scene import Camera
    from wingloop.lab.watch import IR_FOV, IR_SIZE, _level

    if fly_eye is None or fly_eye.pose is None:
        return None
    rng = rng if rng is not None else np.random.default_rng()
    image = fly_eye.thermal_view(rng)
    step = IR_FOV / (IR_SIZE - 1)
    mid = (IR_SIZE - 1) // 2
    band = max(1, int(round(PASMO / step)))
    cols = slice(mid - band, mid + band + 1)
    strip = image[:, cols]
    r, c = np.unravel_index(int(np.argmax(strip)), strip.shape)
    c += cols.start
    o, R = fly_eye.pose
    ray = Camera(size=IR_SIZE, fov=IR_FOV).rays()[r, c] @ _level(R).T
    d = _beam(fly_eye.world, o, ray) if image[r, c] >= CIEPLE else np.inf
    return image, (r, c), ray, d


def polozenie(fly_eye, rng=None):
    """Where what the eye looks at is, (x, y) mm, by the gimbal and the rangefinder; or None."""
    aim = celownik(fly_eye, rng)
    if aim is None or not np.isfinite(aim[3]):
        return None
    o = fly_eye.pose[0]
    p = o + aim[3] * aim[2]
    return float(p[0]), float(p[1])


def _beam(world, o, d) -> float:
    """How far a rangefinder's beam goes along ``d`` before it hits something, mm."""
    from wingloop.lab.scene import Camera, depth
    from wingloop.lab.watch import LRF_BEAM, LRF_RANGE

    x = d / np.linalg.norm(d)
    y = np.cross([0.0, 0.0, 1.0], x)
    y /= np.linalg.norm(y) + 1e-9
    R = np.stack([x, y, np.cross(x, y)], axis=1)
    t = float(depth(world, o, Camera(size=3, fov=LRF_BEAM), rotation_matrix=R).min())
    return t if t <= LRF_RANGE else np.inf


def kod(readings, gain: int = 1) -> np.ndarray:
    """Readings, (n, :data:`POLA`) or one, as lines: a bump of activity round each value.

    ``gain``: each line repeated, so that more Kenyon cells sample the
    senses -- a calyx can give a sense more or less of its room.
    """
    r = np.atleast_2d(np.asarray(readings, dtype=float))
    rows = []
    for j, name in enumerate(POLA):
        x = r[:, j]
        if name == "top":
            rows.append(np.nan_to_num(x)[:, None])
            continue
        lo, hi, n, log = STROJENIE[name]
        f = np.log if log else (lambda v: v)
        centres = np.linspace(f(lo), f(hi), n)
        width = centres[1] - centres[0]
        v = f(np.clip(np.nan_to_num(x, nan=lo), lo, hi))
        bump = np.exp(-0.5 * ((v[:, None] - centres[None]) / width) ** 2)
        rows.append(np.where(np.isnan(x)[:, None], 0.0, bump))
    lines = np.concatenate(rows, axis=1)
    lines = np.repeat(lines, gain, axis=1) if gain > 1 else lines
    return lines[0] if np.ndim(readings) == 1 else lines
