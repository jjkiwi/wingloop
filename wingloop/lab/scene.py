"""A small world to look at: a textured ground, a sky, and solid objects.

Rendered by casting one ray per pixel through a wide-angle camera, because the
fly's eye is wide-angle and because optic flow needs texture to exist -- an
untextured ground gives a moving animal nothing to see. Pure numpy, so it runs
anywhere the rest of this package does.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

#: Brightness of the sky, the ground's two checker tones, and the default
#: object. Objects are dark against the sky on purpose: flies fixate dark
#: vertical objects, and the OFF pathway is half the visual system.
SKY = 0.85
GROUND = (0.35, 0.55)
CHECKER = 2.0  # metres per ground tile


@dataclass
class Thing:
    """A solid in the world: a sphere, a vertical bar (cylinder) or a box."""

    kind: str  # "sphere", "bar" or "box"
    position: np.ndarray  # metres, world frame (x forward, y left, z up)
    size: float = 0.5  # radius for sphere/bar, half-edge for box
    height: float = 3.0  # bar height, metres
    brightness: float = 0.05
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    label: str = ""
    #: Camouflage, 0-1: how much of its surface wears the ground's checker
    #: instead of its own brightness -- at 1 it is painted like the floor
    #: beneath it, tile for tile.
    camo: float = 0.0
    #: How much warmer than the floor it is, kelvin, to a thermal camera
    #: (:func:`thermal`). Paint that hides it from the eye does not change it.
    heat: float = 0.0

    def __post_init__(self):
        self.position = np.asarray(self.position, dtype=float)
        self.velocity = np.asarray(self.velocity, dtype=float)
        if self.kind not in ("sphere", "bar", "box"):
            raise ValueError(f"unknown kind {self.kind!r}")
        if not self.label:
            self.label = self.kind


@dataclass
class World:
    things: list[Thing] = field(default_factory=list)
    ground_height: float = 0.0

    def step(self, dt: float) -> None:
        for t in self.things:
            t.position = t.position + t.velocity * dt


def rotation(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Body-to-world rotation, ZYX (yaw, then pitch, then roll)."""
    cr, sr, cp, sp, cy, sy = (
        np.cos(roll),
        np.sin(roll),
        np.cos(pitch),
        np.sin(pitch),
        np.cos(yaw),
        np.sin(yaw),
    )
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ]
    )


@dataclass
class Camera:
    """A forward-looking wide-angle camera, equidistant (angle-linear) projection.

    ``fov`` is the full horizontal and vertical field, degrees. The fly's single
    eye covers roughly 160-180 degrees; 120 keeps the image from being mostly
    sky and ground while still giving objects off to the side somewhere to be.
    """

    size: int = 96
    fov: float = 120.0

    def rays(self) -> np.ndarray:
        """Unit ray directions in the camera/body frame, shape (size, size, 3)."""
        half = np.deg2rad(self.fov) / 2
        a = np.linspace(half, -half, self.size)  # azimuth: + is left, image column 0
        e = np.linspace(half, -half, self.size)  # elevation: + is up, image row 0
        az, el = np.meshgrid(a, e)
        return np.stack([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)], axis=-1)


def _sphere_hits(o, d, centre, r):
    oc = o - centre
    b = d @ oc
    c = oc @ oc - r * r
    disc = b * b - c
    t = -b - np.sqrt(np.maximum(disc, 0.0))
    return np.where((disc >= 0) & (t > 0), t, np.inf)


def _bar_hits(o, d, centre, r, height):
    # Infinite vertical cylinder about the bar's axis, clipped to its height.
    oc = (o - centre)[:2]
    dx, dy = d[..., 0], d[..., 1]
    a = dx * dx + dy * dy
    b = dx * oc[0] + dy * oc[1]
    c = oc @ oc - r * r
    disc = b * b - a * c
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (-b - np.sqrt(np.maximum(disc, 0.0))) / a
    z = o[2] + t * d[..., 2]
    ok = (disc >= 0) & (t > 0) & (z >= centre[2]) & (z <= centre[2] + height)
    return np.where(ok, t, np.inf)


def _box_hits(o, d, centre, half):
    with np.errstate(divide="ignore", invalid="ignore"):
        inv = 1.0 / d
        t1 = (centre - half - o) * inv
        t2 = (centre + half - o) * inv
    tmin = np.nanmax(np.minimum(t1, t2), axis=-1)
    tmax = np.nanmin(np.maximum(t1, t2), axis=-1)
    ok = (tmax >= tmin) & (tmax > 0)
    return np.where(ok, np.where(tmin > 0, tmin, tmax), np.inf)


def render(
    world: World,
    position,
    attitude,
    camera: Camera | None = None,
    *,
    rotation_matrix=None,
    checker: float = CHECKER,
    match: float = 1.0,
) -> np.ndarray:
    """What the camera sees: a (size, size) image of brightness in [0, 1].

    ``attitude`` is (roll, pitch, yaw), radians, body to world; or pass the
    body-to-world ``rotation_matrix`` directly and ``attitude`` is ignored.
    ``checker`` is the floor tile size in the world's own units. ``match``:
    how much of its camouflage a thing keeps in this camera's light -- 1 in
    visible light, what it was painted for; less in near infrared, where
    paint that matches the floor to the eye need not.
    """
    camera = camera or Camera()
    o = np.asarray(position, dtype=float)
    R = rotation(*attitude) if rotation_matrix is None else np.asarray(rotation_matrix)
    d = camera.rays() @ R.T  # world-frame directions

    # Background: sky above, a checkered ground below.
    image = np.full(d.shape[:2], SKY)
    with np.errstate(divide="ignore", invalid="ignore"):
        tg = (world.ground_height - o[2]) / d[..., 2]
    ground = (tg > 0) & np.isfinite(tg)
    gx = o[0] + tg * d[..., 0]
    gy = o[1] + tg * d[..., 1]
    image = np.where(ground, _floor(gx, gy, checker), image)
    depth = np.where(ground, tg, np.inf)

    for thing in world.things:
        if thing.kind == "sphere":
            t = _sphere_hits(o, d, thing.position, thing.size)
        elif thing.kind == "bar":
            t = _bar_hits(o, d, thing.position, thing.size, thing.height)
        else:
            t = _box_hits(o, d, thing.position, thing.size)
        closer = t < depth
        shade = thing.brightness
        camo = thing.camo * match
        if camo > 0:
            reach = np.where(np.isfinite(t), t, 0.0)
            px, py = o[0] + reach * d[..., 0], o[1] + reach * d[..., 1]
            shade = (1 - camo) * thing.brightness + camo * _floor(px, py, checker)
        image = np.where(closer, shade, image)
        depth = np.where(closer, t, depth)
    return image.astype(np.float32)


def depth(
    world: World, position, camera: Camera | None = None, *, rotation_matrix
) -> np.ndarray:
    """How far each of the camera's rays goes before it hits something: (size, size).

    The geometry :func:`render` shades, without the shading: what an active
    infrared depth camera (time of flight) measures. Paint does not change it,
    so a camouflaged object is as plain here as any other. ``inf`` for sky.
    """
    camera = camera or Camera()
    o = np.asarray(position, dtype=float)
    d = camera.rays() @ np.asarray(rotation_matrix).T
    with np.errstate(divide="ignore", invalid="ignore"):
        tg = (world.ground_height - o[2]) / d[..., 2]
    t = np.where((tg > 0) & np.isfinite(tg), tg, np.inf)
    for thing in world.things:
        if thing.kind == "sphere":
            hit = _sphere_hits(o, d, thing.position, thing.size)
        elif thing.kind == "bar":
            hit = _bar_hits(o, d, thing.position, thing.size, thing.height)
        else:
            hit = _box_hits(o, d, thing.position, thing.size)
        t = np.minimum(t, hit)
    return t


def thermal(
    world: World,
    position,
    camera: Camera | None = None,
    *,
    rotation_matrix,
    clutter: float = 0.0,
    sky: float = -10.0,
) -> np.ndarray:
    """What a thermal camera sees: kelvin above the floor's mean, (size, size).

    Each solid is its own :attr:`Thing.heat` warmer than the floor; the floor
    is uneven by ``clutter`` kelvin (a smooth pattern fixed to the world,
    as sun and shade and damp leave a real one); the clear sky is cold. The
    checker painted on the floor and on camouflaged objects is not in it.
    """
    camera = camera or Camera()
    d = camera.rays() @ np.asarray(rotation_matrix).T
    return thermal_along(world, position, d, clutter=clutter, sky=sky)


def thermal_along(world: World, position, d, *, clutter: float = 0.0, sky: float = -10.0):
    """:func:`thermal` along any world-frame directions ``d`` (..., 3)."""
    o = np.asarray(position, dtype=float)
    image = np.full(d.shape[:-1], sky)
    with np.errstate(divide="ignore", invalid="ignore"):
        tg = (world.ground_height - o[2]) / d[..., 2]
    ground = (tg > 0) & np.isfinite(tg)
    reach = np.where(ground, tg, 0.0)
    gx, gy = o[0] + reach * d[..., 0], o[1] + reach * d[..., 1]
    image = np.where(ground, clutter * _patches(gx, gy), image)
    depth = np.where(ground, tg, np.inf)
    for thing in world.things:
        if thing.kind == "sphere":
            t = _sphere_hits(o, d, thing.position, thing.size)
        elif thing.kind == "bar":
            t = _bar_hits(o, d, thing.position, thing.size, thing.height)
        else:
            t = _box_hits(o, d, thing.position, thing.size)
        closer = t < depth
        image = np.where(closer, thing.heat, image)
        depth = np.where(closer, t, depth)
    return image


def _patches(x, y):
    """A smooth pattern of unit spread, fixed to the world: patches some 20-60 units across."""
    out = np.zeros_like(x)
    for kx, ky, phase in _PATCHES:
        out = out + np.sin(kx * x + ky * y + phase)
    return out * np.sqrt(2.0 / len(_PATCHES))


_rng = np.random.default_rng(1)
_PATCHES = [
    (k * np.cos(a), k * np.sin(a), p)
    for k, a, p in zip(
        2 * np.pi / _rng.uniform(20.0, 60.0, 12),
        _rng.uniform(0, 2 * np.pi, 12),
        _rng.uniform(0, 2 * np.pi, 12),
        strict=True,
    )
]


def _floor(x, y, checker: float):
    """The ground's checker tone at these world coordinates."""
    tile = (np.floor(x / checker) + np.floor(y / checker)) % 2
    return np.where(tile > 0, GROUND[1], GROUND[0])
