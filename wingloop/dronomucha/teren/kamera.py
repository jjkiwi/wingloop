"""The thermal camera looking down, and the laser rangefinder beside it.

A pixel of a camera 60 m up covers a quarter of a metre of ground, of one
100 m up with a 32 x 24 array, three metres: what it reads is the
temperature of everything in that patch, mixed. So each pixel's patch of
ground is found -- where its corners' rays meet the slope -- and where an
object lies in it, the patch is sampled finely and the object's warm parts
mixed in with the ground's in the share of the patch they cover. A person
lying on snow fills a 0.25 m pixel and is 15 K warm in it; in a 2 m pixel
they are 2 K warm, as warm as a sunlit stone. Then the optics' blur and the
sensor's noise.

The camera sits on a two-axis gimbal: it looks ``pochylenie`` degrees below
the horizon (90: straight down) along the heading, however the platform
leans. The rangefinder looks along the camera's middle; it reaches as far
as its model does in daylight (:data:`DALMIERZE`).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..platformy import Kamera
from . import obiekty as O
from . import teren as T

#: Rangefinders: how far each reaches outdoors over rock and snow, m, and its noise.
DALMIERZE = {"TF02-Pro": (13.5, 0.01), "TF03-100": (50.0, 0.01), "TF03-180": (100.0, 0.01)}


@dataclass
class Widok:
    """One picture: the temperatures (rows, columns), C; where each pixel's middle meets the
    ground (rows, columns, 3); how much ground a pixel covers there, m; where
    from (x, y, z) and which way (heading, radians)."""

    temp: np.ndarray
    ziemia: np.ndarray
    gsd: np.ndarray
    skad: np.ndarray
    kurs: float
    rogi: np.ndarray  # the picture's four corners on the ground, (4, 3)


def promienie(kamera: Kamera, kurs: float, pochylenie: float = 90.0, krawedzie=False):
    """The camera's rays in the world: (rows, columns, 3), or at the pixels' corners."""
    w, h = kamera.px
    tw = np.tan(np.deg2rad(kamera.fov) / 2)
    th = tw * h / w
    n = (w + 1, h + 1) if krawedzie else (w, h)
    if krawedzie:
        tx, ty = np.linspace(tw, -tw, n[0]), np.linspace(th, -th, n[1])
    else:
        half_x, half_y = tw * (1 - 1 / w), th * (1 - 1 / h)
        tx, ty = np.linspace(half_x, -half_x, n[0]), np.linspace(half_y, -half_y, n[1])
    el = -np.deg2rad(pochylenie)
    b = np.array([np.cos(kurs) * np.cos(el), np.sin(kurs) * np.cos(el), np.sin(el)])
    left = np.array([-np.sin(kurs), np.cos(kurs), 0.0])
    up = np.cross(b, left)
    TX, TY = np.meshgrid(tx, ty)
    d = b + TX[..., None] * left + TY[..., None] * up
    return d / np.linalg.norm(d, axis=-1, keepdims=True)


def _wewnatrz(obj: O.Obiekt, px, py):
    """Which points (x, y) the object covers seen from above, and the temperature there
    -- its topmost part's -- or NaN."""
    c, s = np.cos(obj.yaw), np.sin(obj.yaw)
    lx = c * (px - obj.x) + s * (py - obj.y)
    ly = -s * (px - obj.x) + c * (py - obj.y)
    temp = np.full(px.shape, np.nan)
    top = np.full(px.shape, -np.inf)
    for part in obj.czesci:
        if part.pod_sniegiem:
            continue
        if part.typ == "kapsula":
            a, b = np.array(part.a), np.array(part.b)
            ab = b[:2] - a[:2]
            f = np.clip(((lx - a[0]) * ab[0] + (ly - a[1]) * ab[1]) / max(ab @ ab, 1e-9), 0, 1)
            inside = np.hypot(lx - a[0] - f * ab[0], ly - a[1] - f * ab[1]) < part.r
            z = max(a[2], b[2]) + part.r
        else:
            (cx, cy, cz), (rx, ry, rz) = part.a, part.r
            inside = ((lx - cx) / rx) ** 2 + ((ly - cy) / ry) ** 2 < 1.0
            z = cz + rz
        win = inside & (z > top)
        temp[win], top[win] = part.temp, z
    return temp


def zdjecie(
    t: T.Teren,
    ground_temp: np.ndarray,
    obiekty: list,
    kamera: Kamera,
    skad,
    kurs: float,
    pochylenie: float = 90.0,
    rng=None,
    rozmycie: float = 0.5,
) -> Widok:
    """A picture from ``skad`` (x, y, z), heading ``kurs`` radians, the camera
    ``pochylenie`` degrees below the horizon."""
    from scipy.ndimage import gaussian_filter

    skad = np.asarray(skad, dtype=float)
    corners = t.trafienie(skad, promienie(kamera, kurs, pochylenie, krawedzie=True))
    mid = 0.25 * (corners[:-1, :-1] + corners[1:, :-1] + corners[:-1, 1:] + corners[1:, 1:])
    span_x = np.linalg.norm(corners[:-1, 1:] - corners[:-1, :-1], axis=-1)
    span_y = np.linalg.norm(corners[1:, :-1] - corners[:-1, :-1], axis=-1)
    gsd = np.sqrt(span_x * span_y)
    temp = t.pole(ground_temp, mid[..., 0], mid[..., 1])
    for obj in obiekty:
        near = np.hypot(mid[..., 0] - obj.x, mid[..., 1] - obj.y) < obj.zasieg() + 1.5 * gsd
        if not near.any():
            continue
        rows, cols = np.nonzero(near)
        k = int(np.clip(np.ceil(gsd[rows, cols].max() / 0.08), 3, 12))
        f = (np.arange(k) + 0.5) / k
        fu, fv = np.meshgrid(f, f)
        c00 = corners[rows, cols][:, None, None, :]
        c01 = corners[rows, cols + 1][:, None, None, :]
        c10 = corners[rows + 1, cols][:, None, None, :]
        c11 = corners[rows + 1, cols + 1][:, None, None, :]
        u, v = fu[None, ..., None], fv[None, ..., None]
        sub = (1 - u) * (1 - v) * c00 + u * (1 - v) * c01 + (1 - u) * v * c10 + u * v * c11
        part_t = _wewnatrz(obj, sub[..., 0], sub[..., 1])
        base = temp[rows, cols][:, None, None]
        mixed = np.where(np.isnan(part_t), base, part_t).mean(axis=(1, 2))
        temp[rows, cols] = mixed
    if rozmycie:
        temp = gaussian_filter(temp, rozmycie)
    rng = rng if rng is not None else np.random.default_rng()
    temp = temp + kamera.netd * rng.standard_normal(temp.shape)
    rogi = corners[[0, 0, -1, -1], [0, -1, -1, 0]]
    return Widok(temp, mid, gsd, skad, kurs, rogi)


def dalmierz(t: T.Teren, skad, kierunek, model: str = "TF03-180", rng=None):
    """How far the beam along ``kierunek`` goes to the ground, m, or None beyond its reach;
    and where it hits."""
    zasieg, szum = DALMIERZE[model]
    hit = t.trafienie(np.asarray(skad, float), np.asarray(kierunek, float)[None])[0]
    d = float(np.linalg.norm(hit - skad))
    rng = rng if rng is not None else np.random.default_rng()
    if d > zasieg:
        return None, hit
    return d * (1 + szum * rng.standard_normal()), hit
