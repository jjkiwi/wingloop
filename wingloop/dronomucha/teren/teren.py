"""The ground: a Tatra valley in real metres, what covers it, and how warm it is.

Not a survey of a real valley -- a valley built the way the Tatras' are: a
glacial trough, U-shaped, falling north, a cirque at its head with a tarn,
ridges either side 400-600 m above the floor, cut by gullies, with spurs; a
trail up the floor and on to a pass (:func:`dolina`). What covers it
follows height and steepness: snow above about 1750 m where it can lie,
rock where it is too steep for snow, scree below the cliffs, dwarf pine
(kosodrzewina) and meadow lower down.

How warm the ground is follows the sun (:func:`temperatura`): a cold
autumn day, the air below zero, the sun low in the south-south-east. Snow
cannot rise above 0 C; rock facing the sun is 15-20 K warmer than in shade
-- and a sunlit boulder is as warm as a person's jacket, which is what a
search by heat has against it.

Coordinates: x east, y north, z up, metres; the search area is the square
kilometre round (0, 0).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

#: What the ground is, cell by cell.
SNIEG, SKALA, PIARG, KOSO, TRAWA, WODA, SZLAK = range(7)
NAZWY = ("śnieg", "skała", "piarg", "kosodrzewina", "hala", "staw", "szlak")
#: Their colours for a person watching (RGB 0-1).
KOLORY = (
    (0.94, 0.95, 0.98),
    (0.42, 0.41, 0.40),
    (0.60, 0.58, 0.55),
    (0.16, 0.30, 0.15),
    (0.55, 0.58, 0.33),
    (0.12, 0.25, 0.38),
    (0.50, 0.40, 0.28),
)


@dataclass(frozen=True)
class Pogoda:
    """The day: the air (C), the sun's bearing from north and height (degrees), cloud (0-1)."""

    powietrze: float = -2.0
    slonce_az: float = 155.0
    slonce_el: float = 25.0
    chmury: float = 0.1

    def slonce(self) -> np.ndarray:
        """A unit vector toward the sun."""
        az, el = np.deg2rad(self.slonce_az), np.deg2rad(self.slonce_el)
        return np.array([np.sin(az) * np.cos(el), np.cos(az) * np.cos(el), np.sin(el)])


#: Days to search on. The lessons are flown on days drawn between a frost and a hot
#: afternoon (:func:`~.lekcje.pogoda_lekcji`): ``jesien``, ``mroz`` and ``lato`` lie
#: among them, ``upal`` -- 27 C and the sun high, the rock far warmer than a person --
#: beyond them.
DNI = {
    "jesien": Pogoda(),
    "mroz": Pogoda(powietrze=-10.0, slonce_az=165.0, slonce_el=14.0, chmury=0.85),
    "lato": Pogoda(powietrze=16.0, slonce_az=200.0, slonce_el=55.0, chmury=0.1),
    "upal": Pogoda(powietrze=27.0, slonce_az=210.0, slonce_el=60.0, chmury=0.0),
}


def _szum(n: int, rng, skale=(400, 200, 100, 50, 25, 12, 6), amp=None) -> np.ndarray:
    """Smooth noise on an n x n grid: octaves of random values, smoothed, summed."""
    from scipy.ndimage import gaussian_filter, zoom

    amp = amp or [s / 400 for s in skale]
    out = np.zeros((n, n))
    for s, a in zip(skale, amp, strict=True):
        k = max(2, int(np.ceil(n * 2 / s)))
        g = rng.standard_normal((k, k))
        up = zoom(g, n / k, order=3)[:n, :n]
        out += a * gaussian_filter(up, n / k / 4)
    return out / np.abs(out).max()


@dataclass
class Teren:
    """A height field and what covers it, ``krok`` metres a cell, ``polowa`` metres each way.

    :meth:`wysokosc` reads the height anywhere (bilinear); :meth:`trafienie`
    finds where rays from the air first meet the ground; :meth:`temperatura` is
    how warm the ground is on a given day.
    """

    z: np.ndarray
    pokrycie: np.ndarray
    krok: float
    polowa: float
    szlak: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))
    staw: tuple = (0.0, 0.0, 0.0)
    baza: tuple = (0.0, 0.0)

    @property
    def n(self) -> int:
        return self.z.shape[0]

    def _ij(self, x, y):
        return (np.asarray(y) + self.polowa) / self.krok, (
            np.asarray(x) + self.polowa
        ) / self.krok

    def wysokosc(self, x, y) -> np.ndarray:
        """The ground's height, m, at (x, y) -- bilinear between cells."""
        from scipy.ndimage import map_coordinates

        i, j = self._ij(x, y)
        shape = np.shape(i)
        out = map_coordinates(self.z, [np.ravel(i), np.ravel(j)], order=1, mode="nearest")
        return out.reshape(shape)

    def gladki(self, skala: float = 40.0):
        """The ground smoothed over ``skala`` m, and its slope east and north: (z, dz/dx,
        dz/dy), each on the grid -- worked out once and kept. What a plane's way is
        laid along: it cannot follow wiggles smaller than its turn."""
        from scipy.ndimage import gaussian_filter

        kept = self.__dict__.setdefault("_gladkie", {})
        if skala not in kept:
            zs = gaussian_filter(np.asarray(self.z, float), skala / self.krok)
            gy, gx = np.gradient(zs, self.krok)
            kept[skala] = (zs, gx, gy)
        return kept[skala]

    def poziomica(self, x: float, y: float, ile: float = 150.0, co: float = 30.0):
        """The contour of the smoothed ground through (x, y), ``ile`` m each way: points
        (n, 2) every ``co`` m, or None where the ground is too level to have one."""
        zs, gx, gy = self.gladki()
        level = float(self.pole(zs, x, y))
        halves = []
        for sign in (-1.0, 1.0):
            px, py, pts = float(x), float(y), []
            for _ in range(int(round(ile / co))):
                g = np.array([float(self.pole(gx, px, py)), float(self.pole(gy, px, py))])
                n = float(np.hypot(*g))
                if n < 0.02:
                    return None
                px, py = px - sign * co * g[1] / n, py + sign * co * g[0] / n
                # Back onto the contour: along the slope, by what it has drifted off.
                g = np.array([float(self.pole(gx, px, py)), float(self.pole(gy, px, py))])
                off = level - float(self.pole(zs, px, py))
                px, py = px + off * g[0] / max(g @ g, 1e-4), py + off * g[1] / max(g @ g, 1e-4)
                pts.append((px, py))
            halves.append(pts)
        return np.array([*halves[0][::-1], (float(x), float(y)), *halves[1]])

    def nachylenie(self) -> np.ndarray:
        """Each cell's slope, degrees."""
        gy, gx = np.gradient(self.z, self.krok)
        return np.degrees(np.arctan(np.hypot(gx, gy)))

    def normalne(self) -> np.ndarray:
        """Each cell's upward unit normal, (n, n, 3)."""
        gy, gx = np.gradient(self.z, self.krok)
        nrm = np.stack([-gx, -gy, np.ones_like(gx)], axis=-1)
        return nrm / np.linalg.norm(nrm, axis=-1, keepdims=True)

    def pole(self, raster, x, y, order=1) -> np.ndarray:
        """Read any raster laid on the same grid at (x, y)."""
        from scipy.ndimage import map_coordinates

        i, j = self._ij(x, y)
        shape = np.shape(i)
        out = map_coordinates(raster, [np.ravel(i), np.ravel(j)], order=order, mode="nearest")
        return out.reshape(shape)

    def trafienie(self, o, d, iteracje: int = 14, dokladnosc: float = 0.05) -> np.ndarray:
        """Where rays from ``o`` along ``d`` (..., 3) first meet the ground: points (..., 3).

        The height of a ray's point above the ground under it, as a function
        of how far along the ray it is, is brought to zero by the secant
        method -- a few steps on a slope, however steep, where stepping to
        the ground's height under the point crawls for rays looking down the
        slope. Then each hit is checked: is it on the ground (within
        ``dokladnosc`` m), and was the ray above the ground all the way to
        it? Rays that fail -- over a cliff's edge, behind a spur -- are
        walked along from ``o`` to where they first go under, and that
        crossing is halved down.
        """
        o = np.asarray(o, dtype=float)
        d = np.asarray(d, dtype=float)
        shape = d.shape[:-1]
        d = d.reshape(-1, 3)
        dz = np.minimum(d[:, 2], -1e-3)

        def above(t, rays=slice(None)):
            x, y = o[0] + t * d[rays, 0], o[1] + t * d[rays, 1]
            return o[2] + t * dz[rays] - self.wysokosc(x, y)

        t0 = (o[2] - float(self.wysokosc(o[0], o[1]))) / -dz
        f0 = above(t0)
        t1 = np.maximum(t0 + f0 / -dz, 0.0)
        f1 = above(t1)
        for _ in range(iteracje):
            if np.abs(f1).max() < dokladnosc:
                break
            den = f1 - f0
            ok = np.abs(den) > 1e-9
            step = np.where(ok, f1 * (t1 - t0) / np.where(ok, den, 1.0), 0.0)
            reach = 0.5 * t1 + 5.0
            t0, f0 = t1, f1
            t1 = np.maximum(t1 - np.clip(step, -reach, reach), 0.0)
            f1 = above(t1)
        # Under the ground on the way there, or not on it: walk those rays.
        bad = np.abs(f1) > 4 * dokladnosc
        for share in (0.5, 0.8, 0.93):
            bad |= above(share * t1) < -4 * dokladnosc
        if bad.any():
            t1[bad] = self._krokami(above, o[2], dz, bad, t1[bad])
        return (o + t1[:, None] * np.stack([d[:, 0], d[:, 1], dz], axis=1)).reshape(*shape, 3)

    def _krokami(self, above, oz, dz, rays, guess, kroki: int = 96) -> np.ndarray:
        """A ray's first crossing of the ground, walked: ``kroki`` steps to the lowest ground
        there is, the first step that ends under the ground halved 14 times."""
        idx = np.nonzero(rays)[0]
        far = np.maximum((oz - float(self.z.min())) / -dz[idx], guess) + 1.0
        lo, hi = np.zeros(len(idx)), far.copy()
        found = np.zeros(len(idx), bool)
        for k in range(1, kroki + 1):
            t = far * k / kroki
            under = (above(t, idx) <= 0) & ~found
            hi[under] = t[under]
            lo[~found & ~under] = t[~found & ~under]
            found |= under
            if found.all():
                break
        for _ in range(14):
            mid = 0.5 * (lo + hi)
            under = above(mid, idx) <= 0
            hi = np.where(under, mid, hi)
            lo = np.where(under, lo, mid)
        return hi

    def temperatura(self, pogoda: Pogoda, seed: int = 0) -> np.ndarray:
        """How warm each cell's surface is, C, on this day.

        Lit by the sun as its normal faces it (no cast shadows), dulled by
        cloud; snow cannot be warmer than 0 C; a texture of half a kelvin.
        """
        rng = np.random.default_rng(seed + 17)
        lit = np.clip(self.normalne() @ pogoda.slonce(), 0.0, 1.0) * (1 - 0.8 * pogoda.chmury)
        a = pogoda.powietrze
        t = np.select(
            [
                self.pokrycie == SNIEG,
                self.pokrycie == SKALA,
                self.pokrycie == PIARG,
                self.pokrycie == KOSO,
                self.pokrycie == TRAWA,
                self.pokrycie == WODA,
            ],
            [
                np.minimum(a - 2.0 + 3.0 * lit, 0.0),
                a - 3.0 + 22.0 * lit,
                a - 2.0 + 15.0 * lit,
                a + 1.0 + 4.0 * lit,
                a - 1.0 + 9.0 * lit,
                np.full_like(lit, 1.5),
            ],
            default=a + 6.0 * lit,
        )
        return t + 0.5 * _szum(self.n, rng, skale=(40, 16, 6), amp=(0.4, 0.4, 0.2))


def dolina(seed: int = 0, krok: float = 2.0, polowa: float = 800.0) -> Teren:
    """A Tatra-like valley, ``polowa`` m each way of the search's middle, ``krok`` m a cell.

    A glacial trough falling north from 1750 to 1550 m, its floor winding,
    its sides rising to ridges 2100-2250 m high; a cirque at its head with a
    tarn; gullies down the sides; a trail from the hut at its foot up the
    floor and over a pass on the east ridge.
    """
    from scipy.ndimage import gaussian_filter

    rng = np.random.default_rng(seed)
    n = int(round(2 * polowa / krok)) + 1
    a = np.linspace(-polowa, polowa, n)
    x, y = np.meshgrid(a, a)
    xc = 70 * np.sin(y / 320.0) + 40 * np.sin(y / 130.0 + 1.0)
    floor = 1640.0 - 0.11 * y
    warp = _szum(n, rng, skale=(400, 200, 100), amp=(1.0, 0.5, 0.25))
    across = np.abs(x - xc + 60.0 * warp)
    s = np.clip(across / 650.0, 0, None)
    # The U of a glacial trough, steepening up to the ridges, gentler past them.
    walls = 430.0 * np.minimum(s, 1.0) ** 2.0 + 60.0 * np.clip(s - 1.0, 0, None)
    # The cirque at the head: the floor rises into a bowl, its back wall steep.
    head = np.clip((-y - 260.0) / 420.0, 0, None)
    cirque = 260.0 * head**1.8 + 90.0 * head * np.clip(across / 400.0, 0, 1)
    # Ridges: uneven crests, peaks and saddles.
    ridge_noise = _szum(n, rng, skale=(800, 400, 200, 100), amp=(1.0, 0.6, 0.35, 0.2))
    crest = 1.0 - np.abs(_szum(n, rng, skale=(300, 150, 60), amp=(1.0, 0.5, 0.25)))
    rough = _szum(n, rng)
    z = floor + walls + cirque
    z += 70.0 * ridge_noise * np.clip(across / 500.0, 0, 1)
    z += 45.0 * crest**3 * np.clip((across - 350.0) / 300.0, 0, 1)
    z += 18.0 * rough
    # Gullies down the sides: winding troughs down the fall line, 150-250 m apart.
    wiggle = 50.0 * _szum(n, rng, skale=(300, 150), amp=(1.0, 0.5))
    gully = np.cos(2 * np.pi * (y + wiggle) / 190.0 + rng.uniform(0, 2 * np.pi))
    z -= 16.0 * np.clip(gully, 0, 1) ** 4 * np.clip((across - 180.0) / 250.0, 0, 1)
    z = gaussian_filter(z, 1.0)
    # The tarn: a flat sheet of water in a hollow of the cirque's floor, its
    # shores sloping up to the ground round it over 60 m -- not a well cut
    # into the slope.
    sx, sy, sr = float(np.interp(-400.0, a, xc[:, 0])), -400.0, 55.0
    r = np.hypot(x - sx, y - sy)
    lake = r < sr
    level = float(np.percentile(z[lake], 20))
    shore = np.clip((r - sr) / 60.0, 0.0, 1.0) ** 1.5
    z = np.where(r < sr + 60.0, level + (z - level) * shore, z)
    # The trail: up the floor from the hut, then zig-zag to a pass east.
    base = (float(np.interp(720.0, a, xc[:, 0])) + 25.0, 720.0)
    tr = [(float(np.interp(v, a, xc[:, 0])) + 25.0, v) for v in np.arange(720.0, -150.0, -30.0)]
    zx, zy = tr[-1]
    for k in range(10):
        zx += 45.0
        zy += -25.0 if k % 2 else 25.0
        tr.append((zx, zy))
    szlak = np.array(tr)
    # What covers it.
    gy, gx = np.gradient(z, krok)
    slope = np.degrees(np.arctan(np.hypot(gx, gy)))
    patch = _szum(n, rng, skale=(120, 60, 30, 15), amp=(1.0, 0.6, 0.4, 0.2))
    cover = np.full(z.shape, TRAWA, dtype=np.int8)
    cover[(z < 1800 + 40 * patch) & (slope < 32) & (patch > -0.2)] = KOSO
    snow = (z > 1790 + 60 * patch) & (slope < 42 + 5 * patch)
    cover[snow] = SNIEG
    cover[(slope > 32) & ~snow & (z > 1650)] = PIARG
    cover[slope > 46 + 5 * patch] = SKALA
    cover[lake] = WODA
    for (x0, y0), (x1, y1) in zip(szlak[:-1], szlak[1:], strict=True):
        for f in np.linspace(0, 1, 30):
            px, py = x0 + (x1 - x0) * f, y0 + (y1 - y0) * f
            i, j = int(round((py + polowa) / krok)), int(round((px + polowa) / krok))
            cover[max(0, i - 1) : i + 2, max(0, j - 1) : j + 2] = SZLAK
    return Teren(
        z=z.astype(np.float32),
        pokrycie=cover,
        krok=krok,
        polowa=polowa,
        szlak=szlak,
        staw=(sx, sy, sr),
        baza=base,
    )
