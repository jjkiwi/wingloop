"""A search filmed as its operator sees it: the thermal camera full screen.

What the drone's thermal camera -- on a gimbal, level, looking where the
drone heads -- sees, 640 x 360, white hot, its gain set to the warmest
thing in view; on it, as a thermal camera's screen has: a crosshair, the
temperature there, the laser rangefinder's distance, the heading, the time,
and the mission's map in a corner. The world is the one the eye and the
sensors see (:mod:`wingloop.lab.watch`), the temperatures the job's.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from wingloop.lab import autopilot, watch

from .misja import obszar, pilot
from .mozg import mozg
from .swiat import oko, swiat
from .zadanie import Zadanie

#: The thermal camera's picture, pixels, its field across, degrees, and how far
#: down it looks, degrees, as a search camera on its gimbal does.
SIZE, FOV, TILT = (640, 360), 60.0, 15.0


def promienie(size=SIZE, fov: float = FOV) -> np.ndarray:
    """The camera's rays in its own frame (x ahead, y left, z up): (height, width, 3)."""
    w, h = size
    half = np.tan(np.deg2rad(fov) / 2)
    u = np.linspace(half, -half, w)
    v = np.linspace(half * h / w, -half * h / w, h)
    uu, vv = np.meshgrid(u, v)
    d = np.stack([np.ones_like(uu), uu, vv], axis=-1)
    return d / np.linalg.norm(d, axis=-1, keepdims=True)


def gimbal(R, tilt: float = TILT) -> np.ndarray:
    """The camera's rotation: the drone's heading, level, then ``tilt`` degrees down."""
    from wingloop.lab.scene import rotation
    from wingloop.lab.watch import _level

    return _level(R) @ rotation(0.0, np.deg2rad(tilt), 0.0)


def kule(parts, at, heat: float) -> list:
    """A thermal camera's solids for a drawn model (:func:`~wingloop.lab.arena.add_parts`).

    Spheres along each capsule and each ellipsoid's longest axis, boxes for
    boxes, each part as warm as :data:`~.wyglad.CIEPLO` says. For the
    operator's screen only: the drone's eye and sensors see the kind's solid.
    """
    from wingloop.lab.scene import Thing

    from .wyglad import CIEPLO

    at = np.asarray(at, dtype=float)
    out = []
    for part in parts:
        h = heat * CIEPLO.get(tuple(part.get("rgba", ())), 1.0)
        kind, size = part["type"], np.ravel(part["size"]).astype(float)
        if kind == "capsule":
            a, b = np.split(np.asarray(part["fromto"], dtype=float), 2)
            r = size[0]
        elif kind == "ellipsoid":
            c = np.asarray(part["pos"], dtype=float)
            axis = int(np.argmax(size))
            r = float(np.sort(size)[1])
            step = np.zeros(3)
            step[axis] = max(size[axis] - r, 0.0)
            a, b = c - step, c + step
        elif kind in ("box", "cylinder"):
            half = size if kind == "box" else np.array([size[0], size[0], size[1]])
            out.append(Thing("box", at + part["pos"], half, heat=h))
            continue
        else:
            out.append(Thing("sphere", at + part["pos"], float(size[0]), heat=h))
            continue
        n = max(2, int(np.ceil(np.linalg.norm(b - a) / (0.6 * r))) + 1)
        for f in np.linspace(0, 1, n):
            out.append(Thing("sphere", at + a + (b - a) * f, r, heat=h))
    return out


def swiat_cieplny(fly_eye, items, t: float):
    """What the operator's thermal camera shows: (the eye's world without the items, the
    items as groups of their drawn models' warm parts, :func:`kule`)."""
    from wingloop.lab import arena
    from wingloop.lab.scene import World

    solids = fly_eye.world.things[len(fly_eye.background) :]
    plain, groups = list(fly_eye.background), []
    for it, solid in zip(items, solids, strict=True):
        spec = arena.KINDS[it.kind]
        if "parts" not in spec:
            plain.append(solid)
            continue
        x, y = it.at(t)
        at = (x, y, spec.get("z", it.z))
        reach = max(np.linalg.norm(np.ravel(spec["size"])) * 1.5 + 4.0, 12.0)
        groups.append((at, reach, kule(spec["parts"], at, solid.heat)))
    return World(plain, ground_height=fly_eye.world.ground_height), groups


def obraz(fly_eye, rays, rng, world=None, groups=()) -> tuple[np.ndarray, float]:
    """Kelvin above the ground, as the camera's optics and noise give it, and the
    distance along its middle (its laser rangefinder), mm -- from where the drone
    looked last."""
    from scipy.ndimage import gaussian_filter

    from wingloop.lab.scene import Camera, depth, thermal_along

    o, R = fly_eye.pose
    G = gimbal(R)
    world = world or fly_eye.world
    k = thermal_along(
        world, o, rays @ G.T, clutter=watch.IR_CLUTTER, sky=-watch.THERMAL_SPAN, groups=groups
    )
    k = gaussian_filter(k, 1.2) + watch.IR_NETD * rng.standard_normal(k.shape)
    far = float(depth(fly_eye.world, o, Camera(size=3, fov=1.0), rotation_matrix=G).min())
    return k, far


def klatka(seen, p, z: Zadanie, size, font, small) -> np.ndarray:
    """One frame: the picture white hot, full size, and the screen's marks on it."""
    from PIL import Image, ImageDraw
    from scipy import ndimage

    k, rng = seen

    hot = float(
        np.clip(np.percentile(k, 99.9), watch.THERMAL_SPAN_MIN, 1.5 * watch.THERMAL_SPAN)
    )
    x = np.clip((k + 1.5) / (hot + 1.5), 0, 1) ** 0.8
    grey = (255 * x).astype(np.uint8)
    img = Image.fromarray(grey).convert("RGB").resize(size, Image.BILINEAR)
    d = ImageDraw.Draw(img)
    W, H = size
    green = (90, 255, 120)
    # The crosshair, and what is under it.
    cx, cy = W // 2, H // 2
    for a, b in (
        ((-40, 0), (-12, 0)),
        ((12, 0), (40, 0)),
        ((0, -40), (0, -12)),
        ((0, 12), (0, 40)),
    ):
        d.line((cx + a[0], cy + a[1], cx + b[0], cy + b[1]), fill=green, width=2)
    h, w = k.shape
    centre = float(np.median(k[h // 2 - 2 : h // 2 + 3, w // 2 - 2 : w // 2 + 3]))
    far = f"{rng / 20:.1f} m" if rng <= watch.LRF_RANGE else "---"
    # Hot spots, as a search camera's screen marks them: what stands out
    # from the ground by more than a third of the span, with its warmest.
    sx, sy = W / w, H / h
    spots, n = ndimage.label(k > max(4.0, hot / 3))
    for j, box in enumerate(ndimage.find_objects(spots), start=1):
        rows, cols = box
        if (rows.stop - rows.start) * (cols.stop - cols.start) < 12:
            continue
        x0, y0 = cols.start * sx - 6, rows.start * sy - 6
        x1, y1 = cols.stop * sx + 6, rows.stop * sy + 6
        d.rectangle((x0, y0, x1, y1), outline=(255, 210, 40), width=2)
        warmest = float(k[box][spots[box] == j].max())
        d.text(
            (x0, max(0, y0 - 26)),
            f"{z.ground_c + warmest:+.0f}°C",
            fill=(255, 210, 40),
            font=small,
        )
    st = p.status()
    heading = st["fly"]["heading"] % 360
    lines_left = [
        f"T {z.ground_c + centre:+5.1f} °C",
        f"ODL {far}",
        f"KURS {heading:03.0f}°",
    ]
    for i, line in enumerate(lines_left):
        d.text((20, H - 30 - 34 * (len(lines_left) - i)), line, fill=green, font=font)
    found = ", ".join(st["reached"]) or "-"
    top = [
        f"IR WHITE HOT  {z.pl}",
        f"t = {p.flight.t:4.1f} s   dron {st['phase_pl']}",
        f"znaleziono: {found}",
    ]
    for i, line in enumerate(top):
        d.text((20, 16 + 30 * i), line, fill=green, font=small)
    # Gain scale: the coldest and the warmest shown, in degrees.
    bar_x, bar_y, bar_h = W - 46, H - 250, 200
    for j in range(bar_h):
        c = int(255 * (1 - j / bar_h) ** 0.8)
        d.line((bar_x, bar_y + j, bar_x + 14, bar_y + j), fill=(c, c, c))
    d.text((bar_x - 70, bar_y - 6), f"{z.ground_c + hot:+.0f}°", fill=green, font=small)
    d.text(
        (bar_x - 70, bar_y + bar_h - 18), f"{z.ground_c - 1.5:+.0f}°", fill=green, font=small
    )
    # The mission's map, top right.
    side = H // 3
    m = Image.fromarray(autopilot.map_image(st, side))
    img.paste(m, (W - side - 16, 16))
    return np.asarray(img)


def film_termo(
    z: Zadanie,
    path,
    seed: int = 0,
    size=(1280, 720),
    progress=print,
    seconds=None,
    dodatki=None,
) -> Path:
    """Record one search, the thermal camera full screen, with ``dodatki``."""
    import imageio.v2 as imageio

    from .dodatki import MUCHA

    dodatki = dodatki or MUCHA
    fly_eye = oko(z)
    mb, valence = mozg(fly_eye, z, progress=progress, dodatki=dodatki)
    p = pilot(fly_eye, mb, valence, z, obszar(z, seed), dodatki)
    seconds = seconds or autopilot.MISSION_S[True]
    rays = promienie()
    noise = np.random.default_rng(seed)
    font = autopilot._font(size[1] // 22)
    small = autopilot._font(size[1] // 30)
    fps, frame_ms = 30, 33.0
    looks = max(1, int(round(frame_ms / fly_eye.every_ms)))
    path = Path(path)
    writer = imageio.get_writer(path, fps=fps, codec="libx264", quality=8, macro_block_size=1)
    try:
        with swiat(z):
            while p.flight.t < seconds:
                for _ in range(looks):
                    p.step()
                if not p.flight.upright():
                    break
                world, groups = swiat_cieplny(fly_eye, p.items, p.flight.t)
                seen = obraz(fly_eye, rays, noise, world, groups)
                frame = klatka(seen, p, z, size, font, small)
                writer.append_data(frame)
                if p.done:
                    for _ in range(fps):
                        writer.append_data(frame)
                    break
            if progress:
                progress(p.summary())
    finally:
        writer.close()
    return path
