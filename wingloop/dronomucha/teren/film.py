"""A search filmed from behind the platform, with what its sensors reach drawn in.

The valley in 3-D (MuJoCo: the height field, coloured by what covers it and
shaded by the sun), the people, chamois, foxes and boulders as their parts
(:mod:`.obiekty`), and the platform -- a foam plane like a Talon GT Rebel, or
a quadcopter -- followed by a camera behind it. Drawn into the scene, so
that it can be seen where they reach:

- the thermal camera's view: four lines from the lens to the corners of the
  picture on the ground, and the picture's outline laid on the slope;
- the laser rangefinder's beam, along the camera's middle: red to where it
  meets the ground if that is within its reach, yellow to the end of its
  reach if not;
- every candidate: a post, yellow while it waits to be looked at again,
  green when the closer look confirms a person, grey when it does not.

On the frame: the thermal picture, in the classic palette, with the warm spots and what
the brain thinks of them; the map, with the lanes, the way flown, the
ground seen, the candidates; and the numbers -- height above the ground, the
rangefinder's reading, speed, time, battery.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np

from ..platformy import Kamera, Platforma
from . import kamera as K
from . import lot as LOT
from . import teren as T
from . import wykrywanie as W

CACHE = Path(tempfile.gettempdir()) / "dronomucha_teren"
_MAPY: dict = {}


def _tekstura(t: T.Teren, size: int = 2048) -> Path:
    """The ground's colours, shaded by the light from the south-east, north up."""
    import hashlib

    from matplotlib.colors import LightSource
    from PIL import Image
    from scipy.ndimage import zoom

    CACHE.mkdir(exist_ok=True)
    key = hashlib.md5(np.ascontiguousarray(t.z[::40, ::40]).tobytes()).hexdigest()[:10]
    path = CACHE / f"tekstura_{key}_{size}.png"
    if path.exists():
        return path
    rgb = np.array(T.KOLORY)[t.pokrycie]
    ls = LightSource(azdeg=135, altdeg=35)
    shaded = ls.shade_rgb(rgb, t.z, vert_exag=1.0, blend_mode="soft", dx=t.krok, dy=t.krok)
    rng = np.random.default_rng(3)
    shaded = np.clip(
        shaded * (1 + 0.05 * rng.standard_normal(shaded.shape[:2]))[..., None], 0, 1
    )
    f = size / shaded.shape[0]
    big = zoom(shaded, (f, f, 1), order=1)
    Image.fromarray((255 * big[::-1]).astype(np.uint8)).save(path)
    return path


def _geom_xml(part, obj, name) -> str:
    """A part of an object (:class:`~.obiekty.Czesc`) as a MuJoCo geom in the world."""
    rgba = " ".join(f"{c:.3f}" for c in part.kolor) + " 1"
    if part.typ == "kapsula":
        a, b = obj.w_swiecie(part.a), obj.w_swiecie(part.b)
        ft = " ".join(f"{v:.3f}" for v in (*a, *b))
        return (
            f'<geom name="{name}" type="capsule" fromto="{ft}" size="{part.r:.3f}" '
            f'rgba="{rgba}"/>'
        )
    c = obj.w_swiecie(part.a)
    r = " ".join(f"{v:.3f}" for v in part.r)
    deg = np.degrees(obj.yaw)
    return (
        f'<geom name="{name}" type="ellipsoid" pos="{c[0]:.3f} {c[1]:.3f} {c[2]:.3f}" '
        f'size="{r}" euler="0 0 {deg:.1f}" rgba="{rgba}"/>'
    )


SAMOLOT_XML = """
<body name="pojazd" mocap="true">
  <geom type="capsule" fromto="-0.45 0 0 0.42 0 0" size="0.065" rgba="0.92 0.92 0.9 1"/>
  <geom type="ellipsoid" pos="0.45 0 -0.01" size="0.09 0.06 0.055" rgba="0.92 0.92 0.9 1"/>
  <geom type="box" pos="0.02 0 0.06" size="0.11 0.5 0.012" rgba="0.95 0.95 0.95 1"/>
  <geom type="box" pos="0.0 0.47 0.06" size="0.1 0.04 0.013" rgba="1 0.45 0.05 1"/>
  <geom type="box" pos="0.0 -0.47 0.06" size="0.1 0.04 0.013" rgba="1 0.45 0.05 1"/>
  <geom type="box" pos="-0.5 0.09 0.09" size="0.07 0.12 0.006" euler="40 0 0"
    rgba="0.95 0.95 0.95 1"/>
  <geom type="box" pos="-0.5 -0.09 0.09" size="0.07 0.12 0.006" euler="-40 0 0"
    rgba="0.95 0.95 0.95 1"/>
  <geom type="cylinder" pos="-0.12 0 0.12" size="0.03 0.03" euler="0 90 0"
    rgba="0.2 0.2 0.2 1"/>
  <geom type="cylinder" pos="-0.16 0 0.12" size="0.11 0.004" euler="0 90 0"
    rgba="0.1 0.1 0.1 0.5"/>
  <geom type="box" pos="0.38 0 -0.07" size="0.03 0.025 0.02" rgba="0.15 0.15 0.15 1"/>
</body>"""

QUAD_XML = """
<body name="pojazd" mocap="true">
  <geom type="box" size="0.09 0.09 0.04" rgba="0.15 0.15 0.18 1"/>
  <geom type="capsule" fromto="-0.16 -0.16 0 0.16 0.16 0" size="0.013"
    rgba="0.2 0.2 0.22 1"/>
  <geom type="capsule" fromto="-0.16 0.16 0 0.16 -0.16 0" size="0.013"
    rgba="0.2 0.2 0.22 1"/>
  <geom type="cylinder" pos="0.16 0.16 0.03" size="0.025 0.02" rgba="0.3 0.3 0.3 1"/>
  <geom type="cylinder" pos="-0.16 0.16 0.03" size="0.025 0.02" rgba="0.3 0.3 0.3 1"/>
  <geom type="cylinder" pos="0.16 -0.16 0.03" size="0.025 0.02" rgba="0.3 0.3 0.3 1"/>
  <geom type="cylinder" pos="-0.16 -0.16 0.03" size="0.025 0.02" rgba="0.3 0.3 0.3 1"/>
  <geom type="cylinder" pos="0.16 0.16 0.055" size="0.13 0.003" rgba="0.1 0.1 0.1 0.45"/>
  <geom type="cylinder" pos="-0.16 0.16 0.055" size="0.13 0.003" rgba="0.1 0.1 0.1 0.45"/>
  <geom type="cylinder" pos="0.16 -0.16 0.055" size="0.13 0.003" rgba="0.1 0.1 0.1 0.45"/>
  <geom type="cylinder" pos="-0.16 -0.16 0.055" size="0.13 0.003" rgba="0.1 0.1 0.1 0.45"/>
  <geom type="box" pos="0.08 0 -0.07" size="0.03 0.025 0.02" rgba="0.95 0.45 0.05 1"/>
</body>"""


def scena(t: T.Teren, obiekty: list, platforma: Platforma, co: int = 2) -> str:
    """The scene's MJCF: the valley (every ``co``-th cell), the objects, the platform."""
    z = t.z[::co, ::co][::-1]
    zmin, zmax = float(z.min()), float(z.max())
    elev = " ".join(f"{v:.4f}" for v in ((z - zmin) / (zmax - zmin)).ravel())
    n = z.shape[0]
    tex = _tekstura(t)
    bodies = []
    for i, obj in enumerate(obiekty):
        for j, part in enumerate(obj.czesci):
            if not part.pod_sniegiem:
                bodies.append(_geom_xml(part, obj, f"o{i}_{j}"))
    vehicle = SAMOLOT_XML if not platforma.zawis else QUAD_XML
    return f"""<mujoco model="dolina">
  <statistic extent="100" center="0 0 {zmin}"/>
  <visual>
    <map znear="0.002" zfar="60" fogstart="12" fogend="40"/>
    <quality shadowsize="4096"/>
    <headlight ambient="0.35 0.35 0.38" diffuse="0.45 0.45 0.45" specular="0.05 0.05 0.05"/>
    <rgba fog="0.72 0.8 0.9 1" haze="0.75 0.82 0.92 1"/>
    <global offwidth="1920" offheight="1080"/>
  </visual>
  <asset>
    <texture name="niebo" type="skybox" builtin="gradient" rgb1="0.55 0.7 0.92"
      rgb2="0.9 0.94 1" width="512" height="512"/>
    <texture name="ziemia" type="2d" file="{tex}"/>
    <material name="ziemia" texture="ziemia" texrepeat="1 1" texuniform="false"
      specular="0.05" shininess="0.05"/>
    <hfield name="dolina" nrow="{n}" ncol="{n}" size="{t.polowa} {t.polowa} {zmax - zmin} 50"
      elevation="{elev}"/>
  </asset>
  <worldbody>
    <light directional="true" dir="-0.4 0.9 -0.45" diffuse="0.7 0.68 0.62" castshadow="false"/>
    <geom type="hfield" hfield="dolina" pos="0 0 {zmin}" material="ziemia"
      contype="0" conaffinity="0"/>
    {"".join(bodies)}
    {vehicle}
  </worldbody>
</mujoco>"""


def _linia(scn, a, b, rgba, width=2.0):
    """A line from ``a`` to ``b`` into the scene being drawn."""
    import mujoco

    if scn.ngeom >= scn.maxgeom:
        return
    g = scn.geoms[scn.ngeom]
    mujoco.mjv_initGeom(
        g,
        mujoco.mjtGeom.mjGEOM_LINE,
        np.zeros(3),
        np.zeros(3),
        np.zeros(9),
        np.asarray(rgba, float),
    )
    mujoco.mjv_connector(
        g, mujoco.mjtGeom.mjGEOM_LINE, width, np.asarray(a, float), np.asarray(b, float)
    )
    scn.ngeom += 1


def _kula(scn, p, r, rgba):
    import mujoco

    if scn.ngeom >= scn.maxgeom:
        return
    g = scn.geoms[scn.ngeom]
    mujoco.mjv_initGeom(
        g,
        mujoco.mjtGeom.mjGEOM_SPHERE,
        np.array([r, r, r]),
        np.asarray(p, float),
        np.eye(3).ravel(),
        np.asarray(rgba, float),
    )
    scn.ngeom += 1


def zasieg_kamery(t: T.Teren, kamera: Kamera, skad, kurs: float, n: int = 16):
    """The four lines from the lens to the picture's corners, and its outline on the ground:
    (corners (4, 3), outline (4n, 3))."""
    w, h = kamera.px
    tw = np.tan(np.deg2rad(kamera.fov) / 2)
    th = tw * h / w
    b = np.array([0.0, 0.0, -1.0])
    left = np.array([-np.sin(kurs), np.cos(kurs), 0.0])
    up = np.cross(b, left)
    edge = []
    for (x0, y0), (x1, y1) in (
        ((tw, th), (-tw, th)),
        ((-tw, th), (-tw, -th)),
        ((-tw, -th), (tw, -th)),
        ((tw, -th), (tw, th)),
    ):
        f = np.linspace(0, 1, n, endpoint=False)[:, None]
        tx, ty = x0 + (x1 - x0) * f, y0 + (y1 - y0) * f
        d = b + tx * left + ty * up
        edge.append(d / np.linalg.norm(d, axis=1, keepdims=True))
    rays = np.concatenate(edge)
    pts = t.trafienie(np.asarray(skad, float), rays)
    return pts[::n], pts


def _ir_panel(widok, plamy, mozg, size, font):
    """The thermal picture in the classic palette, with the warm spots boxed and judged."""
    from PIL import Image, ImageDraw

    temp = widok.temp
    lo, hi = np.percentile(temp, 1), np.percentile(temp, 99.7)
    hi = max(hi, lo + 3.0)
    img = Image.fromarray(ironbow(temp, lo, hi)).resize(size, Image.BILINEAR)
    d = ImageDraw.Draw(img)
    sx, sy = size[0] / temp.shape[1], size[1] / temp.shape[0]
    for p in plamy:
        rc = np.unravel_index(
            np.argmin(
                np.hypot(widok.ziemia[..., 0] - p.gdzie[0], widok.ziemia[..., 1] - p.gdzie[1])
            ),
            temp.shape,
        )
        v = mozg.ocena(p)
        col = (90, 255, 120) if v > 0 else (255, 210, 40)
        x, y = rc[1] * sx, rc[0] * sy
        d.rectangle((x - 9, y - 9, x + 9, y + 9), outline=col, width=2)
        d.text((x + 11, y - 9), f"{v:+.2f}", fill=col, font=font)
    return img


def _mapa(t: T.Teren, wynik_slad, lanes, kandydaci, rogi, size: int, obiekty):
    """The map (a PIL image): hill-shaded ground, the lanes, the way flown, the picture,
    candidates, people."""
    from PIL import Image, ImageDraw

    key = (id(t), size)
    if key not in _MAPY:
        _MAPY[key] = Image.open(_tekstura(t)).resize((size, size), Image.BILINEAR)
    base = _MAPY[key].copy()
    d = ImageDraw.Draw(base)

    def px(x, y):
        return ((x + t.polowa) / (2 * t.polowa) * size, (t.polowa - y) / (2 * t.polowa) * size)

    a, b = px(-LOT.POLE, LOT.POLE), px(LOT.POLE, -LOT.POLE)
    d.rectangle((*a, *b), outline=(0, 200, 255), width=2)
    for (x0, y0), (x1, y1) in lanes:
        d.line((*px(x0, y0), *px(x1, y1)), fill=(255, 255, 255), width=1)
    if len(wynik_slad) > 1:
        d.line([px(s[1], s[2]) for s in wynik_slad[::5]], fill=(255, 60, 30), width=2)
    if rogi is not None:
        d.polygon([px(*r[:2]) for r in rogi], outline=(255, 255, 0))
    for o in obiekty:
        if o.czlowiek:
            x, y = px(o.x, o.y)
            d.ellipse((x - 4, y - 4, x + 4, y + 4), outline=(255, 0, 255), width=2)
    for c in kandydaci:
        x, y = px(*c.gdzie[:2])
        col = {
            "czeka": (255, 210, 0),
            "potwierdzony": (0, 230, 80),
            "odrzucony": (140, 140, 140),
        }
        d.ellipse((x - 3, y - 3, x + 3, y + 3), fill=col[c.stan])
    return base


def film(
    path,
    platforma: Platforma,
    kamera: Kamera,
    teren: T.Teren,
    ground_temp,
    obiekty: list,
    mozg: W.Mozg,
    agl: float | None = None,
    size=(1280, 720),
    fps: int = 24,
    przyspieszenie: float = 12.0,
    zwolnij: float = 3.0,
    do: float | None = None,
    dalmierz: str = "TF03-180",
    seed: int = 0,
    progress=print,
) -> Path:
    """Record a search from behind the platform: ``przyspieszenie`` times real time on the
    lanes, ``zwolnij`` times while it looks again; until ``do`` seconds or the end."""
    import imageio.v2 as imageio
    import mujoco
    from PIL import Image, ImageDraw

    from wingloop.lab.autopilot import _font

    os.environ.setdefault("MUJOCO_GL", "osmesa")
    model = mujoco.MjModel.from_xml_string(scena(teren, obiekty, platforma))
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, size[1], size[0], max_geom=6000)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    font, small = _font(size[1] // 30), _font(size[1] // 42)
    writer = imageio.get_writer(
        str(path), fps=fps, codec="libx264", quality=8, macro_block_size=1
    )
    lanes = LOT.pasy(platforma, kamera, agl or platforma.wysokosc)
    state = dict(view=None, spots=[], clock=0.0, trail=[], frames=0, az=None)
    rng = np.random.default_rng(seed)

    def on_picture(lot, widok, plamy, faza):
        state["view"], state["spots"] = widok, plamy

    def on_step(t, lot, faza, cel, kandydaci):
        state["trail"].append((t, lot.x, lot.y, lot.z))
        speed = zwolnij if faza == "potwierdz" else przyspieszenie
        state["clock"] += 0.2 / speed
        if state["clock"] < 1.0 / fps:
            return do is not None and t > do
        state["clock"] = 0.0
        # The platform, pitched and banked as it flies.
        head = lot.kurs
        bank = 0.0
        if len(state["trail"]) > 3 and not platforma.zawis:
            dh = (head - state.get("head", head) + np.pi) % (2 * np.pi) - np.pi
            bank = np.clip(np.arctan2(lot.v * dh / 0.2, LOT.G), -0.6, 0.6)
        state["head"] = head
        quat = np.zeros(4)
        mujoco.mju_euler2Quat(quat, np.array([bank, 0.0, head]), "xyz")
        data.mocap_pos[0] = (lot.x, lot.y, lot.z)
        data.mocap_quat[0] = quat
        mujoco.mj_forward(model, data)
        # The chase camera: behind, a little above, turning smoothly after it.
        az = np.degrees(head)
        if state["az"] is None:
            state["az"] = az
        state["az"] += ((az - state["az"] + 180) % 360 - 180) * 0.25
        back = 11.0 if not platforma.zawis else 7.0
        cam.lookat[:] = (lot.x + 2 * np.cos(head), lot.y + 2 * np.sin(head), lot.z - 2.5)
        cam.distance, cam.azimuth, cam.elevation = back, state["az"], -20.0
        renderer.update_scene(data, camera=cam)
        scn = renderer.scene
        lens = np.array([lot.x, lot.y, lot.z - 0.08])
        corners, outline = zasieg_kamery(teren, kamera, lens, head)
        for c in corners:
            _linia(scn, lens, c, (1.0, 0.85, 0.1, 0.9), 2.0)
        lifted = outline + np.array([0, 0, 0.6])
        for a, b in zip(lifted, np.roll(lifted, -1, axis=0), strict=True):
            _linia(scn, a, b, (1.0, 0.85, 0.1, 1.0), 3.0)
        reach = K.DALMIERZE[dalmierz][0]
        dist, hit = K.dalmierz(teren, lens, (0, 0, -1), dalmierz, rng)
        if dist is not None:
            _linia(scn, lens, hit, (1.0, 0.1, 0.1, 1.0), 4.0)
            _kula(scn, hit, 0.8, (1.0, 0.1, 0.1, 1.0))
        else:
            _linia(scn, lens, lens + np.array([0, 0, -reach]), (1.0, 1.0, 0.1, 1.0), 4.0)
        for c in kandydaci:
            col = {
                "czeka": (1, 0.85, 0),
                "potwierdzony": (0, 0.9, 0.3),
                "odrzucony": (0.6, 0.6, 0.6),
            }
            g = np.array([c.gdzie[0], c.gdzie[1], float(teren.wysokosc(*c.gdzie[:2]))])
            _linia(scn, g, g + np.array([0, 0, 25.0]), (*col[c.stan], 1.0), 3.0)
            _kula(scn, g + np.array([0, 0, 25.0]), 1.2, (*col[c.stan], 1.0))
        frame = Image.fromarray(renderer.render())
        dr = ImageDraw.Draw(frame)
        W_, H_ = size
        if state["view"] is not None:
            panel = _ir_panel(state["view"], state["spots"], mozg, (W_ // 3, W_ // 4), small)
            frame.paste(panel, (16, H_ - panel.size[1] - 16))
            dr.rectangle(
                (16, H_ - panel.size[1] - 46, 16 + panel.size[0], H_ - panel.size[1] - 16),
                fill=(0, 0, 0),
            )
            dr.text(
                (24, H_ - panel.size[1] - 44),
                f"termowizja {kamera.nazwa}",
                fill=(255, 255, 255),
                font=small,
            )
        side = H_ // 3
        m = _mapa(
            teren,
            state["trail"],
            lanes,
            kandydaci,
            outline[::16] if outline is not None else None,
            side,
            obiekty,
        )
        m = m.resize((side, side))
        frame.paste(m, (W_ - side - 16, 16))
        names = {
            "pasy": "przeszukiwanie pasami",
            "potwierdz": "zniża się, by obejrzeć z bliska",
            "wroc": "wraca na pas",
            "powrot": "powrót do bazy",
        }
        found = sum(c.stan == "potwierdzony" for c in kandydaci)
        lrf = f"{dist:5.1f} m" if dist is not None else f"poza zasięgiem ({reach:.0f} m)"
        battery = max(0.0, 100 * (1 - t / (platforma.lot_min * 60)))
        lines = [
            f"{platforma.pl}  ·  t = {t // 60:02.0f}:{t % 60:02.0f}  ·  ×{speed:.0f}",
            names.get(faza, faza),
            f"nad ziemią {lot.nad_ziemia:5.1f} m   dalmierz {lrf}",
            f"prędkość {lot.v:4.1f} m/s   bateria {battery:3.0f}%   zgłoszeni: {found}",
        ]
        dr.rectangle((10, 10, 10 + W_ // 2.3, 14 + 34 * len(lines)), fill=(0, 0, 0))
        for i, line in enumerate(lines):
            dr.text(
                (20, 16 + 34 * i), line, fill=(255, 255, 255), font=font if i < 1 else small
            )
        writer.append_data(np.asarray(frame))
        state["frames"] += 1
        if progress and state["frames"] % 200 == 0:
            progress(f"  {state['frames']} klatek, t = {t:.0f} s")
        return do is not None and t > do

    try:
        LOT.szukaj(
            platforma,
            kamera,
            teren,
            ground_temp,
            obiekty,
            mozg,
            agl=agl,
            seed=seed,
            klatka=on_picture,
            krok=on_step,
        )
    finally:
        writer.close()
    return Path(path)


#: The classic thermal palette, cold to hot: black, violet, magenta, orange, yellow, white.
IRONBOW = np.array(
    [
        [0, 0, 0],
        [40, 0, 90],
        [120, 10, 140],
        [200, 40, 90],
        [240, 110, 20],
        [250, 200, 40],
        [255, 255, 220],
    ],
    dtype=float,
)


def ironbow(temp, lo: float, hi: float) -> np.ndarray:
    """Temperatures as the classic thermal palette's colours, ``lo`` black to ``hi`` white."""
    x = np.clip((np.asarray(temp) - lo) / max(hi - lo, 1e-6), 0, 1) ** 0.85
    x = x * (len(IRONBOW) - 1)
    rgb = [np.interp(x, np.arange(len(IRONBOW)), IRONBOW[:, c]) for c in range(3)]
    return np.stack(rgb, axis=-1).astype(np.uint8)


def przelot(
    path,
    platforma: Platforma,
    kamera: Kamera,
    teren: T.Teren,
    ground_temp,
    obiekty: list,
    mozg: W.Mozg,
    agl: float | None = None,
    nad=None,
    size=(1280, 720),
    fps: int = 24,
    tempo: float = 1.0,
    dalmierz: str = "TF03-180",
    seed: int = 0,
    zwolnij: float = 0.3,
) -> Path:
    """One pass along one lane, as the thermal camera sees it, full screen, ``tempo`` times
    real time -- ``zwolnij`` times as slow within 45 m of ``nad``: the lane over ``nad``
    (x, y), or the first person's.

    The classic palette, its gain following the picture slowly; a crosshair
    with the temperature under it and the rangefinder's reading; every warm
    spot boxed, with its temperature and what the fly's brain thinks of it
    -- green when it likes it.
    """
    import imageio.v2 as imageio
    from PIL import Image, ImageDraw

    from wingloop.lab.autopilot import _font

    agl = agl or platforma.wysokosc
    if nad is None:
        person = next(o for o in obiekty if o.czlowiek)
        nad = (person.x, person.y)
    lanes = LOT.pasy(platforma, kamera, agl)
    lane = min(lanes, key=lambda ln: abs(ln[0][0] - nad[0]))
    (x0, y0), (x1, y1) = lane
    lot = LOT.Lot(platforma, teren, (x0, y0), agl)
    lot.kurs = float(np.arctan2(y1 - y0, x1 - x0))
    lot.z = float(teren.wysokosc(x0, y0)) + agl
    rng = np.random.default_rng(seed)
    W_, H_ = size
    ih = H_
    iw = int(round(ih * kamera.px[0] / kamera.px[1]))
    left = (W_ - iw) // 2
    font, small = _font(H_ // 28), _font(H_ // 40)
    writer = imageio.get_writer(
        str(path), fps=fps, codec="libx264", quality=8, macro_block_size=1
    )
    lo = hi = None
    t = 0.0
    try:
        while t < 200.0:
            near = np.hypot(lot.x - nad[0], lot.y - nad[1]) < 45.0
            dt = tempo * (zwolnij if near else 1.0) / fps
            if lot.krok(dt, (x1, y1), agl) < 15.0:
                break
            t += dt
            v = K.zdjecie(
                teren, ground_temp, obiekty, kamera, (lot.x, lot.y, lot.z), lot.kurs, rng=rng
            )
            spots = W.plamy(v)
            a, b = np.percentile(v.temp, 1), np.percentile(v.temp, 99.8)
            lo = a if lo is None else lo + 0.15 * (a - lo)
            hi = b if hi is None else hi + 0.15 * (b - hi)
            top = max(hi, lo + 4.0)
            img = Image.fromarray(ironbow(v.temp, lo, top)).resize((iw, ih), Image.BICUBIC)
            frame = Image.new("RGB", size, (8, 8, 12))
            frame.paste(img, (left, 0))
            d = ImageDraw.Draw(frame)
            sx, sy = iw / v.temp.shape[1], ih / v.temp.shape[0]
            for s in spots:
                r, c = np.unravel_index(
                    np.argmin(
                        np.hypot(v.ziemia[..., 0] - s.gdzie[0], v.ziemia[..., 1] - s.gdzie[1])
                    ),
                    v.temp.shape,
                )
                half = max(14.0, 0.6 * s.cechy[3] / max(s.cechy[5], 1e-3) * sx)
                cx, cy = left + (c + 0.5) * sx, (r + 0.5) * sy
                val = mozg.ocena(s)
                col = (60, 255, 110) if val > 0 else (235, 235, 235)
                d.rectangle((cx - half, cy - half, cx + half, cy + half), outline=col, width=2)
                d.text((cx + half + 6, cy - half), f"{s.cechy[0]:+.1f}°C", fill=col, font=small)
                label = "człowiek?" if val > 0 else "nie"
                d.text(
                    (cx + half + 6, cy - half + small.size + 4),
                    f"{label} {val:+.2f}",
                    fill=col,
                    font=small,
                )
            # The crosshair, and what is under it.
            cx, cy = left + iw // 2, ih // 2
            for p, q in (
                ((-40, 0), (-12, 0)),
                ((12, 0), (40, 0)),
                ((0, -40), (0, -12)),
                ((0, 12), (0, 40)),
            ):
                d.line(
                    (cx + p[0], cy + p[1], cx + q[0], cy + q[1]), fill=(90, 255, 120), width=2
                )
            h_, w_ = v.temp.shape
            centre = float(
                np.median(v.temp[h_ // 2 - 2 : h_ // 2 + 3, w_ // 2 - 2 : w_ // 2 + 3])
            )
            dist, _ = K.dalmierz(teren, (lot.x, lot.y, lot.z), (0, 0, -1), dalmierz, rng)
            reach = K.DALMIERZE[dalmierz][0]
            lrf = f"{dist:5.1f} m" if dist is not None else f"--- (>{reach:.0f} m)"
            lines = [
                "IR  ironbow",
                f"{kamera.nazwa}",
                f"{kamera.px[0]}×{kamera.px[1]}  {kamera.fov:.0f}°",
                "",
                f"t  {t:5.1f} s  ×{tempo * (zwolnij if near else 1.0):g}",
                f"T  {centre:+5.1f} °C",
                f"DAL {lrf}",
                f"AGL {lot.nad_ziemia:5.1f} m",
                f"PIX {np.median(v.gsd) * 100:4.0f} cm",
                f"V   {lot.v:4.1f} m/s",
                f"KURS {(90 - np.degrees(lot.kurs)) % 360:03.0f}°",
                "",
                f"plamy: {len(spots)}",
                f"lubi: {sum(mozg.ocena(s) > 0 for s in spots)}",
            ]
            for i, line in enumerate(lines):
                d.text(
                    (14, 16 + int(small.size * 1.45) * i), line, fill=(90, 255, 120), font=small
                )
            # The gain's scale, right.
            bx, by, bh = left + iw + 30, 60, ih - 120
            for j in range(bh):
                col = ironbow(np.array([top - (top - lo) * j / bh]), lo, top)[0]
                d.line((bx, by + j, bx + 22, by + j), fill=tuple(int(c) for c in col))
            d.text(
                (bx - 4, by - font.size - 8), f"{top:+.0f}°C", fill=(240, 240, 240), font=small
            )
            d.text((bx - 4, by + bh + 8), f"{lo:+.0f}°C", fill=(240, 240, 240), font=small)
            writer.append_data(np.asarray(frame))
    finally:
        writer.close()
    return Path(path)
