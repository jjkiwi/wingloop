"""The search at real scale: a Tatra valley, a thermal camera looking down, two platforms."""

from __future__ import annotations

import numpy as np
import pytest

from wingloop.dronomucha.platformy import KAMERY, PLATFORMY
from wingloop.dronomucha.teren import kamera as K
from wingloop.dronomucha.teren import lot as LOT
from wingloop.dronomucha.teren import obiekty as O
from wingloop.dronomucha.teren import teren as T
from wingloop.dronomucha.teren import wykrywanie as W


@pytest.fixture(scope="module")
def dolina():
    t = T.dolina(krok=4.0)
    pg = T.Pogoda()
    return t, pg, t.temperatura(pg)


def test_the_valley_is_a_tatra_valley(dolina):
    """A floor about 1550-1750 m, ridges about 2100-2300 m; snow, rock, dwarf pine, a tarn."""
    t, _, temp = dolina
    floor = t.wysokosc(t.baza[0], t.baza[1])
    assert 1500 < floor < 1700 and 2050 < np.percentile(t.z, 95) < 2500
    for kind in (T.SNIEG, T.SKALA, T.KOSO, T.WODA, T.SZLAK):
        assert (t.pokrycie == kind).any()
    assert temp[t.pokrycie == T.SNIEG].max() <= 0.5  # snow is not warmer than 0 C (+ texture)


def test_a_ray_from_the_air_meets_the_ground(dolina):
    t, _, _ = dolina
    o = np.array([30.0, -40.0, float(t.wysokosc(30.0, -40.0)) + 80.0])
    d = K.promienie(KAMERY["t256s"], 0.4, 90.0)
    hit = t.trafienie(o, d)
    assert np.allclose(hit[..., 2], t.wysokosc(hit[..., 0], hit[..., 1]), atol=0.5)


def test_a_person_fades_into_their_pixels_from_higher_up(dolina):
    """Mixed into a bigger pixel, a person is less warm above the ground round them."""
    t, pg, temp = dolina
    rng = np.random.default_rng(0)
    person = O.Obiekt("czlowiek_lezy", 40.0, 300.0, 0.0, 0.5, O.czlowiek("lezy", -2.0, rng))
    person.z = float(t.wysokosc(person.x, person.y))
    peaks = []
    for h in (30.0, 60.0, 120.0):
        v = K.zdjecie(
            t, temp, [person], KAMERY["t256s"], (40.0, 300.0, person.z + h), 0.0, rng=rng
        )
        spots = [s for s in W.plamy(v) if np.hypot(*(s.gdzie[:2] - (40.0, 300.0))) < 3]
        peaks.append(max(s.cechy[1] for s in spots) if spots else 0.0)
    assert peaks[0] > peaks[1] > peaks[2] and peaks[0] > 8.0
    mlx = K.zdjecie(
        t, temp, [person], KAMERY["mlx90640_55"], (40.0, 300.0, person.z + 60.0), 0.0
    )
    assert not [s for s in W.plamy(mlx) if np.hypot(*(s.gdzie[:2] - (40.0, 300.0))) < 3]


def test_both_platforms_fly_lanes_over_the_square_kilometre(dolina):
    """The plane flies every lane, skipping to turn; the quadcopter the trail's first, from
    the end nearer its start."""
    t, _, _ = dolina
    plane = LOT.pasy(PLATFORMY["samolot"], KAMERY["t256s"], 60.0, start=t.baza)
    quad = LOT.pasy(PLATFORMY["quad"], KAMERY["t256s"], 40.0, start=t.baza)
    xs = sorted(lane[0][0] for lane in plane)
    assert xs[0] < -470 and xs[-1] > 470 and len(set(xs)) == len(xs)
    assert abs(plane[1][0][0] - plane[0][0][0]) > 60  # it skips lanes to turn
    assert abs(quad[0][0][0] - 60.0) < 20  # the lane nearest the trail first
    assert quad[0][0][1] > 0 and t.baza[1] > 0  # from its northern end: the base is north


def test_a_ray_meets_the_ground_on_the_steepest_slopes_too(dolina):
    """Over the valley's walls and looking down them, every pixel's point is on the ground."""
    t, _, _ = dolina
    slope = t.nachylenie()
    a = np.linspace(-t.polowa, t.polowa, t.n)
    inside = (np.abs(a) < 450)[:, None] & (np.abs(a) < 450)[None, :]
    i, j = np.unravel_index(np.argmax(np.where(inside, slope, 0)), slope.shape)
    assert slope[i, j] > 45
    for h in (15.0, 60.0):
        o = np.array([a[j], a[i], float(t.z[i, j]) + h])
        hit = t.trafienie(o, K.promienie(KAMERY["t256s"], 1.0, 90.0, krawedzie=True)[::4, ::4])
        assert np.abs(hit[..., 2] - t.wysokosc(hit[..., 0], hit[..., 1])).max() < 0.3


def test_someone_standing_is_seen_from_the_side_at_a_pictures_edge(dolina):
    """Straight below, a head and shoulders; 20 m to the side, a body lying outward."""
    t, _, temp = dolina
    rng = np.random.default_rng(0)
    person = O.Obiekt("czlowiek_stoi", 40.0, 300.0, 0.0, 0.0, O.czlowiek("stoi", -2.0, rng))
    person.z = float(t.wysokosc(person.x, person.y))
    grid = np.arange(-3.0, 3.0, 0.02)
    gx, gy = np.meshgrid(person.x + grid, person.y + grid)
    above = np.isfinite(K._wewnatrz(person, gx, gy, (person.x, person.y, person.z + 60.0)))
    side = np.isfinite(
        K._wewnatrz(person, gx, gy, (person.x, person.y + 20.0, person.z + 60.0))
    )
    assert side.sum() > 1.3 * above.sum() and person.wysokosc() > 1.6
    # And the picture has them where they are seen: a spot within a metre or so of them.
    v = K.zdjecie(
        t, temp, [person], KAMERY["t256s"], (40.0, 320.0, person.z + 60.0), 0.0, rng=rng
    )
    assert [s for s in W.plamy(v) if np.hypot(*(s.gdzie[:2] - (40.0, 300.0))) < 1.5]


def test_a_boulder_looked_at_from_low_down_is_one_warm_spot(dolina):
    """The ground's own temperature is taken over more than the boulder: it stays a spot."""
    t, _, temp = dolina
    hot = float(t.pole(temp, 0.0, 250.0)) + 15.0
    part = O._elip((0, 0, 0.5), (1.2, 1.0, 0.8), (0.4, 0.4, 0.4), hot)
    rock = O.Obiekt("glaz", 0.0, 250.0, float(t.wysokosc(0.0, 250.0)), 0.3, [part])
    v = K.zdjecie(t, temp, [rock], KAMERY["t256s"], (0.0, 250.0, rock.z + 15.0), 0.0)
    spots = [s for s in W.plamy(v) if np.hypot(*(s.gdzie[:2] - (0.0, 250.0))) < 2.0]
    assert len(spots) == 1 and spots[0].cechy[2] > 0.7 * np.pi * 1.2 * 1.0


def test_as_many_of_each_kind_as_the_table_says():
    rng = np.random.default_rng(0)
    for kind, (least, most) in O.SKLAD.items():
        assert {O._ile(rng, kind) for _ in range(400)} == set(range(least, most + 1))


def test_ground_seen_with_too_big_a_pixel_is_not_searched(dolina):
    t, _, temp = dolina
    o = (0.0, 300.0, float(t.wysokosc(0.0, 300.0)) + 60.0)
    for camera, searched in (("t256s", True), ("mlx90640_55", False)):
        seen = np.zeros((100, 100), bool)
        LOT.przeszukane(K.zdjecie(t, temp, [], KAMERY[camera], o, 0.0), seen)
        assert seen.any() == searched


def test_nothing_lifts_the_plane_over_a_slope(dolina):
    """Flown along a contour lane it holds its height; flown at the valley's wall it climbs
    no faster than it can -- it circles to climb -- and does not meet the ground."""
    t, _, _ = dolina
    plane = PLATFORMY["samolot"]
    lane = max(LOT.poziomice(plane, KAMERY["t256s"], 60.0, t), key=len)
    kurs = np.arctan2(lane[1][1] - lane[0][1], lane[1][0] - lane[0][0])
    lot = LOT.Lot(plane, t, lane[0], 60.0, kurs=kurs)
    route, heights, zs = LOT._Trasa([lane]), [], []
    for _ in range(3000):
        route.krok(lot, 0.2, 60.0, wzdluz=True)
        heights.append(lot.nad_ziemia)
        zs.append(lot.z)
        if route.koniec:
            break
    cruise = np.array(heights[150:])
    assert not lot.rozbity and lot.za_nisko < 2.0
    assert abs(np.median(cruise) - 60.0) < 8.0 and np.percentile(cruise, 95) < 100.0
    assert np.diff(zs).max() <= LOT.WZNOSZENIE["samolot"] * 0.2 + 1e-6
    # At the wall: from over the floor, straight at a point far up the side.
    lot = LOT.Lot(plane, t, (0.0, 100.0), 60.0, kurs=0.0)
    lot.z, lot.wystartowal = float(t.wysokosc(0.0, 100.0)) + 60.0, True
    zs, circled = [lot.z], False
    for _ in range(1500):
        left = lot.krok(0.2, (520.0, 100.0), 60.0)
        zs.append(lot.z)
        circled |= bool(lot.krazy)
        if left < 45.0:
            break
    assert circled and not lot.rozbity and left < 45.0
    assert np.diff(zs).max() <= LOT.WZNOSZENIE["samolot"] * 0.2 + 1e-6


class _Cieple:
    """A stand-in brain: it likes what is clearly warmer than the ground round it."""

    def ocena(self, plama):
        return 0.5 if plama.cechy[1] > 3.0 else -0.5


def _lezacy(t, x, y):
    rng = np.random.default_rng(1)
    person = O.Obiekt("czlowiek_lezy", x, y, 0.0, 0.4, O.czlowiek("lezy", -2.0, rng))
    person.z = float(t.wysokosc(x, y))
    return person


@pytest.mark.parametrize("name", ["quad", "samolot", "vtol"])
def test_a_search_finds_someone_on_its_lane_and_looks_from_close(dolina, name):
    """The whole of it: lanes, a candidate liked twice, the closer look from as low as it is
    meant, a report -- and nothing judged that was not looked at."""
    t, _, temp = dolina
    plat, cam = PLATFORMY[name], KAMERY["lepton35"]
    agl = 40.0 if name == "quad" else 50.0
    # Where it flies a minute in, searching: flown once over an empty valley.
    probe = LOT.szukaj(
        plat, cam, t, temp, [], _Cieple(), agl=agl, seed=0, krok=lambda now, *a: now > 61.0
    )
    _, x, y, _, _, phase = probe.slad[-1]
    assert phase == "pasy"
    person = _lezacy(t, x + 4.0, y)

    def enough(now, lot, phase, target, candidates):
        return now > 500.0 or any(c.stan != "czeka" for c in candidates)

    w = LOT.szukaj(plat, cam, t, temp, [person], _Cieple(), agl=agl, seed=0, krok=enough)
    r = LOT.ocen(w, [person])
    assert r["znalezieni"] == 1 and r["falszywe"] == 0 and not r["rozbicie"]
    found = [c for c in w.kandydaci if c.stan == "potwierdzony"][0]
    assert found.razy >= LOT.RAZY and found.oceny
    looking = "vtol_zawis" if name == "vtol" else name  # the tiltrotor looks in a hover
    assert max(found.wysokosci) <= LOT.NIZEJ[name] + LOT.LUZ[looking] + 0.5
    if name == "vtol":
        # Up as a quadcopter, on as a plane, down to look as a quadcopter again.
        assert w.przejscia >= 2 and w.energia > w.czas  # a hover costs more than cruising
    assert not [c for c in w.kandydaci if c.stan == "odrzucony" and not c.oceny]
    climbs = np.diff([s[3] for s in w.slad])
    assert climbs.max() <= LOT.WZNOSZENIE[name] * 0.2 + 1e-6


def test_lessons_kept_from_another_version_are_flown_again(tmp_path, monkeypatch):
    from wingloop.dronomucha.teren import lekcje as L

    monkeypatch.setattr(L, "DATA", tmp_path)
    cam = KAMERY["mlx90640_55"]
    said = []
    L.lekcje("proba", cam, ujecia=2, doliny=(1,), progress=said.append)
    stamp = (tmp_path / "teren_proba.npz").stat().st_mtime_ns
    L.lekcje("proba", cam, ujecia=2, doliny=(1,), progress=said.append)
    assert (tmp_path / "teren_proba.npz").stat().st_mtime_ns == stamp and not said
    monkeypatch.setattr(L, "WERSJA", L.WERSJA + 1)
    L.lekcje("proba", cam, ujecia=2, doliny=(1,), progress=said.append)
    assert (tmp_path / "teren_proba.npz").stat().st_mtime_ns != stamp and said


def test_the_plane_flies_the_contours_and_the_ground_below_stays_level(dolina):
    """Along a contour lane the ground under the plane rises and falls far less than along a
    north-south lane; the lanes lie over the whole square."""
    t, _, _ = dolina
    plane, cam = PLATFORMY["samolot"], KAMERY["t256s"]

    def spread(lanes):
        out = []
        for lane in lanes:
            p = np.asarray(lane)
            x, y, _ = LOT._po_drodze(p[0], p[1:], 1e9, co=10.0)
            out.append(np.ptp(t.wysokosc(x, y)) / max(np.hypot(*np.diff(p, axis=0).T).sum(), 1))
        return np.median(out)

    contours = LOT.poziomice(plane, cam, 60.0, t)
    assert spread(contours) < 0.25 * spread(LOT.pasy(plane, cam, 60.0))
    pts = np.concatenate([np.asarray(lane) for lane in contours])
    assert pts[:, 0].min() < -450 and pts[:, 0].max() > 450
    assert pts[:, 1].min() < -450 and pts[:, 1].max() > 450


def test_the_tiltrotor_hovers_at_four_times_the_power_it_cruises_at():
    """The arithmetic of docs/osprey.md: thin mountain air costs a hover more."""
    from wingloop.dronomucha.platformy import VTOL, gestosc, moc

    disks = 2 * np.pi * 0.1905**2 + np.pi * 0.1525**2
    high = moc(3.0, disks, 17.0, 10.0, gestosc(1800, 0))
    low = moc(3.0, disks, 17.0, 10.0, gestosc(0, 15))
    assert 3.5 < high["stosunek"] < 4.5 and high["zawis"] > 1.05 * low["zawis"]
    assert VTOL.zawis_koszt == pytest.approx(round(high["stosunek"]))
