"""Write a search for the 3-D simulator: the valley, what lies on it, and the flights.

One JSON file (:func:`eksport`): the ground every ``co`` cells (heights,
cover, temperature), the objects with their parts, the trail and the tarn,
and for each platform asked for its lanes and its whole flight -- a row
every step: time, where, which way, how fast, whether it hovers, its
phase and its battery -- with every candidate's story (seen, judged).
Arrays are little-endian float32 or uint8, base64.

    python -m wingloop.dronomucha.sym3d.eksport scena.json --seed 2
"""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path

import numpy as np

from ..platformy import KAMERY, PLATFORMY
from ..teren import lekcje as L
from ..teren import lot as LOT
from ..teren import obiekty as O
from ..teren import teren as T

#: The phases, as numbers in a flight's rows.
FAZY = ("pasy", "potwierdz", "wroc", "powrot")
#: A flight's row: what each number is.
KOLUMNY = ("t", "x", "y", "z", "kurs", "v", "zawis", "krazy", "faza", "bateria", "nad_ziemia")


def _b64(a: np.ndarray, dtype) -> str:
    return base64.b64encode(np.ascontiguousarray(a, dtype=dtype).tobytes()).decode()


def lot(platforma, kamera, teren, gt, obiekty, mozg, agl, seed, powietrze) -> dict:
    """One platform's search, recorded step by step for the simulator."""
    rows, events, seen = [], [], {}

    def krok(t, lt, faza, cel, kandydaci):
        rows.append(
            (
                t,
                lt.x,
                lt.y,
                lt.z,
                lt.kurs,
                lt.v,
                float(lt.zawis),
                float(lt.krazy),
                float(FAZY.index(faza)),
                1.0 - lt.energia / (platforma.lot_min * 60.0),
                lt.nad_ziemia,
            )
        )
        for c in kandydaci:
            state = "do_sprawdzenia" if c.proba else c.stan
            if seen.get(id(c)) != state:
                seen[id(c)] = state
                events.append(
                    dict(
                        t=round(t, 2),
                        id=len({k for k in seen}) - 1 if state == "czeka" else None,
                        key=id(c),
                        x=float(c.gdzie[0]),
                        y=float(c.gdzie[1]),
                        z=float(c.gdzie[2]),
                        stan=state,
                        ocena=round(
                            float(np.mean([o for o in c.oceny if o is not None]))
                            if any(o is not None for o in c.oceny)
                            else float(c.ocena),
                            3,
                        ),
                    )
                )
        return False

    _, lanes = LOT.trasa_lotu(platforma, kamera, agl, teren)
    w = LOT.szukaj(
        platforma,
        kamera,
        teren,
        gt,
        obiekty,
        mozg,
        agl=agl,
        seed=seed,
        krok=krok,
        powietrze=powietrze,
    )
    keys = {}
    for e in events:  # numbered in the order first seen
        e["id"] = keys.setdefault(e.pop("key"), len(keys))
    r = LOT.ocen(w, obiekty)
    return dict(
        nazwa=platforma.nazwa,
        pl=platforma.pl,
        agl=agl,
        predkosc=platforma.predkosc,
        kolumny=KOLUMNY,
        n=len(rows),
        wiersze=_b64(np.array(rows), "<f4"),
        zdarzenia=events,
        pasy=[[[round(float(x), 1), round(float(y), 1)] for x, y in lane] for lane in lanes],
        wynik={
            k: (float(v) if isinstance(v, (int, float, np.floating)) and v is not None else v)
            for k, v in r.items()
        },
    )


def eksport(
    path,
    seed: int = 2,
    platformy=("samolot", "quad", "vtol"),
    kamera: str = "t256s",
    dzien: str = "jesien",
    co: int = 2,
    progress=print,
) -> Path:
    """Fly the platforms over one search area and write all of it to ``path``."""
    t = T.dolina()
    pg = T.DNI[dzien]
    gt = t.temperatura(pg)
    obiekty = O.uloz(t, pg, seed=seed)
    mozg = L.mozg(kamera, KAMERY[kamera])
    flights = {}
    for name in platformy:
        plat = PLATFORMY[name]
        if progress:
            progress(f"  {plat.pl}: lot ...")
        flights[name] = lot(
            plat, KAMERY[kamera], t, gt, obiekty, mozg, plat.wysokosc, seed, pg.powietrze
        )
        if progress:
            r = flights[name]["wynik"]
            progress(
                f"    znalezieni {r['znalezieni']}/{r['ludzie']}, fałszywe {r['falszywe']}"
            )
    z = t.z[::co, ::co]
    cam = KAMERY[kamera]
    out = dict(
        wersja=1,
        teren=dict(
            n=int(z.shape[0]),
            krok=t.krok * co,
            polowa=t.polowa,
            pole=LOT.POLE,
            wysokosci=_b64(z, "<f4"),
            pokrycie=_b64(t.pokrycie[::co, ::co], "u1"),
            temperatura=_b64(gt[::co, ::co], "<f4"),
            nazwy=T.NAZWY,
            baza=list(t.baza),
            staw=list(t.staw),
            szlak=t.szlak.round(1).tolist(),
        ),
        dzien=dict(
            nazwa=dzien,
            powietrze=pg.powietrze,
            slonce_az=pg.slonce_az,
            slonce_el=pg.slonce_el,
            chmury=pg.chmury,
        ),
        kamera=dict(nazwa=cam.nazwa, px=list(cam.px), fov=cam.fov),
        obiekty=[
            dict(
                rodzaj=o.rodzaj,
                x=o.x,
                y=o.y,
                z=o.z,
                yaw=o.yaw,
                czesci=[
                    dict(
                        typ=c.typ,
                        a=list(c.a),
                        b=list(c.b) if c.b is not None else None,
                        r=c.r if isinstance(c.r, float) else list(c.r),
                        kolor=list(c.kolor),
                        temp=c.temp,
                        snieg=c.pod_sniegiem,
                    )
                    for c in o.czesci
                ],
            )
            for o in obiekty
        ],
        loty=flights,
    )
    path = Path(path)
    path.write_text(json.dumps(out, separators=(",", ":")))
    if progress:
        progress(f"-> {path} ({path.stat().st_size // 1024} kB)")
    return path


def main(argv=None) -> None:
    a = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    a.add_argument("plik")
    a.add_argument("--seed", type=int, default=2)
    a.add_argument("--kamera", default="t256s")
    a.add_argument("--dzien", default="jesien")
    a.add_argument("--platformy", default="samolot,quad,vtol")
    args = a.parse_args(argv)
    eksport(args.plik, args.seed, tuple(args.platformy.split(",")), args.kamera, args.dzien)


if __name__ == "__main__":
    main()
