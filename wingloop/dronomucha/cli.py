"""dronomucha: the drone's jobs from the command line.

dronomucha lista                         the jobs it knows
dronomucha lekcje ZADANIE                teach it a job (once; kept on disk)
dronomucha szukaj ZADANIE [--obszary N]  search N areas, and the tally
dronomucha film ZADANIE --plik X.mp4     record one search
dronomucha rozkaz "Szukaj rozbitka na morzu, pytaj mnie"   an order in words

The functions people add to the fly's brain (docs/hybryda.md), for szukaj
and film: --zestaw mucha|hybryda|hybryda_z_operatorem, or each on its own:
--zmysly N, --kora W, --mapa, --pamiec, --operator.
"""

from __future__ import annotations

import argparse
import json


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="dronomucha", description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("lista", help="the jobs it knows")
    s = sub.add_parser("lekcje", help="teach it a job (kept on disk)")
    s.add_argument("zadanie")
    s.add_argument("--od-nowa", action="store_true", help="fly the lessons again")
    s = sub.add_parser("szukaj", help="search areas it has never seen")
    s.add_argument("zadanie")
    s.add_argument("--seed", type=int, default=7000, help="the first search area")
    s.add_argument("--obszary", type=int, default=6, help="how many areas")
    s.add_argument("--ruch", type=float, default=0.0, help="everything moves, mm/s")
    s.add_argument("--out", default=None, help="write the rows here as JSON")
    s = sub.add_parser("film", help="record one search")
    s.add_argument("zadanie")
    s.add_argument("--plik", default="dronomucha.mp4")
    s.add_argument("--seed", type=int, default=7000, help="which area")
    s.add_argument("--termowizja", choices=("kolor", "bw"), default="bw")
    s.add_argument(
        "--widok",
        choices=("kamera", "termo"),
        default="kamera",
        help="the camera behind the drone, or its thermal camera full screen",
    )
    s = sub.add_parser("rozkaz", help="an order in Polish words: search as it says")
    s.add_argument("tekst")
    s.add_argument("--seed", type=int, default=7000, help="the first search area")
    s.add_argument("--obszary", type=int, default=1, help="how many areas")
    for s in (sub.choices["szukaj"], sub.choices["film"]):
        g = s.add_argument_group("dodatki do mózgu muchy")
        g.add_argument("--zestaw", default="mucha", help="a named set: mucha, hybryda, ...")
        g.add_argument("--zmysly", type=int, default=None, help="senses in the calyx, how much")
        g.add_argument("--kora", type=float, default=None, help="the cortex's say")
        g.add_argument("--mapa", action="store_true", help="search by a map of probability")
        g.add_argument("--pamiec", action="store_true", help="remember objects, not places")
        g.add_argument("--operator", action="store_true", help="ask a person when unsure")
    args = p.parse_args(argv)

    from . import ZADANIA, film, lekcje, oko, szukaj, zadanie

    if args.command == "lista":
        zadanie("ratunek_gory")  # loads the shipped jobs
        for name, z in sorted(ZADANIA.items()):
            others = ", ".join(v["pl"] for k, v in z.kinds.items() if k != z.target)
            print(f"{name:16s} {z.pl}: szuka {z.kinds[z.target]['pl']}, mylące: {others}")
        return
    if args.command == "rozkaz":
        from .rozkaz import rozkaz

        try:
            z, dodatki, understood = rozkaz(args.tekst)
        except ValueError as e:
            raise SystemExit(str(e)) from None
        print(f"Zrozumiałem: {z.pl}; dodatki: {', '.join(understood) or 'brak'}.")
        rows = szukaj(z, range(args.seed, args.seed + args.obszary), dodatki=dodatki)
        found, wanted = sum(r["found"] for r in rows), sum(r["wanted"] for r in rows)
        wrong = sum(r["wrong"] for r in rows)
        print(f"Znalezione: {found} z {wanted}; fałszywe alarmy: {wrong}.")
        return
    z = zadanie(args.zadanie)
    dodatki = _dodatki(args) if args.command in ("szukaj", "film") else None
    if args.command == "lekcje":
        lekcje(oko(z), z, fresh=args.od_nowa)
    elif args.command == "szukaj":
        seeds = range(args.seed, args.seed + args.obszary)
        rows = szukaj(z, seeds, moving=args.ruch, dodatki=dodatki)
        found, wanted = sum(r["found"] for r in rows), sum(r["wanted"] for r in rows)
        wrong = sum(r["wrong"] for r in rows)
        print(f"Znalezione: {found} z {wanted}; fałszywe alarmy: {wrong}.")
        if args.out:
            with open(args.out, "w") as f:
                json.dump(rows, f, indent=1)
    else:
        if args.widok == "termo":
            from .termo import film_termo

            print(f"film -> {film_termo(z, args.plik, seed=args.seed, dodatki=dodatki)}")
        else:
            out = film(z, args.plik, seed=args.seed, thermal=args.termowizja, dodatki=dodatki)
            print(f"film -> {out}")


def _dodatki(args):
    """The add-ons asked for: a named set, with any of its parts changed."""
    from dataclasses import replace

    from .dodatki import ZESTAWY

    if args.zestaw not in ZESTAWY:
        raise SystemExit(f"Nie znam zestawu {args.zestaw!r}. Znane: {', '.join(ZESTAWY)}")
    d = ZESTAWY[args.zestaw]
    if args.zmysly is not None:
        d = replace(d, zmysly=args.zmysly)
    if args.kora is not None:
        d = replace(d, kora=args.kora)
    if args.mapa:
        d = replace(d, mapa={})
    if args.pamiec:
        d = replace(d, pamiec=True)
    if args.operator:
        d = replace(d, operator={})
    return d


if __name__ == "__main__":
    main()
