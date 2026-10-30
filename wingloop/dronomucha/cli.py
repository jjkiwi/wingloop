"""dronomucha: the drone's jobs from the command line.

dronomucha lista                         the jobs it knows
dronomucha lekcje ZADANIE                teach it a job (once; kept on disk)
dronomucha szukaj ZADANIE [--obszary N]  search N areas, and the tally
dronomucha film ZADANIE --plik X.mp4     record one search
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
    args = p.parse_args(argv)

    from . import ZADANIA, film, lekcje, oko, szukaj, zadanie

    if args.command == "lista":
        zadanie("ratunek_gory")  # loads the shipped jobs
        for name, z in sorted(ZADANIA.items()):
            others = ", ".join(v["pl"] for k, v in z.kinds.items() if k != z.target)
            print(f"{name:16s} {z.pl}: szuka {z.kinds[z.target]['pl']}, mylące: {others}")
        return
    z = zadanie(args.zadanie)
    if args.command == "lekcje":
        lekcje(oko(z), z, fresh=args.od_nowa)
    elif args.command == "szukaj":
        rows = szukaj(z, range(args.seed, args.seed + args.obszary), moving=args.ruch)
        found, wanted = sum(r["found"] for r in rows), sum(r["wanted"] for r in rows)
        wrong = sum(r["wrong"] for r in rows)
        print(f"Znalezione: {found} z {wanted}; fałszywe alarmy: {wrong}.")
        if args.out:
            with open(args.out, "w") as f:
                json.dump(rows, f, indent=1)
    else:
        print(f"film -> {film(z, args.plik, seed=args.seed, thermal=args.termowizja)}")


if __name__ == "__main__":
    main()
