"""``flylab`` -- the fly's flight, brain, vision, learning, and a drone, from one command.

    flylab fly     --seconds 0.5 --height 5 [--target X Y]   the physical fly
    flylab see     [--samples 40]                            object recognition
    flylab teach   --reward bar=1 sphere=-1 [--sessions 16]  training flights
    flylab drone   --memory memory.npz [--object bar@120:5]  autonomous mission
    flylab demo                                              all of it, one report
    flylab check                                             is everything installed?

``python -m wingloop.lab ...`` does the same where ``flylab`` is not on PATH.

Every command writes its results as JSON next to ``--out`` and ``flylab
report`` (or ``demo``) turns them into one self-contained HTML page.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

FIXTURES = Path(__file__).resolve().parents[2] / "tests"


def _parse_rewards(items) -> dict:
    out = {}
    for item in items:
        kind, value = item.split("=")
        out[kind] = float(value)
    return out


def _parse_objects(items):
    from .drone import place

    things = []
    for item in items:
        kind, rest = item.split("@")
        heading, distance = (float(v) for v in rest.split(":"))
        things.append(place(kind, heading, distance))
    return things


def cmd_fly(args) -> dict:
    from .flysim import fly

    t0 = time.time()
    run = fly(
        args.seconds, height=args.height, target=tuple(args.target) if args.target else None
    )
    print(
        f"flew {run['t'][-1] * 1000:.0f} ms in {time.time() - t0:.0f} s; final height "
        f"{run['z'][-1]:.2f} mm, heading {run['heading'][-1]:+.1f} deg"
    )
    return {"fly": run}


def cmd_see(args) -> dict:
    from .recognition import evaluate, make_dataset

    if args.samples:
        from .vision import FlyEye

        X, R, y = make_dataset(FlyEye(), n_per_class=args.samples, seed=args.seed)
    else:
        d = np.load(FIXTURES / "visual_features.npz")
        X, R, y = d["X"], d["R"], d["y"]
    result = evaluate(X, R, y)
    for k, v in result.items():
        print(
            f"{k:15s} {v if isinstance(v, float) else f'{100 * v[0]:.0f}% +- {100 * v[1]:.0f}'}"
        )
    return {"recognition": result}


def cmd_teach(args) -> dict:
    from .drone import teach, training_flights
    from .learning import MushroomBody, save
    from .vision import FlyEye

    reward = _parse_rewards(args.reward)
    kinds = tuple(reward)
    eye = FlyEye()
    t0 = time.time()
    F, K = training_flights(eye, kinds, sessions=args.sessions, seed=args.seed)
    mb = MushroomBody(F.shape[1], seed=args.seed)
    teach(mb, F, K, reward)
    save(mb, args.memory)
    valence = {
        k: float(np.mean([mb.valence(f) for f in F[K == k]])) for k in kinds if (K == k).any()
    }
    print(
        f"{len(K)} fixations in {time.time() - t0:.0f} s; learned valence:",
        {k: round(v, 2) for k, v in valence.items()},
        f"-> {args.memory}",
    )
    return {"learning": {"reward": reward, "valence": valence, "fixations": int(len(K))}}


def cmd_drone(args) -> dict:
    from .drone import Mission
    from .learning import load
    from .scene import World
    from .vision import FlyEye

    world = World(_parse_objects(args.object))
    mb = load(args.memory) if args.memory else None
    eye = FlyEye()
    mission = Mission(world, eye, mb)
    t0 = time.time()
    log = mission.run(args.seconds)
    print(
        f"phase {mission.phase} after {mission.t:.1f} s ({time.time() - t0:.0f} s wall); "
        "distances",
        {k: round(v, 2) for k, v in mission.distances().items()},
    )
    from .drone import which

    start = [log[0]["x"], log[0]["y"]] if log else [0.0, 0.0]
    judged = {}
    for m in mission.memories:
        thing = which(world, start, m.heading)
        if thing is not None:
            judged[thing.label] = m.valence
    world_out = [
        dict(
            label=t.label,
            kind=t.kind,
            position=t.position.tolist(),
            size=t.size,
            valence=judged.get(t.label, 0.0),
        )
        for t in world.things
    ]
    return {
        "drone": {
            "log": log,
            "phase": mission.phase,
            "t": mission.t,
            "distances": mission.distances(),
            "world": world_out,
            "memories": [dict(heading=m.heading, valence=m.valence) for m in mission.memories],
        }
    }


def cmd_check(args) -> dict:
    """Say what is installed, what is missing, and how to get it."""
    import importlib.util
    import sys

    ok = True

    def line(good: bool, what: str, fix: str = "") -> None:
        nonlocal ok
        ok &= good
        print(
            ("  ok   " if good else "  MISSING ")
            + what
            + ("" if good else f"\n         -> {fix}")
        )

    v = sys.version_info
    line(
        (3, 10) <= v[:2] <= (3, 12),
        f"Python {v[0]}.{v[1]} (flyvis and flygym need 3.10-3.12)",
        "install Python 3.12 and make a virtual environment with it (see README, 'Windows')",
    )
    for module, package in (
        ("numpy", "numpy"),
        ("mujoco", "mujoco"),
        ("flygym_gymnasium", "flygym-gymnasium"),
        ("torch", "torch"),
        ("flyvis", "flyvis"),
        ("sklearn", "scikit-learn"),
        ("matplotlib", "matplotlib"),
        ("googleapiclient", "google-api-python-client"),
    ):
        line(importlib.util.find_spec(module) is not None, module, f'pip install "{package}"')
    if importlib.util.find_spec("flyvis") is not None:
        try:
            import flyvis

            have = (flyvis.results_dir / "flow/0000/000").exists()
        except Exception:  # pragma: no cover - a broken install
            have = False
        line(
            have,
            "flyvis pretrained models",
            "python -m flyvis_cli.download_pretrained_models --skip_large_files",
        )
    line(
        (FIXTURES / "rwing_vertices.npy").exists(),
        "wing mesh and connectome fixtures",
        "install from a clone of the repository: pip install -e .[body,lab]",
    )
    print("ready" if ok else "not ready: fix the lines marked MISSING, then run this again")
    return {}


def cmd_report(args) -> dict:
    from .report import build

    runs = {}
    for path in args.runs:
        runs.update(json.loads(Path(path).read_text()))
    print("report ->", build(runs, args.html))
    return {}


def cmd_demo(args) -> dict:
    from .report import build

    out = Path(args.dir)
    out.mkdir(parents=True, exist_ok=True)
    runs = {}
    memory = out / "memory.npz"
    steps = [
        ("fly", argparse.Namespace(seconds=args.fly_seconds, height=5.0, target=[50.0, 50.0])),
        ("see", argparse.Namespace(samples=0, seed=0)),
        (
            "teach",
            argparse.Namespace(
                reward=["bar=1", "sphere=-1"], sessions=args.sessions, seed=0, memory=memory
            ),
        ),
        (
            "drone",
            argparse.Namespace(
                object=["bar@120:5", "sphere@-100:5"], memory=memory, seconds=60.0
            ),
        ),
    ]
    for name, ns in steps:
        print(f"== {name}")
        result = COMMANDS[name][0](ns)
        (out / f"{name}.json").write_text(json.dumps(result, default=float))
        runs.update(result)
    print("report ->", build(runs, out / "flylab.html"))
    return runs


COMMANDS = {
    "fly": (cmd_fly, "fly the physical fly"),
    "see": (cmd_see, "object recognition from the optic lobe"),
    "teach": (cmd_teach, "training flights with dopamine"),
    "drone": (cmd_drone, "autonomous drone mission"),
    "report": (cmd_report, "build the HTML report"),
    "check": (cmd_check, "is everything installed?"),
    "demo": (cmd_demo, "everything, one report"),
}


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="flylab", description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="command", required=True)
    for name, (_, help_) in COMMANDS.items():
        s = sub.add_parser(name, help=help_)
        s.add_argument("--out", default=None, help="write the results here as JSON")
        if name == "fly":
            s.add_argument("--seconds", type=float, default=0.5)
            s.add_argument("--height", type=float, default=5.0, help="mm")
            s.add_argument("--target", type=float, nargs=2, default=None, help="x y, mm")
        if name == "see":
            s.add_argument(
                "--samples",
                type=int,
                default=0,
                help="new flights per class (0: use the recorded set)",
            )
            s.add_argument("--seed", type=int, default=0)
        if name == "teach":
            s.add_argument("--reward", nargs="+", default=["bar=1", "sphere=-1"])
            s.add_argument("--sessions", type=int, default=16)
            s.add_argument("--seed", type=int, default=0)
            s.add_argument("--memory", default="memory.npz")
        if name == "drone":
            s.add_argument(
                "--object",
                nargs="+",
                default=["bar@120:5", "sphere@-100:5"],
                help="kind@heading_deg:distance_m",
            )
            s.add_argument("--memory", default=None)
            s.add_argument("--seconds", type=float, default=60.0)
        if name == "report":
            s.add_argument("runs", nargs="+")
            s.add_argument("--html", default="flylab.html")
        if name == "demo":
            s.add_argument("--dir", default="flylab-run")
            s.add_argument("--sessions", type=int, default=16)
            s.add_argument("--fly-seconds", type=float, default=0.3)
    args = p.parse_args(argv)
    result = COMMANDS[args.command][0](args)
    if getattr(args, "out", None) and result:
        Path(args.out).write_text(json.dumps(result, default=float))


if __name__ == "__main__":
    main()
