"""One HTML page with everything a run produced: figures, numbers, and what they mean.

Self-contained -- figures are embedded as PNG -- so it opens anywhere and can
be sent to someone as a single file.
"""

from __future__ import annotations

import base64
import html
import io
import json
from pathlib import Path

import numpy as np


def _png(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    import matplotlib.pyplot as plt

    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def _figure(width=7.0, height=3.0, n=1):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, n, figsize=(width, height))
    return fig, np.atleast_1d(axes)


def fly_figures(run: dict) -> list[tuple[str, str]]:
    t = np.asarray(run["t"]) * 1000
    out = []
    fig, (a, b) = _figure(9, 3, 2)
    a.plot(t, run["z"], color="#2a6f97")
    a.set(xlabel="time (ms)", ylabel="height (mm)", title="Height, throttle loop")
    b.plot(t, np.degrees(run["pitch"]), label="pitch")
    b.plot(t, np.degrees(run["roll"]), label="roll")
    b.plot(t, run["heading"], label="heading")
    b.set(xlabel="time (ms)", ylabel="degrees", title="Attitude and heading")
    b.legend(frameon=False)
    out.append(("The fly in flight", _png(fig)))
    fig, axes = _figure(9, 3, 1)
    names = list(run["neurons"][0].keys())
    for name in names:
        axes[0].plot(t, [n[name] for n in run["neurons"]], label=name)
    axes[0].set(
        xlabel="time (ms)", ylabel="activation", title="Flight neurons (connectome readout)"
    )
    axes[0].legend(frameon=False, fontsize=8, ncol=2)
    out.append(("Its flight neurons", _png(fig)))
    return out


def drone_figures(run: dict) -> list[tuple[str, str]]:
    log = run["log"]
    x, y = [r["x"] for r in log], [r["y"] for r in log]
    fig, (a, b) = _figure(10, 4, 2)
    phases = sorted({r["phase"] for r in log}, key=[r["phase"] for r in log].index)
    for ph in phases:
        idx = [i for i, r in enumerate(log) if r["phase"] == ph]
        a.plot([x[i] for i in idx], [y[i] for i in idx], ".", ms=3, label=ph)
    for thing in run["world"]:
        colour = "#2b9348" if thing["valence"] > 0 else "#c1121f"
        a.add_patch(
            __import__("matplotlib.patches", fromlist=["Circle"]).Circle(
                thing["position"][:2], max(thing["size"], 0.25), color=colour, alpha=0.6
            )
        )
        a.annotate(
            thing["label"],
            thing["position"][:2],
            textcoords="offset points",
            xytext=(6, 6),
            fontsize=8,
        )
    a.set_aspect("equal")
    a.set(xlabel="x (m)", ylabel="y (m)", title="Drone path from above")
    a.legend(frameon=False, fontsize=8)
    t = [r["t"] for r in log]
    b.plot(t, [r["z"] for r in log], label="height (m)")
    b.plot(t, [np.degrees(r["yaw"]) / 100 for r in log], label="heading / 100 deg")
    b.plot(t, [r["valence"] for r in log], label="mushroom-body valence")
    b.set(xlabel="time (s)", title="Height, heading and what it thinks of what it sees")
    b.legend(frameon=False, fontsize=8)
    return [("The drone's mission", _png(fig))]


def recognition_table(result: dict) -> str:
    rows = "".join(
        f"<tr><td>{html.escape(k)}</td><td>{100 * v[0]:.0f}% &plusmn; {100 * v[1]:.0f}</td></tr>"
        for k, v in result.items()
        if isinstance(v, (list, tuple))
    )
    return (
        "<table><tr><th>readout</th><th>accuracy</th></tr>"
        + rows
        + f"<tr><td>chance</td><td>{100 * result['chance']:.0f}%</td></tr></table>"
    )


def build(runs: dict, path) -> Path:
    sections = []
    if "fly" in runs:
        sections.append(("Fly flight simulator", fly_figures(runs["fly"]), ""))
    if "recognition" in runs:
        sections.append(
            (
                "Object recognition from the optic lobe",
                [],
                recognition_table(runs["recognition"]),
            )
        )
    if "learning" in runs:
        lr = runs["learning"]
        table = (
            "<table><tr><th>object</th><th>dopamine</th><th>learned valence</th></tr>"
            + "".join(
                f"<tr><td>{html.escape(k)}</td><td>{lr['reward'].get(k, 0):+.0f}</td><td>{v:+.2f}</td></tr>"
                for k, v in lr["valence"].items()
            )
            + "</table>"
        )
        sections.append(("Learning (mushroom body)", [], table))
    if "drone" in runs:
        d = runs["drone"]
        summary = (
            f"<p>Finished in phase <b>{html.escape(d['phase'])}</b> after {d['t']:.1f} s. "
            "Distances at the end: "
            + ", ".join(f"{html.escape(k)} {v:.2f} m" for k, v in d["distances"].items())
            + ".</p>"
        )
        sections.append(
            ("Autonomous drone on the fly's control laws", drone_figures(d), summary)
        )
    body = ""
    for title, figs, extra in sections:
        body += f"<section><h2>{html.escape(title)}</h2>{extra}"
        for caption, png in figs:
            body += f'<figure><img alt="{html.escape(caption)}" src="data:image/png;base64,{png}"><figcaption>{html.escape(caption)}</figcaption></figure>'
        body += "</section>"
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>flylab run</title>
<style>
:root {{ --bg:#fbfaf7; --fg:#1d1d1b; --muted:#6b6a65; --line:#e3e1da; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg:#16161a; --fg:#ecebe6; --muted:#a3a29c; --line:#2c2c33; }} img {{ background:#fff; }} }}
body {{ background:var(--bg); color:var(--fg); font:16px/1.5 system-ui, sans-serif; margin:0 auto; max-width:1000px; padding:24px 16px; }}
h1 {{ font-size:1.6rem; }} h2 {{ font-size:1.2rem; border-top:1px solid var(--line); padding-top:16px; }}
img {{ max-width:100%; height:auto; border-radius:6px; }} figcaption {{ color:var(--muted); font-size:.9rem; }}
table {{ border-collapse:collapse; }} td, th {{ border-bottom:1px solid var(--line); padding:4px 12px; text-align:left; }}
</style></head><body><h1>flylab</h1>
<p>A fruit fly's flight, vision, learning and a drone flown on its control laws.</p>{body}
<details><summary>Raw numbers</summary><pre>{html.escape(json.dumps({k: v for k, v in runs.items() if k != "drone"}, indent=1, default=float)[:20000])}</pre></details>
</body></html>"""
    path = Path(path)
    path.write_text(page)
    return path
