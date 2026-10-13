"""The brain panel as a page in the browser, updated live from the flight.

The simulation keeps one snapshot of the nervous system up to date; a small
local web server hands it to the page as JSON a few times a second, and the
page draws it. Nothing leaves the computer: the server listens on 127.0.0.1
only.

A browser was chosen over a plotting window because it is legible -- real
type, real layout, a light and a dark theme -- and because it costs the
simulation nothing while it is not being redrawn.
"""

from __future__ import annotations

import base64
import json
import threading
import webbrowser
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

from .brainpanel import signals

PAGE = Path(__file__).with_name("livebrain.html")
#: Seconds of simulated flight the traces keep.
HISTORY = 2.0
MAPS = {"t4": ("T4a", "T4b"), "t5": ("T5a", "T5b"), "mi1": ("Mi1",), "tm9": ("Tm9",)}


def _round(a, digits: int = 3) -> list:
    return [round(float(v), digits) for v in np.asarray(a).ravel()]


class LiveBrain:
    """Holds the latest snapshot and serves it, with the page, on localhost."""

    def __init__(self, eye=None, port: int = 0):
        self.eye = eye
        self.lock = threading.Lock()
        self.snapshot: dict = {"ready": False}
        self.history = {k: deque() for k in ("t", "drive", "power", "yaw", "speed")}
        self.layout = self._layout()
        handler = self._handler()
        self.server = ThreadingHTTPServer(("127.0.0.1", port), handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}/"
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def _layout(self) -> dict:
        if self.eye is None:
            return {"eye": False}
        e = self.eye
        out = {"eye": True, "columns": {"x": _round(e.column_x), "y": _round(e.column_y)}}
        for key, types in MAPS.items():
            idx = e._by_type[types[0]]
            out[key] = {"x": _round(e.x[idx]), "y": _round(e.y[idx])}
        return out

    def _handler(self):
        brain = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):  # keep the console for the flight
                pass

            def _send(self, body: bytes, kind: str) -> None:
                self.send_response(200)
                self.send_header("Content-Type", kind)
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):  # noqa: N802 - the http.server name
                if self.path.startswith("/state.json"):
                    with brain.lock:
                        body = json.dumps(brain.snapshot).encode()
                    self._send(body, "application/json")
                elif self.path.startswith("/layout.json"):
                    self._send(json.dumps(brain.layout).encode(), "application/json")
                elif self.path in ("/", "/index.html"):
                    self._send(PAGE.read_bytes(), "text/html; charset=utf-8")
                else:
                    self.send_error(404)

        return Handler

    def open(self) -> None:
        webbrowser.open(self.url)

    def update(self, state, knobs, rates, slowdown: float, image=None, activity=None) -> None:
        """Take a new snapshot. Cheap: no drawing happens here."""
        values = signals(state, knobs, rates)
        h = self.history
        t = state["t"]
        for key, value in (
            ("t", round(t * 1000, 1)),
            ("drive", values["DNa08/DNp31 drive"]),
            ("power", state["neurons"]["power MNs (DLMn, DVMn)"]),
            ("yaw", values["wing: rotation phase L-R"]),
            ("speed", state["speed"]),
        ):
            h[key].append(round(float(value), 4))
        while h["t"] and h["t"][0] < (t - HISTORY) * 1000:
            for q in h.values():
                q.popleft()
        snap = {
            "ready": True,
            "t": round(t * 1000, 1),
            "speed": round(state["speed"], 1),
            "z": round(state["z"], 2),
            "heading": round(state["heading"], 1),
            "pitch": round(state["pitch"], 1),
            "roll": round(state["roll"], 1),
            "slowdown": round(slowdown, 1),
            "bars": {k: round(v, 3) for k, v in values.items()},
            "raw": {
                "rates": _round(rates, 2),
                "drive": round(state["neurons"]["DNa08/DNp31 drive"], 3),
                "power": round(state["neurons"]["power MNs (DLMn, DVMn)"], 3),
                "steering": round(state["neurons"]["steering MNs"], 4),
            },
            "history": {k: list(v) for k, v in h.items()},
            "eye": None,
        }
        if image is not None and activity is not None and self.eye is not None:
            img = (np.clip(np.asarray(image), 0, 1) * 255).astype(np.uint8)
            e = self.eye
            base = e.baseline if e.baseline is not None else activity
            maps = {}
            for key, types in MAPS.items():
                if len(types) == 2:
                    a = e.type_activity(activity, types[0]) - e.type_activity(
                        activity, types[1]
                    )
                else:
                    a = e.type_activity(activity, types[0]) - e.type_activity(base, types[0])
                scale = max(float(np.abs(a).max()), 1e-3)
                maps[key] = _round(np.clip(a / scale, -1, 1), 2)
            snap["eye"] = {
                "size": int(img.shape[0]),
                "image": base64.b64encode(img.tobytes()).decode(),
                "photoreceptors": _round(e.photoreceptors, 2),
                "maps": maps,
                "neurons": int(e.n_neurons),
            }
        with self.lock:
            self.snapshot = snap

    def close(self) -> None:
        self.server.shutdown()
