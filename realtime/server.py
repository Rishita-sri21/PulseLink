"""PulseLink Live: a zero-dependency web dashboard for the real-time engine.

    python main.py demo                 # then open http://localhost:8050
    python main.py demo --port 9000 --speed 2

The server (Python standard library only) runs the :class:`StreamEngine`
in a background thread at real time (or faster) and pushes every frame to
the browser with Server-Sent Events. The browser sends control changes
(policy, packet loss, bandwidth, injected events, ...) back as JSON.
"""
from __future__ import annotations

import json
import logging
import queue
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

from config import ChannelConfig
from realtime.detectors import default_detectors
from realtime.engine import POLICIES, EngineConfig, StreamEngine
from realtime.sources import LiveSynthSource, demo_tour, local_chbmit_files

log = logging.getLogger(__name__)
STATIC = Path(__file__).resolve().parent / "static"


def _r(a, nd=1):
    return np.round(np.asarray(a, dtype=float), nd).tolist()


def _clean(o):
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else round(float(o), 4)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


class LiveApp:
    def __init__(self, model=None, speed: float = 1.0, extra_detectors=None, edf: str | None = None, edf_channels=None):
        self.model = model
        self.speed = speed
        self.extra = extra_detectors or []
        self.edf, self.edf_channels = edf, edf_channels
        self.lock = threading.Lock()
        self.clients: list[queue.Queue] = []
        self.paused = False
        self.policy_cfg = EngineConfig()
        self.channel = ChannelConfig(bandwidth_bps=16_000.0)
        self.source_name = "tour"
        self.engine = self._build(self.source_name)
        self._stop = threading.Event()

    # -- sources ------------------------------------------------------------- #
    def sources(self) -> list[dict]:
        out = [
            {"id": "tour", "title": "Demo tour (synthetic, every use case in 4 min)"},
            {"id": "free", "title": "Synthetic, calm (use the buttons to trigger events)"},
        ]
        try:
            from preprocessing.load_eeg import find_local_recordings

            for i, (p, _) in enumerate(find_local_recordings()):
                out.append({"id": f"sleep:{i}", "title": f"Sleep-EDF {p.name}"})
        except Exception:  # noqa: BLE001
            pass
        for p, sz in local_chbmit_files():
            if sz:
                out.append({"id": f"chb:{p}", "title": f"CHB-MIT {p.name} ({len(sz)} seizure{'s' * (len(sz) > 1)})"})
        if self.edf:
            out.append({"id": f"edf:{self.edf}", "title": f"EDF {Path(self.edf).name}"})
        return out

    def _make_source(self, sid: str):
        if sid == "tour":
            return LiveSynthSource(seed=int(time.time()) % 1000, schedule=demo_tour())
        if sid == "free":
            return LiveSynthSource(seed=int(time.time()) % 1000)
        if sid.startswith("sleep:"):
            from realtime.sources import sleep_edf_source

            return sleep_edf_source(int(sid.split(":")[1]))
        if sid.startswith("chb:"):
            from realtime.sources import chbmit_source

            path = Path(sid[4:])
            sz = dict((str(p), s) for p, s in local_chbmit_files())[str(path)]
            return chbmit_source(path, sz, lead_s=90.0)
        if sid.startswith("edf:"):
            from realtime.sources import RecordingSource, load_edf

            data, picks = load_edf(Path(sid[4:]), self.edf_channels)
            return RecordingSource(data, 100.0, name=Path(sid[4:]).name, channels=picks)
        raise ValueError(sid)

    def _build(self, sid: str) -> StreamEngine:
        src = self._make_source(sid)
        names = ("seizure", "drowsiness", "attention", "artifact")
        dets = default_detectors(names) + [d.clone() for d in self.extra]
        return StreamEngine(src, self.model, self.channel, cfg=EngineConfig(**self.policy_cfg.__dict__), detectors=dets)

    def state(self) -> dict:
        e = self.engine
        return _clean({
            "sources": self.sources(),
            "source": self.source_name,
            "policies": list(POLICIES) if self.model is not None else [p for p in POLICIES if p != "pulselink"],
            "policy": e.cfg.policy,
            "channel": {k: getattr(e.channel_cfg, k) for k in ("bandwidth_bps", "loss_rate", "burst_length", "max_retries")},
            "alert_copies": e.cfg.alert_copies,
            "speed": self.speed,
            "paused": self.paused,
            "fs": e.fs,
            "frame_s": e.cfg.frame_s,
            "channels": getattr(e.source, "channels", [f"ch{i}" for i in range(e.c)]),
            "detectors": [d.info() for d in e.device.detectors],
            "can_inject": hasattr(e.source, "schedule"),
        })

    # -- control ------------------------------------------------------------- #
    def control(self, msg: dict) -> dict:
        a = msg.get("action")
        with self.lock:
            e = self.engine
            if a == "set":
                if "policy" in msg:
                    e.cfg.policy = self.policy_cfg.policy = msg["policy"]
                if "alert_copies" in msg:
                    e.cfg.alert_copies = self.policy_cfg.alert_copies = int(msg["alert_copies"])
                if "speed" in msg:
                    self.speed = float(msg["speed"])
                ch = {k: float(msg[k]) for k in ("bandwidth_bps", "loss_rate", "burst_length") if k in msg}
                if "max_retries" in msg:
                    ch["max_retries"] = int(msg["max_retries"])
                if ch:
                    e.set_channel(**ch)
                    for k, v in ch.items():
                        setattr(self.channel, k, v)
            elif a == "inject":
                e.source.inject(msg["kind"], msg.get("duration"))
            elif a == "outage":
                e.outage_now(float(msg.get("seconds", 10)))
            elif a == "pause":
                self.paused = True
            elif a == "resume":
                self.paused = False
            elif a in ("reset", "source"):
                if a == "source":
                    self.source_name = msg["source"]
                self.engine = self._build(self.source_name)
                self._broadcast({"type": "reset", "state": self.state()})
        return self.state()

    # -- streaming ----------------------------------------------------------- #
    def _broadcast(self, payload: dict) -> None:
        data = json.dumps(_clean(payload))
        for q in list(self.clients):
            try:
                q.put_nowait(data)
            except queue.Full:
                pass

    def loop(self) -> None:
        while not self._stop.is_set():
            t0 = time.time()
            if not self.paused:
                with self.lock:
                    e = self.engine
                    try:
                        rec = e.step()
                    except StopIteration:
                        self.paused = True
                        continue
                    fid = rec["frame"]
                    payload = {
                        "type": "frame",
                        **{k: rec[k] for k in ("frame", "t", "mode", "codec", "bytes", "cr", "status", "latency_s", "prd", "truth", "label", "device", "receiver", "alerts", "backfilled", "packets", "attempts")},
                        "orig": _r(e.orig[fid]),
                        "recon": _r(e.recon[fid]),
                        "backfill": {str(b): _r(e.recon[b]) for b in rec["backfilled"]},
                        "summary": e.summary(),
                        "events": [ev.__dict__ for ev in e.events[-12:]],
                    }
                    for k in [k for k in e.orig if k < fid - 20]:  # free memory
                        e.orig.pop(k, None)
                        e.recon.pop(k, None)
                    if len(e.records) > 400:
                        del e.records[:100]
                self._broadcast(payload)
                wait = e.cfg.frame_s / max(self.speed, 0.05)
            else:
                wait = 0.2
            time.sleep(max(0.0, wait - (time.time() - t0)))

    def start(self) -> None:
        threading.Thread(target=self.loop, daemon=True).start()


def make_handler(app: LiveApp):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):  # quiet
            log.debug(fmt, *args)

        def _send(self, code, body: bytes, ctype="application/json"):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                self._send(200, (STATIC / "index.html").read_bytes(), "text/html; charset=utf-8")
            elif self.path == "/api/state":
                self._send(200, json.dumps(app.state()).encode())
            elif self.path == "/api/stream":
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "keep-alive")
                self.end_headers()
                q: queue.Queue = queue.Queue(maxsize=200)
                app.clients.append(q)
                try:
                    while True:
                        try:
                            data = q.get(timeout=15)
                            self.wfile.write(f"data: {data}\n\n".encode())
                        except queue.Empty:
                            self.wfile.write(b": keep-alive\n\n")
                        self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError, OSError):
                    pass
                finally:
                    app.clients.remove(q)
            else:
                self._send(404, b'{"error": "not found"}')

        def do_POST(self):
            if self.path != "/api/control":
                return self._send(404, b'{"error": "not found"}')
            n = int(self.headers.get("Content-Length", 0))
            try:
                msg = json.loads(self.rfile.read(n) or b"{}")
                self._send(200, json.dumps(app.control(msg)).encode())
            except Exception as exc:  # noqa: BLE001
                log.exception("control failed")
                self._send(400, json.dumps({"error": str(exc)}).encode())

    return Handler


def serve(host: str = "127.0.0.1", port: int = 8050, model=None, speed: float = 1.0, extra_detectors=None, edf=None, edf_channels=None, open_browser: bool = True) -> None:
    app = LiveApp(model, speed, extra_detectors, edf, edf_channels)
    app.start()
    httpd = ThreadingHTTPServer((host, port), make_handler(app))
    httpd.daemon_threads = True
    url = f"http://{'localhost' if host in ('127.0.0.1', '0.0.0.0') else host}:{port}"
    print(f"PulseLink Live running at {url}  (Ctrl+C to stop)")
    if open_browser:
        try:
            import webbrowser

            threading.Timer(1.0, lambda: webbrowser.open(url)).start()
        except Exception:  # noqa: BLE001
            pass
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("stopped")
