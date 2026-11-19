"""PulseLink Live: a zero-dependency web dashboard for the real-time engine.

    python main.py demo                 # then open http://localhost:8050
    python main.py demo --profile driver --speed 2

The server (Python standard library only) runs two engines on the *same*
EEG in a background thread: PulseLink, and a typical streaming wearable
(compressed EEG over one queue, detection on the phone). Every frame is
pushed to the browser with Server-Sent Events, so the audience sees, side by
side, when each receiver learns about the event. The browser sends control
changes (link quality, injected events, acknowledgements...) back as JSON.
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

from communication.priority_link import CLASS_NAMES
from config import ChannelConfig
from realtime.engine import POLICIES, EngineConfig, StreamEngine, preset
from realtime.profiles import PROFILES, profile_detectors
from realtime.sources import LiveSynthSource, demo_tour, local_chbmit_files

log = logging.getLogger(__name__)
STATIC = Path(__file__).resolve().parent / "static"
PRE_S, TOL_S = 5.0, 10.0
TRUTH_OF = {"seizure": "seizure", "drowsiness": "drowsiness", "attention": "attention", "artifact": "artifact"}


def _r(a, nd=1):
    return np.round(np.asarray(a, dtype=float), nd).tolist()


def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple, set)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else round(float(o), 4)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


class LiveApp:
    def __init__(self, model=None, speed: float = 1.0, extra_detectors=None, edf: str | None = None, edf_channels=None, profile: str = "general"):
        self.model = model
        self.speed = speed
        self.extra = extra_detectors or []
        self.edf, self.edf_channels = edf, edf_channels
        self.profile = profile if profile in PROFILES else "general"
        self.lock = threading.Lock()
        self.clients: list[queue.Queue] = []
        self.paused = False
        self.policy = "event"
        self.alert_copies = EngineConfig().alert_copies
        self.channel = ChannelConfig(bandwidth_bps=16_000.0)
        self.source_name = "tour"
        self._build_all()
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

    def _build_all(self) -> None:
        src = self._make_source(self.source_name)
        dets = profile_detectors(self.profile, extra=self.extra)
        cfg = preset("pulselink", policy=self.policy, alert_copies=self.alert_copies)
        self.engine = StreamEngine(src, self.model, self.channel, cfg=cfg, detectors=dets)
        self.baseline = StreamEngine(src, self.model, self.channel, cfg=preset("stream"), detectors=[d.clone() for d in dets])
        self._pushed: set[int] = set()

    @property
    def target(self) -> str | None:
        """The profile's critical detector (what the race is about)."""
        crit = [d.name for d in self.engine.device.detectors if d.critical]
        return crit[0] if crit else None

    def state(self) -> dict:
        e = self.engine
        return _clean({
            "sources": self.sources(),
            "source": self.source_name,
            "profiles": [{"id": k, "title": p.title, "description": p.description} for k, p in PROFILES.items()],
            "profile": self.profile,
            "target": self.target,
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
            "classes": CLASS_NAMES,
            "can_inject": hasattr(e.source, "schedule"),
            "push": bool(e.notify.push_topic),
        })

    # -- control ------------------------------------------------------------- #
    def control(self, msg: dict) -> dict:
        a = msg.get("action")
        with self.lock:
            e = self.engine
            if a == "set":
                if "policy" in msg:
                    e.cfg.policy = self.policy = msg["policy"]
                if "alert_copies" in msg:
                    e.cfg.alert_copies = self.alert_copies = int(msg["alert_copies"])
                if "speed" in msg:
                    self.speed = float(msg["speed"])
                ch = {k: float(msg[k]) for k in ("bandwidth_bps", "loss_rate", "burst_length") if k in msg}
                if "max_retries" in msg:
                    ch["max_retries"] = int(msg["max_retries"])
                if ch:
                    for eng in (self.engine, self.baseline):
                        eng.set_channel(**ch)
                    for k, v in ch.items():
                        setattr(self.channel, k, v)
            elif a == "inject":
                e.source.inject(msg["kind"], msg.get("duration"))
            elif a == "outage":
                for eng in (self.engine, self.baseline):
                    eng.outage_now(float(msg.get("seconds", 10)))
            elif a == "ack":
                e.notify.acknowledge(msg.get("id"), e.t)
            elif a == "pause":
                self.paused = True
            elif a == "resume":
                self.paused = False
            elif a in ("reset", "source", "profile"):
                if a == "source":
                    self.source_name = msg["source"]
                if a == "profile":
                    self.profile = msg["profile"]
                self._build_all()
                self._broadcast({"type": "reset", "state": self.state()})
        return self.state()

    # -- the race: when did each receiver know? ------------------------------ #
    def race(self) -> dict | None:
        tgt = self.target
        if tgt is None:
            return None
        e, b, t = self.engine, self.baseline, self.engine.t
        fs = e.cfg.frame_s
        start = None
        for r in reversed(e.records):  # latest ground-truth onset of the target event
            if TRUTH_OF.get(tgt, tgt) in r["truth"]:
                start = r["t"] - fs
            elif start is not None:
                break
        if start is None:
            return {"target": tgt, "onset": None, "t": t}
        ev = next((x for x in e.events if x.detector == tgt and np.isfinite(x.t_onset_device) and start - PRE_S <= x.t_onset_device), None)
        early = ev.t_early_arrival if ev and ev.t_early_arrival is not None and ev.t_early_arrival <= t else None
        if ev is None:  # early warning raised but not confirmed yet
            el = next((x for x in e.early_logs if x.detector == tgt and x.t_device >= start - PRE_S and x.t_arrival is not None and x.t_arrival <= t), None)
            early = el.t_arrival if el else None
        conf = ev.t_alert_arrival if ev and ev.t_alert_arrival is not None and ev.t_alert_arrival <= t else None
        bev = next((x for x in b.events if x.detector == tgt and x.t_onset_receiver is not None and start - PRE_S <= x.t_onset_receiver <= t), None)
        return {
            "target": tgt,
            "title": e.device._det(tgt).title,
            "onset": start,
            "t": t,
            "pl_early": None if early is None else early - start,
            "pl_confirmed": None if conf is None else conf - start,
            "pl_path": ev.alert_path if ev and conf is not None else None,
            "pl_local_alarm": None if not (ev and ev.t_local_alarm and ev.t_local_alarm <= t) else ev.t_local_alarm - start,
            "base": None if bev is None else bev.t_onset_receiver - start,
            "base_lag_s": max(0.0, t - b._rx_time) if b.records else None,
        }

    # -- streaming ----------------------------------------------------------- #
    def _broadcast(self, payload: dict) -> None:
        data = json.dumps(_clean(payload))
        for q in list(self.clients):
            try:
                q.put_nowait(data)
            except queue.Full:
                pass

    def _frame_payload(self, rec: dict, brec: dict) -> dict:
        e, b = self.engine, self.baseline
        fid = rec["frame"]
        resolved = [{**r, "recon": _r(e.recon[r["frame"]])} for r in rec["resolved"] if e.recon.get(r["frame"]) is not None]
        bres = [{"frame": r["frame"], "status": r["status"], "latency_s": r["latency_s"], "receiver": r["receiver"]} for r in brec["resolved"]]
        bl = e.link.backlog()
        s, bs = e.summary(), b.summary()
        return {
            "type": "frame",
            **{k: rec[k] for k in ("frame", "t", "mode", "codec", "priority", "bytes", "packets", "truth", "label", "device", "early", "alerts")},
            "orig": _r(e.orig[fid]),
            "resolved": resolved,
            "backfill": {str(x): _r(e.recon[x]) for x in rec["backfilled"] if e.recon.get(x) is not None},
            "backlog": {str(k): v for k, v in bl.items()},
            "base_backlog": sum(b.link.backlog().values()),
            "link_up": e.link.link_up(e.t),
            "offline": e.offline_since is not None,
            "base_offline": b.offline_since is not None,
            "base_resolved": bres,
            "summary": s,
            "base_summary": {k: bs[k] for k in ("compression_ratio", "bitrate_bps", "frames_lost", "mean_latency_s", "max_latency_s", "tx_power_mw", "battery_h", "backlog_bytes")},
            "notifications": e.notify.visible(e.t),
            "events": [{k: v for k, v in ev.__dict__.items() if not k.endswith("_mid")} for ev in e.events[-12:]],
            "race": self.race(),
        }

    def _trim(self, eng: StreamEngine, fid: int) -> None:
        for k in [k for k in eng.orig if k < fid - 40]:  # free memory
            eng.orig.pop(k, None)
            eng.recon.pop(k, None)
            eng._rec_by_frame.pop(k, None)
        if len(eng.records) > 400:
            del eng.records[:100]

    def loop(self) -> None:
        while not self._stop.is_set():
            t0 = time.time()
            if not self.paused:
                with self.lock:
                    e, b = self.engine, self.baseline
                    try:
                        chunk = e.source.read(e.n)
                    except StopIteration:
                        self.paused = True
                        continue
                    rec = e.step(chunk)
                    brec = b.step(chunk)
                    payload = self._frame_payload(rec, brec)
                    for eng in (e, b):
                        self._trim(eng, rec["frame"])
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


def serve(host: str = "127.0.0.1", port: int = 8050, model=None, speed: float = 1.0, extra_detectors=None, edf=None, edf_channels=None, open_browser: bool = True, profile: str = "general") -> None:
    app = LiveApp(model, speed, extra_detectors, edf, edf_channels, profile)
    app.start()
    httpd = ThreadingHTTPServer((host, port), make_handler(app))
    httpd.daemon_threads = True
    url = f"http://{'localhost' if host in ('127.0.0.1', '0.0.0.0') else host}:{port}"
    print(f"PulseLink Live running at {url}  (Ctrl+C to stop)")
    if app.engine.notify.push_topic:
        print(f"phone push on: critical and warning notifications go to ntfy topic '{app.engine.notify.push_topic}'")
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
