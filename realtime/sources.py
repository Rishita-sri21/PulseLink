"""EEG sources for real-time streaming.

* :class:`LiveSynthSource`: endless synthetic EEG whose brain state changes
  on a schedule *or on demand* (``inject("seizure")``), so the demo can show
  any use case live without a dataset.
* :class:`RecordingSource`: replays a real recording (Sleep-EDF, CHB-MIT or
  any EDF) in real time, with its annotated events as ground truth.

Every source yields 2-D chunks ``(C, n)`` in microvolts at ``fs`` plus the
set of ground-truth events active at the end of the chunk.
"""
from __future__ import annotations

import logging
import re
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy import signal

from config import DATA_DIR

log = logging.getLogger(__name__)

EVENT_KINDS = ("seizure", "drowsiness", "attention", "artifact")

# Component amplitudes (uV) of each background brain state.
# keys: pink (1/f background), delta, theta, alpha, beta, gamma, spindle
_STATES = {
    "relaxed": dict(pink=9, delta=4, theta=5, alpha=20, beta=4, gamma=1),
    "focus": dict(pink=9, delta=3, theta=4, alpha=6, beta=13, gamma=4),
    "drowsy": dict(pink=9, delta=10, theta=22, alpha=4, beta=2, gamma=0.5),
    "sleep": dict(pink=10, delta=18, theta=10, alpha=2, beta=2, gamma=0.5, spindle=1),
    "postictal": dict(pink=6, delta=30, theta=6, alpha=1, beta=1, gamma=0.3),
}
_FREQS = dict(delta=2.0, theta=6.0, alpha=10.0, beta=19.0, gamma=35.0)
STATE_TO_EVENT = {"drowsy": "drowsiness", "focus": "attention"}


@dataclass
class Chunk:
    data: np.ndarray  # (C, n) uV
    t_end: float  # stream time at the end of the chunk (s)
    truth: set[str] = field(default_factory=set)  # events active in the chunk
    label: str | None = None  # e.g. sleep stage or background state


class _Osc:
    """Oscillator with continuous phase and slowly wandering frequency."""

    def __init__(self, f0: float, rng):
        self.f0, self.rng, self.phase, self.df = f0, rng, rng.uniform(0, 2 * np.pi), 0.0

    def run(self, n: int, fs: float, f0: float | None = None) -> np.ndarray:
        f0 = self.f0 if f0 is None else f0
        self.df = 0.98 * self.df + self.rng.normal(0, 0.05)
        freq = f0 * (1 + 0.04 * np.tanh(self.df))
        ph = self.phase + 2 * np.pi * freq * np.arange(1, n + 1) / fs
        self.phase = ph[-1] % (2 * np.pi)
        return ph


class LiveSynthSource:
    """Endless 2-channel synthetic EEG with schedulable / injectable events.

    ``schedule`` is a list of ``(start_s, duration_s, kind)`` where kind is a
    background state (``relaxed``, ``focus``, ``drowsy``, ``sleep``) or an
    overlay event (``seizure``, ``artifact``, ``flat``).
    """

    def __init__(self, fs: float = 100.0, n_channels: int = 2, seed: int = 0, schedule=None, base: str = "relaxed"):
        self.fs, self.c = fs, n_channels
        self.channels = ["Frontal (sim)", "Parietal (sim)", "Temporal L (sim)", "Temporal R (sim)"][:n_channels] + [f"ch{i} (sim)" for i in range(4, n_channels)]
        self.rng = np.random.default_rng(seed)
        self.t = 0.0
        self.base = base
        self.schedule = sorted(schedule or [], key=lambda s: s[0])
        self.amp = {k: float(v) for k, v in _STATES[base].items()}
        self.osc = {k: _Osc(f, self.rng) for k, f in _FREQS.items()}
        self.sz_osc = _Osc(5.0, self.rng)
        self.gain = self.rng.uniform(0.8, 1.2, size=(n_channels, 1))
        # pink-ish noise: white noise through a 1-pole low-pass, per channel
        self._pink_sos = signal.butter(1, 3.0, btype="lowpass", fs=fs, output="sos")
        self._pink_zi = np.zeros((self.c, self._pink_sos.shape[0], 2))
        self.sz_amp = 0.0
        self._spindle_left = 0
        self._flat_val = None

    # -- control ------------------------------------------------------------- #
    def inject(self, kind: str, duration_s: float | None = None, start_s: float | None = None) -> None:
        dur = duration_s or {"seizure": 30.0, "artifact": 6.0, "flat": 6.0}.get(kind, 30.0)
        self.schedule.append((self.t if start_s is None else start_s, dur, kind))
        self.schedule.sort(key=lambda s: s[0])

    def active(self, t: float) -> list[str]:
        return [k for s, d, k in self.schedule if s <= t < s + d]

    def state_at(self, t: float) -> str:
        bases = [k for k in self.active(t) if k in _STATES and k != "postictal"]
        if any(k == "seizure" and s + d <= t < s + d + 20 for s, d, k in self.schedule):
            return "postictal"
        return bases[-1] if bases else self.base

    def truth_at(self, t: float) -> set[str]:
        out = {k for k in self.active(t) if k in ("seizure", "artifact")}
        if "flat" in self.active(t):
            out.add("artifact")
        state = self.state_at(t)
        ev = STATE_TO_EVENT.get(state)
        if ev:
            out.add(ev)
        if state == "postictal":
            out.add("postictal")  # slowing after a seizure (not scored as an event)
        return out

    # -- generation ---------------------------------------------------------- #
    def read(self, n: int) -> Chunk:
        fs, rng = self.fs, self.rng
        t_mid = self.t + n / (2 * fs)
        state = self.state_at(t_mid)
        target = _STATES[state]
        alpha_s = 1 - np.exp(-(n / fs) / 1.5)  # ~1.5 s state transitions
        for k in set(self.amp) | set(target):
            self.amp[k] = self.amp.get(k, 0.0) + alpha_s * (target.get(k, 0.0) - self.amp.get(k, 0.0))

        white = rng.normal(size=(self.c, n))
        pink = np.empty_like(white)
        for ch in range(self.c):
            pink[ch], self._pink_zi[ch] = signal.sosfilt(self._pink_sos, white[ch], zi=self._pink_zi[ch])
        pink = pink / 0.17 * self.amp["pink"] * 0.6 + 0.4 * self.amp["pink"] * white * 0.3
        x = pink
        for k, osc in self.osc.items():
            ph = osc.run(n, fs)
            env = 1 + 0.3 * np.sin(ph / 23.0)
            amp = self.amp.get(k, 0.0)
            x = x + amp * env * np.vstack([np.sin(ph + 0.3 * c) for c in range(self.c)]) * self.gain

        # sleep spindles
        if self.amp.get("spindle", 0) > 0.5 and self._spindle_left <= 0 and rng.random() < 0.15 * n / fs:
            self._spindle_left = int(rng.uniform(0.6, 1.2) * fs)
        if self._spindle_left > 0:
            m = min(n, self._spindle_left)
            tt = np.arange(m) / fs
            x[:, :m] += 30 * np.sin(np.pi * np.linspace(0, 1, m)) * np.sin(2 * np.pi * 13 * tt)
            self._spindle_left -= m

        # seizure: evolving rhythmic spike-wave, slowing from ~6 Hz to ~3 Hz
        act = self.active(t_mid)
        sz = [(s, d) for s, d, k in self.schedule if k == "seizure" and s <= t_mid < s + d]
        target_amp = 170.0 if sz else 0.0
        self.sz_amp += (1 - np.exp(-(n / fs) / 2.0)) * (target_amp - self.sz_amp)
        if self.sz_amp > 1.0:
            prog = (t_mid - sz[0][0]) / sz[0][1] if sz else 1.0
            f0 = 6.0 - 3.0 * np.clip(prog, 0, 1)
            ph = self.sz_osc.run(n, fs, f0)
            wave = np.sin(ph) + 0.5 * np.sin(2 * ph + 0.5) + 0.25 * np.sin(3 * ph + 1.0)  # spiky
            x = x + self.sz_amp * wave[None] * np.array([[1.0], [0.8]])[: self.c]

        if "artifact" in act:  # movement burst on one channel
            ch = int(self.t // 7) % self.c
            x[ch] += rng.normal(scale=180.0, size=n) * (1 + np.sin(np.arange(n) / fs * 3))
        if "flat" in act:  # electrode off: channel stuck at one value
            if self._flat_val is None:
                self._flat_val = float(x[0, 0])
            x[0] = self._flat_val
        else:
            self._flat_val = None

        self.t += n / fs
        return Chunk(x, self.t, self.truth_at(self.t - n / (2 * fs)), state)


def demo_tour() -> list[tuple[float, float, str]]:
    """The scripted demo: every use case once in ~4 minutes."""
    return [
        (45.0, 30.0, "focus"),
        (90.0, 30.0, "drowsy"),
        (135.0, 6.0, "artifact"),
        (160.0, 35.0, "seizure"),
        (230.0, 8.0, "flat"),
    ]


def random_schedule(rng, duration_s: float, kinds=("focus", "drowsy", "seizure", "artifact"), calib_s: float = 45.0, gap_s: float = 25.0):
    """Random non-overlapping events after a quiet calibration period."""
    out, t = [], calib_s
    while True:
        kind = rng.choice(kinds)
        dur = {"seizure": rng.uniform(20, 60), "artifact": rng.uniform(3, 10)}.get(kind, rng.uniform(25, 50))
        t += rng.uniform(0.5, 1.5) * gap_s
        if t + dur + 25 > duration_s:
            return out
        out.append((float(t), float(dur), str(kind)))
        t += dur + (20 if kind == "seizure" else 0)


class RecordingSource:
    """Replays a recording; ``events`` maps kind -> [(start_s, end_s), ...]."""

    def __init__(self, data: np.ndarray, fs: float, events: dict | None = None, labels: np.ndarray | None = None, label_names=None, name: str = "recording", loop: bool = True, channels=None, start_s: float = 0.0):
        self.data, self.fs, self.events = np.asarray(data, dtype=np.float64), fs, events or {}
        self.labels, self.label_names, self.name, self.loop = labels, label_names, name, loop
        self.channels = channels or [f"ch{i}" for i in range(self.data.shape[0])]
        self.c = self.data.shape[0]
        self.start = int(round(start_s * fs)) if 0 <= start_s * fs < self.data.shape[1] else 0
        self.pos = self.start
        self.t = 0.0

    @property
    def duration_s(self) -> float:
        return self.data.shape[1] / self.fs

    def inject(self, kind, duration_s=None, start_s=None):  # recordings cannot be changed
        log.info("cannot inject %s into a recording", kind)

    def read(self, n: int) -> Chunk:
        if self.pos + n > self.data.shape[1]:
            if not self.loop:
                raise StopIteration
            self.pos = self.start
        x = self.data[:, self.pos : self.pos + n]
        t_rec = (self.pos + n / 2) / self.fs
        self.pos += n
        self.t += n / self.fs
        truth = {k for k, iv in self.events.items() if any(s <= t_rec < e for s, e in iv)}
        label = None
        if self.labels is not None:
            i = min(int(t_rec // 30), len(self.labels) - 1)
            lab = int(self.labels[i])
            label = self.label_names[lab] if self.label_names and lab >= 0 else str(lab)
        return Chunk(x, self.t, truth, label)


# --------------------------------------------------------------------------- #
# Real datasets
# --------------------------------------------------------------------------- #
# PhysioNet first, then the byte-identical AWS Open Data mirror (PhysioNet is
# blocked on some networks, e.g. cloud sandboxes and campus proxies).
CHBMIT_MIRRORS = (
    "https://physionet.org/files/chbmit/1.0.0",
    "https://physionet-open.s3.amazonaws.com/chbmit/1.0.0",
)
CHBMIT_DIR = DATA_DIR / "chbmit"
DEFAULT_CHB_CHANNELS = ("F7-T7", "F8-T8")


def parse_chbmit_summary(text: str) -> dict[str, list[tuple[float, float]]]:
    """``chbXX-summary.txt`` -> {edf file name: [(start_s, end_s), ...]}."""
    out, current, starts = {}, None, []
    for line in text.splitlines():
        line = line.strip()
        m = re.match(r"File Name:\s*(\S+)", line)
        if m:
            current = m.group(1)
            out[current] = []
            continue
        m = re.match(r"Seizure(?:\s*\d+)?\s*Start Time:\s*(\d+)", line)
        if m and current:
            starts.append(float(m.group(1)))
            continue
        m = re.match(r"Seizure(?:\s*\d+)?\s*End Time:\s*(\d+)", line)
        if m and current and starts:
            out[current].append((starts.pop(0), float(m.group(1))))
    return out


def _fetch(rel: str, dest: Path, mirrors=CHBMIT_MIRRORS) -> Path:
    """Download ``rel`` (path under the dataset root) from the first mirror that works."""
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    errors = []
    for base in mirrors:
        url = f"{base}/{rel}"
        try:
            log.warning("downloading %s", url)
            req = urllib.request.Request(url, headers={"User-Agent": "PulseLink"})
            with urllib.request.urlopen(req, timeout=60) as r, open(tmp, "wb") as f:  # noqa: S310 - fixed https mirrors
                while chunk := r.read(1 << 20):
                    f.write(chunk)
            tmp.rename(dest)
            return dest
        except Exception as exc:  # noqa: BLE001 - try the next mirror
            errors.append(f"{base}: {exc}")
            tmp.unlink(missing_ok=True)
    raise OSError("all mirrors failed: " + "; ".join(errors))


def fetch_chbmit(patient: str = "chb01", n_seizure_files: int = 4, n_normal_files: int = 2, path: Path | None = None, download: bool = True):
    """Download (or find) part of one CHB-MIT patient.

    Returns ``[(edf_path, [(start_s, end_s), ...]), ...]``. Each EDF is about
    40 MB (1 h, 23 channels at 256 Hz).
    """
    root = Path(path or CHBMIT_DIR) / patient
    summary_path = root / f"{patient}-summary.txt"
    if download:
        _fetch(f"{patient}/{patient}-summary.txt", summary_path)
    if not summary_path.exists():
        raise FileNotFoundError(f"no CHB-MIT summary at {summary_path} (use --download)")
    seizures = parse_chbmit_summary(summary_path.read_text())
    with_sz = [f for f, s in seizures.items() if s][:n_seizure_files]
    without = [f for f, s in seizures.items() if not s][:n_normal_files]
    out = []
    for name in with_sz + without:
        p = root / name
        if download:
            _fetch(f"{patient}/{name}", p)
        if p.exists():
            out.append((p, seizures[name]))
    return out


def load_edf(path: Path, channels=None, fs_out: float = 100.0) -> tuple[np.ndarray, list[str]]:
    """Any EDF -> (C, N) uV at ``fs_out``. Bipolar names like ``F7-T7`` that
    appear more than once (CHB-MIT duplicates) are matched by prefix."""
    import mne

    raw = mne.io.read_raw_edf(str(path), preload=True, verbose="error")
    names = raw.ch_names
    if channels:
        picks = []
        for ch in channels:
            hit = [n for n in names if n == ch] or [n for n in names if n.upper().startswith(ch.upper())]
            if not hit:
                raise ValueError(f"channel {ch} not in {path.name}: {names}")
            picks.append(hit[0])
    else:
        picks = [n for n in names if raw.get_channel_types([n])[0] == "eeg"][:2] or names[:2]
    data = raw.get_data(picks=picks) * 1e6
    fs = raw.info["sfreq"]
    if abs(fs - fs_out) > 1e-6:
        from fractions import Fraction

        fr = Fraction(fs_out / fs).limit_denominator(1000)
        data = signal.resample_poly(data, fr.numerator, fr.denominator, axis=-1)
    return data, list(picks)


def chbmit_source(edf_path: Path, seizures, channels=DEFAULT_CHB_CHANNELS, fs: float = 100.0, loop: bool = True, lead_s: float | None = None) -> RecordingSource:
    """``lead_s``: start this many seconds before the first seizure (live demo)."""
    data, picks = load_edf(edf_path, channels, fs)
    start = max(0.0, seizures[0][0] - lead_s) if (lead_s is not None and seizures) else 0.0
    return RecordingSource(data, fs, {"seizure": list(seizures)}, name=Path(edf_path).name, loop=loop, channels=picks, start_s=start)


def local_chbmit_files(path: Path | None = None):
    """[(edf_path, seizures)] for CHB-MIT files already downloaded."""
    out = []
    for summary in sorted(Path(path or CHBMIT_DIR).glob("chb*/chb*-summary.txt")):
        sz = parse_chbmit_summary(summary.read_text())
        for name, s in sz.items():
            p = summary.parent / name
            if p.exists():
                out.append((p, s))
    return out


def sleep_edf_source(index: int = 0, loop: bool = True) -> RecordingSource:
    from config import SLEEP_STAGES
    from preprocessing.load_eeg import find_local_recordings, load_sleep_edf_recording

    pairs = find_local_recordings()
    if not pairs:
        raise FileNotFoundError("no Sleep-EDF recordings (run: python main.py run --download)")
    psg, hyp = pairs[index % len(pairs)]
    rec = load_sleep_edf_recording(psg, hyp)
    return RecordingSource(rec["data"], rec["fs"], {}, labels=rec["labels"], label_names=SLEEP_STAGES, name=psg.name, loop=loop, channels=rec["channels"])
