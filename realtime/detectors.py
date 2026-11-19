"""Pluggable real-time EEG event detectors.

A detector turns the shared window :class:`~realtime.features.Features`
into a score; the base class adds a per-person running baseline and
hysteresis (``on_frames`` consecutive hits to switch on, ``off_frames``
misses to switch off) so alerts do not flicker.

Built-in use cases (all registered in :data:`REGISTRY`):

========== ============================================== ==========
name       what it looks for                              critical
========== ============================================== ==========
seizure    rhythmic, high line-length, high-power activity  yes
drowsiness alpha drop-out with rising theta (vs baseline)  no
attention  engagement index beta / (alpha + theta) rising  no
artifact   flat line, saturation, one-sided bursts         no
sleep      30 s sleep stage from the trained stager        no
========== ============================================== ==========

Add a new use case by subclassing :class:`Detector` and decorating it with
``@register``, or train one from labelled windows with
:class:`LearnedDetector` (works for any binary label: seizure, drowsy,
motor imagery, ...).
"""
from __future__ import annotations

import copy
from collections import deque
from dataclasses import dataclass

import numpy as np

from realtime.features import Features

REGISTRY: dict[str, type["Detector"]] = {}


def register(cls):
    REGISTRY[cls.name] = cls
    return cls


@dataclass
class DetectorOutput:
    name: str
    score: float | None  # None while calibrating / skipped
    active: bool
    onset: bool = False
    offset: bool = False
    label: str | None = None  # categorical output (e.g. sleep stage)


class RunningBaseline:
    """Robust per-person baseline of a scalar feature (median / MAD)."""

    def __init__(self, size: int = 60, min_count: int = 5, floor: float = 0.15):
        self.values: deque[float] = deque(maxlen=size)
        self.min_count, self.floor = min_count, floor

    def ready(self) -> bool:
        return len(self.values) >= self.min_count

    def z(self, v: float) -> float:
        if not self.ready():
            return 0.0
        a = np.asarray(self.values)
        med = np.median(a)
        scale = max(1.4826 * np.median(np.abs(a - med)), self.floor)
        return float((v - med) / scale)

    def add(self, v: float) -> None:
        self.values.append(float(v))


class Detector:
    name = "detector"
    title = "Detector"
    description = ""
    critical = False  # critical events trigger an alert packet + high-fidelity mode
    priority: int | None = None  # notification class on onset (0 = P0 ... 4 = P4); None = P0 if critical else P1
    early: bool | None = None  # two-stage early warning; None = only for critical detectors
    early_window_s = 2.0  # short window for the early stage, evaluated every tick
    early_threshold: float | None = None  # None = same as threshold
    early_hits = 2  # consecutive ticks over threshold (2 x 0.5 s)
    confirm_s = 8.0  # early warning cancelled if not confirmed within this time
    window_s = 4.0
    hop_s: float | None = None  # None = every frame
    threshold = 3.0
    on_frames = 2
    off_frames = 2
    suppress_on_artifact = True
    color = "#888888"

    def __init__(self, **kw):
        for k, v in kw.items():
            if not hasattr(self, k):
                raise TypeError(f"{type(self).__name__} has no parameter {k!r}")
            setattr(self, k, v)
        self.reset()

    # -- override these ----------------------------------------------------- #
    def baseline_values(self, f: Features) -> dict[str, float]:
        """Scalar features tracked by a running baseline (name -> value)."""
        return {}

    def raw_score(self, f: Features, z: dict[str, float]) -> float:
        raise NotImplementedError

    # -- machinery ---------------------------------------------------------- #
    def reset(self) -> None:
        self.active = False
        self._pos = self._neg = 0
        self._baselines: dict[str, RunningBaseline] = {}

    def calibrating(self) -> bool:
        return any(not b.ready() for b in self._baselines.values())

    def step(self, f: Features, artifact: bool = False) -> DetectorOutput:
        vals = self.baseline_values(f)
        for k in vals:
            self._baselines.setdefault(k, RunningBaseline())
        z = {k: self._baselines[k].z(v) for k, v in vals.items()}
        if vals and self.calibrating():
            score = None
        else:
            score = float(self.raw_score(f, z))
        hit = score is not None and score >= self.threshold and not (artifact and self.suppress_on_artifact)
        was = self.active
        if hit:
            self._pos, self._neg = self._pos + 1, 0
            if self._pos >= self.on_frames:
                self.active = True
        else:
            self._neg, self._pos = self._neg + 1, 0
            if self._neg >= self.off_frames:
                self.active = False
        # Only learn the baseline from quiet, clean windows.
        if not self.active and not hit and not artifact:
            for k, v in vals.items():
                self._baselines[k].add(v)
        return DetectorOutput(self.name, score, self.active, onset=self.active and not was, offset=was and not self.active)

    def clone(self) -> "Detector":
        d = copy.deepcopy(self)
        d.reset()
        return d

    @property
    def notify_class(self) -> int:
        return self.priority if self.priority is not None else (0 if self.critical else 1)

    @property
    def has_early(self) -> bool:
        if getattr(self, "uses_raw", False):
            return False
        return bool(self.critical if self.early is None else self.early)

    def info(self) -> dict:
        return {"name": self.name, "title": self.title, "description": self.description, "critical": self.critical, "threshold": self.threshold, "color": self.color, "window_s": self.window_s, "priority": self.notify_class, "early": self.has_early}


@dataclass
class EarlyOutput:
    name: str
    score: float | None
    onset: bool = False  # "possible event" raised now
    cancelled: bool = False  # not confirmed in time
    pending: bool = False  # a possible event is waiting for confirmation


class EarlyStage:
    """Fast first stage of a detector: same score, shorter window, every tick.

    It raises a *possible event* warning (sent as P1) after ``early_hits``
    consecutive ticks over threshold, typically 1-2 s after onset instead
    of ~4 s. The full detector then confirms it (P0 alert) or the warning
    is cancelled after ``confirm_s``. The early stage keeps its own
    baseline because short-window features have a different spread.
    """

    def __init__(self, det: Detector):
        self.det = det
        self.window_s = det.early_window_s
        self.threshold = det.threshold if det.early_threshold is None else det.early_threshold
        self.reset()

    def reset(self) -> None:
        self._baselines: dict[str, RunningBaseline] = {}
        self._pos = 0
        self.pending_since: float | None = None
        self._refractory_until = -1.0

    def step(self, f: Features, t: float, artifact: bool = False) -> EarlyOutput:
        d = self.det
        vals = d.baseline_values(f)
        for k in vals:
            self._baselines.setdefault(k, RunningBaseline(size=240, min_count=20, floor=0.15))
        if any(not b.ready() for b in self._baselines.values()):
            score = None
        else:
            score = float(d.raw_score(f, {k: self._baselines[k].z(v) for k, v in vals.items()}))
        hit = score is not None and score >= self.threshold and not (artifact and d.suppress_on_artifact)
        self._pos = self._pos + 1 if hit else 0
        out = EarlyOutput(d.name, score)
        if d.active:  # confirmed: the warning has done its job
            self.pending_since = None
            self._refractory_until = t + d.confirm_s
        elif self.pending_since is not None and t - self.pending_since > d.confirm_s:
            out.cancelled = True
            self.pending_since = None
            self._refractory_until = t + d.confirm_s
        elif self.pending_since is None and self._pos >= d.early_hits and t >= self._refractory_until:
            out.onset = True
            self.pending_since = t
        out.pending = self.pending_since is not None
        if not hit and not d.active and not artifact and self.pending_since is None:
            for k, v in vals.items():
                self._baselines[k].add(v)
        return out


@register
class SeizureDetector(Detector):
    name = "seizure"
    title = "Seizure"
    description = "Rhythmic activity with line length and power far above this person's baseline"
    critical = True
    window_s = 4.0
    threshold = 4.0
    on_frames = 2
    off_frames = 3
    suppress_on_artifact = False
    color = "#ef4444"
    min_rhythmicity = 2.5

    def baseline_values(self, f):
        return {"ll": np.log(f.line_length + 1e-6), "pow": np.log(f.total_power + 1e-6)}

    def raw_score(self, f, z):
        s = (z["ll"] + 2.0 * z["pow"]) / 3.0  # power weighs double
        # Movement artifacts are broadband and one-sided; seizures are rhythmic.
        gate = np.clip((f.rhythmicity - 1.5) / (self.min_rhythmicity - 1.5), 0.0, 1.0)
        if f.channel_asymmetry > 4.0 or f.flat:
            gate = 0.0
        return s * gate


@register
class DrowsinessDetector(Detector):
    name = "drowsiness"
    title = "Drowsiness"
    description = "Alpha drop-out with rising theta (theta / (alpha + beta) above baseline) and no rise in beta"
    window_s = 4.0
    threshold = 3.0
    on_frames = 3
    off_frames = 3
    color = "#f59e0b"

    def baseline_values(self, f):
        return {"ta": np.log((f.band["theta"] + 1e-6) / (f.band["alpha"] + f.band["beta"] + 1e-6)), "beta": np.log(f.band["beta"] + 1e-6)}

    def raw_score(self, f, z):
        # Rising beta means engagement, not sleepiness: it cancels the score.
        return z["ta"] - max(0.0, z["beta"])


@register
class AttentionDetector(Detector):
    name = "attention"
    title = "Focus"
    description = "Engagement index beta / (alpha + theta) above baseline (Pope et al., 1995)"
    window_s = 4.0
    threshold = 3.0
    on_frames = 3
    off_frames = 3
    color = "#3b82f6"
    priority = 4  # a log entry, not a notification

    def baseline_values(self, f):
        return {"eng": np.log((f.band["beta"] + 1e-6) / (f.band["alpha"] + f.band["theta"] + 1e-6))}

    def raw_score(self, f, z):
        return z["eng"]


@register
class ArtifactDetector(Detector):
    name = "artifact"
    title = "Artifact / electrode"
    description = "Flat line, saturation or a burst on one channel only: signal not trustworthy"
    window_s = 2.0
    threshold = 1.0
    on_frames = 1
    off_frames = 1
    suppress_on_artifact = False
    color = "#a855f7"
    max_uv = 400.0
    max_asymmetry = 4.0

    def raw_score(self, f, z):
        return float(f.flat or f.max_abs > self.max_uv or f.channel_asymmetry > self.max_asymmetry)


@register
class SleepStageDetector(Detector):
    """Wraps the offline sleep stager (needs 30 s windows; outputs a label)."""

    priority = 4

    name = "sleep"
    title = "Sleep stage"
    description = "Sleep stage every 30 s from the band-power random forest"
    window_s = 30.0
    hop_s = 30.0
    threshold = np.inf
    suppress_on_artifact = False
    color = "#14b8a6"

    def __init__(self, stager=None, **kw):
        self.stager = stager
        super().__init__(**kw)

    def step_window(self, x: np.ndarray) -> DetectorOutput:
        from config import SLEEP_STAGES

        if self.stager is None:
            return DetectorOutput(self.name, None, False)
        stage = int(self.stager.predict(x[None])[0])
        return DetectorOutput(self.name, float(stage), False, label=SLEEP_STAGES[stage])

    def raw_score(self, f, z):  # pragma: no cover - uses step_window instead
        return 0.0


class LearnedDetector(Detector):
    """Binary detector learned from labelled windows (any use case).

    >>> det = LearnedDetector(name="seizure_learned", critical=True, kind="gbm")
    >>> det.fit(features_list, labels)          # labels: 0/1 per window

    ``kind``: ``"logreg"`` (logistic regression, a few hundred bytes on an
    MCU) or ``"gbm"`` (gradient-boosted trees, captures interactions such
    as "high power *and* rhythmic").
    """

    title = "Learned"
    description = "Classifier on the shared feature vector"
    threshold = 0.5
    color = "#ec4899"
    early = False  # trained on full windows; enable only after checking short windows

    def __init__(self, name: str = "learned", title: str | None = None, critical: bool = False, kind: str = "logreg", **kw):
        self.name = name
        self.title = title or name
        self.critical = critical
        self.kind = kind
        self.model = None
        super().__init__(**kw)

    def fit(self, feats: list[Features], y) -> "LearnedDetector":
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        X = np.stack([f.vector() for f in feats])
        y = np.asarray(y).astype(int)
        if self.kind == "gbm":
            from sklearn.ensemble import HistGradientBoostingClassifier

            self.model = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.08, max_leaf_nodes=15, l2_regularization=1.0, class_weight="balanced", random_state=0)
        else:
            from sklearn.linear_model import LogisticRegression

            self.model = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=2000, C=0.5))
        self.model.fit(X, y)
        self.description = {"gbm": "Gradient-boosted trees", "logreg": "Logistic regression"}.get(self.kind, self.kind) + " on the shared feature vector"
        return self

    def raw_score(self, f, z):
        if self.model is None:
            return 0.0
        return float(self.model.predict_proba(f.vector()[None])[0, 1])


class CNNDetector(Detector):
    """Tiny 1-D CNN on the raw window (about 18 k weights, under 20 kB as int8).

    Learns its own features from labelled windows, so it can pick up shapes
    that band powers miss (spike-and-wave, K-complexes, eye blinks...).
    """

    title = "CNN"
    description = "Tiny 1-D convolutional network on the raw EEG window"
    threshold = 0.5
    color = "#06b6d4"
    uses_raw = True
    early = False
    scale_uv = 50.0

    def __init__(self, name: str = "cnn", title: str | None = None, critical: bool = False, **kw):
        self.name = name
        self.title = title or name
        self.critical = critical
        self.state = None
        self.shape = None
        super().__init__(**kw)

    @staticmethod
    def _net(c: int):
        import torch.nn as nn

        def block(i, o):
            return nn.Sequential(nn.Conv1d(i, o, 7, stride=2, padding=3), nn.BatchNorm1d(o), nn.ReLU())

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.f = nn.Sequential(block(c, 16), block(16, 32), block(32, 32), block(32, 32))
                self.head = nn.Linear(64, 1)

            def forward(self, x):
                import torch

                h = self.f(x)
                return self.head(torch.cat([h.mean(-1), h.amax(-1)], 1)).squeeze(1)

        return Net()

    def _prep(self, x):
        x = np.asarray(x, dtype=np.float32)
        x = x - np.median(x, axis=-1, keepdims=True)
        return np.sign(x) * np.log1p(np.abs(x) / self.scale_uv)  # compress amplitude, keep sign

    def fit(self, windows: np.ndarray, y, epochs: int = 25, seed: int = 0) -> "CNNDetector":
        import torch

        torch.manual_seed(seed)
        X = torch.tensor(np.stack([self._prep(w) for w in windows]))
        Y = torch.tensor(np.asarray(y, dtype=np.float32))
        self.shape = tuple(X.shape[1:])
        net = self._net(X.shape[1])
        pos = float(Y.sum())
        pw = torch.tensor((len(Y) - pos) / max(pos, 1.0))
        opt = torch.optim.Adam(net.parameters(), 3e-3, weight_decay=1e-4)
        lossf = torch.nn.BCEWithLogitsLoss(pos_weight=pw)
        g = torch.Generator().manual_seed(seed)
        for _ in range(epochs):
            net.train()
            for idx in torch.randperm(len(Y), generator=g).split(64):
                xb = X[idx]
                xb = xb * (1 + 0.1 * torch.randn(len(idx), 1, 1))  # amplitude jitter
                opt.zero_grad()
                lossf(net(xb), Y[idx]).backward()
                opt.step()
        net.eval()
        self.state = {k: v.clone() for k, v in net.state_dict().items()}
        self._cache = None
        return self

    def n_params(self) -> int:
        return int(sum(v.numel() for k, v in (self.state or {}).items() if "running" not in k and "num_batches" not in k))

    def predict(self, windows) -> np.ndarray:
        import torch

        if self.state is None:
            return np.zeros(len(windows))
        if getattr(self, "_cache", None) is None:
            net = self._net(self.shape[0])
            net.load_state_dict(self.state)
            net.eval()
            self._cache = net
        with torch.no_grad():
            return torch.sigmoid(self._cache(torch.tensor(np.stack([self._prep(w) for w in windows])))).numpy()

    def step(self, x, artifact: bool = False) -> DetectorOutput:  # raw window instead of features
        score = float(self.predict([x])[0])
        hit = score >= self.threshold and not (artifact and self.suppress_on_artifact)
        was = self.active
        if hit:
            self._pos, self._neg = self._pos + 1, 0
            self.active = self.active or self._pos >= self.on_frames
        else:
            self._neg, self._pos = self._neg + 1, 0
            if self._neg >= self.off_frames:
                self.active = False
        return DetectorOutput(self.name, score, self.active, onset=self.active and not was, offset=was and not self.active)

    def clone(self) -> "CNNDetector":
        cache, self._cache = getattr(self, "_cache", None), None
        d = copy.deepcopy(self)
        self._cache = cache
        d.reset()
        return d

    def raw_score(self, f, z):  # pragma: no cover
        return 0.0


def default_detectors(names=("seizure", "drowsiness", "attention", "artifact"), stager=None) -> list[Detector]:
    out = []
    for n in names:
        if n == "sleep":
            out.append(SleepStageDetector(stager=stager))
        else:
            out.append(REGISTRY[n]())
    return out
