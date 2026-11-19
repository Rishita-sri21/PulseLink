"""Receiver-side notification centre.

Every alert that reaches the receiver becomes a notification with a
priority (P0 critical ... P4 log). Critical notifications must be
acknowledged; if nobody acknowledges one within ``escalate_after_s`` it is
escalated (in a deployment: call the next contact). Receiver-generated
notices ("monitoring offline") come from the link watchdog, because silence
must never look like "all clear".

Optional phone push: set the environment variable ``PULSELINK_NTFY_TOPIC``
to a topic name and install the free ntfy app (https://ntfy.sh); the live
demo then pushes P0/P1 notifications to the phone. Off by default, and
nothing leaves the computer unless the variable is set.
"""
from __future__ import annotations

import itertools
import logging
import os
import threading
import urllib.request
from dataclasses import asdict, dataclass, field

log = logging.getLogger(__name__)

PRIORITY_LABEL = {0: "P0 critical", 1: "P1 warning", 2: "P2", 3: "P3", 4: "P4 log"}
KIND_TEXT = {
    "alert": "{title} detected",
    "notice": "{title}",
    "early": "Possible {title_l}: checking",
    "offline": "{title}",
    "online": "{title}",
    "log": "{title}",
}


@dataclass
class Notification:
    id: int
    t: float  # arrival at the receiver
    priority: int
    kind: str  # alert, notice, early, offline, online, log
    detector: str | None
    text: str
    t_event: float | None = None  # when the wearable raised it
    path: str | None = None  # primary / backup / receiver
    state: str = "new"  # new, acknowledged, escalated
    t_ack: float | None = None
    t_escalated: float | None = None
    extra: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return asdict(self)


class NotificationCenter:
    def __init__(self, escalate_after_s: float = 30.0, auto_ack_s: float | None = None, push: bool | None = None):
        self.items: list[Notification] = []
        self._ids = itertools.count(1)
        self.escalate_after_s = escalate_after_s
        self.auto_ack_s = auto_ack_s
        topic = os.environ.get("PULSELINK_NTFY_TOPIC")
        self.push_topic = topic if (push is None or push) and topic else None

    def add(self, t: float, priority: int, kind: str, detector: str | None, title: str, t_event: float | None = None, path: str | None = None, **extra) -> Notification:
        text = KIND_TEXT.get(kind, "{title}").format(title=title, title_l=title.lower())
        n = Notification(next(self._ids), float(t), int(priority), kind, detector, text, t_event, path, extra=extra)
        if priority >= 4:
            n.state = "logged"
        self.items.append(n)
        if self.push_topic and priority <= 1:
            self._push(n)
        return n

    def acknowledge(self, nid: int | None = None, t: float | None = None) -> int:
        """Acknowledge one notification (or all open critical ones)."""
        k = 0
        for n in self.items:
            if (nid is None or n.id == nid) and n.state in ("new", "escalated"):
                n.state, n.t_ack = "acknowledged", t
                k += 1
        return k

    def tick(self, t: float) -> list[Notification]:
        """Escalate unacknowledged critical notifications."""
        out = []
        for n in self.items:
            if n.priority == 0 and n.state == "new" and n.t <= t:
                if self.auto_ack_s is not None and t - n.t >= self.auto_ack_s:
                    n.state, n.t_ack = "acknowledged", n.t + self.auto_ack_s
                elif t - n.t >= self.escalate_after_s:
                    n.state, n.t_escalated = "escalated", n.t + self.escalate_after_s
                    out.append(n)
        return out

    def visible(self, t: float, limit: int = 30) -> list[dict]:
        return [n.as_dict() for n in self.items if n.t <= t][-limit:]

    def open_critical(self, t: float) -> list[Notification]:
        return [n for n in self.items if n.priority == 0 and n.state in ("new", "escalated") and n.t <= t]

    def _push(self, n: Notification) -> None:
        def go():
            try:
                req = urllib.request.Request(
                    f"https://ntfy.sh/{self.push_topic}",
                    data=n.text.encode(),
                    headers={"Title": f"PulseLink {PRIORITY_LABEL[n.priority]}", "Priority": "urgent" if n.priority == 0 else "high"},
                )
                urllib.request.urlopen(req, timeout=5).read()
            except Exception as exc:  # noqa: BLE001
                log.warning("push failed: %s", exc)

        threading.Thread(target=go, daemon=True).start()
