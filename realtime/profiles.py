"""Use-case profiles: which detectors run and how urgent each one is.

The engine, link and dashboard are the same for every use case; a profile
only picks detectors and sets their priority class:

* P0 critical: confirmed event -> alert with ACK, spaced copies, backup radio,
  local alarm; a fast early stage sends a "possible event" warning first.
* P1 warning: a notification that should arrive within seconds.
* P4 log: recorded at the receiver, nobody is disturbed.

Add a profile by adding an entry to :data:`PROFILES`.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from realtime.detectors import Detector, REGISTRY, SleepStageDetector


@dataclass
class Profile:
    name: str
    title: str
    description: str
    detectors: dict[str, int]  # detector name -> priority class (0 = critical)
    learned: list = field(default_factory=list)  # extra trained detectors (instances)

    def build(self, stager=None, extra: list[Detector] | None = None) -> list[Detector]:
        out = []
        for name, cls in self.detectors.items():
            d = SleepStageDetector(stager=stager) if name == "sleep" else REGISTRY[name]()
            d.priority = cls
            d.critical = cls == 0
            out.append(d)
        for d in list(self.learned) + list(extra or []):
            out.append(d.clone())
        return out


PROFILES: dict[str, Profile] = {
    "epilepsy": Profile(
        "epilepsy", "Epilepsy monitoring",
        "Seizure alarm for a caregiver; electrode problems are reported so silence is never mistaken for safety.",
        {"seizure": 0, "artifact": 1, "drowsiness": 4},
    ),
    "driver": Profile(
        "driver", "Driver / worker fatigue",
        "Drowsiness is the critical event: early warning to the driver, confirmed alert to the fleet.",
        {"drowsiness": 0, "attention": 4, "artifact": 1},
    ),
    "focus": Profile(
        "focus", "Focus and neurofeedback",
        "Nothing is urgent: focus and fatigue are logged and shown as feedback.",
        {"attention": 4, "drowsiness": 1, "artifact": 4},
    ),
    "general": Profile(
        "general", "General monitoring (demo)",
        "Every built-in detector: seizure critical, the rest as notifications or logs.",
        {"seizure": 0, "drowsiness": 1, "attention": 4, "artifact": 1},
    ),
}


def profile_detectors(name: str = "general", stager=None, extra=None) -> list[Detector]:
    return PROFILES[name].build(stager, extra)
