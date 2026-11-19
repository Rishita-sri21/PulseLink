"""One command to produce every real-time result and pack it for sharing.

    python run_realtime_study.py                 # synthetic + CHB-MIT chb01 (downloads ~250 MB once)
    python run_realtime_study.py --patient chb03 # another CHB-MIT patient
    python run_realtime_study.py --no-chbmit     # offline: synthetic experiments only

Writes results/realtime/ (summary.md, CSVs, figures, run.log, learned
detector) and zips it to pulselink_realtime_results.zip. Send that zip back.
Takes roughly 15-30 minutes on a laptop CPU, less on Colab.
"""
from __future__ import annotations

import argparse
import logging
import platform
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            st.write(s)
            st.flush()

    def flush(self):
        for st in self.streams:
            st.flush()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--patient", default="chb01")
    p.add_argument("--chb-files", type=int, default=4, help="CHB-MIT files with seizures (each ~40 MB)")
    p.add_argument("--no-chbmit", action="store_true")
    p.add_argument("--runs", type=int, default=12)
    p.add_argument("--minutes", type=float, default=10.0)
    p.add_argument("--quick", action="store_true", help="small run to check that everything works (~2 min)")
    args = p.parse_args()

    out = ROOT / "results" / "realtime"
    out.mkdir(parents=True, exist_ok=True)
    log_file = open(out / "run.log", "w")
    sys.stdout = Tee(sys.__stdout__, log_file)
    sys.stderr = Tee(sys.__stderr__, log_file)
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(name)s: %(message)s", stream=sys.stderr)

    import numpy as np
    import torch

    from compression.autoencoder import load_model
    from pipeline import DEFAULT_MODEL
    from realtime.experiments import run_all

    print(f"PulseLink real-time study  {time.strftime('%Y-%m-%d %H:%M')}")
    print(f"python {platform.python_version()}  numpy {np.__version__}  torch {torch.__version__}  {platform.platform()}")
    model = None
    if DEFAULT_MODEL.exists():
        model = load_model(DEFAULT_MODEL)
        print(f"autoencoder: {DEFAULT_MODEL} trained on {model.extra.get('source', '?')} (synthetic={model.extra.get('synthetic', '?')})")
    t0 = time.time()
    run_all(
        out,
        model=model,
        runs=args.runs,
        minutes=args.minutes,
        chbmit=None if args.no_chbmit else args.patient,
        chb_files=args.chb_files,
        download=not args.no_chbmit,
        quick=args.quick,
    )
    sys.stdout.flush()
    log_file.flush()
    zip_path = shutil.make_archive(str(ROOT / "pulselink_realtime_results"), "zip", out)
    print(f"\nAll done in {(time.time() - t0) / 60:.1f} min.")
    print(f"Send back: {zip_path}")
    print("(it contains summary.md, the CSVs, figures/ and run.log)")


if __name__ == "__main__":
    main()
