# datasets/

Downloaded EEG recordings live here and are git-ignored.

* `python main.py run --download` fetches Sleep-EDF (Sleep Cassette, subjects 0-3,
  night 1, about 200 MB) into `datasets/sleep-edf/` using
  `mne.datasets.sleep_physionet.age.fetch_data`.
* You can also copy `SC4xxxE0-PSG.edf` and matching `SC4xxxXX-Hypnogram.edf`
  files from https://physionet.org/content/sleep-edfx/1.0.0/ into
  `datasets/sleep-edf/` by hand; any pair found there is used automatically.
* If nothing is here, PulseLink falls back to synthetic sleep EEG
  (`preprocessing/synthetic.py`) so every script still runs.
