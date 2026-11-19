# Real-time EEG on wearables: processing and transmission for fast event detection

This note answers the question our teacher raised: beyond compressing EEG, how do
we **process and transmit EEG in real time** so that an event (a seizure, a driver
falling asleep, a lapse of attention, a bad electrode) is noticed within seconds,
on a wearable that runs for days on a small battery? It summarises the main
techniques and design choices and says which ones PulseLink now implements
(`realtime/`, `communication/priority_link.py`). Seizures are one example; the
same design serves any use case. Section 8 lists what is built; the measured
numbers are in `results/realtime/summary.md` after running the study.

> Treat the product and paper details below as a starting reading list. Check each
> one against the original source before citing it in the report.

## 1. The key idea: detect on the wearable, send the decision first

There are two ways to detect an event:

| | Stream everything, detect at the receiver | Detect on the wearable, send an alert |
|---|---|---|
| What goes over the air | the full EEG (compressed or not) | a 10–20 byte alert, plus EEG at whatever rate the link allows |
| Delay | detection delay + queueing + retransmissions + decoding | detection delay + one short packet (milliseconds) |
| Slow or congested link | delay grows without limit; raw EEG may never arrive | alert still gets through (it skips the queue) |
| Effect of compression | lossy compression can hide the features the detector needs | the detector sees the clean signal |
| Wearable compute | very little | a small feature extractor or classifier (fine on a Cortex-M4) |

Implanted closed-loop devices and clinical seizure wearables already run their
detectors on the device. PulseLink's results show why: on a 1.6 kbit/s link, a
receiver that waits for raw EEG never detects the seizure (the queue overflows),
while the on-device alert arrives about 4 s after onset whatever the link speed
(`results/realtime/receiver_latency_vs_bandwidth.png`).

### Where the seconds go (latency budget)

| Stage | Typical value | Lever |
|---|---|---|
| Analysis window | 1–4 s of EEG | shorter windows react faster but are noisier |
| Decision rule | 1–3 consecutive windows | hysteresis prevents flickering alarms |
| On-device compute | < 10 ms (features + small model) | int8 models, shared features |
| Radio, alert packet | 5–50 ms on BLE | priority queue, redundant copies |
| Radio, data stream | 0.1 s to minutes | compression, bandwidth-aware codec choice |
| Receiver processing | < 100 ms | n/a |

The decision rule, not the radio, usually dominates. PulseLink's seizure detector
uses 4 s windows every 2 s and needs 2 hits, so it reacts about 4 s after onset.
Shoeb and Guttag's patient-specific CHB-MIT detector reported a median latency
of a few seconds at very high sensitivity, which is the right order of magnitude
to aim for.

## 2. Hardware: what a real-time EEG wearable is made of

* **Electrodes and form factor.** Wet gel gives the best signal but is unpleasant
  for long use. Dry and semi-dry electrodes, headbands, behind-the-ear arrays
  (e.g. cEEGrid), in-ear EEG and subcutaneous electrodes trade channel count and
  coverage for comfort. For seizures, a few well-placed temporal channels often
  catch most focal seizures. That is why PulseLink defaults to two bipolar
  temporal channels (F7-T7, F8-T8) on CHB-MIT.
* **Analog front end (AFE).** Low-noise biopotential ADCs such as TI ADS1299
  (8 channels, 24-bit) or ADS1292, and Intan RHD chips. The AFE often uses
  more power than the radio. PulseLink's battery estimate assumes a 0.5 mW
  always-on front end, which is why compression extends battery life by about
  25 % rather than 5× (see `policy_cost.csv`).
* **Microcontroller plus radio.** BLE SoCs such as Nordic nRF52/nRF53 (Cortex-M4F/M33),
  ultra-low-power MCUs (Ambiq Apollo), ESP32 when Wi-Fi is needed. They are fast
  enough for band powers, line length, wavelets and small int8 CNNs in real time.
* **Open platforms for student projects:** OpenBCI (Cyton or Ganglion boards with
  dry-electrode headsets), Muse headbands (4 channels, BLE) and BITalino. They
  stream to a PC, so PulseLink's receiver and detectors could run on live data.
* **Clinical examples to look at:** wireless EEG patches for rapid seizure
  screening, subcutaneous long-term EEG recorders, and seizure-alert wristbands
  that use motion and skin conductance instead of EEG. Implanted responsive
  neurostimulators are the extreme case: detection is entirely on the device.

## 3. Transmission: getting the important bits through

* **Radio choice.** BLE 5 is the default for body-worn sensors: tens of nJ per
  bit, 244-byte packets, enough throughput for many EEG channels. Wi-Fi gives
  more bandwidth for more power. LoRa and LTE-M/NB-IoT carry little data but
  reach far, which suits alerts when the phone is not nearby. IEEE 802.15.6 is
  the body-area-network standard.
* **Prioritise events.** Use two logical channels: a tiny, high-priority alert
  channel and a best-effort data channel. In PulseLink, alerts skip the data
  queue and the data channel switches to high fidelity during the event.
* **Redundancy has to be spread out in time.** Sending an alert 3 times turns a
  20 % packet loss into about 0.6 % alert loss *if losses are independent*. Real
  links lose packets in bursts (body shadowing, interference), and back-to-back
  copies then die together: about 12 % alert loss in our bursty simulation.
  Spacing the copies about 8 packet slots apart brings it back to about 1 %
  (`alert_delivery.png`). Forward error correction with interleaving (Reed-Solomon,
  fountain codes) does the same for data.
* **ARQ vs FEC.** Retransmission (ARQ) is cheap when loss is low but adds a round
  trip each time and fails during outages. FEC adds a fixed overhead and no
  delay. Use ARQ for the data stream and redundancy or FEC for alerts.
* **Disconnections.** Keep a ring buffer in on-device flash (store and forward),
  re-send the pre-event context after reconnecting, and **alarm locally**
  (vibration or buzzer) when the link is down. In our 30 s outage scenario the
  alert can only arrive after reconnection (about 25 s), so a local alarm or a
  second radio path is the only way to do better.
* **Pre-trigger buffer.** Like cardiac event recorders, keep the last 10 s and
  send them at high fidelity when an event starts, so the clinician sees the onset.
  PulseLink does this.

## 4. Processing: real-time detection techniques

* **Causal preprocessing.** Zero-phase filtering (`filtfilt`) needs future
  samples, so it cannot run live. Use causal IIR filters with saved state
  (PulseLink: `StreamingBandpass`), plus artifact checks for flat lines,
  saturation and one-sided bursts *before* any detector runs.
* **Cheap, robust features.** Line length, signal energy, band powers (delta,
  theta, alpha, beta), ratios such as theta/alpha (drowsiness) and
  beta/(alpha+theta) (engagement index), spectral peak or rhythmicity, spectral
  entropy, Teager energy. One FFT per window can feed every detector.
* **Personal baselines.** EEG differs a lot between people and over the day.
  Comparing each feature with the wearer's own running median and spread
  (z-score) works much better than fixed thresholds. PulseLink learns the baseline
  only from quiet, artifact-free windows.
* **Learned detectors.** Patient-specific classifiers (SVM or logistic regression
  on features) work well for seizures with few training seizures. Small CNNs and
  temporal convolution networks run on MCUs after int8 quantisation (TinyML).
  PulseLink's `LearnedDetector` trains on any labelled windows, so a new use case
  needs labels, not new code.
* **Decision logic.** Hysteresis (k-of-n windows), refractory periods and muting
  lower-priority detectors during a critical event cut false alarms. False alarms
  per hour (or per day) is the metric that makes people stop wearing a device.
* **Use cases with the same pipeline:** seizure alerting, driver or worker
  drowsiness, attention and neurofeedback, sleep staging, BCI (SSVEP, P300,
  motor imagery), anaesthesia depth, neonatal monitoring.

## 5. Compression in a real-time system

* **Short frames hurt compression.** Every frame carries a header and the codec
  has less context. In PulseLink, 2 s frames compress about 7–11× against
  14–16× for 30 s segments. Real-time systems pick the frame length as a
  trade-off between latency and efficiency.
* **Compression can break a detector.** At the receiver, the PulseLink
  autoencoder (trained on synthetic sleep EEG) kept seizures and drowsiness
  detectable but lost enough beta power that the engagement detector mostly
  failed. This argues for **task-aware compression**: train or choose the codec
  for the features the downstream task needs, or detect on the device.
* **Event-aware rates.** Most of the time nothing is happening. Sending low-rate
  data then and high-fidelity data around events gave about 7 % PRD during
  seizures at an average bitrate close to the low-rate codec.
* **Low-encoder-cost codecs.** Compressed sensing and simple delta-plus-entropy
  coding put almost no load on the MCU and are worth comparing with the DWT and
  the autoencoder.

## 6. How to evaluate a real-time system

Report all of these, not only compression ratio and PRD:

* sensitivity (events detected), false alarms per hour, detection latency
  (median and 90th percentile);
* alert delivery probability and alert latency under loss, bursts and outages;
* end-to-end data latency and frames lost;
* fidelity *during events* (not only on average);
* transmitter energy and battery life, with the assumptions stated.

`python run_realtime_study.py` produces each of these for PulseLink.

## 7. Datasets

* **CHB-MIT Scalp EEG** (PhysioNet): 23 paediatric patients, 256 Hz, annotated
  seizures. PulseLink downloads it with `--chbmit chb01 --download`.
* **TUH EEG Seizure Corpus (TUSZ):** much larger, adult clinical EEG, needs
  registration.
* **Siena Scalp EEG** (PhysioNet): adult seizures.
* **SEED-VIG:** vigilance and drowsiness during simulated driving.
* **Sleep-EDF:** already used for compression and sleep staging.
* **BCI Competition IV:** motor imagery.

## 8. What PulseLink implements now, and what is next

PulseLink's distinct advantage is the **fastest reliable alert**: the receiver
should learn about an event within seconds even when the link is slow, lossy or
down. Implemented (`realtime/`, `communication/priority_link.py`):

* **Two-stage detection on the wearable.** Every 0.5 s a fast early stage (2 s
  window) raises a *possible event* warning; the full detector (4 s window)
  confirms it. Typically 1.5 s instead of 4 s for the first warning.
* **Priority link (P0-P4).** Critical alerts (P0) and warnings (P1) go ahead of
  every EEG packet and may interrupt one in flight; EEG around events (P2) goes
  ahead of routine EEG (P3); stale routine EEG is dropped rather than delivered
  late.
* **Reliability for alerts.** End-to-end ACK and retry, two copies spaced in time
  (bursty loss), a backup radio path when the main link does not ACK within
  0.5 s, a local alarm on the wearable if nothing is ACKed within 2 s, and a
  heartbeat so the receiver says "monitoring offline" instead of staying silent.
* **Notification centre** with acknowledgement and escalation, optional push to
  a phone (ntfy).
* **Use-case profiles** (epilepsy, driver, focus, general) that only change which
  detector is critical; the rest of the system is shared.
* **ML:** richer features (Hjorth, spectral entropy), logistic regression,
  gradient-boosted trees and a tiny 1-D CNN (about 18 k weights), compared on
  CHB-MIT leave-one-file-out; a **task-aware** fine-tune of the autoencoder that
  keeps the features the detectors need.
* **Compression stays in the loop:** event-aware codec choice, pre-trigger buffer,
  bandwidth-aware fidelity.
* **Evaluation:** time until the receiver knows vs link quality for PulseLink and
  typical streaming wearables, an ablation of every feature, detection quality,
  energy and battery.

Next steps, roughly in order of value:

1. Run on more CHB-MIT patients (`python run_realtime_study.py --patient chb03`) and report per-patient results.
2. Port the early stage, the detector and the priority scheduler to an nRF52 and measure real latency and energy.
3. Stream from an OpenBCI or Muse headset (via Lab Streaming Layer) into `RecordingSource` for a live demo with a person.
4. Replace the analytical backup radio with a real LTE-M or LoRa module and measure its latency.
5. Seizure *prediction* (minutes ahead) would be the next jump in speed, but needs long recordings and careful validation.

## References (starting points)

* Shoeb, A., Guttag, J. *Application of machine learning to epileptic seizure detection.* ICML 2010 (patient-specific CHB-MIT detector).
* Esteller, R. et al. *Line length: an efficient feature for seizure onset detection.* IEEE EMBS 2001.
* Gotman, J. *Automatic recognition of epileptic seizures in the EEG.* EEG and Clinical Neurophysiology, 1982.
* Pope, A. T., Bogart, E. H., Bartolome, D. S. *Biocybernetic system evaluates indices of operator engagement in automated task.* Biological Psychology, 1995 (engagement index).
* Goldberger, A. L. et al. *PhysioBank, PhysioToolkit, and PhysioNet.* Circulation, 2000 (CHB-MIT, Sleep-EDF, Siena).
* Bluetooth SIG, *Bluetooth Core Specification 5.x* (LE 2M PHY, data length extension).
* Debener, S. et al. *Unobtrusive ambulatory EEG using a smartphone and flexible printed electrodes around the ear.* Scientific Reports, 2015 (cEEGrid).
