# PulseLink results

Data: SC4001E0-PSG.edf, SC4002E0-PSG.edf, SC4011E0-PSG.edf, SC4012E0-PSG.edf, SC4021E0-PSG.edf, SC4022E0-PSG.edf, SC4031E0-PSG.edf, SC4032E0-PSG.edf. Test segments: 120 x 30 s, 2 channels at 100 Hz.

Sleep-stage classifier on original test EEG: accuracy 0.892, kappa 0.820.

## Codec benchmark (ideal channel)

| method | compression_ratio | prdn | snr_db | psd_diff_db | bp_err_mean | stage_accuracy | encode_mops | transmitter_mj | transmitter_saving_pct |
|---|---|---|---|---|---|---|---|---|---|
| raw | 0.999 | 0.0459 | 74.6 | 0.00276 | 0.000468 | 0.892 | 0.012 | 5.22 | 0 |
| dwt(k=0.05,b=6) | 26.8 | 55.4 | 5.39 | 10.3 | 0.567 | 0.758 | 0.337 | 0.229 | 95.6 |
| dwt(k=0.1,b=8) | 13.7 | 40.4 | 8.37 | 4.78 | 0.363 | 0.825 | 0.352 | 0.419 | 92 |
| dwt(k=0.2,b=8) | 8.09 | 25.4 | 12.8 | 2.37 | 0.165 | 0.867 | 0.376 | 0.688 | 86.8 |
| dwt(k=0.3,b=8) | 5.91 | 17.5 | 16.2 | 1.48 | 0.0904 | 0.875 | 0.401 | 0.926 | 82.3 |
| dwt(k=0.5,b=10) | 2.77 | 8.92 | 22.3 | 0.562 | 0.0295 | 0.883 | 0.48 | 1.93 | 63 |
| pulselink(k=2,b=3) | 75 | 79.3 | 2.24 | 3.71 | 0.481 | 0.7 | 4.84 | 0.556 | 89.3 |
| pulselink(k=2,b=4) | 58 | 77.5 | 2.47 | 3.78 | 0.496 | 0.717 | 4.84 | 0.575 | 89 |
| pulselink(k=2,b=6) | 39.9 | 77.1 | 2.53 | 3.8 | 0.501 | 0.708 | 4.85 | 0.622 | 88.1 |
| pulselink(k=4,b=3) | 39.9 | 66 | 4.05 | 2.96 | 0.362 | 0.725 | 4.85 | 0.622 | 88.1 |
| pulselink(k=4,b=4) | 30.5 | 63 | 4.58 | 3 | 0.372 | 0.725 | 4.85 | 0.659 | 87.4 |
| pulselink(k=4,b=6) | 20.6 | 62.2 | 4.72 | 3.01 | 0.378 | 0.717 | 4.86 | 0.744 | 85.7 |
| pulselink(k=8,b=3) | 20.6 | 46.4 | 7.24 | 2.07 | 0.182 | 0.817 | 4.86 | 0.744 | 85.7 |
| pulselink(k=8,b=4) | 16 | 41.6 | 8.56 | 2.06 | 0.175 | 0.85 | 4.87 | 0.82 | 84.3 |
| pulselink(k=8,b=6) | 10.5 | 40.2 | 9.04 | 2.05 | 0.18 | 0.858 | 4.88 | 0.988 | 81.1 |
| pulselink(k=16,b=3) | 10.6 | 34.8 | 9.76 | 1.43 | 0.108 | 0.883 | 4.88 | 0.984 | 81.1 |
| pulselink(k=16,b=4) | 8.49 | 29.3 | 11.8 | 1.37 | 0.0909 | 0.9 | 4.9 | 1.1 | 78.8 |
| pulselink(k=16,b=6) | 5.28 | 27.5 | 12.7 | 1.36 | 0.0932 | 0.9 | 4.93 | 1.48 | 71.6 |


## Packet loss, no retransmission

| loss_rate | method | compression_ratio | prdn_mean_all | segments_lost | reliable_fraction | flag_recall | stage_accuracy | transmitter_energy_per_segment_mj |
|---|---|---|---|---|---|---|---|---|
| 0 | raw | 0.999 | 0.0459 | 0 | 1 | 1 | 0.892 | 5.22 |
| 0 | dwt | 13.7 | 40.4 | 0 | 1 | 0 | 0.825 | 0.419 |
| 0 | pulselink | 16 | 41.6 | 0 | 1 | 0 | 0.85 | 0.819 |
| 0 | adaptive | 1.67 | 4.42 | 0 | 1 | 1 | 0.875 | 3.18 |
| 0.02 | raw | 0.999 | 13.1 | 1 | 0.4 | 1 | 0.875 | 5.22 |
| 0.02 | dwt | 13.7 | 46.8 | 13 | 0.892 | 0.14 | 0.75 | 0.419 |
| 0.02 | pulselink | 16 | 48.1 | 13 | 0.892 | 0.159 | 0.758 | 0.819 |
| 0.02 | adaptive | 4.19 | 31.6 | 20 | 0.692 | 1 | 0.742 | 1.33 |
| 0.05 | raw | 0.999 | 29.6 | 5 | 0.0417 | 1 | 0.875 | 5.22 |
| 0.05 | dwt | 13.7 | 50.4 | 19 | 0.842 | 0.198 | 0.725 | 0.419 |
| 0.05 | pulselink | 16 | 50.5 | 18 | 0.85 | 0.22 | 0.733 | 0.819 |
| 0.05 | adaptive | 7.9 | 49.9 | 31 | 0.292 | 0.963 | 0.692 | 0.936 |
| 0.1 | raw | 0.999 | 45.5 | 10 | 0 | 1 | 0.85 | 5.22 |
| 0.1 | dwt | 13.7 | 56.5 | 33 | 0.725 | 0.333 | 0.625 | 0.419 |
| 0.1 | pulselink | 16 | 55.6 | 32 | 0.733 | 0.381 | 0.633 | 0.819 |
| 0.1 | adaptive | 13.7 | 58.2 | 30 | 0.283 | 0.966 | 0.625 | 0.629 |


## Packet loss, ARQ with 2 retries

| loss_rate | method | compression_ratio | prdn_mean_all | segments_lost | retransmissions | stage_accuracy | transmitter_energy_per_segment_mj |
|---|---|---|---|---|---|---|---|
| 0 | raw | 0.999 | 0.0459 | 0 | 0 | 0.892 | 5.22 |
| 0 | dwt | 13.7 | 40.4 | 0 | 0 | 0.825 | 0.419 |
| 0 | pulselink | 16 | 41.6 | 0 | 0 | 0.85 | 0.819 |
| 0 | adaptive | 1.67 | 4.42 | 0 | 0 | 0.875 | 3.18 |
| 0.02 | raw | 0.999 | 0.0459 | 0 | 116 | 0.892 | 5.32 |
| 0.02 | dwt | 13.7 | 40.4 | 0 | 13 | 0.825 | 0.43 |
| 0.02 | pulselink | 16 | 41.6 | 0 | 13 | 0.85 | 0.829 |
| 0.02 | adaptive | 1.67 | 4.42 | 0 | 66 | 0.875 | 3.24 |
| 0.05 | raw | 0.999 | 0.208 | 0 | 321 | 0.892 | 5.5 |
| 0.05 | dwt | 13.7 | 40.4 | 0 | 21 | 0.825 | 0.436 |
| 0.05 | pulselink | 16 | 41.6 | 0 | 19 | 0.85 | 0.831 |
| 0.05 | adaptive | 1.67 | 4.42 | 0 | 197 | 0.875 | 3.35 |
| 0.1 | raw | 0.999 | 0.931 | 0 | 704 | 0.892 | 5.83 |
| 0.1 | dwt | 13.7 | 40.4 | 0 | 43 | 0.825 | 0.453 |
| 0.1 | pulselink | 16 | 41.6 | 0 | 39 | 0.85 | 0.848 |
| 0.1 | adaptive | 1.67 | 5.15 | 0 | 438 | 0.875 | 3.56 |


## Bandwidth limits

| bandwidth_bps | method | compression_ratio | prdn_mean_all | segments_lost | buffer_overflow | max_latency_s | stage_accuracy |
|---|---|---|---|---|---|---|---|
| 800 | raw | 0.999 | 73.2 | 0 | 3532 | 2,560 | 0.4 |
| 800 | dwt | 13.7 | 40.4 | 0 | 0 | 10.3 | 0.825 |
| 800 | pulselink | 16 | 41.6 | 0 | 0 | 8.08 | 0.85 |
| 800 | adaptive | 5.3 | 17.9 | 0 | 0 | 52.9 | 0.858 |
| 1600 | raw | 0.999 | 58.6 | 0 | 2095 | 1,280 | 0.533 |
| 1600 | dwt | 13.7 | 40.4 | 0 | 0 | 5.16 | 0.825 |
| 1600 | pulselink | 16 | 41.6 | 0 | 0 | 4.07 | 0.85 |
| 1600 | adaptive | 3.43 | 11.5 | 0 | 0 | 23.5 | 0.858 |
| 3200 | raw | 0.999 | 0.0459 | 0 | 0 | 155 | 0.892 |
| 3200 | dwt | 13.7 | 40.4 | 0 | 0 | 2.59 | 0.825 |
| 3200 | pulselink | 16 | 41.6 | 0 | 0 | 2.06 | 0.85 |
| 3200 | adaptive | 2.39 | 7.77 | 0 | 0 | 37.3 | 0.875 |
| 6400 | raw | 0.999 | 0.0459 | 0 | 0 | 15.6 | 0.892 |
| 6400 | dwt | 13.7 | 40.4 | 0 | 0 | 1.31 | 0.825 |
| 6400 | pulselink | 16 | 41.6 | 0 | 0 | 1.05 | 0.85 |
| 6400 | adaptive | 1.67 | 4.42 | 0 | 0 | 15.6 | 0.875 |
| 64000 | raw | 0.999 | 0.0459 | 0 | 0 | 1.6 | 0.892 |
| 64000 | dwt | 13.7 | 40.4 | 0 | 0 | 0.168 | 0.825 |
| 64000 | pulselink | 16 | 41.6 | 0 | 0 | 0.157 | 0.85 |
| 64000 | adaptive | 1.67 | 4.42 | 0 | 0 | 1.59 | 0.875 |


## Outage with and without buffering

| buffer_bytes | method | segments_lost | buffered_by_outage | buffer_overflow | max_latency_s | prdn_mean_all |
|---|---|---|---|---|---|---|
| 2000 | raw | 30 | 10 | 5278 | 900 | 96.9 |
| 2000 | adaptive | 34 | 11 | 58 | 900 | 69.8 |
| 256000 | raw | 10 | 1031 | 519 | 902 | 9.1 |
| 256000 | adaptive | 4 | 1031 | 36 | 902 | 29.4 |


Energy numbers come from the transparent model in `evaluation/energy_model.py` with the assumptions in `config.EnergyConfig`; they are estimates, not measurements.
