# Data findings (development captures only)

Every number here is produced by `scripts/01_data_inspection.py` (`results/inspection_stats.csv`),
`scripts/10_build_intermediate.py` (`results/qc_*.csv`) or `scripts/12_window_rule_ablation.py`. Design decisions were taken on **development** captures only.

## 1. Room A (Experiment 2, PCAP) structure
* Classic pcap, Ethernet/IPv4/UDP, one source MAC, Nexmon magic `0x1111`, chanspec `0xe02a`
  (80 MHz), chip field `0x006a`, core/stream always 0. 256 x 32-bit CSI words per packet.
* ~3.9 s per capture. Reports arrive in pairs about 0.3 ms apart, with pairs about 30–35 ms apart
  (so "30ms" is a packet interval, not a duration).
* One reader-level reject: 2 truncated packets and 2 packets without a UDP payload.

## 2. Three kinds of observation in the raw data
| Kind | How identified | Evidence that it is (in)valid |
|---|---|---|
| Full-band 80 MHz CSI | >= 3 of the four 20 MHz quarters carry energy; guard bins 123-133 deep | smooth across subcarriers; walk >> stand temporal variance |
| `0xFFFF` reports (Room A only) | Nexmon sequence field = 0xFFFF; strictly alternate with normal reports | energy confined to the lowest 20 MHz quarter (like the narrow-band frames), no guard nulls; the band mask makes the profile nearly identical in every capture (corr. 0.99 vs 0.22 for normal reports) and the median data-subcarrier amplitude ~34x lower → not valid 80 MHz observations |
| Narrow-band (20 MHz) frames (both rooms) | energy in only one quarter (Room A: subcarriers -128..-65; Room B: other quarters) | the other three quarters are at the noise floor (~5 log units lower) -> not a valid 256-subcarrier observation |

Rejected (QC, both splits, all captures): Room A 118,013 seq.-0xFFFF + 47,359 other narrow-band of
234,239 decoded; Room B 39,196 narrow-band of 600,500 rows. No NaN, zero or duplicate rows.

**Why this matters beyond cleanliness.** In Room B the narrow-band share depends on the class:
sit 12.7 %, stand 14.0 %, lying 6.8 %, walk 1.4 %, empty 0.1 % of rows (development captures). These frames look
completely different from full-band CSI, so a model would learn "narrow-band rate -> sit/stand", a
session artefact that cannot transfer across rooms. Removing them by a physical criterion that is
identical for both rooms eliminates that shortcut.

## 3. Valid observations per capture and the window rule
Room A: median 57-61 per capture (min 1, max 128); **no capture reaches 150**. Room B: 319-500.
Official window rule (identical for both rooms, `window.mode: full`): the capture's whole
valid-observation sequence is resampled onto 150 uniform time points (packet timestamps for PCAP,
row index for CSV) and rounded to the integer amplitude grid. No padding rows are created, so capture
length cannot leak through repeated rows, and both rooms undergo the same operation.
*Why "full" rather than "150 consecutive rows when available":* with consecutive rows, a Room-B
window would cover a much shorter stretch of time than a Room-A window (see §6). With the
consecutive rule, handcrafted cross-room transfer of the shared classes was essentially zero
(results/task3 runs before the change). A capture is one recorded activity instance, so covering
the whole instance is the physically consistent choice; the choice uses no labels.

## 4. Subcarrier order and scale
Both rooms are in FFT-bin order (guard nulls at bins 123-133, DC next to bin 0), so no re-ordering is
needed. Room B is integer amplitude with per-row maxima ~800-1900. Room-A amplitudes are mapped to
the same convention (median data-subcarrier level 297.5, measured on Room-B dev rows, then
rounded), so quantisation and scale are not room cues.

## 5. Duplicates
`stand/stand65(1).pcap` in the TA package is byte-identical to `stand65.pcap` and is excluded
(content hash). No other duplicate captures exist; the four Room-B files outside the documented
name patterns (`Empty_11`, `Lying_0`, `Sitting_0`, `Standing_0`) are not duplicates (max corr 0.41)
and belong to development.

## 6. Sampling-rate mismatch
Room A's valid full-band observations arrive at roughly 15 Hz (about 60 per 3.9 s capture). The lag-1
autocorrelation of the top-3 motion components of WALK is 0.32 in Room A (native valid observations),
0.40 in Room B, and 0.31 when Room B is decimated by 4. So Room B is sampled roughly 4x faster, and a
full Room-B capture (500 rows) spans about twice the time of a Room-A capture. The exact rate cannot be
recovered (no timestamps); this estimate is descriptive only and is not used by any model.

## 7. Session-dependent noise floor and low-rank denoising
In Room B, the naive motion energy (temporal std of the dynamic part) is *inverted*: sit (0.79)
and stand (0.77) score above walk (0.65), because the sit/stand sessions are noisier (the same
low-SNR sessions that contain many narrow-band frames). Body motion changes many subcarriers
coherently (low rank), while noise is spread over all components. Keeping only the singular
components above the Gavish-Donoho optimal hard threshold (unknown-noise version, Gavish &
Donoho 2014; parameter-free, per window) restores the physical ordering (walk highest in both
rooms). In the final Task-3 runs (room-align+LR, `results/task3_runs.csv`), handcrafted features on
the denoised dynamics raise shared-class F1 from 0.051 to 0.169 (target A) and from 0.035 to 0.273
(target B) compared with the same features on the raw dynamic part.

## 8. Activity zones (paper, Schäfer et al. 2021)
In Experiment 2, WALK, the SIT/STAND/SIT-DOWN/STAND-UP group and FALL were performed in
different areas of the room, so in Room A, location and activity are partly confounded.
