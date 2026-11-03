# Data findings (development captures only)

Every number here is produced by `scripts/10_build_intermediate.py` (`results/qc_*.csv`) or by the
exploratory checks listed. Design decisions were taken on **development** captures only.

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
| `0xFFFF` reports (Room A only) | Nexmon sequence field = 0xFFFF; strictly alternate with normal reports | the **same fixed profile in every capture** (cross-capture corr. 0.98 vs 0.39 for real CSI), activity-independent temporal variance, no guard nulls, ~30x lower amplitude -> zero-information |
| Narrow-band (20 MHz) frames (both rooms) | energy in only one quarter (Room A: subcarriers -128..-65; Room B: other quarters) | the other three quarters are at the noise floor (~5 log units lower) -> not a valid 256-subcarrier observation |

Rejected (QC, both splits, all captures): Room A 118,013 fixed-pattern + 47,359 narrow-band of
234,239 decoded; Room B 39,196 narrow-band of 600,500 rows. No NaN, zero or duplicate rows.

**Why this matters beyond cleanliness.** In Room B the narrow-band share depends on the class:
about 15–18 % of rows for sit/stand but ~0.2–2 % for empty/lying/walk. These frames look
completely different from full-band CSI, so a model would learn "narrow-band rate -> sit/stand", a
session artefact that cannot transfer across rooms. Removing them by a physical criterion that is
identical for both rooms eliminates that shortcut.

## 3. Valid observations per capture
Room A: median 57-61 per capture (min 1, max 128); **no capture reaches 150**. Room B: 319-500.
The challenge's 150-observation window is therefore built with one rule for both rooms
(`windowing.py`): 150 consecutive valid observations when available (all Room-B captures), otherwise
the whole capture's valid sequence resampled onto 150 uniform time points using packet timestamps
(all Room-A captures). No padding rows are created, so capture length cannot leak through repeats.

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
Room A's valid full-band observations arrive at roughly 15 Hz. Walking decorrelates Room-A CSI within
about one report interval, but takes about 24 rows in Room B, so Room B is sampled much faster
(likely more than 10x). The exact Room-B rate cannot be recovered (no timestamps), so the models
use rate-robust features (statistics pooling, time-scale augmentation), and the mismatch is reported
as a limitation.

## 7. Activity zones (paper, Schäfer et al. 2021)
In Experiment 2, WALK, the SIT/STAND/SIT-DOWN/STAND-UP group and FALL were performed in
different areas of the room, so in Room A, location and activity are partly confounded.
