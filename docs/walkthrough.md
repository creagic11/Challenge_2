# Walkthrough — every stage in plain words (viva preparation)

This explains *what* each stage does, *why*, and *what evidence* supports the choice.
Numbers refer to files in `results/`.

---

## 0. Ground rules we followed
* **Clean room.** The authors' MATLAB code, `.mat` files, notebook and result folders were never opened.
  The decoder was written from the public Nexmon CSI format description and validated against physics.
* **Validation is only for evaluation.** All fitting uses development captures. The Room-B validation
  split is `sorted()` filenames + `random.Random(42).sample(…, 25)` per activity over the documented
  filename patterns (`configs/splits/roomB_validation.csv`).
* **No metadata shortcuts.** File format only picks the reader. Window ids are a random permutation.
  Routing and recognition see only the (150, 256) numbers.

## 1. Reading Room A (PCAP) — `pcap_reader.py`, `csi_decode.py`
* A pcap file is a 24-byte global header followed by records (16-byte header + packet bytes). We walk the
  records, skip Ethernet (14 B), IPv4 (IHL×4 B) and UDP (8 B) headers and keep the UDP payload.
* Nexmon payload = 18-byte header (magic 0x1111, RSSI, frame control, MAC, sequence number, core/stream,
  chanspec, chip) + 256 × 32-bit words.
* **BCM4366c0 word:** bit 31 = real sign, bits 30–19 = real mantissa, bit 18 = imaginary sign,
  bits 17–6 = imaginary mantissa, bits 5–0 = signed exponent. Value = ±mantissa · 2^exponent.
  `unpack_4366c0` does this with vectorised bit masks; amplitude = |complex value|.
* **How do we know it is right?** (`tests/test_decode.py`) Synthetic words decode to the expected numbers;
  real spectra are smooth across neighbouring subcarriers (much smoother than a shuffled order); the 80 MHz
  guard subcarriers (FFT bins 123–133) are deep nulls; walking varies more over time than standing.

## 2. Reading Room B (CSV) — `ingest.py`
256 integer amplitude columns, one row per observation, same FFT-bin order as Room A (the guard nulls sit at
the same bins), so no re-ordering is needed.

## 3. Quality control — `cleaning.py` (see `docs/data_findings.md`)
Three kinds of observation exist in the raw data:
1. **Valid full-band 80 MHz CSI** — kept.
2. **Sequence-number 0xFFFF reports (Room A)** — half of all packets, strictly alternating with normal ones.
   Their profile is the *same fixed pattern in every capture* (correlation 0.99 across captures, vs 0.22 for
   normal reports), independent of activity, ~34× weaker, without guard nulls → zero information → rejected.
3. **Narrow-band (20 MHz) frames (both rooms)** — energy in only one of the four 20 MHz quarters, the
   rest is noise floor → not a valid 256-subcarrier observation → rejected with one rule for both rooms.
   Important: in Room B these are frequent only in the sit/stand sessions, so keeping them would let a model
   "recognise" sit/stand from a recording artefact.
Also: truncated packets, non-finite / all-zero rows and duplicate rows; the TA package's
`stand65(1).pcap` is a byte-identical copy and is excluded by content hash.

## 4. Common form, harmonisation, windows — `amplitude.py`, `windowing.py`
* Room-A amplitudes have an arbitrary power-of-two scale and are not integers; Room B is integer.
  We rescale each Room-A observation to Room B's median level (measured on Room-B *development* rows) and
  round. This prevents "integer or not" and absolute scale from becoming room cues.
* **Window rule (both rooms):** the capture's valid observations are resampled onto 150 uniform time points
  (timestamps for PCAP, row index for CSV) and rounded. Room-A captures have only ~60 valid observations
  (never 150). The whole capture = one complete activity instance, and the rule is identical for both rooms;
  an ablation (`results/ablation_window_rule.csv`) shows it beats "150 consecutive rows" on target B
  (macro-F1 0.366 vs 0.316) and ties on target A. No padding → capture length cannot leak.
* Provenance: `results/windows_manifest.csv` maps every window to its raw file, SHA-1, valid-observation
  count, method and configuration hash.

## 5. The STRIDE transform — `transform.py`
L = log(A+1) turns multiplicative channel effects into additive ones. Subtracting each observation's median
removes gain (AGC/transmit power). The per-subcarrier median over time is the **static profile s**
(environment + posture); the rest is the **dynamic part D** (body motion).
**Denoising:** body motion changes many subcarriers coherently (low rank); noise does not. We keep only
singular components of D above the Gavish–Donoho optimal threshold (parameter-free). Without this, Room-B
sit/stand (noisy sessions) looked like *more* motion than walking.

## 6. Task 1 — room router (`router.py`)
Logistic regression on the static profile s. Validation accuracy 1.00 at L = 150 and 0.95 from a single
observation. Audit (`results/task1_audit.csv`): motion energy and a temporal-texture probe also separate the
rooms, i.e. rooms also differ by session properties (sampling rate, noise level) — Known Limitation 1.

## 7. Task 2 — self-supervised encoder (`models/encoder.py`, `ssl.py`)
* 121 k-parameter 1-D CNN: 1×1 "spectral mixing" conv → 4 dilated depthwise-separable temporal blocks →
  mean‖std pooling → 128-d embedding.
* SimCLR-style loss; positives = two random crops of the **same capture** (only capture identity is used —
  no activity or room labels). Augmentations: random time span (time-scale/shift), subcarrier-band mask,
  time mask, jitter. **No time reversal** (sit-down reversed is stand-up).
* Input = denoised dynamic part (STRIDE). Ablation: raw log amplitude (S-raw).
* Evaluation: frozen encoder + logistic-regression probe on development windows → validation; compared
  with handcrafted features (H / Hd) and the same network with random weights (R).

## 8. Task 3 — few-label cross-room adaptation (`fewshot.py`)
* Router chooses the room → that room's label space and normalisation statistics.
* Features are standardised per room (statistics from unlabelled development windows of that room).
  The SAPT variant estimates the statistics only from the shared classes (labels in the source; predicted
  probabilities in the target) to avoid label-shift bias.
* Classifier: logistic regression on aligned source shared-class windows + the K target support captures
  (K = distinct captures per exclusive class; nested supports; 10 episodes). Variants: prototypes, balanced
  optimal transport (Sinkhorn) on the unlabelled target development pool, self-training.
* Routing study: predicted vs oracle room vs no routing (union label space, pooled statistics).

## 9. Likely viva questions
* *Why reject the 0xFFFF reports?* Same fixed pattern in all captures, activity-independent, no guard nulls.
* *Why not 150 consecutive observations?* Room A never has 150 valid observations; for Room B we tested it:
  full-capture windows are better on target B and equal on target A.
* *Is the router "too good"?* Yes — rooms differ in environment **and** session (rate, noise). We show which
  cues separate them; single-observation static profiles alone give 95 %.
* *Where did validation get used?* Only for reported numbers and for choosing among reported variants
  (declared). Never for fitting, normalisation, supports or pseudo-labels.
* *Why is cross-room SIT/STAND hard?* Both are quasi-static; posture lives mostly in the static profile,
  which is room-specific by construction. Walk vs static transfers; sit vs stand largely does not.

## 10. Results to remember (validation)
* Task 1: room accuracy 1.000 (L = 150), 0.949 from one observation; audit shows session cues too.
* Task 2 (macro-F1, Room A / Room B): handcrafted 0.752 / 0.840, random CNN 0.553 / 0.887,
  SSL raw 0.872 / 0.987, STRIDE-SSL (denoised dynamics) 0.669 / 0.931.
* Task 3 headline (room-wise alignment + LR on denoised handcrafted features, K = 20): 0.341 (B->A),
  0.418 (A->B); routing adds about +0.07 / +0.09 mean macro-F1; denoising raises shared-class F1
  from 0.05 to 0.17 (A) and 0.04 to 0.27 (B).
* Honest negatives: SAPT's shared-anchored alignment and balanced OT help shared classes on target A
  but lose on the combined metric; removing static-profile features hurt; SSL does not beat denoised
  handcrafted features across rooms; WALK and STAND swap across rooms.

## 11. Demo checklist
1. `pip install -r requirements.txt` (+ CPU torch).  2. `python -m pytest -q tests`.
3. `python run_episode.py --episode <TA folder> --out predictions.csv` (raw .pcap/.csv or window .npy).
4. If the artefact cannot be loaded: `python scripts/prepare_deployment.py` (a few minutes).
