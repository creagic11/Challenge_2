"""Nexmon CSI payload parsing and BCM4366c0 packed-CSI reconstruction.

Clean-room implementation from the public Nexmon CSI documentation
(https://github.com/seemoo-lab/nexmon_csi, README "CSI format" section):

UDP payload (port 5500) =
    magic 0x1111 (2 B) | RSSI (1 B) | frame-control (1 B) | source MAC (6 B) |
    sequence number (2 B) | core & spatial-stream (2 B) | chanspec (2 B) |
    chip version (2 B) | CSI (nfft x 4 B)

BCM4366c0 CSI word (32 bit, little-endian), one per subcarrier:
    bit 31      sign of real part
    bits 30..19 real mantissa (12 bit, magnitude)
    bit 18      sign of imaginary part
    bits 17..6  imaginary mantissa (12 bit, magnitude)
    bits 5..0   shared exponent (6-bit two's complement)
    value = (+/-) mantissa * 2**exponent

We reconstruct the exact floating-point value of every component. (The Nexmon
reference tools additionally rescale each packet to a fixed bit width; that is
a per-packet power-of-two scale, which our later gain normalisation removes
anyway. Harmonising quantisation with Room B is handled separately in
amplitude.py.)
"""
from __future__ import annotations

import struct
from dataclasses import dataclass

import numpy as np

NEXMON_MAGIC = 0x1111
HDR_LEN = 18
N_MANT, N_EXP = 12, 6


@dataclass
class NexmonPackets:
    csi: np.ndarray          # complex128, (N, nfft), FFT-bin order as delivered by the chip
    seq: np.ndarray          # uint16 sequence numbers
    core: np.ndarray         # core index
    stream: np.ndarray       # spatial-stream index
    rssi: np.ndarray         # int8
    chanspec: np.ndarray     # uint16
    chip: np.ndarray         # uint16
    src_mac: list            # str per packet
    packet_index: np.ndarray # index into the capture's packet list (temporal order)


def parse_header(payload: bytes) -> dict:
    magic, rssi, fctl = struct.unpack_from("<HbB", payload, 0)
    mac = payload[4:10].hex(":")
    seq, css, chanspec, chip = struct.unpack_from("<HHHH", payload, 10)
    return dict(magic=magic, rssi=rssi, fctl=fctl, mac=mac, seq=seq,
                core=css & 0x7, stream=(css >> 3) & 0x7, chanspec=chanspec, chip=chip)


def unpack_4366c0(words: np.ndarray) -> np.ndarray:
    """Vectorised decode of BCM4366c0 32-bit CSI words -> complex128 (same shape)."""
    w = words.astype(np.uint32)
    mask = (1 << N_MANT) - 1
    exp = (w & ((1 << N_EXP) - 1)).astype(np.int32)
    exp = np.where(exp >= (1 << (N_EXP - 1)), exp - (1 << N_EXP), exp)   # two's complement
    im_mag = ((w >> N_EXP) & mask).astype(np.float64)
    im_sgn = np.where((w >> (N_EXP + N_MANT)) & 1, -1.0, 1.0)
    re_mag = ((w >> (N_EXP + N_MANT + 1)) & mask).astype(np.float64)
    re_sgn = np.where((w >> 31) & 1, -1.0, 1.0)
    scale = np.ldexp(1.0, exp)
    return (re_sgn * re_mag + 1j * im_sgn * im_mag) * scale


def decode_payloads(payloads: list, nfft: int = 256) -> tuple[NexmonPackets, list]:
    """Decode every payload; return decoded packets and a per-packet reject reason list."""
    expected = HDR_LEN + 4 * nfft
    rows, hdrs, idx, reasons = [], [], [], []
    for i, p in enumerate(payloads):
        if p is None:
            reasons.append("no_udp_payload")
            continue
        if len(p) != expected:
            reasons.append(f"payload_len_{len(p)}")
            continue
        h = parse_header(p)
        if h["magic"] != NEXMON_MAGIC:
            reasons.append("bad_magic")
            continue
        rows.append(np.frombuffer(p, dtype="<u4", offset=HDR_LEN, count=nfft))
        hdrs.append(h)
        idx.append(i)
        reasons.append("")
    if rows:
        csi = unpack_4366c0(np.stack(rows))
    else:
        csi = np.zeros((0, nfft), complex)
    pk = NexmonPackets(
        csi=csi,
        seq=np.array([h["seq"] for h in hdrs], dtype=np.uint16),
        core=np.array([h["core"] for h in hdrs], dtype=np.int8),
        stream=np.array([h["stream"] for h in hdrs], dtype=np.int8),
        rssi=np.array([h["rssi"] for h in hdrs], dtype=np.int16),
        chanspec=np.array([h["chanspec"] for h in hdrs], dtype=np.uint16),
        chip=np.array([h["chip"] for h in hdrs], dtype=np.uint16),
        src_mac=[h["mac"] for h in hdrs],
        packet_index=np.array(idx, dtype=np.int32),
    )
    return pk, reasons
