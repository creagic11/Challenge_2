"""Minimal libpcap reader (classic .pcap format, Ethernet/IPv4/UDP).

Written from the public libpcap file-format description. We only need to
locate the UDP payload of each packet; protocol semantics beyond that are not
decoded. Each packet keeps its index so temporal order is preserved.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass, field

import numpy as np

PCAP_MAGIC_US_LE = 0xA1B2C3D4   # microsecond timestamps
PCAP_MAGIC_NS_LE = 0xA1B23C4D   # nanosecond timestamps
ETH_HDR, IPV4_MIN_HDR, UDP_HDR = 14, 20, 8


@dataclass
class PcapCapture:
    """All packets of one capture, in file order."""
    timestamps: np.ndarray                 # float64 seconds, shape (N,)
    payloads: list                         # list[bytes], UDP payloads (None if not UDP/IPv4)
    incl_len: np.ndarray                   # captured length per packet
    orig_len: np.ndarray                   # original length on the wire
    udp_dst_port: np.ndarray               # -1 where not UDP
    reject_reason: list = field(default_factory=list)  # '' if ok


def read_pcap(path: str) -> PcapCapture:
    with open(path, "rb") as fh:
        buf = fh.read()
    if len(buf) < 24:
        raise ValueError(f"{path}: file shorter than pcap global header")
    magic_le = struct.unpack("<I", buf[:4])[0]
    magic_be = struct.unpack(">I", buf[:4])[0]
    if magic_le in (PCAP_MAGIC_US_LE, PCAP_MAGIC_NS_LE):
        endian, magic = "<", magic_le
    elif magic_be in (PCAP_MAGIC_US_LE, PCAP_MAGIC_NS_LE):
        endian, magic = ">", magic_be
    else:
        raise ValueError(f"{path}: not a classic pcap file (magic {buf[:4].hex()})")
    ts_div = 1e9 if magic == PCAP_MAGIC_NS_LE else 1e6
    linktype = struct.unpack(endian + "I", buf[20:24])[0]
    if linktype != 1:
        raise ValueError(f"{path}: unsupported link type {linktype} (expected Ethernet)")

    ts, pls, il, ol, ports, rej = [], [], [], [], [], []
    off = 24
    rec = struct.Struct(endian + "IIII")
    while off + 16 <= len(buf):
        sec, frac, incl, orig = rec.unpack_from(buf, off)
        off += 16
        pkt = buf[off: off + incl]
        off += incl
        ts.append(sec + frac / ts_div)
        il.append(incl)
        ol.append(orig)
        payload, port, reason = _udp_payload(pkt, incl, orig)
        pls.append(payload)
        ports.append(port)
        rej.append(reason)
    return PcapCapture(np.asarray(ts, float), pls, np.asarray(il), np.asarray(ol),
                       np.asarray(ports), rej)


def _udp_payload(pkt: bytes, incl: int, orig: int):
    """Return (payload, udp_dst_port, reject_reason)."""
    if incl < orig or len(pkt) < incl:
        return None, -1, "truncated_packet"
    if len(pkt) < ETH_HDR + IPV4_MIN_HDR + UDP_HDR:
        return None, -1, "too_short"
    ethertype = struct.unpack(">H", pkt[12:14])[0]
    if ethertype != 0x0800:
        return None, -1, "not_ipv4"
    ihl = (pkt[ETH_HDR] & 0x0F) * 4
    if pkt[ETH_HDR + 9] != 17:
        return None, -1, "not_udp"
    u = ETH_HDR + ihl
    dport = struct.unpack(">H", pkt[u + 2:u + 4])[0]
    ulen = struct.unpack(">H", pkt[u + 4:u + 6])[0]
    payload = pkt[u + UDP_HDR: u + ulen]
    if len(payload) != ulen - UDP_HDR:
        return None, dport, "udp_length_mismatch"
    return payload, dport, ""
