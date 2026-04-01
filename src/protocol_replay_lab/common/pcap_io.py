from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

import dpkt

from .hexio import bytes_to_hex, hex_to_bytes


def read_pcap_packets(path: str | Path) -> List[bytes]:
    packets: List[bytes] = []
    with Path(path).open('rb') as pcap_file:
        reader = dpkt.pcap.Reader(pcap_file)
        for _timestamp, buf in reader:
            packets.append(buf)
    return packets


def read_pcap_as_hex(path: str | Path) -> List[str]:
    return [bytes_to_hex(packet) for packet in read_pcap_packets(path)]


def extract_payload_from_hex_packets(hex_packets: Iterable[str], offset_bytes: int) -> List[str]:
    start_index = offset_bytes * 2
    return [packet[start_index:] for packet in hex_packets]


def extract_payload_from_pcap(path: str | Path, offset_bytes: int) -> List[str]:
    return extract_payload_from_hex_packets(read_pcap_as_hex(path), offset_bytes)


def write_pcap_packets(packets: Iterable[bytes], path: str | Path) -> None:
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open('wb') as pcap_file:
        writer = dpkt.pcap.Writer(pcap_file)
        for packet in packets:
            writer.writepkt(packet)


def write_hex_packets_to_pcap(hex_packets: Iterable[str], path: str | Path) -> None:
    write_pcap_packets((hex_to_bytes(packet) for packet in hex_packets), path)
