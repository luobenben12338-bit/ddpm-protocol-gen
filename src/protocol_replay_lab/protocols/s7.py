from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from ..common import (
    BitSquareDataset,
    SendSummary,
    TcpSender,
    dump_hex_lines,
    hex_to_bytes,
    load_hex_lines,
    read_pcap_as_hex,
    write_hex_packets_to_pcap,
)
from ..ddpm import SamplingResult, TrainingResult, sample_hex_rows, train_echo_ddpm

DEFAULT_S7_HANDSHAKE: tuple[bytes, ...] = (
    hex_to_bytes('0300001611e00000000100c0010ac1020100c2020102'),
    hex_to_bytes('0300001902f08032010000000000080000f0000001000101e0'),
)

DEFAULT_S7_REQUESTS: tuple[bytes, ...] = (
    hex_to_bytes('0300002102f080320700000100000800080001120411440100ff09000400110000'),
    hex_to_bytes('0300002102f080320700000200000800080001120411440100ff090004001c0000'),
    hex_to_bytes('0300002102f080320700000300000800080001120411440100ff09000401310001'),
)


@dataclass(slots=True)
class S7ExtractionResult:
    source_pcap: Path
    offset_bytes: int
    packet_prefixes: list[str]
    payload_hex_rows: list[str]


@dataclass(slots=True)
class S7GenerationResult:
    training: TrainingResult | None
    sampling: SamplingResult
    output_path: Path


def load_s7_hex_rows(path: str | Path) -> list[str]:
    return load_hex_lines(path)


def build_s7_dataset_from_hex(path: str | Path) -> BitSquareDataset:
    return BitSquareDataset.from_hex_rows(load_s7_hex_rows(path))


def extract_s7_payloads_from_pcap(path: str | Path, *, offset_bytes: int = 88) -> S7ExtractionResult:
    packets = read_pcap_as_hex(path)
    split_at = offset_bytes * 2
    prefixes = [packet[:split_at] for packet in packets]
    payloads = [packet[split_at:] for packet in packets]
    return S7ExtractionResult(
        source_pcap=Path(path),
        offset_bytes=offset_bytes,
        packet_prefixes=prefixes,
        payload_hex_rows=payloads,
    )


def export_s7_payloads_from_pcap(
    input_pcap: str | Path,
    *,
    output_hex: str | Path,
    offset_bytes: int = 88,
) -> S7ExtractionResult:
    result = extract_s7_payloads_from_pcap(input_pcap, offset_bytes=offset_bytes)
    dump_hex_lines((hex_to_bytes(row) for row in result.payload_hex_rows), output_hex)
    return result


def train_s7_model(
    input_path: str | Path,
    *,
    model_out: str | Path,
    preview_dir: str | Path,
    preview_count: int = 4,
    epochs: int = 2,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    timesteps: int = 200,
    dim: int = 16,
    dim_mults: tuple[int, ...] = (1, 2, 4),
    schedule: str = 'linear',
) -> TrainingResult:
    dataset = build_s7_dataset_from_hex(input_path)
    return train_echo_ddpm(
        dataset,
        model_out=model_out,
        preview_dir=preview_dir,
        preview_count=preview_count,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        timesteps=timesteps,
        dim=dim,
        dim_mults=dim_mults,
        schedule=schedule,
    )


def sample_s7_hex(
    input_path: str | Path,
    *,
    model_in: str | Path,
    count: int,
    output_path: str | Path | None = None,
) -> S7GenerationResult:
    dataset = build_s7_dataset_from_hex(input_path)
    sampling = sample_hex_rows(dataset, model_in=model_in, count=count)
    target = Path(output_path) if output_path is not None else Path('outputs/csv/s7_generated.txt')
    dump_hex_lines((hex_to_bytes(row) for row in sampling.hex_rows), target)
    return S7GenerationResult(training=None, sampling=sampling, output_path=target)


def write_generated_s7_pcap(
    source_pcap: str | Path,
    *,
    generated_hex_rows: Sequence[str],
    output_pcap: str | Path,
    offset_bytes: int = 88,
) -> Path:
    extracted = extract_s7_payloads_from_pcap(source_pcap, offset_bytes=offset_bytes)
    if len(generated_hex_rows) > len(extracted.packet_prefixes):
        raise ValueError('生成报文数量超过源 pcap 可拼装的包数量')
    rebuilt_packets = [prefix + payload for prefix, payload in zip(extracted.packet_prefixes, generated_hex_rows)]
    write_hex_packets_to_pcap(rebuilt_packets, output_pcap)
    return Path(output_pcap)


def send_s7_messages(
    host: str,
    messages: Sequence[bytes] | None = None,
    *,
    handshake: Sequence[bytes] | None = None,
    port: int = 102,
    timeout: float = 2.0,
    delay: float = 0.0,
) -> SendSummary:
    intro = tuple(handshake) if handshake is not None else DEFAULT_S7_HANDSHAKE
    payloads = tuple(messages) if messages is not None else DEFAULT_S7_REQUESTS
    sender = TcpSender(host=host, port=port, timeout=timeout)
    return sender.send_messages([*intro, *payloads], reconnect_on_error=True, delay=delay)


def send_s7_hex_rows(
    host: str,
    hex_rows: Sequence[str],
    *,
    handshake: Sequence[bytes] | None = None,
    port: int = 102,
    timeout: float = 2.0,
    delay: float = 0.0,
) -> SendSummary:
    return send_s7_messages(
        host,
        [hex_to_bytes(row) for row in hex_rows],
        handshake=handshake,
        port=port,
        timeout=timeout,
        delay=delay,
    )
