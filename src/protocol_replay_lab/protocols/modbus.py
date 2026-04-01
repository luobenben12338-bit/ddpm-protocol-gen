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
    load_hex_bytes,
    load_hex_lines,
)
from ..ddpm import SamplingResult, TrainingResult, sample_hex_rows, train_echo_ddpm

DEFAULT_MODBUS_REQUESTS: tuple[bytes, ...] = (
    hex_to_bytes('000100000006010300000002'),
    hex_to_bytes('000200000006010300020002'),
    hex_to_bytes('000300000006010600010001'),
)


@dataclass(slots=True)
class ModbusGenerationResult:
    training: TrainingResult | None
    sampling: SamplingResult
    output_path: Path


def load_modbus_messages(path: str | Path) -> list[bytes]:
    return load_hex_bytes(path)


def load_modbus_hex_rows(path: str | Path) -> list[str]:
    return load_hex_lines(path)


def build_modbus_dataset(path: str | Path) -> BitSquareDataset:
    return BitSquareDataset.from_hex_rows(load_modbus_hex_rows(path))


def train_modbus_model(
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
    dataset = build_modbus_dataset(input_path)
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


def sample_modbus_hex(
    input_path: str | Path,
    *,
    model_in: str | Path,
    count: int,
    output_path: str | Path | None = None,
) -> ModbusGenerationResult:
    dataset = build_modbus_dataset(input_path)
    sampling = sample_hex_rows(dataset, model_in=model_in, count=count)
    target = Path(output_path) if output_path is not None else Path('outputs/csv/modbus_generated.txt')
    dump_hex_lines((hex_to_bytes(row) for row in sampling.hex_rows), target)
    return ModbusGenerationResult(training=None, sampling=sampling, output_path=target)


def send_modbus_messages(
    host: str,
    messages: Sequence[bytes] | None = None,
    *,
    port: int = 502,
    timeout: float = 2.0,
    delay: float = 0.0,
) -> SendSummary:
    payloads: Sequence[bytes] = messages if messages is not None else DEFAULT_MODBUS_REQUESTS
    sender = TcpSender(host=host, port=port, timeout=timeout)
    return sender.send_messages(payloads, reconnect_on_error=True, delay=delay)


def send_modbus_hex_rows(
    host: str,
    hex_rows: Sequence[str],
    *,
    port: int = 502,
    timeout: float = 2.0,
    delay: float = 0.0,
) -> SendSummary:
    return send_modbus_messages(
        host,
        [hex_to_bytes(row) for row in hex_rows],
        port=port,
        timeout=timeout,
        delay=delay,
    )


def iter_modbus_message_hex(messages: Sequence[bytes]) -> list[str]:
    return [message.hex() for message in messages]
