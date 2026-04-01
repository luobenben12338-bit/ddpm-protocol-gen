from __future__ import annotations

import argparse
import json
from typing import Sequence

from .common import load_hex_lines
from .protocols import (
    export_s7_payloads_from_pcap,
    sample_modbus_hex,
    sample_s7_hex,
    send_modbus_hex_rows,
    send_s7_hex_rows,
    train_modbus_model,
    train_s7_model,
    write_generated_s7_pcap,
)


def _parse_dim_mults(value: str) -> tuple[int, ...]:
    parts = [item.strip() for item in value.split(',') if item.strip()]
    if not parts:
        raise argparse.ArgumentTypeError('dim_mults 不能为空')
    try:
        return tuple(int(item) for item in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError('dim_mults 必须是逗号分隔的整数列表') from exc


def _add_training_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument('--preview-count', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--learning-rate', type=float, default=1e-3)
    parser.add_argument('--timesteps', type=int, default=200)
    parser.add_argument('--dim', type=int, default=16)
    parser.add_argument('--dim-mults', type=_parse_dim_mults, default=(1, 2, 4))
    parser.add_argument('--schedule', choices=['linear', 'cosine', 'quadratic', 'sigmoid'], default='linear')


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog='protocol-replay-lab')
    subparsers = parser.add_subparsers(dest='protocol', required=True)

    modbus = subparsers.add_parser('modbus', help='Modbus 相关命令')
    modbus_sub = modbus.add_subparsers(dest='action', required=True)

    modbus_train = modbus_sub.add_parser('train', help='训练 Modbus 模型')
    modbus_train.add_argument('--input', required=True)
    modbus_train.add_argument('--model-out', required=True)
    modbus_train.add_argument('--preview-dir', required=True)
    _add_training_args(modbus_train)

    modbus_sample = modbus_sub.add_parser('sample', help='采样生成 Modbus 报文')
    modbus_sample.add_argument('--input', required=True)
    modbus_sample.add_argument('--model-in', required=True)
    modbus_sample.add_argument('--count', type=int, required=True)
    modbus_sample.add_argument('--output', required=True)

    modbus_send = modbus_sub.add_parser('send', help='发送 Modbus 报文')
    modbus_send.add_argument('--host', required=True)
    modbus_send.add_argument('--input', required=True)
    modbus_send.add_argument('--port', type=int, default=502)
    modbus_send.add_argument('--timeout', type=float, default=2.0)
    modbus_send.add_argument('--delay', type=float, default=0.0)

    s7 = subparsers.add_parser('s7', help='S7 相关命令')
    s7_sub = s7.add_subparsers(dest='action', required=True)

    s7_extract = s7_sub.add_parser('extract-pcap', help='从 pcap 提取 S7 payload')
    s7_extract.add_argument('--input', required=True)
    s7_extract.add_argument('--output', required=True)
    s7_extract.add_argument('--offset', type=int, default=88)

    s7_train = s7_sub.add_parser('train', help='训练 S7 模型')
    s7_train.add_argument('--input', required=True)
    s7_train.add_argument('--model-out', required=True)
    s7_train.add_argument('--preview-dir', required=True)
    _add_training_args(s7_train)

    s7_sample = s7_sub.add_parser('sample', help='采样生成 S7 payload')
    s7_sample.add_argument('--input', required=True)
    s7_sample.add_argument('--model-in', required=True)
    s7_sample.add_argument('--count', type=int, required=True)
    s7_sample.add_argument('--output', required=True)

    s7_write = s7_sub.add_parser('write-pcap', help='将生成的 S7 payload 回写为 pcap')
    s7_write.add_argument('--source-pcap', required=True)
    s7_write.add_argument('--input', required=True)
    s7_write.add_argument('--output', required=True)
    s7_write.add_argument('--offset', type=int, default=88)

    s7_send = s7_sub.add_parser('send', help='发送 S7 payload')
    s7_send.add_argument('--host', required=True)
    s7_send.add_argument('--input', required=True)
    s7_send.add_argument('--port', type=int, default=102)
    s7_send.add_argument('--timeout', type=float, default=2.0)
    s7_send.add_argument('--delay', type=float, default=0.0)

    return parser


def _print_json(data: dict) -> None:
    print(json.dumps(data, ensure_ascii=False, indent=2))


def _summary_to_dict(summary) -> dict:
    return {
        'host': summary.host,
        'port': summary.port,
        'success': summary.success,
        'sent_count': summary.sent_count,
        'total_count': summary.total_count,
        'elapsed_seconds': summary.elapsed_seconds,
        'failed_indices': summary.failed_indices,
        'responses': [
            {
                'request_hex': exchange.request.hex(),
                'response_hex': exchange.response.hex() if exchange.response is not None else None,
                'error': exchange.error,
            }
            for exchange in summary.exchanges
        ],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.protocol == 'modbus' and args.action == 'train':
        result = train_modbus_model(
            args.input,
            model_out=args.model_out,
            preview_dir=args.preview_dir,
            preview_count=args.preview_count,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            timesteps=args.timesteps,
            dim=args.dim,
            dim_mults=args.dim_mults,
            schedule=args.schedule,
        )
        _print_json(
            {
                'model_path': str(result.model_path),
                'preview_dir': str(result.preview_dir),
                'sample_count': result.sample_count,
                'side': result.side,
                'epochs': result.epochs,
                'device': result.device,
                'loss_history_tail': result.loss_history[-10:],
            }
        )
        return 0

    if args.protocol == 'modbus' and args.action == 'sample':
        result = sample_modbus_hex(
            args.input,
            model_in=args.model_in,
            count=args.count,
            output_path=args.output,
        )
        _print_json({'count': result.sampling.count, 'output_path': str(result.output_path), 'hex_rows': result.sampling.hex_rows})
        return 0

    if args.protocol == 'modbus' and args.action == 'send':
        summary = send_modbus_hex_rows(
            args.host,
            load_hex_lines(args.input),
            port=args.port,
            timeout=args.timeout,
            delay=args.delay,
        )
        _print_json(_summary_to_dict(summary))
        return 0

    if args.protocol == 's7' and args.action == 'extract-pcap':
        result = export_s7_payloads_from_pcap(args.input, output_hex=args.output, offset_bytes=args.offset)
        _print_json(
            {
                'source_pcap': str(result.source_pcap),
                'offset_bytes': result.offset_bytes,
                'packet_count': len(result.payload_hex_rows),
                'output_path': args.output,
            }
        )
        return 0

    if args.protocol == 's7' and args.action == 'train':
        result = train_s7_model(
            args.input,
            model_out=args.model_out,
            preview_dir=args.preview_dir,
            preview_count=args.preview_count,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            timesteps=args.timesteps,
            dim=args.dim,
            dim_mults=args.dim_mults,
            schedule=args.schedule,
        )
        _print_json(
            {
                'model_path': str(result.model_path),
                'preview_dir': str(result.preview_dir),
                'sample_count': result.sample_count,
                'side': result.side,
                'epochs': result.epochs,
                'device': result.device,
                'loss_history_tail': result.loss_history[-10:],
            }
        )
        return 0

    if args.protocol == 's7' and args.action == 'sample':
        result = sample_s7_hex(
            args.input,
            model_in=args.model_in,
            count=args.count,
            output_path=args.output,
        )
        _print_json({'count': result.sampling.count, 'output_path': str(result.output_path), 'hex_rows': result.sampling.hex_rows})
        return 0

    if args.protocol == 's7' and args.action == 'write-pcap':
        output_path = write_generated_s7_pcap(
            args.source_pcap,
            generated_hex_rows=load_hex_lines(args.input),
            output_pcap=args.output,
            offset_bytes=args.offset,
        )
        _print_json({'output_pcap': str(output_path)})
        return 0

    if args.protocol == 's7' and args.action == 'send':
        summary = send_s7_hex_rows(
            args.host,
            load_hex_lines(args.input),
            port=args.port,
            timeout=args.timeout,
            delay=args.delay,
        )
        _print_json(_summary_to_dict(summary))
        return 0

    parser.error('未识别的命令')
    return 1


if __name__ == '__main__':
    raise SystemExit(main())
