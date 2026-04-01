from __future__ import annotations

from pathlib import Path
from typing import Iterable, List


def normalize_hex(hex_text: str) -> str:
    cleaned = ''.join(hex_text.strip().split()).lower()
    if cleaned.startswith('0x'):
        cleaned = cleaned[2:]
    if len(cleaned) % 2 != 0:
        raise ValueError(f'hex 字符串长度必须为偶数: {hex_text!r}')
    return cleaned


def hex_to_bytes(hex_text: str) -> bytes:
    return bytes.fromhex(normalize_hex(hex_text))


def bytes_to_hex(data: bytes) -> str:
    return data.hex()


def load_hex_lines(path: str | Path) -> List[str]:
    file_path = Path(path)
    lines = []
    for line in file_path.read_text(encoding='utf-8').splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith('#'):
            continue
        lines.append(normalize_hex(stripped))
    return lines


def load_hex_bytes(path: str | Path) -> List[bytes]:
    return [hex_to_bytes(line) for line in load_hex_lines(path)]


def dump_hex_lines(messages: Iterable[bytes], path: str | Path) -> None:
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    content = '\n'.join(bytes_to_hex(message) for message in messages)
    file_path.write_text(content + ('\n' if content else ''), encoding='utf-8')
