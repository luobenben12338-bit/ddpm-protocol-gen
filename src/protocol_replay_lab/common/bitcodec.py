from __future__ import annotations

import math
from typing import Iterable, Sequence

HEX_TO_BITS = {
    '0': '0000',
    '1': '0001',
    '2': '0010',
    '3': '0011',
    '4': '0100',
    '5': '0101',
    '6': '0110',
    '7': '0111',
    '8': '1000',
    '9': '1001',
    'a': '1010',
    'b': '1011',
    'c': '1100',
    'd': '1101',
    'e': '1110',
    'f': '1111',
}
BITS_TO_HEX = {value: key for key, value in HEX_TO_BITS.items()}


def hex_to_bitstring(hex_text: str) -> str:
    cleaned = ''.join(hex_text.strip().split()).lower()
    return ''.join(HEX_TO_BITS[ch] for ch in cleaned)


def bitstring_to_hex(bit_text: str) -> str:
    cleaned = ''.join(bit_text.strip().split())
    if len(cleaned) % 4 != 0:
        raise ValueError('bit 字符串长度必须是 4 的倍数')
    return ''.join(BITS_TO_HEX[cleaned[index:index + 4]] for index in range(0, len(cleaned), 4))


def square_side_for_bits(bit_length: int) -> int:
    if bit_length <= 0:
        raise ValueError('bit_length 必须大于 0')
    side = math.ceil(math.sqrt(bit_length))
    if side % 4 != 0:
        side += 4 - (side % 4)
    return side


def pad_bitstring_to_square(bit_text: str, side: int | None = None) -> tuple[str, int]:
    cleaned = ''.join(bit_text.strip().split())
    actual_side = side or square_side_for_bits(len(cleaned))
    total_bits = actual_side * actual_side
    if len(cleaned) > total_bits:
        raise ValueError('给定 side 太小，无法容纳 bit 数据')
    return cleaned.ljust(total_bits, '0'), actual_side


def hex_to_square_bitstring(hex_text: str) -> tuple[str, int, int]:
    bit_text = hex_to_bitstring(hex_text)
    padded, side = pad_bitstring_to_square(bit_text)
    return padded, side, len(bit_text)


def restore_hex_from_square_bitstring(bit_text: str, original_bit_length: int) -> str:
    trimmed = ''.join(bit_text.strip().split())[:original_bit_length]
    return bitstring_to_hex(trimmed)


def bitstring_to_int_list(bit_text: str) -> list[int]:
    return [int(char) for char in ''.join(bit_text.strip().split())]


def int_list_to_bitstring(values: Sequence[int]) -> str:
    return ''.join('1' if int(value) else '0' for value in values)


def batch_hex_to_square_bitstrings(hex_lines: Iterable[str]) -> tuple[list[str], int, list[int]]:
    lines = list(hex_lines)
    if not lines:
        return [], 0, []

    bit_lengths = [len(hex_to_bitstring(line)) for line in lines]
    side = square_side_for_bits(max(bit_lengths))
    padded_rows = [pad_bitstring_to_square(hex_to_bitstring(line), side=side)[0] for line in lines]
    return padded_rows, side, bit_lengths


def batch_restore_hex_from_square_bitstrings(bit_rows: Iterable[str], original_bit_lengths: Sequence[int]) -> list[str]:
    rows = list(bit_rows)
    if len(rows) != len(original_bit_lengths):
        raise ValueError('bit_rows 与 original_bit_lengths 数量不一致')
    return [restore_hex_from_square_bitstring(row, bit_length) for row, bit_length in zip(rows, original_bit_lengths)]
