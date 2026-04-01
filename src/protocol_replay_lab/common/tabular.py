from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .bitcodec import batch_hex_to_square_bitstrings, batch_restore_hex_from_square_bitstrings, bitstring_to_int_list, int_list_to_bitstring


@dataclass(slots=True)
class BitSquareDataset:
    hex_rows: list[str]
    bit_rows: list[str]
    original_bit_lengths: list[int]
    side: int

    @classmethod
    def from_hex_rows(cls, hex_rows: Sequence[str]) -> 'BitSquareDataset':
        padded_rows, side, bit_lengths = batch_hex_to_square_bitstrings(hex_rows)
        return cls(
            hex_rows=list(hex_rows),
            bit_rows=padded_rows,
            original_bit_lengths=bit_lengths,
            side=side,
        )

    @property
    def width(self) -> int:
        return self.side * self.side

    def as_int_matrix(self) -> np.ndarray:
        if not self.bit_rows:
            return np.zeros((0, 0), dtype=np.int64)
        return np.array([bitstring_to_int_list(row) for row in self.bit_rows], dtype=np.int64)

    def as_tensor_input(self) -> np.ndarray:
        matrix = self.as_int_matrix()
        if matrix.size == 0:
            return np.zeros((0, 1, 0, 0), dtype=np.float32)
        scaled = matrix * 2 - 1
        return scaled.reshape(matrix.shape[0], 1, self.side, self.side).astype(np.float32)

    def restore_hex_rows(self, bit_rows: Sequence[str]) -> list[str]:
        return batch_restore_hex_from_square_bitstrings(bit_rows, self.original_bit_lengths[: len(bit_rows)])

    def restore_from_samples(self, samples: np.ndarray, clamp: bool = True) -> list[str]:
        if samples.ndim != 4:
            raise ValueError('samples 必须是四维数组 [n, c, h, w]')
        flattened = samples.reshape(samples.shape[0], samples.shape[2] * samples.shape[3])
        rounded = (np.round(flattened) + 1) / 2
        if clamp:
            rounded = np.clip(rounded, 0, 1)
        bit_rows = [int_list_to_bitstring(row.astype(np.int64).tolist()) for row in rounded]
        return self.restore_hex_rows(bit_rows)
