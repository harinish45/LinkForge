"""Compact per-record feature payload stored as numpy arrays (memmapped or RAM).

A payload row is 9 scalars + 4 x (4 uint64) hash-bitsets, encoded by
``fastnorm.encode_record``. Rows are buffered and written in bulk to keep
Python-object memory bounded during full-scale passes.
"""

import json
import os
import re
from typing import Dict, List, Optional, Tuple

import numpy as np

SCALAR_FIELDS = [
    ("name_hash", np.uint32),
    ("addr_hash", np.uint32),
    ("postal", np.uint32),
    ("country", np.uint32),
    ("name_len", np.uint16),
    ("addr_len", np.uint16),
    ("n_name_tok", np.uint8),
    ("n_addr_tok", np.uint8),
    ("n_addr_num", np.uint8),
]
BIT_FIELDS = ["name_tok_bits", "name_char_bits", "addr_tok_bits", "addr_num_bits"]
N_LANES = 4
N_SCALARS = len(SCALAR_FIELDS)
PAYLOAD_WORDS = N_SCALARS + len(BIT_FIELDS) * N_LANES

_DTYPE_RE = re.compile(r"^(u?int|float|bool)")
DTYPES = {name: np.dtype(dt) for name, dt in SCALAR_FIELDS}
SCALAR_NAMES = [name for name, _ in SCALAR_FIELDS]


class PayloadStore:
    """Row-major store of encoded payloads with burst flushing."""

    def __init__(self, scalars: Dict[str, np.ndarray], bits: Dict[str, np.ndarray],
                 count: int, backing: str, base_path: Optional[str]):
        self.scalars = scalars
        self.bits = bits
        self.count = count
        self.backing = backing
        self.base_path = base_path
        self._buf: List[tuple] = []
        self.capacity = int(next(iter(scalars.values())).shape[0]) if scalars else 0

    # ---- construction -------------------------------------------------
    @classmethod
    def create(cls, capacity: int, base_path: Optional[str] = None, prefix: str = "p_") -> "PayloadStore":
        scalars: Dict[str, np.ndarray] = {}
        bits: Dict[str, np.ndarray] = {}
        if base_path:
            os.makedirs(base_path, exist_ok=True)
        for name, dtype in SCALAR_FIELDS:
            scalars[name] = _alloc(base_path, f"{prefix}{name}.npy", dtype, capacity)
        for name in BIT_FIELDS:
            bits[name] = _alloc(base_path, f"{prefix}{name}.npy", np.uint64, (capacity, N_LANES))
        backing = "memmap" if base_path else "ram"
        return cls(scalars, bits, 0, backing, base_path)

    @classmethod
    def open(cls, base_path: str, count: int, prefix: str = "p_") -> "PayloadStore":
        scalars = {name: np.load(os.path.join(base_path, f"{prefix}{name}.npy"), mmap_mode="r")
                   for name, _ in SCALAR_FIELDS}
        bits = {name: np.load(os.path.join(base_path, f"{prefix}{name}.npy"), mmap_mode="r")
                for name in BIT_FIELDS}
        return cls(scalars, bits, count, "memmap", base_path)

    # ---- writing ------------------------------------------------------
    def add(self, payload: tuple) -> None:
        self._buf.append(payload)
        if len(self._buf) >= 200_000:
            self.flush()

    def add_many(self, payloads) -> None:
        for p in payloads:
            self.add(p)

    def flush(self) -> None:
        if not self._buf:
            return
        cols = list(zip(*self._buf))
        start = self.count
        end = start + len(self._buf)
        if end > self.capacity:
            raise ValueError(f"PayloadStore overflow: need {end}, capacity {self.capacity}")
        for i, name in enumerate(SCALAR_NAMES):
            self.scalars[name][start:end] = np.asarray(cols[i], dtype=DTYPES[name])
        for j, name in enumerate(BIT_FIELDS):
            base = N_SCALARS + j * N_LANES
            block = np.asarray(cols[base:base + N_LANES], dtype=np.uint64).T
            self.bits[name][start:end] = block
        self.count = end
        self._buf.clear()

    def finalize(self) -> None:
        self.flush()
        if self.base_path:
            with open(os.path.join(self.base_path, "meta.json"), "w", encoding="utf-8") as f:
                json.dump({"count": int(self.count), "words": PAYLOAD_WORDS}, f)

    def slice_rows(self, idx: np.ndarray) -> Dict[str, np.ndarray]:
        """Gather payload rows for the given row indices (returns raw arrays)."""
        out = {name: self.scalars[name][idx] for name in SCALAR_NAMES}
        for name in BIT_FIELDS:
            out[name] = self.bits[name][idx]
        return out


def _alloc(base_path: Optional[str], fname: str, dtype, shape):
    if isinstance(shape, int):
        shape = (shape,)
    if base_path:
        os.makedirs(base_path, exist_ok=True)
        return np.lib.format.open_memmap(os.path.join(base_path, fname), mode="w+",
                                         dtype=dtype, shape=shape)
    return np.zeros(shape, dtype=dtype)
