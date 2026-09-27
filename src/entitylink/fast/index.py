"""Hashed blocking index: class-tagged 32-bit keys -> target row ids."""

from typing import List, Sequence, Tuple

import numpy as np

from entitylink.fast.fastnorm import KEY_CLASSES

CLASS_CAP_BY_NAME = {
    "name_exact": 400,
    "name_tokbag": 400,
    "postal": 40,
    "postal_namepfx": 200,
    "addr_exact": 100,
    "addrnum_namepfx": 200,
    "country_addrnum": 200,
    "longest_tok": 200,
    "street_tok": 300,
}
CAPS_BY_CLASS = np.zeros(16, dtype=np.int64)
for _name, _cls in KEY_CLASSES.items():
    CAPS_BY_CLASS[_cls] = CLASS_CAP_BY_NAME[_name]


class BlockIndexBuilder:
    """Accumulates (key, target_row) pairs and packs them into a sorted lookup."""

    def __init__(self, capacity: int):
        self.keys = np.empty(capacity, dtype=np.uint32)
        self.ids = np.empty(capacity, dtype=np.int32)
        self.n = 0

    def add_batch(self, key_lists: Sequence[Sequence[int]], start_row: int) -> None:
        lens = np.fromiter((len(k) for k in key_lists), dtype=np.int64, count=len(key_lists))
        total = int(lens.sum())
        if total == 0:
            return
        if self.n + total > self.keys.shape[0]:
            self._grow(self.n + total)
        flat: List[int] = []
        for ks in key_lists:
            flat.extend(ks)
        self.keys[self.n:self.n + total] = np.fromiter(flat, dtype=np.uint32, count=total)
        rows = np.repeat(np.arange(start_row, start_row + len(key_lists), dtype=np.int32), lens)
        self.ids[self.n:self.n + total] = rows
        self.n += total

    def _grow(self, need: int) -> None:
        new_cap = max(need, int(self.keys.shape[0] * 1.5) + 1)
        nk = np.empty(new_cap, dtype=np.uint32)
        nk[:self.n] = self.keys[:self.n]
        ni = np.empty(new_cap, dtype=np.int32)
        ni[:self.n] = self.ids[:self.n]
        self.keys, self.ids = nk, ni

    def finalize(self) -> "BlockIndex":
        packed = np.empty(self.n, dtype=np.uint64)
        np.left_shift(self.keys[:self.n].astype(np.uint64), np.uint64(32), out=packed)
        packed |= self.ids[:self.n].astype(np.uint64)
        packed.sort()
        uniq = np.empty(self.n, dtype=bool)
        uniq[0] = True
        np.not_equal(packed[1:], packed[:-1], out=uniq[1:])
        packed = packed[uniq]
        keys = (packed >> np.uint64(32)).astype(np.uint32)
        ids = (packed & np.uint64(0xFFFFFFFF)).astype(np.int32)
        return BlockIndex(keys, ids)


class BlockIndex:
    """Sorted (key, target_row) arrays with range lookup + ragged expansion."""

    def __init__(self, keys: np.ndarray, ids: np.ndarray):
        self.keys = keys
        self.ids = ids

    @property
    def size(self) -> int:
        return int(self.keys.shape[0])

    def ranges(self, q_keys: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        keys = q_keys.astype(np.uint32, copy=False)
        lo = np.searchsorted(self.keys, keys, side="left")
        hi = np.searchsorted(self.keys, keys, side="right")
        return lo, hi

    def expand(self, owners: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        lengths = hi - lo
        total = int(lengths.sum())
        if total == 0:
            return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int32)
        starts = np.cumsum(lengths) - lengths
        positions = np.repeat(lo, lengths) + (np.arange(total, dtype=np.int64) - np.repeat(starts, lengths))
        tids = self.ids[positions]
        owner_rep = np.repeat(owners.astype(np.int64), lengths)
        return owner_rep, tids
