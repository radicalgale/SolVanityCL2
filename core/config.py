# -*- coding: utf-8 -*-
import secrets
from math import ceil

import numpy as np

DEFAULT_ITERATION_BITS = 24
DEFAULT_LOCAL_WORK_SIZE = 256  # Увеличиваем для лучшей производительности


class HostSetting:
    def __init__(self, kernel_source: str, iteration_bits: int, gpu_index: int = 0, total_gpus: int = 1, local_work_size: int = None, fixed_seed: bytes = None, increment_at_start: bool = False, reverse_seed_bytes: bool = False, decrement: bool = False):
        self.iteration_bits = iteration_bits
        self.gpu_index = gpu_index
        self.total_gpus = total_gpus
        self.increment_at_start = bool(increment_at_start)
        self.reverse_seed_bytes = bool(reverse_seed_bytes)
        self.decrement = bool(decrement)
        # iteration_bytes 为需要被迭代覆盖的字节数（向上取整）
        self.iteration_bytes = np.ubyte(ceil(iteration_bits / 8))
        self.global_work_size = 1 << iteration_bits
        self.local_work_size = int(local_work_size) if local_work_size else DEFAULT_LOCAL_WORK_SIZE
        self.kernel_source = kernel_source
        if fixed_seed is not None:
            if not isinstance(fixed_seed, (bytes, bytearray)) or len(fixed_seed) != 32:
                raise ValueError("fixed_seed must be 32 bytes")
            # Ensure writable array (frombuffer over bytes can be read-only)
            self.key32 = np.frombuffer(bytes(fixed_seed), dtype=np.ubyte).copy()
        else:
            self.key32 = self.generate_key32()

    def generate_key32(self) -> np.ndarray:
        # Случайная основа и нулевой хвост длиной iteration_bytes — без GPU‑смещения в seed
        base_key = secrets.token_bytes(32 - int(self.iteration_bytes))
        tail = bytes([0]) * int(self.iteration_bytes)
        return np.array(list(base_key + tail), dtype=np.ubyte)

    def increase_key32(self) -> None:
        self.advance_key32(batches=1, across_all_gpus=True)

    def advance_key32(self, batches: int = 1, across_all_gpus: bool = True) -> None:
        """Advance base seed by a number of batches.

        If across_all_gpus is True, a "batch" equals global_work_size * total_gpus.
        If False, a batch equals just global_work_size (useful for drill-down on a single GPU).
        """
        stride = (1 << self.iteration_bits) * (max(int(self.total_gpus), 1) if across_all_gpus else 1)
        increment = stride * max(int(batches), 1)
        direction = -1 if self.decrement else 1
        if self.increment_at_start:
            # Reverse significance: increment from MSB side
            rev = bytes(self.key32[::-1])
            current_number = int.from_bytes(rev, "big")
            next_number = (current_number + direction * increment) % (1 << 256)
            new_key32_rev = next_number.to_bytes(32, "big")
            new_key32 = new_key32_rev[::-1]
        else:
            current_number = int.from_bytes(bytes(self.key32), "big")
            next_number = (current_number + direction * increment) % (1 << 256)
            new_key32 = next_number.to_bytes(32, "big")
        self.key32[:] = np.frombuffer(new_key32, dtype=np.ubyte)

    def get_device_seed(self) -> np.ndarray:
        """Return the seed bytes to send to the device (optionally reversed)."""
        if self.reverse_seed_bytes:
            return self.key32[::-1].copy()
        return self.key32
