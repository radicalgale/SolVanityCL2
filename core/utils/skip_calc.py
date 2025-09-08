# -*- coding: utf-8 -*-
from dataclasses import dataclass


@dataclass
class SkipMetrics:
    distance: int
    avg_distance: float
    deviation: int
    avg_abs_deviation: float
    avg_signed_deviation: float
    total_keys: int
    avg_sec_per_match: float
    matches_per_hour: float
    current_skip: int = 0


@dataclass
class SkipOptions:
    base_skip: int = 0
    mode: str = "base"  # "base", "adaptive", "avg_distance"
    min_skip: int = 1


def compute_effective_skip(metrics: SkipMetrics, opts: SkipOptions) -> tuple[int, str]:
    """
    Compute effective skip based on configured mode.

    Returns (effective_skip, display_label).
    """
    base = int(max(opts.base_skip, 0))
    label = f"{base:,}"
    total = base
    #
    # SANDBOX AREA START
    #
    # Adaptive 
    if opts.mode == "adaptive":
        devavg = int(metrics.avg_signed_deviation)          # достаем переменную таким вот образом
        total = base - devavg if base > 0 else -devavg      # тут пишем формулу для вычисления скипа
        label = f"{base:,} - {devavg:,} = {total:,}"        # сюда пишем формулу для вывода в логах
    # Average Distance
    if opts.mode == "avg_distance":
        deviation = int(metrics.deviation)
        candidate = int(metrics.avg_distance) + deviation
        if candidate <= 0:
            total = base
            label = f"{base:,}"
        else:
            total = candidate
            label = f"{int(metrics.avg_distance):,} + {deviation:,} = {total:,}"
    #
    # SANDBOX AREA END
    #
    # Enforce minimum skip outside per-mode logic
    total = max(int(total), int(opts.min_skip))
    return total, label


