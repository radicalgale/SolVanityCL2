# -*- coding: utf-8 -*-
import logging
import multiprocessing
import sys
import time
import json
import os

import click
import pyopencl as cl
from base58 import b58encode
from nacl.signing import SigningKey
import numpy as np

from core.config import DEFAULT_ITERATION_BITS, HostSetting
from core.opencl.manager import (
    get_all_gpu_devices,
    get_chosen_devices,
    get_selected_gpu_devices,
)
from core.searcher import Searcher
from core.utils.helpers import load_kernel_source
from core.utils.crypto import save_keypair, get_public_key_from_private_bytes
from core.utils.skip_calc import SkipMetrics, SkipOptions, compute_effective_skip

logging.basicConfig(level=logging.INFO, format="[%(levelname)s %(asctime)s] %(message)s")


@click.group()
def cli():
    """Solana Vanity Address Searcher"""
    pass


@cli.command(context_settings={"show_default": True})
@click.option(
    "--config", "-c", type=click.Path(exists=True, dir_okay=False),
    help="Если указан, параметры берутся из JSON‑файла."
)
@click.option(
    "--starts-with", type=str, default=[], multiple=True,
    help="Префикс(ы) адреса (можно указать несколько раз)."
)
@click.option(
    "--ends-with", type=str, default="",
    help="Суффикс адреса."
)
@click.option(
    "--count", type=int, default=1,
    help="Сколько совпадений дождаться (0 — бесконечно)."
)
@click.option(
    "--iteration-bits", type=int, default=DEFAULT_ITERATION_BITS,
    help="2^битов ключей за батч на GPU."
)
@click.option(
    "--local-work-size", type=int, default=None,
    help="Размер локальной группы работы (work-group size)."
)
@click.option(
    "--auto-tune-lws/--no-auto-tune-lws", default=True,
    help="Автонастройка размера локальной группы (work-group size)."
)
@click.option(
    "--seed-b58", type=str, default=None,
    help="Фиксированный seed (base58, 32 байта) для детерминированных тестов."
)
@click.option(
    "--use-last-seed/--no-use-last-seed", default=False,
    help="Стартовать с последнего сохранённого seed (если seedB58 пустой — читать из last_seed.txt)."
)
@click.option(
    "--increment-at-start/--increment-at-end", default=False,
    help="Инкрементировать seed с начала (MSB) вместо конца (LSB)."
)
@click.option(
    "--reverse-seed-bytes/--no-reverse-seed-bytes", default=False,
    help="Отправлять на устройство seed в обратном порядке байт."
)
@click.option(
    "--skip-suffix-chars", type=int, default=0,
    help="При совпадении префикса сместить символ на позиции (len- K) base58 seed (0 = выключено)."
)
@click.option(
    "--adaptive-skip/--no-adaptive-skip", default=False,
    help="Адаптивный пропуск: эффективный skip = skipSuffixAmount ± DevAvg (средняя подписанная девиация)."
)
@click.option(
    "--avg-dist-skip/--no-avg-dist-skip", default=False,
    help="AvgDistance‑based skip: эффективный skip = AvgDistance + DevAvg (если > 0), иначе базовый skip."
)
@click.option(
    "--skip-suffix-amount", type=int, default=0,
    help="При совпадении префикса добавить к seed числовое смещение (в base58 единицах), 0 = выключено. Пример: 58 меняет предпоследний символ."
)
@click.option(
    "--is-case-sensitive/--no-case-sensitive", default=True,
    help="Учитывать регистр."
)
@click.option(
    "--output-dir", type=click.Path(file_okay=False, dir_okay=True),
    default="./found", help="Куда складывать результаты."
)
@click.option(
    "--write-json/--no-write-json", default=True,
    help="Сохранять JSON-файлы ключей (found/JSON)."
)
@click.option(
    "--write-prefixes/--no-write-prefixes", default=True,
    help="Сохранять seed по префиксам (found/prefixes)."
)
@click.option(
    "--write-data/--no-write-data", default=True,
    help="Сохранять метрики в found/data/*.csv."
)
@click.option(
    "--select-device/--no-select-device", default=False,
    help="Вручную выбрать GPU."
)
@click.option(
    "--record-all/--no-record-all", default=False,
    help="Записывать «начальный» seed каждого батча в all_keys.txt."
)
@click.option(
    "--build-options", type=str, default="",
    help="Опции сборки OpenCL, через пробел (например: '-O3 -cl-mad-enable -cl-fast-relaxed-math')."
)
@click.option(
    "--nv-max-rregcount", type=int, default=None,
    help="NVIDIA: ограничение регистров на поток (например: 64)."
)
@click.option(
    "--max-results-per-batch", type=int, default=1024,
    help="Максимум совпадений, возвращаемых ядром за один батч (на GPU)."
)
@click.option(
    "--accumulate-batches/--no-accumulate-batches", default=False,
    help="Копить результаты в видеопамяти между батчами и сбрасывать пачкой."
)
@click.option(
    "--flush-every-batches", type=int, default=1,
    help="Как часто сбрасывать результаты при аккумуляции (в батчах)."
)
@click.option(
    "--flush-policy",
    type=click.Choice(["interval", "interval_or_near_full"], case_sensitive=False),
    default="interval_or_near_full",
    help="Стратегия сброса: только по интервалу или интервал/почти-полный буфер.",
)
def search_pubkey(
    config,
    starts_with,
    ends_with,
    count,
    iteration_bits,
    local_work_size,
    auto_tune_lws,
    seed_b58,
    use_last_seed,
    increment_at_start,
    reverse_seed_bytes,
    skip_suffix_chars,
    adaptive_skip,
    avg_dist_skip,
    skip_suffix_amount,
    is_case_sensitive,
    output_dir,
    write_json,
    write_prefixes,
    write_data,
    select_device,
    record_all,
    build_options,
    nv_max_rregcount,
    max_results_per_batch,
    accumulate_batches,
    flush_every_batches,
    flush_policy,
):
    """
    Поиск Solana vanity‑адресов.
    --record-all включит запись «начального» seed‑ключа каждого батча в all_keys.txt.
    """
    # --- 1) читаем config.json, если есть ---
    if config:
        try:
            cfg = json.load(open(config, "r", encoding="utf-8"))
        except Exception as e:
            logging.error("Не удалось загрузить {}: {}".format(config, e))
            sys.exit(1)
        # Nested groups with backward-compatible fallbacks
        search_cfg        = cfg.get("search", {}) if isinstance(cfg.get("search", {}), dict) else {}
        perf_cfg          = cfg.get("performance", {}) if isinstance(cfg.get("performance", {}), dict) else {}
        seed_cfg          = cfg.get("seed", {}) if isinstance(cfg.get("seed", {}), dict) else {}
        skip_cfg          = cfg.get("skip", {}) if isinstance(cfg.get("skip", {}), dict) else {}
        out_cfg           = cfg.get("output", {}) if isinstance(cfg.get("output", {}), dict) else {}
        dev_cfg           = cfg.get("devices", {}) if isinstance(cfg.get("devices", {}), dict) else {}
        acc_cfg           = cfg.get("accumulation", {}) if isinstance(cfg.get("accumulation", {}), dict) else {}
        rec_cfg           = cfg.get("record", {}) if isinstance(cfg.get("record", {}), dict) else {}

        starts_with       = tuple(search_cfg.get("startsWith", cfg.get("startsWith", cfg.get("starts_with", []))))
        ends_with         = search_cfg.get("endsWith", cfg.get("endsWith", cfg.get("ends_with", "")))
        count             = search_cfg.get("count", cfg.get("count", count))

        iteration_bits    = perf_cfg.get("iterationBits", cfg.get("iterationBits", cfg.get("iteration_bits", iteration_bits)))
        local_work_size   = perf_cfg.get("localWorkSize", cfg.get("localWorkSize", cfg.get("local_work_size", local_work_size)))
        auto_tune_lws     = perf_cfg.get("autoTuneLws", cfg.get("autoTuneLws", cfg.get("auto_tune_lws", auto_tune_lws)))
        build_options     = perf_cfg.get("buildOptions", cfg.get("buildOptions", cfg.get("build_options", build_options)))
        nv_max_rregcount  = perf_cfg.get("nvMaxRregcount", cfg.get("nvMaxRregcount", cfg.get("nv_max_rregcount", nv_max_rregcount)))

        seed_b58          = seed_cfg.get("seedB58", cfg.get("seedB58", cfg.get("seed_b58", seed_b58)))
        # Treat empty string as disabled
        if isinstance(seed_b58, str) and seed_b58.strip() == "":
            seed_b58 = None
        increment_at_start = seed_cfg.get("incrementAtStart", cfg.get("incrementAtStart", cfg.get("increment_at_start", increment_at_start)))
        reverse_seed_bytes = seed_cfg.get("reverseSeedBytes", cfg.get("reverseSeedBytes", cfg.get("reverse_seed_bytes", reverse_seed_bytes)))
        seed_decrement    = seed_cfg.get("decrement", cfg.get("seedDecrement", False))

        skip_suffix_chars  = skip_cfg.get("skipSuffixChars", cfg.get("skipSuffixChars", cfg.get("skip_suffix_chars", skip_suffix_chars)))
        skip_enabled       = bool(skip_cfg.get("enabled", cfg.get("skipEnabled", True)))
        adaptive_skip      = skip_cfg.get("adaptiveSkip", cfg.get("adaptiveSkip", cfg.get("adaptive_skip", adaptive_skip)))
        avg_dist_skip      = skip_cfg.get("avgDistanceSkip", cfg.get("avgDistanceSkip", cfg.get("avg_dist_skip", avg_dist_skip)))
        # Single mode string (free-form). The implementation lives in skip_calc.py
        skip_mode_cfg      = skip_cfg.get("skipMode", cfg.get("skipMode", cfg.get("skip_mode", None)))
        use_last_seed      = seed_cfg.get("useLastSeed", cfg.get("useLastSeed", cfg.get("use_last_seed", use_last_seed)))
        skip_suffix_amount = skip_cfg.get("skipSuffixAmount", cfg.get("skipSuffixAmount", cfg.get("skip_suffix_amount", skip_suffix_amount)))
        is_case_sensitive  = search_cfg.get("caseSensitive", cfg.get("caseSensitive", cfg.get("case_sensitive",  is_case_sensitive)))
        # Nested: output settings already loaded as out_cfg
        output_dir        = out_cfg.get("dir", cfg.get("outputDir", cfg.get("output_dir", output_dir)))
        write_json        = bool(out_cfg.get("writeJson", cfg.get("writeJson", write_json)))
        write_prefixes    = bool(out_cfg.get("writePrefixes", cfg.get("writePrefixes", write_prefixes)))
        write_data        = bool(out_cfg.get("writeData", cfg.get("writeData", write_data)))
        # Devices / record
        select_device     = dev_cfg.get("selectDevice", cfg.get("selectDevice", cfg.get("select_device", select_device)))
        # recordAll is now under accumulation; keep legacy fallback from record
        record_all        = acc_cfg.get("recordAll", rec_cfg.get("recordAll", cfg.get("record_all", record_all)))
        # Accumulation
        max_results_per_batch = acc_cfg.get("maxResultsPerBatch", cfg.get("maxResultsPerBatch", cfg.get("max_results_per_batch", max_results_per_batch)))
        accumulate_batches = acc_cfg.get("accumulateBatches", cfg.get("accumulateBatches", cfg.get("accumulate_batches", accumulate_batches)))
        flush_every_batches = acc_cfg.get("flushEveryBatches", cfg.get("flushEveryBatches", cfg.get("flush_every_batches", flush_every_batches)))
        flush_policy = (acc_cfg.get("flushPolicy", cfg.get("flushPolicy", cfg.get("flush_policy", flush_policy))) or "interval_or_near_full")
        record_all_mode = (acc_cfg.get("recordAllMode", "tiled") or "tiled").strip().lower()
    else:
        starts_with = tuple(starts_with)

    if not starts_with and not ends_with:
        click.echo("Нужно задать --starts-with или --ends-with либо --config.")
        sys.exit(1)

    # --- 2) выбираем GPU ---
    if select_device:
        plat, devs = get_chosen_devices()
        devices = get_selected_gpu_devices(plat, devs)
    else:
        devices = get_all_gpu_devices()
        plat, devs = None, None
    gpu_count = len(devices)
    logging.info("Используем {} GPU device(s)".format(gpu_count))

    # --- 3) компилируем OpenCL‑ядро с вашими префиксами/суффиксом ---
    kernel_src = load_kernel_source(starts_with, ends_with, is_case_sensitive)

    # --- 4) инициализируем Searcher-ы ---
    # Helper: read last seed from file (skip comments and blanks)
    def read_last_seed_from_file(path: str):
        try:
            if not os.path.isfile(path):
                return None
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    if line.startswith("#") or line.startswith("//") or line.startswith(";"):
                        continue
                    # take first whitespace-delimited token
                    token = line.split()[0]
                    from base58 import b58decode
                    try:
                        b = b58decode(token)
                        if len(b) == 32:
                            return b
                    except Exception:
                        continue
        except Exception:
            return None
        return None

    # Normalize and decode fixed seed if provided / or use last seed
    if isinstance(seed_b58, str):
        seed_b58 = seed_b58.strip()
        if seed_b58 == "":
            seed_b58 = None
    fixed_seed_bytes = None
    last_seed_path = os.path.join(output_dir, "last_seed.txt")
    if use_last_seed:
        fixed_seed_bytes = read_last_seed_from_file(last_seed_path)
        if fixed_seed_bytes is not None:
            logging.info("Loaded last seed from {}".format(last_seed_path))
        else:
            if seed_b58:
                from base58 import b58decode
                try:
                    fixed_seed_bytes = b58decode(seed_b58)
                    if len(fixed_seed_bytes) != 32:
                        logging.error("seedB58 must decode to 32 bytes; got {}".format(len(fixed_seed_bytes)))
                        sys.exit(1)
                    logging.info("useLastSeed enabled but missing; falling back to provided seedB58")
                except Exception as e:
                    logging.error("useLastSeed enabled but no valid last seed; also failed to decode seedB58: {}".format(e))
                    sys.exit(1)
            else:
                logging.warning("useLastSeed enabled, but no valid last seed found at {} and no seedB58 provided".format(last_seed_path))
                fixed_seed_bytes = None
    elif seed_b58:
        try:
            from base58 import b58decode
            fixed_seed_bytes = b58decode(seed_b58)
            if len(fixed_seed_bytes) != 32:
                logging.error("seedB58 must decode to 32 bytes; got {}".format(len(fixed_seed_bytes)))
                sys.exit(1)
        except Exception as e:
            logging.error("Failed to decode seedB58: {}".format(e))
            sys.exit(1)

    settings = [HostSetting(kernel_src, iteration_bits, idx, gpu_count, local_work_size, fixed_seed=fixed_seed_bytes, increment_at_start=increment_at_start, reverse_seed_bytes=reverse_seed_bytes, decrement=bool(seed_decrement)) for idx in range(gpu_count)]
    options_list = [opt for opt in (build_options.split() if build_options else []) if opt]
    searchers = []
    for idx in range(gpu_count):
        try:
            s = Searcher(
                kernel_src,
                idx,
                settings[idx],
                (plat, devs) if select_device else None,
                build_options=options_list,
                nv_max_rregcount=nv_max_rregcount,
                max_results_per_batch=max(1, int(max_results_per_batch)),
                auto_tune_lws=bool(auto_tune_lws),
                accumulate_batches=bool(accumulate_batches),
                flush_every_batches=max(1, int(flush_every_batches)),
                flush_policy=str(flush_policy).strip().lower(),
                record_all=bool(record_all),
                record_all_mode=str(record_all_mode),
            )
        except TypeError as _te:
            logging.warning("Searcher signature mismatch ({}). Falling back to minimal init.".format(_te))
            try:
                s = Searcher(
                    kernel_src,
                    idx,
                    settings[idx],
                    (plat, devs) if select_device else None,
                )
            except Exception as _e2:
                logging.error("Failed to initialize Searcher: {}".format(_e2))
                raise SystemExit(1)
        searchers.append(s)

    # Log current base seed(s) at start
    try:
        for idx in range(gpu_count):
            host_seed_b58 = b58encode(bytes(settings[idx].key32)).decode()
            dev_seed_bytes = settings[idx].get_device_seed()
            dev_seed_b58 = b58encode(bytes(dev_seed_bytes)).decode()
            logging.info("CURRENT SEED [GPU {}] host:{} device:{}".format(idx, host_seed_b58, dev_seed_b58))
    except Exception as e:
        logging.error("Failed to log current seed(s): {}".format(e))

    # --- 5) готовим выходные папки и файлы ---
    os.makedirs(output_dir, exist_ok=True)
    # тут JSON-ки всех найденных сохраняем в отдельную папку
    json_dir = os.path.join(output_dir, "JSON")
    if write_json:
        os.makedirs(json_dir, exist_ok=True)
    # папка для логирования seed по префиксу
    prefixes_dir = os.path.join(output_dir, "prefixes")
    if write_prefixes:
        os.makedirs(prefixes_dir, exist_ok=True)
    # папка для табличных данных по префиксам
    data_dir = os.path.join(output_dir, "data")
    if write_data:
        os.makedirs(data_dir, exist_ok=True)

    csv_file  = os.path.join(output_dir, "results.csv")
    txt_all   = os.path.join(output_dir, "all_keys.txt")

    # создаём CSV с заголовком, если нужно
    if not os.path.isfile(csv_file):
        with open(csv_file, "w", encoding="utf-8") as f:
            f.write("publicKey,privateKey\n")
    if record_all and not os.path.isfile(txt_all):
        open(txt_all, "a", encoding="utf-8").close()

    total_tested = 0
    grand_total_tested = 0
    total_found  = 0
    last_report  = time.time()
    run_start    = time.time()
    # Track absolute starting seed (min across GPUs) for total-keys distance
    try:
        run_start_seed_int = min(int.from_bytes(bytes(s.key32), "big") for s in settings)
    except Exception:
        run_start_seed_int = None
    # Deviation aggregates
    sum_abs_deviation = 0
    sum_signed_deviation = 0
    # Distance tracking based on binary difference between consecutive successful seeds
    last_success_seed_int = None
    sum_binary_distances = 0
    num_distance_events = 0

    logging.info("Старт поиска… Ctrl‑C для остановки")
    # Log skip mode at start
    try:
        mode_start = str(skip_mode_cfg).strip().lower() if skip_mode_cfg else ("avg_distance" if avg_dist_skip else ("adaptive" if adaptive_skip else "base"))
        logging.info("Skip mode: {}".format(mode_start))
    except Exception:
        pass
    # Helper to pretty-print a boxed info block
    def log_pretty_box(title: str, fields: list):
        try:
            label_w = max(len(k) for k, _ in fields) if fields else 0
            value_w = max(len(v) for _, v in fields) if fields else 0
            inner_w = max(len(title), label_w + 2 + value_w)
            top = "+-" + ("-" * inner_w) + "-+"
            mid_title = "| " + title.ljust(inner_w) + " |"
            lines = [top, mid_title, top]
            for k, v in fields:
                lines.append("| " + (k.ljust(label_w) + ": " + v.ljust(inner_w - label_w - 2)) + " |")
            lines.append(top)
            logging.info("\n" + "\n".join(lines))
        except Exception:
            pass
    # Helper to mutate seed: bump the first char of the last K base58 chars, wrap in alphabet; keep 32-byte length
    def mutate_seed_suffix_b58(seed_bytes: bytes, k: int) -> bytes:
        try:
            if not k or k <= 0:
                return None
            s = b58encode(seed_bytes).decode()
            if len(s) < k:
                return None
            ALPH = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"
            idx = len(s) - k
            pos = ALPH.find(s[idx])
            if pos < 0:
                return None
            # Try up to 58 steps to find a variant that decodes to 32 bytes
            for step in range(1, 59):
                new_c = ALPH[(pos + step) % 58]
                s2 = s[:idx] + new_c + s[idx + 1:]
                try:
                    from base58 import b58decode
                    b = b58decode(s2)
                    if len(b) == 32:
                        return b
                except Exception:
                    continue
        except Exception:
            return None
        return None

    # Helper to mutate by integer amount (adds to 256-bit seed, wraps)
    def mutate_seed_add_amount(seed_bytes: bytes, amount: int) -> bytes:
        try:
            if not amount or int(amount) == 0:
                return None
            n = int.from_bytes(seed_bytes, "big")
            next_n = (n + int(amount)) % (1 << 256)
            return next_n.to_bytes(32, "big")
        except Exception:
            return None
    # Helper to mutate by integer amount (adds to 256-bit seed, wraps)
    def mutate_seed_add_amount(seed_bytes: bytes, amount: int) -> bytes:
        try:
            if not amount or int(amount) == 0:
                return None
            n = int.from_bytes(seed_bytes, "big")
            next_n = (n + int(amount)) % (1 << 256)
            return next_n.to_bytes(32, "big")
        except Exception:
            return None
    try:
        while True:
            # 1) Логируем начальные seed всех GPU один раз за итерацию
            # В режиме recordAll сохраняем все ключи в CSV, а не в all_keys.txt
            # (отключаем запись в txt для снижения I/O)

            # 2) Запуск всех GPU параллельно
            kernel_events = []
            for idx, searcher in enumerate(searchers):
                try:
                    evt = searcher.start()
                    kernel_events.append((idx, evt))
                except Exception as e:
                    logging.error("GPU {} start error: {}".format(idx, e))
                    kernel_events.append((idx, None))

            # 3) Ожидание завершения и обработка результатов
            for idx, evt in kernel_events:
                try:
                    if evt is None:
                        continue
                    # обычный шаг — продвигаем seed через все GPU
                    matches = searchers[idx].finish(evt, advance_across_all_gpus=True)
                    tested = settings[idx].global_work_size
                    total_tested += tested
                    grand_total_tested += tested
                    # Collect and persist all matches for this GPU/batch (bulk CSV write)
                    if matches and len(matches) > 0:
                        csv_rows = []
                        for pv_bytes in matches:
                            pv_bytes = bytes(pv_bytes)
                            if len(pv_bytes) != 32:
                                continue
                            # Compute pub; optionally write JSON
                            if write_json:
                                pub = save_keypair(pv_bytes, json_dir)
                            else:
                                pub = get_public_key_from_private_bytes(pv_bytes)
                            pb_bytes = bytes(SigningKey(pv_bytes).verify_key)
                            full = pv_bytes + pb_bytes
                            priv_b58 = b58encode(full).decode()
                            csv_rows.append("{},{}\n".format(pub, priv_b58))
                            # Compute matching prefixes
                            try:
                                if starts_with:
                                    matching_prefixes = [p for p in starts_with if pub.startswith(p)]
                                else:
                                    matching_prefixes = []
                                # Save per-match seed under matching prefixes
                                if write_prefixes and matching_prefixes:
                                    seed_b58 = b58encode(pv_bytes).decode()
                                    for p in matching_prefixes:
                                        pfile = os.path.join(prefixes_dir, "{}{}.txt".format(p, ""))
                                        with open(pfile, "a", encoding="utf-8") as pf:
                                            pf.write("{},{}\n".format(pub, seed_b58))
                            except Exception as _e:
                                logging.error("Save prefix-seed error: {}".format(_e))
                        if csv_rows:
                            try:
                                with open(csv_file, "a", encoding="utf-8") as f_csv:
                                    f_csv.writelines(csv_rows)
                            except Exception as _e:
                                logging.error("Bulk CSV write error: {}".format(_e))
                        # Сохраняем seed по совпавшим префиксам
                        try:
                            # For each match, log under all matching prefixes
                            matching_prefixes = []
                            if starts_with:
                                # compute once per match inside the loop above
                                pass
                        except Exception as _e:
                            logging.error("Save prefix-seed error: {}".format(_e))
                        total_found += len([1 for _ in matches])
                        # In record-all mode, skip heavy host-side logic (skip, metrics, plotting)
                        if record_all:
                            continue
                        # Apply optional seed suffix mutation for further scanning on this GPU
                        try:
                            mutated = None
                            # character-based skip first (does not depend on deviations)
                            if skip_enabled and skip_suffix_chars and int(skip_suffix_chars) > 0 and len(matches) > 0:
                                mutated = mutate_seed_suffix_b58(bytes(matches[-1]), int(skip_suffix_chars))
                                if mutated is not None:
                                    settings[idx].key32[:] = np.frombuffer(mutated, dtype=np.ubyte)
                                    logging.info("Applied seed suffix skip (K={}): {}".format(
                                        int(skip_suffix_chars), b58encode(mutated).decode()
                                    ))
                            # numeric skip amount is applied AFTER distance metrics are updated below using current match
                        except Exception as _e:
                            logging.error("Seed suffix skip failed: {}".format(_e))
                        # Distance metrics (binary difference between consecutive successful seeds)
                        try:
                            # Compute distance using device-order seed bytes
                            if matches and len(matches) > 0:
                                pv_bytes = bytes(matches[-1])
                            cur_n = int.from_bytes(pv_bytes, "big")
                            if last_success_seed_int is not None:
                                if cur_n >= last_success_seed_int:
                                    dist = cur_n - last_success_seed_int
                                else:
                                    dist = (1 << 256) - last_success_seed_int + cur_n
                                sum_binary_distances += dist
                                num_distance_events += 1
                                avg_dist = (sum_binary_distances / num_distance_events) if num_distance_events > 0 else 0
                                deviation = int(dist - avg_dist)
                                sum_abs_deviation += abs(deviation)
                                sum_signed_deviation += deviation
                                avg_abs_dev = (sum_abs_deviation / num_distance_events) if num_distance_events > 0 else 0
                                avg_signed_dev = (sum_signed_deviation / num_distance_events) if num_distance_events > 0 else 0
                                # totalKeys since run start (absolute progression)
                                if run_start_seed_int is not None:
                                    if cur_n >= run_start_seed_int:
                                        total_since_start = cur_n - run_start_seed_int + 1
                                    else:
                                        total_since_start = (1 << 256) - run_start_seed_int + cur_n + 1
                                else:
                                    total_since_start = 0
                                # Apply numeric skip amount using centralized strategy
                                try:
                                    if skip_enabled:
                                        base_skip = int(skip_suffix_amount) if (skip_suffix_amount and int(skip_suffix_amount) != 0) else 0
                                        # Determine skip mode: let any string pass to skip_calc
                                        mode = str(skip_mode_cfg).strip().lower() if skip_mode_cfg else ("avg_distance" if avg_dist_skip else ("adaptive" if adaptive_skip else "base"))
                                        metrics = SkipMetrics(
                                            distance=int(dist),
                                            avg_distance=float(avg_dist),
                                            deviation=int(deviation),
                                            avg_abs_deviation=float(avg_abs_dev),
                                            avg_signed_deviation=float(avg_signed_dev),
                                            total_keys=int(total_since_start),
                                            avg_sec_per_match=float(avg_sec_per_match),
                                            matches_per_hour=float(matches_per_hour),
                                            current_skip=base_skip,
                                        )
                                        opts = SkipOptions(base_skip=base_skip, mode=mode, min_skip=1)
                                        eff_skip_amount, skip_label = compute_effective_skip(metrics, opts)
                                        if eff_skip_amount != 0:
                                            mutated2 = mutate_seed_add_amount(pv_bytes, eff_skip_amount)
                                            if mutated2 is not None:
                                                settings[idx].key32[:] = np.frombuffer(mutated2, dtype=np.ubyte)
                                        if mode == "adaptive":
                                            skip_suffix_amount = eff_skip_amount
                                    else:
                                        skip_label = "disabled"
                                except Exception:
                                    pass
                                # average efficiency
                                try:
                                    elapsed_so_far = time.time() - run_start
                                    avg_sec_per_match = (elapsed_so_far / total_found) if total_found > 0 else 0.0
                                    matches_per_hour = (3600.0 / avg_sec_per_match) if avg_sec_per_match > 0 else 0.0
                                except Exception:
                                    avg_sec_per_match = 0.0
                                    matches_per_hour = 0.0
                                # Compose skip amount display using the returned label
                                log_pretty_box(
                                    "Match #{}".format(total_found),
                                    [
                                        ("Public Key", pub),
                                        ("Seed (b58)", b58encode(pv_bytes).decode()),
                                        ("Skip Amount", skip_label if skip_enabled else "disabled"),
                                        ("Distance", f"{int(dist):,}"),
                                        ("Avg Distance", f"{int(avg_dist):,}"),
                                        ("Deviation", f"{deviation:,}"),
                                        ("AbsDevAvg", f"{int(avg_abs_dev):,}"),
                                        ("DevAvg", f"{int(avg_signed_dev):,}"),
                                        ("Total Keys", f"{int(total_since_start):,}"),
                                        ("Avg Sec/Match", f"{avg_sec_per_match:.2f}"),
                                        ("Matches/Hour", f"{matches_per_hour:.2f}"),
                                    ],
                                )
                                # Persist row(s) for plotting per matching prefix (and ALL)
                                try:
                                    ts = time.strftime('%Y-%m-%d %H:%M:%S')
                                    row_seed = b58encode(pv_bytes).decode()
                                    if write_data:
                                        for p in (matching_prefixes if matching_prefixes else ["ALL"]):
                                            dfile = os.path.join(data_dir, "{}.csv".format(p))
                                            if not os.path.isfile(dfile):
                                                with open(dfile, "w", encoding="utf-8") as df:
                                                    df.write("time,seedB58,privateKey,publicKey,skipAmount,distance,deviation,avgAbsDeviation,avgDeviation,totalKeys,avgSecPerMatch,matchesPerHour\n")
                                            # totalKeys: absolute keys reached since start seed (includes skips)
                                            if run_start_seed_int is not None:
                                                if cur_n >= run_start_seed_int:
                                                    total_since_start = cur_n - run_start_seed_int + 1
                                                else:
                                                    total_since_start = (1 << 256) - run_start_seed_int + cur_n + 1
                                            else:
                                                total_since_start = 0
                                            # average efficiency metrics
                                            try:
                                                elapsed_so_far = time.time() - run_start
                                                avg_sec_per_match = (elapsed_so_far / total_found) if total_found > 0 else 0.0
                                                matches_per_hour = (3600.0 / avg_sec_per_match) if avg_sec_per_match > 0 else 0.0
                                            except Exception:
                                                avg_sec_per_match = 0.0
                                                matches_per_hour = 0.0
                                            with open(dfile, "a", encoding="utf-8") as df:
                                                df.write("{},{},{},{},{},{},{},{},{},{},{:.2f},{:.2f}\n".format(
                                                    ts,
                                                    row_seed,
                                                    priv_b58,
                                                    pub,
                                                    int(skip_suffix_amount) if (skip_suffix_amount and int(skip_suffix_amount) != 0) else 0,
                                                    int(dist),
                                                    int(deviation),
                                                    int(avg_abs_dev),
                                                    int(avg_signed_dev),
                                                    int(total_since_start),
                                                    avg_sec_per_match,
                                                    matches_per_hour
                                                ))
                                except Exception:
                                    pass
                            else:
                                # First match: log and persist with zero-based metrics
                                dist = 0
                                avg_dist = 0
                                deviation = 0
                                avg_abs_dev = 0
                                avg_signed_dev = 0
                                if run_start_seed_int is not None:
                                    if cur_n >= run_start_seed_int:
                                        total_since_start = cur_n - run_start_seed_int + 1
                                    else:
                                        total_since_start = (1 << 256) - run_start_seed_int + cur_n + 1
                                else:
                                    total_since_start = 0
                                # efficiency
                                try:
                                    elapsed_so_far = time.time() - run_start
                                    avg_sec_per_match = (elapsed_so_far / total_found) if total_found > 0 else 0.0
                                    matches_per_hour = (3600.0 / avg_sec_per_match) if avg_sec_per_match > 0 else 0.0
                                except Exception:
                                    avg_sec_per_match = 0.0
                                    matches_per_hour = 0.0
                                if skip_enabled:
                                    skip_base_str = f"{int(skip_suffix_amount):,}" if (skip_suffix_amount and int(skip_suffix_amount) != 0) else "0"
                                    devavg_str = "0"
                                    skip_total_str = skip_base_str
                                else:
                                    skip_base_str = "disabled"
                                    devavg_str = "-"
                                    skip_total_str = "-"
                                log_pretty_box(
                                    "Match #{}".format(total_found),
                                    [
                                        ("Public Key", pub),
                                        ("Seed (b58)", b58encode(pv_bytes).decode()),
                                        ("Skip Amount", f"{skip_base_str} + {devavg_str} = {skip_total_str}"),
                                        ("Distance", f"{int(dist):,}"),
                                        ("Avg Distance", f"{int(avg_dist):,}"),
                                        ("Deviation", f"{deviation:,}"),
                                        ("AbsDevAvg", f"{int(avg_abs_dev):,}"),
                                        ("DevAvg", f"{int(avg_signed_dev):,}"),
                                        ("Total Keys", f"{int(total_since_start):,}"),
                                        ("Avg Sec/Match", f"{avg_sec_per_match:.2f}"),
                                        ("Matches/Hour", f"{matches_per_hour:.2f}"),
                                    ],
                                )
                                try:
                                    ts = time.strftime('%Y-%m-%d %H:%M:%S')
                                    row_seed = b58encode(pv_bytes).decode()
                                    for p in (matching_prefixes if matching_prefixes else ["ALL"]):
                                        dfile = os.path.join(data_dir, "{}.csv".format(p))
                                        if not os.path.isfile(dfile):
                                            with open(dfile, "w", encoding="utf-8") as df:
                                                df.write("time,seedB58,privateKey,publicKey,skipAmount,distance,deviation,avgAbsDeviation,avgDeviation,totalKeys,avgSecPerMatch,matchesPerHour\n")
                                        with open(dfile, "a", encoding="utf-8") as df:
                                            df.write("{},{},{},{},{},{},{},{},{},{},{:.2f},{:.2f}\n".format(
                                                ts,
                                                row_seed,
                                                priv_b58,
                                                pub,
                                                (int(skip_suffix_amount) if (skip_enabled and skip_suffix_amount and int(skip_suffix_amount) != 0) else 0),
                                                int(dist),
                                                int(deviation),
                                                int(avg_abs_dev),
                                                int(avg_signed_dev),
                                                int(total_since_start),
                                                avg_sec_per_match,
                                                matches_per_hour
                                            ))
                                except Exception:
                                    pass
                            last_success_seed_int = cur_n
                        except Exception:
                            pass
                        # Finding rate (average time per match)
                        try:
                            elapsed_so_far = time.time() - run_start
                            if total_found > 0 and elapsed_so_far > 0:
                                avg_time_per_find = elapsed_so_far / total_found
                                finds_per_hour = 3600.0 / avg_time_per_find
                        except Exception:
                            pass
                        if count > 0 and total_found >= count:
                            raise KeyboardInterrupt()
                except KeyboardInterrupt:
                    raise
                except Exception as e:
                    logging.error("GPU {} finish error: {}".format(idx, e))
                    continue

            # 4) Каждую секунду — агрегированный отчёт
            now = time.time()
            if now - last_report >= 1.0:
                elapsed = now - last_report
                rate = total_tested / elapsed if elapsed > 0 else 0
                mh = rate / 1e6
                # Compute current total keys progressed since run start (absolute), using max seed across GPUs
                try:
                    current_max_seed = max(int.from_bytes(bytes(s.key32), "big") for s in settings)
                    if run_start_seed_int is not None:
                        if current_max_seed >= run_start_seed_int:
                            total_keys_abs = current_max_seed - run_start_seed_int + 1
                        else:
                            total_keys_abs = (1 << 256) - run_start_seed_int + current_max_seed + 1
                    else:
                        total_keys_abs = 0
                except Exception:
                    total_keys_abs = 0
                click.echo(
                    "[{}] Tested: {:,} keys | Rate: {:,} keys/s ({:.1f} MH/s) | Found: {} | Total Keys: {:,}".format(
                        time.strftime('%H:%M:%S'), 
                        total_tested, 
                        int(rate), 
                        mh,
                        total_found,
                        int(total_keys_abs)
                    )
                )
                total_tested = 0
                last_report = now

    except KeyboardInterrupt:
        logging.info("=== Summary ===")
        now = time.time()
        elapsed_total = now - run_start
        hms = time.strftime('%H:%M:%S', time.gmtime(elapsed_total))
        logging.info("Total runtime: {} ({:.1f} s)".format(hms, elapsed_total))
        logging.info("Total keys tested: {:,}".format(grand_total_tested))
        logging.info("Total matches found: {}".format(total_found))
        # Log skip mode at end
        try:
            mode_end = str(skip_mode_cfg).strip().lower() if skip_mode_cfg else ("avg_distance" if avg_dist_skip else ("adaptive" if adaptive_skip else "base"))
            logging.info("Skip mode: {}".format(mode_end))
        except Exception:
            pass
        if total_found > 0 and elapsed_total > 0:
            avg_time_per_find = elapsed_total / total_found
            finds_per_hour = 3600.0 / avg_time_per_find
            logging.info("Efficiency: avg {:.2f}s per match ({:.2f} matches/hour)".format(
                avg_time_per_find, finds_per_hour
            ))
        # Save last seed for future runs
        try:
            last_seed_path = os.path.join(output_dir, "last_seed.txt")
            # choose most advanced host seed across GPUs
            best = None
            best_n = -1
            for s in settings:
                b = bytes(s.key32)
                n = int.from_bytes(b, "big")
                if n > best_n:
                    best_n = n
                    best = b
            if best is not None:
                with open(last_seed_path, "w", encoding="utf-8") as lf:
                    lf.write("# Last seed saved at {}\n".format(time.strftime('%Y-%m-%d %H:%M:%S')))
                    lf.write("# Lines starting with #, //, ; are ignored. First token is parsed as base58.\n")
                    lf.write(b58encode(best).decode() + "\n")
                logging.info("Saved last seed to {}".format(last_seed_path))
        except Exception as e:
            logging.error("Failed to save last seed: {}".format(e))
        if record_all:
            logging.info("All batch seeds in: {}".format(txt_all))
        logging.info("Matches in: {}".format(csv_file))
        
        # Закрываем OpenCL контексты
        logging.info("Closing OpenCL contexts...")
        for searcher in searchers:
            try:
                searcher.cleanup()
            except:
                pass
        
        sys.exit(0)


@cli.command()
def show_device():
    """Show available OpenCL устройства."""
    for p_index, platform in enumerate(cl.get_platforms()):
        click.echo("Platform {}: {}".format(p_index, platform.name))
        for d_index, device in enumerate(platform.get_devices(device_type=cl.device_type.GPU)):
            click.echo("  - Device {}: {}".format(d_index, device.name))


@cli.command()
def benchmark():
    """Benchmark each GPU individually."""
    devices = get_all_gpu_devices()
    gpu_count = len(devices)
    logging.info("Benchmarking {} GPU device(s)".format(gpu_count))
    
    # Простой тест производительности
    kernel_src = load_kernel_source(("TEST",), "", True)
    
    for idx in range(gpu_count):
        try:
            setting = HostSetting(kernel_src, 24, idx, gpu_count)  # 24 bits для быстрого теста
            searcher = Searcher(kernel_src, idx, setting)
            
            # Тестируем 3 батча для стабильности
            total_time = 0
            total_keys = 0
            
            for _ in range(3):
                t0 = time.time()
                searcher.find(log_stats=False)
                t1 = time.time()
                batch_time = t1 - t0
                batch_keys = setting.global_work_size
                total_time += batch_time
                total_keys += batch_keys
            
            avg_rate = total_keys / total_time if total_time > 0 else 0
            logging.info("GPU {}: {:,} keys/s ({:.2f} MH/s)".format(
                idx, int(avg_rate), avg_rate / 1e6
            ))
            
        except Exception as e:
            logging.error("GPU {} error: {}".format(idx, e))


# --- Plotting utilities ---
@cli.command(name="plot-csv", context_settings={"show_default": True})
@click.option(
    "--csv",
    "csv_path",
    type=click.Path(exists=True, dir_okay=False),
    required=True,
    help="Path to metrics CSV (e.g., found-*/data/DEATH.csv)",
)
@click.option(
    "--output", "output_path", type=click.Path(dir_okay=False), default=None,
    help="Where to save the figure (PNG). Defaults to <CSV_DIR>/<CSV_NAME>.png"
)
@click.option(
    "--show/--no-show", default=False,
    help="Display the plot window after saving."
)
@click.option(
    "--dpi", type=int, default=140,
    help="DPI for the saved figure."
)
@click.option(
    "--title", type=str, default=None,
    help="Custom plot title. Defaults to the CSV file name."
)
def plot_csv(csv_path, output_path, show, dpi, title):
    """Plot metrics from a CSV file: X=totalKeys, Y1=distance + MA(5), Y2=matches/hour."""
    import os
    import csv as _csv
    try:
        import matplotlib
        if not show:
            matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
        from matplotlib.ticker import FuncFormatter
        # Dark theme
        try:
            plt.style.use("dark_background")
        except Exception:
            pass
    except Exception as e:
        click.echo("matplotlib is required for plotting: {}".format(e))
        sys.exit(1)

    # Read CSV
    xs = []             # totalKeys
    ys_distance = []    # distance
    ys_mph = []         # matchesPerHour
    dts = []            # parsed timestamps
    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = _csv.DictReader(f)
            for row in reader:
                try:
                    tk = int(row.get("totalKeys", "0") or 0)
                    dist = int(row.get("distance", "0") or 0)
                    mph = float(row.get("matchesPerHour", "0") or 0.0)
                except Exception:
                    continue
                # parse timestamp if present
                try:
                    ts_str = (row.get("time") or "").strip()
                    if ts_str:
                        from datetime import datetime as _dt
                        dts.append(_dt.strptime(ts_str, "%Y-%m-%d %H:%M:%S"))
                except Exception:
                    pass
                xs.append(tk)
                ys_distance.append(dist)
                ys_mph.append(mph)
    except Exception as e:
        click.echo("Failed to read CSV: {}".format(e))
        sys.exit(1)

    if not xs:
        click.echo("No rows parsed from CSV: {}".format(csv_path))
        sys.exit(1)

    # Sort by X in case input is unordered
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    xs = [xs[i] for i in order]
    ys_distance = [ys_distance[i] for i in order]
    ys_mph = [ys_mph[i] for i in order]

    # Compute 5-point trailing moving average of distance
    ys_ma5 = []
    try:
        for i in range(len(ys_distance)):
            start = 0 if i < 4 else i - 4
            window = ys_distance[start:i + 1]
            if window:
                ys_ma5.append(sum(window) / float(len(window)))
            else:
                ys_ma5.append(0)
    except Exception:
        ys_ma5 = [0 for _ in ys_distance]

    # Human number formatter
    def _fmt_num(n):
        try:
            n = float(n)
            absn = abs(n)
            for unit in ["", "K", "M", "B", "T", "P"]:
                if absn < 1000.0 or unit == "P":
                    if unit:
                        return "{:.0f}{}".format(n, unit)
                    return "{:,}".format(int(n))
                n /= 1000.0
                absn = abs(n)
        except Exception:
            return str(n)

    # Build figure
    fig, ax1 = plt.subplots(figsize=(11, 6))
    ax2 = ax1.twinx()

    # Primary Y: dots for distance and MA(5)
    l1, = ax1.plot(
        xs, ys_distance,
        linestyle="None", marker="o", markersize=3,
        label="Distance", color="tab:blue", alpha=0.85,
    )
    l2, = ax1.plot(
        xs, ys_ma5,
        linestyle="-", linewidth=1.5,
        label="MA(5) Distance", color="tab:orange", alpha=0.9,
    )
    # Secondary Y: line for matches/hour
    l3, = ax2.plot(xs, ys_mph, label="Matches/Hour", color="tab:green", linewidth=1.3)

    ax1.set_xlabel("Total Keys")
    ax1.set_ylabel("Distance / MA(5) (keys)")
    ax2.set_ylabel("Matches per Hour")

    ax1.grid(True, which="both", linestyle=":", linewidth=0.6, alpha=0.7)

    # Format ticks
    ax1.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: _fmt_num(x)))
    ax1.yaxis.set_major_formatter(FuncFormatter(lambda y, pos: _fmt_num(y)))
    ax2.yaxis.set_major_formatter(FuncFormatter(lambda y, pos: _fmt_num(y)))

    # Legend combining both axes
    lines = [l1, l2, l3]
    labels = [ln.get_label() for ln in lines]
    ax1.legend(lines, labels, loc="upper left")

    # Title
    if not title:
        title = os.path.basename(csv_path)
    ax1.set_title(title)

    # Add start/end/elapsed time annotation if timestamps exist
    try:
        if dts:
            start_dt = min(dts)
            end_dt = max(dts)
            elapsed = end_dt - start_dt
            secs = int(elapsed.total_seconds())
            hh = secs // 3600
            mm = (secs % 3600) // 60
            ss = secs % 60
            elapsed_str = f"{hh}:{mm:02d}:{ss:02d}"
            info_text = (
                f"Start: {start_dt.strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"End:   {end_dt.strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"Elapsed: {elapsed_str}\n"
                f"Total Keys: {max(xs)}"
            )
            ax1.text(
                0.98, 0.98, info_text,
                transform=ax1.transAxes,
                ha="right", va="top",
                fontsize=9, color="white",
                bbox=dict(facecolor="black", alpha=0.35, boxstyle="round,pad=0.3"),
            )
    except Exception:
        pass

    plt.tight_layout()

    # Resolve output path
    if output_path is None:
        base = os.path.splitext(os.path.basename(csv_path))[0] + ".png"
        output_path = os.path.join(os.path.dirname(csv_path), base)

    try:
        fig.savefig(output_path, dpi=dpi)
        click.echo("Saved plot to: {}".format(output_path))
    except Exception as e:
        click.echo("Failed to save figure: {}".format(e))
        sys.exit(1)

    if show:
        try:
            plt.show()
        except Exception:
            pass

# --- CPU self-test for multi-result flow ---
@cli.command(name="selftest-multi", context_settings={"show_default": True})
@click.option(
    "--starts-with",
    type=str,
    default=[],
    multiple=True,
    help="Префикс(ы) адреса (опционально, CPU-подбор может занять время).",
)
@click.option(
    "--count",
    type=int,
    default=5,
    help="Сколько совпадений сгенерировать (CPU).",
)
@click.option(
    "--is-case-sensitive/--no-case-sensitive",
    default=True,
    help="Учитывать регистр для префиксов (CPU тест).",
)
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False, dir_okay=True),
    default="./found",
    help="Куда сохранять результаты теста.",
)

def selftest_multi(starts_with, count, is_case_sensitive, output_dir):
    """CPU self-test: находит несколько совпадений, копит в памяти и сохраняет разом."""
    import os as _os
    import secrets as _secrets
    from nacl.signing import SigningKey as _SK
    try:
        _os.makedirs(output_dir, exist_ok=True)
        json_dir = _os.path.join(output_dir, "JSON")
        _os.makedirs(json_dir, exist_ok=True)
        prefixes_dir = _os.path.join(output_dir, "prefixes")
        _os.makedirs(prefixes_dir, exist_ok=True)
        csv_file = _os.path.join(output_dir, "results.csv")
        if not _os.path.isfile(csv_file):
            with open(csv_file, "w", encoding="utf-8") as f:
                f.write("publicKey,privateKey\n")

        def _match(pub: str) -> bool:
            if not starts_with:
                return True
            if is_case_sensitive:
                return any(pub.startswith(p) for p in starts_with)
            up = pub.upper()
            return any(up.startswith(p.upper()) for p in starts_with)

        # Accumulate matches in memory
        matches = []  # list of (pv_bytes, pub_str, priv_b58)
        logging.info("CPU self-test: generating {} match(es){}...".format(
            int(count),
            (" with prefixes {}".format(list(starts_with)) if starts_with else "")
        ))
        attempts = 0
        while len(matches) < int(count):
            attempts += 1
            pv_bytes = _secrets.token_bytes(32)
            try:
                pb_bytes = bytes(_SK(pv_bytes).verify_key)
            except Exception:
                continue
            pub = b58encode(pb_bytes).decode()
            if not _match(pub):
                continue
            full = pv_bytes + pb_bytes
            priv_b58 = b58encode(full).decode()
            matches.append((pv_bytes, pub, priv_b58))
            if len(matches) % 10 == 0:
                logging.info("Collected {} / {} ({} attempts)".format(len(matches), int(count), attempts))

        # Save all collected matches at once
        logging.info("Persisting {} collected match(es)".format(len(matches)))
        try:
            with open(csv_file, "a", encoding="utf-8") as f_csv:
                for pv_bytes, pub, priv_b58 in matches:
                    save_keypair(pv_bytes, json_dir)
                    f_csv.write("{},{}\n".format(pub, priv_b58))
                    # Save per‑prefix seeds too
                    try:
                        if starts_with:
                            matching_prefixes = [p for p in starts_with if (pub.startswith(p) if is_case_sensitive else pub.upper().startswith(p.upper()))]
                        else:
                            matching_prefixes = []
                        if matching_prefixes:
                            seed_b58 = b58encode(pv_bytes).decode()
                            for p in matching_prefixes:
                                pfile = _os.path.join(prefixes_dir, "{}{}.txt".format(p, ""))
                                with open(pfile, "a", encoding="utf-8") as pf:
                                    pf.write("{},{}\n".format(pub, seed_b58))
                    except Exception:
                        pass
        except Exception as e:
            logging.error("Self-test persist error: {}".format(e))
            raise SystemExit(1)

        click.echo("Self-test done. Wrote {} matches to {}".format(len(matches), csv_file))
    except Exception as e:
        logging.error("Self-test failed: {}".format(e))
        raise SystemExit(1)

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
    cli()


@cli.command(name="seed-from-private")
def seed_from_private():
    """Print the 32-byte seed (Base58) from a Base58 private key (32 or 64 bytes)."""
    try:
        pk_b58 = click.prompt("Paste private key (Base58)", type=str).strip()
        from base58 import b58decode
        raw = b58decode(pk_b58)
    except Exception as e:
        click.echo("Invalid Base58 private key: {}".format(e))
        raise SystemExit(1)
    if len(raw) == 64:
        seed = raw[:32]
    elif len(raw) == 32:
        seed = raw
    else:
        click.echo("Unexpected private key length: {} bytes (expected 32 or 64)".format(len(raw)))
        raise SystemExit(1)
    click.echo(b58encode(seed).decode())
