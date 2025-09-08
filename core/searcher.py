# -*- coding: utf-8 -*-
import logging
import time
from typing import List, Optional, Tuple

import numpy as np
import pyopencl as cl

from core.config import HostSetting
from core.opencl.manager import get_all_gpu_devices, get_selected_gpu_devices


class Searcher:
    def __init__(
        self,
        kernel_source: str,
        index: int,
        setting: HostSetting,
        chosen_devices: Optional[Tuple[int, List[int]]] = None,
        build_options: Optional[List[str]] = None,
        nv_max_rregcount: Optional[int] = None,
        max_results_per_batch: int = 256,
        # optional extended behaviors
        record_all: bool = False,
        record_all_mode: str = "tiled",
    ):
        # Выбор устройства
        if chosen_devices is None:
            devices = get_all_gpu_devices()
        else:
            devices = get_selected_gpu_devices(*chosen_devices)
        enabled = devices[index]

        self.context = cl.Context([enabled])
        self.gpu_chunks = len(devices)
        self.command_queue = cl.CommandQueue(self.context)
        self.setting = setting
        self.index = index
        self.display_index = (
            index if chosen_devices is None else chosen_devices[1][index]
        )
        self.prev_time = None
        self.is_nvidia = "NVIDIA" in enabled.platform.name.upper()

        # Компиляция OpenCL‑ядра с опциями
        options: List[str] = []
        if build_options:
            options.extend(build_options)
        if self.is_nvidia and nv_max_rregcount:
            try:
                options.append("-cl-nv-maxrregcount={}".format(int(nv_max_rregcount)))
            except Exception:
                pass
        try:
            program = cl.Program(self.context, kernel_source).build(options=options or None)
        except Exception as e:
            # Fallback: filter unsupported flags (e.g., -O3 on some NVIDIA OpenCL)
            allowed_flags = {
                "-cl-fast-relaxed-math",
                "-cl-mad-enable",
                "-cl-no-signed-zeros",
                "-cl-unsafe-math-optimizations",
                "-cl-finite-math-only",
                "-cl-denorms-are-zero",
            }
            filtered = []
            for opt in (options or []):
                if opt.startswith("-D"):
                    filtered.append(opt)
                elif opt in allowed_flags:
                    filtered.append(opt)
                elif opt.startswith("-cl-nv-maxrregcount") and self.is_nvidia:
                    filtered.append(opt)
                else:
                    logging.warning("Filtering unsupported OpenCL option: {}".format(opt))
            program = cl.Program(self.context, kernel_source).build(options=filtered or None)
        self.kernel = cl.Kernel(program, "generate_pubkey")
        try:
            self._kernel_num_args = int(getattr(self.kernel, 'num_args', self.kernel.get_info(cl.kernel_info.NUM_ARGS)))
        except Exception:
            self._kernel_num_args = 6

        # Буферы
        self.memobj_key32 = cl.Buffer(
            self.context,
            cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
            32 * np.ubyte().itemsize,
            hostbuf=self.setting.key32,
        )
        # Results buffer: count (uint32) + items (max_results * 33 bytes)
        self.max_results = int(max_results_per_batch) if int(max_results_per_batch) > 0 else 256
        self.memobj_output_items = cl.Buffer(
            self.context,
            cl.mem_flags.READ_WRITE,
            self.max_results * 33 * np.ubyte().itemsize,
        )
        self.memobj_output_count = cl.Buffer(
            self.context,
            cl.mem_flags.READ_WRITE,
            np.int32().itemsize,
        )

        # Локальная копия вывода
        self.output_items = np.zeros(self.max_results * 33, dtype=np.ubyte)
        self.output_count = np.zeros(1, dtype=np.int32)

        # Аргументы ядра
        self.kernel.set_arg(0, self.memobj_key32)
        self.kernel.set_arg(1, self.memobj_output_items)
        self.kernel.set_arg(2, self.memobj_output_count)
        # scalar args: occupied_bytes (uchar), group_offset (uint), max_results (uint)
        self.kernel.set_arg(3, np.uint8(self.setting.iteration_bytes))
        self.kernel.set_arg(4, np.uint32(self.index))
        self.kernel.set_arg(5, np.uint32(self.max_results))
        # Extended flags
        self.record_all = bool(record_all)
        self.record_all_mode = (record_all_mode or "tiled").strip().lower()
        self._tile_index = 0
        self._tile_start = np.uint32(0)
        self._tile_size = np.uint32(self.max_results)
        if self._kernel_num_args >= 7:
            self.kernel.set_arg(6, np.uint8(1 if self.record_all else 0))
        if self._kernel_num_args >= 9:
            self.kernel.set_arg(7, self._tile_start)
            self.kernel.set_arg(8, self._tile_size)
        # Compatibility: optional extra args (record_all, tile_start, tile_size)
        try:
            num_args = int(getattr(self.kernel, 'num_args', self.kernel.get_info(cl.kernel_info.NUM_ARGS)))
        except Exception:
            num_args = 6
        if num_args >= 7:
            # record_all flag default = 0
            self.kernel.set_arg(6, np.uint8(0))
        if num_args >= 9:
            # tile_start = 0, tile_size = max_results by default
            self.kernel.set_arg(7, np.uint32(0))
            self.kernel.set_arg(8, np.uint32(self.max_results))

    def start(self):
        """Запускает батч асинхронно, возвращает событие выполнения ядра."""
        # Обновить ключ на устройстве (неблокирующе)
        # Use device seed view (may be reversed)
        device_seed = self.setting.get_device_seed()
        evt_copy = cl.enqueue_copy(self.command_queue, self.memobj_key32, device_seed)
        # Обнулить выходные буферы на устройстве
        evt_fill_items = cl.enqueue_fill_buffer(
            self.command_queue,
            self.memobj_output_items,
            np.uint8(0),
            0,
            self.output_items.nbytes,
            wait_for=[evt_copy],
        )
        evt_fill_count = cl.enqueue_fill_buffer(
            self.command_queue,
            self.memobj_output_count,
            np.uint8(0),
            0,
            self.output_count.nbytes,
            wait_for=[evt_fill_items],
        )
        # Update tiling for record-all if supported
        if self.record_all and self._kernel_num_args >= 7:
            total = int(self.setting.global_work_size)
            cap = int(self.max_results)
            if self.record_all_mode == "monolithic":
                start = 0
                size = total
            else:
                num_tiles = max(1, (total + cap - 1) // cap)
                if self._tile_index >= num_tiles:
                    self._tile_index = 0
                start = int(self._tile_index) * cap
                remain = total - start
                size = cap if cap <= remain else remain
                if size <= 0:
                    size = min(cap, total)
            self._tile_start = np.uint32(start)
            self._tile_size = np.uint32(size)
            if self._kernel_num_args >= 7:
                self.kernel.set_arg(6, np.uint8(1))
            if self._kernel_num_args >= 9:
                self.kernel.set_arg(7, self._tile_start)
                self.kernel.set_arg(8, self._tile_size)
        # Запустить ядро
        global_ws = self.setting.global_work_size
        evt_kernel = cl.enqueue_nd_range_kernel(
            self.command_queue,
            self.kernel,
            (global_ws,),
            (self.setting.local_work_size,),
            wait_for=[evt_fill_count],
        )
        return evt_kernel

    def finish(self, kernel_event, advance_across_all_gpus: bool = True) -> np.ndarray:
        """Дожидается завершения батча, считывает результат и увеличивает ключ."""
        try:
            if self.record_all and self._kernel_num_args >= 7:
                # Copy only the current tile
                size_bytes = int(self._tile_size) * 33 * np.ubyte().itemsize
                if self.output_items.size < int(self._tile_size) * 33:
                    self.output_items = np.zeros(int(self._tile_size) * 33, dtype=np.ubyte)
                dest_view = self.output_items[: (size_bytes // np.ubyte().itemsize)]
                evt_items = cl.enqueue_copy(self.command_queue, dest_view, self.memobj_output_items)
                evt_items.wait()
            else:
                # Read count first
                evt_cnt = cl.enqueue_copy(self.command_queue, self.output_count, self.memobj_output_count, wait_for=[kernel_event])
                evt_cnt.wait()
                # Read items buffer
                evt_items = cl.enqueue_copy(self.command_queue, self.output_items, self.memobj_output_items)
                evt_items.wait()
        except KeyboardInterrupt:
            self.command_queue.finish()
            raise
        # Увеличиваем seed согласно выбранной стратегии
        if advance_across_all_gpus:
            self.setting.advance_key32(batches=1, across_all_gpus=True)
        else:
            self.setting.advance_key32(batches=1, across_all_gpus=False)
        # Build list of matches (each 32-byte private key)
        matches: List[bytes] = []
        if self.record_all and self._kernel_num_args >= 7:
            try:
                n = int(self._tile_size)
                for i in range(n):
                    base = i * 33
                    pv = bytes(self.output_items[base + 1: base + 33])
                    if len(pv) == 32:
                        matches.append(pv)
            except Exception:
                matches = []
            # next tile
            self._tile_index += 1
        else:
            try:
                count = int(self.output_count[0])
                if count > self.max_results:
                    count = self.max_results
                for i in range(count):
                    base = i * 33
                    pv = bytes(self.output_items[base + 1: base + 33])
                    if len(pv) == 32 and self.output_items[base + 0] != 0:
                        matches.append(pv)
            except Exception:
                matches = []
        return matches

    def find(self, log_stats: bool = True) -> np.ndarray:
        start_time = time.time()
        evt = self.start()
        out = self.finish(evt)
        self.prev_time = time.time() - start_time
        if log_stats:
            global_ws = self.setting.global_work_size
            mhz = global_ws / (self.prev_time * 1e6) if self.prev_time > 0 else 0
            logging.info("GPU {} Speed: {:.2f} MH/s ({:,} keys/s)".format(
                self.display_index, mhz, int(global_ws / self.prev_time) if self.prev_time > 0 else 0
            ))
        return out

    def cleanup(self):
        """Корректно закрывает OpenCL ресурсы"""
        try:
            if hasattr(self, 'command_queue'):
                self.command_queue.finish()
                self.command_queue.release()
            if hasattr(self, 'context'):
                self.context.release()
        except:
            pass


def multi_gpu_init(
    index: int,
    setting: HostSetting,
    gpu_counts: int,
    stop_flag,
    lock,
    chosen_devices: Optional[Tuple[int, List[int]]] = None,
) -> List[int]:
    try:
        searcher = Searcher(
            kernel_source=setting.kernel_source,
            index=index,
            setting=setting,
            chosen_devices=chosen_devices,
        )
        i = 0
        st = time.time()
        while True:
            result = searcher.find(i == 0)
            if result:
                # Return first match in legacy format [1,<32 bytes>] for compatibility
                first = result[0]
                if isinstance(first, (bytes, bytearray)) and len(first) == 32:
                    return [1] + list(first)
                else:
                    return [0]
            if time.time() - st > max(gpu_counts, 1):
                i = 0
                st = time.time()
                with lock:
                    if stop_flag.value:
                        return result.tolist()
            else:
                i += 1
    except Exception:
        logging.exception("multi_gpu_init error")
    return [0]


def save_result(outputs: List[List[int]], output_dir: str) -> int:
    from core.utils.crypto import save_keypair

    result_count = 0
    for output in outputs:
        if not output[0]:
            continue
        result_count += 1
        pv = bytes(output[1:])
        save_keypair(pv, output_dir)
    return result_count
