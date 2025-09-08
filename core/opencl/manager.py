# -*- coding: utf-8 -*-
import logging
import os
import sys
from typing import List, Tuple

import click
import pyopencl as cl

# Default to using the OpenCL cache for faster rebuilds; enable extra output optionally
if os.environ.get("PYOPENCL_COMPILER_OUTPUT") is None:
    os.environ["PYOPENCL_COMPILER_OUTPUT"] = "0"
if os.environ.get("PYOPENCL_NO_CACHE") is not None and os.environ["PYOPENCL_NO_CACHE"].strip().upper() == "TRUE":
    pass
else:
    # Ensure cache is enabled by default
    os.environ.pop("PYOPENCL_NO_CACHE", None)


def get_all_gpu_devices() -> List[cl.Device]:
    return [
        device
        for platform_obj in cl.get_platforms()
        for device in platform_obj.get_devices(device_type=cl.device_type.GPU)
    ]


def get_selected_gpu_devices(
    platform_id: int, device_ids: List[int]
) -> List[cl.Device]:
    platform_obj = cl.get_platforms()[platform_id]
    devices = platform_obj.get_devices(device_type=cl.device_type.GPU)
    return [devices[d_id] for d_id in device_ids]


def get_chosen_devices() -> Tuple[int, List[int]]:
    if "CHOSEN_OPENCL_DEVICES" in os.environ:
        platform_str, devices_str = os.environ.get("CHOSEN_OPENCL_DEVICES", "").split(
            ":"
        )
        return int(platform_str), list(map(int, devices_str.split(",")))
    platforms = cl.get_platforms()
    click.echo("Choose platform:")
    for idx, plat in enumerate(platforms):
        click.echo("{}. {}".format(idx, plat.name))
    platform_id = click.prompt(
        "Choice", default=0, type=click.IntRange(0, len(platforms) - 1)
    )
    all_devices = platforms[platform_id].get_devices(device_type=cl.device_type.GPU)
    if not all_devices:
        logging.error("Platform {} doesn't have GPU devices.".format(platform_id))
        sys.exit(-1)
    click.echo("Choose device(s):")
    for d_idx, device in enumerate(all_devices):
        click.echo("{}. {}".format(d_idx, device.name))
    device_ids_str = click.prompt("Choice, comma-separated", default="0", type=str)
    devices_list = list(map(int, device_ids_str.split(",")))
    click.echo(
        "Set environment variable CHOSEN_OPENCL_DEVICES='{}:{}' to avoid future prompts.".format(platform_id, device_ids_str)
    )
    return platform_id, devices_list
