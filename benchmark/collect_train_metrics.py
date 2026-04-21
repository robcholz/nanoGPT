#!/usr/bin/env python3
"""
Collect training metrics for train.py without editing the training script.

The wrapper executes the target script in-process and samples the active
train.py line every 100 ms to classify the current phase.
"""

from __future__ import annotations

import argparse
import ast
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from typing import Optional

try:
    import torch
except Exception:  # pragma: no cover - torch is expected in this repo
    torch = None


MB = 1024 * 1024


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def csv_value(value: Optional[float]) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


@dataclass(frozen=True)
class FunctionRange:
    start: int
    end: int

    def contains(self, lineno: int) -> bool:
        return self.start <= lineno <= self.end


class TrainSourceMap:
    def __init__(self, script_path: Path) -> None:
        source = script_path.read_text()
        tree = ast.parse(source, filename=str(script_path))
        self.lines = source.splitlines()
        self.functions = self._extract_function_ranges(tree)
        self.forward_lines = self._find_lines("logits, loss = model(X, Y)")
        self.backward_lines = self._find_lines("scaler.scale(loss).backward()")
        self.checkpoint_lines = self._find_lines("torch.save(checkpoint")
        self.optimizer_lines = set()
        for needle in (
            "scaler.unscale_(optimizer)",
            "clip_grad_norm_(",
            "scaler.step(optimizer)",
            "scaler.update()",
            "optimizer.zero_grad(set_to_none=True)",
        ):
            self.optimizer_lines.update(self._find_lines(needle))
        self.dataloader_lines = set()
        for needle in (
            "X, Y = get_batch('train')",
            "X, Y = get_batch(split)",
        ):
            self.dataloader_lines.update(self._find_lines(needle))

    def _extract_function_ranges(self, tree: ast.AST) -> dict[str, FunctionRange]:
        functions: dict[str, FunctionRange] = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.end_lineno is not None:
                functions[node.name] = FunctionRange(node.lineno, node.end_lineno)
        return functions

    def _find_lines(self, needle: str) -> set[int]:
        return {index + 1 for index, line in enumerate(self.lines) if needle in line}

    def classify(self, lineno: int) -> str:
        get_batch_range = self.functions.get("get_batch")
        if get_batch_range and get_batch_range.contains(lineno):
            return "dataloader"
        if lineno in self.checkpoint_lines:
            return "checkpoint"
        if lineno in self.backward_lines:
            return "backward"
        if lineno in self.optimizer_lines:
            return "optimizer"
        if lineno in self.forward_lines:
            return "forward"
        if lineno in self.dataloader_lines:
            return "dataloader"
        return "idle"


class ProcSampler:
    def __init__(self, script_path: Path, interval_s: float, output_path: Path) -> None:
        self.script_path = script_path.resolve()
        self.interval_s = interval_s
        self.output_path = output_path
        self.source_map = TrainSourceMap(self.script_path)
        self.main_thread_id = threading.main_thread().ident
        self.stop_event = threading.Event()
        self.writer = None
        self.file_handle = None
        self.thread = None
        self.last_sample = None

    def start(self) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.file_handle = self.output_path.open("w", newline="")
        self.writer = csv.DictWriter(
            self.file_handle,
            fieldnames=[
                "timestamp",
                "step",
                "phase",
                "cpu_util_percent",
                "gpu_util_percent",
                "gpu_mem_mb",
                "host_mem_mb",
                "disk_read_mb_s",
                "disk_write_mb_s",
            ],
        )
        self.writer.writeheader()
        self.last_sample = self._read_counters()
        self.thread = threading.Thread(target=self._run, name="train-metrics-sampler", daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=max(self.interval_s * 5, 1.0))
        if self.file_handle is not None:
            self.file_handle.flush()
            self.file_handle.close()

    def _run(self) -> None:
        while not self.stop_event.is_set():
            time.sleep(self.interval_s)
            current = self._read_counters()
            previous = self.last_sample
            self.last_sample = current
            if previous is None or self.writer is None or self.file_handle is None:
                continue

            elapsed = max(current["wall_time"] - previous["wall_time"], 1e-9)
            cpu_delta = current["cpu_time"] - previous["cpu_time"]
            read_delta = current["read_bytes"] - previous["read_bytes"]
            write_delta = current["write_bytes"] - previous["write_bytes"]
            step, phase = self._current_step_and_phase()
            gpu_util, gpu_mem = self._read_gpu_metrics()

            self.writer.writerow(
                {
                    "timestamp": utc_now(),
                    "step": step,
                    "phase": phase,
                    "cpu_util_percent": csv_value(100.0 * cpu_delta / elapsed),
                    "gpu_util_percent": csv_value(gpu_util),
                    "gpu_mem_mb": csv_value(gpu_mem),
                    "host_mem_mb": csv_value(current["rss_bytes"] / MB),
                    "disk_read_mb_s": csv_value(read_delta / elapsed / MB),
                    "disk_write_mb_s": csv_value(write_delta / elapsed / MB),
                }
            )
            self.file_handle.flush()

    def _current_step_and_phase(self) -> tuple[int, str]:
        frame = sys._current_frames().get(self.main_thread_id)
        while frame is not None:
            filename = os.path.abspath(frame.f_code.co_filename)
            if filename == str(self.script_path):
                step = frame.f_locals.get("iter_num", frame.f_globals.get("iter_num", -1))
                if not isinstance(step, int):
                    step = -1
                return step, self.source_map.classify(frame.f_lineno)
            frame = frame.f_back
        return -1, "idle"

    def _read_counters(self) -> dict[str, float]:
        cpu_time = self._read_process_cpu_time()
        read_bytes, write_bytes = self._read_host_disk_bytes()
        return {
            "wall_time": time.monotonic(),
            "cpu_time": cpu_time,
            "read_bytes": read_bytes,
            "write_bytes": write_bytes,
            "rss_bytes": self._read_process_rss_bytes(),
        }

    def _read_process_cpu_time(self) -> float:
        with open("/proc/self/stat") as handle:
            fields = handle.read().split()
        ticks = os.sysconf(os.sysconf_names["SC_CLK_TCK"])
        return (int(fields[13]) + int(fields[14])) / ticks

    def _read_process_rss_bytes(self) -> int:
        with open("/proc/self/statm") as handle:
            fields = handle.read().split()
        page_size = os.sysconf("SC_PAGE_SIZE")
        return int(fields[1]) * page_size

    def _read_host_disk_bytes(self) -> tuple[int, int]:
        read_sectors = 0
        write_sectors = 0
        devices = self._host_disk_devices()
        with open("/proc/diskstats") as handle:
            for line in handle:
                fields = line.split()
                if len(fields) < 10 or fields[2] not in devices:
                    continue
                read_sectors += int(fields[5])
                write_sectors += int(fields[9])
        sector_size = 512
        return read_sectors * sector_size, write_sectors * sector_size

    def _host_disk_devices(self) -> set[str]:
        if not hasattr(self, "_disk_devices"):
            devices = set()
            for name in os.listdir("/sys/block"):
                if name.startswith(("loop", "ram", "zram", "dm-", "md")):
                    continue
                devices.add(name)
            self._disk_devices = devices
        return self._disk_devices

    def _read_gpu_metrics(self) -> tuple[Optional[float], Optional[float]]:
        if torch is None or not torch.cuda.is_available():
            return None, None

        device = self._current_cuda_device()
        gpu_util = None
        gpu_mem = None

        try:
            gpu_util = float(torch.cuda.utilization(device))
        except Exception:
            gpu_util = self._read_gpu_util_via_nvidia_smi(device)

        try:
            gpu_mem = torch.cuda.memory_allocated(device) / MB
        except Exception:
            gpu_mem = None

        return gpu_util, gpu_mem

    def _read_gpu_util_via_nvidia_smi(self, device: int) -> Optional[float]:
        try:
            proc = subprocess.run(
                [
                    "nvidia-smi",
                    "-i",
                    str(device),
                    "--query-gpu=utilization.gpu",
                    "--format=csv,noheader,nounits",
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=1.0,
            )
        except Exception:
            return None
        output = proc.stdout.strip().splitlines()
        if not output:
            return None
        try:
            return float(output[0].strip())
        except ValueError:
            return None

    def _current_cuda_device(self) -> int:
        frame = sys._current_frames().get(self.main_thread_id)
        while frame is not None:
            filename = os.path.abspath(frame.f_code.co_filename)
            if filename == str(self.script_path):
                device = frame.f_globals.get("device", "cuda")
                if isinstance(device, str) and device.startswith("cuda:"):
                    try:
                        return int(device.split(":", 1)[1])
                    except ValueError:
                        break
                break
            frame = frame.f_back
        return torch.cuda.current_device()


def rank_aware_output_path(output_path: Path) -> Path:
    world_size = os.environ.get("WORLD_SIZE")
    rank = os.environ.get("RANK")
    if world_size is None or rank is None or int(world_size) <= 1:
        return output_path
    suffix = output_path.suffix or ".csv"
    stem = output_path.stem if output_path.suffix else output_path.name
    return output_path.with_name(f"{stem}.rank{rank}{suffix}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default="benchmark/train_metrics.csv",
        help="CSV file to write. In DDP, .rankN is inserted automatically.",
    )
    parser.add_argument(
        "--interval-ms",
        type=int,
        default=100,
        help="Sampling interval in milliseconds.",
    )
    parser.add_argument(
        "script",
        nargs="?",
        default="train.py",
        help="Python training script to execute.",
    )
    parser.add_argument(
        "train_args",
        nargs=argparse.REMAINDER,
        help="Arguments passed through to the training script.",
    )
    args = parser.parse_args()
    if args.train_args and args.train_args[0] == "--":
        args.train_args = args.train_args[1:]
    return args


def run_script(script_path: Path, train_args: list[str]) -> None:
    old_argv = sys.argv[:]
    old_sys_path = sys.path[:]
    try:
        sys.argv = [str(script_path)] + train_args
        sys.path.insert(0, str(script_path.parent))
        globals_dict = {
            "__name__": "__main__",
            "__file__": str(script_path),
            "__package__": None,
            "__cached__": None,
        }
        source = script_path.read_text()
        code = compile(source, str(script_path), "exec")
        exec(code, globals_dict, globals_dict)
    finally:
        sys.argv = old_argv
        sys.path[:] = old_sys_path


def main() -> int:
    args = parse_args()
    script_path = Path(args.script).resolve()
    output_path = rank_aware_output_path(Path(args.output).resolve())

    sampler = ProcSampler(
        script_path=script_path,
        interval_s=args.interval_ms / 1000.0,
        output_path=output_path,
    )
    sampler.start()
    try:
        run_script(script_path, args.train_args)
    finally:
        sampler.stop()

    print(f"wrote metrics to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
