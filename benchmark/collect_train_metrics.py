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
        self.checkpoint_lines = set()
        for needle in (
            "checkpoint = {",
            "'model': raw_model.state_dict(),",
            "'optimizer': optimizer.state_dict(),",
            "print(f\"saving checkpoint to {out_dir}\")",
            "torch.save(checkpoint",
        ):
            self.checkpoint_lines.update(self._find_lines(needle))
        self.loss_scale_lines = self._find_lines("loss = loss / gradient_accumulation_steps")
        self.eval_lines = set()
        for needle in (
            "if iter_num % eval_interval == 0 and master_process:",
            "losses = estimate_loss()",
            "print(f\"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}\")",
            "wandb.log({",
            "if losses['val'] < best_val_loss or always_save_checkpoint:",
            "best_val_loss = losses['val']",
        ):
            self.eval_lines.update(self._find_lines(needle))
        self.logging_lines = set()
        for needle in (
            "lossf = loss.item() * gradient_accumulation_steps",
            "mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)",
            'print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")',
        ):
            self.logging_lines.update(self._find_lines(needle))
        self.optimizer_lines = set()
        for needle in (
            "if grad_clip != 0.0:",
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
        self.eval_range = self._find_function_range("estimate_loss")

    def _extract_function_ranges(self, tree: ast.AST) -> dict[str, FunctionRange]:
        functions: dict[str, FunctionRange] = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.end_lineno is not None:
                functions[node.name] = FunctionRange(node.lineno, node.end_lineno)
        return functions

    def _find_function_range(self, name: str) -> Optional[FunctionRange]:
        return self.functions.get(name)

    def _find_lines(self, needle: str) -> set[int]:
        return {index + 1 for index, line in enumerate(self.lines) if needle in line}

    def classify(self, lineno: int) -> str:
        get_batch_range = self.functions.get("get_batch")
        if get_batch_range and get_batch_range.contains(lineno):
            return "dataloader"
        if self.eval_range and self.eval_range.contains(lineno):
            return "eval"
        if lineno in self.eval_lines:
            return "eval"
        if lineno in self.checkpoint_lines:
            return "checkpoint"
        if lineno in self.backward_lines:
            return "backward"
        if lineno in self.optimizer_lines:
            return "optimizer"
        if lineno in self.forward_lines:
            return "forward"
        if lineno in self.loss_scale_lines:
            return "forward"
        if lineno in self.dataloader_lines:
            return "dataloader"
        if lineno in self.logging_lines:
            return "idle"
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
                "interval_s",
                "cpu_util_percent",
                "gpu_util_percent",
                "gpu_mem_mb",
                "gpu_power_w",
                "host_mem_mb",
                "disk_read_mb_s",
                "disk_write_mb_s",
            ],
        )
        self.writer.writeheader()
        self.last_sample = self._read_sample()
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
            current = self._read_sample()
            previous = self.last_sample
            self.last_sample = current
            if previous is None or self.writer is None or self.file_handle is None:
                continue

            elapsed = max(current["wall_time"] - previous["wall_time"], 1e-9)
            cpu_delta = current["cpu_time"] - previous["cpu_time"]
            read_delta = current["read_bytes"] - previous["read_bytes"]
            write_delta = current["write_bytes"] - previous["write_bytes"]

            self.writer.writerow(
                {
                    "timestamp": previous["timestamp"],
                    "step": previous["step"],
                    "phase": previous["phase"],
                    "interval_s": csv_value(elapsed),
                    "cpu_util_percent": csv_value(100.0 * cpu_delta / elapsed),
                    "gpu_util_percent": csv_value(previous["gpu_util"]),
                    "gpu_mem_mb": csv_value(previous["gpu_mem"]),
                    "gpu_power_w": csv_value(previous["gpu_power"]),
                    "host_mem_mb": csv_value(previous["rss_bytes"] / MB),
                    "disk_read_mb_s": csv_value(read_delta / elapsed / MB),
                    "disk_write_mb_s": csv_value(write_delta / elapsed / MB),
                }
            )
            self.file_handle.flush()

    def _current_step_and_phase(self) -> tuple[int, str]:
        frame = sys._current_frames().get(self.main_thread_id)
        train_frames = []
        while frame is not None:
            filename = os.path.abspath(frame.f_code.co_filename)
            if filename == str(self.script_path):
                train_frames.append(frame)
            frame = frame.f_back
        if not train_frames:
            return -1, "idle"

        phase = "idle"
        for train_frame in train_frames:
            phase = self.source_map.classify(train_frame.f_lineno)
            if phase != "idle":
                break

        outermost_frame = train_frames[-1]
        step = outermost_frame.f_globals.get("iter_num", -1)
        if not isinstance(step, int):
            step = -1
        return step, phase

    def _read_sample(self) -> dict[str, float | int | str | None]:
        step, phase = self._current_step_and_phase()
        gpu_util, gpu_mem, gpu_power = self._read_gpu_metrics()
        cpu_time = self._read_process_cpu_time()
        read_bytes, write_bytes = self._read_host_disk_bytes()
        return {
            "timestamp": utc_now(),
            "wall_time": time.monotonic(),
            "step": step,
            "phase": phase,
            "cpu_time": cpu_time,
            "read_bytes": read_bytes,
            "write_bytes": write_bytes,
            "rss_bytes": self._read_process_rss_bytes(),
            "gpu_util": gpu_util,
            "gpu_mem": gpu_mem,
            "gpu_power": gpu_power,
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

    def _read_gpu_metrics(self) -> tuple[Optional[float], Optional[float], Optional[float]]:
        if torch is None or not torch.cuda.is_available():
            return None, None, None

        device = self._current_cuda_device()
        gpu_util = None
        gpu_mem = None
        gpu_power = None
        smi_util, smi_power = self._read_gpu_telemetry_via_nvidia_smi(device)

        try:
            gpu_util = float(torch.cuda.utilization(device))
        except Exception:
            gpu_util = smi_util

        try:
            gpu_mem = torch.cuda.memory_allocated(device) / MB
        except Exception:
            gpu_mem = None

        gpu_power = smi_power

        return gpu_util, gpu_mem, gpu_power

    def _read_gpu_telemetry_via_nvidia_smi(self, device: int) -> tuple[Optional[float], Optional[float]]:
        try:
            proc = subprocess.run(
                [
                    "nvidia-smi",
                    "-i",
                    str(device),
                    "--query-gpu=utilization.gpu,power.draw",
                    "--format=csv,noheader,nounits",
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=1.0,
            )
        except Exception:
            return None, None
        output = proc.stdout.strip().splitlines()
        if not output:
            return None, None
        values = [part.strip() for part in output[0].split(",")]
        if len(values) != 2:
            return None, None
        return self._parse_optional_float(values[0]), self._parse_optional_float(values[1])

    def _parse_optional_float(self, value: str) -> Optional[float]:
        if not value or value.upper() == "N/A":
            return None
        try:
            return float(value)
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
