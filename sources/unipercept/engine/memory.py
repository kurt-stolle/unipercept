"""
Memory tracker for the engine. This is a modified version of the original memory tracker from
https://github.com/huggingface/transformers/blob/main/src/transformers/engine_utils.py
"""

from __future__ import annotations

import dataclasses as D
import gc
import threading
import typing as T

import psutil
import torch

__all__ = ["MemoryTracker"]


MetricsDictType: T.TypeAlias = T.Dict[str, T.Any]
MetricsTrainStage: T.TypeAlias = T.Literal["init", "train", "eval", "pred"]


@D.dataclass(slots=True, kw_only=True)
class MemoryTracker:
    """
    Tracks process memory usage during the training process.
    """

    enabled: bool = D.field(
        default=False,
        init=True,
        metadata={
            "help": "Whether to run analysis that tracks memory usage metrics, i.e. when disabled this class will be a no-op."
        },
    )
    process: psutil.Process = D.field(default_factory=psutil.Process, init=False)
    cur_stage: T.Optional[str] = D.field(default=None, init=False)
    cpu: MetricsDictType = D.field(default_factory=dict, init=False)
    gpu: MetricsDictType = D.field(default_factory=dict, init=False)
    init_reported: bool = D.field(default=False, init=False)
    gpu_mem_used_at_start: T.Optional[int] = D.field(default=None, init=False)
    gpu_mem_used_peak: T.Optional[int] = D.field(default=None, init=False)
    gpu_mem_used_now: T.Optional[int] = D.field(default=None, init=False)
    cpu_mem_used_at_start: T.Optional[int] = D.field(default=None, init=False)
    cpu_mem_used_peak: T.Optional[int] = D.field(default=None, init=False)
    cpu_mem_used_now: T.Optional[int] = D.field(default=None, init=False)
    peak_monitoring: T.Optional[bool] = D.field(default=None, init=False)

    def cpu_mem_used(self):
        """get resident set size memory for the current process"""
        return self.process.memory_info().rss

    def peak_monitor_func(self, event: threading.Event):
        self.cpu_mem_used_peak = -1

        event.set()

        while True:
            self.cpu_mem_used_peak = max(self.cpu_mem_used(), self.cpu_mem_used_peak)

            # can't sleep or will not catch the peak right (this comment is here on purpose)
            # time.sleep(0.001) # 1msec

            if not self.peak_monitoring:
                break

    def start(self, stage: str):
        if not self.enabled:
            return

        # deal with nested calls of eval during train - simply ignore those
        if self.cur_stage is not None and self.cur_stage != stage:
            return

        self.cur_stage = stage

        gc.collect()

        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

        self.gpu_mem_used_at_start = torch.cuda.memory_allocated()
        self.cpu_mem_used_at_start = self.cpu_mem_used()
        self.peak_monitoring = True
        peak_monitor_event = threading.Event()
        peak_monitor_thread = threading.Thread(
            target=self.peak_monitor_func, args=(peak_monitor_event,)
        )
        peak_monitor_thread.daemon = True
        peak_monitor_thread.start()
        peak_monitor_event.wait()

    def stop(self, stage: str):
        """
        Stop tracking and update the metrics.
        """

        # deal with nested calls of eval during train - simply ignore those
        if self.cur_stage is not None and self.cur_stage != stage:
            return

        # this sends a signal to peak_monitor_func to complete its loop
        self.peak_monitoring = False

        # first ensure all objects get collected and their memory is freed
        gc.collect()

        if torch is not None:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif is_torch_xpu_available():
                torch.xpu.empty_cache()

        # concepts:
        # - alloc_delta:  the difference of allocated memory between the end and the start
        # - peaked_delta: the difference between the peak memory and the current memory
        # in order to know how much memory the measured code consumed one needs to sum these two

        # gpu
        if torch.cuda.is_available():
            self.gpu_mem_used_now = torch.cuda.memory_allocated()
            self.gpu_mem_used_peak = torch.cuda.max_memory_allocated()
        else:
            raise ValueError("No available GPU device found!")

        self.gpu[self.cur_stage] = {
            "begin": self.gpu_mem_used_at_start,
            "end": self.gpu_mem_used_now,
            "alloc": (self.gpu_mem_used_now - self.gpu_mem_used_at_start),
            "peaked": max(0, self.gpu_mem_used_peak - self.gpu_mem_used_now),
        }

        # cpu
        self.cpu_mem_used_now = self.cpu_mem_used()
        self.cpu[self.cur_stage] = {
            "begin": self.cpu_mem_used_at_start,
            "end": self.cpu_mem_used_now,
            "alloc": (self.cpu_mem_used_now - self.cpu_mem_used_at_start),
            "peaked": max(0, self.cpu_mem_used_peak - self.cpu_mem_used_now),
        }

        # reset - cycle finished
        self.cur_stage = None

    def update_metrics(self, stage: int, metrics: dict[str, T.Any]):
        """
        Updates the metrics

        Parameters
        ----------
        stage : str
            The current stage of the training process
        metrics : dict
            The metrics to update

        """
        if not self.enabled:
            return

        # deal with nested calls of eval during train - simply ignore those
        if self.cur_stage is not None and self.cur_stage != stage:
            return

        # since we don't have a way to return init metrics, we push them into the first of train/val/predict
        stages = [stage]
        if not self.init_reported:
            stages.insert(0, "init")
            self.init_reported = True

        for stage in stages:
            for t in ["alloc", "peaked"]:
                if stage in self.cpu and t in self.cpu[stage]:
                    metrics[f"{stage}_mem_cpu_{t}_delta"] = self.cpu[stage][t]
                if torch is not None and stage in self.gpu and t in self.gpu[stage]:
                    metrics[f"{stage}_mem_gpu_{t}_delta"] = self.gpu[stage][t]
            # if we need additional debug info, enable the following
            # for t in ["begin", "end"]:
            #     if stage in self.cpu and t in self.cpu[stage]:
            #         metrics[f"{stage}_mem_cpu_{t}"] = self.cpu[stage][t]
            #     if torch is not None and stage in self.gpu and t in self.gpu[stage]:
            #         metrics[f"{stage}_mem_gpu_{t}"] = self.gpu[stage][t]

        # since memory can be allocated before init, and it might be difficult to track overall
        # memory usage, in particular for GPU, let's report memory usage at the point init was called
        if stages[0] == "init":
            metrics["before_init_mem_cpu"] = self.cpu["init"]["begin"]
            if torch is not None:
                metrics["before_init_mem_gpu"] = self.gpu["init"]["begin"]
            # if we also wanted to report any additional memory allocations in between init and
            # whatever the next stage was we could also report this:
            # if self.cpu["init"]["end"] != self.cpu[stage]["begin"]:
            #     metrics[f"after_init_mem_cpu_delta"] = self.cpu[stage]["begin"] - self.cpu["init"]["end"]
            # if torch is not None and self.gpu["init"]["end"] != self.gpu[stage]["begin"]:
            #     metrics[f"after_init_mem_gpu_delta"] = self.gpu[stage]["begin"] - self.gpu["init"]["end"]

    def stop_and_update_metrics(self, stage: str, metrics=None):
        """
        Combine stop and metrics update in one call for simpler code

        Parameters
        ----------
        metrics : dict, optional
            The metrics to update, by default None

        """
        if not self.enabled:
            return

        self.stop(stage)

        # init doesn't have metrics to update so we just save that data for later stages to retrieve
        if metrics is not None:
            self.update_metrics(stage, metrics)
