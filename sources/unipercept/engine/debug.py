from __future__ import annotations

import collections
import enum

import torch

from unipercept.log import get_logger

logger = get_logger(__name__)


class DebugMode(enum.IntFlag):
    NONE = 0
    UNDERFLOW_OVERFLOW = enum.auto()
    INSPECT_MODEL = enum.auto()


class DebugUnderflowOverflow:
    def __init__(
        self,
        model,
        max_frames_to_save=21,
        trace_batch_nums=[],
        abort_after_batch_num=None,
    ):
        self.model = model
        self.trace_batch_nums = trace_batch_nums
        self.abort_after_batch_num = abort_after_batch_num

        # keep a LIFO buffer of frames to dump as soon as inf/nan is encountered to give context to the problem emergence
        self.frames = collections.deque([], max_frames_to_save)
        self.frame = []
        self.batch_number = 0
        self.total_calls = 0
        self.detected_overflow = False
        self.prefix = ""

        self.analyse_model()

        self.register_forward_hook()

    def save_frame(self, frame=None):
        if frame is not None:
            self.expand_frame(frame)
        self.frames.append("\n".join(self.frame))
        self.frame = []  # start a new frame

    def expand_frame(self, line):
        self.frame.append(line)

    def trace_frames(self):
        print("\n".join(self.frames))
        self.frames = []

    def reset_saved_frames(self):
        self.frames = []

    def dump_saved_frames(self):
        print(f"\nDetected inf/nan during batch_number={self.batch_number}")
        print(f"Last {len(self.frames)} forward frames:")
        print(f"{'abs min':8} {'abs max':8} metadata")
        print("\n".join(self.frames))
        print("\n\n")
        self.frames = []

    def analyse_model(self):
        # extract the fully qualified module names, to be able to report at run time. e.g.:
        # encoder.block.2.layer.0.SelfAttention.o
        #
        # for shared weights only the first shared module name will be registered
        self.module_names = {m: name for name, m in self.model.named_modules()}
        # self.longest_module_name = max(len(v) for v in self.module_names.values())

    def analyse_variable(self, var, ctx):
        if torch.is_tensor(var):
            self.expand_frame(get_abs_min_max(var, ctx))
            if detect_overflow(var, ctx):
                self.detected_overflow = True
        elif var is None:
            self.expand_frame(f"{'None':>17} {ctx}")
        else:
            self.expand_frame(f"{'not a tensor':>17} {ctx}")

    def batch_start_frame(self):
        self.expand_frame(
            f"\n\n{self.prefix} *** Starting batch number={self.batch_number} ***"
        )
        self.expand_frame(f"{'abs min':8} {'abs max':8} metadata")

    def batch_end_frame(self):
        self.expand_frame(
            f"{self.prefix} *** Finished batch number={self.batch_number-1} ***\n\n"
        )

    def create_frame(self, module, input, output):
        self.expand_frame(
            f"{self.prefix} {self.module_names[module]} {module.__class__.__name__}"
        )

        # params
        for name, p in module.named_parameters(recurse=False):
            self.analyse_variable(p, name)

        # inputs
        if isinstance(input, tuple):
            for i, x in enumerate(input):
                self.analyse_variable(x, f"input[{i}]")
        else:
            self.analyse_variable(input, "input")

        # outputs
        if isinstance(output, tuple):
            for i, x in enumerate(output):
                # possibly a tuple of tuples
                if isinstance(x, tuple):
                    for j, y in enumerate(x):
                        self.analyse_variable(y, f"output[{i}][{j}]")
                else:
                    self.analyse_variable(x, f"output[{i}]")
        else:
            self.analyse_variable(output, "output")

        self.save_frame()

    def register_forward_hook(self):
        def __register(module):
            module.register_forward_hook(self.forward_hook)

        self.model.apply(__register)

    def forward_hook(self, module, input, output):
        # - input is a tuple of packed inputs (could be non-Tensors)
        # - output could be a Tensor or a tuple of Tensors and non-Tensors

        last_frame_of_batch = False

        trace_mode = True if self.batch_number in self.trace_batch_nums else False
        if trace_mode:
            self.reset_saved_frames()

        if self.total_calls == 0:
            self.batch_start_frame()
        self.total_calls += 1

        # count batch numbers - the very first forward hook of the batch will be called when the
        # batch completes - i.e. it gets called very last - we know this batch has finished
        if module == self.model:
            self.batch_number += 1
            last_frame_of_batch = True

        self.create_frame(module, input, output)

        # if last_frame_of_batch:
        #     self.batch_end_frame()

        if trace_mode:
            self.trace_frames()

        if last_frame_of_batch:
            self.batch_start_frame()

        if self.detected_overflow and not trace_mode:
            self.dump_saved_frames()

            # now we can abort, as it's pointless to continue running
            raise ValueError(
                "DebugUnderflowOverflow: inf/nan detected, aborting as there is no point running further. "
                "Please scroll up above this traceback to see the activation values prior to this event."
            )

        # abort after certain batch if requested to do so
        if (
            self.abort_after_batch_num is not None
            and self.batch_number > self.abort_after_batch_num
        ):
            raise ValueError(
                f"DebugUnderflowOverflow: aborting after {self.batch_number} batches due to"
                f" `abort_after_batch_num={self.abort_after_batch_num}` arg"
            )


def get_abs_min_max(var, ctx):
    if var.numel() == 0:
        min_max = "empty"
    else:
        abs_var = var.abs()
        min_max = f"{abs_var.min():8.2e} {abs_var.max():8.2e}"
    return f"{min_max:>17} {ctx}"


def detect_overflow(var, ctx):
    """
    Report whether the tensor contains any `nan` or `inf` entries.

    This is useful for detecting overflows/underflows and best to call right after the function that did some math that
    modified the tensor in question.

    This function contains a few other helper features that you can enable and tweak directly if you want to track
    various other things.

    Args:
        var: the tensor variable to check
        ctx: the message to print as a context

    Return:
        `True` if `inf` or `nan` was detected, `False` otherwise
    """
    detected = False
    if torch.isnan(var).any().item():
        detected = True
        print(f"{ctx} has nans")
    if torch.isinf(var).any().item():
        detected = True
        print(f"{ctx} has infs")

    # if needed to monitor large elements can enable the following
    if 0:  # and detected:
        n100 = var[torch.ge(var.abs(), 100)]
        if n100.numel() > 0:
            print(f"{ctx}:  n100={n100.numel()}")
        n1000 = var[torch.ge(var.abs(), 1000)]
        if n1000.numel() > 0:
            print(f"{ctx}: n1000={n1000.numel()}")
        n10000 = var[torch.ge(var.abs(), 10000)]
        if n10000.numel() > 0:
            print(f"{ctx}: n10000={n10000.numel()}")

    if 0:
        print(f"min={var.min():9.2e} max={var.max():9.2e}")

    if 0:
        print(
            f"min={var.min():9.2e} max={var.max():9.2e} var={var.var():9.2e} mean={var.mean():9.2e} ({ctx})"
        )

    return detected
