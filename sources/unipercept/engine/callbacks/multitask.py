import typing as T

import regex as re
import torch
from torch import nn

from unipercept.types import Tensor

from .._params import EngineParams
from ._base import Signal, State, StatefulCallbackDispatcher


class _GroupedLossWeightingCallback(StatefulCallbackDispatcher):
    __slots__ = ("tasks", "groups", "missing_ok")

    def __init__(
        self,
        *,
        tasks: (
            T.Iterable[str | re.Pattern | T.Iterable[str | re.Pattern]]
            | T.Mapping[str, T.Iterable[str | re.Pattern] | str | re.Pattern]
        ),
        missing_ok: bool = False,
        **kwargs,
    ):
        if isinstance(tasks, T.Mapping):
            task_names = T.cast(list[str], list(tasks.keys()))
            tasks = tasks.values()
        else:
            task_names = None

        self.groups: list[list[re.Pattern]] = [
            list(
                map(
                    lambda n: (re.compile(n) if isinstance(n, str) else n),
                    [task] if isinstance(task, (str, re.Pattern)) else task,
                )
            )
            for task in tasks
        ]
        self.tasks: list[str] = (
            [str(t[0].pattern) for t in self.groups]
            if task_names is None
            else task_names
        )
        self.missing_ok = missing_ok


class UncertaintyLossWeightingCallback(_GroupedLossWeightingCallback):
    """
    Implements the Uncertrainty loss weighting algorithm from [1]

    References
    ----------
    [1] Kendall et al., "Multi-Task Learning Using Uncertrainty to Weigh Losses for Scene Geometry and Semantics". CVPR 2018. https://arxiv.org/pdf/1705.07115
    """

    __slots__ = ("gamma", "hook_handle", "weights")

    def __init__(
        self,
        gamma: float = -0.7,
        lr: float = 1e-2,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.gamma = gamma
        self.lr = lr
        self.weights: Tensor | None = None
        self.hook_handle: torch.utils.hooks.RemovableHandle | None = None

    @property
    def task_weights(self) -> dict[str, float]:
        assert self.weights is not None
        return dict(zip(self.tasks, self.weights.tolist(), strict=False))

    @T.override
    def on_optimizer_setup(
        self,
        params: EngineParams,
        state: State,
        control: Signal,
        *,
        stage: EngineStage,
        extra_params: list[ParameterDefinition],
        **kwargs,
    ):
        assert self.weights is not None

        extra_params.append(
            {
                "params": self.weights,
                "lr": self.lr,
                "weight_decay": 0.0,
            }
        )

    @T.override
    def on_accelerator_setup(self, *args, accelerator: Accelerator, **kwargs):
        self.weights = torch.full(
            (len(self.groups),),
            self.gamma,
            device=accelerator.device,
            dtype=torch.float32,
            requires_grad=True,
        )

    @T.override
    def on_model_setup(self, *args, model: nn.Module, training: bool, **kwargs):
        if not training:
            return
        if self.hook_handle is not None:
            self.hook_handle.remove()
            self.hook_handle = None

        def apply_weights_hook(
            module: nn.Module,
            inputs: ModelInput,
            outputs: ModelOutput | dict[str, Tensor],
        ) -> ModelOutput | None | dict[str, Tensor]:
            r"""
            Hook that applies the loss weights in the forward of the model.
            """
            if not module.training:
                return None
            assert self.weights is not None

            if isinstance(outputs, ModelOutput):
                loss_dict = outputs.losses
            elif isinstance(outputs, dict):
                loss_dict = outputs
            else:
                msg = f"Outputs must be a ModelOutput or a dict, got {type(outputs)}"
                raise ValueError(msg)
            assert loss_dict is not None
            for loss_key in list(loss_dict.keys()):
                for i, group in enumerate(self.groups):
                    if any(pattern.search(loss_key) for pattern in group):
                        w = self.weights[i]
                        loss_dict[loss_key] = (
                            loss_dict[loss_key] * 1 / (2 * w.exp()) + w / 2
                        )
                        break
                else:
                    if self.missing_ok:
                        pass
                    else:
                        msg = f"Loss {loss_key} not found in groups: {self.groups}"
                        raise ValueError(msg)
            return outputs

        self.hook_handle = model.register_forward_hook(apply_weights_hook)

    @T.override
    def on_train_begin(self, params, state, control, *, model: nn.Module, **kwargs):
        if self.weights is None:
            msg = f"{self.__class__.__name__} requires the accelerator to be set up before training."
            raise RuntimeError(msg)

        nn.init.constant_(self.weights, self.gamma)

    @T.override
    def on_train_step_end(
        self, params, state, control, *, losses: dict[str, Tensor], **kwargs
    ):
        if self.weights is None:
            msg = f"{self.__class__.__name__} requires the accelerator to be set up before training."
            raise RuntimeError(msg)

        reduce(self.weights, op="mean", inplace=True).wait()


class TaskRebalanceCallback(_GroupedLossWeightingCallback):
    """
    Implements a task rebalancing callback without optimization.

    If window is 1, then this is equivalent to Dynamic Weight Averaging (DWA) [1].

    References
    ----------
    [1] Liu et al., End-to-End Multi-Task Learning with Attention. CVPR 2019.
    """

    __slots__ = ("gamma", "window", "hook_handle", "weights", "losses")

    def __init__(
        self,
        gamma: float = 0.5,
        window: int = 2,
        **kwargs,
    ):
        super().__init__(**kwargs)

        assert window >= 2, window

        self.window = window
        self.gamma = gamma
        self.weights: Tensor | None = None
        self.losses: Tensor | None = None
        self.hook_handle: torch.utils.hooks.RemovableHandle | None = None

    @property
    def task_weights(self) -> dict[str, float]:
        assert self.weights is not None
        return dict(zip(self.tasks, self.weights.tolist(), strict=False))

    @T.override
    def on_accelerator_setup(self, *args, accelerator: Accelerator, **kwargs):
        self.weights = torch.full(
            (len(self.groups),),
            torch.nan,
            device=accelerator.device,
            dtype=torch.float32,
            requires_grad=False,
        )
        self.losses = torch.full(
            (len(self.groups), self.window),
            torch.nan,
            device=accelerator.device,
            dtype=torch.float32,
            requires_grad=False,
        )

    @T.override
    def on_model_setup(self, *args, model: nn.Module, training: bool, **kwargs):
        if not training:
            return
        if self.hook_handle is not None:
            self.hook_handle.remove()
            self.hook_handle = None

        def apply_weights_hook(
            module: nn.Module,
            inputs: ModelInput,
            outputs: ModelOutput | dict[str, Tensor],
        ) -> ModelOutput | None | dict[str, Tensor]:
            r"""
            Hook that applies the loss weights in the forward of the model.
            """
            if not module.training:
                return None
            assert self.weights is not None

            if isinstance(outputs, ModelOutput):
                loss_dict = outputs.losses
            elif isinstance(outputs, dict):
                loss_dict = outputs
            else:
                msg = f"Outputs must be a ModelOutput or a dict, got {type(outputs)}"
                raise ValueError(msg)
            assert loss_dict is not None
            for loss_key in list(loss_dict.keys()):
                for i, group in enumerate(self.groups):
                    if any(pattern.search(loss_key) for pattern in group):
                        w = self.weights[i].detach()
                        loss_dict[loss_key] = loss_dict[loss_key] * w
                        break
                else:
                    if self.missing_ok:
                        pass
                    else:
                        msg = f"Loss {loss_key} not found in groups: {self.groups}"
                        raise ValueError(msg)
            return outputs

        self.hook_handle = model.register_forward_hook(apply_weights_hook)

    @T.override
    def on_train_begin(self, params, state, control, *, model: nn.Module, **kwargs):
        if self.weights is None or self.losses is None:
            msg = f"{self.__class__.__name__} requires the accelerator to be set up before training."
            raise RuntimeError(msg)

        self.weights.fill_(1.0)
        self.losses.fill_(torch.nan)

    @T.override
    @torch.no_grad()
    def on_train_step_begin(self, *args, **kwargs):
        r"""
        Compute the weights given the current losses.
        """
        assert self.losses is not None
        if torch.isnan(self.losses).any():
            return
        loss_1, loss_2 = self.losses.chunk(2, dim=-1)
        loss_1 = loss_1.mean(dim=-1)
        loss_2 = loss_2.mean(dim=-1)
        w = (loss_1 / loss_2) * self.gamma

        self.weights = w.softmax(dim=0) * len(self.groups)

    @T.override
    @torch.no_grad()
    def on_train_step_end(
        self, params, state, control, *, losses: dict[str, Tensor], **kwargs
    ):
        assert self.losses is not None
        self.losses = self.losses.roll(1, dims=-1)

        losses_keys = set(losses.keys())
        losses_keys.remove("total")
        for i, patterns in enumerate(self.groups):
            group_losses = []

            for loss_pattern in patterns:
                matches = set()
                for loss_key in losses_keys:
                    if not loss_pattern.search(loss_key):
                        continue
                    group_losses.append(reduce(losses[loss_key].detach()))
                    matches.add(loss_key)
                losses_keys -= matches
            if len(group_losses) == 0:
                msg = f"No losses found for group: {patterns}"
                raise RuntimeError(msg)
            self.losses[i, 0] = torch.stack([g.wait() for g in group_losses]).sum()

        if len(losses_keys) > 0 and not self.missing_ok:
            msg = f"Unmatched losses: {losses_keys}. Groups: {self.groups}"
            raise RuntimeError(msg)


class TaskParameterRebalanceCallback(StatefulCallbackDispatcher):
    """
    Implements a task parameter rebalancing callback.
    """

    def __init__(self, optimizer: OptimizerFactory):
        self.optimizer_factory = optimizer
        self.optimizer: Optimizer | None = None
        self.model: ModelBase | None = None

    def virtual_step(self, train_x, train_y, alpha, model_optim):
        """
        Compute unrolled network theta' (virtual step)
        """

        # forward & compute loss
        if type(train_x) == list:  # multi-domain setting [many-to-many]
            train_pred = [self.model(x, t) for t, x in enumerate(train_x)]
        else:  # single-domain setting [one-to-many]
            train_pred = self.model(train_x)

        train_loss = self.model_fit(train_pred, train_y)

        loss = sum([w * train_loss[i] for i, w in enumerate(self.meta_weights)])

        # compute gradient
        gradients = torch.autograd.grad(loss, self.model.parameters())

        # do virtual step (update gradient): theta' = theta - alpha * sum_i lambda_i * L_i(f_theta(x_i), y_i)
        with torch.no_grad():
            for weight, weight_, grad in zip(
                self.model.parameters(),
                self.model_.parameters(),
                gradients,
                strict=False,
            ):
                if (
                    "momentum" in model_optim.param_groups[0].keys()
                ):  # used in SGD with momentum
                    m = (
                        model_optim.state[weight].get("momentum_buffer", 0.0)
                        * model_optim.param_groups[0]["momentum"]
                    )
                else:
                    m = 0
                weight_.copy_(
                    weight
                    - alpha
                    * (m + grad + model_optim.param_groups[0]["weight_decay"] * weight)
                )

    def unrolled_backward(self, train_x, train_y, val_x, val_y, alpha, model_optim):
        """
        Compute un-rolled loss and backward its gradients
        """

        # do virtual step (calc theta`)
        self.virtual_step(train_x, train_y, alpha, model_optim)

        # define weighting for primary tasks (with binary weights)
        pri_weights = []
        for t in self.train_tasks:
            if t in self.pri_tasks:
                pri_weights += [1.0]
            else:
                pri_weights += [0.0]

        # compute validation data loss on primary tasks
        if type(val_x) == list:
            val_pred = [self.model_(x, t) for t, x in enumerate(val_x)]
        else:
            val_pred = self.model_(val_x)
        val_loss = self.model_fit(val_pred, val_y)
        loss = sum([w * val_loss[i] for i, w in enumerate(pri_weights)])

        # compute hessian via finite difference approximation
        model_weights_ = tuple(self.model_.parameters())
        d_model = torch.autograd.grad(loss, model_weights_, allow_unused=True)
        hessian = self.compute_hessian(d_model, train_x, train_y)

        # update final gradient = - alpha * hessian
        with torch.no_grad():
            for mw, h in zip([self.meta_weights], hessian, strict=False):
                mw.grad = -alpha * h

    def compute_hessian(self, d_model, train_x, train_y):
        norm = torch.cat([w.view(-1) for w in d_model]).norm()
        eps = 0.01 / norm

        # \theta+ = \theta + eps * d_model
        with torch.no_grad():
            for p, d in zip(self.model.parameters(), d_model, strict=False):
                p += eps * d

        if type(train_x) == list:
            train_pred = [self.model(x, t) for t, x in enumerate(train_x)]
        else:
            train_pred = self.model(train_x)
        train_loss = self.model_fit(train_pred, train_y)
        loss = sum([w * train_loss[i] for i, w in enumerate(self.meta_weights)])
        d_weight_p = torch.autograd.grad(loss, self.meta_weights)

        # \theta- = \theta - eps * d_model
        with torch.no_grad():
            for p, d in zip(self.model.parameters(), d_model, strict=False):
                p -= 2 * eps * d

        if type(train_x) == list:
            train_pred = [self.model(x, t) for t, x in enumerate(train_x)]
        else:
            train_pred = self.model(train_x)
        train_loss = self.model_fit(train_pred, train_y)
        loss = sum([w * train_loss[i] for i, w in enumerate(self.meta_weights)])
        d_weight_n = torch.autograd.grad(loss, self.meta_weights)

        # recover theta
        with torch.no_grad():
            for p, d in zip(self.model.parameters(), d_model, strict=False):
                p += eps * d

        hessian = [
            (p - n) / (2.0 * eps) for p, n in zip(d_weight_p, d_weight_n, strict=False)
        ]
        return hessian
