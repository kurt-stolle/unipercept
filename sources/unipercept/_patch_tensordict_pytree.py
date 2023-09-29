"""
This patch is required to make `tensordict.TensorDict` compatible with `torch.utils._pytree`.

Update
------
As of 2023-08-01, this patch is no longer required, as `tensordict.TensorDict` now supports this. 
We will leave it in the `unipercept` package until the next release of `tensordict` that includes this fix.

See: https://github.com/pytorch-labs/tensordict/pull/501
"""
from __future__ import annotations

import collections
import typing as T
import warnings

import tensordict
import torch.utils._pytree as pytree

__all__ = []


def _flatten(d: tensordict.TensorDict) -> T.Tuple[T.List[T.Any], pytree.Context]:
    return list(d.values()), {
        "keys": list(d.keys()),  # type: ignore
        "batch_size": d.batch_size,
        "names": d.names,
    }


def _unflatten(values: T.List[T.Any], context: pytree.Context) -> T.Dict[T.Any, T.Any]:
    return tensordict.TensorDict(
        dict(zip(context["keys"], values)),
        context["batch_size"],
        names=context["names"],
    )  # type: ignore


def _register(t: tensordict.TensorDictBase) -> None:
    if t not in pytree.SUPPORTED_NODES:
        pytree._register_pytree_node(t, _flatten, _unflatten)
        print(f"PATCH: Registered pytree node {t}")
    else:
        # warnings.warn(f"Pytree node {t} is already registered!", RuntimeWarning)
        pass


_register(tensordict.TensorDict)
_register(tensordict.SubTensorDict)
_register(tensordict.LazyStackedTensorDict)
_register(tensordict.PersistentTensorDict)
