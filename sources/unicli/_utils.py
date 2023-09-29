"""
Utilities for the CLI.
"""
from typing import Any, NoReturn

import omegaconf
import torch
import torch.nn as nn
import torch.utils.data
from detectron2.config import instantiate


def setup_config(cfg: Any) -> omegaconf.DictConfig:
    if not isinstance(cfg, omegaconf.DictConfig):
        raise ValueError("Configuration must be an OmegaConf DictConfig!")
    return cfg


def load_dataset(cfg: omegaconf.DictConfig, *args, **kwargs) -> torch.utils.data.DataLoader:
    """
    Instantiate a dataset from configuration.
    Args are passed to the dataset constructor.
    """
    loader = instantiate(cfg)
    assert isinstance(loader, torch.utils.data.DataLoader)
    return loader


def load_model(cfg: omegaconf.DictConfig, device: torch.device | str) -> nn.Module:
    """
    Instantiate a model from configuration.
    """
    model = instantiate(cfg)
    assert isinstance(model, nn.Module)
    model.to(device)
    return model


def prompt_confirm(message: str, condition: bool, default: bool = False) -> None | NoReturn:
    """
    Prompt the user for confirmation when a potentially destructive action is
    about to be performed.

    Parameters
    ----------
    message
        The message to display to the user, concatenated with "Are you sure? [y/N]".
    condition
        The condition that triggers the prompt (e.g. `True` if the action is about to be performed).
    default
        The default choice if the user does not provide any input. Defaults to `False`, i.e. "No".

    Returns
    -------
    Exits if the user aborts the action, otherwise None.
    """
    if not condition:
        return

    if default:
        message = f"{message} [Y/n] "
    else:
        message = f"{message} [y/N] "

    def input_choice() -> bool:
        while True:
            choice = input(message).lower()

            if choice in {"y", "yes"}:
                return True
            elif choice in {"n", "no"}:
                return False
            elif choice == "":
                return default

    choice = input_choice()
    if not choice:
        print("Aborting.")
        exit(0)
