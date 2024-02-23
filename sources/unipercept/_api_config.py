"""
This file defines some basic API methods for working with UniPercept models, data and other submodules.
"""
from __future__ import annotations
import contextlib

import os
import re
import typing as T

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
from omegaconf import DictConfig
from PIL import Image as pil_image

from unipercept.integrations.wandb_integration import WANDB_RUN_PREFIX
from unipercept.integrations.wandb_integration import read_run as _wandb_read_run
from unipercept.log import get_logger
from unipercept.state import check_main_process
from unipercept.utils.typings import Pathable

if T.TYPE_CHECKING:
    import torch.types

    import unipercept

    from unipercept.data.ops import Op
    from unipercept.data.sets import Metadata
    from unipercept.model import InputData, ModelBase

    StateParam: T.TypeAlias = (
        str | os.PathLike | dict[str, torch.Tensor] | unipercept.engine.Engine
    )
    StateDict: T.TypeAlias = dict[str, torch.Tensor]
    ConfigParam: T.TypeAlias = str | os.PathLike | DictConfig
    ImageParam: T.TypeAlias = (
        str | os.PathLike | pil_image.Image | np.ndarray | torch.Tensor
    )

__all__ = [
    "read_config",
    "read_run",
    "load_checkpoint",
    "create_engine",
    "create_model",
    "create_model_factory",
    "create_dataset",
    "create_loaders",
    "create_inputs",
    "prepare_images",
]

_logger = get_logger(__name__)


KEY_WEIGHTS = "_WEIGHTS_"  # Key used to store the path used to initialize the config through the API


##########################
# SUPPORT FOR W&B REMOTE #
##########################


def _read_model_wandb(path: str) -> str:
    """
    Read a model from W7B. Prefix wandb-run://entity/project/name
    """
    from unipercept import file_io

    run = _wandb_read_run(path)
    import wandb
    from wandb.sdk.wandb_run import Run

    assert path.startswith(WANDB_RUN_PREFIX)

    _logger.info("Reading W&B model checkpoint from %s", path)

    run_name = path[len(WANDB_RUN_PREFIX) :]
    wandb_api = wandb.Api()
    run: Run = wandb_api.run(run_name)

    model_artifact_name = (
        f"{run.entity}/{run.project}/{run.id}-model:latest/model.safetensors"
    )

    _logger.info("Downloading model artifact %s", model_artifact_name)
    return file_io.get_local_path(f"wandb-artifact://{model_artifact_name}")


##########################
# READING CONFIGURATIONS #
##########################


def read_run(
    path: str,
) -> tuple[DictConfig, unipercept.engine.Engine, unipercept.model.ModelFactory]:
    """
    Read the run from a configuration file, directory or remote.

    Parameters
    ----------
    path
        Path to a configuration or remote location.

    Returns
    -------
    config
        The configuration object.
    engine
        The engine object.
    model_factory
        The model factory object.
    """
    from unipercept import file_io

    if file_io.isdir(path):
        path = file_io.join(path, "config.yaml")
        if not file_io.isfile(path):
            raise FileNotFoundError(f"Could not find configuration file at {path}")
    config = read_config(path)
    engine = create_engine(config)
    model = create_model_factory(config)

    return config, engine, model


def read_config(config: ConfigParam) -> DictConfig:
    """
    Load a configuration file into a DictConfig object. If a path is passed, the configuration file will be loaded
    from disk. Otherwise, the object will be returned as-is, if it is already a DictConfig.

    Parameters
    ----------
    config
        Path to the configuration file (.py or .yaml file) or a DictConfig object.

    Returns
    -------
    config
        A DictConfig object.
    """
    from unipercept.engine._engine import _sort_children_by_suffix
    from unipercept import file_io
    from unipercept.config import load_config, load_config_remote

    if isinstance(config, str):
        try:
            cfg_obj = load_config_remote(config)
            cfg_obj[KEY_WEIGHTS] = config
            return cfg_obj
        except FileNotFoundError:
            pass
    if isinstance(config, Pathable):
        _logger.info("Reading configuration from path %s", config)

        config_path = file_io.Path(config).resolve().expanduser()
        if not config_path.is_file():
            msg = f"Could not find configuration file at {config_path}"
            raise FileNotFoundError(msg)
        if config_path.suffix not in (".py", ".yaml"):
            msg = f"Configuration file must be a .py or .yaml file, got {config_path}"
            raise ValueError(msg)
        obj = load_config(str(config_path))
        if not isinstance(obj, DictConfig):
            msg = f"Expected a DictConfig, got {obj}"
            raise TypeError(msg)
        
        if config_path.suffix == ".yaml":
            # Check if the config has a latest checkpoint
            models_path = config_path.parent / "outputs" / "checkpoints"
            if models_path.is_dir():
                step_dirs = list(_sort_children_by_suffix(models_path))
                if len(step_dirs) > 0:
                    latest_step = file_io.Path(step_dirs[-1])
                    latest_weights = latest_step / "model.safetensors"
                    if latest_weights.is_file():
                        obj[KEY_WEIGHTS] = latest_weights.as_posix()
        return obj
    elif isinstance(config, DictConfig):
        return config
    else:
        msg = f"Expected a configuration file path or a DictConfig, got {config}"
        raise TypeError(msg)


#######################
# WORKING WITH MODELS #
#######################


def load_checkpoint(state: StateParam, target: nn.Module) -> None:
    """
    Load a checkpoint into a model from a file or a dictionary. The model is modified in-place.

    Parameters
    ----------
    state
        Path to the checkpoint file (.pth/.safetensors file) or a dictionary containing the model state.
    target
        The model to load the state into.
    """
    from unipercept import file_io
    from unipercept.engine import Engine

    # Check remote
    if isinstance(state, str) and state.startswith(WANDB_RUN_PREFIX):
        state = _read_model_wandb(state)

    # Check argument type
    if isinstance(state, str) or isinstance(state, os.PathLike):
        _logger.info("Loading checkpoint from path %s", state)
        # State was passed as a file path
        state_path = file_io.Path(state)
        if state_path.is_dir():
            state_path = state_path / "model.safetensors"
        if not state_path.is_file():
            raise FileNotFoundError(f"Could not find checkpoint file at {state_path}")

        match state_path.suffix:
            case ".pth":
                # Load PyTorch pickled checkpoint
                state_dict = torch.load(state_path, map_location="cpu")
                if isinstance(state_dict, nn.Module):
                    # If the checkpoint is a nn.Module, extract its state_dict
                    state_dict = state_dict.state_dict()
                elif not isinstance(state_dict, dict):
                    pass  # OK
                else:
                    raise TypeError(
                        f"Expected a state_dict or a nn.Module, got {type(state_dict)}"
                    )
                target.load_state_dict(state_dict, strict=True)
            case ".safetensors":
                # Load SafeTensors checkpoint
                import safetensors.torch as st

                st.load_model(target, state_path, strict=True)
            case _:
                raise ValueError(
                    f"Checkpoint file must be a .pth or .safetensors file, got {state_path}"
                )
    elif isinstance(state, Engine):
        _logger.info("Loading checkpoint from engine")
        # State was passed as a Engine object
        state.recover(model=target)
    elif isinstance(state, T.Mapping):
        _logger.info("Loading state from weights mapping")
        target.load_state_dict(state, strict=True)
    else:
        raise TypeError(f"Expected a checkpoint file path or a dictionary, got {state}")


####################
# CREATION METHODS #
####################


def create_engine(config: ConfigParam) -> unipercept.engine.Engine:
    """
    Create a engine from a configuration file. The engine will be initialized with the default parameters, and
    the configuration file will be used to override them.

    Parameters
    ----------
    config
        Path to the configuration file (.py or .yaml file).

    Returns
    -------
    engine
        A engine instance.
    """
    from .config import instantiate
    from unipercept.engine import Engine
    from unipercept.state import barrier

    config = read_config(config)
    engine = T.cast(Engine, instantiate(config.ENGINE))

    with contextlib.suppress(FileExistsError):
        engine.config = config
    barrier()

    return engine


def create_model_factory(
    config: ConfigParam, *, weights: str | None = None
) -> unipercept.model.ModelFactory:
    """
    Create a factory for models from a configuration file. The factory will be initialized with the default parameters,
    and the configuration file will be used to override them.
    """
    from unipercept.model import ModelFactory

    config = read_config(config)
    return ModelFactory(config.MODEL, weights=weights or config.get(KEY_WEIGHTS))


def create_model(
    config: ConfigParam,
    *,
    state: StateParam | None = None,
    device: str | torch.types.Device = "cpu",
) -> ModelBase:
    """
    Load a model from a configuration file. If the configuration file is part of a traning session, the latest
    checkpoint will be loaded. Otherwise, the model will be returned as-is, with its parameters default-initialized.

    Parameters
    ----------
    config
        Path to the configuration file (.py or .yaml file).
    state
        Path to the checkpoint file (.pth/.safetensors file) or a dictionary containing the model state.
        Optional, defaults to loading the latest checkpoint from the training session, if the configuration references
        one.

    Returns
    -------
    model
        A model instance.
    """
    from unipercept import file_io
    from unipercept.config import instantiate

    # Check remote
    if isinstance(config, str) and config.startswith(WANDB_RUN_PREFIX):
        if state is None:
            state = _read_model_wandb(config)
        config = read_config(config)

    # Handle binary PyTorch model
    if (
        isinstance(config, (str, os.PathLike))
        and (pickle_path := file_io.Path(config)).is_file()
        and pickle_path.suffix == ".bin"
    ):
        _logger.info("Loading binary PyTorch model from %s", pickle_path)

        if state is not None:
            raise ValueError(
                "Cannot specify both a binary PyTorch model (`.bin` suffix) and a state"
            )
        with open(pickle_path, "rb") as f:
            model = torch.load(f)
        if not isinstance(model, nn.Module):
            raise TypeError(
                f"Expected binary file to load an `nn.Module` class, got {type(model)}"
            )
        return model

    # Default handling
    config = read_config(config)
    model: ModelBase = instantiate(config.MODEL)

    if state is not None:
        load_checkpoint(state, model)
    elif KEY_WEIGHTS in config:
        _logger.info(
            "Loading remote checkpoint matching configuration read path",
        )
        load_checkpoint(config[KEY_WEIGHTS], model)

    return model.eval().to(device)


@T.overload
def create_dataset(
    config: ConfigParam,
    variant: T.Optional[str | re.Pattern] = None,
    batch_size: int = 1,
    return_loader: bool = True,
) -> tuple[torch.utils.data.DataLoader[InputData], Metadata]:
    ...


@T.overload
def create_dataset(
    config: ConfigParam,
    variant: T.Optional[str | re.Pattern] = None,
    batch_size: int = 1,
    return_loader: bool = True,
) -> tuple[T.Iterator[InputData], Metadata]:
    ...


def create_dataset(
    config: ConfigParam,
    variant: T.Optional[str | re.Pattern] = None,
    batch_size: int = 1,
    return_loader: bool = True,
) -> tuple[T.Iterator[InputData] | torch.utils.data.DataLoader[InputData], Metadata]:
    """
    Create an iterator of a dataloader as specified in a configuration file.

    Parameters
    ----------
    config
        The configuration file to use.
    variant
        The variant of the dataset to use. Will be compiled to a regular expression.
        If ``None``, then the default test dataset will be used, which is found via
        the pattern ``.+/val$``. Defaults to ``None``.
    batch_size
        The batch size to use. Defaults to 1.
    return_loader
        Whether to return the dataloader object instead of an iterator. Defaults to `False`.

    Returns
    -------
    iterator
        An iterator of `InputData` objects.
    info
        The dataset metadata.
    """

    from .config import instantiate

    config = read_config(config)

    loaders = dict(config.ENGINE.loaders)

    if isinstance(variant, str) and variant in loaders:
        # Lookup by direct key
        _logger.info("Found dataset loader %s via key", variant)
        key = variant
    else:
        # Search in keys
        if variant is None:
            variant = re.compile(r".+/val$")
        elif isinstance(variant, str):
            variant = re.compile(variant)
        elif not isinstance(variant, re.Pattern):
            raise TypeError(
                f"Expected a string or a regular expression, got {type(variant)}"
            )

        # Find the first key that matches the pattern
        key_list = list(loaders.keys())
        key = next((k for k in key_list if variant.match(k)), None)
        if key is None:
            raise ValueError(
                f"Could not find a dataset matching {variant.pattern!r} in {key_list}"
            )
        else:
            _logger.info(
                "Found dataset loader %s matching %s, available are: %s",
                key,
                variant.pattern,
                key_list,
            )

    datafactory = instantiate(loaders[key])
    dataloader = datafactory(batch_size)
    if return_loader:
        return dataloader, datafactory.dataset.info
    else:
        return iter(datafactory(batch_size)), datafactory.dataset.info


def create_loaders(config: ConfigParam, *, keys: T.Optional[T.Collection[str]] = None):
    """
    Create a dictionary of dataloaders as specified in a configuration file.

    Parameters
    ----------
    config
        The configuration file to use.
    keys
        The keys of the loaders to create. If `None`, then all loaders will be created. Defaults to `None`.

    Returns
    -------
    loaders
        A dictionary of dataloaders.
    """
    from .config import instantiate

    config = read_config(config)

    loaders = {}
    for key, loader in config.ENGINE.loaders.items():
        if keys is not None and key not in keys:
            continue
        loaders[key] = instantiate(loader)

    return loaders


####################################################
# FEEDING IMAGES TO MODELS WITHOUT USING A DATASET #
####################################################


def prepare_images(
    images_dir: str | os.PathLike,
    *,
    batch_size: int = 1,
    suffix: T.Collection[str] = (".jpg", ".png"),
    separator: str | None = os.sep,
    return_loader: bool = True,
    ops: T.Sequence[Op] | None = None,
):
    """
    Create an interator of a dataloader the mocks a dataset from a directory of images.
    All images are converted to `InputData` objects, with a single capture per image.

    Parameters
    ----------
    images_dir
        The directory containing the images. By default, if the directory contains subdirectories of images, then these
        will be treated as different sequences, and frames are chronologically ordered by alphabetically sorting their
        filename. This can be modified using the `separator` and `sort` parameters.
    batch_size
        The batch size to use. Defaults to 1.
    suffix
        The suffix of the image files to look for. Defaults to `.png`.
    separator
        The separator to use to split the filename into sequence and frame ID. Defaults to the default separator as
        dictated by the current OS (i.e. '/' or '\\'). If set to `None`, then the filename is used as the frame number
        and all images are treated as part of the same sequence.
    return_loader
        Whether to return the dataloader object instead of an iterator. Defaults to `False`.
    ops
        A sequence of operations to apply to the images before they are returned. Defaults to no operations.

    Returns
    -------
    iterator
        An iterator of `InputData` objects.
    info
        The dataset metadata.
    """
    from torch.utils.data import DataLoader
    from unipercept import file_io
    from unipercept.model import InputData

    if ops is None:
        ops = []

    # List the images using a Glob pattern, such that we can determine whether we are dealing with a flat directory of
    # images or a directory of subdirectories of images.
    images_root = file_io.Path(images_dir).resolve().expanduser()

    image_paths = []
    for s in suffix:
        image_paths += list(images_root.glob(f"**/*{s}"))
    image_paths = filter(lambda p: p.is_file(), image_paths)
    image_paths = map(lambda p: p.relative_to(images_root), image_paths)
    image_paths = map(str, image_paths)
    image_paths = list(image_paths)
    image_paths = sorted(image_paths)

    # If the separator is None, then we treat all images as part of the same sequence.
    image_ordered: list[tuple[str, tuple[str | int, str | int]]]
    if separator is None:
        image_ordered = [(p, ("all", p)) for p in image_paths]
    else:
        # Otherwise, we split the filename into sequence and frame ID.
        image_ordered = []
        for path in image_paths:
            split = str(reversed(path)).split(separator, maxsplit=1)
            if len(split) != 2:
                seq = "all"
                frame = split[0]
            else:
                frame, seq = split
            image_ordered.append((path, (seq, frame)))

    # # Sort by sequence ID, then by frame number.
    # image_ordered.sort(key=lambda p: p[1])

    seq_num = 0
    sequence_frame_indices = {}
    image_intseq = []
    for path, (seq, _) in image_ordered:
        if seq not in sequence_frame_indices:
            seq_num += 1
        frame_num = sequence_frame_indices.setdefault(seq, -1) + 1
        sequence_frame_indices[seq] = frame_num

        path_full = images_root / path

        assert path_full.is_file(), f"Could not find image file at {path_full}"

        image_intseq.append((path_full, (seq_num, frame_num)))

    # Make the dataset object
    dataset = _ImagePathsDataset(image_intseq, ops)

    # Create the dataloader
    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        collate_fn=InputData.collate,
        shuffle=False,
        num_workers=0,
    )

    return loader if return_loader else iter(loader)


class _ImagePathsDataset(torch.utils.data.Dataset):
    def __init__(
        self, paths: T.Sequence[tuple[str, tuple[int, int]]], ops: T.Sequence[Op]
    ):
        self.paths = paths
        self.ops = ops

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path, (sequence_id, frame_id) = self.paths[index]

        input = create_inputs(path, sequence_offset=sequence_id, frame_offset=frame_id)[
            0
        ]
        for op in self.ops:
            input = op(input)
        return input


def create_inputs(
    images: ImageParam | T.Sequence[ImageParam],
    *,
    frame_offset: int = 0,
    sequence_offset: int = 0,
) -> InputData:
    """
    Creates ``InputData`` from a single image, useful for simple inference use-cases. Also accepts a sequence if a
    batch is to be created. The returned ``InputData`` will have a single capture with a single frame.

    Parameters
    ----------
    images
        The image(s) to create the ``InputData`` from. If a tensor is passed, then it will be moved to the CPU.

    Returns
    -------
    input
        An ``InputData`` object.
    """
    from torchvision.io import read_image
    from torchvision.transforms.v2.functional import pil_to_tensor

    from unipercept import file_io
    from unipercept.data.tensors import Image
    from unipercept.model import CameraModel, CaptureData, InputData

    batch: list[torch.Tensor] = []

    if not isinstance(images, T.Sequence):
        images = [images]

    for image_spec in images:
        if isinstance(image_spec, str) or isinstance(image_spec, os.PathLike):
            image_path = file_io.Path(image_spec)
            if not image_path.is_file():
                raise FileNotFoundError(f"Could not find image file at {image_path}")
            image = read_image(str(image_path))
        elif isinstance(image_spec, pil_image.Image):
            image = pil_to_tensor(image_spec)
        else:
            image = torch.as_tensor(image_spec).cpu()
        assert (
            image.ndim == 3 and image.shape[0] == 3
        ), f"Expected am RGB image, got {image.shape}"
        batch.append(image)

    ids = torch.stack(
        [
            torch.ones(len(batch), dtype=torch.int64) * sequence_offset,  # Sequence ID
            torch.arange(len(batch), dtype=torch.int64) * frame_offset,  # Frame ID
        ],
        dim=1,
    )

    inputs = InputData(
        ids=ids,
        captures=CaptureData(
            times=torch.arange(len(batch), dtype=torch.float32).unsqueeze_(1),
            images=(torch.stack(batch)).as_subclass(Image).unsqueeze_(1),
            segmentations=None,
            depths=None,
            batch_size=[len(batch), 1],
        ),
        motions=None,
        cameras=CameraModel(
            image_size=torch.tensor(
                [img.shape[-2:] for img in batch], dtype=torch.float32
            ),
            matrix=torch.eye(4, dtype=torch.float32).repeat(len(batch), 1, 1),
            pose=torch.eye(4, dtype=torch.float32).repeat(len(batch), 1, 1),
            batch_size=[len(batch)],
        ),
        content_boxes=torch.stack(
            [
                torch.tensor([0, 0, img.shape[-2], img.shape[-1]], dtype=torch.float32)
                for img in batch
            ]
        ),
        batch_size=[len(batch)],
    )
    return inputs
