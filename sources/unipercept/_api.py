"""
This file defines some basic API methods for working with UniPercept models, data and other submodules.
"""
from __future__ import annotations

import os
import typing as T



if T.TYPE_CHECKING:
    import numpy as np
    import torch
    import torch.nn as nn
    from omegaconf import DictConfig
    from unicore import file_io
    import torch.types
    from PIL import Image as pil_image
    from .data.sets import Metadata
    from .model import CameraModel, CaptureData, InputData, ModelBase
    from .utils.config.templates import LazyConfigFile
    from .data.ops import Op
    from .trainer import Trainer

    StateParam: T.TypeAlias = str | os.PathLike | dict[str, torch.Tensor] | Trainer
    StateDict: T.TypeAlias = dict[str, torch.Tensor]
    ConfigParam: T.TypeAlias = str | os.PathLike | DictConfig | LazyConfigFile
    ImageParam: T.TypeAlias = str | os.PathLike | pil_image.Image | np.ndarray | torch.Tensor

__all__ = [
    "read_config",
    "load_checkpoint",
    "create_trainer",
    "create_model",
    "create_dataset",
    "create_inputs",
    "prepare_images",
    "get_dataset",
    "get_info",
]


###########################
# ALIASES FROM SUBMODULES #
###########################


def get_dataset(name: str):
    """
    Alias for `unipercept.data.sets.get_dataset`.
    """
    from .data.sets import get_dataset

    return get_dataset(name)


def get_info():
    """
    Alias for `unipercept.data.sets.get_info`.
    """
    from .data.sets import get_info

    return get_info(name)

##########################
# READING CONFIGURATIONS #
##########################


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
    from .utils.config import LazyConfig

    if isinstance(config, str) or isinstance(config, os.PathLike):
        config_path = file_io.Path(config).resolve().expanduser()
        if not config_path.is_file():
            raise FileNotFoundError(f"Could not find configuration file at {config_path}")
        if not config_path.suffix in (".py", ".yaml"):
            raise ValueError(f"Configuration file must be a .py or .yaml file, got {config_path}")
        obj = LazyConfig.load(str(config_path))
        if not isinstance(obj, DictConfig):
            raise TypeError(f"Expected a DictConfig, got {obj}")
        return obj
    elif isinstance(config, DictConfig):
        return config
    else:
        raise TypeError(f"Expected a configuration file path or a DictConfig, got {config}")

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
    if isinstance(state, str) or isinstance(state, os.PathLike):
        # State was passed as a file path
        state_path = file_io.Path(state)
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
                    raise TypeError(f"Expected a state_dict or a nn.Module, got {type(state_dict)}")
                target.load_state_dict(state_dict, strict=True)
            case ".safetensors":
                # Load SafeTensors checkpoint
                import safetensors.torch as st

                st.load_model(target, state_path, strict=True)
            case _:
                raise ValueError(f"Checkpoint file must be a .pth or .safetensors file, got {state_path}")
    elif isinstance(state, Trainer):
        # State was passed as a Trainer object
        state.recover(model=target)
    elif isinstance(state, T.Mapping):
        target.load_state_dict(state, strict=True)
    else:
        raise TypeError(f"Expected a checkpoint file path or a dictionary, got {state}")


####################
# CREATION METHODS #
####################

def create_trainer(config: ConfigParam, *, model: nn.Module | None = None) -> Trainer:
    """
    Create a trainer from a configuration file. The trainer will be initialized with the default parameters, and
    the configuration file will be used to override them.

    Parameters
    ----------
    config
        Path to the configuration file (.py or .yaml file).

    Returns
    -------
    trainer
        A trainer instance.
    """
    from .utils.config import instantiate

    config = read_config(config)
    trainer: Trainer = instantiate(config.trainer)
    trainer.recover(model=model)

    return trainer


def create_model(
    config: ConfigParam, *, state: StateParam | None = None, device: str | torch.types.Device = "cpu"
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

    from .trainer import Trainer
    from .utils.config import instantiate

    if (
        isinstance(config, (str, os.PathLike))
        and (pickle_path := file_io.Path(config)).is_file()
        and pickle_path.suffix == ".bin"
    ):
        if state is not None:
            raise ValueError("Cannot specify both a binary PyTorch model (`.bin` suffix) and a state")
        with open(pickle_path, "rb") as f:
            model = torch.load(f)
        if not isinstance(model, nn.Module):
            raise TypeError(f"Expected binary file to load an `nn.Module` class, got {type(model)}")
        return model

    config = read_config(config)
    model: ModelBase = instantiate(config.model)

    if state is not None:
        load_checkpoint(state, model)

    return model.eval().to(device)


def create_dataset(
    config: ConfigParam, variant: str = "test", batch_size: int = 1, return_loader: bool = True
) -> tuple[T.Iterator[InputData], Metadata]:
    """
    Create an iterator of a dataloader as specified in a configuration file.

    Parameters
    ----------
    config
        The configuration file to use.
    variant
        The variant of the dataset to use. Defaults to `test`.
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

    from .utils.config import instantiate

    config = read_config(config)

    if variant not in config.data.loaders:
        raise KeyError(
            f"Variant {variant} not found in data configuration, available variants are {dict(config.data.loaders).keys()}"
        )

    datafactory = instantiate(config.data.loaders[variant])
    dataloader = datafactory(batch_size)
    if return_loader:
        return dataloader, datafactory.dataset.info
    else:
        return iter(datafactory(batch_size)), datafactory.dataset.info


def prepare_images(
    images_dir: str | os.PathLike,
    *,
    batch_size: int = 1,
    suffix: T.Collection[str] = (".jpg", ".png"),
    separator: str | None = os.sep,
    return_loader: bool = True,
    ops: T.Sequence[Op] = [],
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

    from .model import InputData

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
    dataset = _ImageFolder(image_intseq, ops)

    # Create the dataloader
    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        collate_fn=InputData.collate,
        shuffle=False,
        num_workers=0,
    )

    return loader if return_loader else iter(loader)



# def read_image(image: ImageParam) -> torch.Tensor:
#     """
#     Read an image from a file or coerce it to a Tensor if it is in PIL or NumPy. If a Tensor is passed, it is
#     returned as-is.

#     Parameters
#     ----------
#     image
#         The image file path, a PIL image, a NumPy array or a Tensor.

#     Returns
#     -------
#     image
#         The image as a Tensor.
#     """
#     from torchvision.transforms.v2.functional import pil_to_tensor

#     if isinstance(image, str) or isinstance(image, os.PathLike):
#         image_path = file_io.Path(image)
#         if not image_path.is_file():
#             raise FileNotFoundError(f"Could not find image file at {image_path}")
#         image = pil_image.open(image_path)
#     if isinstance(image, pil_image.Image):
#         return pil_to_tensor(image)
#     elif isinstance(image, np.ndarray):
#         return torch.from_numpy(image)
#     elif isinstance(image, torch.Tensor):
#         return image
#     else:
#         raise TypeError(f"Expected an image file path, a PIL image, a NumPy array or a Tensor, got {image}")


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

    from .data.tensors import Image
    from .model import InputData, CaptureData, CameraModel

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
        assert image.ndim == 3 and image.shape[0] == 3, f"Expected am RGB image, got {image.shape}"
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
            image_size=torch.tensor([img.shape[-2:] for img in batch], dtype=torch.float32),
            matrix=torch.eye(4, dtype=torch.float32).repeat(len(batch), 1, 1),
            pose=torch.eye(4, dtype=torch.float32).repeat(len(batch), 1, 1),
            batch_size=[len(batch)],
        ),
        content_boxes=torch.stack(
            [torch.tensor([0, 0, img.shape[-2], img.shape[-1]], dtype=torch.float32) for img in batch]
        ),
        batch_size=[len(batch)],
    )
    return inputs
