"""
This file is used to configure `setuptools` to build the extension modules.

See Also
--------
- `SetupTools Docs <https://setuptools.pypa.io/en/latest/userguide/ext_modules.html>`_
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension

ENVIRON_TRUE = frozenset({"1", "true"})


def locate_cuda_includes():
    result = []
    cub_home = os.environ.get("CUB_HOME", None)
    if cub_home is None:
        prefix = os.environ.get("CONDA_PREFIX", None)
        if prefix is not None and os.path.isdir(prefix + "/include/cub"):
            cub_home = prefix + "/include"
    if cub_home is None:
        warnings.warn(
            "The environment variable `CUB_HOME` was not found. "
            "See: `https://github.com/NVIDIA/cub/releases` "
        )
    else:
        result.append(os.path.realpath(cub_home).replace("\\ ", " "))
    return result


def should_skip(name: str):
    skip_names = os.environ.get("UP_EXTENSIONS_SKIP", "").lower().split(",")
    if "*" in skip_names:
        return True
    return name.lower() in skip_names


def find_extension(name: str):
    from torch import cuda
    from torch.utils.cpp_extension import CUDA_HOME, CppExtension, CUDAExtension

    if should_skip(name):
        warnings.warn(f"Skipping extension: {name} (disabled by user)")
        return []

    root = Path(__file__).parent / "sources" / name.replace(".", "/")
    assert root.is_dir(), f"Directory not found: {root}"

    ext_files = list(root.glob("**/*.cpp"))
    ext_compile_args = {
        "cxx": ["-std=c++17"],
    }
    ext_define_macros = []
    ext_include = [str(root)]

    if cuda.is_available() or os.environ.get("FORCE_CUDA", "").lower() in ENVIRON_TRUE:
        if CUDA_HOME is None:
            msg = (
                "CUDA_HOME environment variable not set. Check your CUDA installation."
            )
            raise OSError(msg)

        ext_cls = CUDAExtension
        ext_include.extend(locate_cuda_includes())
        ext_files.extend(root.rglob("cuda/*.cu"))
        ext_define_macros += [("WITH_CUDA", None)]
        ext_compile_args["nvcc"] = [
            "-DCUDA_HAS_FP16=1",
            "-D__CUDA_NO_HALF_OPERATORS__",
            "-D__CUDA_NO_HALF_CONVERSIONS__",
            "-D__CUDA_NO_HALF2_OPERATORS__",
            "-O3",
        ]

        if os.name != "nt":
            ext_compile_args["nvcc"].append("--std=c++17")
    else:
        ext_cls = CppExtension
        # ext_files = list(root.glob("cpu/*.cpp"))
    return [
        ext_cls(
            f"{name}.extension",
            sorted(map(str, ext_files)),  # sort for reproducibility
            include_dirs=ext_include,
            define_macros=ext_define_macros,
            extra_compile_args=ext_compile_args,
        )
    ]


setup(
    ext_modules=[
        *find_extension("unipercept.nn.layers.deform_conv"),
        *find_extension("unipercept.nn.layers.deform_attn"),
        *find_extension("unipercept.vision.knn_points"),
    ],
    cmdclass={"build_ext": BuildExtension},
)
