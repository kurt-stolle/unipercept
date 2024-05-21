"""
This file is used to configure `setuptools` to build the extension modules.

See Also
--------
- `SetupTools Docs <https://setuptools.pypa.io/en/latest/userguide/ext_modules.html>`_
"""

from __future__ import annotations

from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension


def find_extension(name: str):
    from torch import cuda
    from torch.utils.cpp_extension import CUDA_HOME, CppExtension, CUDAExtension

    root = Path(__file__).parent / "sources" / name.replace(".", "/")
    assert root.is_dir(), f"Directory not found: {root}"

    ext_files = list(root.glob("*.cpp"))
    ext_compile_args = {
        "cxx": [],
    }
    ext_define_macros = []

    if cuda.is_available() and CUDA_HOME is not None:
        if CUDA_HOME is None:
            msg = (
                "CUDA_HOME environment variable not set. Check your CUDA installation."
            )
            raise EnvironmentError(msg)

        ext_cls = CUDAExtension
        ext_files.extend(root.rglob("cuda/*.cu"))
        ext_define_macros += [("WITH_CUDA", None)]
        ext_compile_args["nvcc"] = [
            "-DCUDA_HAS_FP16=1",
            "-D__CUDA_NO_HALF_OPERATORS__",
            "-D__CUDA_NO_HALF_CONVERSIONS__",
            "-D__CUDA_NO_HALF2_OPERATORS__",
            "-O3",
        ]
    else:
        ext_cls = CppExtension
        ext_files = list(root.glob("cpu/*.cpp"))
    return ext_cls(
        f"{name}._ext",
        sorted(map(str, ext_files)),  # sort for reproducibility
        include_dirs=[str(root)],
        define_macros=ext_define_macros,
        extra_compile_args=ext_compile_args,
    )


setup(
    ext_modules=[
        find_extension("unipercept.nn.layers.deform_conv"),
        find_extension("unipercept.nn.layers.flash_attn"),
    ],
    cmdclass={"build_ext": BuildExtension},
)
