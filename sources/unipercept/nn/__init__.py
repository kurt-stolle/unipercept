from __future__ import annotations

from unicore.utils.module import lazy_module_factory

__all__ = []
__getattr__, __dir__ = lazy_module_factory(__name__, ["backbones", "layers", "losses", "typings"])

del lazy_module_factory
