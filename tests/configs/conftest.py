from pathlib import Path

import pytest
import unipercept as up
from omegaconf import DictConfig, ListConfig

cfg_skip = set()
cfg_root = Path(__file__).parent.parent.parent / "configs"
cfg_files = [
    cfg_path.relative_to(cfg_root)
    for cfg_path in cfg_root.glob("**/*.py")
    if not cfg_path.name.startswith("_") and not any(skip in cfg_path.as_posix() for skip in cfg_skip)
]

assert len(cfg_files) > 0, "No configuration files found in: " + cfg_root.as_posix()


@pytest.fixture(params=cfg_files, ids=lambda p: "cfg:" + p.as_posix().replace(".py", ""), scope="session")
def cfg(request):
    cfg_path = cfg_root / request.param

    cfg = up.read_config(cfg_path)

    assert cfg is not None
    assert isinstance(cfg, (DictConfig)), type(cfg)
    return cfg
