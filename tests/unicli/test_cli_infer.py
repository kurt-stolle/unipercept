import omegaconf
import pytest
from unicli.infer import main

TEST_CONFIG = """
from detectron2.config import LazyCall as L
from torch import nn
from omegaconf import OmegaConf

train = dict(
    output_dir="{cfg_out}/test"
)
model = L(nn.Module)()
"""


@pytest.mark.unit
def test_cli_infer(parser, tmp_path):
    cfg_out = tmp_path / "output"  # output dir created by CLI

    cfg_dir = tmp_path / "configs"  # config dir populated with mock data
    cfg_dir.mkdir()
    cfg_path = cfg_dir / "config.py"
    cfg_path.write_text(
        TEST_CONFIG.format(
            cfg_out=cfg_out.as_posix(),
        )
    )

    cmd = ["infer", "--config", cfg_path.as_posix(), "--output", cfg_out.as_posix()]

    args = parser.parse_args(cmd)

    # When the error `Missing key device` is raised, the test passes. This means
    # the model was successfully initialized from configuration.
    with pytest.raises(omegaconf.errors.ConfigAttributeError, match="Missing key dataset"):
        main(args)
