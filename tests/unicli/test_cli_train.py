import omegaconf
import pytest
from unicli import command
from unicli.train import main

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
def test_cli_train(parser, tmp_path):
    cfg_out = tmp_path / "output"
    cfg_out.mkdir()

    cfg_dir = tmp_path / "configs"
    cfg_dir.mkdir()

    cfg_path = cfg_dir / "config.py"
    cfg_path.write_text(
        TEST_CONFIG.format(
            cfg_out=cfg_out.as_posix(),
        )
    )

    cmd = ["train", "--config", cfg_path.as_posix()]

    args = parser.parse_args(cmd)

    # When the error `Missing key device` is raised, the test passes. This means
    # the model was successfully initialized from configuration.
    with pytest.raises(omegaconf.errors.ConfigAttributeError, match="Missing key device"):
        main(args)

    assert (cfg_out / "test").exists()
    assert (cfg_out / "test" / "config.yaml").exists()

    with pytest.raises(FileExistsError, match="File exists:.+test"):
        main(args)

    # Reset mode should delete the output directory.
    cmd.append("--headless")
    cmd.append("--reset")

    args = parser.parse_args(cmd)

    with pytest.raises(omegaconf.errors.ConfigAttributeError, match="Missing key device"):
        main(args)


@pytest.mark.unit
def test_cli_train_eval_only(parser, tmp_path):
    cfg_out = tmp_path / "output"
    cfg_out.mkdir()

    cfg_dir = tmp_path / "configs"
    cfg_dir.mkdir()

    cfg_path = cfg_dir / "config.py"
    cfg_path.write_text(
        TEST_CONFIG.format(
            cfg_out=cfg_out.as_posix(),
        )
    )

    cmd = ["train", "--config", cfg_path.as_posix(), "--eval-only"]
    args = parser.parse_args(cmd)
    with pytest.raises(omegaconf.errors.ConfigAttributeError, match="Missing key device"):
        main(args)
