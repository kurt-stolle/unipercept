import omegaconf
import pytest


@pytest.mark.unit
def test_cli_echo_invalid(parser):
    with pytest.raises(SystemExit):
        parser.parse_args("echo")

    with pytest.raises(SystemExit):
        parser.parse_args("echo --config")

    with pytest.raises(SystemExit):
        parser.parse_args("echo --config 1")

    with pytest.raises(SystemExit):
        parser.parse_args("echo --config 1 2")


TEST_CONFIG = """
test_key = dict(
    test_value=True, 
    other_value="x", 
    nested_dict=dict(
        a=1, 
        b=2
    )
)
"""


@pytest.mark.unit
def test_cli_echo_config(parser, tmp_path):
    d = tmp_path / "configs"
    d.mkdir()

    f = d / "config.py"
    f.write_text(TEST_CONFIG)

    cmd = ["echo", "--config", str(f)]
    args = parser.parse_args(cmd)

    assert args is not None
    assert args.config is not None, args

    cfg = args.config
    assert isinstance(cfg, omegaconf.dictconfig.DictConfig)

    with pytest.raises(omegaconf.errors.ConfigAttributeError):
        cfg.not_there

    cfg_obj = omegaconf.OmegaConf.to_object(cfg)
    assert isinstance(cfg_obj, dict), cfg_obj
    assert "test_key" in cfg_obj, cfg_obj
    assert cfg_obj["test_key"]["test_value"] is True, cfg_obj


@pytest.mark.parametrize(
    "ov_arg,ov_value",
    [
        ("test_key.other_value=y", "y"),
        ("test_key.nested_dict.a=1000", 1000),
    ],
)
@pytest.mark.unit
def test_cli_echo_overrides(parser, tmp_path, ov_arg, ov_value):
    d = tmp_path / "configs"
    d.mkdir()

    f = d / "config.py"
    f.write_text(TEST_CONFIG)

    cmd = ["echo", "--config", str(f), ov_arg]
    args = parser.parse_args(cmd)

    assert args is not None
    assert args.config is not None, args

    cfg = args.config
    assert isinstance(cfg, omegaconf.dictconfig.DictConfig)

    with pytest.raises(omegaconf.errors.ConfigAttributeError):
        cfg.other_value

    for sub in ov_arg.split("=")[0].split("."):
        cfg = cfg.get(sub)

    assert cfg == ov_value
