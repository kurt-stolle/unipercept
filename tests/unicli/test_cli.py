import pytest
from unicli import command


@pytest.mark.unit
@pytest.mark.parametrize("flag", ["--version", "--help"])
def test_cli_root_exit(flag):
    with pytest.raises(SystemExit):
        command.root(["flag"])
