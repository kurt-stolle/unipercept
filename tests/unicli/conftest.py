import pytest
from unicli import command


@pytest.fixture(scope="session")
def parser():
    p = command.get_parser()
    return p
