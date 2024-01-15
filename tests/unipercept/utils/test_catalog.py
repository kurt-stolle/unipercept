import pytest

from unipercept.catalog import DataManager


@pytest.fixture
def catalog():
    return DataManager()


def test_parse_key(catalog):
    canon = catalog.parse_key

    assert canon("foo") == "foo"
    assert canon("Foo") == "foo"
    assert canon("FooBar") == "foobar"


def test_split_query(catalog):
    split = catalog.split_query

    assert split("foo") == ("foo", [])
    assert split("foo-bar") == ("foo-bar", [])
    assert split("foo-bar/baz") == ("foo-bar", ["baz"])
    assert split("foo-bar/baz/qux") == ("foo-bar", ["baz/qux"])


def test_catalog_register(catalog):
    @catalog.register_dataset(info=lambda: {"abc": 1})
    class FooData:
        test_value = "foo"

        @staticmethod
        def info(id_: str):
            return {"test_info": "bar"}

    assert catalog.get_dataset("foodata").test_value == "foo"

    with pytest.raises(KeyError):
        catalog.register_dataset("foodata")(FooData)


def test_catalog_info(catalog):
    @catalog.register_info("foodata")
    def _(variant):
        return {"test_info": variant}

    info = catalog.get_info("foodata/bar")
    assert info["test_info"] == "bar"
    info = catalog.get_info("foodata/baz")
    assert info["test_info"] == "baz"

    with pytest.raises(KeyError):
        catalog.get_info("bardata")
        nonexists = info["test_value"]
        assert nonexists is None
