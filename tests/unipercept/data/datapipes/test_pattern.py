import pytest
from torchdata.datapipes import iter

from unipercept.datapipes.pattern import MatchMode, PatternFilter, PatternMatcher


@pytest.mark.parametrize(
    "mode",
    [MatchMode.ERROR, MatchMode.IGNORE, MatchMode("warn"), MatchMode.FILTER],
)
def test_pattern_matcher_pipe(mode):
    wrapper = iter.IterableWrapper(["foo", "bar", "foo-bar"])
    matcher = PatternMatcher(wrapper, r"^(?P<value_one>\w+)-(?P<value_two>\w+)$", mode="ignore")  # type: ignore

    it = (m for m in matcher.__iter__())

    match matcher.mode:
        case MatchMode.ERROR:
            with pytest.raises(ValueError):
                next(it)
            with pytest.raises(ValueError):
                next(it)
        case MatchMode.WARN:
            with pytest.warns(UserWarning):
                next(it)
                next(it)
        case MatchMode.FILTER:
            pass  # next will be foobar due to filter
        case MatchMode.IGNORE:
            foo, foo_txt = next(it)
            assert foo is None
            assert foo_txt == "foo"
            bar, bar_txt = next(it)
            assert bar is None
            assert bar_txt == "bar"
        case _:
            pytest.fail("Invalid mode")

    foobar, foobar_txt = next(it)
    assert foobar is not None
    assert foobar.group("value_one") == "foo"
    assert foobar.group("value_two") == "bar"
    assert foobar_txt == "foo-bar"

    with pytest.raises(StopIteration):
        next(it)


def test_pattern_filter_pipe():
    wrapper = iter.IterableWrapper(["foo", "bar", "foo-bar"])
    filter = PatternFilter(wrapper, r"^(?P<value_one>\w+)-(?P<value_two>\w+)$")  # type: ignore

    it = (m for m in filter.__iter__())

    foobar = next(it)
    assert foobar == "foo-bar"

    with pytest.raises(StopIteration):
        next(it)
