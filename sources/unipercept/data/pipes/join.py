from typing import Any, Callable, Generic, Iterable, Iterator, Sequence, TypeVar, cast

from torchdata.datapipes import functional_datapipe
from torchdata.datapipes.iter import IterDataPipe
from typing_extensions import override

__all__ = ["LeftIndexJoin"]

_T = TypeVar("_T", covariant=True)
_O = TypeVar("_O", contravariant=True)
_I = TypeVar("_I")
_J = TypeVar("_J")


@functional_datapipe("left_index_join_by_unicore")
class LeftIndexJoin(IterDataPipe[_T], Generic[_T, _J]):
    def __init__(
        self,
        target: IterDataPipe[_T],
        sources: Iterable[IterDataPipe],
        target_index: Callable[[_T], _I] = lambda x: x,
        source_index: Callable[[_O], _I] = lambda x: x,
        join: Callable[[_T, Sequence[_O | None]], _J] = lambda x, y: (x, *y),
    ):
        self.target = target
        self.sources = tuple(sources)
        self.source_index = source_index
        self.target_index = target_index
        self.join = join

    def __len__(self):
        return len(self.target)  # type: ignore

    @override
    def __iter__(self) -> Iterator[_J]:
        for item in self.target:
            idx_tgt = self.target_index(item)
            merge: list[Any | None] = [None] * len(self.sources)
            buffers = ({} for _ in range(len(self.sources)))

            for i_merge, (o, buf) in enumerate(zip(self.sources, buffers)):
                if idx_tgt in buf:
                    merge[i_merge] = buf[idx_tgt]
                    continue

                for o_item in o:
                    idx_src = self.source_index(o_item)
                    buf[idx_src] = o_item

                    if idx_src != idx_tgt:
                        continue

                    merge[i_merge] = o_item
                    break

            if self.join is None:
                yield cast(_J, (item, *merge))
            else:
                yield self.join(item, merge)
