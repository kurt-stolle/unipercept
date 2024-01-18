from pathlib import Path

from torchdata.datapipes import iter, map

from unipercept.data.pipes.io_wrap import UniCoreFileLister


def test_file_lister_pipe():
    file_lister = UniCoreFileLister(".", masks="*.py")
    assert isinstance(file_lister, iter.IterDataPipe)
    assert not isinstance(file_lister, map.MapDataPipe)


def test_file_opener_pipe(tmp_path: Path):
    file_input = tmp_path / "test.txt"
    file_input.write_text("test")

    file_opener = iter.IterableWrapper([str(file_input)]).open_files_by_unicore()
    assert isinstance(file_opener, iter.IterDataPipe)
    assert not isinstance(file_opener, map.MapDataPipe)

    for path, stream in file_opener:
        assert path == str(file_input)
        content = stream.read()
        assert content == "test"


def test_file_saver_pipe():
    pass
