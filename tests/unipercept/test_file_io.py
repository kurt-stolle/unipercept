from unipercept import file_io


def test_file_io_globals():
    for d in dir(file_io):
        assert getattr(file_io, d) is not None
