r"""
Quick utility to export metric data in a format that can be easily imported into further
analysis tools or reports.
"""

import enum as E

import pandas as pd

from unipercept import file_io

__all__ = ["ExportFormat", "export_dataframe"]


class ExportFormat(E.StrEnum):
    CSV = ".csv"
    EXCEL = ".xlsx"
    JSON = ".json"
    LATEX = ".tex"
    PARQUET = ".parquet"
    PICKLE = ".pkl"


def export_dataframe(
    df: pd.DataFrame,
    path: file_io.Pathable,
    format: ExportFormat | str | None = None,
):
    """
    Export metric data to a file.

    Parameters
    ----------
    df
        The DataFrame containing the metric data.
    path
        The path to save the data to.
    format
        The format to save the data in. If None, the format will be inferred from the file extension.
    """
    path = file_io.Path(path)
    if format is None:
        format = ExportFormat(path.suffix)
    match format:
        case ExportFormat.CSV:
            df.to_csv(path, index=False)
        case ExportFormat.EXCEL:
            df.to_excel(path, index=False)
        case ExportFormat.JSON:
            df.to_json(path, orient="records")
        case ExportFormat.LATEX:
            df.to_latex(path, index=False)
        case ExportFormat.PARQUET:
            df.to_parquet(path, index=False)
        case ExportFormat.PICKLE:
            df.to_pickle(path)
        case _:
            msg = f"Unsupported format: {format}"
            raise NotImplementedError(msg)
