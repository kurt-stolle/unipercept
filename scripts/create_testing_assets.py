"""
Quick script to generate the assets for testing.
"""

from __future__ import annotations

import re
from pathlib import Path

import unipercept as up


class TestingAssetDataset(up.data.sets.FolderPatternDataset, root=Path(__file__).parent / "assets" / "testing", pattern=re.compile(r"(\d{4})/(\d{6}).png$")):
        Path(__file__).parent / "assets" / "testing",
        re.compile(r"(\d{4})/(\d{6}).png$"),
        depth_path=lambda p: 
    )

if __name__ == "__main__":
    