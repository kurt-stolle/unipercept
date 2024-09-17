from __future__ import annotations

import time

from unipercept.utils.ulid import ULID, Randomness, Timestamp


def test_ulid_timestamp():
    ts = Timestamp.generate()

    ts_enc = ts.str
    ts_dec = Timestamp.create(ts_enc)

    assert ts_dec == ts


def test_ulid_randomness():
    rand = Randomness.generate()

    rand_enc = rand.str
    rand_dec = Randomness.create(rand_enc)

    assert rand_dec == rand


def test_ulid_ordering():
    ulids = []
    for _ in range(10):
        ulids.append(ULID.generate())
        time.sleep(1e-3)

    ulids_sorted = sorted(ulids)

    assert ulids == ulids_sorted
