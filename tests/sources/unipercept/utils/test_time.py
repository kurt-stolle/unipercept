from unipercept.utils.time import profile, ProfileAccumulator
import time


def test_time_profiler():
    acc = ProfileAccumulator()

    for i in range(10):
        with profile(acc, "test"):
            time.sleep(i * 1e-4)

    for i in range(5):
        with profile(acc, "test2"):
            time.sleep(i * 1e-3)

    assert len(acc) == 2
    assert len(acc.means) == 2
    assert len(acc.sums) == 2
    assert len(acc.mins) == 2
    assert len(acc.maxs) == 2
    assert len(acc.stds) == 2
    assert len(acc.medians) == 2
    assert len(acc.variances) == 2
    assert len(acc.counts) == 2

    df_profile = acc.to_dataframe()
    assert len(df_profile) == 15

    print(df_profile.to_markdown(index=False, floatfmt=".3f"))

    df_summary = acc.to_summary()
    assert len(df_summary) == 2

    print(df_summary.to_markdown(index=False, floatfmt=".3f"))
