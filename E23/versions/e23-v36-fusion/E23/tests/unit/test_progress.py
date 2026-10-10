from __future__ import annotations

import malr.cli as cli


def test_progress_batches_reports_consumed_work_to_stderr(capsys) -> None:
    # Removing the progress wrapper or reporting on yield instead of completion
    # must make this test fail.
    progress_batches = getattr(cli, "_progress_batches", None)
    assert progress_batches is not None

    seen: list[int] = []

    def batches():
        for value in (1, 2, 3):
            seen.append(value)
            yield value

    tracked = iter(progress_batches(batches(), total=3, label="test fixture"))
    assert next(tracked) == 1
    assert seen == [1]
    assert "0/3" in capsys.readouterr().err

    assert next(tracked) == 2
    assert "1/3" in capsys.readouterr().err
    assert list(tracked) == [3]
    output = capsys.readouterr().err
    assert "3/3" in output
    assert "100%" in output
    assert "phase=test fixture" in output
    assert "ETA=" in output


def test_progress_batch_count_respects_partial_and_dropped_batches() -> None:
    batch_count = getattr(cli, "_progress_batch_count", None)
    assert batch_count is not None
    assert batch_count(5, 2, drop_last=False) == 3
    assert batch_count(5, 2, drop_last=True) == 2
    assert batch_count(0, 2, drop_last=False) == 0
