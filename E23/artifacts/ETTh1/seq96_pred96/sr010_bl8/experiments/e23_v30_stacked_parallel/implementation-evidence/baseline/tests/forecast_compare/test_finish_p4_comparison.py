"""Guard the one-off detached P4 continuation against duplicate or failed jobs."""

import json

import pytest

from scripts import finish_p4_comparison as finish


def test_wait_requires_successful_exit(tmp_path):
    job = tmp_path / "search"
    job.mkdir()
    (job / "status.json").write_text(json.dumps({"state": "finished", "exit_code": 2}))
    with pytest.raises(RuntimeError, match="exit 2"):
        finish.wait_for_success(job, 0.01)
    (job / "status.json").write_text(json.dumps({"state": "finished", "exit_code": 0}))
    finish.wait_for_success(job, 0.01)


def test_launch_refuses_existing_job(tmp_path):
    job = tmp_path / "existing"
    job.mkdir()
    with pytest.raises(RuntimeError, match="refusing to overwrite"):
        finish.launch_and_wait(job, ["/bin/true"], 0.01)


def test_selected_tests_and_report_are_distinct():
    missing = finish.s4m_evaluation_command("evaluate-missing")
    clean = finish.s4m_evaluation_command("evaluate-clean")
    report = finish.report_command()
    assert missing[-1] == "evaluate-missing"
    assert clean[-1] == "evaluate-clean"
    assert "--saits-evaluation" in report
    assert str(finish.BASE / "saits/evaluations/saits-test-v2.json") in report
    assert str(finish.BASE / "reports/comparison_p4_v2") in report
    with pytest.raises(ValueError, match="invalid S4M evaluation phase"):
        finish.s4m_evaluation_command("search")
