"""
Tests for summarize_pr.py.

Gemini is replaced by a stub client, so these need no API key and make no
network calls.
"""

import io
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from google.genai import errors as genai_errors

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import summarize_pr  # noqa: E402
from summarize_pr import PRSummary, summarize_diff  # noqa: E402

DIFF = "diff --git a/auth.py b/auth.py\n+    limiter.check(request.ip)\n"

SAMPLE = PRSummary(
    title="Add rate limiting to login",
    summary="Login attempts are now rate limited per IP.",
    risk_level="medium",
    key_changes=["Add a rate limiter", "Lock the account after 5 failures"],
    testing_notes="Send six bad passwords in a row and expect a lockout.",
)


class StubClient:
    """Stands in for genai.Client: raises each queued error, then answers."""

    def __init__(self, errors=(), parsed=SAMPLE):
        self.errors = list(errors)
        self.parsed = parsed
        self.calls = []
        self.models = SimpleNamespace(generate_content=self._generate_content)

    def _generate_content(self, **kwargs):
        self.calls.append(kwargs)
        if self.errors:
            raise self.errors.pop(0)
        return SimpleNamespace(parsed=self.parsed)


def server_error():
    return genai_errors.ServerError(503, {"error": {"message": "model overloaded"}})


@pytest.fixture
def sleeps(monkeypatch):
    """Record retry delays instead of actually waiting."""
    recorded = []
    monkeypatch.setattr(summarize_pr.time, "sleep", recorded.append)
    return recorded


@pytest.fixture
def stub(monkeypatch):
    """Give main() an API key and a stub client in place of the real one."""
    client = StubClient()
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setattr(summarize_pr.genai, "Client", lambda api_key: client)
    return client


def test_returns_the_parsed_summary():
    client = StubClient()

    assert summarize_diff(client, DIFF) == SAMPLE
    assert len(client.calls) == 1


def test_sends_the_diff_and_asks_for_schema_validated_json():
    client = StubClient()

    summarize_diff(client, DIFF)

    call = client.calls[0]
    assert call["model"] == summarize_pr.MODEL
    assert DIFF in call["contents"]
    assert call["config"].response_mime_type == "application/json"
    assert call["config"].response_schema is PRSummary


def test_retries_when_the_model_is_temporarily_unavailable(sleeps):
    client = StubClient(errors=[server_error(), server_error()])

    assert summarize_diff(client, DIFF) == SAMPLE
    assert len(client.calls) == 3
    assert sleeps == [summarize_pr.RETRY_DELAY_SECONDS] * 2


def test_retry_notice_goes_to_stderr_not_stdout(sleeps, capsys):
    client = StubClient(errors=[server_error()])

    summarize_diff(client, DIFF)

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "retrying" in captured.err


def test_exits_clearly_when_the_reply_has_no_parsed_summary(sleeps):
    client = StubClient(parsed=None)

    with pytest.raises(SystemExit, match="no usable summary"):
        summarize_diff(client, DIFF)

    assert len(client.calls) == 1
    assert sleeps == []


def test_gives_up_after_the_last_retry(sleeps):
    client = StubClient(errors=[server_error() for _ in range(summarize_pr.MAX_RETRIES)])

    with pytest.raises(genai_errors.ServerError):
        summarize_diff(client, DIFF)

    assert len(client.calls) == summarize_pr.MAX_RETRIES
    assert len(sleeps) == summarize_pr.MAX_RETRIES - 1


def test_does_not_retry_a_bad_request(sleeps):
    bad_request = genai_errors.ClientError(400, {"error": {"message": "invalid argument"}})
    client = StubClient(errors=[bad_request])

    with pytest.raises(genai_errors.ClientError):
        summarize_diff(client, DIFF)

    assert len(client.calls) == 1
    assert sleeps == []


def test_main_needs_an_api_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setattr(sys, "argv", ["summarize_pr.py"])

    with pytest.raises(SystemExit, match="GEMINI_API_KEY is not set"):
        summarize_pr.main()


def test_main_rejects_an_empty_diff(monkeypatch, stub, tmp_path):
    empty = tmp_path / "empty.diff"
    empty.write_text("  \n", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["summarize_pr.py", str(empty)])

    with pytest.raises(SystemExit, match="No diff provided"):
        summarize_pr.main()

    assert stub.calls == []


def test_main_reads_the_diff_from_a_file_and_prints_the_summary(monkeypatch, stub, capsys):
    sample = ROOT / "sample_diff.txt"
    monkeypatch.setattr(sys, "argv", ["summarize_pr.py", str(sample)])

    summarize_pr.main()

    assert sample.read_text(encoding="utf-8") in stub.calls[0]["contents"]
    out = capsys.readouterr().out
    assert SAMPLE.title in out
    assert SAMPLE.risk_level in out
    assert SAMPLE.testing_notes in out
    for change in SAMPLE.key_changes:
        assert f"  - {change}" in out


def test_main_reads_the_diff_from_stdin(monkeypatch, stub):
    monkeypatch.setattr(sys, "argv", ["summarize_pr.py"])
    monkeypatch.setattr(sys, "stdin", io.StringIO(DIFF))

    summarize_pr.main()

    assert DIFF in stub.calls[0]["contents"]
