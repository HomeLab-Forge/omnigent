"""Pi must always be told the model's context window.

Omitting it leaves Pi on an internal default sized for no model in
particular. For a gateway model the catalog has no entry for, that default
is far below the real window, and Pi acts on it: it compacted at
``token_count = 20684`` in session c5723032, and after its compaction was
switched off, session d636e884 reached ~14.6k tokens and then called the
model without the user's query still in the conversation — which Qwen's
chat template rejects with "No user query found in messages."
"""

from __future__ import annotations

import pytest

from omnigent import model_catalog
from omnigent.inner.pi_executor import _pi_model_json_entry

MODEL_ID = "qwen3.6-27b-a3b-coder"


def _entry(context_window: int | None) -> dict:
    model = model_catalog.ModelEntry(
        id=MODEL_ID,
        family="qwen",
        metadata=model_catalog.ModelMetadata(context_window=context_window),
    )
    return _pi_model_json_entry(model)


@pytest.fixture(autouse=True)
def _no_ambient_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AP_CONTEXT_WINDOW_OVERRIDE", raising=False)


def test_catalog_metadata_is_used_when_present() -> None:
    assert _entry(65536)["contextWindow"] == 65536


def test_a_model_the_catalog_does_not_describe_still_gets_a_window() -> None:
    """The case that matters — a self-hosted model behind a gateway."""
    assert "contextWindow" in _entry(None)


def test_the_environment_override_reaches_pi(monkeypatch: pytest.MonkeyPatch) -> None:
    """``AP_CONTEXT_WINDOW_OVERRIDE`` exists for exactly this case."""
    monkeypatch.setenv("AP_CONTEXT_WINDOW_OVERRIDE", "131072")
    assert _entry(None)["contextWindow"] == 131072


def test_catalog_metadata_wins_over_the_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """A model the catalog describes is described correctly already."""
    monkeypatch.setenv("AP_CONTEXT_WINDOW_OVERRIDE", "131072")
    assert _entry(65536)["contextWindow"] == 65536


def test_max_output_tokens_stays_optional() -> None:
    """Only the context window is always sent; Pi defaults the rest sanely."""
    assert "maxTokens" not in _entry(None)
