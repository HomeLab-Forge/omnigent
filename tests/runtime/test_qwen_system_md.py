"""
Tests that ``executor.config.system_md`` reaches qwen as ``QWEN_SYSTEM_MD``.

ACP has no system-prompt slot, so Omnigent delivers the agent prompt as a user
turn and Qwen Code's own baked prompt keeps the system role — memory subsystem,
bundled skill catalogue, todo doctrine, and few-shot examples that write tool
calls as prose. ``QWEN_SYSTEM_MD`` is Qwen's supported override; this locks the
wiring, and the containment that stops a bundle naming a path outside itself.

Unit test — no subprocess spawn, no real CLI.
"""

from __future__ import annotations

from pathlib import Path

from omnigent.runtime.workflow import _build_qwen_spawn_env
from omnigent.spec.types import AgentSpec, ExecutorSpec


def _spec(config: dict[str, object]) -> AgentSpec:
    """
    Build a minimal qwen agent spec carrying *config*.

    :param config: The ``executor.config`` mapping under test.
    :returns: An :class:`AgentSpec` with a qwen executor.
    """
    return AgentSpec(
        spec_version=1,
        name="watchdog",
        executor=ExecutorSpec(type="omnigent", config=dict(config)),
    )


def test_system_md_resolves_against_the_bundle(tmp_path: Path) -> None:
    """A bundle-relative ``system_md`` becomes an absolute ``QWEN_SYSTEM_MD``."""
    (tmp_path / "system.md").write_text("agent instructions", encoding="utf-8")
    env = _build_qwen_spawn_env(_spec({"system_md": "system.md"}), workdir=tmp_path)
    assert env["QWEN_SYSTEM_MD"] == str((tmp_path / "system.md").resolve())


def test_system_md_absent_when_unset(tmp_path: Path) -> None:
    """No ``system_md`` leaves Qwen's own prompt in place."""
    env = _build_qwen_spawn_env(_spec({}), workdir=tmp_path)
    assert "QWEN_SYSTEM_MD" not in env


def test_system_md_cannot_escape_the_bundle(tmp_path: Path) -> None:
    """A path climbing out of the bundle is refused, not resolved."""
    outside = tmp_path.parent / "escape.md"
    outside.write_text("elsewhere", encoding="utf-8")
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    env = _build_qwen_spawn_env(
        _spec({"system_md": "../escape.md"}), workdir=bundle
    )
    assert "QWEN_SYSTEM_MD" not in env


def test_system_md_missing_file_is_skipped(tmp_path: Path) -> None:
    """A named file that is not there is a warning, not a broken spawn."""
    env = _build_qwen_spawn_env(_spec({"system_md": "missing.md"}), workdir=tmp_path)
    assert "QWEN_SYSTEM_MD" not in env


def test_system_md_needs_a_bundle_dir(tmp_path: Path) -> None:
    """Without a bundle on disk the relative path has no meaning."""
    env = _build_qwen_spawn_env(_spec({"system_md": "system.md"}), workdir=None)
    assert "QWEN_SYSTEM_MD" not in env
