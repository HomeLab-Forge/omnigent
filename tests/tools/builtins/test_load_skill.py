"""Tests for omnigent.tools.builtins.load_skill."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from omnigent.spec.types import SkillSpec
from omnigent.tools.base import ToolContext
from omnigent.tools.builtins import LoadSkillTool


@pytest.fixture()
def skill_with_resources(tmp_path: Path) -> SkillSpec:
    """
    A skill with a ``references/`` directory containing a
    file, for testing resource listing in load_skill output.

    :returns: A ``SkillSpec`` pointing at a real directory
        with a reference file.
    """
    skill_dir = tmp_path / "skills" / "code-review"
    skill_dir.mkdir(parents=True)
    refs_dir = skill_dir / "references"
    refs_dir.mkdir()
    (refs_dir / "style-guide.md").write_text("# Style Guide\n\nUse snake_case.")
    return SkillSpec(
        name="code-review",
        description="Reviews code.",
        content="Review the code.",
        skill_dir=skill_dir,
    )


@pytest.fixture()
def skill_no_resources() -> SkillSpec:
    """
    A skill with no ``skill_dir`` (in-memory only).

    :returns: A ``SkillSpec`` with ``skill_dir=None``.
    """
    return SkillSpec(
        name="summarize",
        description="Summarizes text.",
        content="Summarize the input concisely.",
    )


def test_load_skill_returns_content(
    skill_no_resources: SkillSpec,
    tool_ctx: ToolContext,
) -> None:
    """
    LoadSkillTool.invoke returns the skill's content string.
    """
    tool = LoadSkillTool([skill_no_resources])
    result = tool.invoke(json.dumps({"name": "summarize"}), tool_ctx)
    assert result == "Summarize the input concisely."


def test_load_skill_not_found(
    skill_no_resources: SkillSpec,
    tool_ctx: ToolContext,
) -> None:
    """
    LoadSkillTool.invoke returns error for unknown skill name.
    """
    tool = LoadSkillTool([skill_no_resources])
    result = tool.invoke(json.dumps({"name": "nonexistent"}), tool_ctx)
    assert "not found" in result
    assert "summarize" in result


def test_load_skill_with_resources_lists_files(
    skill_with_resources: SkillSpec,
    tool_ctx: ToolContext,
) -> None:
    """
    LoadSkillTool.invoke appends a resource listing when the
    skill has bundled reference files.
    """
    tool = LoadSkillTool([skill_with_resources])
    result = tool.invoke(
        json.dumps({"name": "code-review"}),
        tool_ctx,
    )
    assert "Review the code." in result
    assert "references/style-guide.md" in result
    assert "read_skill_file" in result


def test_load_skill_missing_name_argument(
    skill_no_resources: SkillSpec,
    tool_ctx: ToolContext,
) -> None:
    """
    LoadSkillTool.invoke returns error when 'name' is missing.
    """
    tool = LoadSkillTool([skill_no_resources])
    result = tool.invoke(json.dumps({}), tool_ctx)
    assert "missing required 'name'" in result


@pytest.mark.parametrize("arguments", ["not-json", "[]"])
def test_load_skill_rejects_invalid_arguments(
    arguments: str,
    skill_no_resources: SkillSpec,
    tool_ctx: ToolContext,
) -> None:
    """
    Malformed or non-object arguments return an error string.
    """
    tool = LoadSkillTool([skill_no_resources])
    result = tool.invoke(arguments, tool_ctx)

    assert result.startswith("Error:")


def test_load_skill_rejects_non_string_name(
    skill_no_resources: SkillSpec,
    tool_ctx: ToolContext,
) -> None:
    """
    ``name`` must be a string skill name.
    """
    tool = LoadSkillTool([skill_no_resources])
    result = tool.invoke(json.dumps({"name": 123}), tool_ctx)

    assert result == "Error: 'name' must be a string"


@pytest.fixture()
def conversation_ctx() -> ToolContext:
    """
    A :class:`ToolContext` carrying a conversation id.

    The shared ``tool_ctx`` fixture leaves ``conversation_id`` unset, which is
    the "every load is a first load" path.

    :returns: A context scoped to ``"conv_alice"``.
    """
    return ToolContext(
        task_id="task_test",
        agent_id="agent_test",
        conversation_id="conv_alice",
    )


def test_repeat_load_returns_the_content_behind_a_note(
    skill_no_resources: SkillSpec,
    conversation_ctx: ToolContext,
) -> None:
    """
    A second load of the same skill repeats the instructions, prefixed.

    Withholding them would be worse than the loop it prevents: a skill lives
    only as this tool's output in the transcript, so once compaction drops
    that output the second call is the only way back to the instructions.
    """
    tool = LoadSkillTool([skill_no_resources])
    args = json.dumps({"name": "summarize"})

    first = tool.invoke(args, conversation_ctx)
    second = tool.invoke(args, conversation_ctx)

    assert not first.startswith("[Already loaded")
    assert second.startswith("[Already loaded")
    assert skill_no_resources.content in second, (
        f"the repeat load must still carry the instructions; got: {second!r}"
    )


def test_already_loaded_is_scoped_per_conversation(
    skill_no_resources: SkillSpec,
) -> None:
    """One conversation's load must not mark the skill loaded in another."""
    tool = LoadSkillTool([skill_no_resources])
    args = json.dumps({"name": "summarize"})
    alice = ToolContext(task_id="t", agent_id="a", conversation_id="conv_alice")
    bob = ToolContext(task_id="t", agent_id="a", conversation_id="conv_bob")

    tool.invoke(args, alice)

    assert not tool.invoke(args, bob).startswith("[Already loaded")
    assert tool.invoke(args, alice).startswith("[Already loaded")


def test_a_failed_load_is_not_recorded(
    skill_no_resources: SkillSpec,
    conversation_ctx: ToolContext,
) -> None:
    """An unknown skill name must not mark anything as loaded."""
    tool = LoadSkillTool([skill_no_resources])

    tool.invoke(json.dumps({"name": "nonexistent"}), conversation_ctx)
    result = tool.invoke(json.dumps({"name": "summarize"}), conversation_ctx)

    assert not result.startswith("[Already loaded")


def test_load_skill_schema_lists_skill_names(
    skill_no_resources: SkillSpec,
    skill_with_resources: SkillSpec,
) -> None:
    """
    LoadSkillTool.get_schema includes all skill names in the
    description.
    """
    tool = LoadSkillTool(
        [skill_no_resources, skill_with_resources],
    )
    schema = tool.get_schema()
    desc = schema["function"]["description"]
    assert "summarize" in desc
    assert "code-review" in desc
