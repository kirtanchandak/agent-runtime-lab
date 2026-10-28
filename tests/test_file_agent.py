from agent_runtime import ToolCall
from examples.file_agent import build_file_tools


def test_file_tools_read_and_write_inside_the_workspace(tmp_path):
    workspace = tmp_path / "workspace"
    outside = tmp_path / "secret.txt"
    outside.write_text("nope")
    registry = build_file_tools(workspace)

    written = registry.execute(
        ToolCall(id="1", name="write_file", arguments={"path": "notes/hello.txt", "content": "hi"})
    )
    listed = registry.execute(ToolCall(id="2", name="list_files"))
    read = registry.execute(ToolCall(id="3", name="read_file", arguments={"path": "notes/hello.txt"}))
    escaped = registry.execute(ToolCall(id="4", name="read_file", arguments={"path": "../secret.txt"}))
    overwritten = registry.execute(
        ToolCall(id="5", name="write_file", arguments={"path": "../secret.txt", "content": "pwned"})
    )

    assert written.output == "wrote notes/hello.txt"
    assert "notes/hello.txt" in (listed.output or "")
    assert read.output == "hi"
    assert escaped.ok is False
    assert "escapes" in (escaped.error or "")
    assert overwritten.ok is False
    assert outside.read_text() == "nope"


def test_missing_file_is_a_tool_error(tmp_path):
    registry = build_file_tools(tmp_path / "workspace")

    result = registry.execute(ToolCall(id="1", name="read_file", arguments={"path": "missing.txt"}))

    assert result.ok is False
    assert "missing.txt" in (result.error or "")
