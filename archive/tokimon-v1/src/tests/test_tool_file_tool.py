from __future__ import annotations

from pathlib import Path

from tools.file_tool import FileTool


def test_file_tool_write_and_read_roundtrip(tmp_path: Path) -> None:
    tool = FileTool(tmp_path)

    write_result = tool.write("notes/todo.txt", "hello")
    assert write_result.ok is True

    read_result = tool.read("notes/todo.txt")
    assert read_result.ok is True
    assert read_result.data["content"] == "hello"
    assert Path(read_result.data["path"]) == tmp_path / "notes" / "todo.txt"


def test_file_tool_blocks_path_traversal(tmp_path: Path) -> None:
    tool = FileTool(tmp_path)
    result = tool.read("../secrets.txt")
    assert result.ok is False
    assert result.error == "Path traversal detected"


def test_file_tool_read_missing_file_returns_error(tmp_path: Path) -> None:
    tool = FileTool(tmp_path)
    result = tool.read("does-not-exist.txt")
    assert result.ok is False
    assert result.summary == "read failed"
    assert result.error
