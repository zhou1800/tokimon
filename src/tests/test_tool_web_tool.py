from tools.web_tool import WebTool


def test_web_tool_rejects_non_http_urls() -> None:
    tool = WebTool()
    result = tool.fetch("file:///etc/passwd")
    assert result.ok is False


def test_web_tool_rejects_localhost() -> None:
    tool = WebTool()
    result = tool.fetch("http://localhost:8000/")
    assert result.ok is False

