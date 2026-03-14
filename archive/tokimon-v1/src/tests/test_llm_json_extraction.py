from __future__ import annotations

import json

from llm import client as llm_client


def test_extract_json_text_returns_object_on_empty() -> None:
    assert llm_client._extract_json_text("") == "{}"
    assert json.loads(llm_client._extract_json_text("")) == {}


def test_extract_json_text_strips_empty_code_fence() -> None:
    assert llm_client._extract_json_text("```json\n\n```") == "{}"
    assert llm_client._extract_json_text("```\n\n```") == "{}"


def test_extract_embedded_json_text_finds_object() -> None:
    snippet = llm_client._extract_embedded_json_text("prefix {\"a\": 1, \"b\": 2} suffix")
    assert snippet is not None
    assert json.loads(snippet) == {"a": 1, "b": 2}


def test_extract_embedded_json_text_finds_array() -> None:
    snippet = llm_client._extract_embedded_json_text("noise [1, 2, 3] trailing")
    assert snippet is not None
    assert json.loads(snippet) == [1, 2, 3]


def test_extract_embedded_json_text_returns_none_without_json() -> None:
    assert llm_client._extract_embedded_json_text("no json here") is None

