"""
Test cho llm/safe_call.py.

Không gọi OpenAI API thật (không tốn tiền, không cần OPENAI_API_KEY,
không cần mạng) — mock lại đúng shape mà langchain's
`with_structured_output(schema, include_raw=True).invoke(messages)` trả về:
    {"raw": <AIMessage-like>, "parsed": <schema instance | None>, "parsing_error": <Exception | None>}
"""
from unittest.mock import MagicMock
import pytest
from pydantic import BaseModel

from llm.safe_call import safe_llm_call


class DummySchema(BaseModel):
    value: str


def _make_llm_client(invoke_results):
    """invoke_results: list các dict trả về lần lượt mỗi lần .invoke() được gọi."""
    structured_llm = MagicMock()
    structured_llm.invoke.side_effect = invoke_results
    client = MagicMock()
    client.with_structured_output.return_value = structured_llm
    return client


class TestSafeLlmCallSuccess:
    def test_success_on_first_attempt_extracts_tokens(self):
        parsed = DummySchema(value="ok")
        raw_msg = MagicMock()
        raw_msg.usage_metadata = {"total_tokens": 42}
        client = _make_llm_client([
            {"raw": raw_msg, "parsed": parsed, "parsing_error": None},
        ])

        result, stats = safe_llm_call(
            client, DummySchema, "some prompt",
            model_name="gpt-test", node_name="test_node",
        )

        assert result.value == "ok"
        assert stats == {"model_name": "gpt-test", "node_name": "test_node", "tokens": 42}

    def test_accepts_message_list_input(self):
        from langchain_core.messages import HumanMessage
        parsed = DummySchema(value="ok")
        client = _make_llm_client([
            {"raw": MagicMock(usage_metadata=None), "parsed": parsed, "parsing_error": None},
        ])

        result, _ = safe_llm_call(client, DummySchema, [HumanMessage(content="hi")])
        assert result.value == "ok"

    def test_falls_back_to_response_metadata_token_usage(self):
        parsed = DummySchema(value="ok")
        raw_msg = MagicMock()
        raw_msg.usage_metadata = None
        raw_msg.response_metadata = {"token_usage": {"total_tokens": 7}}
        client = _make_llm_client([
            {"raw": raw_msg, "parsed": parsed, "parsing_error": None},
        ])

        _, stats = safe_llm_call(client, DummySchema, "prompt")
        assert stats["tokens"] == 7


class TestSafeLlmCallRetry:
    def test_retries_after_parsing_error_then_succeeds(self):
        parsed = DummySchema(value="ok")
        client = _make_llm_client([
            {"raw": None, "parsed": None, "parsing_error": "bad json format"},
            {"raw": MagicMock(usage_metadata=None), "parsed": parsed, "parsing_error": None},
        ])

        result, _ = safe_llm_call(client, DummySchema, "prompt", max_retries=3)
        assert result.value == "ok"
        assert client.with_structured_output.return_value.invoke.call_count == 2

    def test_retries_when_parsed_is_none_without_explicit_error(self):
        parsed = DummySchema(value="ok")
        client = _make_llm_client([
            {"raw": MagicMock(usage_metadata=None), "parsed": None, "parsing_error": None},
            {"raw": MagicMock(usage_metadata=None), "parsed": parsed, "parsing_error": None},
        ])

        result, _ = safe_llm_call(client, DummySchema, "prompt", max_retries=3)
        assert result.value == "ok"

    def test_raises_runtime_error_after_max_retries_exhausted(self):
        client = _make_llm_client([
            {"raw": None, "parsed": None, "parsing_error": "bad"},
            {"raw": None, "parsed": None, "parsing_error": "bad"},
            {"raw": None, "parsed": None, "parsing_error": "bad"},
        ])

        # RuntimeError message contains both the node name and "unknown_node" default
        # (it interpolates node_name into the message), so we just match the node name.
        with pytest.raises(RuntimeError, match="node_1_router"):
            safe_llm_call(client, DummySchema, "prompt", max_retries=3, node_name="node_1_router")

        assert client.with_structured_output.return_value.invoke.call_count == 3
