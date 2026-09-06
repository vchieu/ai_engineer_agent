"""
Test cho agent/nodes.py.

- node_2b_cleaner: pure function (regex), test trực tiếp không cần mock gì.
- node_1_router: có gọi LLM qua safe_llm_call -> mock hàm này ở cấp module
  "agent.nodes.safe_llm_call" để không tốn API call thật.
"""
from unittest.mock import patch, MagicMock

from schemas.payload import AgentState, RouterDecision, MissingFieldRequest
from agent.nodes import node_2b_cleaner, node_1_router, node_terminal_missing_info


class TestNodeCleaner:
    def test_removes_iso_timestamps(self):
        state = AgentState(user_input="2024-01-01T10:00:00 Error: crash")
        result = node_2b_cleaner(state)
        assert "2024-01-01" not in result.cleaned_logs
        assert "Error: crash" in result.cleaned_logs

    def test_removes_ansi_escape_codes(self):
        state = AgentState(user_input="\x1b[31mError\x1b[0m: crash")
        result = node_2b_cleaner(state)
        assert "\x1b" not in result.cleaned_logs
        assert "Error: crash" in result.cleaned_logs

    def test_strips_blank_lines_and_whitespace(self):
        state = AgentState(user_input="line1\n\n   \nline2   ")
        result = node_2b_cleaner(state)
        assert result.cleaned_logs == "line1\nline2"

    def test_combined_timestamp_ansi_and_blank_lines(self):
        raw = "2024-01-01T10:00:00 \x1b[31mError\x1b[0m: crash\n\n  \nline2"
        state = AgentState(user_input=raw)
        result = node_2b_cleaner(state)
        assert result.cleaned_logs == "Error: crash\nline2"


class TestNodeRouter:
    @patch("agent.nodes.safe_llm_call")
    def test_fix_bug_intent_sets_state_correctly(self, mock_safe_call):
        decision = RouterDecision(intent="FIX_BUG", reasoning="có log lỗi rõ ràng", missing_fields=None)
        mock_safe_call.return_value = (
            decision,
            {"model_name": "gpt-test", "node_name": "node_1_router", "tokens": 15},
        )

        state = AgentState(user_input="Bug: crash khi chia cho 0")
        result = node_1_router(state, llm_strong=MagicMock())

        assert result.intent == "FIX_BUG"
        assert result.missing_fields is None
        assert result.total_tokens == 15
        assert result.token_usage_by_node.get("node_1_router") == 15

    @patch("agent.nodes.safe_llm_call")
    def test_missing_info_intent_populates_missing_fields(self, mock_safe_call):
        missing = [MissingFieldRequest(field="error_log", question="Bạn có log lỗi không?")]
        decision = RouterDecision(intent="MISSING_INFO", reasoning="thiếu log", missing_fields=missing)
        mock_safe_call.return_value = (
            decision,
            {"model_name": "gpt-test", "node_name": "node_1_router", "tokens": 10},
        )

        state = AgentState(user_input="Sửa giúp tôi cái bug")
        result = node_1_router(state, llm_strong=MagicMock())

        assert result.intent == "MISSING_INFO"
        assert result.missing_fields[0].field == "error_log"

    @patch("agent.nodes.safe_llm_call")
    def test_token_accumulates_across_multiple_router_calls(self, mock_safe_call):
        """Mô phỏng vòng lặp MISSING_INFO -> router chạy lại nhiều lần, token phải cộng dồn."""
        decision = RouterDecision(intent="FIX_BUG", reasoning="ok", missing_fields=None)
        mock_safe_call.return_value = (
            decision,
            {"model_name": "gpt-test", "node_name": "node_1_router", "tokens": 10},
        )

        state = AgentState(user_input="bug")
        state = node_1_router(state, llm_strong=MagicMock())
        state = node_1_router(state, llm_strong=MagicMock())

        assert state.total_tokens == 20
        assert state.token_usage_by_node["node_1_router"] == 20


class TestNodeTerminalMissingInfo:
    def test_sets_failed_missing_info_status(self):
        state = AgentState(user_input="x", max_missing_info_retries=3)
        result = node_terminal_missing_info(state)
        assert result.final_status == "FAILED_MISSING_INFO"
        assert result.error_message is not None
