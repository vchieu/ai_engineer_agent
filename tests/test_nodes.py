"""
Test cho agent/nodes.py.

- node_2b_cleaner: pure function (regex), test trực tiếp không cần mock gì.
- node_1_router: có gọi LLM qua safe_llm_call -> mock hàm này ở cấp module
  "agent.nodes.safe_llm_call" để không tốn API call thật.
"""
from unittest.mock import patch, MagicMock

from schemas.payload import (
    AgentState, RouterDecision, MissingFieldRequest,
    PlanSchema, DiagnosisSchema, PlanAudit,
)
from agent.nodes import (
    node_2b_cleaner, node_1_router, node_terminal_missing_info,
    node_2a_planner, node_3b_diagnosis, node_plan_dispatch,
    node_3c_plan_reviewer,
)


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
        decision = RouterDecision(intent="FIX_BUG", reasoning="có log lỗi rõ ràng", missing_fields=None, complexity_hint="STANDARD")
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
        decision = RouterDecision(intent="MISSING_INFO", reasoning="thiếu log", missing_fields=missing, complexity_hint="STANDARD")
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
        decision = RouterDecision(intent="FIX_BUG", reasoning="ok", missing_fields=None, complexity_hint="STANDARD")
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


class TestNodePlanDispatch:
    """
    Test node THUẦN PYTHON, không cần mock LLM/Docker gì cả -> chạy siêu nhanh.
    Đây là lớp test quan trọng nhất của cả tính năng review plan, vì nó quyết định
    liệu 1 thay đổi rủi ro có thực sự được chặn lại để review hay không.
    """

    def _diagnosis_state(self, affected_files=None, approach="fix logic đơn giản", user_input="sửa bug nhỏ"):
        from schemas.payload import DiagnosisSchema
        diag = DiagnosisSchema(
            root_cause="null pointer",
            affected_files=affected_files or ["utils.py"],
            suggested_approach=approach,
            acceptance_criteria="test pass",
        )
        return AgentState(user_input=user_input, intent="FIX_BUG", diagnosis=diag)

    def test_small_simple_bug_fix_skips_review(self):
        state = self._diagnosis_state(affected_files=["utils.py"])
        result = node_plan_dispatch(state)
        assert result.review_strategy == "skip_review"

    def test_many_affected_files_triggers_llm_review(self):
        state = self._diagnosis_state(affected_files=[f"f{i}.py" for i in range(5)])
        result = node_plan_dispatch(state)
        assert result.review_strategy == "llm_review"

    def test_risk_keyword_in_approach_triggers_llm_review(self):
        state = self._diagnosis_state(approach="cần sửa lại logic auth token hết hạn")
        result = node_plan_dispatch(state)
        assert result.review_strategy == "llm_review"

    def test_risk_keyword_in_user_input_triggers_llm_review(self):
        state = self._diagnosis_state(user_input="Giúp tôi sửa bug liên quan tới database migration")
        result = node_plan_dispatch(state)
        assert result.review_strategy == "llm_review"

    def test_complexity_complex_triggers_llm_review(self):
        state = self._diagnosis_state()
        state.complexity_hint = "COMPLEX"
        result = node_plan_dispatch(state)
        assert result.review_strategy == "llm_review"

    def test_touching_critical_file_triggers_human_review(self):
        state = self._diagnosis_state(affected_files=["docker_runner.py"])
        result = node_plan_dispatch(state)
        assert result.review_strategy == "human_review"

    def test_force_human_review_overrides_everything(self):
        """force_human_review=True phải thắng tất cả các điều kiện khác, kể cả bug fix nhỏ."""
        state = self._diagnosis_state(affected_files=["utils.py"])
        state.force_human_review = True
        result = node_plan_dispatch(state)
        assert result.review_strategy == "human_review"

    def test_new_feature_non_trivial_triggers_llm_review_even_without_other_signals(self):
        plan = PlanSchema(steps=["a"], target_files=["a.py"], acceptance_criteria="ok")
        state = AgentState(user_input="thêm tính năng mới", intent="NEW_FEATURE", plan=plan, complexity_hint="STANDARD")
        result = node_plan_dispatch(state)
        assert result.review_strategy == "llm_review"


class TestPlanFeedbackInjection:
    @patch("agent.nodes.safe_llm_call")
    def test_planner_includes_reviewer_feedback_on_retry(self, mock_safe_call):
        plan = PlanSchema(steps=["a"], target_files=["a.py"], acceptance_criteria="ok")
        mock_safe_call.return_value = (
            plan,
            {"model_name": "gpt-test", "node_name": "node_2a_planner", "tokens": 5},
        )
        audit = PlanAudit(
            passed=False,
            identified_risks=["rủi ro"],
            audit_feedback="THIEU_XU_LY_EDGE_CASE",
            confidence=0.4,
        )
        state = AgentState(
            user_input="thêm tính năng",
            intent="NEW_FEATURE",
            plan_audit_result=audit,
        )

        node_2a_planner(state, llm_strong=MagicMock())

        sent_prompt = mock_safe_call.call_args[0][2]
        assert "THIEU_XU_LY_EDGE_CASE" in sent_prompt

    @patch("agent.nodes.safe_llm_call")
    def test_diagnosis_includes_human_feedback_on_retry(self, mock_safe_call):
        diagnosis = DiagnosisSchema(
            root_cause="null pointer",
            affected_files=["utils.py"],
            suggested_approach="fix logic",
            acceptance_criteria="test pass",
        )
        mock_safe_call.return_value = (
            diagnosis,
            {"model_name": "gpt-test", "node_name": "node_3b_diagnosis", "tokens": 5},
        )
        state = AgentState(
            user_input="sửa bug",
            intent="FIX_BUG",
            cleaned_logs="crash",
            plan_review_feedback="KIEM_TRA_NHIỀU_EDGE_CASE",
        )

        node_3b_diagnosis(state, llm_strong=MagicMock())

        sent_prompt = mock_safe_call.call_args[0][2]
        assert "KIEM_TRA_NHIỀU_EDGE_CASE" in sent_prompt


class TestPlanReviewerFallback:
    @patch("agent.nodes.safe_llm_call")
    def test_missing_plan_and_diagnosis_uses_safe_fallback_prompt(self, mock_safe_call):
        audit = PlanAudit(
            passed=False,
            identified_risks=["rủi ro"],
            audit_feedback="cần bổ sung ngữ cảnh",
            confidence=0.4,
        )
        mock_safe_call.return_value = (
            audit,
            {"model_name": "gpt-test", "node_name": "node_3c_plan_reviewer", "tokens": 5},
        )
        state = AgentState(user_input="x", intent="FIX_BUG")

        result = node_3c_plan_reviewer(state, llm_strong=MagicMock())

        sent_prompt = mock_safe_call.call_args[0][2]
        assert "Không có Plan hoặc Diagnosis nào được tạo trước đó" in sent_prompt
        assert result.plan_review_retries == 1
