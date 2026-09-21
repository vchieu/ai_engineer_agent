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
    SandboxResult, CodeFixProposal, FileChange, VerifierAudit,
)
from agent.nodes import (
    node_2b_cleaner, node_1_router, node_terminal_missing_info,
    node_2a_planner, node_3b_diagnosis, node_plan_dispatch,
    node_3c_plan_reviewer, node_plan_human_interrupt, node_5_verifier,
    _redact_secrets,
)


class TestRedactSecrets:
    """Regression test cho việc redact secret trước khi lưu HistoryEntry vào
    checkpoint DB (không redact ở prompt Verifier cùng vòng lặp)."""

    def test_redacts_password_assignment(self):
        assert "hunter2" not in _redact_secrets("Error: password=hunter2 invalid")

    def test_redacts_bearer_token(self):
        out = _redact_secrets("Authorization: Bearer eyJhbGciOiJIUzI1NiJ9.abc")
        assert "eyJhbGciOiJIUzI1NiJ9" not in out

    def test_redacts_aws_access_key_pattern(self):
        out = _redact_secrets("AWS key AKIAABCDEFGHIJKLMNOP leaked")
        assert "AKIAABCDEFGHIJKLMNOP" not in out

    def test_does_not_touch_normal_output(self):
        text = "ZeroDivisionError: division by zero"
        assert _redact_secrets(text) == text

    def test_empty_string_passthrough(self):
        assert _redact_secrets("") == ""


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

    def test_touching_critical_file_via_full_relative_path_triggers_human_review(self):
        """Regression test cho bug: trước đây CRITICAL_FILES so khớp full path
        thay vì basename, nên 'schemas/payload.py' (đường dẫn thật của project)
        không khớp với 'payload.py' trong CRITICAL_FILES và lọt qua gate."""
        state = self._diagnosis_state(affected_files=["schemas/payload.py"])
        result = node_plan_dispatch(state)
        assert result.review_strategy == "human_review"

    def test_no_files_identified_triggers_llm_review_not_skip(self):
        """Regression test: diagnosis tồn tại nhưng không tự sinh trạng thái
        affected_files=[] được nữa (schema có min_length=1), nhưng path phòng thủ
        khi cả plan lẫn diagnosis đều vắng mặt (files=[]) vẫn phải không skip_review."""
        state = AgentState(user_input="sửa gì đó", intent="FIX_BUG")
        result = node_plan_dispatch(state)
        assert result.review_strategy == "llm_review"

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


class TestNodePlanHumanInterrupt:
    """
    Regression test cho bug: node_plan_human_interrupt trước đây KHÔNG có giới
    hạn số lần người dùng từ chối (khác với llm_review vốn có
    max_plan_review_retries) -> có thể loop Planner/Diagnosis vô hạn.
    Mock trực tiếp `agent.nodes.interrupt` (không cần chạy graph/checkpointer
    thật) để mô phỏng phản hồi "reject" liên tục từ người dùng.
    """

    @patch("agent.nodes.interrupt")
    def test_approve_routes_to_coder(self, mock_interrupt):
        mock_interrupt.return_value = {"decision": "approve"}
        state = AgentState(user_input="x", intent="FIX_BUG")

        result = node_plan_human_interrupt(state)

        assert result.goto == "node_4_coder"

    @patch("agent.nodes.interrupt")
    def test_single_reject_under_limit_loops_back_and_increments_counter(self, mock_interrupt):
        mock_interrupt.return_value = {"decision": "reject", "feedback": "chưa ổn"}
        state = AgentState(
            user_input="x", intent="FIX_BUG",
            human_review_retries=0, max_human_review_retries=3,
        )

        result = node_plan_human_interrupt(state)

        assert result.goto == "node_3b_diagnosis"
        assert result.update["human_review_retries"] == 1
        assert result.update["plan_review_feedback"] == "chưa ổn"

    @patch("agent.nodes.interrupt")
    def test_reject_at_limit_routes_to_terminal_instead_of_looping_forever(self, mock_interrupt):
        mock_interrupt.return_value = {"decision": "reject", "feedback": "vẫn chưa ổn"}
        state = AgentState(
            user_input="x", intent="FIX_BUG",
            human_review_retries=2, max_human_review_retries=3,
        )

        result = node_plan_human_interrupt(state)

        assert result.goto == "node_terminal_plan_rejected"
        assert result.update["human_review_retries"] == 3

    @patch("agent.nodes.interrupt")
    def test_reject_routes_to_planner_for_new_feature(self, mock_interrupt):
        mock_interrupt.return_value = {"decision": "reject", "feedback": "no"}
        state = AgentState(
            user_input="x", intent="NEW_FEATURE",
            human_review_retries=0, max_human_review_retries=3,
        )

        result = node_plan_human_interrupt(state)

        assert result.goto == "node_2a_planner"


class TestNodeTerminalPlanRejected:
    def test_message_reflects_both_llm_and_human_retry_counters(self):
        from agent.nodes import node_terminal_plan_rejected
        state = AgentState(
            user_input="x", plan_review_retries=2, max_plan_review_retries=2,
            human_review_retries=1, max_human_review_retries=3,
        )

        result = node_terminal_plan_rejected(state)

        assert result.final_status == "FAILED_PLAN_REJECTED"
        assert "2/2" in result.error_message
        assert "1/3" in result.error_message


class TestNodeVerifierDeterministicOverride:
    """Regression test cho R7: audit.passed phải bị ép về False bằng CODE khi
    sandbox thực tế fail, không phụ thuộc việc LLM có tuân thủ prompt hay không."""

    def _proposal(self):
        return CodeFixProposal(
            explanation="fix", language="python", entrypoint_filename="main.py",
            is_test_file=False, files=[FileChange(filename="main.py", content="print(1)")],
        )

    @patch("agent.nodes.safe_llm_call")
    @patch("agent.nodes.execute_in_docker_sandbox")
    def test_forces_passed_false_when_sandbox_fails_even_if_llm_says_true(self, mock_sandbox, mock_safe_call):
        mock_sandbox.return_value = SandboxResult(
            success=False, returncode=1, stdout="", stderr="boom", error_message="Exited with code 1"
        )
        # LLM "sai" (bỏ qua instruction trong prompt) vẫn trả passed=True dù sandbox fail.
        mock_safe_call.return_value = (
            VerifierAudit(passed=True, audit_feedback="trông ổn", confidence=0.9),
            {"model_name": "gpt-test", "node_name": "node_5_verifier", "tokens": 5},
        )
        state = AgentState(user_input="x", intent="FIX_BUG", code_proposal=self._proposal())

        result = node_5_verifier(state, llm_strong=MagicMock())

        assert result.audit_result.passed is False
        assert result.final_status != "SUCCESS"

    @patch("agent.nodes.safe_llm_call")
    @patch("agent.nodes.execute_in_docker_sandbox")
    def test_keeps_passed_true_when_sandbox_succeeds_and_llm_agrees(self, mock_sandbox, mock_safe_call):
        mock_sandbox.return_value = SandboxResult(
            success=True, returncode=0, stdout="ok", stderr="", error_message=None
        )
        mock_safe_call.return_value = (
            VerifierAudit(passed=True, audit_feedback="đạt", confidence=0.9),
            {"model_name": "gpt-test", "node_name": "node_5_verifier", "tokens": 5},
        )
        state = AgentState(user_input="x", intent="FIX_BUG", code_proposal=self._proposal())

        result = node_5_verifier(state, llm_strong=MagicMock())

        assert result.audit_result.passed is True
        assert result.final_status == "SUCCESS"
