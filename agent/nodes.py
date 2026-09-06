import re
from typing import Any, Literal, Union, Dict
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.types import interrupt, Command

from schemas.payload import (
    AgentState, RouterDecision, PlanSchema, DiagnosisSchema,
    CodeFixProposal, VerifierAudit, HistoryEntry
)
from llm.safe_call import safe_llm_call
from sandbox.docker_runner import execute_in_docker_sandbox


def _get_model_name(llm_client: Any) -> str:
    return getattr(llm_client, "model_name", getattr(llm_client, "model", "unknown_model"))


def _update_tokens(state: AgentState, token_stats: dict):
    model = token_stats.get("model_name", "unknown_model")
    node = token_stats.get("node_name", "unknown_node")
    tokens = token_stats.get("tokens", 0)

    state.total_tokens += tokens
    state.token_usage_by_model[model] = state.token_usage_by_model.get(model, 0) + tokens
    state.token_usage_by_node[node] = state.token_usage_by_node.get(node, 0) + tokens


def _truncate_for_prompt(text: str, max_chars: int = 2500) -> str:
    if not text or len(text) <= max_chars:
        return text or ""
    half = max_chars // 2
    return f"{text[:half]}\n\n... [Đã cắt bớt {len(text) - max_chars} ký tự ở giữa] ...\n\n{text[-half:]}"


def node_1_router(state: AgentState, llm_strong: Any) -> AgentState:
    print("\n---> [Node 1] Router Agent đang phân loại yêu cầu...")
    prompt = f"""Bạn là Router Agent trong hệ thống AI Software Engineering.
Hãy phân tích yêu cầu người dùng và quyết định luồng:
1. 'MISSING_INFO': Thiếu file, thiếu mô tả lỗi hoặc không đủ ngữ cảnh. Hãy chọn field chuẩn trong: ['language', 'error_log', 'source_code', 'file_list', 'additional_context'].
2. 'FIX_BUG': Sửa bug/lỗi đang xảy ra cùng đoạn code/log lỗi.
3. 'NEW_FEATURE': Phát triển hoặc viết mới tính năng/module.

User Input:
{state.user_input}
"""
    model_name = _get_model_name(llm_strong)
    decision, token_stats = safe_llm_call(llm_strong, RouterDecision, prompt, model_name=model_name, node_name="node_1_router")
    _update_tokens(state, token_stats)

    state.intent = decision.intent
    state.missing_fields = decision.missing_fields
    print(f"     Kết quả Router: {decision.intent} | Lý do: {decision.reasoning}")
    return state


def node_missing_info_interrupt(state: AgentState) -> Command[Literal["node_1_router"]]:
    interrupt_payload = {
        "status": "AWAITING_USER_INPUT",
        "missing_info_retries": state.missing_info_retries,
        "requests": [req.model_dump() for req in (state.missing_fields or [])]
    }

    user_response: Union[str, Dict[str, str]] = interrupt(interrupt_payload)

    print("\n---> [Interrupt Resumed] Nhận phản hồi bổ sung từ người dùng...")

    if isinstance(user_response, dict):
        formatted_answers = "\n".join([f"- {k}: {v}" for k, v in user_response.items()])
    else:
        formatted_answers = str(user_response)

    updated_input = f"{state.user_input}\n\n[Thông tin bổ sung từ Người dùng]:\n{formatted_answers}"

    return Command(
        goto="node_1_router",
        update={
            "user_input": updated_input,
            "missing_fields": None,
            "missing_info_retries": state.missing_info_retries + 1
        }
    )


def node_terminal_missing_info(state: AgentState) -> AgentState:
    print("\n---> [Terminal] Đạt giới hạn số lần hỏi bổ sung thông tin.")
    state.final_status = "FAILED_MISSING_INFO"
    state.error_message = f"Hệ thống không thể thu thập đủ thông tin sau {state.max_missing_info_retries} lần thử."
    return state


def node_2a_planner(state: AgentState, llm_strong: Any) -> AgentState:
    print("\n---> [Node 2A] Planner Agent đang lập kế hoạch...")
    prompt = f"Bạn là Lead Architect. Hãy lập kế hoạch phát triển tính năng:\n{state.user_input}"
    model_name = _get_model_name(llm_strong)
    plan, token_stats = safe_llm_call(llm_strong, PlanSchema, prompt, model_name=model_name, node_name="node_2a_planner")
    _update_tokens(state, token_stats)

    state.plan = plan
    return state


def node_2b_cleaner(state: AgentState) -> AgentState:
    print("\n---> [Node 2B] Cleaner đang làm sạch log...")
    raw_text = state.user_input
    cleaned = re.sub(r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?", "", raw_text)
    cleaned = re.sub(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])", "", cleaned)
    lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
    state.cleaned_logs = "\n".join(lines)
    return state


def node_3b_diagnosis(state: AgentState, llm_strong: Any) -> AgentState:
    print("\n---> [Node 3B] Diagnosis Agent đang chẩn đoán...")
    prompt = f"Bạn là Senior Debugging Expert. Hãy chẩn đoán nguyên nhân bug:\n{state.cleaned_logs}"
    model_name = _get_model_name(llm_strong)
    diagnosis, token_stats = safe_llm_call(llm_strong, DiagnosisSchema, prompt, model_name=model_name, node_name="node_3b_diagnosis")
    _update_tokens(state, token_stats)

    state.diagnosis = diagnosis
    return state


def node_4_coder(state: AgentState, llm_cheap: Any) -> AgentState:
    print(f"\n---> [Node 4] Coder Agent đang lập trình (Vòng #{state.iteration + 1})...")

    context_str = ""
    if state.intent == "NEW_FEATURE" and state.plan:
        context_str = f"KẾ HOẠCH:\nSteps: {state.plan.steps}\nFiles: {state.plan.target_files}\nAcceptance Criteria: {state.plan.acceptance_criteria}"
    elif state.intent == "FIX_BUG" and state.diagnosis:
        context_str = f"CHẨN ĐOÁN:\nRoot Cause: {state.diagnosis.root_cause}\nSuggested Approach: {state.diagnosis.suggested_approach}\nAcceptance Criteria: {state.diagnosis.acceptance_criteria}"

    sys_prompt = """Bạn là Coder Agent chuyên nghiệp. Hãy viết mã nguồn hoàn chỉnh.
QUY TẮC BẮT BUỘC:
1. Khi xử lý FIX_BUG: Hãy ƯU TIÊN viết 1 file test tái hiện/kiểm tra lỗi làm entrypoint (is_test_file=True).
2. Nếu 'is_test_file' là True và language='python': Bắt buộc các class test kế thừa từ 'unittest.TestCase'.
3. 'entrypoint_filename' phải nằm trong danh sách 'files' và có đuôi tệp khớp với 'language' (.py cho python; .js cho javascript)."""

    messages = [
        SystemMessage(content=sys_prompt),
        HumanMessage(content=f"Yêu cầu:\n{state.user_input}\n\nNgữ cảnh:\n{context_str}")
    ]

    if state.iteration > 0 and state.history:
        last_hist = state.history[-1]
        formatted_prev_code = ""
        for file_item in last_hist.code_proposal:
            formatted_prev_code += f"### File: {file_item.filename}\n```\n{file_item.content}\n```\n\n"

        messages.append(AIMessage(content=f"Mã nguồn vòng trước:\n{formatted_prev_code}"))
        messages.append(HumanMessage(content=f"Lỗi kiểm thử trước đó:\n{last_hist.feedback}\nHãy sửa lại code."))

    model_name = _get_model_name(llm_cheap)
    code_proposal, token_stats = safe_llm_call(
        llm_cheap, CodeFixProposal, messages, model_name=model_name, node_name="node_4_coder"
    )
    _update_tokens(state, token_stats)

    state.code_proposal = code_proposal
    return state


def node_5_verifier(state: AgentState, llm_strong: Any) -> AgentState:
    print(f"\n---> [Node 5] Verifier Agent đang audit & test (Vòng #{state.iteration + 1})...")

    prop = state.code_proposal
    files_dict = {f.filename: f.content for f in prop.files}
    entry_file = prop.entrypoint_filename

    if prop.language == "python":
        entry_cmd = ["python", "-m", "unittest", entry_file] if prop.is_test_file else ["python", entry_file]
    else:
        entry_cmd = ["node", "--test", entry_file] if prop.is_test_file else ["node", entry_file]

    sb_result = execute_in_docker_sandbox(
        files=files_dict,
        entrypoint_cmd=entry_cmd,
        language=prop.language,
        timeout_seconds=30
    )

    if prop.is_test_file and sb_result.success:
        combined_out = (sb_result.stdout + "\n" + sb_result.stderr).lower()
        if prop.language == "python" and "ran 0 tests" in combined_out:
            sb_result.success = False
            sb_result.stderr += "\n[Runner Validation Error]: Unittest không tìm thấy test case nào (Ran 0 tests)."
        elif prop.language == "javascript" and ("# tests 0" in combined_out or "pass 0" in combined_out):
            sb_result.success = False
            sb_result.stderr += "\n[Runner Validation Error]: Node --test không tìm thấy test case nào (# tests 0)."

    state.sandbox_result = sb_result

    acceptance_criteria = state.plan.acceptance_criteria if state.plan else (state.diagnosis.acceptance_criteria if state.diagnosis else "")
    full_code_text = "".join([f"=== File: {f.filename} ===\n{f.content}\n\n" for f in prop.files])

    prompt = f"""Bạn là Verifier Audit Agent. Hãy đánh giá mã nguồn và kết quả thực thi sandbox.

Tiêu chí chấp nhận (Acceptance Criteria):
{acceptance_criteria}

Mã nguồn hoàn chỉnh:
{full_code_text}

Kết quả Sandbox Execution:
- Success Flag: {sb_result.success}
- Error/Warning Message: {sb_result.error_message or 'None'}
- Standard Output (Truncated):
{_truncate_for_prompt(sb_result.stdout)}
- Standard Error (Truncated):
{_truncate_for_prompt(sb_result.stderr)}

Yêu cầu:
1. Nếu Sandbox Success=False, bắt buộc passed=False.
2. Đặt passed=True chỉ khi Sandbox Success=True VÀ Mã nguồn thỏa mãn Tiêu chí chấp nhận.
"""
    model_name = _get_model_name(llm_strong)
    audit, token_stats = safe_llm_call(llm_strong, VerifierAudit, prompt, model_name=model_name, node_name="node_5_verifier")
    _update_tokens(state, token_stats)

    state.audit_result = audit

    state.history.append(HistoryEntry(
        iteration=state.iteration,
        code_proposal=state.code_proposal.files if state.code_proposal else [],
        sandbox_stdout=sb_result.stdout,
        sandbox_stderr=sb_result.stderr,
        feedback=audit.audit_feedback if audit else (sb_result.stderr or "Unknown error")
    ))
    state.iteration += 1

    if sb_result.success and audit.passed:
        state.final_status = "SUCCESS"
    elif state.iteration >= state.max_iteration:
        state.final_status = "FAILED_MAX_ITERATION"
        state.error_message = f"Thất bại sau khi thử tối đa {state.max_iteration} vòng lập trình."

    return state
