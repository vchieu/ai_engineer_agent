from typing import Any, Literal, Optional, Dict
from langgraph.graph import StateGraph, START, END

from schemas.payload import AgentState
from agent.nodes import (
    node_1_router,
    node_missing_info_interrupt,
    node_terminal_missing_info,
    node_2a_planner,
    node_2b_cleaner,
    node_3b_diagnosis,
    node_4_coder,
    node_5_verifier,
    node_plan_dispatch,
    node_3c_plan_reviewer,
    node_plan_human_interrupt,
    node_terminal_plan_rejected,
)


def route_after_router(
    state: AgentState,
) -> Literal[
    "node_missing_info_interrupt", "node_terminal_missing_info",
    "node_2b_cleaner", "node_2a_planner", "node_4_coder",
]:
    if state.intent == "MISSING_INFO":
        if state.missing_info_retries >= state.max_missing_info_retries:
            return "node_terminal_missing_info"
        return "node_missing_info_interrupt"
    elif state.intent == "FIX_BUG":
        return "node_2b_cleaner"
    elif state.intent == "NEW_FEATURE" and state.complexity_hint == "TRIVIAL":
        return "node_4_coder"
    else:
        return "node_2a_planner"


def route_after_plan_dispatch(
    state: AgentState,
) -> Literal["node_4_coder", "node_3c_plan_reviewer", "node_plan_human_interrupt"]:
    mapping = {
        "skip_review": "node_4_coder",
        "llm_review": "node_3c_plan_reviewer",
        "human_review": "node_plan_human_interrupt",
    }
    return mapping[state.review_strategy]


def route_after_plan_review(
    state: AgentState,
) -> Literal["node_4_coder", "node_2a_planner", "node_3b_diagnosis", "node_terminal_plan_rejected"]:
    if state.plan_audit_result and state.plan_audit_result.passed:
        return "node_4_coder"
    if state.plan_review_retries >= state.max_plan_review_retries:
        return "node_terminal_plan_rejected"
    return "node_2a_planner" if state.intent == "NEW_FEATURE" else "node_3b_diagnosis"


def route_after_verifier(state: AgentState) -> Literal["node_4_coder", "__end__"]:
    # Trước đây chỉ END khi final_status in ["SUCCESS", "FAILED_MAX_ITERATION"] —
    # về mặt lý thuyết, không node nào trong nodes.py hiện set final_status="ERROR"
    # (giá trị "ERROR" trong FinalStatusType chỉ được main.py dùng ở tầng response
    # dict khi graph.stream() ném exception RA NGOÀI graph, không phải giá trị
    # được ghi vào state rồi feed lại cho route này). Tuy nhiên nếu sau này có
    # thêm node nào set final_status="ERROR" hoặc 1 trạng thái terminal khác, để
    # route dựa theo whitelist 2 giá trị cụ thể là 1 bẫy dễ quên cập nhật.
    # Đảo lại logic: BẤT KỲ final_status nào đã được set (khác None) đều là
    # terminal -> END; chỉ None (chưa quyết định) mới tiếp tục vòng lặp Coder.
    if state.final_status is None:
        return "node_4_coder"
    return END


def extract_interrupt_data(state_snapshot: Any) -> Optional[Dict[str, Any]]:
    # `task.interrupts[0].value` phụ thuộc cấu trúc nội bộ của LangGraph — có thể
    # đổi giữa các phiên bản. Bọc try/except để lỗi cấu trúc trả về None (coi như
    # "không có interrupt") thay vì làm crash toàn bộ start/resume_agent_session.
    if not state_snapshot:
        return None
    try:
        tasks = getattr(state_snapshot, "tasks", None)
        if not tasks:
            return None
        for task in tasks:
            interrupts = getattr(task, "interrupts", None)
            if interrupts:
                return interrupts[0].value
    except (AttributeError, IndexError, TypeError):
        return None
    return None


def create_agent_graph(llm_strong: Any, llm_cheap: Any, checkpointer: Optional[Any] = None):
    builder = StateGraph(AgentState)

    builder.add_node("node_1_router", lambda state: node_1_router(state, llm_strong))
    builder.add_node("node_missing_info_interrupt", node_missing_info_interrupt)
    builder.add_node("node_terminal_missing_info", node_terminal_missing_info)
    builder.add_node("node_2a_planner", lambda state: node_2a_planner(state, llm_strong))
    builder.add_node("node_2b_cleaner", node_2b_cleaner)
    builder.add_node("node_3b_diagnosis", lambda state: node_3b_diagnosis(state, llm_strong))
    builder.add_node("node_plan_dispatch", node_plan_dispatch)
    builder.add_node("node_3c_plan_reviewer", lambda state: node_3c_plan_reviewer(state, llm_strong))
    builder.add_node("node_plan_human_interrupt", node_plan_human_interrupt)
    builder.add_node("node_terminal_plan_rejected", node_terminal_plan_rejected)
    builder.add_node("node_4_coder", lambda state: node_4_coder(state, llm_cheap))
    builder.add_node("node_5_verifier", lambda state: node_5_verifier(state, llm_strong))

    builder.add_edge(START, "node_1_router")
    builder.add_conditional_edges("node_1_router", route_after_router)
    builder.add_edge("node_terminal_missing_info", END)
    builder.add_edge("node_2a_planner", "node_plan_dispatch")
    builder.add_edge("node_2b_cleaner", "node_3b_diagnosis")
    builder.add_edge("node_3b_diagnosis", "node_plan_dispatch")
    builder.add_conditional_edges("node_plan_dispatch", route_after_plan_dispatch)
    builder.add_conditional_edges("node_3c_plan_reviewer", route_after_plan_review)
    builder.add_edge("node_terminal_plan_rejected", END)
    builder.add_edge("node_4_coder", "node_5_verifier")
    builder.add_conditional_edges("node_5_verifier", route_after_verifier)

    return builder.compile(checkpointer=checkpointer)
