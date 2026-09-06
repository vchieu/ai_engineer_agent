"""
Test cho các hàm routing thuần (pure function: State -> str) trong agent/graph.py.

Đây là logic điều khiển luồng quan trọng nhất của cả hệ thống (quyết định
lặp lại, dừng, hay đi hỏi thêm thông tin) nhưng lại rẻ nhất để test vì
không cần LLM, không cần Docker — chỉ cần dựng AgentState với các field
liên quan rồi assert kết quả routing.
"""
from langgraph.graph import END
from schemas.payload import AgentState
from agent.graph import route_after_router, route_after_verifier


class TestRouteAfterRouter:
    def test_missing_info_goes_to_interrupt_when_retries_under_limit(self):
        state = AgentState(
            user_input="x", intent="MISSING_INFO",
            missing_info_retries=0, max_missing_info_retries=3,
        )
        assert route_after_router(state) == "node_missing_info_interrupt"

    def test_missing_info_goes_to_interrupt_at_second_to_last_retry(self):
        state = AgentState(
            user_input="x", intent="MISSING_INFO",
            missing_info_retries=2, max_missing_info_retries=3,
        )
        assert route_after_router(state) == "node_missing_info_interrupt"

    def test_missing_info_goes_to_terminal_when_retry_limit_reached(self):
        state = AgentState(
            user_input="x", intent="MISSING_INFO",
            missing_info_retries=3, max_missing_info_retries=3,
        )
        assert route_after_router(state) == "node_terminal_missing_info"

    def test_missing_info_goes_to_terminal_when_retry_limit_exceeded(self):
        """Phòng trường hợp retries > max do lỗi logic ở đâu đó khác cộng dồn quá tay."""
        state = AgentState(
            user_input="x", intent="MISSING_INFO",
            missing_info_retries=5, max_missing_info_retries=3,
        )
        assert route_after_router(state) == "node_terminal_missing_info"

    def test_fix_bug_routes_to_cleaner(self):
        state = AgentState(user_input="x", intent="FIX_BUG")
        assert route_after_router(state) == "node_2b_cleaner"

    def test_new_feature_routes_to_planner(self):
        state = AgentState(user_input="x", intent="NEW_FEATURE")
        assert route_after_router(state) == "node_2a_planner"


class TestRouteAfterVerifier:
    def test_success_ends_graph(self):
        state = AgentState(user_input="x", final_status="SUCCESS")
        assert route_after_verifier(state) == END

    def test_failed_max_iteration_ends_graph(self):
        state = AgentState(user_input="x", final_status="FAILED_MAX_ITERATION")
        assert route_after_verifier(state) == END

    def test_no_final_status_loops_back_to_coder(self):
        state = AgentState(user_input="x", final_status=None)
        assert route_after_verifier(state) == "node_4_coder"

    def test_error_status_loops_back_to_coder(self):
        """
        LƯU Ý: route_after_verifier hiện tại chỉ dừng graph cho SUCCESS/FAILED_MAX_ITERATION.
        final_status="ERROR" (nếu từng được set ở node_5_verifier) sẽ bị coi là "chưa xong"
        và quay lại node_4_coder. Test này ghi lại hành vi HIỆN TẠI để nếu ai đổi logic
        thì phải sửa test một cách có chủ đích, không phải vô tình.
        """
        state = AgentState(user_input="x", final_status="ERROR")
        assert route_after_verifier(state) == "node_4_coder"
