import logging
import os
import sqlite3
import threading
from typing import Dict, Any, Union, Optional
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

from schemas.payload import AgentState
from agent.graph import create_agent_graph, extract_interrupt_data

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("ai_engineer_agent")

# Khởi tạo RLock toàn cục (reentrant) cho SQLite Connection và Graph singleton đảm bảo thread-safety an toàn
db_lock = threading.RLock()
_graph_lock = threading.Lock()
_verifier_graph = None
# Cờ đánh dấu close_db_connection() đã chạy: checkpointer giữ tham chiếu tới
# connection object GỐC (không phải biến `conn` toàn cục), nên set `conn = None`
# không khiến checkpointer "quên" connection cũ — nó vẫn cầm 1 connection đã bị
# .close(), và lần gọi tiếp theo sẽ ném lỗi sqlite3 khó hiểu. Cờ này giúp báo lỗi
# rõ ràng ngay từ đầu thay vì để traceback nội bộ SQLite lộ ra ngoài.
_db_closed = False


# 1. KHỞI TẠO VÀ KIỂM TRA MÔI TRƯỜNG (LAZY LOADING CHO UNIT TESTS)
def get_llm_clients():
    """Khởi tạo LLM clients chỉ khi thực sự cần chạy session."""
    api_key = os.getenv("OPENAI_API_KEY")
    INVALID_KEYS = {"", "your_openai_api_key_here", "none", "null"}

    if not api_key or api_key.strip().lower() in INVALID_KEYS:
        raise RuntimeError(
            "❌ OPENAI_API_KEY không hợp lệ hoặc chưa được cấu hình đúng trong file .env."
        )

    # Cho phép override model qua .env thay vì hardcode — cần thiết khi muốn thử
    # model khác (vd gpt-4.1, gpt-4o-2024-...) mà không phải sửa code.
    strong_model = os.getenv("LLM_STRONG_MODEL", "gpt-4o")
    cheap_model = os.getenv("LLM_CHEAP_MODEL", "gpt-4o-mini")
    llm_strong = ChatOpenAI(model=strong_model, temperature=0.0, api_key=api_key)
    llm_cheap = ChatOpenAI(model=cheap_model, temperature=0.0, api_key=api_key)
    return llm_strong, llm_cheap


# 2. KHỞI TẠO SQLITE CHECKPOINTER TỐI ƯU LOCAL (WAL MODE)
DB_PATH = os.getenv("DB_PATH", os.path.join(os.path.dirname(__file__), "agent_sessions.db"))

conn = sqlite3.connect(DB_PATH, check_same_thread=False)
with db_lock:
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA busy_timeout=5000;")
    conn.execute("PRAGMA synchronous=NORMAL;")

checkpointer = SqliteSaver(conn)

assert hasattr(checkpointer, "delete_thread"), (
    "Phiên bản langgraph-checkpoint-sqlite hiện tại không hỗ trợ delete_thread() — "
    "vui lòng nâng cấp package trước khi chạy (ví dụ: pip install -U 'langgraph-checkpoint>=2.0.25')."
)

# Tự quản lý HOẠT ĐỘNG GẦN NHẤT của mỗi thread (không chỉ lúc "hoàn tất") để
# phục vụ cronjob dọn dẹp an toàn.
# LƯU Ý FIX: bảng cũ tên "session_completions" chỉ được ghi khi session HOÀN TẤT
# (is_completed=True) — 1 session dừng ở interrupt (chờ người dùng bổ sung info /
# duyệt plan) và KHÔNG BAO GIỜ quay lại resume sẽ không bao giờ xuất hiện trong
# bảng này, nên cleanup_old_sessions() không bao giờ dọn được nó -> checkpoint DB
# phình vô hạn cho các session bị người dùng bỏ dở. Đổi sang track "last_activity"
# ghi ở MỌI lần gọi start/resume (bất kể kết quả completed hay interrupted).
with db_lock:
    with conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS session_activity (
                thread_id TEXT PRIMARY KEY,
                last_activity_at TEXT NOT NULL
            )
        """)


def _touch_session_activity(thread_id: str):
    """Cập nhật thời điểm hoạt động gần nhất của thread (Thread-safe). Gọi ở MỌI
    lần start/resume, kể cả khi session dừng lại ở interrupt hoặc lỗi — để
    cleanup_old_sessions() có thể dọn được cả các session bị bỏ dở, không chỉ
    session đã hoàn tất."""
    try:
        with db_lock:
            with conn:
                conn.execute(
                    "INSERT OR REPLACE INTO session_activity (thread_id, last_activity_at) VALUES (?, datetime('now'))",
                    (thread_id,)
                )
    except Exception as e:
        logger.error(f"[Tracking Error] Không thể ghi nhận hoạt động cho thread {thread_id}: {e}", exc_info=True)


# 3. QUẢN LÝ DỌN DẸP & VÒNG ĐỜI DATABASE
def delete_thread_data(thread_id: str):
    """Xóa toàn bộ checkpoint history qua API chính thức và dọn bảng tracking (Thread-safe)."""
    try:
        with db_lock:
            checkpointer.delete_thread(thread_id)
            with conn:
                conn.execute("DELETE FROM session_activity WHERE thread_id = ?", (thread_id,))

        logger.info(f"🧹 [Database Cleanup] Đã xóa toàn bộ dữ liệu của thread: {thread_id}")
    except Exception as e:
        logger.error(f"⚠️ [Database Cleanup Error] Không thể xóa thread {thread_id}: {e}", exc_info=True)


def cleanup_old_sessions(days_retention: int = 7):
    """Dọn dẹp các checkpoint cũ dựa trên `last_activity_at` (Thread-safe).
    Áp dụng cho MỌI thread không hoạt động quá `days_retention` ngày, kể cả các
    session đang dừng ở interrupt (chờ user) mà không bao giờ được resume —
    được coi là bị bỏ dở và dọn như session đã hoàn tất."""
    try:
        with db_lock:
            with conn:
                rows = conn.execute(
                    "SELECT thread_id FROM session_activity WHERE last_activity_at < datetime('now', '-' || ? || ' days')",
                    (days_retention,)
                ).fetchall()

            for (tid,) in rows:
                checkpointer.delete_thread(tid)
                with conn:
                    conn.execute("DELETE FROM session_activity WHERE thread_id = ?", (tid,))

            with conn:
                conn.execute("PRAGMA wal_checkpoint(TRUNCATE);")

        logger.info(f"🧹 [Database Maintenance] Đã dọn dẹp {len(rows)} phiên cũ hơn {days_retention} ngày.")
    except Exception as e:
        logger.error(f"⚠️ [Database Maintenance Error]: {e}", exc_info=True)


def close_db_connection():
    """Flush WAL vĩnh viễn vào file chính và đóng connection (dùng khi shutdown app/FastAPI)."""
    global conn, _db_closed
    if conn:
        try:
            with db_lock:
                conn.execute("PRAGMA wal_checkpoint(TRUNCATE);")
                conn.close()
                conn = None
                _db_closed = True
            logger.info("🔒 [Database] Đã flush WAL và đóng kết nối SQLite an toàn.")
        except Exception as e:
            logger.error(f"⚠️ [Database Close Error]: {e}", exc_info=True)


# 4. GRAPH INSTANCE (LAZY LOADED / THREAD-SAFE SINGLETON)
def get_verifier_graph():
    """Khởi tạo verifier_graph dạng singleton (thread-safe)."""
    global _verifier_graph
    if _db_closed:
        raise RuntimeError(
            "❌ close_db_connection() đã được gọi — checkpointer đang giữ 1 connection SQLite "
            "đã đóng, không thể mở session mới. Cần khởi động lại process để tạo connection mới."
        )
    if _verifier_graph is None:
        with _graph_lock:
            if _verifier_graph is None:
                llm_strong, llm_cheap = get_llm_clients()
                _verifier_graph = create_agent_graph(llm_strong, llm_cheap, checkpointer=checkpointer)
    return _verifier_graph


def __getattr__(name: str):
    if name == "verifier_graph":
        return get_verifier_graph()
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


# 5. API SERVICES BỌC AN TOÀN TOÀN DIỆN
def start_agent_session(
    thread_id: str,
    user_input: str,
    auto_cleanup: bool = False,
    force_human_review: bool = False,
) -> Dict[str, Any]:
    """
    Khởi chạy phiên làm việc mới.
    - auto_cleanup=True: Tự động xóa history khỏi DB ngay khi session kết thúc (tránh phình đĩa).
    """
    # recursion_limit mặc định của LangGraph (25) có thể chạm giới hạn khi có
    # nhiều vòng human/llm review + iteration coder/verifier cộng dồn trong 1
    # session — nới ra và cho phép cấu hình qua env thay vì hardcode ngầm.
    recursion_limit = int(os.getenv("GRAPH_RECURSION_LIMIT", "50"))
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": recursion_limit}
    initial_state = AgentState(
        user_input=user_input, max_iteration=3, max_missing_info_retries=3,
        force_human_review=force_human_review,
    )
    graph = get_verifier_graph()
    # Ghi nhận hoạt động NGAY LẬP TỨC, trước khi biết kết quả completed/interrupted/
    # error — xem giải thích ở định nghĩa _touch_session_activity().
    _touch_session_activity(thread_id)

    try:
        # TRADE-OFF: Khóa bao gồm toàn bộ graph.stream() (LLM calls + Docker sandbox),
        # không chỉ riêng phần ghi checkpoint. Nghĩa là các session chạy TUẦN TỰ,
        # không song song thật sự. Xem AGENTS.md mục "API Services & Persistence" để
        # biết hướng nâng cấp lên concurrency thật (AsyncSqliteSaver / per-request connection / Postgres).
        with db_lock:
            for _ in graph.stream(initial_state, config=config, stream_mode="values"):
                pass

            state_snapshot = graph.get_state(config)
            
        interrupt_info = extract_interrupt_data(state_snapshot)

        if interrupt_info:
            return {
                "thread_id": thread_id,
                "is_completed": False,
                "final_status": None,
                "interrupt_data": interrupt_info
            }

        final_st: AgentState = state_snapshot.values
        res = {
            "thread_id": thread_id,
            "is_completed": True,
            "final_status": final_st.final_status or "ERROR",
            "final_state": final_st.model_dump()
        }

        if auto_cleanup:
            delete_thread_data(thread_id)
        else:
            # Ghi nhận thời gian để hàm cleanup_old_sessions() dọn dẹp sau này
            _touch_session_activity(thread_id)

        return res

    except Exception as e:
        logger.error(f"❌ [Fatal Runtime Error in Session {thread_id}]: {str(e)}", exc_info=True)
        if auto_cleanup:
            delete_thread_data(thread_id)
        else:
            _touch_session_activity(thread_id)
        return {
            "thread_id": thread_id,
            "is_completed": True,
            "final_status": "ERROR",
            "error": str(e),
            "final_state": None
        }


def resume_agent_session(
    thread_id: str,
    user_answer: Union[str, Dict[str, str]],
    auto_cleanup: bool = False
) -> Dict[str, Any]:
    """
    Tiếp tục phiên bị tạm ngắt.
    - auto_cleanup=True: Tự động xóa history khỏi DB ngay khi session kết thúc.
    """
    recursion_limit = int(os.getenv("GRAPH_RECURSION_LIMIT", "50"))
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": recursion_limit}
    graph = get_verifier_graph()
    # Ghi nhận hoạt động ngay khi resume được gọi — quan trọng cho các session
    # trước đó dừng ở interrupt: nếu user resume rồi lại bị interrupt tiếp (vd
    # human_review nhiều vòng), mỗi lần resume phải "touch" lại để không bị
    # cleanup_old_sessions() dọn nhầm 1 session vẫn đang hoạt động dở dang.
    _touch_session_activity(thread_id)

    try:
        # TRADE-OFF: Khóa bao gồm toàn bộ graph.stream() (LLM calls + Docker sandbox),
        # không chỉ riêng phần ghi checkpoint. Nghĩa là các session chạy TUẦN TỰ,
        # không song song thật sự. Xem AGENTS.md mục "API Services & Persistence" để
        # biết hướng nâng cấp lên concurrency thật (AsyncSqliteSaver / per-request connection / Postgres).
        with db_lock:
            for _ in graph.stream(Command(resume=user_answer), config=config, stream_mode="values"):
                pass

            state_snapshot = graph.get_state(config)

        interrupt_info = extract_interrupt_data(state_snapshot)

        if interrupt_info:
            return {
                "thread_id": thread_id,
                "is_completed": False,
                "final_status": None,
                "interrupt_data": interrupt_info
            }

        final_st: AgentState = state_snapshot.values
        res = {
            "thread_id": thread_id,
            "is_completed": True,
            "final_status": final_st.final_status or "ERROR",
            "final_state": final_st.model_dump()
        }

        if auto_cleanup:
            delete_thread_data(thread_id)
        else:
            # Ghi nhận thời gian để hàm cleanup_old_sessions() dọn dẹp sau này
            _touch_session_activity(thread_id)

        return res

    except Exception as e:
        logger.error(f"❌ [Fatal Runtime Error in Session {thread_id}]: {str(e)}", exc_info=True)
        if auto_cleanup:
            delete_thread_data(thread_id)
        else:
            _touch_session_activity(thread_id)
        return {
            "thread_id": thread_id,
            "is_completed": True,
            "final_status": "ERROR",
            "error": str(e),
            "final_state": None
        }


if __name__ == "__main__":
    SESSION_ID = "session_local_persist_002"

    prompt_thieu = "Sửa giúp tôi hàm calculate_average bị crash khi tính toán."
    print("=== TEST START SESSION ===")
    res1 = start_agent_session(thread_id=SESSION_ID, user_input=prompt_thieu, auto_cleanup=False)
    print("Response Status:", res1["is_completed"], "| Interrupt Data:", res1.get("interrupt_data"))

    if not res1["is_completed"]:
        structured_reply = {
            "error_log": "ZeroDivisionError: division by zero",
            "source_code": "def calculate_average(nums):\n    return sum(nums) / len(nums)"
        }
        print("\n=== TEST RESUME SESSION (Auto Cleanup = True) ===")
        res2 = resume_agent_session(thread_id=SESSION_ID, user_answer=structured_reply, auto_cleanup=True)
        print("Final Status Payload:", res2["final_status"])

    # Đóng DB an toàn
    close_db_connection()
