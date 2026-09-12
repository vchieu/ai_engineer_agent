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


# 1. KHỞI TẠO VÀ KIỂM TRA MÔI TRƯỜNG (LAZY LOADING CHO UNIT TESTS)
def get_llm_clients():
    """Khởi tạo LLM clients chỉ khi thực sự cần chạy session."""
    api_key = os.getenv("OPENAI_API_KEY")
    INVALID_KEYS = {"", "your_openai_api_key_here", "none", "null"}

    if not api_key or api_key.strip().lower() in INVALID_KEYS:
        raise RuntimeError(
            "❌ OPENAI_API_KEY không hợp lệ hoặc chưa được cấu hình đúng trong file .env."
        )

    llm_strong = ChatOpenAI(model="gpt-4o", temperature=0.0, api_key=api_key)
    llm_cheap = ChatOpenAI(model="gpt-4o-mini", temperature=0.0, api_key=api_key)
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

# Tự quản lý thời gian hoàn tất session để dọn dẹp an toàn
with db_lock:
    with conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS session_completions (
                thread_id TEXT PRIMARY KEY,
                completed_at TEXT NOT NULL
            )
        """)


def _mark_session_completed(thread_id: str):
    """Lưu vết thời điểm session kết thúc để phục vụ cronjob dọn dẹp (Thread-safe)."""
    try:
        with db_lock:
            with conn:
                conn.execute(
                    "INSERT OR REPLACE INTO session_completions (thread_id, completed_at) VALUES (?, datetime('now'))",
                    (thread_id,)
                )
    except Exception as e:
        logger.error(f"[Tracking Error] Không thể đánh dấu hoàn tất cho thread {thread_id}: {e}", exc_info=True)


# 3. QUẢN LÝ DỌN DẸP & VÒNG ĐỜI DATABASE
def delete_thread_data(thread_id: str):
    """Xóa toàn bộ checkpoint history qua API chính thức và dọn bảng tracking (Thread-safe)."""
    try:
        with db_lock:
            checkpointer.delete_thread(thread_id)
            with conn:
                conn.execute("DELETE FROM session_completions WHERE thread_id = ?", (thread_id,))

        logger.info(f"🧹 [Database Cleanup] Đã xóa toàn bộ dữ liệu của thread: {thread_id}")
    except Exception as e:
        logger.error(f"⚠️ [Database Cleanup Error] Không thể xóa thread {thread_id}: {e}", exc_info=True)


def cleanup_old_sessions(days_retention: int = 7):
    """Dọn dẹp các checkpoint cũ dựa trên bảng tracking độc lập (Thread-safe)."""
    try:
        with db_lock:
            with conn:
                rows = conn.execute(
                    "SELECT thread_id FROM session_completions WHERE completed_at < datetime('now', '-' || ? || ' days')",
                    (days_retention,)
                ).fetchall()

            for (tid,) in rows:
                checkpointer.delete_thread(tid)
                with conn:
                    conn.execute("DELETE FROM session_completions WHERE thread_id = ?", (tid,))

            with conn:
                conn.execute("PRAGMA wal_checkpoint(TRUNCATE);")

        logger.info(f"🧹 [Database Maintenance] Đã dọn dẹp {len(rows)} phiên cũ hơn {days_retention} ngày.")
    except Exception as e:
        logger.error(f"⚠️ [Database Maintenance Error]: {e}", exc_info=True)


def close_db_connection():
    """Flush WAL vĩnh viễn vào file chính và đóng connection (dùng khi shutdown app/FastAPI)."""
    global conn
    if conn:
        try:
            with db_lock:
                conn.execute("PRAGMA wal_checkpoint(TRUNCATE);")
                conn.close()
                conn = None
            logger.info("🔒 [Database] Đã flush WAL và đóng kết nối SQLite an toàn.")
        except Exception as e:
            logger.error(f"⚠️ [Database Close Error]: {e}", exc_info=True)


# 4. GRAPH INSTANCE (LAZY LOADED / THREAD-SAFE SINGLETON)
def get_verifier_graph():
    """Khởi tạo verifier_graph dạng singleton (thread-safe)."""
    global _verifier_graph
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
    config = {"configurable": {"thread_id": thread_id}}
    initial_state = AgentState(
        user_input=user_input, max_iteration=3, max_missing_info_retries=3,
        force_human_review=force_human_review,
    )
    graph = get_verifier_graph()

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
            _mark_session_completed(thread_id)

        return res

    except Exception as e:
        logger.error(f"❌ [Fatal Runtime Error in Session {thread_id}]: {str(e)}", exc_info=True)
        if auto_cleanup:
            delete_thread_data(thread_id)
        else:
            _mark_session_completed(thread_id)
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
    config = {"configurable": {"thread_id": thread_id}}
    graph = get_verifier_graph()

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
            _mark_session_completed(thread_id)

        return res

    except Exception as e:
        logger.error(f"❌ [Fatal Runtime Error in Session {thread_id}]: {str(e)}", exc_info=True)
        if auto_cleanup:
            delete_thread_data(thread_id)
        else:
            _mark_session_completed(thread_id)
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
