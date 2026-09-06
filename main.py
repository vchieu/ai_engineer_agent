import os
import sqlite3
from typing import Dict, Any, Union, Optional
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

from schemas.payload import AgentState
from agent.graph import create_agent_graph, extract_interrupt_data

load_dotenv()

# 1. KHỞI TẠO LLM CLIENTS
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("❌ Missing OPENAI_API_KEY environment variable. Vui lòng cấu hình file .env trước khi chạy.")

llm_strong = ChatOpenAI(model="gpt-4o", temperature=0.0, api_key=api_key)
llm_cheap = ChatOpenAI(model="gpt-4o-mini", temperature=0.0, api_key=api_key)


# 2. KHỞI TẠO SQLITE CHECKPOINTER TỐI ƯU LOCAL (WAL MODE)
DB_PATH = os.getenv("DB_PATH", os.path.join(os.path.dirname(__file__), "agent_sessions.db"))

conn = sqlite3.connect(DB_PATH, check_same_thread=False)
conn.execute("PRAGMA journal_mode=WAL;")
conn.execute("PRAGMA busy_timeout=5000;")
conn.execute("PRAGMA synchronous=NORMAL;")

checkpointer = SqliteSaver(conn)


# 3. QUẢN LÝ DỌN DẸP & VÒNG ĐỜI DATABASE
def delete_thread_data(thread_id: str):
    """Xóa toàn bộ checkpoint history của 1 thread hoàn tất để giải phóng dung lượng."""
    try:
        with conn:
            conn.execute("DELETE FROM checkpoints WHERE thread_id = ?", (thread_id,))
            conn.execute("DELETE FROM checkpoint_blobs WHERE thread_id = ?", (thread_id,))
            conn.execute("DELETE FROM checkpoint_writes WHERE thread_id = ?", (thread_id,))
        print(f"🧹 [Database Cleanup] Đã xóa checkpoint history của thread: {thread_id}")
    except Exception as e:
        print(f"⚠️ [Database Cleanup Error] Không thể xóa thread {thread_id}: {e}")


def cleanup_old_sessions(days_retention: int = 7):
    """
    Dọn dẹp các checkpoint cũ hơn N ngày và thực hiện TRUNCATE WAL để gom dung lượng.
    Phù hợp chạy định kỳ/cronjob khi muốn giữ lại log vài ngày để debug.
    """
    try:
        with conn:
            conn.execute(
                "DELETE FROM checkpoints WHERE datetime(timestamp) < datetime('now', '-' || ? || ' days')",
                (days_retention,)
            )
            conn.execute("PRAGMA wal_checkpoint(TRUNCATE);")
        print(f"🧹 [Database Maintenance] Đã dọn dẹp các phiên cũ hơn {days_retention} ngày.")
    except Exception as e:
        print(f"⚠️ [Database Maintenance Error]: {e}")


def close_db_connection():
    """Flush WAL vĩnh viễn vào file chính và đóng connection (dùng khi shutdown app/FastAPI)."""
    global conn
    if conn:
        try:
            conn.execute("PRAGMA wal_checkpoint(TRUNCATE);")
            conn.close()
            print("🔒 [Database] Đã flush WAL và đóng kết nối SQLite an toàn.")
        except Exception as e:
            print(f"⚠️ [Database Close Error]: {e}")


# 4. GRAPH INSTANCE
verifier_graph = create_agent_graph(llm_strong, llm_cheap, checkpointer=checkpointer)


# 5. API SERVICES BỌC AN TOÀN TOÀN DIỆN
def start_agent_session(
    thread_id: str, 
    user_input: str, 
    auto_cleanup: bool = False
) -> Dict[str, Any]:
    """
    Khởi chạy phiên làm việc mới.
    - auto_cleanup=True: Tự động xóa history khỏi DB ngay khi session kết thúc (tránh phình đĩa).
    """
    config = {"configurable": {"thread_id": thread_id}}
    initial_state = AgentState(user_input=user_input, max_iteration=3, max_missing_info_retries=3)

    try:
        for _ in verifier_graph.stream(initial_state, config=config, stream_mode="values"):
            pass

        state_snapshot = verifier_graph.get_state(config)
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

        return res

    except Exception as e:
        print(f"❌ [Fatal Runtime Error in Session {thread_id}]: {str(e)}")
        if auto_cleanup:
            delete_thread_data(thread_id)
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

    try:
        for _ in verifier_graph.stream(Command(resume=user_answer), config=config, stream_mode="values"):
            pass

        state_snapshot = verifier_graph.get_state(config)
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

        return res

    except Exception as e:
        print(f"❌ [Fatal Runtime Error in Session {thread_id}]: {str(e)}")
        if auto_cleanup:
            delete_thread_data(thread_id)
        return {
            "thread_id": thread_id,
            "is_completed": True,
            "final_status": "ERROR",
            "error": str(e),
            "final_state": None
        }


if __name__ == "__main__":
    if not os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY") == "your_openai_api_key_here":
        print("⚠️ OPENAI_API_KEY chưa được cấu hình. Vui lòng thêm key vào file .env.")
    else:
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
