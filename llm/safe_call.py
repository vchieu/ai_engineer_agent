from typing import Any, Type, TypeVar, Tuple, List
from pydantic import BaseModel
from langchain_core.messages import BaseMessage, HumanMessage

T = TypeVar("T", bound=BaseModel)


def safe_llm_call(
    llm_client: Any,
    schema_class: Type[T],
    prompt_or_messages: Any,
    model_name: str = "unknown_model",
    node_name: str = "unknown_node",
    max_retries: int = 3
) -> Tuple[T, dict]:
    structured_llm = llm_client.with_structured_output(schema_class, include_raw=True)

    if isinstance(prompt_or_messages, str):
        messages: List[BaseMessage] = [HumanMessage(content=prompt_or_messages)]
    elif isinstance(prompt_or_messages, list):
        messages = list(prompt_or_messages)
    else:
        messages = [HumanMessage(content=str(prompt_or_messages))]

    last_error = None

    for attempt in range(1, max_retries + 1):
        try:
            res = structured_llm.invoke(messages)
            raw_msg = res.get("raw")
            parsed_obj = res.get("parsed")
            parsing_error = res.get("parsing_error")

            if parsing_error:
                raise ValueError(f"Lỗi Format JSON/Schema: {parsing_error}")

            if parsed_obj is None:
                raise ValueError("LLM không trả về dữ liệu cấu trúc hợp lệ.")

            tokens = 0
            if raw_msg:
                if hasattr(raw_msg, "usage_metadata") and raw_msg.usage_metadata:
                    tokens = raw_msg.usage_metadata.get("total_tokens", 0)
                elif hasattr(raw_msg, "response_metadata") and "token_usage" in raw_msg.response_metadata:
                    tokens = raw_msg.response_metadata["token_usage"].get("total_tokens", 0)

            token_stats = {
                "model_name": model_name,
                "node_name": node_name,
                "tokens": tokens
            }
            return parsed_obj, token_stats

        except Exception as e:
            last_error = e
            print(f"⚠️ [safe_llm_call] Retry {attempt}/{max_retries} tại node '{node_name}' do lỗi: {str(e)[:150]}")
            
            error_feedback = (
                f"\n\n[HỆ THỐNG]: Lần phản hồi trước đó của bạn tạo ra lỗi Validation: {str(e)}."
                f"\nHãy kiểm tra kỹ định dạng schema của class '{schema_class.__name__}' và tạo lại câu trả lời hợp lệ."
            )
            messages.append(HumanMessage(content=error_feedback))

    raise RuntimeError(f"Lỗi gọi LLM tại node '{node_name}' sau {max_retries} lần thử. Chi tiết: {last_error}")
