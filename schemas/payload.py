import os
from typing import List, Optional, Literal, Dict, Any
from pydantic import BaseModel, Field, model_validator


def _is_safe_filename(filename: str) -> bool:
    if os.path.isabs(filename):
        return False
    # Trên Windows, os.path.isabs() không nhận Unix-style absolute paths như "/etc/passwd"
    if filename.startswith("/") and filename.lstrip("/"):
        return False
    normalized = os.path.normpath(filename)
    parts = normalized.split(os.sep)
    parts_alt = normalized.split("/")
    if ".." in parts or ".." in parts_alt or normalized.startswith(".."):
        return False
    return True


class SandboxResult(BaseModel):
    success: bool
    returncode: int
    stdout: str
    stderr: str
    error_message: Optional[str] = None


class FileChange(BaseModel):
    filename: str = Field(min_length=1, description="Tên file mã nguồn (bao gồm đường dẫn tương đối).")
    content: str = Field(description="Nội dung mã nguồn hoàn chỉnh.")


class HistoryEntry(BaseModel):
    iteration: int = Field(description="Vòng lặp hiện tại.")
    code_proposal: List[FileChange] = Field(description="Danh sách file từ proposal.")
    sandbox_stdout: str = Field(default="", description="Log stdout từ sandbox.")
    sandbox_stderr: str = Field(default="", description="Log stderr từ sandbox.")
    feedback: str = Field(description="Nhận xét từ audit hoặc lỗi hệ thống.")


AllowedFieldKey = Literal["language", "error_log", "source_code", "file_list", "additional_context"]


class MissingFieldRequest(BaseModel):
    field: AllowedFieldKey = Field(description="Tên biến cố định dùng cho Frontend key binding.")
    question: str = Field(description="Câu hỏi hiển thị trực tiếp cho user")
    suggested_options: List[str] = Field(default_factory=list, description="Gợi ý có sẵn để hiển thị dạng Button/Chips")
    allow_freeform: bool = Field(default=True, description="Cho phép user nhập văn bản tự do")


class RouterDecision(BaseModel):
    intent: Literal["MISSING_INFO", "FIX_BUG", "NEW_FEATURE"] = Field(description="Phân loại yêu cầu người dùng")
    reasoning: str = Field(description="Lý do phân loại")
    missing_fields: Optional[List[MissingFieldRequest]] = Field(
        default=None, 
        description="Chi tiết từng thông tin còn thiếu nếu intent là MISSING_INFO"
    )
    complexity_hint: Literal["TRIVIAL", "STANDARD", "COMPLEX"] = Field(
        description=(
            "TRIVIAL: thay đổi rất nhỏ, rủi ro thấp (sửa 1 dòng, đổi text/label, thêm log). "
            "STANDARD: logic rõ ràng, phạm vi vừa phải. "
            "COMPLEX: ảnh hưởng nhiều file, đổi kiến trúc, hoặc đụng khu vực nhạy cảm "
            "(auth, database, payment, security)."
        )
    )


class PlanSchema(BaseModel):
    steps: List[str] = Field(description="Các bước thực hiện")
    target_files: List[str] = Field(description="Danh sách file liên quan")
    acceptance_criteria: str = Field(description="Tiêu chí nghiệm thu rõ ràng")


class DiagnosisSchema(BaseModel):
    root_cause: str = Field(description="Nguyên nhân gốc rễ bug")
    affected_files: List[str] = Field(description="Các file bị ảnh hưởng")
    suggested_approach: str = Field(description="Hướng tiếp cận kỹ thuật đề xuất")
    acceptance_criteria: str = Field(description="Tiêu chí để xác nhận bug đã hết")


class CodeFixProposal(BaseModel):
    explanation: str = Field(description="Mô tả các thay đổi mã nguồn đã thực hiện.")
    language: Literal["python", "javascript"] = Field(default="python", description="Ngôn ngữ lập trình.")
    entrypoint_filename: str = Field(description="Tên file chính dùng để khởi chạy execution.")
    is_test_file: bool = Field(default=False, description="True nếu entrypoint là file chạy Unit Test.")
    files: List[FileChange] = Field(description="Danh sách các file mã nguồn.")

    @model_validator(mode="after")
    def validate_proposal(self) -> "CodeFixProposal":
        filenames = [f.filename for f in self.files]
        if len(filenames) != len(set(filenames)):
            raise ValueError("Validation Error: Danh sách file thay đổi chứa các đường dẫn bị trùng lặp (duplicate filenames).")

        created_filenames = set()

        for f in self.files:
            if not _is_safe_filename(f.filename):
                raise ValueError(
                    f"Validation Error: Phát hiện hành vi Path Traversal nguy hiểm trong tên file '{f.filename}'."
                )
            created_filenames.add(f.filename)

        if not _is_safe_filename(self.entrypoint_filename):
            raise ValueError(f"Validation Error: Entrypoint filename chứa đường dẫn không an toàn '{self.entrypoint_filename}'.")

        if self.entrypoint_filename not in created_filenames:
            raise ValueError(
                f"Validation Error: entrypoint_filename '{self.entrypoint_filename}' không nằm trong danh sách files."
            )

        ext = os.path.splitext(self.entrypoint_filename)[1].lower()
        if self.language == "python" and ext != ".py":
            raise ValueError(f"Validation Error: Language là 'python' nhưng entrypoint '{self.entrypoint_filename}' thiếu đuôi .py")
        elif self.language == "javascript" and ext not in [".js", ".mjs", ".cjs"]:
            raise ValueError(f"Validation Error: Language là 'javascript' nhưng entrypoint '{self.entrypoint_filename}' thiếu đuôi .js/.mjs/.cjs")

        return self


class VerifierAudit(BaseModel):
    passed: bool = Field(description="True nếu mã nguồn vượt qua kiểm thử VÀ đạt tiêu chí chấp nhận")
    audit_feedback: str = Field(description="Nhận xét chi tiết, lý do thất bại hoặc chỉ dẫn sửa đổi")
    confidence: float = Field(description="Độ tin cậy của phán quyết (0.0 đến 1.0)")


class PlanAudit(BaseModel):
    passed: bool = Field(
        description="True nếu kế hoạch đủ chi tiết, khả thi, không có rủi ro nghiêm trọng chưa xử lý."
    )
    identified_risks: List[str] = Field(
        min_length=1,
        description=(
            "Danh sách rủi ro/điểm yếu tiềm ẩn của kế hoạch. BẮT BUỘC có ít nhất 1 mục, "
            "kể cả khi passed=True — để chống confirmation bias (LLM có xu hướng tự khen)."
        ),
    )
    audit_feedback: str = Field(description="Nhận xét chi tiết, hướng dẫn khắc phục cụ thể nếu passed=False.")
    confidence: float = Field(description="Độ tin cậy của phán quyết (0.0 đến 1.0).")


FinalStatusType = Literal["SUCCESS", "FAILED_MAX_ITERATION", "FAILED_MISSING_INFO", "FAILED_PLAN_REJECTED", "ERROR"]


class AgentState(BaseModel):
    user_input: str
    intent: Optional[Literal["MISSING_INFO", "FIX_BUG", "NEW_FEATURE"]] = None
    missing_fields: Optional[List[MissingFieldRequest]] = None
    plan: Optional[PlanSchema] = None
    cleaned_logs: Optional[str] = None
    diagnosis: Optional[DiagnosisSchema] = None
    code_proposal: Optional[CodeFixProposal] = None
    sandbox_result: Optional[SandboxResult] = None
    audit_result: Optional[VerifierAudit] = None
    
    iteration: int = 0
    max_iteration: int = 3
    missing_info_retries: int = 0
    max_missing_info_retries: int = 3
    complexity_hint: Optional[Literal["TRIVIAL", "STANDARD", "COMPLEX"]] = None
    plan_audit_result: Optional[PlanAudit] = None
    plan_review_retries: int = 0
    max_plan_review_retries: int = 2
    plan_review_feedback: Optional[str] = None
    review_strategy: Optional[Literal["skip_review", "llm_review", "human_review"]] = None
    force_human_review: bool = False
    
    final_status: Optional[FinalStatusType] = None
    error_message: Optional[str] = None
    
    history: List[HistoryEntry] = Field(default_factory=list)
    
    total_tokens: int = 0
    token_usage_by_model: Dict[str, int] = Field(default_factory=dict)
    token_usage_by_node: Dict[str, int] = Field(default_factory=dict)
