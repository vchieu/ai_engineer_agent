import os
import re
import tempfile
from typing import Dict, List, Optional
import docker

from schemas.payload import SandboxResult

_DOCKER_CLIENT_SINGLETON: Optional[docker.DockerClient] = None

# Xem giải thích tương tự trong schemas/payload.py::_is_safe_filename — giữ nguyên
# logic ở đây vì đây là lớp phòng thủ thứ 2 (defense-in-depth), độc lập với
# validation ở tầng Pydantic.
_WINDOWS_DRIVE_RE = re.compile(r"^[A-Za-z]:")

# Phải khớp (hoặc nhỏ hơn) max_length=200_000 của SandboxResult.stdout/stderr ở
# schemas/payload.py. QUAN TRỌNG: việc truncate phải xảy ra ở ĐÂY, TRƯỚC khi
# construct SandboxResult — nếu để pydantic tự raise ValidationError vì vượt
# max_length, exception đó sẽ bị `except Exception` bên dưới (dùng chung cho
# timeout) nuốt mất và báo nhầm thành "Docker Wait/Exec Exception", che mất
# việc code thực ra đã chạy xong (chỉ là in ra quá nhiều).
_MAX_LOG_CHARS = 200_000


def _cap_log(text: str) -> str:
    if len(text) <= _MAX_LOG_CHARS:
        return text
    return text[:_MAX_LOG_CHARS] + f"\n...[TRUNCATED - vượt quá {_MAX_LOG_CHARS} ký tự]"


def _get_docker_client() -> docker.DockerClient:
    global _DOCKER_CLIENT_SINGLETON
    if _DOCKER_CLIENT_SINGLETON is None:
        _DOCKER_CLIENT_SINGLETON = docker.from_env()
    return _DOCKER_CLIENT_SINGLETON


def _safe_join(base_dir: str, filename: str) -> str:
    if _WINDOWS_DRIVE_RE.match(filename):
        raise ValueError(f"Filename không hợp lệ (Windows Drive Letter): {filename}")
    if filename.startswith(("/", "\\")):
        raise ValueError(f"Filename không hợp lệ (Absolute/UNC Path): {filename}")
    if os.path.isabs(filename):
        raise ValueError(f"Filename không hợp lệ (Path Traversal): {filename}")
    normalized = os.path.normpath(filename)
    if (
        ".." in normalized.split(os.sep)
        or ".." in normalized.replace("\\", "/").split("/")
        or normalized.startswith("..")
    ):
        raise ValueError(f"Filename không hợp lệ (Path Traversal): {filename}")

    full_path = os.path.normpath(os.path.join(base_dir, filename))
    abs_base = os.path.abspath(base_dir)

    # LƯU Ý: trước đây `full_path == abs_base` được coi là HỢP LỆ — nghĩa là
    # filename="." hoặc "" (normpath("") == ".") sẽ "pass" validation, rồi sau đó
    # open(file_path, "w") ném IsADirectoryError vì file_path chính là thư mục
    # base_dir, KHÔNG phải 1 file. Giờ bắt buộc full_path phải nằm THỰC SỰ bên
    # trong abs_base (không được trùng chính nó).
    if not full_path.startswith(abs_base + os.sep):
        raise ValueError(f"Filename không hợp lệ — trỏ tới base dir hoặc thoát khỏi sandbox: {filename}")
    return full_path


def execute_in_docker_sandbox(
    files: Dict[str, str],
    entrypoint_cmd: List[str],
    language: str = "python",
    timeout_seconds: int = 30,
    mem_limit: str = "512m"
) -> SandboxResult:
    # Image tag KHÔNG pin digest theo mặc định (reproducibility/supply-chain risk —
    # tag "slim" có thể trỏ tới digest khác theo thời gian). Cho phép override qua
    # env để pin dạng "python:3.11-slim@sha256:<digest>" khi đã resolve được digest
    # thật (cần mạng/registry access, không thể hardcode 1 giá trị chính xác ở đây).
    image_map = {
        "python": os.getenv("SANDBOX_PYTHON_IMAGE", "python:3.11-slim"),
        "javascript": os.getenv("SANDBOX_NODE_IMAGE", "node:20-slim")
    }
    image = image_map.get(language, image_map["python"])

    try:
        client = _get_docker_client()
    except Exception as e:
        return SandboxResult(
            success=False,
            returncode=-1,
            stdout="",
            stderr="",
            error_message=f"Docker Client Initialization Error: {str(e)}"
        )

    with tempfile.TemporaryDirectory() as tmp_dir:
        os.chmod(tmp_dir, 0o777)

        for filename, content in files.items():
            try:
                file_path = _safe_join(tmp_dir, filename)
            except ValueError as ve:
                return SandboxResult(
                    success=False,
                    returncode=-1,
                    stdout="",
                    stderr="",
                    error_message=f"Security Violation: {str(ve)}"
                )

            # Trước đây makedirs/open/chmod nằm NGOÀI try/except — bất kỳ OSError nào
            # (quyền, tên file lạ, đĩa đầy...) sẽ ném exception thẳng ra khỏi hàm thay
            # vì trả về SandboxResult(success=False) an toàn như phần còn lại của hàm.
            try:
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(content)
                os.chmod(file_path, 0o666)
            except OSError as ose:
                return SandboxResult(
                    success=False,
                    returncode=-1,
                    stdout="",
                    stderr="",
                    error_message=f"File Write Error cho '{filename}': {str(ose)}"
                )

        try:
            container = client.containers.run(
                image=image,
                command=entrypoint_cmd,
                volumes={tmp_dir: {"bind": "/app", "mode": "rw"}},
                working_dir="/app",
                detach=True,
                mem_limit=mem_limit,
                pids_limit=100,
                user="1000:1000",
                cap_drop=["ALL"],
                security_opt=["no-new-privileges:true"],
                read_only=True,
                tmpfs={'/tmp': 'rw,size=32m,noexec'},
                network_disabled=True,
                nano_cpus=1000000000
            )

            try:
                result = container.wait(timeout=timeout_seconds)
                returncode = result.get("StatusCode", -1)
                stdout = _cap_log(container.logs(stdout=True, stderr=False).decode("utf-8", errors="replace"))
                stderr = _cap_log(container.logs(stdout=False, stderr=True).decode("utf-8", errors="replace"))

                return SandboxResult(
                    success=(returncode == 0),
                    returncode=returncode,
                    stdout=stdout,
                    stderr=stderr,
                    error_message=None if returncode == 0 else f"Exited with code {returncode}"
                )
            except Exception as e:
                try:
                    container.kill()
                except Exception:
                    pass
                # container.wait(timeout=...) hết hạn phía client (requests.exceptions.ReadTimeout)
                # không đồng nghĩa process bên trong container đã dừng — đó là lý do container.kill()
                # ở trên luôn được gọi vô điều kiện trong except này (không chỉ khi đúng là timeout).
                # Phân biệt rõ 2 trường hợp chỉ để error_message không gây hiểu lầm cho log/debugging.
                is_timeout = type(e).__name__ in ("ReadTimeout", "ConnectionTimeout", "ConnectTimeout")
                if is_timeout:
                    msg = f"Execution Timeout sau {timeout_seconds}s — container đã bị kill() ở trên."
                else:
                    msg = f"Docker Wait/Exec Exception (không phải timeout): {type(e).__name__}: {str(e)}"
                return SandboxResult(
                    success=False,
                    returncode=-1,
                    stdout="",
                    stderr="",
                    error_message=msg
                )
            finally:
                try:
                    container.remove(force=True)
                except Exception:
                    pass

        except Exception as e:
            return SandboxResult(
                success=False,
                returncode=-1,
                stdout="",
                stderr="",
                error_message=f"Docker Run Exception: {str(e)}"
            )
