import os
import tempfile
from typing import Dict, List, Optional
import docker

from schemas.payload import SandboxResult

_DOCKER_CLIENT_SINGLETON: Optional[docker.DockerClient] = None


def _get_docker_client() -> docker.DockerClient:
    global _DOCKER_CLIENT_SINGLETON
    if _DOCKER_CLIENT_SINGLETON is None:
        _DOCKER_CLIENT_SINGLETON = docker.from_env()
    return _DOCKER_CLIENT_SINGLETON


def _safe_join(base_dir: str, filename: str) -> str:
    if os.path.isabs(filename):
        raise ValueError(f"Filename không hợp lệ (Path Traversal): {filename}")
    # Trên Windows, os.path.isabs() không nhận Unix-style absolute paths như "/etc/passwd"
    if filename.startswith("/") and filename.lstrip("/"):
        raise ValueError(f"Filename không hợp lệ (Path Traversal): {filename}")
    if ".." in filename.split(os.sep) or ".." in filename.split("/"):
        raise ValueError(f"Filename không hợp lệ (Path Traversal): {filename}")
    
    full_path = os.path.normpath(os.path.join(base_dir, filename))
    abs_base = os.path.abspath(base_dir)
    
    if not (full_path == abs_base or full_path.startswith(abs_base + os.sep)):
        raise ValueError(f"Filename thoát khỏi sandbox dir: {filename}")
    return full_path


def execute_in_docker_sandbox(
    files: Dict[str, str],
    entrypoint_cmd: List[str],
    language: str = "python",
    timeout_seconds: int = 30,
    mem_limit: str = "512m"
) -> SandboxResult:
    image_map = {
        "python": "python:3.11-slim",
        "javascript": "node:20-slim"
    }
    image = image_map.get(language, "python:3.11-slim")

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

            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)
            os.chmod(file_path, 0o666)

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
                stdout = container.logs(stdout=True, stderr=False).decode("utf-8", errors="replace")
                stderr = container.logs(stdout=False, stderr=True).decode("utf-8", errors="replace")

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
                return SandboxResult(
                    success=False,
                    returncode=-1,
                    stdout="",
                    stderr="",
                    error_message=f"Execution Timeout ({timeout_seconds}s) or Exception: {str(e)}"
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
