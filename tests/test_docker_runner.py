"""
Test cho sandbox/docker_runner.py.

Chia làm 2 nhóm rõ ràng:
- TestSafeJoin: pure function, KHÔNG cần Docker daemon -> luôn chạy.
- TestExecuteInDockerSandbox: đánh dấu @pytest.mark.integration vì cần
  Docker daemon đang chạy thật (và cần image python:3.11-slim đã pull sẵn).
  Chạy nhóm này bằng:  pytest -m integration
  Bỏ qua nhóm này bằng: pytest -m "not integration"  (mặc định trong CI/pre-commit)
"""
import os
import pytest

from sandbox.docker_runner import _safe_join


class TestSafeJoin:
    def test_normal_relative_path_resolves_inside_base(self, tmp_path):
        result = _safe_join(str(tmp_path), "app.py")
        assert result == os.path.join(str(tmp_path), "app.py")

    def test_nested_relative_path_resolves_inside_base(self, tmp_path):
        result = _safe_join(str(tmp_path), "sub/dir/app.py")
        assert result.startswith(str(tmp_path))

    @pytest.mark.parametrize("filename", [
        "/etc/passwd",
        "../escape.py",
        "sub/../../escape.py",
    ])
    def test_path_traversal_rejected(self, tmp_path, filename):
        with pytest.raises(ValueError):
            _safe_join(str(tmp_path), filename)


@pytest.mark.integration
class TestExecuteInDockerSandbox:
    """
    Yêu cầu: Docker daemon đang chạy + đã `docker pull python:3.11-slim`.
    Đây là những test quan trọng nhất để đảm bảo các rào chắn bảo mật
    (network_disabled, read_only, cap_drop, timeout) THỰC SỰ có hiệu lực,
    không chỉ đúng trên giấy.
    """

    def test_successful_python_execution_returns_stdout(self):
        from sandbox.docker_runner import execute_in_docker_sandbox

        result = execute_in_docker_sandbox(
            files={"main.py": "print('hello_sandbox')"},
            entrypoint_cmd=["python", "main.py"],
            language="python",
            timeout_seconds=10,
        )
        assert result.success is True
        assert result.returncode == 0
        assert "hello_sandbox" in result.stdout

    def test_nonzero_exit_code_reported_as_failure(self):
        from sandbox.docker_runner import execute_in_docker_sandbox

        result = execute_in_docker_sandbox(
            files={"main.py": "raise SystemExit(1)"},
            entrypoint_cmd=["python", "main.py"],
            language="python",
            timeout_seconds=10,
        )
        assert result.success is False
        assert result.returncode == 1

    def test_network_is_actually_disabled(self):
        """Kiểm chứng network_disabled=True có tác dụng thật, không chỉ là config trên giấy."""
        from sandbox.docker_runner import execute_in_docker_sandbox

        code = (
            "import socket\n"
            "try:\n"
            "    socket.create_connection(('8.8.8.8', 53), timeout=3)\n"
            "    print('NETWORK_REACHABLE')\n"
            "except OSError:\n"
            "    print('NETWORK_BLOCKED')\n"
        )
        result = execute_in_docker_sandbox(
            files={"main.py": code},
            entrypoint_cmd=["python", "main.py"],
            language="python",
            timeout_seconds=10,
        )
        assert "NETWORK_BLOCKED" in result.stdout

    def test_filesystem_is_read_only_outside_app(self):
        """Kiểm chứng read_only=True: không ghi được ra ngoài /app và /tmp."""
        from sandbox.docker_runner import execute_in_docker_sandbox

        code = (
            "try:\n"
            "    open('/root/should_fail.txt', 'w').write('x')\n"
            "    print('WRITE_SUCCEEDED')\n"
            "except OSError:\n"
            "    print('WRITE_BLOCKED')\n"
        )
        result = execute_in_docker_sandbox(
            files={"main.py": code},
            entrypoint_cmd=["python", "main.py"],
            language="python",
            timeout_seconds=10,
        )
        assert "WRITE_BLOCKED" in result.stdout

    def test_long_running_code_is_killed_on_timeout(self):
        """Kiểm chứng cơ chế timeout + container.kill() không làm crash toàn bộ hàm."""
        from sandbox.docker_runner import execute_in_docker_sandbox

        result = execute_in_docker_sandbox(
            files={"main.py": "import time\ntime.sleep(60)"},
            entrypoint_cmd=["python", "main.py"],
            language="python",
            timeout_seconds=3,
        )
        # Quan trọng: hàm phải trả về SandboxResult(success=False,...),
        # KHÔNG được raise exception ra ngoài (đây chính là bug đã fix ở round trước).
        assert result.success is False
