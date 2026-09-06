"""
Unit test cho schemas/payload.py

Đây là lớp test QUAN TRỌNG NHẤT trong toàn bộ dự án: mọi validation an ninh
(chống Path Traversal, chống duplicate filename, khớp extension/language)
đều nằm ở đây. Nếu ai đó (kể cả AI) vô tình sửa lỏng các rule này, test ở
đây phải là nơi đầu tiên báo đỏ.

Không cần Docker, không cần LLM, không cần mạng -> chạy trong mili-giây.
"""
import pytest
from schemas.payload import _is_safe_filename, CodeFixProposal, FileChange


class TestIsSafeFilename:
    @pytest.mark.parametrize("filename", [
        "main.py",
        "src/utils.py",
        "a/b/c/d.js",
        "test_file.py",
    ])
    def test_safe_filenames_accepted(self, filename):
        assert _is_safe_filename(filename) is True

    @pytest.mark.parametrize("filename", [
        "/etc/passwd",           # absolute path
        "../secret.py",          # traversal ra ngoài trực tiếp
        "a/../../b.py",          # traversal sau khi normalize ("../b.py")
        "..",                    # chỉ toàn dấu chấm
    ])
    def test_unsafe_filenames_rejected(self, filename):
        assert _is_safe_filename(filename) is False


class TestCodeFixProposalValidation:
    """Test model_validator 'validate_proposal' trong CodeFixProposal."""

    def _base_kwargs(self, **overrides):
        kwargs = dict(
            explanation="fix bug",
            language="python",
            entrypoint_filename="main.py",
            is_test_file=False,
            files=[FileChange(filename="main.py", content="print(1)")],
        )
        kwargs.update(overrides)
        return kwargs

    def test_valid_proposal_passes(self):
        proposal = CodeFixProposal(**self._base_kwargs())
        assert proposal.entrypoint_filename == "main.py"

    def test_path_traversal_in_entrypoint_rejected(self):
        with pytest.raises(ValueError, match="Path Traversal"):
            CodeFixProposal(**self._base_kwargs(
                entrypoint_filename="../evil.py",
                files=[FileChange(filename="../evil.py", content="pass")],
            ))

    def test_path_traversal_in_other_file_rejected(self):
        with pytest.raises(ValueError, match="Path Traversal"):
            CodeFixProposal(**self._base_kwargs(
                files=[
                    FileChange(filename="main.py", content="print(1)"),
                    FileChange(filename="../../etc/passwd", content="pwned"),
                ],
            ))

    def test_entrypoint_not_in_files_rejected(self):
        with pytest.raises(ValueError, match="không nằm trong danh sách files"):
            CodeFixProposal(**self._base_kwargs(entrypoint_filename="other.py"))

    def test_duplicate_filenames_rejected(self):
        with pytest.raises(ValueError, match="trùng lặp"):
            CodeFixProposal(**self._base_kwargs(files=[
                FileChange(filename="main.py", content="a"),
                FileChange(filename="main.py", content="b"),
            ]))

    def test_python_extension_mismatch_rejected(self):
        with pytest.raises(ValueError, match="thiếu đuôi"):
            CodeFixProposal(**self._base_kwargs(
                entrypoint_filename="main.js",
                files=[FileChange(filename="main.js", content="pass")],
            ))

    def test_javascript_extension_variants_accepted(self):
        for ext in ["index.js", "index.mjs", "index.cjs"]:
            proposal = CodeFixProposal(**self._base_kwargs(
                language="javascript",
                entrypoint_filename=ext,
                files=[FileChange(filename=ext, content="console.log(1)")],
            ))
            assert proposal.language == "javascript"

    def test_javascript_wrong_extension_rejected(self):
        with pytest.raises(ValueError, match="thiếu đuôi"):
            CodeFixProposal(**self._base_kwargs(
                language="javascript",
                entrypoint_filename="index.py",
                files=[FileChange(filename="index.py", content="pass")],
            ))
