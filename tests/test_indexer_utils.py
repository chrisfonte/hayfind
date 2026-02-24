from pathlib import Path

from hayfind.indexer import chunk_text, is_binary


def test_chunk_text_non_empty() -> None:
    text = "one\n\n" + ("x" * 1300) + "\n\nthree"
    chunks = chunk_text(text, max_chars=500)
    assert chunks
    assert all(len(c) <= 500 for c in chunks)


def test_is_binary(tmp_path: Path) -> None:
    txt = tmp_path / "a.txt"
    txt.write_text("hello\nworld", encoding="utf-8")
    assert not is_binary(txt)

    bin_file = tmp_path / "a.bin"
    bin_file.write_bytes(b"abc\x00def")
    assert is_binary(bin_file)
