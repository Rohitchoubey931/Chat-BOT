import pytest

from app.loader import chunk_text


def test_empty_text_returns_no_chunks():
    assert chunk_text("", 100, 10) == []


def test_short_text_is_one_chunk():
    assert chunk_text("hello world", 100, 10) == ["hello world"]


def test_chunks_respect_size_limit_and_overlap():
    text = "word " * 500
    chunks = chunk_text(text, 200, 50)
    assert len(chunks) > 1
    assert all(len(c) <= 200 for c in chunks)


def test_overlap_must_be_smaller_than_size():
    with pytest.raises(ValueError):
        chunk_text("some text here", 10, 10)
