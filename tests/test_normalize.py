import pytest
from embcache._normalize import normalize

def test_normalize():
    assert normalize("") == ""
    assert normalize("hello world") == "hello world"
    assert normalize("HELLO WORLD") == "hello world"
    # Unicode NFC
    assert normalize("café") == normalize("cafe\u0301")
    # Punctuation
    assert normalize("hello, world!") == "hello world"
    # Whitespace collapse
    assert normalize("foo   bar\t\nbaz") == "foo bar baz"
    # Mixed unicode punctuation
    assert normalize("smart quote: “hey” — ellipsis…") == "smart quote hey ellipsis"
    # Only punctuation
    assert normalize("!!! ??? ,,,") == ""
    assert normalize("   ") == ""
