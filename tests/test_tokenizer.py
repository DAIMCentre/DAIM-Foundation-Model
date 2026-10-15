import pytest

from daim_fm import ByteTokenizer


def test_byte_tokenizer_round_trips_unicode() -> None:
    tokenizer = ByteTokenizer()
    text = "Hull, café, 数据"
    assert tokenizer.decode(tokenizer.encode(text)) == text


def test_byte_tokenizer_rejects_ids_outside_vocabulary() -> None:
    with pytest.raises(ValueError, match="between 0 and 255"):
        ByteTokenizer().decode([256])
