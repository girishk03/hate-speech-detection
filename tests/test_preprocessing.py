def test_clean_text_converts_to_lowercase(comment_processor):
    assert comment_processor.clean_text("HELLO World") == "hello world"


def test_clean_text_removes_punctuation(comment_processor):
    assert comment_processor.clean_text("Hello, world!!! #Topic") == "hello world"


def test_clean_text_handles_empty_input(comment_processor):
    assert comment_processor.clean_text("") == ""
