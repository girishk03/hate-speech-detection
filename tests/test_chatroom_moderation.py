def test_toxic_message_is_detected(combined_module):
    assert combined_module.is_toxic("I hate you") is True


def test_clean_message_is_allowed(combined_module):
    assert combined_module.is_toxic("Thank you for your thoughtful help") is False


def test_empty_message_is_handled(combined_module):
    assert combined_module.is_toxic("") is False
    assert combined_module.make_polite("") == ""
