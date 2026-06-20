def test_model_returns_valid_prediction(comment_processor, model_and_vectorizer):
    model, vectorizer = model_and_vectorizer
    result = comment_processor.classify_comments(["thank you for the helpful video"], model, vectorizer)

    assert result[0]["prediction"] in {"positive", "negative", "neutral"}


def test_model_result_includes_confidence(comment_processor, model_and_vectorizer):
    model, vectorizer = model_and_vectorizer
    result = comment_processor.classify_comments(["this is terrible"], model, vectorizer)

    assert 0.0 <= result[0]["confidence"] <= 1.0


def test_model_supports_batch_prediction(comment_processor, model_and_vectorizer):
    model, vectorizer = model_and_vectorizer
    result = comment_processor.classify_comments(
        ["excellent explanation", "this was disappointing"],
        model,
        vectorizer,
    )

    assert len(result) == 2
    assert all(item["prediction"] in {"positive", "negative", "neutral"} for item in result)
