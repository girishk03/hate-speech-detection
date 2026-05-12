# comment_processor.py

import re
import os
import time
import numpy as np
import pandas as pd
from youtube_comment_downloader import YoutubeCommentDownloader, SORT_BY_POPULAR, SORT_BY_RECENT

# -----------------------------
# Text Cleaning Function
# -----------------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#\w+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# -----------------------------
# Fetch + Clean + Save Comments
# -----------------------------
def fetch_clean_comments(
    video_url,
    max_comments=50,
    save_path="data/fetched_comments.csv",
    max_duration_sec=None,
    sort_by="newest",
    progress_callback=None,
    return_records=False
):
    try:
        downloader = YoutubeCommentDownloader()
        start_time = time.time()
        sort_mode = SORT_BY_RECENT if sort_by == "newest" else SORT_BY_POPULAR

        raw_comments = []
        cleaned_comments = []
        authors = []
        author_urls = []
        comment_ids = []

        # -----------------------------
        # Fetch comments until we have enough cleaned ones
        # -----------------------------
        for comment in downloader.get_comments_from_url(video_url, sort_by=sort_mode):
            if max_duration_sec is not None and (time.time() - start_time) >= max_duration_sec:
                print(f"⏱️ Stopped fetch at {max_duration_sec}s time budget")
                break

            text = comment.get("text", "").strip()

            if text:
                raw_comments.append(text)
                cleaned = clean_text(text)
                
                # Only count non-empty cleaned comments
                if cleaned.strip() != "":
                    cleaned_comments.append(cleaned)
                    authors.append(
                        comment.get("author")
                        or comment.get("username")
                        or comment.get("name")
                        or comment.get("channel")
                        or "Unknown User"
                    )
                    channel_id = comment.get("channel")
                    if channel_id:
                        author_urls.append(f"https://www.youtube.com/channel/{channel_id}")
                    else:
                        author_urls.append(None)
                    comment_ids.append(
                        comment.get("cid")
                        or comment.get("commentId")
                        or comment.get("comment_id")
                        or comment.get("id")
                        or None
                    )

            # Stop when we have enough cleaned comments
            if len(cleaned_comments) >= max_comments:
                break

            if progress_callback and len(cleaned_comments) % 25 == 0:
                progress_callback(len(cleaned_comments), max_comments)

        if progress_callback:
            progress_callback(len(cleaned_comments), max_comments)

        print(f"✅ Fetched {len(raw_comments)} raw comments, {len(cleaned_comments)} usable comments")

        if not cleaned_comments:
            return []

        # Get corresponding raw comments
        raw_comments = raw_comments[:len(cleaned_comments)]

        # -----------------------------
        # Save to CSV
        # -----------------------------
        df = pd.DataFrame({
            "original_comment": raw_comments,
            "clean_text": cleaned_comments,
            "author": authors,
            "author_url": author_urls,
            "comment_id": comment_ids
        })

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df.to_csv(save_path, index=False)

        print(f"💾 Saved cleaned comments to {save_path}")

        if return_records:
            return [
                {
                    "author": author,
                    "author_url": author_url,
                    "comment_id": comment_id,
                    "comment": original,
                    "clean_comment": cleaned,
                }
                for author, author_url, comment_id, original, cleaned in zip(authors, author_urls, comment_ids, raw_comments, cleaned_comments)
            ]

        return cleaned_comments

    except Exception as e:
        print(f"❌ Error: {e}")
        return []
# -----------------------------
# Predict using saved model
# -----------------------------
def predict_from_comments(cleaned_comments, model, vectorizer, save_path="data/predicted_comments.csv"):
    try:
        if not cleaned_comments:
            return []

        # Vectorize
        vec = vectorizer.transform(cleaned_comments)

        # Predict
        preds = model.predict(vec)

        # Save results
        df = pd.DataFrame({
            "clean_text": cleaned_comments,
            "prediction": preds
        })

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df.to_csv(save_path, index=False)

        print(f"💾 Saved predictions to {save_path}")

        # Return results for frontend
        results = []
        for c, p in zip(cleaned_comments, preds):
            results.append({
                "clean_text": c,
                "prediction": p
            })

        return results

    except Exception as e:
        print(f"❌ Prediction Error: {e}")
        return []

# -----------------------------
# Stats Function
# -----------------------------
def calculate_stats(results):
    stats = {"total": len(results), "positive": 0, "negative": 0, "neutral": 0}

    for r in results:
        pred = r["prediction"].lower()
        if pred in stats:
            stats[pred] += 1

    return stats


TOXIC_KEYWORDS = {
    "hate", "kill", "die", "stupid", "idiot", "dumb", "moron", "retard",
    "ugly", "disgusting", "nasty", "racist", "nazi", "terrorist",
    "fuck", "shit", "bitch", "asshole", "bastard", "worthless", "useless"
}


def _softmax(arr):
    arr = np.asarray(arr, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    shifted = arr - np.max(arr, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=1, keepdims=True)


def _find_keywords(text):
    tokens = re.findall(r"[a-zA-Z']+", str(text).lower())
    return sorted(set(t for t in tokens if t in TOXIC_KEYWORDS))

def classify_comments(cleaned_comments, model, vectorizer):
    """
    Takes cleaned comments → returns predictions
    """

    if not cleaned_comments:
        return []

    if isinstance(cleaned_comments[0], dict):
        original_texts = [item.get("comment", "") for item in cleaned_comments]
        texts = [item.get("clean_comment", clean_text(item.get("comment", ""))) for item in cleaned_comments]
        authors = [item.get("author", "Unknown User") for item in cleaned_comments]
        author_urls = [item.get("author_url") for item in cleaned_comments]
        comment_ids = [item.get("comment_id") for item in cleaned_comments]
    else:
        original_texts = cleaned_comments
        texts = cleaned_comments
        authors = ["Unknown User"] * len(cleaned_comments)
        author_urls = [None] * len(cleaned_comments)
        comment_ids = [None] * len(cleaned_comments)

    # Convert text → numerical (TF-IDF)
    X = vectorizer.transform(texts)

    # Predict
    predictions = model.predict(X)
    classes = list(getattr(model, "classes_", []))

    probabilities = None
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X)
    elif hasattr(model, "decision_function"):
        decision_vals = model.decision_function(X)
        probabilities = _softmax(decision_vals)

    # Combine results
    results = []
    for idx, (author, author_url, comment_id, original_comment, pred) in enumerate(zip(authors, author_urls, comment_ids, original_texts, predictions)):
        confidence = 0.0
        if probabilities is not None and len(classes) > 0:
            pred_idx = classes.index(pred) if pred in classes else int(np.argmax(probabilities[idx]))
            confidence = float(probabilities[idx][pred_idx])
        elif hasattr(model, "decision_function"):
            # fallback approximation from decision margin
            margin = model.decision_function(X[idx])
            confidence = float(np.max(_softmax(margin)))

        matched_keywords = _find_keywords(original_comment)
        label = str(pred).lower()
        if label == "negative":
            toxicity_score = min(1.0, 0.6 * confidence + 0.08 * len(matched_keywords))
        elif label == "neutral":
            toxicity_score = min(1.0, 0.2 * confidence + 0.06 * len(matched_keywords))
        else:
            toxicity_score = max(0.0, 0.08 * len(matched_keywords))

        results.append({
            "author": author,
            "author_url": author_url,
            "comment_id": comment_id,
            "comment": original_comment,
            "prediction": pred,
            "confidence": round(confidence, 4),
            "toxicity_score": round(toxicity_score, 4),
            "matched_keywords": matched_keywords
        })

    return results
