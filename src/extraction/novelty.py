"""
Novelty score: TF-IDF based, computed externally after extraction.

For each extracted event on asset A at time t:
  1. Retrieve all EventRecords for asset A in [t - 48h, t).
  2. Compute TF-IDF vector of (title + event_type) for the current item vs. the window.
  3. novelty = 1 - max_cosine_similarity(current, window_items).
  4. If the window is empty, novelty = 1.0.

This gives high novelty to the first report of a hack and low novelty to
the fifteenth repetition of the same story.

Point-in-time safety: novelty must only use events with published_at < t
in the lookback window.
"""

from __future__ import annotations

import logging
from datetime import timedelta

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.config import EXTRACTION_LOOKBACK_HOURS

log = logging.getLogger("extraction.novelty")


def compute_novelty(
    event: dict,
    all_events: pd.DataFrame,
    lookback_hours: int = EXTRACTION_LOOKBACK_HOURS,
) -> float:
    """Compute novelty score for one event against a window of prior events.

    Args:
        event: EventRecord dict with keys: asset, published_at, content_hash.
        all_events: DataFrame of all EventRecords (must have asset,
                    published_at, event_type columns).
        lookback_hours: Size of the trailing window for novelty comparison.

    Returns:
        Novelty score in [0.0, 1.0]. 1.0 = completely novel.
    """
    asset = event["asset"]
    event_published = event.get("published_at")

    # Point-in-time anchor: novelty must be computed against events that were
    # knowably published BEFORE this one. extracted_at is the batch wall-clock
    # and is NOT a valid availability anchor (it is post-hoc), so we refuse to
    # fall back to it — an event without published_at cannot be scored safely.
    if event_published is None:
        return 1.0

    if all_events.empty:
        return 1.0

    window_start = event_published - timedelta(hours=lookback_hours)
    mask = (
        (all_events["asset"] == asset)
        & (all_events["published_at"] >= window_start)
        & (all_events["published_at"] < event_published)
        & (all_events["content_hash"] != event["content_hash"])
    )

    window = all_events.loc[mask]

    if window.empty:
        return 1.0

    # Build TF-IDF features
    # Current item text: title + event_type
    current_text = f"{event.get('title', '')} {event['event_type']}"
    window_texts = window.apply(
        lambda r: f"{r.get('title', '')} {r['event_type']}", axis=1
    ).tolist()

    # Add current to the vectorizer fit to avoid zero division
    all_texts = window_texts + [current_text]
    vectorizer = TfidfVectorizer(token_pattern=r'[a-zA-Z0-9_]+')
    tfidf = vectorizer.fit_transform(all_texts)

    # Get the current item's vector (last in the list)
    current_vector = tfidf[-1:]
    window_vectors = tfidf[:-1]

    # Compute cosine similarities
    similarities = cosine_similarity(current_vector, window_vectors).flatten()

    # Novelty = 1 - max similarity
    max_sim = similarities.max() if len(similarities) > 0 else 0.0
    novelty = max(0.0, min(1.0, 1.0 - max_sim))

    return float(novelty)


def compute_novelty_batch(
    events_df: pd.DataFrame,
) -> pd.Series:
    """Compute novelty scores for all events in a DataFrame.

    Returns a Series aligned with events_df index.
    """
    log.info("Computing novelty for %d events...", len(events_df))
    scores = []
    for i, event_row in events_df.iterrows():
        event_dict = event_row.to_dict()
        score = compute_novelty(event_dict, events_df)
        scores.append(score)
    return pd.Series(scores, index=events_df.index, name="novelty")
