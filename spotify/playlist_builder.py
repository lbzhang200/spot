from __future__ import annotations

import pandas as pd
import spotipy

from spotify_client import (
    get_recommendations,
    create_playlist,
    add_tracks_to_playlist,
)
from analyzer import seed_tracks_for_mood, FEATURE_COLS


def build_and_create_playlist(
    sp: spotipy.Spotify,
    user_id: str,
    df: pd.DataFrame,
    mood: str,
    centroid: dict,
    playlist_name: str,
    num_tracks: int,
    exclude_ids: set[str],
) -> tuple[str, list[dict]]:
    """
    End-to-end: recommend tracks → filter → create playlist → add tracks.
    Returns (playlist_url, list_of_track_dicts).
    """
    seeds = seed_tracks_for_mood(df, mood, centroid, n=5)
    if not seeds:
        raise ValueError(f"No seed tracks found for mood '{mood}'.")

    # Build target feature kwargs for recommendations API
    target_features = {
        f"target_{feat}": centroid[feat]
        for feat in FEATURE_COLS
        if feat != "tempo"
    }
    target_features["target_tempo"] = centroid.get("tempo", 120)

    recs = get_recommendations(
        sp, seeds, target_features, limit=min(num_tracks * 2, 100)
    )

    # Filter already-listened tracks
    filtered = [t for t in recs if t["id"] not in exclude_ids][:num_tracks]

    if not filtered:
        # Relax filter if everything was excluded
        filtered = recs[:num_tracks]

    playlist_id = create_playlist(
        sp, user_id, playlist_name,
        description=f"Auto-generated {mood} playlist by Spotify Mood Analyzer"
    )

    uris = [t["uri"] for t in filtered]
    add_tracks_to_playlist(sp, playlist_id, uris)

    playlist_url = f"https://open.spotify.com/playlist/{playlist_id}"
    return playlist_url, filtered
