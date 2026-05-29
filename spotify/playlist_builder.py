from __future__ import annotations

import spotipy
import pandas as pd

from spotify_client import create_playlist, add_tracks_to_playlist
from analyzer import top_artist_names_for_mood


def _search_artist_tracks(sp: spotipy.Spotify, artist_name: str, limit: int = 10) -> list[dict]:
    """Search for tracks by artist name."""
    try:
        res = sp.search(q=f'artist:"{artist_name}"', type="track", limit=limit)
        return res.get("tracks", {}).get("items", [])
    except Exception:
        return []


def build_and_create_playlist(
    sp: spotipy.Spotify,
    user_id: str,
    df: pd.DataFrame,
    mood: str,
    playlist_name: str,
    num_tracks: int,
    exclude_ids: set[str],
) -> tuple[str, list[dict]]:
    """
    Find tracks from the same artists as the mood cluster via search,
    then create a playlist.
    """
    artist_names = top_artist_names_for_mood(df, mood, n=8)
    if not artist_names:
        raise ValueError(f"No artists found for mood '{mood}'.")

    candidates: list[dict] = []
    seen_ids: set[str] = set()

    for name in artist_names:
        results = _search_artist_tracks(sp, name, limit=10)
        for t in results:
            tid = t.get("id")
            if tid and tid not in exclude_ids and tid not in seen_ids:
                seen_ids.add(tid)
                candidates.append(t)
        if len(candidates) >= num_tracks * 2:
            break

    if not candidates:
        raise ValueError(
            "Could not find new tracks to add. Try a different time range or mood."
        )

    selected = candidates[:num_tracks]

    playlist_id = create_playlist(
        sp, user_id, playlist_name,
        description=f"Auto-generated {mood} playlist by Spotify Mood Analyzer",
    )
    uris = [t["uri"] for t in selected]
    add_tracks_to_playlist(sp, playlist_id, uris)

    playlist_url = f"https://open.spotify.com/playlist/{playlist_id}"
    return playlist_url, selected
