import os
import spotipy
from spotipy.oauth2 import SpotifyOAuth
from dotenv import load_dotenv

load_dotenv()

SCOPES = " ".join([
    "user-read-recently-played",
    "user-top-read",
    "playlist-modify-public",
    "playlist-modify-private",
    "user-library-read",
])

def get_spotify_client() -> spotipy.Spotify:
    return spotipy.Spotify(
        auth_manager=SpotifyOAuth(
            client_id=os.getenv("SPOTIFY_CLIENT_ID"),
            client_secret=os.getenv("SPOTIFY_CLIENT_SECRET"),
            redirect_uri=os.getenv("SPOTIFY_REDIRECT_URI", "http://localhost:8501"),
            scope=SCOPES,
            cache_path=".spotify_cache",
            open_browser=True,
        )
    )


def get_current_user(sp: spotipy.Spotify) -> dict:
    return sp.current_user()


def get_top_tracks(sp: spotipy.Spotify, time_range: str, limit: int = 50) -> list[dict]:
    results = sp.current_user_top_tracks(limit=limit, time_range=time_range)
    return results.get("items", [])


def get_top_artists(sp: spotipy.Spotify, time_range: str, limit: int = 50) -> list[dict]:
    results = sp.current_user_top_artists(limit=limit, time_range=time_range)
    return results.get("items", [])


def get_recently_played(sp: spotipy.Spotify, limit: int = 50) -> list[dict]:
    """Paginate recently played tracks (up to 200)."""
    tracks = []
    result = sp.current_user_recently_played(limit=min(limit, 50))
    tracks.extend(result.get("items", []))

    while result.get("next") and len(tracks) < limit:
        cursor = result["cursors"]["before"] if result.get("cursors") else None
        if not cursor:
            break
        result = sp.current_user_recently_played(limit=50, before=cursor)
        tracks.extend(result.get("items", []))

    # Deduplicate by track id
    seen = set()
    unique = []
    for item in tracks:
        tid = item["track"]["id"]
        if tid and tid not in seen:
            seen.add(tid)
            unique.append(item["track"])
    return unique[:limit]


def get_audio_features(sp: spotipy.Spotify, track_ids: list[str]) -> list[dict]:
    """Fetch audio features in batches of 100 (Spotify API limit)."""
    features = []
    for i in range(0, len(track_ids), 100):
        batch = track_ids[i : i + 100]
        try:
            result = sp.audio_features(batch)
            if result:
                features.extend([f for f in result if f is not None])
        except Exception:
            pass
    return features


def get_recommendations(
    sp: spotipy.Spotify,
    seed_tracks: list[str],
    target_features: dict,
    limit: int = 30,
) -> list[dict]:
    """Fetch recommendations based on seed tracks and target audio features."""
    allowed_targets = {
        "target_energy", "target_valence", "target_danceability",
        "target_tempo", "target_acousticness", "target_instrumentalness",
        "target_speechiness",
    }
    kwargs = {k: v for k, v in target_features.items() if k in allowed_targets}
    try:
        result = sp.recommendations(seed_tracks=seed_tracks[:5], limit=limit, **kwargs)
        return result.get("tracks", [])
    except Exception as e:
        raise RuntimeError(f"Recommendations failed: {e}")


def create_playlist(sp: spotipy.Spotify, user_id: str, name: str, description: str = "") -> str:
    """Create a private playlist and return its ID."""
    playlist = sp.user_playlist_create(
        user=user_id,
        name=name,
        public=False,
        description=description,
    )
    return playlist["id"]


def add_tracks_to_playlist(sp: spotipy.Spotify, playlist_id: str, track_uris: list[str]) -> None:
    """Add tracks in batches of 100."""
    for i in range(0, len(track_uris), 100):
        sp.playlist_add_items(playlist_id, track_uris[i : i + 100])
