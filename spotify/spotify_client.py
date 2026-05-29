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
            redirect_uri=os.getenv("SPOTIFY_REDIRECT_URI", "http://127.0.0.1:8888"),
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
    tracks = []
    result = sp.current_user_recently_played(limit=min(limit, 50))
    tracks.extend(result.get("items", []))

    while result.get("next") and len(tracks) < limit:
        cursor = result["cursors"]["before"] if result.get("cursors") else None
        if not cursor:
            break
        result = sp.current_user_recently_played(limit=50, before=cursor)
        tracks.extend(result.get("items", []))

    seen = set()
    unique = []
    for item in tracks:
        tid = item["track"]["id"]
        if tid and tid not in seen:
            seen.add(tid)
            unique.append(item["track"])
    return unique[:limit]


def get_artists_by_ids(sp: spotipy.Spotify, artist_ids: list[str]) -> list[dict]:
    """Fetch full artist objects (with genres) in batches of 50."""
    artists = []
    for i in range(0, len(artist_ids), 50):
        batch = artist_ids[i : i + 50]
        try:
            result = sp.artists(batch)
            artists.extend(result.get("artists") or [])
        except Exception:
            pass
    return [a for a in artists if a]


def get_related_artists(sp: spotipy.Spotify, artist_id: str) -> list[dict]:
    try:
        result = sp.artist_related_artists(artist_id)
        return result.get("artists", [])
    except Exception:
        return []


def get_artist_top_tracks(sp: spotipy.Spotify, artist_id: str, market: str = "US") -> list[dict]:
    try:
        result = sp.artist_top_tracks(artist_id, country=market)
        return result.get("tracks", [])
    except Exception:
        return []


def create_playlist(sp: spotipy.Spotify, user_id: str, name: str, description: str = "") -> str:
    playlist = sp.user_playlist_create(
        user=user_id,
        name=name,
        public=False,
        description=description,
    )
    return playlist["id"]


def add_tracks_to_playlist(sp: spotipy.Spotify, playlist_id: str, track_uris: list[str]) -> None:
    for i in range(0, len(track_uris), 100):
        sp.playlist_add_items(playlist_id, track_uris[i : i + 100])
