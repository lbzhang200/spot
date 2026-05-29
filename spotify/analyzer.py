from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

FEATURE_COLS = [
    "energy", "valence", "danceability",
    "tempo", "acousticness", "instrumentalness", "speechiness",
]

MOOD_RULES: list[tuple[str, dict]] = [
    ("Energetic",  {"energy": (0.6, 1), "tempo_norm": (0.6, 1)}),
    ("Happy",      {"valence": (0.6, 1), "danceability": (0.6, 1)}),
    ("Melancholy", {"valence": (0, 0.4), "acousticness": (0.5, 1)}),
    ("Focus",      {"tempo_norm": (0, 0.4), "energy": (0, 0.4), "instrumentalness": (0.4, 1)}),
    ("Chill",      {}),  # fallback
]


def _label_cluster(centroid: dict) -> str:
    for mood, rules in MOOD_RULES:
        if all(
            lo <= centroid.get(feat, 0) <= hi
            for feat, (lo, hi) in rules.items()
        ):
            return mood
    return "Chill"


def build_track_dataframe(tracks: list[dict], features: list[dict]) -> pd.DataFrame:
    """Merge track metadata with audio features."""
    feat_map = {f["id"]: f for f in features if f.get("id")}

    rows = []
    for t in tracks:
        tid = t.get("id")
        if not tid or tid not in feat_map:
            continue
        f = feat_map[tid]
        rows.append({
            "id": tid,
            "name": t.get("name", ""),
            "artist": ", ".join(a["name"] for a in t.get("artists", [])),
            "album": t.get("album", {}).get("name", ""),
            "image_url": (t.get("album", {}).get("images") or [{}])[0].get("url", ""),
            "uri": t.get("uri", ""),
            "external_url": t.get("external_urls", {}).get("spotify", ""),
            "energy": f.get("energy", 0),
            "valence": f.get("valence", 0),
            "danceability": f.get("danceability", 0),
            "tempo": f.get("tempo", 0),
            "acousticness": f.get("acousticness", 0),
            "instrumentalness": f.get("instrumentalness", 0),
            "speechiness": f.get("speechiness", 0),
        })
    return pd.DataFrame(rows)


def cluster_tracks(df: pd.DataFrame, k: int = 5) -> tuple[pd.DataFrame, list[dict], list[str]]:
    """Run KMeans on audio features. Returns augmented df, centroids, mood labels."""
    if df.empty or len(df) < k:
        k = max(1, len(df))

    scaler = MinMaxScaler()
    X = df[FEATURE_COLS].copy()
    X["tempo"] = (X["tempo"] - 60) / (200 - 60)  # rough BPM normalisation
    X = X.clip(0, 1)
    X_scaled = scaler.fit_transform(X)

    km = KMeans(n_clusters=k, n_init=20, random_state=42)
    df = df.copy()
    df["cluster"] = km.fit_predict(X_scaled)

    # Build centroid dicts in original (normalised) feature space
    centroids = []
    labels = []
    for i in range(k):
        cluster_rows = df[df["cluster"] == i]
        centroid = cluster_rows[FEATURE_COLS].mean().to_dict()
        # add normalised tempo for rule matching
        centroid["tempo_norm"] = (centroid["tempo"] - 60) / (200 - 60)
        centroid["tempo_norm"] = max(0, min(1, centroid["tempo_norm"]))
        label = _label_cluster(centroid)
        centroids.append(centroid)
        labels.append(label)

    # Deduplicate labels by appending suffix if needed
    seen: dict[str, int] = {}
    deduped: list[str] = []
    for lbl in labels:
        if lbl in seen:
            seen[lbl] += 1
            deduped.append(f"{lbl} {seen[lbl]}")
        else:
            seen[lbl] = 1
            deduped.append(lbl)

    df["mood"] = df["cluster"].map(lambda i: deduped[i])
    return df, centroids, deduped


def aggregate_genres(artists: list[dict]) -> pd.DataFrame:
    """Return genre counts weighted by artist rank (index 0 = highest)."""
    genre_scores: dict[str, float] = {}
    n = len(artists)
    for rank, artist in enumerate(artists):
        weight = (n - rank) / n
        for genre in artist.get("genres", []):
            genre_scores[genre] = genre_scores.get(genre, 0) + weight

    df = pd.DataFrame(
        list(genre_scores.items()), columns=["genre", "score"]
    ).sort_values("score", ascending=False)
    return df


def dominant_mood(df: pd.DataFrame) -> str:
    return df["mood"].value_counts().idxmax() if not df.empty else "Unknown"


def top_tracks_per_mood(df: pd.DataFrame, n: int = 5) -> dict[str, pd.DataFrame]:
    result = {}
    for mood, group in df.groupby("mood"):
        result[mood] = group.head(n)
    return result


def seed_tracks_for_mood(df: pd.DataFrame, mood: str, centroid: dict, n: int = 5) -> list[str]:
    """Return track IDs closest to the centroid for a given mood."""
    subset = df[df["mood"] == mood].copy()
    if subset.empty:
        return []

    feat_arr = subset[FEATURE_COLS].values.astype(float)
    centroid_arr = np.array([centroid.get(f, 0) for f in FEATURE_COLS])

    # Normalise tempo in both
    tempo_idx = FEATURE_COLS.index("tempo")
    feat_arr[:, tempo_idx] = (feat_arr[:, tempo_idx] - 60) / 140
    centroid_arr[tempo_idx] = centroid.get("tempo_norm", 0)

    dists = np.linalg.norm(feat_arr - centroid_arr, axis=1)
    subset = subset.copy()
    subset["_dist"] = dists
    return subset.nsmallest(n, "_dist")["id"].tolist()
