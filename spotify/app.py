"""Spotify Mood & Genre Analyzer — Streamlit app."""

from __future__ import annotations

import datetime
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from spotify_client import (
    get_spotify_client,
    get_current_user,
    get_top_tracks,
    get_top_artists,
    get_recently_played,
    get_audio_features,
)
from analyzer import (
    build_track_dataframe,
    cluster_tracks,
    aggregate_genres,
    dominant_mood,
    top_tracks_per_mood,
    FEATURE_COLS,
)
from playlist_builder import build_and_create_playlist

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Spotify Mood Analyzer",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    :root { --green: #1DB954; }
    body, .stApp { background-color: #121212; color: #FFFFFF; }
    .stSidebar { background-color: #000000; }
    h1, h2, h3, h4 { color: #FFFFFF; }
    .stButton > button {
        background-color: #1DB954;
        color: #000000;
        font-weight: 700;
        border-radius: 500px;
        border: none;
        padding: 0.5rem 1.5rem;
    }
    .stButton > button:hover { background-color: #1ed760; }
    .stSelectbox label, .stSlider label, .stTextInput label { color: #b3b3b3; }
    .stTabs [data-baseweb="tab"] { color: #b3b3b3; }
    .stTabs [aria-selected="true"] { color: #1DB954 !important; border-bottom-color: #1DB954 !important; }
    .track-card {
        display: flex; align-items: center; gap: 12px;
        background: #282828; border-radius: 8px;
        padding: 10px 14px; margin-bottom: 8px;
    }
    .track-card img { border-radius: 4px; }
    .track-title { font-weight: 600; font-size: 14px; }
    .track-sub { color: #b3b3b3; font-size: 12px; }
    a { color: #1DB954 !important; text-decoration: none; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Auth ────────────────────────────────────────────────────────────────────────
@st.cache_resource
def _get_sp():
    return get_spotify_client()


def _require_auth():
    try:
        sp = _get_sp()
        user = get_current_user(sp)
        return sp, user
    except Exception as e:
        st.error(f"Authentication failed: {e}")
        st.info("Make sure your .env file contains SPOTIFY_CLIENT_ID, SPOTIFY_CLIENT_SECRET, and SPOTIFY_REDIRECT_URI.")
        st.stop()


sp, user = _require_auth()

# ── Header ──────────────────────────────────────────────────────────────────────
col_img, col_name = st.columns([1, 8])
images = user.get("images") or []
if images:
    col_img.image(images[0]["url"], width=60)
col_name.markdown(
    f"### 👋 Welcome, **{user.get('display_name', 'Spotify User')}**",
    unsafe_allow_html=True,
)
st.divider()

# ── Sidebar nav ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎵 Spotify Analyzer")
    section = st.radio(
        "Navigate",
        ["My Listening", "Mood Analysis", "Generate Playlist"],
        label_visibility="collapsed",
    )
    st.divider()
    st.markdown("<small style='color:#b3b3b3'>Data cached for 1 hour</small>", unsafe_allow_html=True)

# ── Helpers ─────────────────────────────────────────────────────────────────────
TIME_RANGES = {
    "Last 4 weeks": "short_term",
    "Last 6 months": "medium_term",
    "All time": "long_term",
    "Recent plays": "recent",
}

MOOD_EMOJI = {
    "Energetic": "⚡",
    "Happy": "😊",
    "Melancholy": "🌧️",
    "Focus": "🧠",
    "Chill": "😌",
}

GREEN = "#1DB954"
RADAR_COLORS = [GREEN, "#E91E63", "#2196F3", "#FF9800", "#9C27B0"]


def _track_card(track: dict, rank: int | None = None) -> None:
    img = track.get("image_url") or (track.get("album", {}).get("images") or [{}])[0].get("url", "")
    name = track.get("name", "")
    artist = track.get("artist", "") or ", ".join(a["name"] for a in track.get("artists", []))
    url = track.get("external_url") or track.get("external_urls", {}).get("spotify", "#")
    prefix = f"{rank}. " if rank else ""
    img_tag = f'<img src="{img}" width="48" height="48">' if img else ""
    st.markdown(
        f'<div class="track-card">{img_tag}'
        f'<div><div class="track-title">{prefix}<a href="{url}" target="_blank">{name}</a></div>'
        f'<div class="track-sub">{artist}</div></div></div>',
        unsafe_allow_html=True,
    )


@st.cache_data(ttl=3600)
def fetch_listening_data(time_range_key: str):
    if time_range_key == "recent":
        tracks = get_recently_played(sp, limit=50)
        artists = get_top_artists(sp, "short_term", limit=50)
    else:
        tracks = get_top_tracks(sp, time_range_key, limit=50)
        artists = get_top_artists(sp, time_range_key, limit=50)
    return tracks, artists


@st.cache_data(ttl=3600)
def fetch_features_and_cluster(time_range_key: str):
    tracks, _ = fetch_listening_data(time_range_key)
    track_ids = [t["id"] for t in tracks if t.get("id")]
    features = get_audio_features(sp, track_ids)
    df = build_track_dataframe(tracks, features)
    if df.empty:
        return df, [], []
    df, centroids, labels = cluster_tracks(df)
    return df, centroids, labels


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — My Listening
# ══════════════════════════════════════════════════════════════════════════════
if section == "My Listening":
    st.header("🎧 My Listening")

    range_label = st.selectbox("Time range", list(TIME_RANGES.keys()))
    range_key = TIME_RANGES[range_label]

    with st.spinner("Fetching your listening data…"):
        tracks, artists = fetch_listening_data(range_key)

    if not tracks and not artists:
        st.warning("No data found for this time range.")
        st.stop()

    # ── Genre chart ───────────────────────────────────────────────────────────
    st.subheader("Top Genres")
    genre_df = aggregate_genres(artists)
    top_genres = genre_df.head(15)

    fig = px.bar(
        top_genres,
        x="score",
        y="genre",
        orientation="h",
        color_discrete_sequence=[GREEN],
        labels={"score": "Weighted Score", "genre": ""},
    )
    fig.update_layout(
        paper_bgcolor="#121212",
        plot_bgcolor="#121212",
        font_color="#FFFFFF",
        yaxis=dict(autorange="reversed"),
        margin=dict(l=0, r=0, t=20, b=0),
        height=400,
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Top artists & tracks ──────────────────────────────────────────────────
    col_a, col_t = st.columns(2)

    with col_a:
        st.subheader("Top 5 Artists")
        for i, artist in enumerate(artists[:5], 1):
            img_url = (artist.get("images") or [{}])[0].get("url", "")
            url = artist.get("external_urls", {}).get("spotify", "#")
            img_tag = f'<img src="{img_url}" width="48" height="48" style="border-radius:50%">' if img_url else ""
            st.markdown(
                f'<div class="track-card">{img_tag}'
                f'<div><div class="track-title">{i}. <a href="{url}" target="_blank">{artist["name"]}</a></div>'
                f'<div class="track-sub">{", ".join(artist.get("genres", [])[:2])}</div></div></div>',
                unsafe_allow_html=True,
            )

    with col_t:
        st.subheader("Top 5 Tracks")
        for i, track in enumerate(tracks[:5], 1):
            _track_card(track, rank=i)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — Mood Analysis
# ══════════════════════════════════════════════════════════════════════════════
elif section == "Mood Analysis":
    st.header("🧠 Mood Analysis")

    range_label = st.selectbox("Time range", list(TIME_RANGES.keys()))
    range_key = TIME_RANGES[range_label]

    with st.spinner("Clustering your tracks…"):
        df, centroids, labels = fetch_features_and_cluster(range_key)

    if df.empty:
        st.warning("Not enough tracks with audio features for this time range.")
        st.stop()

    dom = dominant_mood(df)
    emoji = MOOD_EMOJI.get(dom.split()[0], "🎵")
    st.success(f"{emoji} Your dominant mood is **{dom}** ({df[df['mood']==dom].shape[0]} tracks)")

    # ── Radar chart ───────────────────────────────────────────────────────────
    st.subheader("Mood Profiles (Audio Feature Radar)")

    radar_features = ["energy", "valence", "danceability", "acousticness", "instrumentalness", "speechiness"]

    fig = go.Figure()
    for i, (label, centroid) in enumerate(zip(labels, centroids)):
        vals = [centroid.get(f, 0) for f in radar_features]
        vals += vals[:1]  # close the loop
        fig.add_trace(go.Scatterpolar(
            r=vals,
            theta=radar_features + [radar_features[0]],
            fill="toself",
            name=f"{MOOD_EMOJI.get(label.split()[0], '🎵')} {label}",
            line_color=RADAR_COLORS[i % len(RADAR_COLORS)],
            opacity=0.7,
        ))

    fig.update_layout(
        polar=dict(
            bgcolor="#282828",
            radialaxis=dict(visible=True, range=[0, 1], color="#b3b3b3", gridcolor="#333"),
            angularaxis=dict(color="#b3b3b3", gridcolor="#333"),
        ),
        paper_bgcolor="#121212",
        font_color="#FFFFFF",
        legend=dict(bgcolor="#121212"),
        margin=dict(l=40, r=40, t=40, b=40),
        height=500,
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Example tracks per mood ───────────────────────────────────────────────
    st.subheader("Example Tracks per Mood")
    mood_tracks = top_tracks_per_mood(df, n=5)
    tabs = st.tabs([f"{MOOD_EMOJI.get(m.split()[0],'🎵')} {m}" for m in labels])
    for tab, label in zip(tabs, labels):
        with tab:
            subset = mood_tracks.get(label, pd.DataFrame())
            if subset.empty:
                st.write("No tracks in this cluster.")
            else:
                for _, row in subset.iterrows():
                    _track_card(row.to_dict())

    # ── Cluster size bar ─────────────────────────────────────────────────────
    st.subheader("Cluster Sizes")
    counts = df["mood"].value_counts().reset_index()
    counts.columns = ["mood", "count"]
    fig2 = px.bar(
        counts, x="mood", y="count",
        color_discrete_sequence=[GREEN],
        labels={"mood": "", "count": "Tracks"},
    )
    fig2.update_layout(
        paper_bgcolor="#121212", plot_bgcolor="#121212",
        font_color="#FFFFFF", margin=dict(t=20, b=0),
    )
    st.plotly_chart(fig2, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — Generate Playlist
# ══════════════════════════════════════════════════════════════════════════════
elif section == "Generate Playlist":
    st.header("🎛️ Generate Playlist")

    range_label = st.selectbox("Base time range for mood data", list(TIME_RANGES.keys()))
    range_key = TIME_RANGES[range_label]

    with st.spinner("Loading mood clusters…"):
        df, centroids, labels = fetch_features_and_cluster(range_key)

    if df.empty:
        st.warning("Not enough data to generate playlists. Try a different time range.")
        st.stop()

    mood_centroid_map = dict(zip(labels, centroids))

    selected_mood = st.selectbox(
        "Select a mood",
        [f"{MOOD_EMOJI.get(l.split()[0], '🎵')} {l}" for l in labels],
    )
    # Strip emoji prefix
    clean_mood = " ".join(selected_mood.split()[1:]) if selected_mood else labels[0]

    num_songs = st.slider("Number of songs", min_value=10, max_value=50, value=25, step=5)

    month_year = datetime.datetime.now().strftime("%B %Y")
    default_name = f"{clean_mood} Mix — {month_year}"
    playlist_name = st.text_input("Playlist name", value=default_name)

    st.divider()

    if st.button("🎵 Generate & Add to Spotify"):
        centroid = mood_centroid_map[clean_mood]
        known_ids = set(df["id"].tolist())

        with st.spinner("Fetching recommendations and creating playlist…"):
            try:
                playlist_url, rec_tracks = build_and_create_playlist(
                    sp=sp,
                    user_id=user["id"],
                    df=df,
                    mood=clean_mood,
                    centroid=centroid,
                    playlist_name=playlist_name,
                    num_tracks=num_songs,
                    exclude_ids=known_ids,
                )

                st.success(f"✅ Playlist **{playlist_name}** created!")
                st.markdown(
                    f'<a href="{playlist_url}" target="_blank">'
                    f'<button style="background:#1DB954;color:#000;font-weight:700;border:none;'
                    f'border-radius:500px;padding:0.5rem 1.5rem;cursor:pointer;font-size:15px;">'
                    f'🔗 Open in Spotify</button></a>',
                    unsafe_allow_html=True,
                )
                st.subheader(f"Tracks added ({len(rec_tracks)})")
                for track in rec_tracks:
                    _track_card(track)

            except Exception as e:
                st.error(f"Failed to create playlist: {e}")
