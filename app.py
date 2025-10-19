pip install streamlit openai moviepy requests
import os
import io
import math
import tempfile
import requests
from pathlib import Path

import streamlit as st
from moviepy.editor import (
    VideoFileClip,
    AudioFileClip,
    ImageClip,
    CompositeVideoClip,
    TextClip,
)

# -----------------------------
# Config & Secrets
# -----------------------------
st.set_page_config(page_title="KSM AutoVideo Creator", page_icon="🎥", layout="centered")
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", None)
PEXELS_API_KEY = st.secrets.get("PEXELS_API_KEY", None)

# -----------------------------
# UI
# -----------------------------
st.title("🎥 KSM AutoVideo Creator")
st.caption("Paste Title + Content → get a 30–90 sec video with AI voice, dynamic background, and optional logo overlay.")

col1, col2 = st.columns(2)
with col1:
    title = st.text_input("Video Title", placeholder="e.g., DMAIC meets Process Intelligence")
with col2:
    target_duration = st.slider("Target Duration (sec)", 30, 90, 60, 5)

content = st.text_area(
    "Video Content (1–3 short paragraphs)",
    height=200,
    placeholder=(
        "Paste your learning here… We'll auto-trim to fit your target duration, "
        "add AI voice, choose a relevant background, and render captions."
    ),
)

with st.expander("Advanced Options", expanded=False):
    voice = st.selectbox(
        "Voice (OpenAI TTS)",
        ["alloy", "verse", "aria", "sage"],
        index=0,
        help="Requires OPENAI_API_KEY in Streamlit secrets.",
    )
    topic_hint = st.text_input(
        "Topic / Keywords (optional)",
        placeholder="e.g., Lean Six Sigma, BPM, AI, Process Mining",
        help="Improves background selection if Pexels is available.",
    )
    enable_captions = st.checkbox("Embed rolling captions", value=True)
    add_title_card = st.checkbox("Add branded title card (first 3s)", value=True)
    bgm_mix = st.checkbox("Add light background music (royalty-free demo)", value=False)

logo_file = st.file_uploader("Optional: Upload your logo (PNG with transparency works best)", type=["png", "jpg", "jpeg"])    

# -----------------------------
# Helpers
# -----------------------------
CURATED_BG = {
    "lean": [
        # Curated, permissive stock loops (fallback if no Pexels)
        "https://player.vimeo.com/external/357479265.sd.mp4?s=9a18c1a7b09f3f5c6e237ef55ff29ce7cbe9da9a&profile_id=164",
        "https://player.vimeo.com/external/214857965.sd.mp4?s=3b7627a6d4a485a4f33a61f54e63a60b9d2e6cda&profile_id=164",
    ],
    "process": [
        "https://player.vimeo.com/external/357478777.sd.mp4?s=4f4c3ee6c52777c3339f6f1a6fcba6d8b49e6c38&profile_id=164",
    ],
    "ai": [
        "https://player.vimeo.com/external/331695260.sd.mp4?s=6f1b8a8b3022c1b3b3c9f94d0d8de7b4d8b3b3a0&profile_id=164",
    ],
    "default": [
        "https://player.vimeo.com/external/357479265.sd.mp4?s=9a18c1a7b09f3f5c6e237ef55ff29ce7cbe9da9a&profile_id=164",
    ],
}


def estimate_speaking_time_seconds(text: str, wps: float = 2.2) -> int:
    words = max(1, len(text.strip().split()))
    return math.ceil(words / wps)


def trim_to_duration(text: str, target_sec: int, wps: float = 2.2) -> str:
    """Trim text to roughly fit target duration using words-per-second heuristic."""
    max_words = max(35, int(target_sec * wps))  # keep a sensible minimum
    words = text.strip().split()
    if len(words) <= max_words:
        return text.strip()
    return " ".join(words[:max_words]) + "…"


def chunk_text(text: str, approx_words: int = 10):
    words = text.split()
    for i in range(0, len(words), approx_words):
        yield " ".join(words[i : i + approx_words])


def pexels_search_video(query: str) -> str | None:
    if not PEXELS_API_KEY:
        return None
    try:
        headers = {"Authorization": PEXELS_API_KEY}
        params = {"query": query, "per_page": 1, "orientation": "landscape"}
        r = requests.get("https://api.pexels.com/videos/search", headers=headers, params=params, timeout=15)
        if r.status_code == 200:
            data = r.json()
            if data.get("videos"):
                # pick the highest quality file
                files = data["videos"][0].get("video_files", [])
                if files:
                    files = sorted(files, key=lambda f: f.get("width", 0), reverse=True)
                    return files[0].get("link")
    except Exception:
        return None
    return None


def pick_background_url(title: str, content: str, topic_hint: str | None) -> str:
    # Prefer Pexels search if key is present
    query = (topic_hint or title or content or "").lower()
    candidates = []
    if any(k in query for k in ["lean", "six sigma", "kaizen", "dmaic", "process"]):
        candidates += CURATED_BG["lean"] + CURATED_BG["process"]
    if any(k in query for k in ["ai", "analytics", "data", "automation", "mining"]):
        candidates += CURATED_BG["ai"]
    if PEXELS_API_KEY:
        # Try a few smart queries
        for q in [topic_hint, title, "abstract technology background", "minimal gradient", "office workflow b-roll"]:
            if not q:
                continue
            url = pexels_search_video(q)
            if url:
                return url
    # fallback to curated
    return (candidates or CURATED_BG["default"])[0]


def download_to_temp(url: str, suffix: str) -> str:
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    fd, path = tempfile.mkstemp(suffix=suffix)
    with os.fdopen(fd, "wb") as f:
        f.write(resp.content)
    return path


# -----------------------------
# OpenAI TTS (with graceful fallback)
# -----------------------------

def synthesize_speech_openai(text: str, voice: str = "alloy") -> str:
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY missing. Add it in Streamlit secrets.")
    # Lazy import to avoid hard dependency if not used
    from openai import OpenAI

    client = OpenAI(api_key=OPENAI_API_KEY)
    # Save to temp file
    fd, path = tempfile.mkstemp(suffix=".mp3")
    os.close(fd)
    try:
        # Using the current OpenAI TTS endpoint name
        # If the SDK version differs, adjust to the installed version's TTS method
        with client.audio.speech.with_streaming_response.create(
            model="gpt-4o-mini-tts",
            voice=voice,
            input=text,
            format="mp3",
        ) as response:
            response.stream_to_file(path)
        return path
    except AttributeError:
        # Older SDK fallback
        resp = client.audio.speech.create(model="gpt-4o-mini-tts", voice=voice, input=text)
        audio_bytes = resp.get("data") or getattr(resp, "content", None)
        if not audio_bytes:
            raise RuntimeError("TTS failed: empty response body")
        Path(path).write_bytes(audio_bytes)
        return path


# -----------------------------
# Video Assembly
# -----------------------------

def build_video(
    title: str,
    narration_text: str,
    voice: str,
    target_duration: int,
    topic_hint: str | None,
    logo_bytes: bytes | None,
    enable_captions: bool,
    add_title_card: bool,
    bgm_mix: bool,
) -> str:
    # Trim narration to target duration
    trimmed_text = trim_to_duration(narration_text, target_duration)

    # TTS
    audio_path = synthesize_speech_openai(trimmed_text, voice=voice)
    audio = AudioFileClip(audio_path)

    # Background
    bg_url = pick_background_url(title, narration_text, topic_hint)
    bg_path = download_to_temp(bg_url, suffix=".mp4")
    video_bg = VideoFileClip(bg_path)

    # Align duration: use max of audio duration and min(target_duration, bg length)
    out_dur = min(max(audio.duration, 30), float(target_duration), float(video_bg.duration))
    video_bg = video_bg.subclip(0, out_dur).volumex(1.0)
    video_bg = video_bg.set_audio(audio)

    overlays = [video_bg]

    # Optional title card (first 3s)
    if add_title_card:
        try:
            ttl = TextClip(title, fontsize=64, color="white", bg_color="black", size=video_bg.size)
            ttl = ttl.set_duration(min(3, out_dur)).set_position("center")
            overlays.append(ttl)
        except Exception:
            # If ImageMagick not present, skip title card
            pass

    # Optional logo overlay
    if logo_bytes:
        try:
            fd, logo_path = tempfile.mkstemp(suffix=".png")
            with os.fdopen(fd, "wb") as f:
                f.write(logo_bytes)
            logo_clip = ImageClip(logo_path).set_duration(out_dur)
            # Resize to ~10% width
            w, h = video_bg.size
            target_w = max(96, int(w * 0.12))
            logo_clip = logo_clip.resize(width=target_w).set_position(("right", "bottom"), relative=False).margin(
                right=40, bottom=30, opacity=0
            )
            overlays.append(logo_clip)
        except Exception:
            pass

    # Rolling captions (bottom), chunk text proportional to audio duration
    if enable_captions:
        try:
            total_words = max(1, len(trimmed_text.split()))
            approx_segments = max(3, int(out_dur // 2.5))
            words_per_seg = max(6, total_words // approx_segments)
            t = 0.0
            for chunk in chunk_text(trimmed_text, words_per_seg):
                seg_dur = min(4.0, max(1.5, out_dur / approx_segments))
                try:
                    txt = TextClip(
                        chunk,
                        fontsize=38,
                        color="white",
                        bg_color="rgba(0,0,0,0.55)",
                        method="caption",
                        size=(int(video_bg.w * 0.9), None),
                    ).set_duration(seg_dur).set_position(("center", int(video_bg.h * 0.82)))
                    overlays.append(txt.set_start(t))
                    t += seg_dur
                    if t >= out_dur:
                        break
                except Exception:
                    # If caption rendering fails, stop trying
                    break
        except Exception:
            pass

    # Optional BGM (very low volume overlay) — quick demo using a free loop
    if bgm_mix:
        try:
            song_url = "https://cdn.pixabay.com/download/audio/2022/03/15/audio_3f2bf7b7a6.mp3?filename=soft-ambient-110241.mp3"
            bgm_path = download_to_temp(song_url, suffix=".mp3")
            bgm = AudioFileClip(bgm_path).volumex(0.06).audio_loop(duration=out_dur)
            # Mix: narration already on video; we overlay BGM quietly by setting as composite
            video_bg_audio = overlays[0].audio.set_duration(out_dur)
            from moviepy.audio.AudioClip import CompositeAudioClip

            mixed = CompositeAudioClip([video_bg_audio, bgm])
            overlays[0] = overlays[0].set_audio(mixed)
        except Exception:
            pass

    final = CompositeVideoClip(overlays)
    out_path = Path(tempfile.gettempdir()) / "KSM_AutoVideo_Creator.mp4"
    final.write_videofile(
        str(out_path),
        fps=24,
        codec="libx264",
        audio_codec="aac",
        threads=4,
        preset="medium",
        bitrate="3000k",
        verbose=False,
        logger=None,
    )
    return str(out_path)


# -----------------------------
# Action
# -----------------------------
if st.button("🎬 Generate Video", use_container_width=True, disabled=not title or not content):
    if not title or not content:
        st.error("Please provide both title and content.")
        st.stop()
    try:
        with st.spinner("Generating narration, selecting background, and rendering video…"):
            logo_bytes = logo_file.read() if logo_file else None
            out_path = build_video(
                title=title.strip(),
                narration_text=content.strip(),
                voice=voice,
                target_duration=target_duration,
                topic_hint=topic_hint.strip() if topic_hint else None,
                logo_bytes=logo_bytes,
                enable_captions=enable_captions,
                add_title_card=add_title_card,
                bgm_mix=bgm_mix,
            )
        st.success("✅ Video generated!")
        st.video(out_path)
        with open(out_path, "rb") as f:
            st.download_button("⬇️ Download MP4", f, file_name="KSM_AutoVideo.mp4")
        st.info("Tip: Post on LinkedIn with a 1–2 line hook + 3 bullets in the caption for best reach.")
    except Exception as e:
        st.error(f"Error: {e}\nCheck that your OPENAI_API_KEY is set. If captions fail, disable them in Advanced Options.")

# -----------------------------
# Footer
# -----------------------------
st.caption(
    "Needs secrets: OPENAI_API_KEY (required), PEXELS_API_KEY (optional for smarter backgrounds).\n"
    "If ImageMagick is not present on the host, title card / captions will auto-skip."
)
