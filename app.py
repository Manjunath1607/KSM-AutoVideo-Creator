import os
import re
import math
import textwrap
import tempfile
from pathlib import Path

import requests
import streamlit as st
from gtts import gTTS
from moviepy.editor import (
    VideoFileClip,
    AudioFileClip,
    ImageClip,
    CompositeVideoClip,
)
from PIL import Image, ImageDraw, ImageFont

# =============================================================
# KSM AutoVideo Creator — gTTS Edition (no OpenAI required)
# - TTS: gTTS (free)
# - Captions & title via Pillow (no ImageMagick)
# - Pexels backgrounds optional via PEXELS_API_KEY (Streamlit secrets)
# =============================================================

st.set_page_config(page_title="KSM AutoVideo Creator", page_icon="🎥", layout="centered")

# Optional: Pexels API key (for smarter backgrounds). If absent, we use curated fallbacks.
PEXELS_API_KEY = st.secrets.get("PEXELS_API_KEY")

# ---------- UI ----------
st.title("KSM AutoVideo Creator")
st.caption("Paste Title + Content → get a 30–90 sec video with free TTS (gTTS), dynamic background, and optional logo overlay.")

c1, c2 = st.columns([2, 1])
with c1:
    title = st.text_input("Video Title", placeholder="e.g., DMAIC meets Process Intelligence")
with c2:
    target_duration = st.slider("Target Duration (sec)", 30, 90, 60, 5)

content = st.text_area(
    "Video Content (1–3 short paragraphs)",
    height=220,
    placeholder=(
        "Paste your learning here… We'll auto-trim to fit your target duration, "
        "create voice with gTTS, choose a relevant background, and render captions."
    ),
)

with st.expander("Advanced Options", expanded=False):
    language = st.selectbox("Voice language (gTTS)", ["en", "en-uk", "en-au", "hi", "ta"], index=0, help="gTTS language code")
    topic_hint = st.text_input("Topic / Keywords (optional)", placeholder="Lean Six Sigma, BPM, AI, Process Mining")
    enable_captions = st.checkbox("Embed rolling captions", value=True)
    add_title_card = st.checkbox("Add branded title card (first 3s)", value=True)
    bgm_mix = st.checkbox("Add light background music (royalty-free demo)", value=False)

logo_file = st.file_uploader("Optional: Upload your logo (PNG with transparency works best)", type=["png", "jpg", "jpeg"])

# ---------- Helpers ----------
CURATED_BG = {
    "lean": [
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


def slugify(text: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", (text or "ksm-video").strip().lower())
    return s.strip("-") or "ksm-video"


def estimate_speaking_time_seconds(text: str, wps: float = 2.2) -> int:
    words = max(1, len(text.strip().split()))
    return math.ceil(words / wps)


def trim_to_duration(text: str, target_sec: int, wps: float = 2.2) -> str:
    max_words = max(35, int(target_sec * wps))
    words = text.strip().split()
    if len(words) <= max_words:
        return text.strip()
    return " ".join(words[:max_words]) + "…"


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
                files = data["videos"][0].get("video_files", [])
                if files:
                    files = sorted(files, key=lambda f: f.get("width", 0), reverse=True)
                    return files[0].get("link")
    except Exception:
        return None
    return None


def pick_background_url(title: str, content: str, topic_hint: str | None) -> str:
    query = (topic_hint or title or content or "").lower()
    candidates = []
    if any(k in query for k in ["lean", "six sigma", "kaizen", "dmaic", "process"]):
        candidates += CURATED_BG["lean"] + CURATED_BG["process"]
    if any(k in query for k in ["ai", "analytics", "data", "automation", "mining"]):
        candidates += CURATED_BG["ai"]
    if PEXELS_API_KEY:
        for q in [topic_hint, title, "abstract technology background", "minimal gradient", "office workflow b-roll"]:
            if not q:
                continue
            url = pexels_search_video(q)
            if url:
                return url
    return (candidates or CURATED_BG["default"])[0]


def download_to_temp(url: str, suffix: str) -> str:
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    fd, path = tempfile.mkstemp(suffix=suffix)
    with os.fdopen(fd, "wb") as f:
        f.write(resp.content)
    return path


# ---------- TTS (gTTS: free, no key required) ----------
def synthesize_speech(text: str, lang: str = "en") -> str:
    fd, path = tempfile.mkstemp(suffix=".mp3")
    os.close(fd)
    # gTTS may take a couple seconds; keep text concise (we auto-trim)
    gTTS(text=text, lang=lang.split("-")[0]).save(path)
    return path


# ---------- Pillow-based text rendering (no ImageMagick) ----------
def make_text_image(
    text: str,
    width: int,
    pad: int = 24,
    fontsize: int = 34,
    fg=(255, 255, 255),
    bg=(0, 0, 0, 160),
) -> Image.Image:
    max_chars = max(10, width // (fontsize // 2))
    wrapped = []
    for para in text.splitlines():
        wrapped += textwrap.wrap(para, width=max_chars) or [""]
    lines = wrapped or [""]

    font = ImageFont.load_default()
    line_h = int(fontsize * 1.6)
    img_h = pad * 2 + line_h * len(lines)

    img = Image.new("RGBA", (width, img_h), (0, 0, 0, 0))
    bg_img = Image.new("RGBA", (width, img_h), (0, 0, 0, 0))
    draw_bg = ImageDraw.Draw(bg_img)
    draw_bg.rectangle([0, 0, width, img_h], fill=bg)
    img.alpha_composite(bg_img, (0, 0))

    draw = ImageDraw.Draw(img)
    y = pad
    for line in lines:
        draw.text((pad, y), line, font=font, fill=fg)
        y += line_h
    return img


# ---------- Video Assembly ----------
def build_video(
    title: str,
    narration_text: str,
    lang: str,
    target_duration: int,
    topic_hint: str | None,
    logo_bytes: bytes | None,
    enable_captions: bool,
    add_title_card: bool,
    bgm_mix: bool,
) -> str:
    trimmed_text = trim_to_duration(narration_text, target_duration)

    # TTS
    audio_path = synthesize_speech(trimmed_text, lang=lang)
    audio = AudioFileClip(audio_path)

    # Background video
    bg_url = pick_background_url(title, narration_text, topic_hint)
    bg_path = download_to_temp(bg_url, suffix=".mp4")
    video_bg = VideoFileClip(bg_path)

    # Duration
    out_dur = min(max(audio.duration, 30), float(target_duration), float(video_bg.duration))
    video_bg = video_bg.subclip(0, out_dur).volumex(1.0)
    video_bg = video_bg.set_audio(audio)
    overlays = [video_bg]

    # Title card (Pillow)
    if add_title_card:
        try:
            w, h = video_bg.size
            img = make_text_image(title, width=min(w - 80, int(w * 0.9)), fontsize=42)
            title_clip = ImageClip(img).set_duration(min(3, out_dur))
            title_clip = title_clip.set_position(("center", "center"))
            overlays.append(title_clip)
        except Exception:
            pass

    # Logo
    if logo_bytes:
        try:
            fd, logo_path = tempfile.mkstemp(suffix=".png")
            with os.fdopen(fd, "wb") as f:
                f.write(logo_bytes)
            logo_clip = ImageClip(logo_path).set_duration(out_dur)
            w, h = video_bg.size
            target_w = max(96, int(w * 0.12))
            logo_clip = (
                logo_clip.resize(width=target_w)
                .set_position(("right", "bottom"))
                .margin(right=40, bottom=30, opacity=0)
            )
            overlays.append(logo_clip)
        except Exception:
            pass

    # Captions
    if enable_captions:
        try:
            total_words = max(1, len(trimmed_text.split()))
            approx_segments = max(3, int(out_dur // 2.5))
            words_per_seg = max(6, total_words // approx_segments)
            t = 0.0
            w, h = video_bg.size
            cap_w = int(w * 0.9)
            words = trimmed_text.split()
            for i in range(0, len(words), words_per_seg):
                chunk = " ".join(words[i : i + words_per_seg])
                if not chunk:
                    break
                seg_dur = min(4.0, max(1.4, out_dur / approx_segments))
                img = make_text_image(chunk, width=cap_w, fontsize=32)
                cap_clip = ImageClip(img).set_duration(seg_dur)
                cap_clip = cap_clip.set_position(("center", int(h * 0.82))).set_start(t)
                overlays.append(cap_clip)
                t += seg_dur
                if t >= out_dur:
                    break
        except Exception:
            pass

    # Optional BGM
    if bgm_mix:
        try:
            music_url = "https://cdn.pixabay.com/download/audio/2022/03/15/audio_3f2bf7b7a6.mp3?filename=soft-ambient-110241.mp3"
            music_path = download_to_temp(music_url, suffix=".mp3")
            bgm = AudioFileClip(music_path).volumex(0.06).audio_loop(duration=out_dur)
            from moviepy.audio.AudioClip import CompositeAudioClip

            mixed = CompositeAudioClip([overlays[0].audio.set_duration(out_dur), bgm])
            overlays[0] = overlays[0].set_audio(mixed)
        except Exception:
            pass

    final = CompositeVideoClip(overlays)

    out_name = f"{slugify(title)}.mp4"
    out_path = Path(tempfile.gettempdir()) / out_name
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


# ---------- Action ----------
btn = st.button("🎬 Generate Video", use_container_width=True, disabled=not title or not content)
if btn:
    try:
        with st.spinner("Generating voice (gTTS), selecting background, and rendering video…"):
            logo_bytes = logo_file.read() if logo_file else None
            out_path = build_video(
                title=title.strip(),
                narration_text=content.strip(),
                lang=language,
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
            st.download_button("⬇️ Download MP4", f, file_name=Path(out_path).name)
    except Exception as e:
        st.error(f"Error: {e}")

# ---------- Footer ----------
st.caption(
    "Pexels key (optional) can be added in Streamlit Secrets as PEXELS_API_KEY for smarter backgrounds. "
    "Captions & title use Pillow (no ImageMagick needed)."
)
