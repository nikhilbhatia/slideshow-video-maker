import streamlit as st
import os
import tempfile
import shutil
from pathlib import Path
from itertools import islice
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import numpy as np
from moviepy import ImageClip, AudioFileClip, concatenate_videoclips

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Slideshow Video Maker", page_icon="🎬", layout="wide")
st.title("🎬 Slideshow Video Maker")
st.caption("Upload photos + names, add audio, and export an MP4 ready to embed in PowerPoint.")

# ── Constants ─────────────────────────────────────────────────────────────────
VIDEO_W, VIDEO_H = 1280, 720
BG_COLOR         = (15, 15, 30)
NAME_COLOR       = (255, 255, 255)
LABEL_COLOR      = (180, 200, 255)
BORDER_COLOR     = (255, 255, 255, 160)
MAX_PEOPLE       = 48
MAX_PER_SLIDE    = 8

# cols, photo_size, font_name, font_label  — indexed by n (1-based)
LAYOUT = {
    1: (1, 380, 54, 32),
    2: (2, 290, 42, 26),
    3: (3, 210, 34, 22),
    4: (2, 250, 34, 22),
    5: (3, 185, 30, 20),
    6: (3, 182, 29, 19),
    7: (4, 152, 26, 17),
    8: (4, 148, 25, 16),
}


# ── Rendering helpers ─────────────────────────────────────────────────────────

def load_font(size: int):
    for path in [
        # macOS
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/Arial.ttf",
        "/Library/Fonts/Arial.ttf",
        # Linux (Streamlit Cloud / Ubuntu)
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ]:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                pass
    return ImageFont.load_default()


def crop_square(img: Image.Image) -> Image.Image:
    s    = min(img.width, img.height)
    left = (img.width - s) // 2
    top  = (img.height - s) // 2
    return img.crop((left, top, left + s, top + s))


def make_circle_photo(img: Image.Image, size: int, border: int = 4) -> Image.Image:
    img  = img.resize((size, size), Image.LANCZOS).convert("RGBA")
    mask = Image.new("L", (size, size), 0)
    ImageDraw.Draw(mask).ellipse((0, 0, size - 1, size - 1), fill=255)
    img.putalpha(mask)
    ring = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    ImageDraw.Draw(ring).ellipse(
        (border // 2, border // 2, size - border // 2 - 1, size - border // 2 - 1),
        outline=BORDER_COLOR, width=border,
    )
    return Image.alpha_composite(img, ring)


def make_square_photo(img: Image.Image, size: int, border: int = 4) -> Image.Image:
    img = img.resize((size, size), Image.LANCZOS).convert("RGB")
    ImageDraw.Draw(img).rectangle(
        [border // 2, border // 2, size - border // 2 - 1, size - border // 2 - 1],
        outline=(255, 255, 255), width=border,
    )
    return img


def draw_centered_text(draw, cx, y, text, font, color, shadow=True):
    bbox = draw.textbbox((0, 0), text, font=font)
    x    = cx - (bbox[2] - bbox[0]) // 2
    if shadow:
        draw.text((x + 2, y + 2), text, font=font, fill=(0, 0, 0))
    draw.text((x, y), text, font=font, fill=color)


def make_slide_frame(
    entries: list,
    img_shape: str,
    size: tuple = (VIDEO_W, VIDEO_H),
) -> np.ndarray:
    n    = len(entries)
    w, h = size

    cols, photo_size, fs_name, fs_label = LAYOUT.get(n, LAYOUT[8])
    rows = (n + cols - 1) // cols

    font_name  = load_font(fs_name)
    font_label = load_font(fs_label)

    has_label  = any(e.get("label", "").strip() for e in entries)
    text_h     = fs_name + 10 + (fs_label + 8 if has_label else 0)
    cell_h     = photo_size + 16 + text_h

    # Dynamic row gap: distribute remaining vertical space
    v_margin      = 40
    usable_h      = h - 2 * v_margin
    row_gap       = max(10, (usable_h - rows * cell_h) // max(1, rows + 1))
    total_grid_h  = rows * cell_h + max(0, rows - 1) * row_gap
    start_y       = (h - total_grid_h) // 2

    # Background
    if n == 1 and entries[0].get("img_path"):
        try:
            bp    = Image.open(entries[0]["img_path"]).convert("RGB")
            ratio = max(w / bp.width, h / bp.height)
            bp    = bp.resize((int(bp.width * ratio), int(bp.height * ratio)), Image.LANCZOS)
            cx_   = (bp.width - w) // 2
            cy_   = (bp.height - h) // 2
            bp    = bp.crop((cx_, cy_, cx_ + w, cy_ + h))
            bp    = bp.filter(ImageFilter.GaussianBlur(radius=22))
            bg    = Image.blend(bp, Image.new("RGB", (w, h), (0, 0, 0)), alpha=0.55)
        except Exception:
            bg = Image.new("RGB", (w, h), BG_COLOR)
    else:
        bg = Image.new("RGB", (w, h), BG_COLOR)

    draw = ImageDraw.Draw(bg)

    for i, entry in enumerate(entries):
        row_idx      = i // cols
        col_idx      = i % cols
        items_in_row = min(cols, n - row_idx * cols)

        # Dynamic col gap for this row
        h_margin    = 40
        usable_w    = w - 2 * h_margin
        col_gap     = max(10, (usable_w - items_in_row * photo_size) // max(1, items_in_row + 1))
        row_total_w = items_in_row * photo_size + max(0, items_in_row - 1) * col_gap
        row_start_x = (w - row_total_w) // 2

        px = row_start_x + col_idx * (photo_size + col_gap)
        py = start_y + row_idx * (cell_h + row_gap)
        cx = px + photo_size // 2

        # Photo
        try:
            photo = Image.open(entry["img_path"]).convert("RGB")
            photo = crop_square(photo)
            if img_shape == "Circle":
                rendered = make_circle_photo(photo, photo_size)
                bg_rgba  = bg.convert("RGBA")
                bg_rgba.paste(rendered, (px, py), rendered)
                bg   = bg_rgba.convert("RGB")
                draw = ImageDraw.Draw(bg)
            else:
                bg.paste(make_square_photo(photo, photo_size), (px, py))
        except Exception:
            draw.rectangle([px, py, px + photo_size, py + photo_size], fill=(50, 50, 80))

        ty = py + photo_size + 16
        draw_centered_text(draw, cx, ty, entry.get("name", ""), font_name, NAME_COLOR)
        if entry.get("label", "").strip():
            draw_centered_text(draw, cx, ty + fs_name + 10,
                               entry["label"].strip(), font_label, LABEL_COLOR, shadow=False)

    return np.array(bg.convert("RGB"))


def chunked(iterable, n):
    it = iter(iterable)
    while chunk := list(islice(it, n)):
        yield chunk


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")

    per_slide = st.selectbox(
        "People per slide",
        options=[1, 2, 3, 4, 5, 6, 7, 8],
        index=0,
        help=f"Up to {MAX_PER_SLIDE} people per slide. Remaining people overflow to the next slide automatically.",
    )
    img_shape = st.radio(
        "Photo shape", options=["Circle", "Square"], horizontal=True,
    )

    st.markdown("---")
    slide_duration = st.slider("Seconds per slide", 2, 15, 5)
    fps            = st.selectbox("Frame rate", [24, 30], index=1)
    fade_duration  = st.slider("Fade transition (s)", 0.0, 1.5, 0.5, step=0.1)
    loop_audio     = st.checkbox("Loop audio if shorter than video", value=True)

    st.markdown("---")
    st.markdown("**Output:** MP4 (H.264)")
    st.markdown("Embed in PPT: **Insert → Video → This Device**")


# ── Session state ─────────────────────────────────────────────────────────────
if "entries" not in st.session_state:
    st.session_state.entries = [{"name": "", "label": "", "image": None}]
if "expand_idx" not in st.session_state:
    st.session_state.expand_idx = 0     # which entry to auto-expand


def add_entry():
    if len(st.session_state.entries) < MAX_PEOPLE:
        st.session_state.entries.append({"name": "", "label": "", "image": None})
        st.session_state.expand_idx = len(st.session_state.entries) - 1


def remove_entry(i):
    st.session_state.entries.pop(i)
    st.session_state.expand_idx = max(0, i - 1)


# ── Step 1: People ────────────────────────────────────────────────────────────
st.subheader("Step 1 — Add people")

n_total  = len(st.session_state.entries)
n_valid  = sum(1 for e in st.session_state.entries if e["name"].strip() and e["image"])
n_slides = max(1, -(-n_valid // per_slide))
col_stat1, col_stat2, col_stat3 = st.columns(3)
col_stat1.metric("People added", f"{n_total} / {MAX_PEOPLE}")
col_stat2.metric("With photo & name", n_valid)
col_stat3.metric("Slides that will be created", n_slides)

st.markdown("")

# Paginate entries into groups of 8 shown as tabs
PAGE_SIZE   = 8
n_pages     = max(1, -(-n_total // PAGE_SIZE))
page_labels = [
    f"People {i*PAGE_SIZE+1}–{min((i+1)*PAGE_SIZE, n_total)}"
    for i in range(n_pages)
]

if n_pages == 1:
    page_idx = 0
    active_entries = list(enumerate(st.session_state.entries))
else:
    tabs = st.tabs(page_labels)
    # Determine which tab the expand_idx falls on
    default_tab = st.session_state.expand_idx // PAGE_SIZE

    # We can't programmatically select a tab, so just render all tabs
    for tab_i, tab in enumerate(tabs):
        with tab:
            start = tab_i * PAGE_SIZE
            end   = min(start + PAGE_SIZE, n_total)
            for idx in range(start, end):
                entry    = st.session_state.entries[idx]
                expanded = (idx == st.session_state.expand_idx)
                with st.expander(
                    f"Person {idx + 1}  —  {entry['name'] or '(unnamed)'}",
                    expanded=expanded,
                ):
                    c1, c2 = st.columns([2, 1])
                    with c1:
                        entry["name"]  = st.text_input("Name", value=entry["name"], key=f"name_{idx}")
                        entry["label"] = st.text_input("Subtitle (optional)", value=entry["label"], key=f"lbl_{idx}")
                    with c2:
                        up = st.file_uploader("Photo", type=["jpg","jpeg","png","webp"], key=f"img_{idx}")
                        if up:
                            entry["image"] = up
                        if entry["image"]:
                            st.image(entry["image"], use_container_width=True)
                    if n_total > 1:
                        if st.button("Remove", key=f"rm_{idx}"):
                            remove_entry(idx)
                            st.rerun()
    page_idx = None   # tabs handled above

# Single-page case (no tabs rendered yet)
if n_pages == 1:
    for idx, entry in enumerate(st.session_state.entries):
        expanded = (idx == st.session_state.expand_idx)
        with st.expander(
            f"Person {idx + 1}  —  {entry['name'] or '(unnamed)'}",
            expanded=expanded,
        ):
            c1, c2 = st.columns([2, 1])
            with c1:
                entry["name"]  = st.text_input("Name", value=entry["name"], key=f"name_{idx}")
                entry["label"] = st.text_input("Subtitle (optional)", value=entry["label"], key=f"lbl_{idx}")
            with c2:
                up = st.file_uploader("Photo", type=["jpg","jpeg","png","webp"], key=f"img_{idx}")
                if up:
                    entry["image"] = up
                if entry["image"]:
                    st.image(entry["image"], use_container_width=True)
            if n_total > 1:
                if st.button("Remove", key=f"rm_{idx}"):
                    remove_entry(idx)
                    st.rerun()

add_col, _ = st.columns([1, 4])
with add_col:
    if n_total < MAX_PEOPLE:
        st.button(f"+ Add person  ({n_total}/{MAX_PEOPLE})", on_click=add_entry)
    else:
        st.warning(f"Maximum of {MAX_PEOPLE} people reached.")

# ── Step 2: Audio ─────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("Step 2 — Upload audio")
audio_file = st.file_uploader("Audio track (MP3, WAV, AAC, M4A)", type=["mp3","wav","aac","m4a"])
if audio_file:
    st.audio(audio_file)

# ── Step 3: Generate ──────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("Step 3 — Generate video")

# Slide preview summary
valid = [e for e in st.session_state.entries if e["name"].strip() and e["image"] is not None]
if valid:
    groups = list(chunked(valid, per_slide))
    with st.expander(f"Slide preview — {len(groups)} slide(s)", expanded=False):
        for si, grp in enumerate(groups):
            names = ", ".join(e["name"] for e in grp)
            st.markdown(f"**Slide {si+1}:** {names}")

if st.button("🎬 Create MP4", type="primary"):
    if not valid:
        st.error("Add at least one person with a name and photo.")
    elif not audio_file:
        st.error("Please upload an audio file.")
    else:
        tmp_dir = tempfile.mkdtemp()
        try:
            with st.spinner(f"Rendering {len(list(chunked(valid, per_slide)))} slide(s) for {len(valid)} people…"):
                for i, entry in enumerate(valid):
                    raw       = entry["image"]
                    img_bytes = raw.getvalue() if hasattr(raw, "getvalue") else raw.read()
                    img_path  = os.path.join(tmp_dir, f"person_{i}.jpg")
                    with open(img_path, "wb") as f:
                        f.write(img_bytes)
                    entry["img_path"] = img_path

                slide_groups = list(chunked(valid, per_slide))
                clips        = []
                progress     = st.progress(0)

                for si, group in enumerate(slide_groups):
                    frame = make_slide_frame(group, img_shape)
                    clip  = ImageClip(frame, duration=slide_duration).with_fps(fps)
                    if fade_duration > 0:
                        from moviepy.video.fx import CrossFadeIn, CrossFadeOut
                        clip = clip.with_effects([CrossFadeIn(fade_duration), CrossFadeOut(fade_duration)])
                    clips.append(clip)
                    progress.progress((si + 1) / len(slide_groups))

                video          = concatenate_videoclips(clips, method="compose")
                total_duration = video.duration

                audio_path = os.path.join(tmp_dir, "audio" + Path(audio_file.name).suffix)
                with open(audio_path, "wb") as f:
                    f.write(audio_file.getvalue())

                audio_clip = AudioFileClip(audio_path)
                if loop_audio and audio_clip.duration < total_duration:
                    from moviepy import concatenate_audioclips
                    loops      = int(np.ceil(total_duration / audio_clip.duration))
                    audio_clip = concatenate_audioclips([audio_clip] * loops)

                audio_clip = audio_clip.with_duration(total_duration)
                video      = video.with_audio(audio_clip)

                out_path = os.path.join(tmp_dir, "slideshow.mp4")
                video.write_videofile(
                    out_path, fps=fps, codec="libx264", audio_codec="aac",
                    temp_audiofile=os.path.join(tmp_dir, "tmp_audio.m4a"),
                    remove_temp=True, logger=None,
                )

            st.success(f"Done — {len(slide_groups)} slide(s), {len(valid)} people, {total_duration:.0f}s")
            with open(out_path, "rb") as f:
                video_bytes = f.read()

            st.video(video_bytes)
            st.download_button("⬇️ Download MP4", video_bytes, "slideshow.mp4", "video/mp4")
            st.info(
                "**Embed in PowerPoint:**  Insert → Video → Video on My PC (Windows) "
                "or Insert → Movie (Mac) → select `slideshow.mp4`"
            )

        except Exception as e:
            st.error(f"Error: {e}")
            raise
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)
