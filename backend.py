# ============================================================
#  AI Khmer Dubbing PRO — Web Backend
#  FastAPI server — by sem samnang
#  Start: double-click setup.bat or run: python backend.py
# ============================================================

import os
import sys
import re
import glob
import uuid
import shutil
import asyncio
import subprocess
import threading

from datetime import timedelta
from pathlib import Path

import whisper
import srt
import edge_tts

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from deep_translator import GoogleTranslator

# ============================================================
# PATHS
# ============================================================

BASE_DIR    = Path(__file__).parent
FFMPEG_EXE  = str(BASE_DIR / "ffmpeg" / "ffmpeg.exe")
FFPROBE_EXE = str(BASE_DIR / "ffmpeg" / "ffprobe.exe")

if not os.path.exists(FFMPEG_EXE):
    FFMPEG_EXE  = "ffmpeg"
    FFPROBE_EXE = "ffprobe"

WORK_DIR  = BASE_DIR / "_jobs"
MODEL_DIR = BASE_DIR / "models"
OUT_DIR   = BASE_DIR / "4_final_output"

for d in [WORK_DIR, MODEL_DIR, OUT_DIR]:
    d.mkdir(exist_ok=True)

CREATION_FLAGS = 0
if sys.platform == "win32":
    CREATION_FLAGS = subprocess.CREATE_NO_WINDOW

# ============================================================
# JOB STORE
# ============================================================

jobs: dict[str, dict] = {}

# ============================================================
# HELPERS
# ============================================================

def is_valid_subtitle(text: str) -> bool:
    if not text:
        return False
    text = text.strip()
    if len(text) <= 1:
        return False
    low = text.lower()
    bad_tokens = ["♪","♫","[music]","[音乐]","(music)","music","la la","oh","ah","mmm","hmm"]
    for b in bad_tokens:
        if b.lower() in low:
            return False
    if len(set(text)) <= 2:
        return False
    return True


def clean_subtitles(subs):
    cleaned = []
    last_text = ""
    for s in subs:
        txt = s.content.strip()
        if txt == last_text:
            continue
        if len(txt) <= 1:
            continue
        if last_text and txt in last_text:
            continue
        cleaned.append(s)
        last_text = txt
    return cleaned


def jlog(job_id: str, msg: str):
    if job_id in jobs:
        jobs[job_id]["logs"].append(msg)
    print(f"[{job_id[:8]}] {msg}")


def jset(job_id: str, **kwargs):
    if job_id in jobs:
        jobs[job_id].update(kwargs)


def run_ffmpeg_via_bat(cmd: str, bat_path: Path) -> subprocess.CompletedProcess:
    """Fix: write ffmpeg command to .bat to avoid 'command line too long' error"""
    bat_content = f"@echo off\n{cmd}\n"
    with open(bat_path, "w", encoding="utf-8") as f:
        f.write(bat_content)

    return subprocess.run(
        f'"{bat_path}"',
        shell=True,
        capture_output=True,
        text=True,
        creationflags=CREATION_FLAGS,
    )

# ============================================================
# PIPELINE
# ============================================================

def run_pipeline(job_id: str):
    job        = jobs[job_id]
    video_path = job["video_path"]
    voice      = job["voice"]
    model_name = job["model"]
    keep_bgm   = job["keep_bgm"]

    base     = Path(video_path).stem
    filename = Path(video_path).name
    job_dir  = WORK_DIR / job_id

    srt_zh   = job_dir / f"{base}.srt"
    srt_kh   = job_dir / f"{base}_kh.srt"
    bg_audio = job_dir / "bg.wav"
    seg_dir  = job_dir / "segments"
    seg_dir.mkdir(exist_ok=True)

    try:
        # ---- STEP 1: Whisper --------------------------------
        jset(job_id, step=1, status="Speech recognition...", progress=5)
        jlog(job_id, "\n[ STEP 1 ] Whisper — loading model...")

        model = whisper.load_model(model_name, download_root=str(MODEL_DIR))

        result = model.transcribe(
            video_path,
            language="zh",
            fp16=False,
            verbose=False,
            temperature=0.0,
            initial_prompt="只转录正常人类对话。",
            condition_on_previous_text=False,
            no_speech_threshold=0.7,
            logprob_threshold=-1.0,
            compression_ratio_threshold=1.8,
        )

        subs_zh = []
        for i, seg in enumerate(result["segments"], start=1):
            text = seg["text"].strip()
            if not is_valid_subtitle(text):
                continue
            subs_zh.append(srt.Subtitle(
                index=i,
                start=timedelta(seconds=float(seg["start"])),
                end=timedelta(seconds=float(seg["end"])),
                content=text,
            ))

        subs_zh = clean_subtitles(subs_zh)

        with open(srt_zh, "w", encoding="utf-8") as f:
            f.write(srt.compose(subs_zh))

        jlog(job_id, f"Chinese SRT saved ({len(subs_zh)} lines)")

        # ---- STEP 1.5: BGM ----------------------------------
        if keep_bgm:
            jset(job_id, step=2, status="Extracting BGM...", progress=15)
            jlog(job_id, "\n[ STEP 1.5 ] Extract Background Music")
            bg_cmd = (
                f'"{FFMPEG_EXE}" -y -i "{video_path}" -vn '
                f'-af "pan=stereo|c0=0.5*c0-0.5*c1|c1=0.5*c1-0.5*c0" '
                f'-ar 44100 "{bg_audio}"'
            )
            subprocess.run(bg_cmd, shell=True, creationflags=CREATION_FLAGS)
            jlog(job_id, "Background music extracted.")

        # ---- STEP 2: Translate ------------------------------
        jset(job_id, step=3, status="Translating...", progress=20)
        jlog(job_id, "\n[ STEP 2 ] Translation — Chinese → Khmer")

        with open(srt_zh, "r", encoding="utf-8") as f:
            subs = list(srt.parse(f.read()))

        total = len(subs)

        for i, sub in enumerate(subs):
            try:
                km = GoogleTranslator(source="zh-CN", target="km").translate(sub.content.strip())
                sub.content = km
                jlog(job_id, f"  [{i+1}/{total}] {km}")
            except Exception as e:
                jlog(job_id, f"  Translation error [{i+1}]: {e}")
            jset(job_id, progress=20 + (i + 1) / total * 25)

        with open(srt_kh, "w", encoding="utf-8") as f:
            f.write(srt.compose(subs))

        jlog(job_id, f"Khmer SRT saved ({len(subs)} lines)")

        # ---- STEP 3: TTS ------------------------------------
        jset(job_id, step=4, status="Khmer TTS...", progress=45)
        jlog(job_id, "\n[ STEP 3 ] Khmer Text-to-Speech")

        tts_voice = (
            "km-KH-SreymomNeural" if voice == "female"
            else "km-KH-PisethNeural"
        )

        sem      = asyncio.Semaphore(3)
        done_tts = {"n": 0}

        async def audio_dur(path):
            p = await asyncio.create_subprocess_shell(
                f'"{FFPROBE_EXE}" -v error -show_entries format=duration '
                f'-of default=noprint_wrappers=1:nokey=1 "{path}"',
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                creationflags=CREATION_FLAGS,
            )
            out, _ = await p.communicate()
            try:
                return float(out.decode().strip())
            except:
                return 0.0

        async def make_voice(sub):
            text = sub.content.strip()
            if not text:
                return
            ms         = int(sub.start.total_seconds() * 1000)
            target_dur = sub.end.total_seconds() - sub.start.total_seconds()
            out_mp3    = seg_dir / f"{sub.index:04d}_{ms}.mp3"
            tmp_mp3    = seg_dir / f"tmp_{sub.index:04d}.mp3"

            async with sem:
                try:
                    rate = "+0%"
                    l = len(text)
                    if l > 120:  rate = "+35%"
                    elif l > 80: rate = "+25%"
                    elif l > 50: rate = "+15%"
                    elif l < 15: rate = "-10%"

                    tts = edge_tts.Communicate(text, tts_voice, rate=rate)
                    await tts.save(str(tmp_mp3))

                    tts_dur = await audio_dur(str(tmp_mp3))

                    if tts_dur <= 0:
                        shutil.copy(str(tmp_mp3), str(out_mp3))
                    else:
                        speed = tts_dur / target_dur
                        speed = max(0.55, min(speed, 2.2))
                        filters = []
                        while speed > 2.0:
                            filters.append("atempo=2.0")
                            speed /= 2.0
                        while speed < 0.5:
                            filters.append("atempo=0.5")
                            speed /= 0.5
                        filters.append(f"atempo={speed:.4f}")
                        ff_cmd = (
                            f'"{FFMPEG_EXE}" -y -i "{tmp_mp3}" '
                            f'-filter:a "{",".join(filters)}" '
                            f'-ar 44100 "{out_mp3}" -loglevel error'
                        )
                        p = await asyncio.create_subprocess_shell(
                            ff_cmd, creationflags=CREATION_FLAGS
                        )
                        await p.communicate()

                    if tmp_mp3.exists():
                        tmp_mp3.unlink()

                except Exception as e:
                    jlog(job_id, f"  TTS sub {sub.index}: {e}")

            done_tts["n"] += 1
            jset(job_id, progress=45 + done_tts["n"] / total * 40)

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(asyncio.gather(*[make_voice(s) for s in subs]))
        loop.close()

        # ---- STEP 4: FFmpeg Merge ---------------------------
        jset(job_id, step=5, status="Merging video...", progress=88)
        jlog(job_id, "\n[ STEP 4 ] FFmpeg Merge")

        segments = sorted(glob.glob(str(seg_dir / "*.mp3")))
        if not segments:
            raise RuntimeError("No audio segments generated")

        jlog(job_id, f"  Mixing {len(segments)} audio segments...")

        mixed_audio = job_dir / "mixed_audio.wav"
        filter_txt  = job_dir / "_filter.txt"
        n           = len(segments)

        fc = ""
        for i, seg in enumerate(segments):
            m    = re.search(r"_(\d+)\.mp3$", seg)
            ms_v = m.group(1) if m else "0"
            fc  += f"[{i}:a]adelay={ms_v}|{ms_v}[a{i}];\n"

        mix  = "".join(f"[a{i}]" for i in range(n))
        fc  += f"{mix}amix=inputs={n}:duration=longest:normalize=0[outa]"

        with open(filter_txt, "w", encoding="utf-8") as f:
            f.write(fc)

        inputs_str = " ".join(f'-i "{x}"' for x in segments)
        mix_cmd = (
            f'"{FFMPEG_EXE}" -y {inputs_str} '
            f'-filter_complex_script "{filter_txt}" '
            f'-map "[outa]" -ar 44100 "{mixed_audio}" -loglevel error'
        )

        mix_bat = job_dir / "_mix.bat"
        proc = run_ffmpeg_via_bat(mix_cmd, mix_bat)

        if proc.returncode != 0:
            jlog(job_id, proc.stderr[-1000:])
            raise RuntimeError("FFmpeg audio mix failed")

        jlog(job_id, "  Audio mix done. Merging with video...")

        final_out  = OUT_DIR / f"Khmer_{filename}"
        final_bat  = job_dir / "_final.bat"

        if keep_bgm:
            final_cmd = (
                f'"{FFMPEG_EXE}" -y '
                f'-i "{video_path}" '
                f'-i "{mixed_audio}" '
                f'-i "{bg_audio}" '
                f'-filter_complex '
                f'"[1:a]volume=1.0[dub];[2:a]volume=0.35[bg];[dub][bg]amix=inputs=2:duration=longest:normalize=0[outa]" '
                f'-map 0:v -map "[outa]" '
                f'-c:v copy -c:a aac -b:a 192k '
                f'"{final_out}" -loglevel error'
            )
        else:
            final_cmd = (
                f'"{FFMPEG_EXE}" -y '
                f'-i "{video_path}" '
                f'-i "{mixed_audio}" '
                f'-map 0:v -map 1:a '
                f'-c:v copy -c:a aac -b:a 192k '
                f'"{final_out}" -loglevel error'
            )

        proc2 = run_ffmpeg_via_bat(final_cmd, final_bat)

        if proc2.returncode != 0:
            jlog(job_id, proc2.stderr[-1000:])
            raise RuntimeError("FFmpeg final merge failed")

        jset(
            job_id,
            state="done",
            progress=100,
            status="DONE!",
            output=str(final_out),
            output_filename=f"Khmer_{filename}",
        )
        jlog(job_id, "\n✅ DUBBING COMPLETE!")
        jlog(job_id, str(final_out))

    except Exception as exc:
        jset(job_id, state="error", error=str(exc))
        jlog(job_id, f"\nERROR: {exc}")

# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(title="AI Khmer Dubbing PRO")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def index():
    return FileResponse(str(BASE_DIR / "index.html"))

@app.get("/ping")
def ping():
    return {"ok": True}

@app.post("/upload")
async def upload(
    video:    UploadFile = File(...),
    voice:    str        = Form("female"),
    model:    str        = Form("base"),
    keep_bgm: str        = Form("1"),
):
    job_id  = str(uuid.uuid4())
    job_dir = WORK_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    suffix     = Path(video.filename).suffix
    video_path = str(job_dir / f"input{suffix}")

    with open(video_path, "wb") as f:
        content = await video.read()
        f.write(content)

    jobs[job_id] = {
        "state":           "running",
        "step":            0,
        "progress":        0,
        "status":          "Uploading...",
        "logs":            [],
        "video_path":      video_path,
        "voice":           voice,
        "model":           model,
        "keep_bgm":        keep_bgm == "1",
        "output":          None,
        "output_filename": None,
        "error":           None,
    }

    t = threading.Thread(target=run_pipeline, args=(job_id,), daemon=True)
    t.start()

    return {"job_id": job_id}


@app.get("/status/{job_id}")
def status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job  = jobs[job_id]
    logs = job.get("logs", []).copy()
    job["logs"] = []

    return {
        "state":           job["state"],
        "step":            job["step"],
        "progress":        job["progress"],
        "status":          job["status"],
        "logs":            logs,
        "output":          job.get("output"),
        "output_filename": job.get("output_filename"),
        "error":           job.get("error"),
    }


# ---- Download endpoint: optimized for all devices (iPhone, Android, PC) --------
@app.get("/download/{job_id}")
def download(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job         = jobs[job_id]
    output_path = job.get("output")
    filename    = job.get("output_filename", "Khmer_output.mp4")

    if not output_path or not Path(output_path).exists():
        raise HTTPException(status_code=404, detail="File not ready")

    # Get file size for download progress
    file_size = Path(output_path).stat().st_size

    return FileResponse(
        path=output_path,
        filename=filename,
        media_type="video/mp4",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
            "Content-Length": str(file_size),
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )


# ============================================================
# ENTRY
# ============================================================

if __name__ == "__main__":
    import uvicorn
    print("=" * 60)
    print("  AI Khmer Dubbing PRO — Backend Server")
    print("  http://localhost:8000")
    print("  http://192.168.1.100:8000  (LAN / Phone)")
    print("=" * 60)
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
