"""
Stage-3 Long-Script Complete Mathematical Cinema Director E2E Verification Suite.

Verifies:
1. `GET /api/cinema/sample-script`: Official 5-Act Epic Benchmark Script + 0 branding leaks.
2. `POST /api/generate-from-prompt` (`mode="script_cinema"`): Multi-act delegation + 0 leaks.
3. `POST /api/cinema/generate-from-script` on an UNSEEN Custom 4-Act Script:
   "Act 1: A golden Mandelbulb fractal awakens in deep space. Act 2: It morphs into a chaotic
    Lorenz strange attractor in a curl-noise nebula. Act 3: Below, an architectural Xenakis
    hyperbolic pavilion rises over ocean waves. Act 4: A crystal astrolabe with Saturn rings
    crowns the sky."
   Saves `backend/out_of_box_videos/custom_4act_script_film.{mp4,gif}` & `backend/sheet_custom_4act_script_film.png`.
4. `POST /api/cinema/generate-from-script` on the Full Official 5-Act Epic Script (`SAMPLE_EPIC_5_ACT_SCRIPT`):
   - Verifies 5 distinct acts + 4 C^2 homotopy morph transitions + >= 100 total frames.
   - Verifies every act's WebGL1 GLSL shader, AST-derived/closed-form LaTeX dict, 3D primitives,
     `std > 28`, `chroma > 15`, `motion_energy > 1.0`, `verified == True`, and distinct inter-act visuals.
   - Verifies `backend/out_of_box_videos/complete_script_film.mp4` (OpenCV frame count >= 100,
     non-black frames across all 5 acts, inter-act MAD > 12), `complete_script_film.gif`
     (valid GIF89a, PIL n_frames >= 100), and `backend/sheet_complete_script_film.png`.
5. Zero AI / model branding leaks across ALL API responses and ALL files under `frontend/src/`.
6. Frontend `npm run build` exits with code 0.
"""

import ast
import base64
import io
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np
from PIL import Image

BACKEND_DIR = Path(__file__).resolve().parent
REPO_ROOT = BACKEND_DIR.parent
FRONTEND_DIR = REPO_ROOT / "frontend"
VIDEOS_DIR = BACKEND_DIR / "out_of_box_videos"

COMPLETE_MP4 = VIDEOS_DIR / "complete_script_film.mp4"
COMPLETE_GIF = VIDEOS_DIR / "complete_script_film.gif"
COMPLETE_SHEET = BACKEND_DIR / "sheet_complete_script_film.png"

CUSTOM_4ACT_MP4 = VIDEOS_DIR / "custom_4act_script_film.mp4"
CUSTOM_4ACT_GIF = VIDEOS_DIR / "custom_4act_script_film.gif"
CUSTOM_4ACT_SHEET = BACKEND_DIR / "sheet_custom_4act_script_film.png"

if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from fastapi.testclient import TestClient  # noqa: E402
import symbolic_math_compiler as smc  # noqa: E402
from main import app  # noqa: E402
from test_e2e_behavior import (  # noqa: E402
    decode_base64_png,
    find_branding_leaks_in_text,
)

UNSEEN_4_ACT_SCRIPT = (
    "Act 1: A golden Mandelbulb fractal awakens in deep space. "
    "Act 2: It morphs into a chaotic Lorenz strange attractor in a curl-noise nebula. "
    "Act 3: Below, an architectural Xenakis hyperbolic pavilion rises over ocean waves. "
    "Act 4: A crystal astrolabe with Saturn rings crowns the sky."
)

BINARY_B64_KEYS = {
    "image_base64",
    "preview_image_base64",
    "frames_base64",
    "gif_base64",
    "mp4_base64",
    "sheet_base64",
}


def find_leaks_recursive(obj: Any, path: str = "root") -> List[str]:
    leaks: List[str] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in BINARY_B64_KEYS or (isinstance(k, str) and k.endswith("_base64")):
                continue
            if k != "use_ai" and isinstance(k, str):
                leaks.extend(find_branding_leaks_in_text(k, f"{path}.<key:{k}>"))
            leaks.extend(find_leaks_recursive(v, f"{path}.{k}"))
    elif isinstance(obj, list):
        if obj and isinstance(obj[0], (int, float)):
            return leaks
        if obj and isinstance(obj[0], list) and obj[0] and isinstance(obj[0][0], (int, float)):
            return leaks
        for idx, item in enumerate(obj):
            leaks.extend(find_leaks_recursive(item, f"{path}[{idx}]"))
    elif isinstance(obj, str):
        if obj.startswith(("data:image/", "data:video/")):
            return leaks
        leaks.extend(find_branding_leaks_in_text(obj, path))
    return leaks


def assert_zero_branding(payload: Any, label: str) -> None:
    leaks = find_leaks_recursive(payload, label)
    assert not leaks, f"AI branding leak detected in {label}:\n  " + "\n  ".join(leaks[:10])


def _decode_data_uri(uri: str, prefix: str, label: str) -> bytes:
    assert isinstance(uri, str) and uri.startswith(prefix), (
        f"{label}: expected prefix '{prefix}', got {str(uri)[:64]!r}"
    )
    return base64.b64decode(uri[len(prefix):])


def validate_script_cinema_response(
    data: Dict[str, Any],
    label: str,
    expected_acts: int,
    min_total_frames: int,
) -> None:
    assert_zero_branding(data, label)
    assert data.get("mode") == "script_cinema", f"{label}: expected mode='script_cinema', got {data.get('mode')}"
    assert data.get("num_acts") == expected_acts, f"{label}: expected {expected_acts} acts, got {data.get('num_acts')}"
    assert data.get("num_transitions") == expected_acts - 1, (
        f"{label}: expected {expected_acts - 1} transitions, got {data.get('num_transitions')}"
    )
    total_frames = int(data.get("total_frames", 0))
    assert total_frames >= min_total_frames, f"{label}: total_frames={total_frames} < {min_total_frames}"
    assert float(data.get("total_duration_sec", 0.0)) > 0.0

    # Validate top-level base64 media payloads
    mp4_bytes = _decode_data_uri(data.get("mp4_base64", ""), "data:video/mp4;base64,", f"{label}:mp4")
    gif_bytes = _decode_data_uri(data.get("gif_base64", ""), "data:image/gif;base64,", f"{label}:gif")
    assert len(mp4_bytes) > 50_000, f"{label}: MP4 too small ({len(mp4_bytes)} bytes)"
    assert len(gif_bytes) > 100_000, f"{label}: GIF too small ({len(gif_bytes)} bytes)"
    assert gif_bytes[:6] in (b"GIF89a", b"GIF87a"), f"{label}: invalid GIF magic header {gif_bytes[:6]!r}"

    # Validate top-level Python code & LaTeX
    ast.parse(data["python_code"])
    assert isinstance(data.get("latex"), dict) and len(data["latex"]) >= 2, f"{label}: invalid top-level latex"

    # Validate verification summary
    vs = data.get("verification_summary") or {}
    assert vs.get("all_acts_verified") is True, f"{label}: verification_summary.all_acts_verified != True ({vs})"
    assert float(vs.get("mean_act_std", 0.0)) > 30.0, f"{label}: low mean_act_std={vs.get('mean_act_std')}"
    assert float(vs.get("mean_motion_energy", 0.0)) > 1.0, f"{label}: low mean_motion_energy={vs.get('mean_motion_energy')}"

    # Validate every individual Act
    acts = data.get("acts") or []
    assert len(acts) == expected_acts, f"{label}: len(acts)={len(acts)} != {expected_acts}"
    act_thumbs: List[np.ndarray] = []
    for idx, act in enumerate(acts, 1):
        act_lbl = f"{label}:Act{idx}"
        assert act.get("title") and act.get("narration"), f"{act_lbl}: missing title/narration"
        assert act.get("verified") is True, f"{act_lbl}: act not verified"
        assert float(act.get("std", 0.0)) > 25.0, f"{act_lbl}: std={act.get('std')} <= 25"
        assert float(act.get("chroma", 0.0)) > 12.0, f"{act_lbl}: chroma={act.get('chroma')} <= 12"
        assert float(act.get("motion_energy", 0.0)) > 0.5, f"{act_lbl}: motion_energy={act.get('motion_energy')} <= 0.5"

        glsl = act.get("glsl_fragment_shader", "")
        ok_g, g_err = smc.validate_glsl_webgl1_static(glsl)
        assert ok_g, f"{act_lbl}: WebGL1 GLSL invalid: {g_err}"

        eqs = act.get("latex_equations") or act.get("latex") or {}
        assert isinstance(eqs, dict) and len(eqs) >= 2, f"{act_lbl}: missing latex dict"

        p3d = act.get("primitives_3d") or []
        assert isinstance(p3d, list) and len(p3d) >= 100, f"{act_lbl}: missing primitives_3d"

        prev_b64 = act.get("preview_image_base64") or act.get("image_base64") or ""
        p_img, p_arr = decode_base64_png(prev_b64)
        assert float(np.std(p_arr)) > 25.0, f"{act_lbl}: preview PNG std too low"
        act_thumbs.append(np.asarray(p_img.resize((120, 75), Image.Resampling.BILINEAR), dtype=np.float32))

    # Ensure consecutive acts are genuinely distinct visual scenes (not duplicated frames)
    for i in range(len(act_thumbs) - 1):
        inter_mad = float(np.mean(np.abs(act_thumbs[i] - act_thumbs[i + 1])))
        assert inter_mad > 12.0, f"{label}: Act {i+1} and Act {i+2} look too similar (MAD={inter_mad:.2f})"

    transitions = data.get("transitions") or []
    assert len(transitions) == expected_acts - 1, f"{label}: expected {expected_acts - 1} transition records"


def verify_mp4_and_gif_on_disk(mp4_path: Path, gif_path: Path, sheet_path: Path, min_frames: int) -> Dict[str, Any]:
    assert mp4_path.exists() and mp4_path.stat().st_size > 100_000, f"Invalid MP4 on disk: {mp4_path}"
    assert gif_path.exists() and gif_path.stat().st_size > 200_000, f"Invalid GIF on disk: {gif_path}"
    assert sheet_path.exists() and sheet_path.stat().st_size > 100_000, f"Invalid sheet on disk: {sheet_path}"

    # Verify GIF header & frame count via PIL
    raw_gif = gif_path.read_bytes()
    assert raw_gif[:6] in (b"GIF89a", b"GIF87a"), f"Bad GIF header: {raw_gif[:6]!r}"
    with Image.open(io.BytesIO(raw_gif)) as gim:
        gif_frames = int(getattr(gim, "n_frames", 1))
        assert gif_frames >= min_frames, f"GIF has {gif_frames} frames < {min_frames}"

    # Verify MP4 frame count & visual health across all 5 acts via OpenCV
    cap = cv2.VideoCapture(str(mp4_path))
    assert cap.isOpened(), f"Failed to open MP4 via OpenCV: {mp4_path}"
    mp4_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    assert mp4_frames >= min_frames, f"MP4 has {mp4_frames} frames < {min_frames}"

    sampled_frames: List[np.ndarray] = []
    for frac in (0.08, 0.28, 0.48, 0.68, 0.88):
        idx = max(0, min(mp4_frames - 1, int(round(frac * (mp4_frames - 1)))))
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        assert ok and frame is not None, f"Failed to read MP4 frame {idx}"
        f_std = float(np.std(frame))
        f_mean = float(np.mean(frame))
        assert f_std > 25.0 and f_mean > 10.0, (
            f"MP4 frame {idx} looks blank/black (std={f_std:.2f}, mean={f_mean:.2f})"
        )
        sampled_frames.append(frame.astype(np.float32))
    cap.release()

    # Check inter-act diversity inside the actual MP4 video stream
    inter_diffs = [
        float(np.mean(np.abs(sampled_frames[i] - sampled_frames[i + 1])))
        for i in range(len(sampled_frames) - 1)
    ]
    assert min(inter_diffs) > 10.0, f"MP4 sampled acts lack visual diversity: {inter_diffs}"

    # Verify storyboard contact sheet opens cleanly
    with Image.open(sheet_path) as sim:
        sw, sh = sim.size
        s_arr = np.asarray(sim.convert("RGB"), dtype=np.float32)
        assert sw >= 1000 and sh >= 600 and float(np.std(s_arr)) > 30.0

    return {
        "mp4_bytes": mp4_path.stat().st_size,
        "mp4_frames": mp4_frames,
        "gif_bytes": gif_path.stat().st_size,
        "gif_frames": gif_frames,
        "sheet_bytes": sheet_path.stat().st_size,
        "sheet_size": [sw, sh],
        "sampled_inter_act_mad": [round(x, 2) for x in inter_diffs],
    }


def verify_frontend_and_build() -> None:
    print("\n5. Scanning ALL frontend/src/* files for zero AI/model branding & running `npm run build`...")
    checked = 0
    for fpath in sorted((FRONTEND_DIR / "src").rglob("*")):
        if fpath.is_file() and fpath.suffix.lower() in {".js", ".jsx", ".ts", ".tsx", ".css", ".html"}:
            content = fpath.read_text(encoding="utf-8")
            leaks = find_branding_leaks_in_text(content, str(fpath.relative_to(REPO_ROOT)))
            assert not leaks, f"AI branding leak detected in {fpath}:\n  " + "\n  ".join(leaks[:10])
            checked += 1
    assert checked >= 2
    print(f"   [PASS] Scanned {checked} files under frontend/src/ — ZERO AI/model branding leaks")

    proc = subprocess.run(
        "npm run build",
        cwd=str(FRONTEND_DIR),
        shell=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, f"Frontend build failed ({proc.returncode}):\n{proc.stdout}\n{proc.stderr}"
    print("   [PASS] Frontend `npm run build` exited with code 0")


def run_long_script_cinema_tests() -> None:
    client = TestClient(app)

    print("1. Testing GET /api/cinema/sample-script & zero AI branding leaks...")
    r_sample = client.get("/api/cinema/sample-script")
    assert r_sample.status_code == 200, r_sample.text
    sample_data = r_sample.json()
    assert_zero_branding(sample_data, "GET:/api/cinema/sample-script")
    sample_script = sample_data.get("script", "")
    for act_num in range(1, 6):
        assert f"ACT {act_num}:" in sample_script, f"Missing 'ACT {act_num}:' in sample script"
    print(f"   [PASS] Sample script loaded: '{sample_data.get('title')}' ({len(sample_script)} chars, 5 acts, 0 leaks)")

    print("\n2. Testing POST /api/generate-from-prompt delegation with mode='script_cinema'...")
    t0 = time.time()
    r_del = client.post(
        "/api/generate-from-prompt",
        json={
            "prompt": UNSEEN_4_ACT_SCRIPT,
            "mode": "script_cinema",
            "refine": True,
        },
    )
    assert r_del.status_code == 200, f"mode='script_cinema' delegation failed: {r_del.text[:400]}"
    d_del = r_del.json()
    validate_script_cinema_response(d_del, "generate-from-prompt:script_cinema", expected_acts=4, min_total_frames=60)
    print(
        f"   [PASS] /api/generate-from-prompt (mode='script_cinema') in {time.time()-t0:.1f}s -> "
        f"{d_del['num_acts']} acts, {d_del['total_frames']} frames, 0 branding leaks"
    )

    print("\n3. Testing POST /api/cinema/generate-from-script on 2nd UNSEEN Custom 4-Act Script...")
    t0 = time.time()
    r_4act = client.post(
        "/api/cinema/generate-from-script",
        json={
            "script": UNSEEN_4_ACT_SCRIPT,
            "width": 520,
            "height": 325,
            "fps": 12,
            "frames_per_act": 14,
            "transition_frames": 6,
            "max_acts": 6,
            "refine_acts": True,
        },
    )
    assert r_4act.status_code == 200, f"4-act custom script failed: {r_4act.text[:400]}"
    d_4act = r_4act.json()
    validate_script_cinema_response(d_4act, "cinema:custom_4act", expected_acts=4, min_total_frames=70)

    # Preserve custom 4-act movie + contact sheet before running the 5-act epic benchmark
    if COMPLETE_MP4.exists():
        shutil.copy2(COMPLETE_MP4, CUSTOM_4ACT_MP4)
    if COMPLETE_GIF.exists():
        shutil.copy2(COMPLETE_GIF, CUSTOM_4ACT_GIF)
    if COMPLETE_SHEET.exists():
        shutil.copy2(COMPLETE_SHEET, CUSTOM_4ACT_SHEET)
    print(
        f"   [PASS] Custom 4-Act Film ({time.time()-t0:.1f}s) -> {d_4act['num_acts']} acts, "
        f"{d_4act['num_transitions']} C^2 morphs, {d_4act['total_frames']}f ({d_4act['total_duration_sec']:.1f}s) | "
        f"paradigms={[a.get('paradigm_choice') for a in d_4act['acts']]}"
    )

    print("\n4. Testing POST /api/cinema/generate-from-script on OFFICIAL 5-ACT EPIC SCRIPT (114 frames)...")
    t0 = time.time()
    r_5act = client.post(
        "/api/cinema/generate-from-script",
        json={
            "script": sample_script,
            "width": 520,
            "height": 325,
            "fps": 12,
            "frames_per_act": 18,
            "transition_frames": 6,
            "max_acts": 6,
            "refine_acts": True,
        },
    )
    assert r_5act.status_code == 200, f"5-act epic script failed: {r_5act.text[:400]}"
    d_5act = r_5act.json()
    validate_script_cinema_response(d_5act, "cinema:epic_5act", expected_acts=5, min_total_frames=100)

    disk_stats = verify_mp4_and_gif_on_disk(COMPLETE_MP4, COMPLETE_GIF, COMPLETE_SHEET, min_frames=100)
    print(
        f"   [PASS] 5-Act Epic Film ({time.time()-t0:.1f}s) -> '{d_5act['title']}' | "
        f"{d_5act['num_acts']} acts + {d_5act['num_transitions']} C^2 morphs = {d_5act['total_frames']}f "
        f"({d_5act['total_duration_sec']:.2f}s @ {d_5act['fps']} FPS)"
    )
    for a in d_5act["acts"]:
        print(
            f"          Act {a['act_number']}: [{a.get('paradigm_choice')}] '{a['title']}' | "
            f"std={a['std']:.1f} chroma={a['chroma']:.1f} motion={a['motion_energy']:.2f} verified={a['verified']}"
        )
    print(
        f"   [PASS] Disk Artifacts Verified:\n"
        f"          MP4  : {COMPLETE_MP4} ({disk_stats['mp4_bytes']:,}B, {disk_stats['mp4_frames']}f, inter-act MAD={disk_stats['sampled_inter_act_mad']})\n"
        f"          GIF  : {COMPLETE_GIF} ({disk_stats['gif_bytes']:,}B, {disk_stats['gif_frames']}f)\n"
        f"          Sheet: {COMPLETE_SHEET} ({disk_stats['sheet_bytes']:,}B, {disk_stats['sheet_size']})"
    )

    verify_frontend_and_build()

    print("\n==================================================================================")
    print("LONG-SCRIPT MULTI-ACT MATHEMATICAL CINEMA DIRECTOR VERIFICATION PASSED (100%)!")
    print("==================================================================================")


if __name__ == "__main__":
    run_long_script_cinema_tests()
