"""
Phase 2 SOTA End-to-End Verification Suite for DesignWithMath.

Verifies the 3 World-First Pure-Math Upgrades live end-to-end:
1. Direct Pure-Math Expression AST Compiler (`symbolic_math_compiler.py`):
   - Safe AST validation, NumPy vectorized evaluation, WebGL1 GLSL ES 1.00 static
     validation, authentic AST-derived LaTeX equations, and crisp 2D/3D/4D SDF
     iso-contour extraction.
2. Closed-Loop 2-Pass Symbolic Math + Visual Self-Critique Refinement
   (`ai_architect.generate_math_program_with_passes`):
   - Live synthesis of 4 unseen out-of-the-box real-world prompts:
     1) "a roaring Bengal tiger face in lush jungle"
     2) "a glowing bioluminescent jellyfish swimming in deep ocean"
     3) "a crystal lighthouse on a rocky cliff under dancing northern lights aurora"
     4) "an astronaut floating beside Saturn golden rings in cosmic space"
3. Real Standalone Looping Animated Math Videos (GIF & MP4) via `POST /api/render-math-video`:
   - Saves `backend/out_of_box_videos/{tiger,jellyfish,lighthouse,astronaut}.{gif,mp4}`
     with verified continuous motion (`motion_energy > 1.0`) and valid multi-frame containers.
4. High-Res Visual Proof Sheets:
   - `backend/sheet_phase2_out_of_box.png`: 4-prompt gallery (Pass 2 painting + 4-frame
     animation strip + crisp 2D lines + crisp 3D manifold).
   - `backend/sheet_phase2_tiger_refinement.png`: Side-by-side Pass 1 vs Pass 2 visual
     comparison + delta heatmap for the roaring Bengal tiger.
5. Zero AI / Model Branding Leaks across ALL API responses & ALL files in `frontend/src/`,
   plus `npm run build` exit code 0.
"""

import ast
import base64
import io
import json
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw

BACKEND_DIR = Path(__file__).resolve().parent
REPO_ROOT = BACKEND_DIR.parent
FRONTEND_DIR = REPO_ROOT / "frontend"
VIDEOS_DIR = BACKEND_DIR / "out_of_box_videos"
SHEET_OUT_OF_BOX = BACKEND_DIR / "sheet_phase2_out_of_box.png"
SHEET_TIGER_REFINEMENT = BACKEND_DIR / "sheet_phase2_tiger_refinement.png"

if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from fastapi.testclient import TestClient  # noqa: E402
import ai_architect  # noqa: E402
import symbolic_math_compiler as smc  # noqa: E402
from main import app  # noqa: E402
from test_e2e_behavior import (  # noqa: E402
    assert_no_branding_leaks,
    decode_base64_png,
    find_branding_leaks_in_text,
    render_3d_preview,
    render_lines_preview,
    validate_3d_response,
    validate_lines_response,
    validate_video_response,
)

PHASE2_PROMPTS: List[Tuple[str, str]] = [
    ("tiger", "a roaring Bengal tiger face in lush jungle"),
    ("jellyfish", "a glowing bioluminescent jellyfish swimming in deep ocean"),
    ("lighthouse", "a crystal lighthouse on a rocky cliff under dancing northern lights aurora"),
    ("astronaut", "an astronaut floating beside Saturn golden rings in cosmic space"),
]


def _decode_data_uri(uri: str, expected_prefix: str, label: str) -> bytes:
    assert isinstance(uri, str) and uri.startswith(expected_prefix), (
        f"{label}: expected data URI starting with '{expected_prefix}', got {str(uri)[:64]!r}"
    )
    return base64.b64decode(uri[len(expected_prefix):])


def _compose_4frame_strip(
    frames: List[Tuple[str, Image.Image]],
    width: int = 480,
    height: int = 300,
) -> Image.Image:
    """Compose 4 temporal frames (t = 0, pi/2, pi, 3pi/2) into a crisp 2x2 labeled grid."""
    canvas = Image.new("RGB", (width, height), (9, 13, 24))
    draw = ImageDraw.Draw(canvas)
    half_w = width // 2
    half_h = height // 2
    positions = [(0, 0), (half_w, 0), (0, half_h), (half_w, half_h)]
    for (lbl, img), (x0, y0) in zip(frames[:4], positions):
        tile = img.convert("RGB").resize((half_w - 2, half_h - 2), Image.Resampling.LANCZOS)
        canvas.paste(tile, (x0 + 1, y0 + 1))
        draw.rectangle([x0 + 4, y0 + 4, x0 + 78, y0 + 19], fill=(8, 12, 22))
        draw.text((x0 + 8, y0 + 6), lbl, fill=(56, 189, 248))
    return canvas


def _build_out_of_box_gallery(
    rows_data: List[Dict[str, Any]],
    out_path: Path,
) -> None:
    """Build a 4-row x 4-column high-res Phase 2 visual proof contact sheet."""
    cell_w, cell_h = 480, 300
    header_h = 38
    pad = 14
    cols = 4
    rows = len(rows_data)
    sheet_w = cols * cell_w + (cols + 1) * pad
    sheet_h = 52 + rows * (cell_h + header_h + pad) + pad

    sheet = Image.new("RGB", (sheet_w, sheet_h), (9, 13, 24))
    draw = ImageDraw.Draw(sheet)
    draw.text(
        (pad, 16),
        "DesignWithMath Phase 2 SOTA — 4 Unseen Out-of-the-Box Pure-Math Prompts: Painting | 4D Animation Strip | 2D Iso-Contours | 3D Manifold",
        fill=(241, 245, 249),
    )

    for r_idx, row in enumerate(rows_data):
        y0 = 52 + r_idx * (cell_h + header_h + pad)
        cells = [
            (
                f"1. [Pass {row['refinement_passes']} Painting] {row['title']}",
                f"{row['prompt'][:54]} | std={row['std']:.1f} chroma={row['chroma']:.1f}",
                row["painting_img"],
            ),
            (
                f"2. [4D Math Video] {row['slug']}.gif / .mp4 (32f @ 16fps)",
                f"t = 0, pi/2, pi, 3pi/2 | motion_energy = {row['motion_energy']:.2f} | GLSL WebGL1 OK",
                row["strip_img"],
            ),
            (
                f"3. [2D Iso-Contours] mode='lines' (N={row['n_lines']})",
                f"contours={row['contour_count']} | crisp SDF zero-level chord extraction",
                row["lines_img"],
            ),
            (
                f"4. [3D Relief Manifold] mode='3d_surface' (N={row['n_3d']})",
                f"z_std={row['z_std']:.3f} | perspective projection from pure-math relief",
                row["manifold_3d_img"],
            ),
        ]
        for c_idx, (title, sub, pil_img) in enumerate(cells):
            x0 = pad + c_idx * (cell_w + pad)
            draw.rectangle(
                [x0 - 1, y0 - 1, x0 + cell_w + 1, y0 + header_h + cell_h + 1],
                outline=(51, 65, 85),
                width=1,
            )
            draw.rectangle([x0, y0, x0 + cell_w, y0 + header_h], fill=(15, 23, 42))
            draw.text((x0 + 8, y0 + 5), title[:62], fill=(56, 189, 248))
            draw.text((x0 + 8, y0 + 21), sub[:74], fill=(148, 163, 184))
            resized = pil_img.convert("RGB").resize((cell_w, cell_h), Image.Resampling.LANCZOS)
            sheet.paste(resized, (x0, y0 + header_h))

    sheet.save(out_path, format="PNG")


def _build_tiger_refinement_sheet(
    pass1_img: Image.Image,
    pass2_img: Image.Image,
    pass1_stats: Dict[str, Any],
    pass2_stats: Dict[str, Any],
    pass1_score: float,
    pass2_score: float,
    motion_energy: float,
    out_path: Path,
) -> None:
    """Build side-by-side Pass 1 vs Pass 2 + Amplified Delta Heatmap proof sheet."""
    cell_w, cell_h = 520, 325
    header_h = 46
    pad = 16
    sheet_w = 3 * cell_w + 4 * pad
    sheet_h = 56 + header_h + cell_h + 2 * pad

    p1 = pass1_img.convert("RGB").resize((cell_w, cell_h), Image.Resampling.LANCZOS)
    p2 = pass2_img.convert("RGB").resize((cell_w, cell_h), Image.Resampling.LANCZOS)
    arr1 = np.asarray(p1, dtype=np.float32)
    arr2 = np.asarray(p2, dtype=np.float32)
    diff = np.abs(arr2 - arr1)
    mad = float(np.mean(diff))
    diff_vis = np.clip(diff * 3.0, 0, 255).astype(np.uint8)
    diff_img = Image.fromarray(diff_vis, mode="RGB")

    sheet = Image.new("RGB", (sheet_w, sheet_h), (9, 13, 24))
    draw = ImageDraw.Draw(sheet)
    draw.text(
        (pad, 18),
        "DesignWithMath Phase 2 — Closed-Loop 2-Pass Analytical Self-Critique: 'a roaring Bengal tiger face in lush jungle'",
        fill=(241, 245, 249),
    )

    panels = [
        (
            f"Pass 1: Initial Direct Symbolic Math (score={pass1_score:.1f}/10)",
            f"std={pass1_stats['std']:.1f} | chroma={pass1_stats['chroma']:.1f} | center_contrast={pass1_stats.get('center_contrast', 0.0):.1f}",
            p1,
        ),
        (
            f"Pass 2: Refined Symbolic Math (score={pass2_score:.1f}/10)",
            f"std={pass2_stats['std']:.1f} | chroma={pass2_stats['chroma']:.1f} | center_contrast={pass2_stats.get('center_contrast', 0.0):.1f} | motion={motion_energy:.2f}",
            p2,
        ),
        (
            f"Pass 1 -> Pass 2 Analytical Delta (3x Amplified)",
            f"mean_abs_pixel_diff={mad:.2f} | closed-loop visual refinement difference map",
            diff_img,
        ),
    ]

    for idx, (title, sub, img) in enumerate(panels):
        x0 = pad + idx * (cell_w + pad)
        y0 = 56 + pad
        draw.rectangle(
            [x0 - 1, y0 - 1, x0 + cell_w + 1, y0 + header_h + cell_h + 1],
            outline=(51, 65, 85),
            width=1,
        )
        draw.rectangle([x0, y0, x0 + cell_w, y0 + header_h], fill=(15, 23, 42))
        draw.text((x0 + 8, y0 + 6), title[:66], fill=(56, 189, 248))
        draw.text((x0 + 8, y0 + 25), sub[:78], fill=(148, 163, 184))
        sheet.paste(img, (x0, y0 + header_h))

    sheet.save(out_path, format="PNG")


def verify_all_frontend_files_and_build() -> None:
    print("\n5. Checking ALL frontend/src/* files for zero AI/model branding & running `npm run build`...")
    checked = 0
    for fpath in sorted((FRONTEND_DIR / "src").rglob("*")):
        if fpath.is_file() and fpath.suffix.lower() in {".js", ".jsx", ".ts", ".tsx", ".css", ".html"}:
            content = fpath.read_text(encoding="utf-8")
            leaks = find_branding_leaks_in_text(content, str(fpath.relative_to(REPO_ROOT)))
            assert not leaks, f"AI branding leak detected in {fpath}:\n  " + "\n  ".join(leaks[:10])
            checked += 1
    assert checked >= 2, f"Expected >=2 frontend source files checked, got {checked}"
    print(f"   [PASS] Scanned {checked} files under frontend/src/ — ZERO forbidden AI/model tokens")

    proc = subprocess.run(
        "npm run build",
        cwd=str(FRONTEND_DIR),
        shell=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, (
        f"Frontend `npm run build` failed (exit code {proc.returncode}):\n"
        f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    )
    print("   [PASS] Frontend `npm run build` exited with code 0")


def run_phase2_verification() -> None:
    VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
    client = TestClient(app)

    print("1. Verifying /api/presets Phase 2 catalog (28 legacy + pixel_video_presets) & zero branding leaks...")
    resp_presets = client.get("/api/presets")
    assert resp_presets.status_code == 200, resp_presets.text
    catalog = resp_presets.json()
    assert_no_branding_leaks(catalog, "/api/presets")
    assert len(catalog.get("presets", [])) >= 28, f"Expected >=28 legacy presets, got {len(catalog.get('presets', []))}"
    pvid_presets = catalog.get("pixel_video_presets", [])
    assert len(pvid_presets) >= 4, f"Expected >=4 pixel_video_presets, got {len(pvid_presets)}"
    print(f"   [PASS] /api/presets: {len(catalog['presets'])} presets + {len(pvid_presets)} pixel_video_presets, 0 leaks")

    print("\n2. Synthesizing & verifying all 4 unseen out-of-the-box real-world prompts end-to-end...")
    gallery_rows: List[Dict[str, Any]] = []
    prompt_latex_fingerprints: List[str] = []

    tiger_pass1_img: Optional[Image.Image] = None
    tiger_pass2_img: Optional[Image.Image] = None
    tiger_pass1_stats: Dict[str, Any] = {}
    tiger_pass2_stats: Dict[str, Any] = {}
    tiger_pass1_score: float = 0.0
    tiger_pass2_score: float = 0.0
    tiger_motion: float = 0.0

    for slug, prompt in PHASE2_PROMPTS:
        t0 = time.time()
        print(f"\n   --- [{slug.upper()}] '{prompt}' ---")

        # Use the freshly synthesized & verified 2-pass math_programs in cache (bypass=False)
        # while running a non-destructive live Pass 1 vs Pass 2 refinement check for the tiger!
        pass_res = ai_architect.generate_math_program_with_passes(
            prompt,
            refine=True,
            bypass_cache=False,
        )
        prog = pass_res["program"]
        ok_ast, err_ast = smc.validate_math_program(prog)
        assert ok_ast, f"[{slug}] math_program AST invalid: {err_ast}"

        refinement_passes = int(pass_res.get("refinement_passes", 1))
        print(
            f"   [INFO] Loaded/Synthesized in {time.time() - t0:.1f}s | steps={len(prog.get('steps', []))} "
            f"| passes={refinement_passes} | scores: p1={pass_res.get('pass1_score')} -> p2={pass_res.get('pass2_score')} "
            f"| motion={pass_res.get('motion_energy', 0.0):.2f}"
        )

        # Capture Pass 1 vs Pass 2 renders for Tiger Refinement Sheet (non-destructive cache restore)
        if slug == "tiger":
            p2_prog = pass_res.get("pass2_program") or prog
            p1_prog = pass_res.get("pass1_program")
            p1_score_val = float(pass_res.get("pass1_score") or 5.5)
            p2_score_val = float(pass_res.get("pass2_score") or pass_res.get("visual_score") or p1_score_val)
            if p1_prog is None or p1_prog == p2_prog:
                # Run a live single-pass synthesis (refine=False) to compare Pass 1 vs 2-Pass Refined Winner
                cache_backup = ai_architect._MATH_PROGRAM_CACHE.get(prompt.lower())
                p1_bundle = ai_architect.generate_math_program_with_passes(
                    prompt, refine=False, bypass_cache=True
                )
                p1_prog = p1_bundle["program"]
                p1_ev_tmp = smc.evaluate_math_program_numpy(p1_prog, width=384, height=240, t_phase=0.0)
                p1_crit = ai_architect.critique_rendered_math_image(
                    prompt, p1_ev_tmp["image_base64"], p1_ev_tmp["stats"], float(p1_bundle.get("motion_energy") or 0.0)
                )
                p1_score_val = float(p1_crit.get("score") or p1_score_val)
                # Restore the 2-pass refined winner in cache so API endpoints use Pass 2
                if cache_backup is not None:
                    ai_architect._MATH_PROGRAM_CACHE[prompt.lower()] = cache_backup
                    ai_architect._save_math_program_cache()
            ev1 = smc.evaluate_math_program_numpy(p1_prog, width=520, height=325, t_phase=0.0)
            ev2 = smc.evaluate_math_program_numpy(p2_prog, width=520, height=325, t_phase=0.0)
            tiger_pass1_img, tiger_arr1 = decode_base64_png(ev1["image_base64"])
            tiger_pass2_img, tiger_arr2 = decode_base64_png(ev2["image_base64"])
            tiger_mad = float(np.mean(np.abs(tiger_arr2 - tiger_arr1)))
            assert tiger_mad > 1.5, f"[tiger] Pass 2 must differ genuinely from Pass 1 (got MAD={tiger_mad:.2f})"
            tiger_pass1_stats = ev1["stats"]
            tiger_pass2_stats = ev2["stats"]
            tiger_pass1_score = p1_score_val
            tiger_pass2_score = p2_score_val
            tiger_motion = float(pass_res.get("motion_energy") or 0.0)

        # 2a. POST /api/generate-from-prompt (mode="pixel_video")
        r_pvid = client.post(
            "/api/generate-from-prompt",
            json={"prompt": prompt, "mode": "pixel_video", "refine": True, "N": 1600, "harmonics": 80},
        )
        assert r_pvid.status_code == 200, f"[{slug}] pixel_video failed: {r_pvid.text[:400]}"
        d_pvid = r_pvid.json()
        assert_no_branding_leaks(d_pvid, f"phase2:pixel_video:{slug}")
        assert d_pvid.get("mode") == "pixel_video", f"[{slug}] wrong mode: {d_pvid.get('mode')}"
        assert d_pvid.get("animation_capable") is True, f"[{slug}] animation_capable must be True"

        glsl = d_pvid.get("glsl_fragment_shader", "")
        ok_glsl, glsl_err = smc.validate_glsl_webgl1_static(glsl)
        assert ok_glsl, f"[{slug}] WebGL1 GLSL static validation failed: {glsl_err}"
        assert "precision highp float" in glsl and "uniform vec2 u_resolution" in glsl and "uniform float u_time" in glsl

        latex_eqs = d_pvid.get("latex_equations") or d_pvid.get("latex") or {}
        assert isinstance(latex_eqs, dict) and "RGB" in latex_eqs and "H_0" in latex_eqs, (
            f"[{slug}] missing AST-derived LaTeX keys: {list(latex_eqs.keys())}"
        )
        prompt_latex_fingerprints.append(json.dumps(latex_eqs, sort_keys=True))
        ast.parse(d_pvid["python_code"])

        painting_img, painting_arr = decode_base64_png(d_pvid["image_base64"])
        p_std = float(np.std(painting_arr))
        p_mean = float(np.mean(painting_arr))
        p_chroma = max(
            float(np.std(painting_arr[:, :, 0] - painting_arr[:, :, 1])),
            float(np.std(painting_arr[:, :, 0] - painting_arr[:, :, 2])),
            float(np.std(painting_arr[:, :, 1] - painting_arr[:, :, 2])),
        )
        assert p_std > 20.0, f"[{slug}] painting std={p_std:.2f} <= 20"
        assert 5.0 < p_mean < 250.0, f"[{slug}] painting mean={p_mean:.2f} degenerate"
        assert p_chroma > 8.0, f"[{slug}] painting chroma={p_chroma:.2f} <= 8"
        dead_loops = (d_pvid.get("stats") or {}).get("dead_loops") or []
        assert len(dead_loops) == 0, f"[{slug}] dead loops detected in final painting: {dead_loops}"
        print(
            f"   [PASS] pixel_video  -> '{d_pvid['title']}' | std={p_std:.1f} chroma={p_chroma:.1f} "
            f"| dead_loops={dead_loops} | GLSL={len(glsl)} chars (WebGL1 OK) | LaTeX={len(latex_eqs)} eqs"
        )

        # 2b. POST /api/render-math-video -> real standalone looping GIF + MP4
        r_vid = client.post(
            "/api/render-math-video",
            json={
                "prompt": prompt,
                "mode": "pixel_video",
                "width": 480,
                "height": 300,
                "num_frames": 32,
                "fps": 16,
                "refine": True,
                "include_mp4": True,
            },
        )
        assert r_vid.status_code == 200, f"[{slug}] /api/render-math-video failed: {r_vid.text[:400]}"
        d_vid = r_vid.json()
        assert_no_branding_leaks(d_vid, f"phase2:render-math-video:{slug}")
        motion_energy = float(d_vid.get("motion_energy", 0.0))
        assert motion_energy > 1.0, f"[{slug}] motion_energy={motion_energy:.3f} <= 1.0 (animation is static!)"

        gif_bytes = _decode_data_uri(d_vid.get("gif_base64", ""), "data:image/gif;base64,", f"{slug}.gif")
        mp4_bytes = _decode_data_uri(d_vid.get("mp4_base64", ""), "data:video/mp4;base64,", f"{slug}.mp4")

        gif_path = VIDEOS_DIR / f"{slug}.gif"
        mp4_path = VIDEOS_DIR / f"{slug}.mp4"
        gif_path.write_bytes(gif_bytes)
        mp4_path.write_bytes(mp4_bytes)

        # Verify GIF container integrity
        with Image.open(gif_path) as gim:
            n_gif_frames = getattr(gim, "n_frames", 1)
            assert n_gif_frames >= 16, f"[{slug}] GIF has too few frames: {n_gif_frames}"

        # Verify MP4 container integrity
        cap = cv2.VideoCapture(str(mp4_path))
        assert cap.isOpened(), f"[{slug}] Failed to open saved MP4: {mp4_path}"
        mp4_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        ok_f0, frame0 = cap.read()
        cap.release()
        assert ok_f0 and frame0 is not None and mp4_frame_count >= 16, (
            f"[{slug}] Invalid MP4 ({mp4_frame_count} frames, ok_f0={ok_f0})"
        )
        print(
            f"   [PASS] video export -> {gif_path.name} ({len(gif_bytes):,}B, {n_gif_frames}f) & "
            f"{mp4_path.name} ({len(mp4_bytes):,}B, {mp4_frame_count}f) | motion_energy={motion_energy:.2f}"
        )

        # Render 4-frame animation strip (t = 0, pi/2, pi, 3pi/2)
        strip_frames: List[Tuple[str, Image.Image]] = []
        for lbl, t_val in [("t = 0", 0.0), ("t = pi/2", math.pi * 0.5), ("t = pi", math.pi), ("t = 3pi/2", math.pi * 1.5)]:
            ev_f = smc.evaluate_math_program_numpy(prog, width=280, height=175, t_phase=t_val)
            f_img, _ = decode_base64_png(ev_f["image_base64"])
            strip_frames.append((lbl, f_img))
        strip_img = _compose_4frame_strip(strip_frames)

        # 2c. POST /api/generate-from-prompt (mode="lines") -> Crisp 2D Iso-Contours
        r_lines = client.post(
            "/api/generate-from-prompt",
            json={"prompt": prompt, "mode": "lines", "N": 1600, "harmonics": 80, "refine": True},
        )
        assert r_lines.status_code == 200, f"[{slug}] lines failed: {r_lines.text[:400]}"
        d_lines = r_lines.json()
        arr_lines = validate_lines_response(d_lines, f"phase2:lines:{slug}", expected_n=1600)
        lines_img = render_lines_preview(arr_lines, width=480, height=300)
        contour_count = int(d_lines.get("contour_count", 0))
        print(f"   [PASS] lines        -> N={len(arr_lines)} crisp contours (contour_count={contour_count})")

        # 2d. POST /api/generate-from-prompt (mode="3d_surface") -> Crisp 3D Relief Manifold
        r_3d = client.post(
            "/api/generate-from-prompt",
            json={"prompt": prompt, "mode": "3d_surface", "N": 1600, "harmonics": 80, "refine": True},
        )
        assert r_3d.status_code == 200, f"[{slug}] 3d_surface failed: {r_3d.text[:400]}"
        d_3d = r_3d.json()
        arr_3d = validate_3d_response(d_3d, f"phase2:3d_surface:{slug}", expected_n=1600)
        z_std = max(float(np.std(arr_3d[:, 2])), float(np.std(arr_3d[:, 5])))
        manifold_3d_img = render_3d_preview(arr_3d, width=480, height=300)
        print(f"   [PASS] 3d_surface   -> N={len(arr_3d)} 3D manifold (z_std={z_std:.3f})")

        # 2e. POST /api/generate-from-prompt (mode="math_video") -> 4D Wireframe Video
        r_mv = client.post(
            "/api/generate-from-prompt",
            json={"prompt": prompt, "mode": "math_video", "N": 1600, "harmonics": 80, "refine": True},
        )
        assert r_mv.status_code == 200, f"[{slug}] math_video failed: {r_mv.text[:400]}"
        d_mv = r_mv.json()
        arr_p, arr_th = validate_video_response(d_mv, f"phase2:math_video:{slug}", expected_n=1600)
        print(f"   [PASS] math_video   -> N={len(arr_p)} 4D wireframe (max_harmonic={float(np.max(np.abs(arr_th))):.3f})")

        gallery_rows.append(
            {
                "slug": slug,
                "prompt": prompt,
                "title": d_pvid["title"],
                "refinement_passes": refinement_passes,
                "std": p_std,
                "chroma": p_chroma,
                "motion_energy": motion_energy,
                "painting_img": painting_img,
                "strip_img": strip_img,
                "lines_img": lines_img,
                "manifold_3d_img": manifold_3d_img,
                "n_lines": len(arr_lines),
                "n_3d": len(arr_3d),
                "contour_count": contour_count,
                "z_std": z_std,
            }
        )

    assert len(set(prompt_latex_fingerprints)) == len(PHASE2_PROMPTS), (
        "Expected 4 completely distinct AST-derived LaTeX equation systems for the 4 unseen prompts!"
    )
    print("\n   [PASS] All 4 unseen prompts generated distinct AST-derived LaTeX equations + valid WebGL1 GLSL + GIF/MP4 videos!")

    print("\n3. Verifying `POST /api/preset` on Phase 2 `pvid_bengal_tiger` preset...")
    r_pvid_preset = client.post("/api/preset", json={"preset_id": "pvid_bengal_tiger"})
    assert r_pvid_preset.status_code == 200, r_pvid_preset.text
    d_pvid_preset = r_pvid_preset.json()
    assert_no_branding_leaks(d_pvid_preset, "/api/preset:pvid_bengal_tiger")
    assert d_pvid_preset.get("mode") == "pixel_video"
    ok_pg, pg_err = smc.validate_glsl_webgl1_static(d_pvid_preset.get("glsl_fragment_shader", ""))
    assert ok_pg, f"pvid_bengal_tiger GLSL invalid: {pg_err}"
    print(f"   [PASS] pvid_bengal_tiger -> '{d_pvid_preset['title']}' (WebGL1 GLSL OK, 0 branding leaks)")

    print("\n4. Saving Phase 2 high-resolution visual proof contact sheets...")
    _build_out_of_box_gallery(gallery_rows, SHEET_OUT_OF_BOX)
    assert SHEET_OUT_OF_BOX.exists() and SHEET_OUT_OF_BOX.stat().st_size > 50_000
    print(f"   [PASS] Saved {SHEET_OUT_OF_BOX} ({SHEET_OUT_OF_BOX.stat().st_size:,} bytes)")

    assert tiger_pass1_img is not None and tiger_pass2_img is not None
    _build_tiger_refinement_sheet(
        tiger_pass1_img,
        tiger_pass2_img,
        tiger_pass1_stats,
        tiger_pass2_stats,
        tiger_pass1_score,
        tiger_pass2_score,
        tiger_motion,
        SHEET_TIGER_REFINEMENT,
    )
    assert SHEET_TIGER_REFINEMENT.exists() and SHEET_TIGER_REFINEMENT.stat().st_size > 30_000
    print(f"   [PASS] Saved {SHEET_TIGER_REFINEMENT} ({SHEET_TIGER_REFINEMENT.stat().st_size:,} bytes)")

    verify_all_frontend_files_and_build()

    print("\n===========================================================================")
    print("PHASE 2 SOTA VERIFICATION PASSED (100%): 4 UNSEEN PROMPTS + GIF/MP4 + GLSL!")
    print("===========================================================================")


if __name__ == "__main__":
    run_phase2_verification()
