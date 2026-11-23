"""
Stage 1 & Stage 2 Master Mathematical Paradigms + Beyond 3D Raymarched Cinema SOTA Verification Suite.

Verifies:
1. Stage 1 ("Be Like Them Perfectly"): All 8 Master Mathematical Paradigms via `POST /api/preset`:
   - `mp_iq_mandelbulb`         (Inigo Quilez — 3D SDFs & Real-Time GPU Raymarching)
   - `mp_ddg_gyroid`            (Helaman Ferguson & Bathsheba Grossman — DDG & Minimal Surfaces)
   - `mp_floraform_pde`         (Nervous System Floraform — Differential Growth & Elastic Buckling PDE)
   - `mp_thurston_eversion`     (William Thurston / Geometry Center — 4D Sphere Eversion & Hopf Projection)
   - `mp_xenakis_metastasis`    (Iannis Xenakis — Stochastic OU & Ruled Hyperbolic Paraboloids)
   - `mp_houdini_curl_attractor`(SideFX Houdini Style — 3D Curl-Noise, RK4 Attractor & Gray-Scott PDE)
   - `mp_manim_conformal`       (Manim / 3Blue1Brown — Complex Conformal Homotopy & Domain Coloring)
   - `mp_kleinian_mandelbox`    (4K/60FPS GLSL Raymarching Cinema — Non-Euclidean Kleinian Mandelbox)
   Checks: valid WebGL1 ES 1.00 GLSL, high-contrast 3D visuals (`std > 28`), non-empty `primitives_3d`,
   authentic LaTeX equations, parseable Python export, looping GIF/MP4 video export on `mp_iq_mandelbulb`,
   zero AI/model branding leaks, and saves `backend/sheet_stage1_8_master_paradigms.png` (2x4 gallery).

2. Stage 2 ("Go Beyond Them"): 4 unseen prompts synthesized in `mode="raymarch_cinema"`:
   - "a golden phoenix rising over turbulent sapphire ocean waves"
   - "a hyperbolic bronze cathedral with glowing gyroid pillars and god rays"
   - "a bioluminescent deep-sea dragonfish surrounded by differential-growth coral"
   - "a non-Euclidean celestial astrolabe with orbiting Saturn rings"
   Checks: valid 3D `raymarch_3d_scene`, valid WebGL1 GLSL, high-contrast 3D raymarched PNG (`std > 25`,
   `chroma > 10`), non-empty `primitives_3d` (`z_std > 0.01`), `primitives` & `time_harmonics`, authentic
   LaTeX, standalone looping `.gif` & `.mp4` videos in `backend/out_of_box_videos/` (`motion_energy > 1.0`),
   zero AI/model branding leaks, and saves `backend/sheet_stage2_beyond_raymarch_cinema.png` (4x4 gallery:
   3D Raymarched Render | 4-Phase 4D Cinema Strip | Extracted 3D Surface Manifold | Exact LaTeX & Curvature Normal Map).

3. Frontend zero-AI-branding check across all `frontend/src/*` files and `npm run build` exit code 0.
"""

import ast
import base64
import json
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw

BACKEND_DIR = Path(__file__).resolve().parent
REPO_ROOT = BACKEND_DIR.parent
FRONTEND_DIR = REPO_ROOT / "frontend"
VIDEOS_DIR = BACKEND_DIR / "out_of_box_videos"
SHEET_STAGE1 = BACKEND_DIR / "sheet_stage1_8_master_paradigms.png"
SHEET_STAGE2 = BACKEND_DIR / "sheet_stage2_beyond_raymarch_cinema.png"

if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from fastapi.testclient import TestClient  # noqa: E402
import master_paradigms_engine as mpe  # noqa: E402
import symbolic_math_compiler as smc  # noqa: E402
from main import app  # noqa: E402
from test_e2e_behavior import (  # noqa: E402
    assert_no_branding_leaks,
    decode_base64_png,
    find_branding_leaks_in_text,
    render_3d_preview,
)

STAGE1_PRESETS: List[Tuple[str, str]] = [
    ("mp_iq_mandelbulb", "1. Inigo Quilez — 3D SDF Mandelbulb Raymarching"),
    ("mp_ddg_gyroid", "2. Ferguson/Grossman — DDG Gyroid Minimal Surface"),
    ("mp_floraform_pde", "3. Nervous System — Floraform Elastic Buckling PDE"),
    ("mp_thurston_eversion", "4. William Thurston — 4D Sphere Eversion Homotopy"),
    ("mp_xenakis_metastasis", "5. Iannis Xenakis — Ruled Hyperbolic Paraboloids"),
    ("mp_houdini_curl_attractor", "6. Houdini — 3D Curl-Noise, RK4 & Gray-Scott PDE"),
    ("mp_manim_conformal", "7. Manim/3B1B — Complex Conformal Homotopy w=f(z,t)"),
    ("mp_kleinian_mandelbox", "8. Non-Euclidean Kleinian Limit Set & Mandelbox"),
]

STAGE2_PROMPTS: List[Tuple[str, str]] = [
    ("golden_phoenix", "a golden phoenix rising over turbulent sapphire ocean waves"),
    ("gyroid_cathedral", "a hyperbolic bronze cathedral with glowing gyroid pillars and god rays"),
    ("deep_sea_dragonfish", "a bioluminescent deep-sea dragonfish surrounded by differential-growth coral"),
    ("celestial_astrolabe", "a non-Euclidean celestial astrolabe with orbiting Saturn rings"),
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
    canvas = Image.new("RGB", (width, height), (8, 12, 22))
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


def _render_latex_and_curvature_panel(
    render_img: Image.Image,
    latex_dict: Dict[str, str],
    stats: Dict[str, Any],
    node_count: int,
    motion_energy: float,
    refinement_passes: int,
    width: int = 480,
    height: int = 300,
) -> Image.Image:
    """
    Column 4 panel for Stage 2 gallery:
    Top half: Tangent-space shading-normal & discrete Laplacian curvature map derived from 3D raymarch render.
    Bottom half: Exact LaTeX governing equation keys & closed-form excerpts + quantitative 3D metrics.
    """
    panel = Image.new("RGB", (width, height), (10, 15, 28))
    draw = ImageDraw.Draw(panel)

    # Top half: Normal-gradient (left) & Laplacian Curvature (right)
    top_h = 148
    half_w = width // 2
    rgb = np.asarray(render_img.convert("RGB").resize((half_w - 8, top_h - 22), Image.Resampling.BILINEAR), dtype=np.float32)
    lum = (0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]) / 255.0

    gx = cv2.Sobel(lum, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(lum, cv2.CV_32F, 0, 1, ksize=3)
    nz = np.full_like(lum, 0.35)
    norm_len = np.sqrt(gx * gx + gy * gy + nz * nz) + 1e-6
    n_rgb = np.stack([
        np.clip((gx / norm_len) * 0.5 + 0.5, 0.0, 1.0),
        np.clip((-gy / norm_len) * 0.5 + 0.5, 0.0, 1.0),
        np.clip((nz / norm_len) * 0.5 + 0.5, 0.0, 1.0),
    ], axis=-1)
    normal_img = Image.fromarray((n_rgb * 255.0).astype(np.uint8), mode="RGB")

    lap = np.abs(cv2.Laplacian(lum, cv2.CV_32F, ksize=3))
    lap_norm = np.clip(lap / (np.percentile(lap, 98) + 1e-5), 0.0, 1.0)
    curv_bgr = cv2.applyColorMap((lap_norm * 255.0).astype(np.uint8), cv2.COLORMAP_INFERNO)
    curv_img = Image.fromarray(cv2.cvtColor(curv_bgr, cv2.COLOR_BGR2RGB), mode="RGB")

    panel.paste(normal_img, (4, 18))
    panel.paste(curv_img, (half_w + 4, 18))
    draw.text((8, 4), "3D Gradient Normal Map N(p)", fill=(56, 189, 248))
    draw.text((half_w + 8, 4), "Discrete Curvature |lap(d)|", fill=(251, 191, 36))

    # Bottom half: Exact LaTeX equations & metrics
    y_cursor = top_h + 4
    draw.line([(4, y_cursor - 2), (width - 4, y_cursor - 2)], fill=(51, 65, 85), width=1)
    metric_line = (
        f"Nodes={node_count} | Passes={refinement_passes} | std={stats.get('std', 0):.1f} | "
        f"chroma={stats.get('chroma', 0):.1f} | norm_var={stats.get('normal_variance', 0):.3f} | mot={motion_energy:.1f}"
    )
    draw.text((8, y_cursor + 2), metric_line[:76], fill=(52, 211, 153))
    y_cursor += 18

    for k_idx, (eq_key, eq_tex) in enumerate(list(latex_dict.items())[:5]):
        clean_tex = " ".join(str(eq_tex).split())
        line_txt = f"{eq_key}: {clean_tex}"
        if len(line_txt) > 74:
            line_txt = line_txt[:71] + "..."
        draw.text((8, y_cursor + k_idx * 24), line_txt, fill=(226, 232, 240))
        # Wrap second line if long
        if len(clean_tex) > 64:
            rem = clean_tex[64:134]
            draw.text((16, y_cursor + k_idx * 24 + 11), rem, fill=(148, 163, 184))

    return panel


def _build_stage1_sheet(panels: List[Tuple[str, str, Image.Image]], out_path: Path) -> None:
    cols, rows = 4, 2
    cell_w, cell_h = 520, 325
    header_h = 40
    pad = 14
    sheet_w = cols * cell_w + (cols + 1) * pad
    sheet_h = 54 + rows * (cell_h + header_h + pad) + pad

    sheet = Image.new("RGB", (sheet_w, sheet_h), (8, 12, 22))
    draw = ImageDraw.Draw(sheet)
    draw.text(
        (pad, 18),
        "DesignWithMath STAGE 1 — All 8 Master Mathematical Paradigms (Quilez • Ferguson/Grossman • Floraform • Thurston • Xenakis • Houdini • Manim • Kleinian)",
        fill=(241, 245, 249),
    )

    for idx, (title, sub, img) in enumerate(panels):
        r, c = divmod(idx, cols)
        x0 = pad + c * (cell_w + pad)
        y0 = 54 + r * (cell_h + header_h + pad)
        draw.rectangle(
            [x0 - 1, y0 - 1, x0 + cell_w + 1, y0 + header_h + cell_h + 1],
            outline=(51, 65, 85),
            width=1,
        )
        draw.rectangle([x0, y0, x0 + cell_w, y0 + header_h], fill=(15, 23, 42))
        draw.text((x0 + 8, y0 + 5), title[:68], fill=(56, 189, 248))
        draw.text((x0 + 8, y0 + 22), sub[:80], fill=(148, 163, 184))
        sheet.paste(img.convert("RGB").resize((cell_w, cell_h), Image.Resampling.LANCZOS), (x0, y0 + header_h))

    sheet.save(out_path, format="PNG")


def _build_stage2_sheet(rows_data: List[Dict[str, Any]], out_path: Path) -> None:
    cols = 4
    rows = len(rows_data)
    cell_w, cell_h = 480, 300
    header_h = 38
    pad = 14
    sheet_w = cols * cell_w + (cols + 1) * pad
    sheet_h = 54 + rows * (cell_h + header_h + pad) + pad

    sheet = Image.new("RGB", (sheet_w, sheet_h), (8, 12, 22))
    draw = ImageDraw.Draw(sheet)
    draw.text(
        (pad, 18),
        "DesignWithMath STAGE 2 — Universal Text-to-3D Raymarched Cinema: 3D Render | 4-Phase 4D Strip | Extracted 3D Manifold | LaTeX & Curvature Normal Map",
        fill=(241, 245, 249),
    )

    for r_idx, row in enumerate(rows_data):
        y0 = 54 + r_idx * (cell_h + header_h + pad)
        cells = [
            (
                f"1. [3D Raymarch Render] {row['title']}",
                f"{row['prompt'][:54]} | std={row['std']:.1f} chroma={row['chroma']:.1f}",
                row["render_img"],
            ),
            (
                f"2. [4D Cinema Strip] {row['slug']}.gif / .mp4 (24f @ 12fps)",
                f"t=0, pi/2, pi, 3pi/2 | motion={row['motion_energy']:.2f} | GLSL WebGL1 OK",
                row["strip_img"],
            ),
            (
                f"3. [3D Surface Manifold] primitives_3d (N={row['n_3d']})",
                f"z_std={row['z_std']:.3f} | Newton surface projection on d(p, t) = 0",
                row["manifold_img"],
            ),
            (
                f"4. [Exact LaTeX & 3D Curvature/Normal Map]",
                f"passes={row['refinement_passes']} | score={row['visual_score']:.1f} | norm_var={row['normal_variance']:.3f}",
                row["latex_curv_img"],
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
            sheet.paste(pil_img.convert("RGB").resize((cell_w, cell_h), Image.Resampling.LANCZOS), (x0, y0 + header_h))

    sheet.save(out_path, format="PNG")


def verify_frontend_and_build() -> None:
    print("\n4. Scanning ALL frontend/src/* files for zero AI/model branding & running `npm run build`...")
    checked = 0
    for fpath in sorted((FRONTEND_DIR / "src").rglob("*")):
        if fpath.is_file() and fpath.suffix.lower() in {".js", ".jsx", ".ts", ".tsx", ".css", ".html"}:
            content = fpath.read_text(encoding="utf-8")
            leaks = find_branding_leaks_in_text(content, str(fpath.relative_to(REPO_ROOT)))
            assert not leaks, f"AI branding leak detected in {fpath}:\n  " + "\n  ".join(leaks[:10])
            checked += 1
    assert checked >= 2, f"Expected >=2 frontend source files scanned, got {checked}"
    print(f"   [PASS] Scanned {checked} files under frontend/src/ — ZERO AI/model branding leaks")

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


def run_master_paradigms_sota_suite() -> None:
    VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
    client = TestClient(app)

    print("1. Verifying /api/presets catalog includes all 8 Master Paradigms & zero branding leaks...")
    r_cat = client.get("/api/presets")
    assert r_cat.status_code == 200, r_cat.text
    catalog = r_cat.json()
    assert_no_branding_leaks(catalog, "/api/presets")
    all_ids = {p["id"] for p in catalog.get("presets", [])}
    mp_ids = {p["id"] for p in catalog.get("master_paradigm_presets", [])}
    expected_mp = {pid for pid, _ in STAGE1_PRESETS}
    assert expected_mp.issubset(all_ids), f"Missing Master Paradigms in presets: {expected_mp - all_ids}"
    assert expected_mp.issubset(mp_ids), f"Missing Master Paradigms in master_paradigm_presets: {expected_mp - mp_ids}"
    print(f"   [PASS] /api/presets: {len(catalog['presets'])} total presets + {len(mp_ids)} master_paradigm_presets (0 branding leaks)")

    print("\n2. STAGE 1 ('Be Like Them Perfectly') — Testing all 8 Master Mathematical Paradigms via /api/preset...")
    stage1_panels: List[Tuple[str, str, Image.Image]] = []
    for pid, label in STAGE1_PRESETS:
        t0 = time.time()
        resp = client.post("/api/preset", json={"preset_id": pid, "N": 1600})
        assert resp.status_code == 200, f"[{pid}] /api/preset failed: {resp.text[:400]}"
        data = resp.json()
        assert_no_branding_leaks(data, f"stage1:{pid}")
        assert data.get("mode") == "raymarch_cinema", f"[{pid}] expected mode='raymarch_cinema', got {data.get('mode')}"
        assert data.get("animation_capable") is True, f"[{pid}] animation_capable must be True"

        glsl = data.get("glsl_fragment_shader", "")
        ok_glsl, glsl_err = smc.validate_glsl_webgl1_static(glsl)
        assert ok_glsl, f"[{pid}] WebGL1 GLSL static validation failed: {glsl_err}"
        assert "precision highp float" in glsl and "uniform vec2 u_resolution" in glsl and "uniform float u_time" in glsl

        img, arr = decode_base64_png(data.get("image_base64", ""))
        std_val = float(np.std(arr))
        mean_val = float(np.mean(arr))
        chroma_val = max(
            float(np.std(arr[:, :, 0] - arr[:, :, 1])),
            float(np.std(arr[:, :, 0] - arr[:, :, 2])),
            float(np.std(arr[:, :, 1] - arr[:, :, 2])),
        )
        assert std_val > 28.0, f"[{pid}] std(rgb)={std_val:.2f} <= 28.0"
        assert 5.0 < mean_val < 250.0, f"[{pid}] degenerate mean(rgb)={mean_val:.2f}"

        p3d = data.get("primitives_3d")
        assert isinstance(p3d, list) and len(p3d) >= 100, f"[{pid}] empty/too-small primitives_3d"
        arr3d = np.asarray(p3d, dtype=np.float64)
        assert arr3d.ndim == 2 and arr3d.shape[1] == 6 and np.isfinite(arr3d).all(), f"[{pid}] invalid primitives_3d shape/values"
        z_std = max(float(np.std(arr3d[:, 2])), float(np.std(arr3d[:, 5])))

        latex_d = data.get("latex_equations") or data.get("latex") or {}
        assert isinstance(latex_d, dict) and len(latex_d) >= 2, f"[{pid}] missing authentic LaTeX equations"
        ast.parse(data["python_code"])

        norm_var = float((data.get("stats") or {}).get("normal_variance", 0.0))
        stage1_panels.append(
            (
                label,
                f"{pid} | std={std_val:.1f} chroma={chroma_val:.1f} norm_var={norm_var:.3f} | 3D N={len(p3d)} (z_std={z_std:.3f})",
                img,
            )
        )
        print(
            f"   [PASS] {pid:26s} ({time.time()-t0:.2f}s) -> std={std_val:.1f}, chroma={chroma_val:.1f}, "
            f"norm_var={norm_var:.3f}, 3D_N={len(p3d)}, GLSL={len(glsl)}ch, LaTeX={len(latex_d)}eqs"
        )

    _build_stage1_sheet(stage1_panels, SHEET_STAGE1)
    assert SHEET_STAGE1.exists() and SHEET_STAGE1.stat().st_size > 80_000
    print(f"   [PASS] Saved Stage-1 2x4 gallery -> {SHEET_STAGE1} ({SHEET_STAGE1.stat().st_size:,} bytes)")

    # Verify /api/render-math-video works with preset_id="mp_iq_mandelbulb"
    print("   Testing POST /api/render-math-video on Stage-1 preset `mp_iq_mandelbulb`...")
    r_mp_vid = client.post(
        "/api/render-math-video",
        json={"preset_id": "mp_iq_mandelbulb", "mode": "raymarch_cinema", "width": 360, "height": 225, "num_frames": 20, "fps": 12, "include_mp4": True},
    )
    assert r_mp_vid.status_code == 200, f"mp_iq_mandelbulb render-math-video failed: {r_mp_vid.text[:400]}"
    d_mp_vid = r_mp_vid.json()
    assert_no_branding_leaks(d_mp_vid, "stage1:render-math-video:mp_iq_mandelbulb")
    assert float(d_mp_vid.get("motion_energy", 0.0)) > 1.0, f"mp_iq_mandelbulb static video: {d_mp_vid.get('motion_energy')}"
    mp_gif_bytes = _decode_data_uri(d_mp_vid.get("gif_base64", ""), "data:image/gif;base64,", "mp_iq_mandelbulb.gif")
    mp_mp4_bytes = _decode_data_uri(d_mp_vid.get("mp4_base64", ""), "data:video/mp4;base64,", "mp_iq_mandelbulb.mp4")
    (VIDEOS_DIR / "mp_iq_mandelbulb.gif").write_bytes(mp_gif_bytes)
    (VIDEOS_DIR / "mp_iq_mandelbulb.mp4").write_bytes(mp_mp4_bytes)
    print(f"   [PASS] Stage-1 video export -> mp_iq_mandelbulb.gif ({len(mp_gif_bytes):,}B) & .mp4 ({len(mp_mp4_bytes):,}B), motion={d_mp_vid['motion_energy']:.2f}")

    print("\n3. STAGE 2 ('Go Beyond Them') — Synthesizing 4 unseen prompts in `mode='raymarch_cinema'`...")
    stage2_rows: List[Dict[str, Any]] = []
    stage2_latex_fps: List[str] = []

    for slug, prompt in STAGE2_PROMPTS:
        t0 = time.time()
        print(f"\n   --- [{slug.upper()}] '{prompt}' ---")

        r_gen = client.post(
            "/api/generate-from-prompt",
            json={"prompt": prompt, "mode": "raymarch_cinema", "N": 1600, "refine": True},
        )
        assert r_gen.status_code == 200, f"[{slug}] /api/generate-from-prompt failed: {r_gen.text[:400]}"
        d_gen = r_gen.json()
        assert_no_branding_leaks(d_gen, f"stage2:generate:{slug}")
        assert d_gen.get("mode") == "raymarch_cinema"
        assert d_gen.get("animation_capable") is True

        scene = d_gen.get("raymarch_scene")
        assert isinstance(scene, dict), f"[{slug}] missing raymarch_scene in response"
        ok_sc, sc_err = mpe.validate_raymarch_scene(scene)
        assert ok_sc, f"[{slug}] raymarch_scene validation failed: {sc_err}"
        node_count = len(scene.get("nodes", []))

        glsl = d_gen.get("glsl_fragment_shader", "")
        ok_glsl, glsl_err = smc.validate_glsl_webgl1_static(glsl)
        assert ok_glsl, f"[{slug}] WebGL1 GLSL static validation failed: {glsl_err}"
        assert "precision highp float" in glsl and "uniform vec2 u_resolution" in glsl and "uniform float u_time" in glsl

        render_img, render_arr = decode_base64_png(d_gen.get("image_base64", ""))
        std_val = float(np.std(render_arr))
        mean_val = float(np.mean(render_arr))
        chroma_val = max(
            float(np.std(render_arr[:, :, 0] - render_arr[:, :, 1])),
            float(np.std(render_arr[:, :, 0] - render_arr[:, :, 2])),
            float(np.std(render_arr[:, :, 1] - render_arr[:, :, 2])),
        )
        assert std_val > 25.0, f"[{slug}] 3D raymarch render std={std_val:.2f} <= 25.0"
        assert 5.0 < mean_val < 250.0, f"[{slug}] degenerate mean={mean_val:.2f}"
        assert chroma_val > 10.0, f"[{slug}] chroma={chroma_val:.2f} <= 10.0"

        arr3d = np.asarray(d_gen.get("primitives_3d", []), dtype=np.float64)
        assert arr3d.ndim == 2 and arr3d.shape[1] == 6 and len(arr3d) >= 400 and np.isfinite(arr3d).all(), (
            f"[{slug}] invalid primitives_3d: {arr3d.shape}"
        )
        z_std = max(float(np.std(arr3d[:, 2])), float(np.std(arr3d[:, 5])))
        assert z_std > 0.01, f"[{slug}] flat 3D manifold (z_std={z_std:.5f})"

        arr2d = np.asarray(d_gen.get("primitives", []), dtype=np.float64)
        assert arr2d.ndim == 2 and arr2d.shape[1] == 4 and len(arr2d) >= 400 and np.isfinite(arr2d).all()

        latex_d = d_gen.get("latex_equations") or d_gen.get("latex") or {}
        assert isinstance(latex_d, dict) and len(latex_d) >= 3, f"[{slug}] missing LaTeX equations"
        stage2_latex_fps.append(json.dumps(latex_d, sort_keys=True))
        ast.parse(d_gen["python_code"])

        stats = d_gen.get("stats") or {}
        norm_var = float(stats.get("normal_variance", 0.0))
        ref_passes = int(d_gen.get("refinement_passes", 1))
        vis_score = float(d_gen.get("visual_score", 0.0))
        print(
            f"   [PASS] raymarch_cinema ({time.time()-t0:.1f}s) -> '{d_gen['title']}' | nodes={node_count} "
            f"| passes={ref_passes} score={vis_score:.1f} | std={std_val:.1f} chroma={chroma_val:.1f} "
            f"norm_var={norm_var:.3f} | GLSL={len(glsl)}ch (WebGL1 OK)"
        )

        # Render standalone looping GIF + MP4 via POST /api/render-math-video
        r_vid = client.post(
            "/api/render-math-video",
            json={
                "prompt": prompt,
                "mode": "raymarch_cinema",
                "width": 420,
                "height": 262,
                "num_frames": 24,
                "fps": 12,
                "refine": True,
                "include_mp4": True,
            },
        )
        assert r_vid.status_code == 200, f"[{slug}] /api/render-math-video failed: {r_vid.text[:400]}"
        d_vid = r_vid.json()
        assert_no_branding_leaks(d_vid, f"stage2:render-math-video:{slug}")
        motion_energy = float(d_vid.get("motion_energy", 0.0))
        assert motion_energy > 1.0, f"[{slug}] video motion_energy={motion_energy:.3f} <= 1.0"

        gif_bytes = _decode_data_uri(d_vid.get("gif_base64", ""), "data:image/gif;base64,", f"{slug}.gif")
        mp4_bytes = _decode_data_uri(d_vid.get("mp4_base64", ""), "data:video/mp4;base64,", f"{slug}.mp4")
        gif_path = VIDEOS_DIR / f"{slug}.gif"
        mp4_path = VIDEOS_DIR / f"{slug}.mp4"
        gif_path.write_bytes(gif_bytes)
        mp4_path.write_bytes(mp4_bytes)

        with Image.open(gif_path) as gim:
            n_gif_frames = getattr(gim, "n_frames", 1)
            assert n_gif_frames >= 16, f"[{slug}] GIF has too few frames: {n_gif_frames}"

        cap = cv2.VideoCapture(str(mp4_path))
        assert cap.isOpened(), f"[{slug}] Failed to open MP4: {mp4_path}"
        n_mp4_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        ok_f0, frame0 = cap.read()
        cap.release()
        assert ok_f0 and frame0 is not None and n_mp4_frames >= 16, f"[{slug}] Invalid MP4 ({n_mp4_frames} frames)"
        print(
            f"   [PASS] video export -> {gif_path.name} ({len(gif_bytes):,}B, {n_gif_frames}f) & "
            f"{mp4_path.name} ({len(mp4_bytes):,}B, {n_mp4_frames}f) | motion_energy={motion_energy:.2f}"
        )

        # Render 4-phase 4D cinema strip
        strip_frames: List[Tuple[str, Image.Image]] = []
        for lbl, t_val in [("t = 0", 0.0), ("t = pi/2", math.pi * 0.5), ("t = pi", math.pi), ("t = 3pi/2", math.pi * 1.5)]:
            ev_t = mpe.evaluate_raymarch_scene_numpy(scene, width=260, height=162, t_phase=t_val, N=200)
            f_img, _ = decode_base64_png(ev_t["image_base64"])
            strip_frames.append((lbl, f_img))
        strip_img = _compose_4frame_strip(strip_frames)

        # Render extracted 3D manifold & exact LaTeX / curvature-normal panel
        manifold_img = render_3d_preview(arr3d, width=480, height=300)
        latex_curv_img = _render_latex_and_curvature_panel(
            render_img=render_img,
            latex_dict=latex_d,
            stats={"std": std_val, "chroma": chroma_val, "normal_variance": norm_var},
            node_count=node_count,
            motion_energy=motion_energy,
            refinement_passes=ref_passes,
        )

        stage2_rows.append(
            {
                "slug": slug,
                "prompt": prompt,
                "title": d_gen["title"],
                "std": std_val,
                "chroma": chroma_val,
                "normal_variance": norm_var,
                "motion_energy": motion_energy,
                "refinement_passes": ref_passes,
                "visual_score": vis_score,
                "n_3d": len(arr3d),
                "z_std": z_std,
                "render_img": render_img,
                "strip_img": strip_img,
                "manifold_img": manifold_img,
                "latex_curv_img": latex_curv_img,
            }
        )

    assert len(set(stage2_latex_fps)) == len(STAGE2_PROMPTS), "Expected distinct LaTeX equations across the 4 Stage-2 scenes"
    _build_stage2_sheet(stage2_rows, SHEET_STAGE2)
    assert SHEET_STAGE2.exists() and SHEET_STAGE2.stat().st_size > 100_000
    print(f"\n   [PASS] Saved Stage-2 4x4 gallery -> {SHEET_STAGE2} ({SHEET_STAGE2.stat().st_size:,} bytes)")

    verify_frontend_and_build()

    print("\n================================================================================")
    print("STAGE 1 (8 MASTER PARADIGMS) & STAGE 2 (BEYOND 3D RAYMARCH CINEMA) PASSED 100%!")
    print("================================================================================")


if __name__ == "__main__":
    run_master_paradigms_sota_suite()
