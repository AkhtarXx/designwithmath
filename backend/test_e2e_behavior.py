"""
Comprehensive E2E & Mathematical Behavioral Verification Suite for DesignWithMath.

Verifies:
1. All 5 synthesis modes (`lines`, `circles`, `pixel_field`, `3d_surface`, `math_video`)
   end-to-end via FastAPI endpoints (`/api/health`, `/api/presets`, `/api/preset`,
   `/api/generate-from-prompt`, `/api/evaluate-custom`).
2. All 7 Hamid Naderi Yeganeh 2D Pixel-Field paintings (`pf_peacock`, `pf_autumn_trees`,
   `pf_kiwi`, `pf_strawberries`, `pf_corals`, `pf_bipolar_nebula`, `pf_moon_footprint`)
   for non-empty, colorful, high-variance RGB rendering (`std(rgb) > 20`, no all-black/white).
3. Simple, Medium, and Complex prompt routing via silent flagship multi-tier router across
   all 5 modes (`primitives`, `primitives_3d`, `time_harmonics`, `image_base64`, animated
   `t_phase` / `num_frames`, LaTeX equations, valid Python AST).
4. Brand-new unseen arbitrary user prompts (`"a glowing jellyfish in the deep ocean"`,
   `"an astronaut floating near Saturn rings in space"`,
   `"snowy mountain peak at golden sunset with pine trees"`,
   `"a roaring Bengal tiger in jungle"`) across `pixel_field`, `3d_surface`, `math_video`,
   and `lines` modes — verifying prompt-specific LaTeX and genuinely distinct visual outputs
   (never copy-pasted presets or ASCII text fallbacks).
5. Zero AI / Model Branding Leak across ALL API responses and frontend files (`App.jsx`,
   `index.css`) — banning Cloudflare, DeepSeek, GLM, Kimi, GPT-OSS, QwQ, Llama, Qwen, 70B,
   120B, AI, etc.
6. Visual verification contact sheets (`sheet_verified_all_modes.png` and
   `sheet_unseen_prompts.png`).
7. Frontend production build (`npm run build` in `frontend/`) exiting with code 0.
"""

import argparse
import ast
import base64
import io
import json
import math
import re
import subprocess
import sys
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw

BACKEND_DIR = Path(__file__).resolve().parent
REPO_ROOT = BACKEND_DIR.parent
FRONTEND_DIR = REPO_ROOT / "frontend"
SHEET_ALL_MODES_PATH = BACKEND_DIR / "sheet_verified_all_modes.png"
SHEET_UNSEEN_PATH = BACKEND_DIR / "sheet_unseen_prompts.png"

if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from fastapi.testclient import TestClient  # noqa: E402
import ai_architect  # noqa: E402
import symbolic_math_compiler as smc  # noqa: E402
from main import app  # noqa: E402


# Forbidden AI / model branding tokens that MUST NEVER appear in any API response or UI file
FORBIDDEN_SUBSTRINGS = (
    "cloudflare",
    "70b",
    "120b",
    "llama",
    "qwen",
    "qwq",
    "deepseek",
    "gemma",
    "mistral",
    "gpt-oss",
    "moonshot",
    "openai",
    "workers ai",
)
FORBIDDEN_WORD_RE = re.compile(
    r"\b(cloudflare|workers\s*ai|llama[-\d\.]*|qwen[-\d\.]*|qwq[-\d\.]*|deepseek[-\d\.]*|"
    r"glm[-\d\.]*|kimi[-\d\.]*|gemma[-\d\.]*|mistral[-\w\.]*|scout|vlm|moonshot|openai|gpt[-\w]*|nemotron|"
    r"120b|70b|32b|27b|24b|17b|12b|8b|fp8|llm|ai)\b",
    re.IGNORECASE,
)

REQUIRED_PIXEL_FIELD_PRESETS = [
    "pf_peacock",
    "pf_autumn_trees",
    "pf_kiwi",
    "pf_strawberries",
    "pf_corals",
    "pf_bipolar_nebula",
    "pf_moon_footprint",
]

CHROMATIC_PIXEL_FIELD_PRESETS = {
    "pf_peacock",
    "pf_autumn_trees",
    "pf_kiwi",
    "pf_strawberries",
    "pf_corals",
    "pf_bipolar_nebula",
}

REQUIRED_SCIAM_PRESETS = {
    "bird_in_flight",
    "japanese_maple_sciam",
    "maple_leaf_sciam",
    "oak_leaf_sciam",
    "olive_branch_sciam",
    "palm_branch_sciam",
}

REQUIRED_3D_PRESETS = {
    "3d_rose_bloom",
    "3d_torus_knot",
    "3d_bird_wings",
    "3d_nautilus_shell",
    "3d_klein_caustic",
}

REQUIRED_VIDEO_PRESETS = {
    "video_soaring_bird",
    "video_harmonic_butterfly",
    "video_swimming_angelfish",
    "video_blooming_lotus",
}

UNSEEN_NOVEL_PROMPTS = [
    "a glowing jellyfish in the deep ocean",
    "an astronaut floating near Saturn rings in space",
    "snowy mountain peak at golden sunset with pine trees",
    "a roaring Bengal tiger in jungle",
]


class ApiAdapter:
    """Unified adapter supporting FastAPI TestClient (default) or live HTTP server."""

    def __init__(self, live_url: Optional[str] = None):
        self.live_url = live_url.rstrip("/") if live_url else None
        self.client = None if self.live_url else TestClient(app)

    def get_json(self, path: str) -> Dict[str, Any]:
        if self.live_url:
            with urllib.request.urlopen(f"{self.live_url}{path}", timeout=30) as resp:
                assert resp.status == 200, f"GET {path}: expected 200, got {resp.status}"
                return json.loads(resp.read().decode("utf-8"))
        resp = self.client.get(path)
        assert resp.status_code == 200, f"GET {path}: expected 200, got {resp.status_code}: {resp.text[:300]}"
        return resp.json()

    def post_json(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        if self.live_url:
            req = urllib.request.Request(
                f"{self.live_url}{path}",
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=180) as resp:
                assert resp.status == 200, f"POST {path}: expected 200, got {resp.status}"
                return json.loads(resp.read().decode("utf-8"))
        resp = self.client.post(path, json=payload)
        assert resp.status_code == 200, f"POST {path} ({payload}): expected 200, got {resp.status_code}: {resp.text[:400]}"
        return resp.json()


def find_branding_leaks_in_text(text: str, context: str) -> List[str]:
    leaks: List[str] = []
    low = text.lower()
    for tok in FORBIDDEN_SUBSTRINGS:
        if tok in low:
            leaks.append(f"{context}: forbidden substring '{tok}' found in {text[:160]!r}")
    for match in FORBIDDEN_WORD_RE.finditer(text):
        word = match.group(0)
        leaks.append(f"{context}: forbidden token '{word}' found in {text[:160]!r}")
    return leaks


def find_branding_leaks_in_payload(obj: Any, path: str = "root") -> List[str]:
    """Recursively inspect every string value (and non-API-key names) in an API response."""
    leaks: List[str] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            # Skip binary base64 media payloads
            if k in ("image_base64", "frames_base64", "gif_base64", "mp4_base64"):
                continue
            # Allow internal boolean request flag name `use_ai` inside sample_prompts templates
            if k != "use_ai" and isinstance(k, str):
                leaks.extend(find_branding_leaks_in_text(k, f"{path}.<key:{k}>"))
            leaks.extend(find_branding_leaks_in_payload(v, f"{path}.{k}"))
    elif isinstance(obj, list):
        # Skip massive numeric arrays (primitives, primitives_3d, time_harmonics)
        if obj and isinstance(obj[0], (int, float)):
            return leaks
        if obj and isinstance(obj[0], list) and obj[0] and isinstance(obj[0][0], (int, float)):
            return leaks
        for idx, item in enumerate(obj):
            leaks.extend(find_branding_leaks_in_payload(item, f"{path}[{idx}]"))
    elif isinstance(obj, str):
        leaks.extend(find_branding_leaks_in_text(obj, path))
    return leaks


def assert_no_branding_leaks(payload: Any, label: str) -> None:
    leaks = find_branding_leaks_in_payload(payload, label)
    assert not leaks, f"AI branding leak detected in {label}:\n  " + "\n  ".join(leaks[:10])


def decode_base64_png(data_uri: str) -> Tuple[Image.Image, np.ndarray]:
    prefix = "data:image/png;base64,"
    assert isinstance(data_uri, str) and data_uri.startswith(prefix), (
        f"image_base64 must start with '{prefix}', got: {str(data_uri)[:60]!r}"
    )
    raw = base64.b64decode(data_uri[len(prefix):])
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    arr = np.asarray(img, dtype=np.float32)
    return img, arr


def validate_pixel_field_response(
    res: Dict[str, Any],
    label: str,
    require_chromatic: bool = True,
) -> Tuple[Image.Image, np.ndarray, Dict[str, float]]:
    assert res.get("mode") == "pixel_field", f"{label}: expected mode 'pixel_field', got {res.get('mode')}"
    assert "rgb_array" not in res, f"{label}: raw 'rgb_array' must not be present in JSON response"
    assert isinstance(res.get("latex"), dict) and len(res["latex"]) > 0, f"{label}: missing latex dict"
    assert "RGB" in res["latex"], f"{label}: missing 'RGB' key in latex"
    assert isinstance(res.get("python_code"), str) and len(res["python_code"]) > 40, f"{label}: missing python_code"
    ast.parse(res["python_code"])
    assert_no_branding_leaks(res, label)

    img, arr = decode_base64_png(res.get("image_base64", ""))
    h, w, c = arr.shape
    assert c == 3 and w >= 200 and h >= 120, f"{label}: unexpected image shape {arr.shape}"

    rgb_std = float(np.std(arr))
    rgb_mean = float(np.mean(arr))
    chroma_rg = float(np.std(arr[:, :, 0] - arr[:, :, 1]))
    chroma_rb = float(np.std(arr[:, :, 0] - arr[:, :, 2]))
    chroma_gb = float(np.std(arr[:, :, 1] - arr[:, :, 2]))
    max_chroma = max(chroma_rg, chroma_rb, chroma_gb)

    assert rgb_std > 20.0, f"{label}: std(rgb)={rgb_std:.2f} <= 20.0 (image too flat/empty)"
    assert 5.0 < rgb_mean < 250.0, f"{label}: mean(rgb)={rgb_mean:.2f} indicates all-black or all-white output"
    if require_chromatic:
        assert max_chroma > 8.0, (
            f"{label}: expected vibrant color painting, but max channel-diff std={max_chroma:.2f} <= 8.0"
        )

    stats = {
        "width": w,
        "height": h,
        "std": rgb_std,
        "mean": rgb_mean,
        "chroma": max_chroma,
    }
    return img, arr, stats


def validate_lines_response(res: Dict[str, Any], label: str, expected_n: Optional[int] = None) -> np.ndarray:
    assert res.get("mode") == "lines", f"{label}: expected mode 'lines', got {res.get('mode')}"
    prims = res.get("primitives")
    assert isinstance(prims, list) and len(prims) > 0, f"{label}: empty primitives"
    if expected_n is not None:
        assert len(prims) == expected_n, f"{label}: expected {expected_n} primitives, got {len(prims)}"
    arr = np.asarray(prims, dtype=np.float64)
    assert arr.ndim == 2 and arr.shape[1] == 4, f"{label}: expected (N, 4), got {arr.shape}"
    assert np.isfinite(arr).all(), f"{label}: NaN or Inf found in primitives"
    assert float(np.max(np.abs(arr))) <= 1.25, f"{label}: coordinates out of normalized range (max={np.max(np.abs(arr)):.3f})"
    assert float(np.std(arr)) > 0.05, f"{label}: degenerate/collapsed geometry (std={np.std(arr):.4f})"
    assert all(k in res.get("latex", {}) for k in ("X1", "Y1", "X2", "Y2")), f"{label}: missing X1/Y1/X2/Y2 latex"
    ast.parse(res["python_code"])
    assert_no_branding_leaks(res, label)
    return arr


def validate_circles_response(res: Dict[str, Any], label: str, expected_n: Optional[int] = None) -> np.ndarray:
    assert res.get("mode") == "circles", f"{label}: expected mode 'circles', got {res.get('mode')}"
    prims = res.get("primitives")
    assert isinstance(prims, list) and len(prims) > 0, f"{label}: empty primitives"
    if expected_n is not None:
        assert len(prims) == expected_n, f"{label}: expected {expected_n} primitives, got {len(prims)}"
    arr = np.asarray(prims, dtype=np.float64)
    assert arr.ndim == 2 and arr.shape[1] == 3, f"{label}: expected (N, 3), got {arr.shape}"
    assert np.isfinite(arr).all(), f"{label}: NaN or Inf found in circle primitives"
    assert np.all(arr[:, 2] > 0.0), f"{label}: non-positive circle radius detected"
    assert all(k in res.get("latex", {}) for k in ("X", "Y", "R")), f"{label}: missing X/Y/R latex"
    ast.parse(res["python_code"])
    assert_no_branding_leaks(res, label)
    return arr


def validate_3d_response(res: Dict[str, Any], label: str, expected_n: Optional[int] = None) -> np.ndarray:
    assert res.get("mode") == "3d_surface", f"{label}: expected mode '3d_surface', got {res.get('mode')}"
    prims_3d = res.get("primitives_3d")
    assert isinstance(prims_3d, list) and len(prims_3d) > 0, f"{label}: empty primitives_3d"
    if expected_n is not None:
        assert len(prims_3d) == expected_n, f"{label}: expected {expected_n} primitives_3d, got {len(prims_3d)}"
    arr = np.asarray(prims_3d, dtype=np.float64)
    assert arr.ndim == 2 and arr.shape[1] == 6, f"{label}: expected (N, 6) primitives_3d, got {arr.shape}"
    assert np.isfinite(arr).all(), f"{label}: NaN or Inf in primitives_3d"
    z_std = max(float(np.std(arr[:, 2])), float(np.std(arr[:, 5])))
    assert z_std > 0.01, f"{label}: 3D manifold has flat Z dimension (z_std={z_std:.5f})"
    assert all(k in res.get("latex", {}) for k in ("X1", "Y1", "Z1", "X2", "Y2", "Z2")), (
        f"{label}: missing 3D latex keys (got {list(res.get('latex', {}).keys())})"
    )
    ast.parse(res["python_code"])
    assert_no_branding_leaks(res, label)
    return arr


def validate_video_response(
    res: Dict[str, Any], label: str, expected_n: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    assert res.get("mode") == "math_video", f"{label}: expected mode 'math_video', got {res.get('mode')}"
    prims = res.get("primitives")
    th = res.get("time_harmonics")
    assert isinstance(prims, list) and len(prims) > 0, f"{label}: empty primitives"
    assert isinstance(th, list) and len(th) == len(prims), (
        f"{label}: time_harmonics length mismatch ({len(th) if isinstance(th, list) else type(th)} vs {len(prims)})"
    )
    if expected_n is not None:
        assert len(prims) == expected_n, f"{label}: expected {expected_n} primitives, got {len(prims)}"
    arr_p = np.asarray(prims, dtype=np.float64)
    arr_th = np.asarray(th, dtype=np.float64)
    assert arr_p.ndim == 2 and arr_p.shape[1] == 4, f"{label}: expected (N, 4) primitives, got {arr_p.shape}"
    assert arr_th.ndim == 2 and arr_th.shape[1] >= 8, f"{label}: expected (N, >=8) time_harmonics, got {arr_th.shape}"
    assert np.isfinite(arr_p).all() and np.isfinite(arr_th).all(), f"{label}: NaN or Inf in math_video data"
    motion_amp = float(np.max(np.abs(arr_th)))
    assert motion_amp > 0.005, f"{label}: static math_video (max time_harmonics amp={motion_amp:.5f})"
    assert all(k in res.get("latex", {}) for k in ("X1", "Y1", "X2", "Y2")), f"{label}: missing X1/Y1/X2/Y2 latex"
    latex_blob = " ".join(str(v) for v in res["latex"].values())
    assert ("\\sin(" in latex_blob or "\\cos(" in latex_blob) and "t" in latex_blob, (
        f"{label}: math_video LaTeX equations should include temporal harmonic dependence on t"
    )
    ast.parse(res["python_code"])
    assert_no_branding_leaks(res, label)
    return arr_p, arr_th


# ============================================================================
# VISUAL VERIFICATION SHEET RENDERERS (PIL — ZERO MATPLOTLIB DEPENDENCY)
# ============================================================================

def render_lines_preview(arr: np.ndarray, width: int = 480, height: int = 288) -> Image.Image:
    img = Image.new("RGB", (width, height), (5, 8, 17))
    draw = ImageDraw.Draw(img, "RGBA")
    cx, cy = width / 2.0, height / 2.0
    scale = min(width, height) * 0.43
    n = len(arr)
    for i, (x1, y1, x2, y2) in enumerate(arr):
        t = i / max(1, n - 1)
        r = int(40 + 180 * (0.5 + 0.5 * math.sin(t * math.pi * 2.0)))
        g = int(195 + 45 * math.cos(t * math.pi * 2.0))
        b = 248
        draw.line(
            [(cx + x1 * scale, cy - y1 * scale), (cx + x2 * scale, cy - y2 * scale)],
            fill=(r, g, b, 90),
            width=1,
        )
    return img


def render_circles_preview(arr: np.ndarray, width: int = 480, height: int = 288) -> Image.Image:
    img = Image.new("RGB", (width, height), (4, 10, 8))
    draw = ImageDraw.Draw(img, "RGBA")
    cx, cy = width / 2.0, height / 2.0
    scale = min(width, height) * 0.42
    n = len(arr)
    for i, (x, y, rad) in enumerate(arr):
        t = i / max(1, n - 1)
        px = cx + x * scale
        py = cy - y * scale
        pr = max(1.0, rad * scale)
        r = int(52 + 120 * t)
        g = int(215 - 35 * t)
        b = int(150 + 80 * (1.0 - t))
        draw.ellipse([px - pr, py - pr, px + pr, py + pr], outline=(r, g, b, 110), width=1)
    return img


def render_3d_preview(arr3d: np.ndarray, width: int = 480, height: int = 288) -> Image.Image:
    img = Image.new("RGB", (width, height), (6, 7, 13))
    draw = ImageDraw.Draw(img, "RGBA")
    cx, cy = width / 2.0, height / 2.0
    scale = min(width, height) * 0.40
    rot_x, rot_y, fov = -0.36, 0.58, 3.2
    cos_y, sin_y = math.cos(rot_y), math.sin(rot_y)
    cos_x, sin_x = math.cos(rot_x), math.sin(rot_x)

    def proj(x: float, y: float, z: float) -> Tuple[float, float, float]:
        x1 = x * cos_y + z * sin_y
        z1 = -x * sin_y + z * cos_y
        y2 = y * cos_x - z1 * sin_x
        z2 = y * sin_x + z1 * cos_x
        persp = fov / max(0.45, fov - z2)
        return cx + x1 * scale * persp, cy - y2 * scale * persp, z2

    projected = []
    n = len(arr3d)
    for i, (x1, y1, z1, x2, y2, z2) in enumerate(arr3d):
        sx1, sy1, zd1 = proj(x1, y1, z1)
        sx2, sy2, zd2 = proj(x2, y2, z2)
        projected.append((0.5 * (zd1 + zd2), i / max(1, n - 1), sx1, sy1, sx2, sy2))
    projected.sort(key=lambda item: item[0])

    for z_avg, t, sx1, sy1, sx2, sy2 in projected:
        depth = max(0.15, min(1.0, 0.55 + 0.42 * z_avg))
        r = int((145 + 95 * math.sin(t * math.pi * 2.0)) * depth)
        g = int((110 + 115 * math.cos(t * math.pi * 2.0 + 1.1)) * depth)
        b = int(245 * depth)
        draw.line([(sx1, sy1), (sx2, sy2)], fill=(r, g, b, 115), width=1)
    return img


def render_math_video_preview(
    arr_p: np.ndarray, arr_th: np.ndarray, width: int = 480, height: int = 288
) -> Image.Image:
    """Superimpose two temporal phases (t = 0 in cyan, t = pi/2 in gold) to visualize 4D motion."""
    img = Image.new("RGB", (width, height), (9, 7, 14))
    draw = ImageDraw.Draw(img, "RGBA")
    cx, cy = width / 2.0, height / 2.0
    scale = min(width, height) * 0.41

    phases = [
        (0.0, (56, 189, 248, 85)),
        (math.pi / 2.0, (251, 191, 36, 95)),
    ]
    for t_val, color in phases:
        st, ct = math.sin(t_val), math.cos(t_val)
        for (x1, y1, x2, y2), h in zip(arr_p, arr_th):
            px1 = x1 + h[0] * st + h[1] * ct
            py1 = y1 + h[2] * st + h[3] * ct
            px2 = x2 + h[4] * st + h[5] * ct
            py2 = y2 + h[6] * st + h[7] * ct
            draw.line(
                [(cx + px1 * scale, cy - py1 * scale), (cx + px2 * scale, cy - py2 * scale)],
                fill=color,
                width=1,
            )
    return img


def build_verification_sheet(
    panels: List[Tuple[str, str, Image.Image]],
    out_path: Path,
    banner: str,
    cols: int = 4,
) -> None:
    rows = math.ceil(len(panels) / cols)
    cell_w, cell_h = 480, 288
    header_h = 34
    pad = 14
    sheet_w = cols * cell_w + (cols + 1) * pad
    sheet_h = rows * (cell_h + header_h) + (rows + 1) * pad + 44

    sheet = Image.new("RGB", (sheet_w, sheet_h), (11, 15, 25))
    draw = ImageDraw.Draw(sheet)
    draw.text((pad, 14), banner, fill=(226, 232, 240))

    for idx, (title, subtitle, pil_img) in enumerate(panels):
        r = idx // cols
        c = idx % cols
        x0 = pad + c * (cell_w + pad)
        y0 = 44 + pad + r * (cell_h + header_h + pad)

        draw.rectangle([x0 - 1, y0 - 1, x0 + cell_w + 1, y0 + header_h + cell_h + 1], outline=(51, 65, 85), width=1)
        draw.rectangle([x0, y0, x0 + cell_w, y0 + header_h], fill=(15, 23, 42))
        draw.text((x0 + 8, y0 + 4), title[:58], fill=(56, 189, 248))
        draw.text((x0 + 8, y0 + 18), subtitle[:74], fill=(148, 163, 184))

        resized = pil_img.convert("RGB").resize((cell_w, cell_h), Image.Resampling.LANCZOS)
        sheet.paste(resized, (x0, y0 + header_h))

    sheet.save(out_path, format="PNG")


# ============================================================================
# FRONTEND BRANDING SCAN & PRODUCTION BUILD CHECK
# ============================================================================

def verify_frontend_zero_branding_and_build() -> None:
    print("\n9. Checking Frontend zero-AI-branding & production build (`npm run build`)...")
    frontend_files = [
        FRONTEND_DIR / "src" / "App.jsx",
        FRONTEND_DIR / "src" / "index.css",
    ]
    for fpath in frontend_files:
        assert fpath.exists(), f"Expected frontend file missing: {fpath}"
        content = fpath.read_text(encoding="utf-8")
        leaks = find_branding_leaks_in_text(content, str(fpath.name))
        assert not leaks, f"AI branding leak detected in {fpath}:\n  " + "\n  ".join(leaks[:10])
    print("   [PASS] Zero forbidden tokens in frontend/src/App.jsx and frontend/src/index.css")

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


# ============================================================================
# MAIN E2E TEST SUITE
# ============================================================================

def run_all_tests(live_url: Optional[str] = None) -> None:
    api = ApiAdapter(live_url=live_url)
    all_modes_panels: List[Tuple[str, str, Image.Image]] = []
    unseen_panels: List[Tuple[str, str, Image.Image]] = []
    preset_thumb_arrays: Dict[str, np.ndarray] = {}

    print("1. Testing /api/health & branding cleanliness...")
    health = api.get_json("/api/health")
    assert health.get("status") == "ok", f"Unexpected health payload: {health}"
    assert_no_branding_leaks(health, "/api/health")
    print(f"   [PASS] Health OK: {health.get('engine')}")

    print("\n2. Testing /api/presets catalog across all 5 modes & zero AI branding leak...")
    catalog = api.get_json("/api/presets")
    assert_no_branding_leaks(catalog, "/api/presets")
    presets = catalog.get("presets", [])
    preset_ids = {p["id"] for p in presets}
    modes_in_catalog = {p["mode"] for p in presets}

    assert set(REQUIRED_PIXEL_FIELD_PRESETS).issubset(preset_ids), (
        f"Missing Pixel-Field presets: {set(REQUIRED_PIXEL_FIELD_PRESETS) - preset_ids}"
    )
    assert REQUIRED_SCIAM_PRESETS.issubset(preset_ids), (
        f"Missing SciAm presets: {REQUIRED_SCIAM_PRESETS - preset_ids}"
    )
    assert REQUIRED_3D_PRESETS.issubset(preset_ids), (
        f"Missing 3D presets: {REQUIRED_3D_PRESETS - preset_ids}"
    )
    assert REQUIRED_VIDEO_PRESETS.issubset(preset_ids), (
        f"Missing 4D Video presets: {REQUIRED_VIDEO_PRESETS - preset_ids}"
    )
    expected_modes = {"lines", "circles", "pixel_field", "3d_surface", "math_video"}
    assert expected_modes.issubset(modes_in_catalog), (
        f"Missing modes in /api/presets: {expected_modes - modes_in_catalog}"
    )
    print(f"   [PASS] {len(presets)} presets verified covering all 5 modes: {sorted(modes_in_catalog)}")

    print("\n3. Testing all 7 Hamid Naderi Yeganeh 2D Pixel-Field paintings via /api/preset...")
    for pid in REQUIRED_PIXEL_FIELD_PRESETS:
        res = api.post_json("/api/preset", {"preset_id": pid})
        img, arr, stats = validate_pixel_field_response(
            res,
            label=f"/api/preset:{pid}",
            require_chromatic=(pid in CHROMATIC_PIXEL_FIELD_PRESETS),
        )
        preset_thumb_arrays[pid] = np.asarray(img.resize((120, 72), Image.Resampling.BILINEAR), dtype=np.float32)
        all_modes_panels.append(
            (
                f"[pixel_field] {res['title']}",
                f"{pid} | {stats['width']}x{stats['height']} | std={stats['std']:.1f} mean={stats['mean']:.1f} chroma={stats['chroma']:.1f}",
                img,
            )
        )
        print(
            f"   [PASS] {pid:18s} -> {stats['width']}x{stats['height']} "
            f"std={stats['std']:.1f}, mean={stats['mean']:.1f}, chroma={stats['chroma']:.1f}"
        )

    print("\n4. Testing ALL remaining presets (lines, circles, 3d_surface, math_video) via /api/preset...")
    preview_lines_img = None
    preview_circles_img = None
    preview_3d_img = None
    preview_video_img = None

    for p in presets:
        pid = p["id"]
        pmode = p["mode"]
        if pmode == "pixel_field":
            continue
        target_n = 1200
        res = api.post_json("/api/preset", {"preset_id": pid, "N": target_n})
        if pmode == "lines":
            arr = validate_lines_response(res, f"/api/preset:{pid}", expected_n=target_n)
            if pid == "bird_in_flight":
                preview_lines_img = render_lines_preview(arr)
        elif pmode == "circles":
            arr = validate_circles_response(res, f"/api/preset:{pid}", expected_n=target_n)
            if pid == "olive_branch_sciam":
                preview_circles_img = render_circles_preview(arr)
        elif pmode == "3d_surface":
            arr3d = validate_3d_response(res, f"/api/preset:{pid}", expected_n=target_n)
            if pid == "3d_rose_bloom":
                preview_3d_img = render_3d_preview(arr3d)
        elif pmode == "math_video":
            arr_p, arr_th = validate_video_response(res, f"/api/preset:{pid}", expected_n=target_n)
            if pid == "video_soaring_bird":
                preview_video_img = render_math_video_preview(arr_p, arr_th)
        elif pmode == "raymarch_cinema":
            assert_no_branding_leaks(res, f"/api/preset:{pid}")
            assert res.get("mode") == "raymarch_cinema", f"/api/preset:{pid}: wrong mode {res.get('mode')}"
            ok_g, g_err = smc.validate_glsl_webgl1_static(res.get("glsl_fragment_shader", ""))
            assert ok_g, f"/api/preset:{pid}: WebGL1 GLSL invalid: {g_err}"
            _, arr_rm = decode_base64_png(res.get("image_base64", ""))
            assert float(np.std(arr_rm)) > 20.0, f"/api/preset:{pid}: low std={np.std(arr_rm):.2f}"
            assert isinstance(res.get("primitives_3d"), list) and len(res["primitives_3d"]) > 0
            assert isinstance(res.get("latex"), dict) and len(res["latex"]) > 0
            ast.parse(res["python_code"])
        else:
            raise AssertionError(f"Unknown preset mode {pmode!r} for {pid}")
        print(f"   [PASS] {pid:26s} [{pmode:10s}] -> {res['title']}")

    print("\n5. Testing silent flagship multi-tier router (`select_model_tier`) & Simple/Medium/Complex prompts...")
    simple_tier = ai_architect.select_model_tier("star")
    medium_tier = ai_architect.select_model_tier("blooming rose flower")
    complex_tier = ai_architect.select_model_tier("Faisal Mosque Islamabad with crescent moon")
    for tier_name, tier_models in (("simple", simple_tier), ("medium", medium_tier), ("complex", complex_tier)):
        assert isinstance(tier_models, list) and len(tier_models) >= 2, f"{tier_name} tier empty: {tier_models}"
        assert all(m.startswith("@cf/") for m in tier_models), f"{tier_name} has invalid model id: {tier_models}"
        assert not any("8b" in m.lower() for m in tier_models), (
            f"{tier_name} tier must use flagship models (no tiny 8B models): {tier_models}"
        )
    print(f"   [PASS] Flagship tiers verified (simple={simple_tier[0]}, medium={medium_tier[0]}, complex={complex_tier[0]})")

    standard_prompt_tests = [
        {"tier": "simple", "prompt": "star", "N": 1200, "harmonics": 64, "mode": "lines", "use_ai": True},
        {"tier": "simple", "prompt": "lotus", "N": 1200, "harmonics": 64, "mode": "circles", "use_ai": False},
        {"tier": "medium", "prompt": "japanese maple leaf", "N": 1400, "harmonics": 72, "mode": "lines", "use_ai": False},
        {"tier": "medium", "prompt": "olive branch zaitoon", "N": 1400, "harmonics": 72, "mode": "circles", "use_ai": False},
        {"tier": "complex", "prompt": "Faisal Mosque Islamabad with crescent moon", "N": 1600, "harmonics": 84, "mode": "lines", "use_ai": True},
        {"tier": "complex", "prompt": "A detailed mechanical bicycle", "N": 1600, "harmonics": 84, "mode": "math_video", "use_ai": True},
    ]

    for q in standard_prompt_tests:
        payload = {k: v for k, v in q.items() if k != "tier"}
        label = f"/api/generate-from-prompt[{q['tier']}:{q['mode']}:{q['prompt']}]"
        res = api.post_json("/api/generate-from-prompt", payload)
        qmode = q["mode"]
        if qmode == "lines":
            validate_lines_response(res, label, expected_n=q["N"])
        elif qmode == "circles":
            validate_circles_response(res, label, expected_n=q["N"])
        elif qmode == "math_video":
            validate_video_response(res, label, expected_n=q["N"])
        print(f"   [PASS] ({q['tier']:7s} | {qmode:11s}) '{q['prompt']}' -> {res['title']}")

    print("\n6. Testing BRAND-NEW UNSEEN arbitrary prompts across pixel_field, 3d_surface, math_video & lines...")
    unseen_pf_thumbs: Dict[str, np.ndarray] = {}
    unseen_latex_fingerprints: List[str] = []

    for idx, unseen_prompt in enumerate(UNSEEN_NOVEL_PROMPTS):
        # 6a. Full-color 2D Pixel-Field synthesis
        pf_payload: Dict[str, Any] = {
            "prompt": unseen_prompt,
            "N": 1600,
            "harmonics": 84,
            "mode": "pixel_field",
            "use_ai": True,
        }
        # Also verify animated pixel_field t_phase on first unseen prompt
        if idx == 0:
            pf_payload["t_phase"] = 0.85
        res_pf = api.post_json("/api/generate-from-prompt", pf_payload)
        img_pf, arr_pf, stats_pf = validate_pixel_field_response(
            res_pf,
            label=f"unseen[pixel_field:{unseen_prompt}]",
            require_chromatic=True,
        )
        thumb = np.asarray(img_pf.resize((120, 72), Image.Resampling.BILINEAR), dtype=np.float32)

        # Ensure this unseen prompt painting is NOT a copy-pasted preset
        for pid, p_thumb in preset_thumb_arrays.items():
            mad = float(np.mean(np.abs(thumb - p_thumb)))
            assert mad > 10.0, f"Unseen prompt '{unseen_prompt}' looks suspiciously identical to preset '{pid}' (MAD={mad:.2f})"
        # Ensure distinct from other unseen prompts
        for other_p, other_thumb in unseen_pf_thumbs.items():
            mad = float(np.mean(np.abs(thumb - other_thumb)))
            assert mad > 10.0, f"Unseen prompt '{unseen_prompt}' too close to '{other_p}' (MAD={mad:.2f})"
        unseen_pf_thumbs[unseen_prompt] = thumb

        # Prompt-specific LaTeX check (non-boilerplate)
        latex_fp = json.dumps(res_pf["latex"], sort_keys=True)
        unseen_latex_fingerprints.append(latex_fp)
        unseen_panels.append(
            (
                f"[pixel_field] {res_pf['title']}",
                f"{unseen_prompt[:42]} | std={stats_pf['std']:.1f} chroma={stats_pf['chroma']:.1f}",
                img_pf,
            )
        )
        if idx == 0:
            all_modes_panels.append(
                (
                    f"[unseen->pixel_field] {res_pf['title']}",
                    f"{unseen_prompt[:42]} | std={stats_pf['std']:.1f} chroma={stats_pf['chroma']:.1f}",
                    img_pf,
                )
            )
        print(
            f"   [PASS] UNSEEN [pixel_field] '{unseen_prompt}' -> {res_pf['title']} "
            f"(std={stats_pf['std']:.1f}, chroma={stats_pf['chroma']:.1f}, latex_keys={list(res_pf['latex'].keys())})"
        )

    # Confirm all 4 unseen prompts produced unique prompt-specific LaTeX
    assert len(set(unseen_latex_fingerprints)) == len(UNSEEN_NOVEL_PROMPTS), (
        "Expected distinct prompt-specific LaTeX equations for each unseen prompt, found duplicates!"
    )
    print("   [PASS] All 4 unseen prompts generated distinct, prompt-specific LaTeX equation systems")

    # 6b. Test unseen prompts across lines, 3d_surface, and math_video modes
    unseen_multimode = [
        ("lines", UNSEEN_NOVEL_PROMPTS[0]),       # glowing jellyfish -> lines
        ("3d_surface", UNSEEN_NOVEL_PROMPTS[1]),  # astronaut near Saturn rings -> 3D manifold
        ("math_video", UNSEEN_NOVEL_PROMPTS[2]),  # snowy mountain peak -> 4D math video
        ("3d_surface", UNSEEN_NOVEL_PROMPTS[3]),  # roaring Bengal tiger -> 3D manifold
        ("math_video", UNSEEN_NOVEL_PROMPTS[0]),  # glowing jellyfish -> 4D math video
        ("lines", UNSEEN_NOVEL_PROMPTS[3]),       # roaring Bengal tiger -> lines
    ]
    for umode, uprompt in unseen_multimode:
        res_u = api.post_json(
            "/api/generate-from-prompt",
            {"prompt": uprompt, "N": 1400, "harmonics": 80, "mode": umode, "use_ai": True},
        )
        lbl = f"unseen[{umode}:{uprompt}]"
        if umode == "lines":
            arr_l = validate_lines_response(res_u, lbl, expected_n=1400)
            if len(unseen_panels) < 8:
                unseen_panels.append((f"[lines] {res_u['title']}", uprompt[:60], render_lines_preview(arr_l)))
        elif umode == "3d_surface":
            arr_3 = validate_3d_response(res_u, lbl, expected_n=1400)
            if len(unseen_panels) < 8:
                unseen_panels.append((f"[3d_surface] {res_u['title']}", uprompt[:60], render_3d_preview(arr_3)))
        elif umode == "math_video":
            arr_p, arr_th = validate_video_response(res_u, lbl, expected_n=1400)
            if len(unseen_panels) < 8:
                unseen_panels.append((f"[math_video] {res_u['title']}", uprompt[:60], render_math_video_preview(arr_p, arr_th)))
        print(f"   [PASS] UNSEEN [{umode:10s}] '{uprompt}' -> {res_u['title']}")

    print("\n7. Testing /api/evaluate-custom for lines & circles modes...")
    custom_lines = api.post_json(
        "/api/evaluate-custom",
        {
            "mode": "lines",
            "N": 1000,
            "formulas": {
                "X1": "np.sin(2*np.pi*k/N)**3",
                "Y1": "np.cos(2*np.pi*k/N)",
                "X2": "0.5*np.sin(6*np.pi*k/N)",
                "Y2": "0.5*np.cos(4*np.pi*k/N)",
            },
        },
    )
    validate_lines_response(custom_lines, "/api/evaluate-custom:lines", expected_n=1000)

    custom_circles = api.post_json(
        "/api/evaluate-custom",
        {
            "mode": "circles",
            "N": 800,
            "formulas": {
                "X": "0.6*np.cos(4*np.pi*k/N)",
                "Y": "0.6*np.sin(4*np.pi*k/N)",
                "R": "0.05 + 0.12*(np.sin(12*np.pi*k/N)**2)",
            },
        },
    )
    validate_circles_response(custom_circles, "/api/evaluate-custom:circles", expected_n=800)
    print("   [PASS] /api/evaluate-custom OK for both lines and circles modes")

    print(f"\n8. Saving visual verification sheets...")
    if preview_lines_img is not None:
        all_modes_panels.append(("[lines] A Bird in Flight (Parinda)", "bird_in_flight | N=1200 | 2D Line Segments", preview_lines_img))
    if preview_circles_img is not None:
        all_modes_panels.append(("[circles] Olive Branch (SciAm)", "olive_branch_sciam | N=1200 | 2D Circle Families", preview_circles_img))
    if preview_3d_img is not None:
        all_modes_panels.append(("[3d_surface] 3D Rose Corolla Manifold", "3d_rose_bloom | N=1200 | 3D Perspective Orbit", preview_3d_img))
    if preview_video_img is not None:
        all_modes_panels.append(("[math_video] 4D Flapping Bird (t=0 & t=pi/2)", "video_soaring_bird | N=1200 | 4D Time Harmonics", preview_video_img))

    build_verification_sheet(
        all_modes_panels,
        SHEET_ALL_MODES_PATH,
        banner="DesignWithMath — Verified E2E Sheet: All 7 Hamid 2D Pixel-Field Paintings + All 5 Mathematical Modes",
        cols=4,
    )
    assert SHEET_ALL_MODES_PATH.exists() and SHEET_ALL_MODES_PATH.stat().st_size > 20_000

    build_verification_sheet(
        unseen_panels,
        SHEET_UNSEEN_PATH,
        banner="DesignWithMath — Verified Unseen Novel Arbitrary Prompts (Pixel-Field, Lines, 3D Manifold & 4D Video)",
        cols=4,
    )
    assert SHEET_UNSEEN_PATH.exists() and SHEET_UNSEEN_PATH.stat().st_size > 20_000
    print(f"   [PASS] Saved {SHEET_ALL_MODES_PATH} ({len(all_modes_panels)} panels, {SHEET_ALL_MODES_PATH.stat().st_size:,} bytes)")
    print(f"   [PASS] Saved {SHEET_UNSEEN_PATH} ({len(unseen_panels)} panels, {SHEET_UNSEEN_PATH.stat().st_size:,} bytes)")

    verify_frontend_zero_branding_and_build()

    print("\n======================================================================")
    print("ALL E2E, VISUAL, 5-MODE, UNSEEN-PROMPT, ZERO-BRANDING & BUILD TESTS PASSED!")
    print("======================================================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run DesignWithMath E2E Verification Suite")
    parser.add_argument("--live-url", type=str, default=None, help="Optional live FastAPI base URL (e.g. http://127.0.0.1:8000)")
    args = parser.parse_args()
    run_all_tests(live_url=args.live_url)
