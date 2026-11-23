import math
import os
import re
import tempfile
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from ai_architect import compile_ir_component, safe_eval_sdf_expr


# ============================================================================
# 0. METADATA SANITIZER (PURE MATHEMATICAL TERMINOLOGY ONLY)
# ============================================================================

_FORBIDDEN_BRAND_PATTERNS = re.compile(
    r"\b(cloudflare|workers\s*ai|llama[-\d\.]*|qwen[-\d\.]*|qwq[-\d\.]*|deepseek[-\w\.]*|glm[-\w\.]*|kimi[-\w\.]*|gpt[-\w\.]*|gemma[-\w\.]*|mistral[-\w\.]*|nemotron[-\w\.]*|120b|70b|32b|27b|26b|24b|17b|12b|8b|fp8|instruct|moonshot|zhipu|openai|anthropic|gemini|vision\s*critic|vlm|llm|ai\s+architect|ai\s+synthesis|ai\s+engine|ai\b)\b",
    re.IGNORECASE,
)


def sanitize_math_text(text: str, fallback: str = "Harmonic Differential-Geometric Synthesis") -> str:
    if not text:
        return fallback
    cleaned = _FORBIDDEN_BRAND_PATTERNS.sub("", str(text))
    cleaned = re.sub(r"\(\s*\)", "", cleaned)
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip(" -—:+|/,")
    return cleaned if len(cleaned) >= 3 else fallback


def sanitize_math_program(prog: Dict[str, Any]) -> Dict[str, Any]:
    """Returns a strictly canonical, branding-free copy of a symbolic math_program."""
    clean: Dict[str, Any] = {
        "title": sanitize_math_text(prog.get("title", ""), "Symbolic Mathematical Synthesis"),
        "subtitle": sanitize_math_text(
            prog.get("subtitle", ""),
            "Unified Analytical SDF & Double-Exponential Level-Set System",
        ),
        "aspect": list(prog.get("aspect", [16, 10])),
        "domain": dict(prog.get("domain", {"x_min": -1.6, "x_max": 1.6, "y_min": -1.0, "y_max": 1.0})),
        "steps": prog.get("steps", []),
        "H0": str(prog.get("H0", "0.0")),
        "H1": str(prog.get("H1", "0.0")),
        "H2": str(prog.get("H2", "0.0")),
    }
    return clean


# ============================================================================
# 1. FOURIER TRIGONOMETRIC DECOMPOSITION FOR 3D & 4D MANIFOLDS
# ============================================================================

def fit_fourier_series_1d(
    signal: np.ndarray,
    num_harmonics: int = 64,
) -> Tuple[float, List[Dict[str, float]]]:
    n = len(signal)
    fft_vals = np.fft.rfft(signal) / float(n)
    a0 = float(np.real(fft_vals[0]))

    raw_harmonics: List[Dict[str, float]] = []
    max_m = len(fft_vals) - 1
    for m in range(1, max_m + 1):
        coeff = fft_vals[m]
        scale = 2.0 if (m < max_m or n % 2 != 0) else 1.0
        a_m = float(scale * np.real(coeff))
        b_m = float(-scale * np.imag(coeff))
        amp = math.hypot(a_m, b_m)
        if amp > 5e-5:
            raw_harmonics.append({
                "m": int(m),
                "a": round(a_m, 5),
                "b": round(b_m, 5),
                "amp": round(amp, 5),
            })

    raw_harmonics.sort(key=lambda h: h["amp"], reverse=True)
    selected = raw_harmonics[: max(1, min(num_harmonics, len(raw_harmonics)))]
    selected.sort(key=lambda h: h["m"])
    return round(a0, 5), selected


def eval_fourier_series_1d(
    a0: float,
    harmonics: List[Dict[str, float]],
    k: np.ndarray,
    N: int,
) -> np.ndarray:
    res = np.full_like(k, fill_value=a0, dtype=np.float64)
    base_angle = (2.0 * np.pi * k) / float(N)
    for h in harmonics:
        m = h["m"]
        a = h["a"]
        b = h["b"]
        angle = m * base_angle
        if abs(a) >= 5e-5:
            res += a * np.cos(angle)
        if abs(b) >= 5e-5:
            res += b * np.sin(angle)
    return res


def format_3d_series_to_latex(
    var_name: str,
    a0: float,
    harmonics: List[Dict[str, float]],
    N: int,
    max_display_terms: int = 6,
) -> str:
    terms: List[str] = []
    if abs(a0) >= 0.005:
        terms.append(f"{a0:.3f}")

    sorted_by_amp = sorted(harmonics, key=lambda h: h["amp"], reverse=True)
    top_subset = sorted(sorted_by_amp[:max_display_terms], key=lambda h: h["m"])

    for h in top_subset:
        m = h["m"]
        a = h["a"]
        b = h["b"]
        freq_coeff = 2 * m
        angle_str = rf"\frac{{{freq_coeff}\pi k}}{{{N}}}"
        if abs(a) >= 0.005:
            sign = "+" if (a > 0 and len(terms) > 0) else ("-" if a < 0 else "")
            terms.append(rf"{sign} {abs(a):.3f}\cos\left({angle_str}\right)")
        if abs(b) >= 0.005:
            sign = "+" if (b > 0 and len(terms) > 0) else ("-" if b < 0 else "")
            terms.append(rf"{sign} {abs(b):.3f}\sin\left({angle_str}\right)")

    remaining = len(harmonics) - len(top_subset)
    if not terms:
        expr = "0"
    else:
        expr = " ".join(terms).lstrip("+ ")
        if remaining > 0:
            expr += rf" + \cdots \text{{ (+{remaining} harmonics)}}"

    pretty_var = {
        "X1": "X_1",
        "Y1": "Y_1",
        "Z1": "Z_1",
        "X2": "X_2",
        "Y2": "Y_2",
        "Z2": "Z_2",
    }.get(var_name, var_name)
    return rf"{pretty_var}(k) = {expr}"


def format_series_to_py_expr(a0: float, harmonics: List[Dict[str, float]], max_terms: int = 40) -> str:
    parts: List[str] = []
    if abs(a0) >= 5e-5:
        parts.append(f"{a0:.4f}")
    sorted_by_amp = sorted(harmonics, key=lambda h: h["amp"], reverse=True)
    subset = sorted(sorted_by_amp[:max_terms], key=lambda h: h["m"])
    for h in subset:
        m = h["m"]
        a = h["a"]
        b = h["b"]
        freq = 2 * m
        if abs(a) >= 5e-5:
            sign = " + " if (a > 0 and parts) else (" - " if a < 0 else "")
            parts.append(f"{sign}{abs(a):.4f}*np.cos({freq}*np.pi*k/N)")
        if abs(b) >= 5e-5:
            sign = " + " if (b > 0 and parts) else (" - " if b < 0 else "")
            parts.append(f"{sign}{abs(b):.4f}*np.sin({freq}*np.pi*k/N)")
    return "".join(parts) if parts else "0.0 * k"


# ============================================================================
# 2. BUILT-IN ICONIC 3D PARAMETRIC SURFACE & SPACE-CURVE PRESETS
# ============================================================================

MATH_3D_PRESETS: List[Dict[str, Any]] = [
    {
        "id": "3d_rose_bloom",
        "name": "3D Parametric Rose Corolla (Gulab Manifold)",
        "author": "Differential-Geometric Harmonic Series",
        "category": "3D Manifolds & 4D Math Video",
        "mode": "3d_surface",
        "N": 2200,
        "description": "3D layered rose corolla surface woven from 2200 spatial chords with 5-fold petal harmonics and helical Z-elevation.",
        "latex": {
            "X1": r"X_1(k) = \left(\frac{13}{20} + \frac{7}{20}\cos^2\left(\frac{5\pi k}{N}\right)\right)\cos\left(\frac{2\pi k}{N}\right)\left(1 - \frac{1}{4}\sin^2\left(\frac{3\pi k}{N}\right)\right)",
            "Y1": r"Y_1(k) = \left(\frac{13}{20} + \frac{7}{20}\cos^2\left(\frac{5\pi k}{N}\right)\right)\sin\left(\frac{2\pi k}{N}\right)\left(1 - \frac{1}{4}\sin^2\left(\frac{3\pi k}{N}\right)\right)",
            "Z1": r"Z_1(k) = \frac{11}{20}\sin\left(\frac{5\pi k}{N}\right)\cos\left(\frac{2\pi k}{N}\right) + \frac{1}{4}\cos^2\left(\frac{10\pi k}{N}\right)",
            "X2": r"X_2(k) = \frac{19}{50}\left(\frac{3}{5} + \frac{2}{5}\sin^2\left(\frac{5\pi k}{N} + \frac{\pi}{4}\right)\right)\cos\left(\frac{4\pi k}{N} + \frac{3}{10}\right)",
            "Y2": r"Y_2(k) = \frac{19}{50}\left(\frac{3}{5} + \frac{2}{5}\sin^2\left(\frac{5\pi k}{N} + \frac{\pi}{4}\right)\right)\sin\left(\frac{4\pi k}{N} + \frac{3}{10}\right)",
            "Z2": r"Z_2(k) = -\frac{7}{20} + \frac{3}{10}\cos\left(\frac{5\pi k}{N}\right)",
        },
        "formulas": {
            "X1": "(0.65 + 0.35*(np.cos(5*np.pi*k/N)**2))*np.cos(2*np.pi*k/N)*(1.0 - 0.25*(np.sin(3*np.pi*k/N)**2))",
            "Y1": "(0.65 + 0.35*(np.cos(5*np.pi*k/N)**2))*np.sin(2*np.pi*k/N)*(1.0 - 0.25*(np.sin(3*np.pi*k/N)**2))",
            "Z1": "0.55*np.sin(5*np.pi*k/N)*np.cos(2*np.pi*k/N) + 0.25*(np.cos(10*np.pi*k/N)**2)",
            "X2": "0.38*(0.6 + 0.4*(np.sin(5*np.pi*k/N + np.pi/4)**2))*np.cos(4*np.pi*k/N + 0.3)",
            "Y2": "0.38*(0.6 + 0.4*(np.sin(5*np.pi*k/N + np.pi/4)**2))*np.sin(4*np.pi*k/N + 0.3)",
            "Z2": "-0.35 + 0.30*np.cos(5*np.pi*k/N)",
        },
    },
    {
        "id": "3d_torus_knot",
        "name": "3D Harmonic Torus Caustic Manifold",
        "author": "Differential-Geometric Harmonic Series",
        "category": "3D Manifolds & 4D Math Video",
        "mode": "3d_surface",
        "N": 2400,
        "description": "3D (3,5)-Torus knot caustic envelope spanned by phase-shifted inner and outer toroidal space curves.",
        "latex": {
            "X1": r"X_1(k) = \left(\frac{17}{25} + \frac{3}{10}\cos\left(\frac{10\pi k}{N}\right)\right)\cos\left(\frac{6\pi k}{N}\right)",
            "Y1": r"Y_1(k) = \left(\frac{17}{25} + \frac{3}{10}\cos\left(\frac{10\pi k}{N}\right)\right)\sin\left(\frac{6\pi k}{N}\right)",
            "Z1": r"Z_1(k) = \frac{21}{50}\sin\left(\frac{10\pi k}{N}\right) + \frac{3}{25}\cos\left(\frac{20\pi k}{N}\right)",
            "X2": r"X_2(k) = \left(\frac{17}{25} + \frac{3}{10}\cos\left(\frac{10\pi k}{N} + \frac{9}{20}\right)\right)\cos\left(\frac{6\pi k}{N} + \frac{7}{25}\right)",
            "Y2": r"Y_2(k) = \left(\frac{17}{25} + \frac{3}{10}\cos\left(\frac{10\pi k}{N} + \frac{9}{20}\right)\right)\sin\left(\frac{6\pi k}{N} + \frac{7}{25}\right)",
            "Z2": r"Z_2(k) = \frac{21}{50}\sin\left(\frac{10\pi k}{N} + \frac{9}{20}\right) - \frac{3}{25}\sin\left(\frac{20\pi k}{N}\right)",
        },
        "formulas": {
            "X1": "(0.68 + 0.30*np.cos(10*np.pi*k/N))*np.cos(6*np.pi*k/N)",
            "Y1": "(0.68 + 0.30*np.cos(10*np.pi*k/N))*np.sin(6*np.pi*k/N)",
            "Z1": "0.42*np.sin(10*np.pi*k/N) + 0.12*np.cos(20*np.pi*k/N)",
            "X2": "(0.68 + 0.30*np.cos(10*np.pi*k/N + 0.45))*np.cos(6*np.pi*k/N + 0.28)",
            "Y2": "(0.68 + 0.30*np.cos(10*np.pi*k/N + 0.45))*np.sin(6*np.pi*k/N + 0.28)",
            "Z2": "0.42*np.sin(10*np.pi*k/N + 0.45) - 0.12*np.sin(20*np.pi*k/N)",
        },
    },
    {
        "id": "3d_bird_wings",
        "name": "3D Soaring Bird Wing Surface (3D Parinda)",
        "author": "Hamid Naderi Yeganeh 3D Extension",
        "category": "3D Manifolds & 4D Math Video",
        "mode": "3d_surface",
        "N": 2000,
        "description": "3D volumetric extension of A Bird in Flight with dihedral wing arch Z₁(k) and tail camber Z₂(k).",
        "latex": {
            "X1": r"X_1(k) = \frac{3}{2}\sin^7\left(\frac{2\pi k}{N} + \frac{\pi}{3}\right)",
            "Y1": r"Y_1(k) = \frac{1}{4}\cos^2\left(\frac{6\pi k}{N}\right)",
            "Z1": r"Z_1(k) = \frac{13}{20}\sin^4\left(\frac{2\pi k}{N} + \frac{\pi}{3}\right) - \frac{11}{50}\cos\left(\frac{4\pi k}{N}\right)",
            "X2": r"X_2(k) = \frac{1}{5}\sin\left(\frac{6\pi k}{N} + \frac{\pi}{5}\right)",
            "Y2": r"Y_2(k) = -\frac{2}{3}\sin^2\left(\frac{4\pi k}{N} - \frac{\pi}{3}\right)",
            "Z2": r"Z_2(k) = -\frac{9}{50}\sin\left(\frac{4\pi k}{N}\right) + \frac{3}{20}\cos^3\left(\frac{2\pi k}{N}\right)",
        },
        "formulas": {
            "X1": "1.5 * (np.sin(2*np.pi*k/N + np.pi/3)**7)",
            "Y1": "0.25 * (np.cos(6*np.pi*k/N)**2)",
            "Z1": "0.65 * (np.sin(2*np.pi*k/N + np.pi/3)**4) - 0.22 * np.cos(4*np.pi*k/N)",
            "X2": "0.2 * np.sin(6*np.pi*k/N + np.pi/5)",
            "Y2": "-(2.0/3.0) * (np.sin(4*np.pi*k/N - np.pi/3)**2)",
            "Z2": "-0.18 * np.sin(4*np.pi*k/N) + 0.15 * (np.cos(2*np.pi*k/N)**3)",
        },
    },
    {
        "id": "3d_nautilus_shell",
        "name": "3D Logarithmic Helicospiral Shell",
        "author": "Differential-Geometric Harmonic Series",
        "category": "3D Manifolds & 4D Math Video",
        "mode": "3d_surface",
        "N": 2200,
        "description": "3D chambered Nautilus shell surface generated by radially modulated helicospiral chordribs.",
        "latex": {
            "X1": r"X_1(k) = \left(\frac{9}{50} + \frac{39}{50}\frac{k}{N}\right)\cos\left(\frac{6\pi k}{N}\right)\left(1 + \frac{11}{50}\cos\left(\frac{24\pi k}{N}\right)\right)",
            "Y1": r"Y_1(k) = \left(\frac{9}{50} + \frac{39}{50}\frac{k}{N}\right)\sin\left(\frac{6\pi k}{N}\right)\left(1 + \frac{11}{50}\cos\left(\frac{24\pi k}{N}\right)\right)",
            "Z1": r"Z_1(k) = \frac{11}{20}\left(\frac{k}{N} - \frac{1}{2}\right) + \frac{6}{25}\left(\frac{9}{50} + \frac{39}{50}\frac{k}{N}\right)\sin\left(\frac{24\pi k}{N}\right)",
            "X2": r"X_2(k) = \left(\frac{3}{25} + \frac{14}{25}\frac{k}{N}\right)\cos\left(\frac{6\pi k}{N} - \frac{11}{50}\right)",
            "Y2": r"Y_2(k) = \left(\frac{3}{25} + \frac{14}{25}\frac{k}{N}\right)\sin\left(\frac{6\pi k}{N} - \frac{11}{50}\right)",
            "Z2": r"Z_2(k) = \frac{11}{20}\left(\frac{k}{N} - \frac{1}{2}\right) - \frac{4}{25}\left(\frac{3}{25} + \frac{14}{25}\frac{k}{N}\right)\cos\left(\frac{24\pi k}{N}\right)",
        },
        "formulas": {
            "X1": "(0.18 + 0.78*(k/N))*np.cos(6*np.pi*k/N)*(1.0 + 0.22*np.cos(24*np.pi*k/N))",
            "Y1": "(0.18 + 0.78*(k/N))*np.sin(6*np.pi*k/N)*(1.0 + 0.22*np.cos(24*np.pi*k/N))",
            "Z1": "0.55*((k/N) - 0.5) + 0.24*(0.18 + 0.78*(k/N))*np.sin(24*np.pi*k/N)",
            "X2": "(0.12 + 0.56*(k/N))*np.cos(6*np.pi*k/N - 0.22)",
            "Y2": "(0.12 + 0.56*(k/N))*np.sin(6*np.pi*k/N - 0.22)",
            "Z2": "0.55*((k/N) - 0.5) - 0.16*(0.12 + 0.56*(k/N))*np.cos(24*np.pi*k/N)",
        },
    },
    {
        "id": "3d_klein_caustic",
        "name": "3D Harmonic Wave Dome (Celestial Vault)",
        "author": "Differential-Geometric Harmonic Series",
        "category": "3D Manifolds & 4D Math Video",
        "mode": "3d_surface",
        "N": 2200,
        "description": "3D architectural ribbed dome manifold with fluted meridian waves and inner caustic oculus ring.",
        "latex": {
            "X1": r"X_1(k) = \sin\left(\frac{\pi k}{N}\right)\left(\frac{3}{4} + \frac{11}{50}\cos\left(\frac{12\pi k}{N}\right)\right)\cos\left(\frac{8\pi k}{N}\right)",
            "Y1": r"Y_1(k) = \cos\left(\frac{\pi k}{N}\right) + \frac{9}{50}\sin\left(\frac{6\pi k}{N}\right)\sin\left(\frac{\pi k}{N}\right)",
            "Z1": r"Z_1(k) = \sin\left(\frac{\pi k}{N}\right)\left(\frac{3}{4} + \frac{11}{50}\cos\left(\frac{12\pi k}{N}\right)\right)\sin\left(\frac{8\pi k}{N}\right)",
            "X2": r"X_2(k) = \frac{31}{50}\sin\left(\frac{\pi k}{N}\right)\cos\left(\frac{8\pi k}{N} + \frac{7}{20}\right)",
            "Y2": r"Y_2(k) = \frac{17}{20}\cos\left(\frac{\pi k}{N}\right) - \frac{3}{20}\cos\left(\frac{6\pi k}{N}\right)",
            "Z2": r"Z_2(k) = \frac{31}{50}\sin\left(\frac{\pi k}{N}\right)\sin\left(\frac{8\pi k}{N} + \frac{7}{20}\right)",
        },
        "formulas": {
            "X1": "np.sin(np.pi*k/N)*(0.75 + 0.22*np.cos(12*np.pi*k/N))*np.cos(8*np.pi*k/N)",
            "Y1": "np.cos(np.pi*k/N) + 0.18*np.sin(6*np.pi*k/N)*np.sin(np.pi*k/N)",
            "Z1": "np.sin(np.pi*k/N)*(0.75 + 0.22*np.cos(12*np.pi*k/N))*np.sin(8*np.pi*k/N)",
            "X2": "0.62*np.sin(np.pi*k/N)*np.cos(8*np.pi*k/N + 0.35)",
            "Y2": "0.85*np.cos(np.pi*k/N) - 0.15*np.cos(6*np.pi*k/N)",
            "Z2": "0.62*np.sin(np.pi*k/N)*np.sin(8*np.pi*k/N + 0.35)",
        },
    },
]


# ============================================================================
# 3. BUILT-IN 4D TIME-EVOLVING MATHEMATICAL VIDEO PRESETS
# ============================================================================

MATH_VIDEO_PRESETS: List[Dict[str, Any]] = [
    {
        "id": "video_soaring_bird",
        "name": "4D Flapping Bird in Flight (Time-Harmonic Parinda)",
        "author": "Hamid Naderi Yeganeh 4D Time-Harmonic Extension",
        "category": "3D Manifolds & 4D Math Video",
        "mode": "math_video",
        "N": 2000,
        "description": "Continuous 60 FPS time-parameterized trigonometric bird in flight with flapping wing dihedral and breathing tail feathers.",
        "latex": {
            "X1": r"X_1(k, t) = \frac{3}{2}\sin^7\left(\frac{2\pi k}{N} + \frac{\pi}{3}\right)\left(1 - \frac{3}{25}\cos(t)\right)",
            "Y1": r"Y_1(k, t) = \frac{1}{4}\cos^2\left(\frac{6\pi k}{N}\right) + \frac{9}{25}\left|\sin^5\left(\frac{2\pi k}{N} + \frac{\pi}{3}\right)\right|\sin(t)",
            "X2": r"X_2(k, t) = \frac{1}{5}\sin\left(\frac{6\pi k}{N} + \frac{\pi}{5}\right) + \frac{1}{25}\sin\left(\frac{4\pi k}{N}\right)\cos(t)",
            "Y2": r"Y_2(k, t) = -\frac{2}{3}\sin^2\left(\frac{4\pi k}{N} - \frac{\pi}{3}\right) - \frac{3}{25}\sin^2\left(\frac{4\pi k}{N}\right)\sin(t)",
        },
        "base_formulas": {
            "X1": "1.5 * (np.sin(2*np.pi*k/N + np.pi/3)**7)",
            "Y1": "0.25 * (np.cos(6*np.pi*k/N)**2)",
            "X2": "0.2 * np.sin(6*np.pi*k/N + np.pi/5)",
            "Y2": "-(2.0/3.0) * (np.sin(4*np.pi*k/N - np.pi/3)**2)",
        },
        "mod_formulas": {
            "ax1": "-0.04 * (np.sin(2*np.pi*k/N + np.pi/3)**5)",
            "bx1": "-0.14 * (np.sin(2*np.pi*k/N + np.pi/3)**7)",
            "ay1": "0.36 * (np.abs(np.sin(2*np.pi*k/N + np.pi/3))**5) - 0.08",
            "by1": "0.06 * np.cos(6*np.pi*k/N)",
            "ax2": "0.03 * np.cos(6*np.pi*k/N)",
            "bx2": "0.04 * np.sin(4*np.pi*k/N)",
            "ay2": "-0.12 * (np.sin(4*np.pi*k/N - np.pi/3)**2)",
            "by2": "0.05 * np.cos(4*np.pi*k/N)",
        },
    },
    {
        "id": "video_harmonic_butterfly",
        "name": "4D Breathing Butterfly Wings (Time-Harmonic Titli)",
        "author": "Temple-Fay / Hamid 4D Time-Harmonic Extension",
        "category": "3D Manifolds & 4D Math Video",
        "mode": "math_video",
        "N": 2200,
        "description": "4D exponential-trigonometric butterfly unfolding and flapping its wings in continuous harmonic time t.",
        "latex": {
            "X1": r"X_1(k, t) = \sin\left(\frac{2\pi k}{N}\right)R(k)\left(1 - \frac{11}{50}\sin(t)\right)",
            "Y1": r"Y_1(k, t) = \cos\left(\frac{2\pi k}{N}\right)R(k) + \frac{4}{25}\left|\sin\left(\frac{2\pi k}{N}\right)\right|R(k)\cos(t)",
            "X2": r"X_2(k, t) = \frac{2}{5}\sin\left(\frac{6\pi k}{N}\right)\cos\left(\frac{2\pi k}{N}\right)\left(1 - \frac{3}{20}\sin(t)\right)",
            "Y2": r"Y_2(k, t) = \frac{2}{5}\cos\left(\frac{4\pi k}{N}\right) + \frac{2}{25}\sin\left(\frac{4\pi k}{N}\right)\cos(t)",
        },
        "base_formulas": {
            "X1": "np.sin(2*np.pi*k/N)*(np.exp(np.cos(2*np.pi*k/N)) - 2*np.cos(8*np.pi*k/N) - np.sin(np.pi*k/(6*N))**5)",
            "Y1": "np.cos(2*np.pi*k/N)*(np.exp(np.cos(2*np.pi*k/N)) - 2*np.cos(8*np.pi*k/N) - np.sin(np.pi*k/(6*N))**5)",
            "X2": "0.4*np.sin(6*np.pi*k/N)*np.cos(2*np.pi*k/N)",
            "Y2": "0.4*np.cos(4*np.pi*k/N)",
        },
        "mod_formulas": {
            "ax1": "-0.22 * X1",
            "bx1": "0.06 * np.sin(4*np.pi*k/N)",
            "ay1": "0.14 * np.abs(X1)",
            "by1": "0.10 * Y1",
            "ax2": "-0.15 * X2",
            "bx2": "0.04 * np.cos(6*np.pi*k/N)",
            "ay2": "0.08 * np.sin(4*np.pi*k/N)",
            "by2": "0.06 * Y2",
        },
    },
    {
        "id": "video_swimming_angelfish",
        "name": "4D Undulating Reef Angelfish (Time-Harmonic Machli)",
        "author": "Hamid Naderi Yeganeh 4D Time-Harmonic Extension",
        "category": "3D Manifolds & 4D Math Video",
        "mode": "math_video",
        "N": 2000,
        "description": "4D travelling-wave aquatic propulsion where dorsal fins and tail undulate via phase-locked sin(t) and cos(t) harmonics.",
        "latex": {
            "X1": r"X_1(k, t) = \cos\left(\frac{2\pi k}{N}\right) - \frac{3}{5}\sin^2\left(\frac{2\pi k}{N}\right) + \frac{3}{50}\sin\left(\frac{4\pi k}{N}\right)\sin(t)",
            "Y1": r"Y_1(k, t) = \frac{7}{10}\sin\left(\frac{4\pi k}{N}\right)\cos\left(\frac{2\pi k}{N}\right) + \frac{9}{50}\left(1 - \cos\left(\frac{2\pi k}{N}\right)\right)\sin\left(\frac{6\pi k}{N} - t\right)",
            "X2": r"X_2(k, t) = -\frac{1}{3}\cos\left(\frac{6\pi k}{N}\right) + \frac{2}{5}\cos\left(\frac{2\pi k}{N}\right) + \frac{1}{25}\cos\left(\frac{4\pi k}{N}\right)\cos(t)",
            "Y2": r"Y_2(k, t) = \frac{1}{3}\sin^3\left(\frac{2\pi k}{N}\right) + \frac{3}{25}\sin\left(\frac{6\pi k}{N} - t\right)",
        },
        "base_formulas": {
            "X1": "np.cos(2*np.pi*k/N) - 0.6*(np.sin(2*np.pi*k/N)**2)",
            "Y1": "0.7*np.sin(4*np.pi*k/N)*np.cos(2*np.pi*k/N)",
            "X2": "-(1/3)*np.cos(6*np.pi*k/N) + 0.4*np.cos(2*np.pi*k/N)",
            "Y2": "(1/3)*(np.sin(2*np.pi*k/N)**3)",
        },
        "mod_formulas": {
            "ax1": "0.06 * np.sin(4*np.pi*k/N)",
            "bx1": "0.04 * np.cos(2*np.pi*k/N)",
            "ay1": "-0.16 * (1.0 - np.cos(2*np.pi*k/N)) * np.cos(6*np.pi*k/N)",
            "by1": "0.16 * (1.0 - np.cos(2*np.pi*k/N)) * np.sin(6*np.pi*k/N)",
            "ax2": "0.04 * np.sin(6*np.pi*k/N)",
            "bx2": "0.04 * np.cos(4*np.pi*k/N)",
            "ay2": "-0.11 * np.cos(6*np.pi*k/N)",
            "by2": "0.11 * np.sin(6*np.pi*k/N)",
        },
    },
    {
        "id": "video_blooming_lotus",
        "name": "4D Blooming & Counter-Rotating Lotus Mandala",
        "author": "Differential-Geometric 4D Time-Harmonic Series",
        "category": "3D Manifolds & 4D Math Video",
        "mode": "math_video",
        "N": 2000,
        "description": "4D 8-petal lotus mandala whose outer corolla breathes radially while inner stamen caustics counter-rotate in time t.",
        "latex": {
            "X1": r"X_1(k, t) = R_1(k)\cos\left(\frac{2\pi k}{N}\right) + \frac{7}{50}\cos^2\left(\frac{8\pi k}{N}\right)\cos\left(\frac{2\pi k}{N}\right)\sin(t) - \frac{2}{25}R_1(k)\sin\left(\frac{2\pi k}{N}\right)\cos(t)",
            "Y1": r"Y_1(k, t) = R_1(k)\sin\left(\frac{2\pi k}{N}\right) + \frac{7}{50}\cos^2\left(\frac{8\pi k}{N}\right)\sin\left(\frac{2\pi k}{N}\right)\sin(t) + \frac{2}{25}R_1(k)\cos\left(\frac{2\pi k}{N}\right)\cos(t)",
            "X2": r"X_2(k, t) = R_2(k)\cos\left(\frac{4\pi k}{N} + \frac{\pi}{8}\right) + \frac{3}{25}\sin\left(\frac{4\pi k}{N}\right)\sin(t)",
            "Y2": r"Y_2(k, t) = R_2(k)\sin\left(\frac{4\pi k}{N} + \frac{\pi}{8}\right) - \frac{3}{25}\cos\left(\frac{4\pi k}{N}\right)\sin(t)",
        },
        "base_formulas": {
            "X1": "(0.55 + 0.45*(np.cos(8*np.pi*k/N)**2))*np.cos(2*np.pi*k/N)",
            "Y1": "(0.55 + 0.45*(np.cos(8*np.pi*k/N)**2))*np.sin(2*np.pi*k/N)",
            "X2": "(0.28 + 0.16*(np.sin(8*np.pi*k/N)**2))*np.cos(4*np.pi*k/N + np.pi/8)",
            "Y2": "(0.28 + 0.16*(np.sin(8*np.pi*k/N)**2))*np.sin(4*np.pi*k/N + np.pi/8)",
        },
        "mod_formulas": {
            "ax1": "0.14*(np.cos(8*np.pi*k/N)**2)*np.cos(2*np.pi*k/N)",
            "bx1": "-0.08*Y1",
            "ay1": "0.14*(np.cos(8*np.pi*k/N)**2)*np.sin(2*np.pi*k/N)",
            "by1": "0.08*X1",
            "ax2": "0.11*np.sin(4*np.pi*k/N)",
            "bx2": "0.06*X2",
            "ay2": "-0.11*np.cos(4*np.pi*k/N)",
            "by2": "0.06*Y2",
        },
    },
]


# ============================================================================
# 4. 3D PYTHON SCRIPT EXPORT GENERATOR (Axes3D Line3DCollection)
# ============================================================================

def build_3d_python_export_script(
    title: str,
    N: int,
    py_formulas: Dict[str, str],
) -> str:
    clean_title = sanitize_math_text(title)
    return f'''"""
3D Parametric Manifold Mathematical Art: {clean_title}
Style: 3D Spatial Chord-Surface (X1(k), Y1(k), Z1(k)) -> (X2(k), Y2(k), Z2(k)) for k = 1..{N}
Generated by DesignWithMath
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection

N = {N}
k = np.arange(1, N + 1, dtype=np.float64)

# Exact 3D Trigonometric Equations for Endpoints (X1, Y1, Z1) and (X2, Y2, Z2)
X1 = {py_formulas["X1"]}
Y1 = {py_formulas["Y1"]}
Z1 = {py_formulas["Z1"]}
X2 = {py_formulas["X2"]}
Y2 = {py_formulas["Y2"]}
Z2 = {py_formulas["Z2"]}

fig = plt.figure(figsize=(9, 9), facecolor="#050811")
ax = fig.add_subplot(111, projection="3d", facecolor="#050811")

p1 = np.column_stack([X1, Y1, Z1])
p2 = np.column_stack([X2, Y2, Z2])
segments_3d = np.stack([p1, p2], axis=1)

colors = plt.cm.cool(np.linspace(0.0, 1.0, N))
colors[:, 3] = 0.42
lc = Line3DCollection(segments_3d, colors=colors, linewidths=0.6)
ax.add_collection3d(lc)

ax.set_xlim(-1.0, 1.0)
ax.set_ylim(-1.0, 1.0)
ax.set_zlim(-1.0, 1.0)
ax.set_axis_off()
ax.view_init(elev=24, azim=-48)
plt.tight_layout()
plt.show()
'''


def build_video_python_export_script(
    title: str,
    N: int,
    py_formulas: Dict[str, str],
) -> str:
    clean_title = sanitize_math_text(title)
    return f'''"""
4D Time-Evolving Mathematical Video: {clean_title}
Style: Time-Parameterized Trigonometric System (X1(k,t), Y1(k,t)) -> (X2(k,t), Y2(k,t))
for k = 1..{N} and continuous time phase t in [0, 2*pi]
Generated by DesignWithMath
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.animation import FuncAnimation

N = {N}
k = np.arange(1, N + 1, dtype=np.float64)

# Base Spatial Trigonometric Equations (t = 0)
X1_0 = {py_formulas["X1"]}
Y1_0 = {py_formulas["Y1"]}
X2_0 = {py_formulas["X2"]}
Y2_0 = {py_formulas["Y2"]}

# Fundamental Temporal Harmonic Amplitudes sin(t) and cos(t)
AX1 = {py_formulas.get("AX1", "0.08 * np.sin(2*np.pi*k/N)")}
BX1 = {py_formulas.get("BX1", "0.04 * np.cos(2*np.pi*k/N)")}
AY1 = {py_formulas.get("AY1", "0.14 * np.sin(4*np.pi*k/N)")}
BY1 = {py_formulas.get("BY1", "0.05 * np.cos(4*np.pi*k/N)")}
AX2 = {py_formulas.get("AX2", "0.05 * np.cos(2*np.pi*k/N)")}
BX2 = {py_formulas.get("BX2", "0.03 * np.sin(2*np.pi*k/N)")}
AY2 = {py_formulas.get("AY2", "-0.09 * np.cos(4*np.pi*k/N)")}
BY2 = {py_formulas.get("BY2", "0.04 * np.sin(4*np.pi*k/N)")}

fig, ax = plt.subplots(figsize=(8, 8), facecolor="#050811")
ax.set_facecolor("#050811")
ax.set_xlim(-1.15, 1.15)
ax.set_ylim(-1.15, 1.15)
ax.set_aspect("equal")
ax.axis("off")

segments = np.stack([np.column_stack([X1_0, Y1_0]), np.column_stack([X2_0, Y2_0])], axis=1)
lc = LineCollection(segments, linewidths=0.65, colors="#38bdf8", alpha=0.45)
ax.add_collection(lc)

def update(frame):
    t = 2.0 * np.pi * (frame / 120.0)
    st, ct = np.sin(t), np.cos(t)
    x1 = X1_0 + AX1 * st + BX1 * ct
    y1 = Y1_0 + AY1 * st + BY1 * ct
    x2 = X2_0 + AX2 * st + BX2 * ct
    y2 = Y2_0 + AY2 * st + BY2 * ct
    lc.set_segments(np.stack([np.column_stack([x1, y1]), np.column_stack([x2, y2])], axis=1))
    return (lc,)

anim = FuncAnimation(fig, update, frames=120, interval=16, blit=True)
plt.tight_layout()
plt.show()
'''


# ============================================================================
# 5. 3D PRESET EVALUATOR & UNIVERSAL IR-TO-3D SURFACE COMPILER
# ============================================================================

def evaluate_3d_preset(preset_id: str, custom_N: Optional[int] = None) -> Dict[str, Any]:
    preset = next((p for p in MATH_3D_PRESETS if p["id"] == preset_id), MATH_3D_PRESETS[0])
    N = int(custom_N or preset["N"])
    k = np.arange(1, N + 1, dtype=np.float64)
    env = {"np": np, "k": k, "N": float(N)}
    formulas = preset["formulas"]

    x1 = np.broadcast_to(eval(formulas["X1"], {"__builtins__": {}}, env), (N,)).astype(np.float64)
    y1 = np.broadcast_to(eval(formulas["Y1"], {"__builtins__": {}}, env), (N,)).astype(np.float64)
    z1 = np.broadcast_to(eval(formulas["Z1"], {"__builtins__": {}}, env), (N,)).astype(np.float64)
    x2 = np.broadcast_to(eval(formulas["X2"], {"__builtins__": {}}, env), (N,)).astype(np.float64)
    y2 = np.broadcast_to(eval(formulas["Y2"], {"__builtins__": {}}, env), (N,)).astype(np.float64)
    z2 = np.broadcast_to(eval(formulas["Z2"], {"__builtins__": {}}, env), (N,)).astype(np.float64)

    max_abs = max(
        float(np.max(np.abs(x1))),
        float(np.max(np.abs(y1))),
        float(np.max(np.abs(z1))),
        float(np.max(np.abs(x2))),
        float(np.max(np.abs(y2))),
        float(np.max(np.abs(z2))),
        1e-5,
    )
    x1, y1, z1 = x1 / max_abs, y1 / max_abs, z1 / max_abs
    x2, y2, z2 = x2 / max_abs, y2 / max_abs, z2 / max_abs

    primitives_3d = np.column_stack([x1, y1, z1, x2, y2, z2]).round(4).tolist()
    primitives_2d = np.column_stack([x1, y1, x2, y2]).round(4).tolist()

    return {
        "title": sanitize_math_text(preset["name"]),
        "subtitle": sanitize_math_text(preset["description"]),
        "author": sanitize_math_text(preset["author"]),
        "mode": "3d_surface",
        "N": N,
        "harmonics_count": "3D Closed-Form Exact",
        "latex": preset["latex"],
        "formulas": formulas,
        "primitives_3d": primitives_3d,
        "primitives": primitives_2d,
        "python_code": build_3d_python_export_script(preset["name"], N, formulas),
    }


def _safe_eval_sdf_relief_1d(sdf_expr: str, x: np.ndarray, y: np.ndarray) -> Optional[np.ndarray]:
    """
    Safely evaluates an analytical implicit_sdf_expr(x, y, E) along 1D curve coordinates
    using ai_architect.safe_eval_sdf_expr to sculpt 3D relief depth Z(k).
    """
    if not sdf_expr:
        return None
    try:
        E = np.zeros_like(x, dtype=np.float64)
        for s in range(1, 5):
            freq = 3.0 * (1.28 ** s)
            a2 = 2.0 * s * s
            u1 = math.cos(a2) * x + math.sin(a2) * y
            v1 = math.cos(a2) * y - math.sin(a2) * x
            E += (0.85 ** s) * np.cos(freq * u1 + 2.0 * math.cos(5.0 * s)) * np.cos(freq * v1 + 2.0 * math.cos(5.0 * s))
        E = E / 2.5
        arr = np.asarray(safe_eval_sdf_expr(str(sdf_expr), x, y, E=E), dtype=np.float64)
        arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=-1.0)
        return np.tanh(arr)
    except Exception:
        return None


def _extrude_component_to_3d(
    comp: Dict[str, Any],
    idx: int,
    n_pts: int,
    sym_mode: Optional[str],
    sym_param: Any,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compiles ANY 2D anatomical or generative component (including ribbon_spine,
    polar_harmonic, crescent_lobe, wheel_rosette, parametric_equation,
    repeated_field, and implicit_sdf_expr) into a 3D parametric tube/ribbon manifold
    (x1, y1, z1, x2, y2, z2) with smooth Clenshaw-Curtis endpoints so C^2
    zero-length 3D bridges have zero derivative kink.
    """
    x1, y1, x2, y2 = compile_ir_component(comp, n_pts)
    if sym_mode == "mirror_x":
        ax = float(sym_param)
        x1, x2 = 2.0 * ax - x1, 2.0 * ax - x2
    elif sym_mode == "radial":
        ang_deg, (rcx, rcy) = sym_param
        rad = np.deg2rad(ang_deg)
        ca, sa = np.cos(rad), np.sin(rad)
        x1, y1 = rcx + ca * (x1 - rcx) - sa * (y1 - rcy), rcy + sa * (x1 - rcx) + ca * (y1 - rcy)
        x2, y2 = rcx + ca * (x2 - rcx) - sa * (y2 - rcy), rcy + sa * (x2 - rcx) + ca * (y2 - rcy)

    u_lin = np.linspace(0.0, 1.0, n_pts, endpoint=False)
    u = 0.5 * (1.0 - np.cos(np.pi * u_lin))
    name_low = (comp.get("name") or "").lower()
    ctype = comp.get("type", "ribbon_spine")

    # Component depth offset & volumetric bulge
    default_z_off = 0.08 * math.sin(1.7 * (idx + 1))
    z_offset = float(comp.get("z_offset", default_z_off))
    default_bulge = 0.22 if ctype in ("polar_harmonic", "crescent_lobe", "repeated_field", "implicit_sdf_expr") else 0.16
    z_bulge = max(0.08, float(comp.get("z_bulge", default_bulge)))

    # Smooth window that vanishes at u=0 and u=1 for C^2 bridge continuity, plus z_offset
    env = np.sin(np.pi * u)
    chord_w = np.hypot(x1 - x2, y1 - y2)

    if ctype == "repeated_field":
        layout = (comp.get("layout") or "scattered_cluster").strip().lower()
        count = max(4, min(60, int(comp.get("count", 16))))
        if layout == "perspective_corridor":
            # True 3D receding tunnel depth: near instances forward (+Z), far horizon backward (-Z)
            depth_grad = 0.55 * (1.0 - 2.0 * u) - 0.22 * y1
            z1 = z_offset + (depth_grad + z_bulge * np.cos(min(count, 12) * np.pi * u)) * env
            z2 = z_offset + (depth_grad - 0.85 * z_bulge * np.sin(min(count, 12) * np.pi * u)) * env
        elif layout == "radial_fan":
            # 3D conical/fluted umbrella curvature across concentric rings
            r_loc = np.hypot(x1, y1)
            corrug = np.cos(min(count, 16) * np.pi * u)
            z1 = z_offset + (z_bulge * (0.6 + 0.5 * r_loc) * corrug + 0.16 * r_loc * np.sin(3.0 * np.pi * u)) * env
            z2 = z_offset - (0.75 * z_bulge * corrug - 0.14 * r_loc * np.cos(3.0 * np.pi * u)) * env
        elif layout == "wave_array":
            # 3D staggered curtains of waves / tendrils / flames across depth
            z1 = z_offset + (0.30 * np.sin(6.0 * np.pi * u + 2.5 * x1) + z_bulge * np.cos(min(count, 14) * np.pi * u)) * env
            z2 = z_offset + (0.28 * np.cos(6.0 * np.pi * u + 2.5 * x2) - z_bulge * np.sin(min(count, 14) * np.pi * u)) * env
        else:  # scattered_cluster
            # Volumetric 3D spherical scatter across Z-depth
            z1 = z_offset + (0.36 * np.sin(min(count, 18) * 1.3 * np.pi * u + 0.9 * idx) + 0.18 * np.cos(5.0 * np.pi * u)) * env
            z2 = z_offset + (0.32 * np.cos(min(count, 18) * 1.3 * np.pi * u + 0.9 * idx) - 0.18 * np.sin(5.0 * np.pi * u)) * env
    elif ctype == "implicit_sdf_expr":
        sdf_expr = comp.get("sdf_expr", "")
        relief_1 = _safe_eval_sdf_relief_1d(sdf_expr, x1, y1)
        relief_2 = _safe_eval_sdf_relief_1d(sdf_expr, x2, y2)
        if relief_1 is not None and relief_2 is not None:
            z1 = z_offset + (z_bulge * (0.55 * np.sin(2.0 * np.pi * u) - 0.65 * relief_1) + 0.45 * chord_w * np.cos(4.0 * np.pi * u)) * env
            z2 = z_offset + (z_bulge * (-0.55 * np.sin(2.0 * np.pi * u) - 0.65 * relief_2) - 0.45 * chord_w * np.sin(4.0 * np.pi * u)) * env
        else:
            z1 = z_offset + (z_bulge + 0.60 * chord_w) * np.cos(3.0 * np.pi * u + 0.5 * idx) * env
            z2 = z_offset - (z_bulge + 0.60 * chord_w) * np.sin(3.0 * np.pi * u + 0.5 * idx) * env
    elif any(k in name_low for k in ["wing", "feather", "plume", "hood", "fin", "petal", "leaf", "bell", "umbrella"]):
        # Sweeping 3D dihedral wing/petal/jellyfish-bell camber
        z1 = z_offset + z_bulge * np.sin(np.pi * u) + 0.28 * np.abs(x1) * np.cos(2.0 * np.pi * u) * env
        z2 = z_offset - 0.75 * z_bulge * np.sin(np.pi * u) + 0.18 * np.sin(6.0 * np.pi * u) * env
    elif ctype == "wheel_rosette" or any(k in name_low for k in ["ring", "saturn", "orbit", "halo"]):
        # 3D Toroidal Ring / Double-Cone Spoked Wheel Hub & Rim
        z1 = z_offset + z_bulge * np.cos(4.0 * np.pi * u) * env + 0.22 * y1 * env
        z2 = z_offset - z_bulge * np.sin(4.0 * np.pi * u) * env - 0.22 * y2 * env
    else:
        # 3D Helical Tube / Ribbed Dome Extrusion around component spine
        z1 = z_offset + (z_bulge + 0.65 * chord_w) * np.sin(2.0 * np.pi * u + 0.4 * idx) * env + 0.12 * np.cos(6.0 * np.pi * u) * env
        z2 = z_offset - (z_bulge + 0.65 * chord_w) * np.sin(2.0 * np.pi * u + 0.4 * idx) * env - 0.12 * np.cos(6.0 * np.pi * u) * env

    return x1, y1, z1, x2, y2, z2


def compile_ir_to_3d_surface(
    ir: Optional[Dict[str, Any]] = None,
    N: int = 2000,
    num_harmonics: int = 64,
    x1_raw: Optional[np.ndarray] = None,
    y1_raw: Optional[np.ndarray] = None,
    dual_raw: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    title: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Universal 3D Parametric Manifold Compiler:
    Takes any anatomical IR from ai_architect.py (or 2D contour pair (x1, y1, x2, y2))
    and extrudes/revolves each component into a true 3D parametric tube/ribbon manifold
    (X1(k), Y1(k), Z1(k)) -> (X2(k), Y2(k), Z2(k)) for k = 1..N where all 6 coordinates
    are decomposed into explicit Fourier trigonometric series.
    """
    raw_comps = (ir or {}).get("components", []) if ir is not None else []

    if raw_comps:
        expanded: List[Tuple[float, int, Dict[str, Any], Optional[str], Any]] = []
        for idx, c in enumerate(raw_comps):
            w = max(0.03, float(c.get("weight", 0.15)))
            rc = max(1, min(16, int(c.get("radial_copies", 1))))
            mx = bool(c.get("mirror_x", False))
            if rc > 1:
                arc = c.get("radial_arc_deg", [0.0, 360.0])
                if not isinstance(arc, list) or len(arc) < 2:
                    arc = [0.0, 360.0]
                is_open_arc = abs(float(arc[1]) - float(arc[0]) - 360.0) > 1e-2
                angles = np.linspace(float(arc[0]), float(arc[1]), rc, endpoint=is_open_arc)
                for ang in angles:
                    expanded.append((w / rc, idx, c, "radial", (float(ang), c.get("radial_center", [0.0, 0.0]))))
            elif mx:
                axis_x = float(c.get("mirror_axis_x", 0.0))
                expanded.append((w / 2.0, idx, c, None, None))
                expanded.append((w / 2.0, idx, c, "mirror_x", axis_x))
            else:
                expanded.append((w, idx, c, None, None))

        M = len(expanded)
        n_bridge = 20 if M > 1 else 0
        total_bridge = M * n_bridge
        n_active = max(M * 24, N - total_bridge)
        total_w = sum(item[0] for item in expanded)

        allocs = [max(24, int(round((item[0] / total_w) * n_active))) for item in expanded]
        diff = (N - total_bridge) - sum(allocs)
        allocs[-1] = max(24, allocs[-1] + diff)

        comp_curves_3d = []
        for (w, idx, c, sym_mode, sym_param), n_pts in zip(expanded, allocs):
            comp_curves_3d.append(_extrude_component_to_3d(c, idx, n_pts, sym_mode, sym_param))

        # Nearest-Neighbor 3D TSP ordering
        ordered = [comp_curves_3d[0]]
        remaining = list(comp_curves_3d[1:])
        while remaining:
            last_x = 0.5 * (ordered[-1][0][-1] + ordered[-1][3][-1])
            last_y = 0.5 * (ordered[-1][1][-1] + ordered[-1][4][-1])
            best_idx, best_dist, reverse_best = 0, 1e9, False
            for idx_r, cur in enumerate(remaining):
                sx, sy = 0.5 * (cur[0][0] + cur[3][0]), 0.5 * (cur[1][0] + cur[4][0])
                ex, ey = 0.5 * (cur[0][-1] + cur[3][-1]), 0.5 * (cur[1][-1] + cur[4][-1])
                d_start = math.hypot(sx - last_x, sy - last_y)
                d_end = math.hypot(ex - last_x, ey - last_y)
                if d_start < best_dist:
                    best_dist, best_idx, reverse_best = d_start, idx_r, False
                if d_end < best_dist:
                    best_dist, best_idx, reverse_best = d_end, idx_r, True
            chosen = remaining.pop(best_idx)
            if reverse_best:
                chosen = tuple(arr[::-1] for arr in chosen)
            ordered.append(chosen)

        X1_l, Y1_l, Z1_l, X2_l, Y2_l, Z2_l = [], [], [], [], [], []
        s = np.linspace(0.0, 1.0, n_bridge + 2)[1:-1]
        sigma = 10.0 * (s ** 3) - 15.0 * (s ** 4) + 6.0 * (s ** 5)

        for i, cur in enumerate(ordered):
            X1_l.append(cur[0])
            Y1_l.append(cur[1])
            Z1_l.append(cur[2])
            X2_l.append(cur[3])
            Y2_l.append(cur[4])
            Z2_l.append(cur[5])
            if n_bridge > 0:
                nxt = ordered[(i + 1) % len(ordered)]
                m1_x, m1_y, m1_z = 0.5 * (cur[0][-1] + cur[3][-1]), 0.5 * (cur[1][-1] + cur[4][-1]), 0.5 * (cur[2][-1] + cur[5][-1])
                m2_x, m2_y, m2_z = 0.5 * (nxt[0][0] + nxt[3][0]), 0.5 * (nxt[1][0] + nxt[4][0]), 0.5 * (nxt[2][0] + nxt[5][0])
                bx = (1.0 - sigma) * m1_x + sigma * m2_x
                by = (1.0 - sigma) * m1_y + sigma * m2_y
                bz = (1.0 - sigma) * m1_z + sigma * m2_z
                X1_l.append(bx)
                Y1_l.append(by)
                Z1_l.append(bz)
                X2_l.append(bx)
                Y2_l.append(by)
                Z2_l.append(bz)

        x1_c = np.concatenate(X1_l)[:N]
        y1_c = np.concatenate(Y1_l)[:N]
        z1_c = np.concatenate(Z1_l)[:N]
        x2_c = np.concatenate(X2_l)[:N]
        y2_c = np.concatenate(Y2_l)[:N]
        z2_c = np.concatenate(Z2_l)[:N]
        effective_harmonics = max(num_harmonics, 84)
    else:
        t = np.linspace(0.0, 2.0 * np.pi, N, endpoint=False)
        if x1_raw is None or y1_raw is None:
            x1_c = 1.5 * (np.sin(t + np.pi / 3.0) ** 7)
            y1_c = 0.25 * (np.cos(3.0 * t) ** 2)
            x2_c = 0.2 * np.sin(3.0 * t + np.pi / 5.0)
            y2_c = -(2.0 / 3.0) * (np.sin(2.0 * t - np.pi / 3.0) ** 2)
        else:
            x1_c = np.asarray(x1_raw, dtype=np.float64)[:N]
            y1_c = np.asarray(y1_raw, dtype=np.float64)[:N]
            if dual_raw is not None:
                x2_c = np.asarray(dual_raw[0], dtype=np.float64)[:N]
                y2_c = np.asarray(dual_raw[1], dtype=np.float64)[:N]
            else:
                delta = max(2, N // 45)
                x2_c = 0.85 * np.roll(x1_c, -delta)
                y2_c = 0.85 * np.roll(y1_c, -delta)

        # Normalize 2D coordinates before 3D surface revolution/extrusion
        m2d = max(float(np.max(np.abs(x1_c))), float(np.max(np.abs(y1_c))), float(np.max(np.abs(x2_c))), float(np.max(np.abs(y2_c))), 1e-5)
        x1_c, y1_c, x2_c, y2_c = x1_c / m2d, y1_c / m2d, x2_c / m2d, y2_c / m2d
        ribbon_w = np.hypot(x1_c - x2_c, y1_c - y2_c)
        z1_c = 0.36 * np.sin(t) * np.cos(2.0 * t) + 0.48 * ribbon_w * np.cos(3.0 * t) + 0.16 * np.sin(5.0 * t)
        z2_c = -0.32 * np.cos(t) * np.sin(2.0 * t) - 0.45 * ribbon_w * np.sin(3.0 * t) + 0.14 * np.cos(4.0 * t)
        effective_harmonics = max(num_harmonics, 64)

    max_abs = max(
        float(np.max(np.abs(x1_c))),
        float(np.max(np.abs(y1_c))),
        float(np.max(np.abs(z1_c))),
        float(np.max(np.abs(x2_c))),
        float(np.max(np.abs(y2_c))),
        float(np.max(np.abs(z2_c))),
        1e-5,
    )
    signals = {
        "X1": x1_c / max_abs,
        "Y1": y1_c / max_abs,
        "Z1": z1_c / max_abs,
        "X2": x2_c / max_abs,
        "Y2": y2_c / max_abs,
        "Z2": z2_c / max_abs,
    }

    k = np.arange(1, N + 1, dtype=np.float64)
    latex_dict: Dict[str, str] = {}
    py_formulas: Dict[str, str] = {}
    harmonics_meta: Dict[str, Any] = {}
    evaluated: Dict[str, np.ndarray] = {}

    for var_name, sig in signals.items():
        a0, harms = fit_fourier_series_1d(sig, num_harmonics=effective_harmonics)
        latex_dict[var_name] = format_3d_series_to_latex(var_name, a0, harms, N)
        py_formulas[var_name] = format_series_to_py_expr(a0, harms)
        harmonics_meta[var_name] = {"a0": a0, "terms": harms}
        evaluated[var_name] = eval_fourier_series_1d(a0, harms, k, N)

    primitives_3d = np.column_stack([
        evaluated["X1"],
        evaluated["Y1"],
        evaluated["Z1"],
        evaluated["X2"],
        evaluated["Y2"],
        evaluated["Z2"],
    ]).round(4).tolist()

    primitives_2d = np.column_stack([
        evaluated["X1"],
        evaluated["Y1"],
        evaluated["X2"],
        evaluated["Y2"],
    ]).round(4).tolist()

    raw_title = title or (ir or {}).get("title") or "3D Parametric Manifold Synthesis"
    clean_title = sanitize_math_text(raw_title, "3D Parametric Manifold Synthesis")
    num_parts = len(raw_comps) if raw_comps else 1
    subtitle = (
        f"3D Differential-Geometric Ribbon Manifold (X₁(k), Y₁(k), Z₁(k)) → (X₂(k), Y₂(k), Z₂(k)) — "
        f"{num_parts} Coupled Spatial Sub-Manifolds ({effective_harmonics} Harmonics)"
    )

    res: Dict[str, Any] = {
        "title": clean_title,
        "subtitle": subtitle,
        "author": "3D Differential-Geometric Fourier-Hamid Compiler",
        "mode": "3d_surface",
        "N": N,
        "harmonics_count": effective_harmonics,
        "latex": latex_dict,
        "formulas": py_formulas,
        "harmonics": harmonics_meta,
        "primitives_3d": primitives_3d,
        "primitives": primitives_2d,
        "python_code": build_3d_python_export_script(clean_title, N, py_formulas),
    }
    if raw_comps:
        res["ir_components"] = [
            {"name": sanitize_math_text(c.get("name", "Sub-Manifold")), "type": c.get("type", "ribbon_spine"), "mirror_x": bool(c.get("mirror_x", False))}
            for c in raw_comps
        ]
    return res


# ============================================================================
# 6. 4D TIME-EVOLVING MATHEMATICAL VIDEO / ANIMATION ENGINE (mode == "math_video")
# ============================================================================

def format_4d_video_latex(
    var_name: str,
    a0_base: float,
    harms_base: List[Dict[str, float]],
    a0_sin: float,
    harms_sin: List[Dict[str, float]],
    a0_cos: float,
    harms_cos: List[Dict[str, float]],
    N: int,
) -> str:
    """
    Formats explicit time-dependent LaTeX equations X_1(k, t), Y_1(k, t), X_2(k, t), Y_2(k, t)
    showing both spatial harmonics (2*pi*m*k / N) and temporal harmonics sin(t), cos(t).
    """
    def _short_spatial_expr(a0_val: float, h_list: List[Dict[str, float]], max_t: int = 3) -> str:
        t_parts: List[str] = []
        if abs(a0_val) >= 0.005:
            t_parts.append(f"{a0_val:.3f}")
        top_h = sorted(sorted(h_list, key=lambda h: h["amp"], reverse=True)[:max_t], key=lambda h: h["m"])
        for h in top_h:
            m = h["m"]
            a = h["a"]
            b = h["b"]
            ang = rf"\frac{{{2*m}\pi k}}{{{N}}}"
            if abs(a) >= 0.004:
                s = "+" if (a > 0 and t_parts) else ("-" if a < 0 else "")
                t_parts.append(rf"{s}{abs(a):.3f}\cos\!\left({ang}\right)")
            if abs(b) >= 0.004:
                s = "+" if (b > 0 and t_parts) else ("-" if b < 0 else "")
                t_parts.append(rf"{s}{abs(b):.3f}\sin\!\left({ang}\right)")
        return " ".join(t_parts).lstrip("+ ") if t_parts else "0.015"

    base_str = _short_spatial_expr(a0_base, harms_base, max_t=4)
    sin_str = _short_spatial_expr(a0_sin, harms_sin, max_t=2)
    cos_str = _short_spatial_expr(a0_cos, harms_cos, max_t=2)

    pretty_var = {"X1": "X_1", "Y1": "Y_1", "X2": "X_2", "Y2": "Y_2"}.get(var_name, var_name)
    return (
        rf"{pretty_var}(k, t) = \Big({base_str}\Big) "
        rf"+ \Big({sin_str}\Big)\sin(\omega t) "
        rf"+ \Big({cos_str}\Big)\cos(\omega t)"
    )


def evaluate_video_preset(preset_id: str, custom_N: Optional[int] = None) -> Dict[str, Any]:
    preset = next((p for p in MATH_VIDEO_PRESETS if p["id"] == preset_id), MATH_VIDEO_PRESETS[0])
    N = int(custom_N or preset["N"])
    k = np.arange(1, N + 1, dtype=np.float64)
    base_f = preset["base_formulas"]
    mod_f = preset["mod_formulas"]

    env: Dict[str, Any] = {"np": np, "k": k, "N": float(N)}
    x1 = np.broadcast_to(eval(base_f["X1"], {"__builtins__": {}}, env), (N,)).astype(np.float64)
    y1 = np.broadcast_to(eval(base_f["Y1"], {"__builtins__": {}}, env), (N,)).astype(np.float64)
    x2 = np.broadcast_to(eval(base_f["X2"], {"__builtins__": {}}, env), (N,)).astype(np.float64)
    y2 = np.broadcast_to(eval(base_f["Y2"], {"__builtins__": {}}, env), (N,)).astype(np.float64)

    env.update({"X1": x1, "Y1": y1, "X2": x2, "Y2": y2})
    ax1 = np.broadcast_to(eval(mod_f["ax1"], {"__builtins__": {}}, env), (N,)).astype(np.float64)
    bx1 = np.broadcast_to(eval(mod_f["bx1"], {"__builtins__": {}}, env), (N,)).astype(np.float64)
    ay1 = np.broadcast_to(eval(mod_f["ay1"], {"__builtins__": {}}, env), (N,)).astype(np.float64)
    by1 = np.broadcast_to(eval(mod_f["by1"], {"__builtins__": {}}, env), (N,)).astype(np.float64)
    ax2 = np.broadcast_to(eval(mod_f["ax2"], {"__builtins__": {}}, env), (N,)).astype(np.float64)
    bx2 = np.broadcast_to(eval(mod_f["bx2"], {"__builtins__": {}}, env), (N,)).astype(np.float64)
    ay2 = np.broadcast_to(eval(mod_f["ay2"], {"__builtins__": {}}, env), (N,)).astype(np.float64)
    by2 = np.broadcast_to(eval(mod_f["by2"], {"__builtins__": {}}, env), (N,)).astype(np.float64)

    max_abs = max(
        float(np.max(np.abs(x1))),
        float(np.max(np.abs(y1))),
        float(np.max(np.abs(x2))),
        float(np.max(np.abs(y2))),
        1e-5,
    )
    x1, y1, x2, y2 = x1 / max_abs, y1 / max_abs, x2 / max_abs, y2 / max_abs
    ax1, bx1, ay1, by1 = ax1 / max_abs, bx1 / max_abs, ay1 / max_abs, by1 / max_abs
    ax2, bx2, ay2, by2 = ax2 / max_abs, bx2 / max_abs, ay2 / max_abs, by2 / max_abs

    primitives = np.column_stack([x1, y1, x2, y2]).round(4).tolist()
    time_harmonics = np.column_stack([ax1, bx1, ay1, by1, ax2, bx2, ay2, by2]).round(4).tolist()

    py_formulas = {
        "X1": base_f["X1"],
        "Y1": base_f["Y1"],
        "X2": base_f["X2"],
        "Y2": base_f["Y2"],
    }

    return {
        "title": sanitize_math_text(preset["name"]),
        "subtitle": sanitize_math_text(preset["description"]),
        "author": sanitize_math_text(preset["author"]),
        "mode": "math_video",
        "N": N,
        "harmonics_count": "4D Space-Time Closed-Form",
        "latex": preset["latex"],
        "formulas": py_formulas,
        "primitives": primitives,
        "time_harmonics": time_harmonics,
        "python_code": build_video_python_export_script(preset["name"], N, py_formulas),
    }


def _infer_component_anim_role(comp: Dict[str, Any]) -> str:
    role = (comp.get("anim_role") or "").strip().lower()
    if role in {"flap", "spin", "sway", "wave", "pulse", "orbit", "static"}:
        return role
    name_low = (comp.get("name") or "").lower()
    ctype = comp.get("type", "ribbon_spine")
    if ctype == "repeated_field":
        layout = (comp.get("layout") or "").strip().lower()
        if layout == "wave_array":
            return "wave"
        if layout == "radial_fan":
            return "pulse"
        if layout == "scattered_cluster":
            return "orbit"
        return "sway"
    if ctype == "implicit_sdf_expr":
        return "wave" if any(k in name_low for k in ["wave", "water", "tentacle", "cloud", "nebula", "flame"]) else "pulse"
    if any(k in name_low for k in ["wing", "plume", "feather", "fin", "arm", "bird", "eagle", "phoenix", "butterfly"]):
        return "flap"
    if any(k in name_low for k in ["ring", "saturn", "planet", "orbit", "astronaut", "satellite", "comet"]):
        return "orbit"
    if ctype == "wheel_rosette" or any(k in name_low for k in ["wheel", "gear", "spoke", "rosette", "propeller", "turbine"]):
        return "spin"
    if any(k in name_low for k in ["water", "pool", "wave", "ocean", "coil", "tail", "snake", "dragon", "stream", "tentacle", "jellyfish", "kelp"]):
        return "wave"
    if any(k in name_low for k in ["moon", "hilal", "crescent", "star", "crown", "sun", "eye", "heart", "ocellus", "bell", "glow"]):
        return "pulse"
    return "sway"


def _compute_component_time_harmonics(
    comp: Dict[str, Any],
    idx: int,
    n_pts: int,
    sym_mode: Optional[str],
    sym_param: Any,
) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray,
    np.ndarray, np.ndarray, np.ndarray, np.ndarray,
    np.ndarray, np.ndarray, np.ndarray, np.ndarray,
]:
    """
    Computes base spatial curves (x1, y1, x2, y2) and analytical temporal harmonic
    coefficients (ax1, bx1, ay1, by1, ax2, bx2, ay2, by2) for ANY component
    (including repeated_field and implicit_sdf_expr) so that:
        X1(k, t) = X1_0(k) + ax1(k)*sin(t) + bx1(k)*cos(t)
    respects Clenshaw-Curtis endpoint damping (vanishing at component bridge joints)
    and exhibits lifelike harmonic kinematics!
    """
    x1, y1, x2, y2 = compile_ir_component(comp, n_pts)
    if sym_mode == "mirror_x":
        ax = float(sym_param)
        x1, x2 = 2.0 * ax - x1, 2.0 * ax - x2
    elif sym_mode == "radial":
        ang_deg, (rcx, rcy) = sym_param
        rad = np.deg2rad(ang_deg)
        ca, sa = np.cos(rad), np.sin(rad)
        x1, y1 = rcx + ca * (x1 - rcx) - sa * (y1 - rcy), rcy + sa * (x1 - rcx) + ca * (y1 - rcy)
        x2, y2 = rcx + ca * (x2 - rcx) - sa * (y2 - rcy), rcy + sa * (x2 - rcx) + ca * (y2 - rcy)

    u_lin = np.linspace(0.0, 1.0, n_pts, endpoint=False)
    u = 0.5 * (1.0 - np.cos(np.pi * u_lin))
    env = np.sin(np.pi * u)  # Smooth C^2 window that vanishes at bridge joints
    role = _infer_component_anim_role(comp)
    ctype = comp.get("type", "ribbon_spine")

    cx = float(np.mean(0.5 * (x1 + x2)))
    cy = float(np.mean(0.5 * (y1 + y2)))
    dx1, dy1 = x1 - cx, y1 - cy
    dx2, dy2 = x2 - cx, y2 - cy

    if ctype == "repeated_field":
        layout = (comp.get("layout") or "scattered_cluster").strip().lower()
        count = max(4, min(60, int(comp.get("count", 16))))
        if layout == "wave_array":
            w_phase = 5.0 * np.pi * u + 2.2 * x1 + 0.6 * idx
            ax1 = 0.09 * np.sin(w_phase) * env
            bx1 = 0.07 * np.cos(w_phase) * env
            ay1 = -0.12 * np.cos(w_phase) * env
            by1 = 0.12 * np.sin(w_phase) * env
            ax2 = 0.09 * np.sin(w_phase + 0.35) * env
            bx2 = 0.07 * np.cos(w_phase + 0.35) * env
            ay2 = -0.12 * np.cos(w_phase + 0.35) * env
            by2 = 0.12 * np.sin(w_phase + 0.35) * env
            return x1, y1, x2, y2, ax1, bx1, ay1, by1, ax2, bx2, ay2, by2
        elif layout == "perspective_corridor":
            depth_scale = (1.15 - 0.75 * u) * env
            ax1 = 0.08 * depth_scale * np.sin(min(count, 10) * np.pi * u)
            bx1 = 0.04 * depth_scale * np.cos(2.0 * np.pi * u)
            ay1 = 0.04 * depth_scale * np.cos(min(count, 10) * np.pi * u)
            by1 = 0.03 * depth_scale * np.sin(2.0 * np.pi * u)
            ax2 = 0.08 * depth_scale * np.sin(min(count, 10) * np.pi * u + 0.25)
            bx2 = -0.04 * depth_scale * np.cos(2.0 * np.pi * u)
            ay2 = 0.04 * depth_scale * np.cos(min(count, 10) * np.pi * u + 0.25)
            by2 = -0.03 * depth_scale * np.sin(2.0 * np.pi * u)
            return x1, y1, x2, y2, ax1, bx1, ay1, by1, ax2, bx2, ay2, by2

    if role == "flap":
        # Wing flapping around root hinge (x near 0): vertical lift proportional to |x|, slight spanwise foreshortening
        span1 = np.abs(x1) + 0.15
        span2 = np.abs(x2) + 0.15
        ax1 = -0.07 * x1 * env
        bx1 = 0.03 * np.sin(2.0 * np.pi * u) * env
        ay1 = 0.24 * span1 * env
        by1 = 0.06 * np.cos(2.0 * np.pi * u) * env
        ax2 = -0.07 * x2 * env
        bx2 = -0.03 * np.sin(2.0 * np.pi * u) * env
        ay2 = 0.22 * span2 * env
        by2 = -0.05 * np.cos(2.0 * np.pi * u) * env

    elif role == "spin":
        # Tangential harmonic rotation velocity vector (-dy, dx) around wheel/rosette center
        ax1 = -0.22 * dy1 * env
        bx1 = -0.10 * dx1 * env
        ay1 = 0.22 * dx1 * env
        by1 = -0.10 * dy1 * env
        ax2 = -0.22 * dy2 * env
        bx2 = -0.10 * dx2 * env
        ay2 = 0.22 * dx2 * env
        by2 = -0.10 * dy2 * env

    elif role == "orbit":
        # Smooth elliptical 3D-projected orbital drift (planets, rings, floating astronauts, star clusters)
        ax1 = (-0.18 * dy1 + 0.05 * np.cos(2.0 * np.pi * u + idx)) * env
        bx1 = (0.07 * dx1 + 0.04 * np.sin(2.0 * np.pi * u)) * env
        ay1 = (0.12 * dx1 + 0.06 * np.sin(2.0 * np.pi * u + idx)) * env
        by1 = (0.06 * dy1 - 0.04 * np.cos(2.0 * np.pi * u)) * env
        ax2 = (-0.18 * dy2 + 0.05 * np.cos(2.0 * np.pi * u + idx)) * env
        bx2 = (0.07 * dx2 + 0.04 * np.sin(2.0 * np.pi * u)) * env
        ay2 = (0.12 * dx2 + 0.06 * np.sin(2.0 * np.pi * u + idx)) * env
        by2 = (0.06 * dy2 - 0.04 * np.cos(2.0 * np.pi * u)) * env

    elif role == "wave":
        # Travelling serpentine / water / tentacle wave sin(4*pi*u - t)
        w_phase = 4.0 * np.pi * u + 0.7 * idx
        ax1 = 0.06 * np.sin(w_phase) * env
        bx1 = 0.05 * np.cos(w_phase) * env
        ay1 = -0.12 * np.cos(w_phase) * env
        by1 = 0.12 * np.sin(w_phase) * env
        ax2 = 0.06 * np.sin(w_phase + 0.3) * env
        bx2 = 0.05 * np.cos(w_phase + 0.3) * env
        ay2 = -0.12 * np.cos(w_phase + 0.3) * env
        by2 = 0.12 * np.sin(w_phase + 0.3) * env

    elif role == "pulse":
        # Radial breathing & gentle celestial levitation
        ax1 = 0.14 * dx1 * env
        bx1 = -0.06 * dy1 * env
        ay1 = 0.14 * dy1 * env + 0.05 * env
        by1 = 0.06 * dx1 * env
        ax2 = 0.14 * dx2 * env
        bx2 = -0.06 * dy2 * env
        ay2 = 0.14 * dy2 * env + 0.05 * env
        by2 = 0.06 * dx2 * env

    elif role == "static":
        # Rock-solid architectural anchor with subtle harmonic light shimmer
        ax1 = 0.014 * np.sin(2.0 * np.pi * u) * env
        bx1 = 0.010 * np.cos(2.0 * np.pi * u) * env
        ay1 = 0.014 * np.cos(2.0 * np.pi * u) * env
        by1 = 0.010 * np.sin(2.0 * np.pi * u) * env
        ax2 = -0.014 * np.sin(2.0 * np.pi * u) * env
        bx2 = 0.010 * np.cos(2.0 * np.pi * u) * env
        ay2 = -0.014 * np.cos(2.0 * np.pi * u) * env
        by2 = 0.010 * np.sin(2.0 * np.pi * u) * env

    else:  # sway
        # Graceful cantilever harmonic sway along u
        sway_profile = (u ** 1.3) * env
        ax1 = 0.09 * sway_profile * np.cos(0.5 * idx)
        bx1 = 0.04 * np.sin(2.0 * np.pi * u) * env
        ay1 = 0.05 * sway_profile * np.sin(2.0 * np.pi * u)
        by1 = 0.03 * np.cos(2.0 * np.pi * u) * env
        ax2 = 0.09 * sway_profile * np.cos(0.5 * idx)
        bx2 = -0.04 * np.sin(2.0 * np.pi * u) * env
        ay2 = 0.05 * sway_profile * np.sin(2.0 * np.pi * u)
        by2 = -0.03 * np.cos(2.0 * np.pi * u) * env

    return x1, y1, x2, y2, ax1, bx1, ay1, by1, ax2, bx2, ay2, by2


def compile_ir_to_math_video(
    ir: Optional[Dict[str, Any]] = None,
    N: int = 2000,
    num_harmonics: int = 64,
    x1_raw: Optional[np.ndarray] = None,
    y1_raw: Optional[np.ndarray] = None,
    dual_raw: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    title: Optional[str] = None,
    archetype: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Universal 4D Time-Evolving Mathematical Video Compiler:
    Synthesizes a time-parameterized trigonometric equation system
        (X_1(k, t), Y_1(k, t), X_2(k, t), Y_2(k, t))
    for k = 1..N and continuous time t in [0, 2*pi].
    Decomposes both the base spatial curve and the sin(t)/cos(t) temporal harmonic
    amplitudes into explicit Fourier trigonometric series for 60 FPS browser rendering.
    """
    raw_comps = (ir or {}).get("components", []) if ir is not None else []

    if raw_comps:
        expanded: List[Tuple[float, int, Dict[str, Any], Optional[str], Any]] = []
        for idx, c in enumerate(raw_comps):
            w = max(0.03, float(c.get("weight", 0.15)))
            rc = max(1, min(16, int(c.get("radial_copies", 1))))
            mx = bool(c.get("mirror_x", False))
            if rc > 1:
                arc = c.get("radial_arc_deg", [0.0, 360.0])
                if not isinstance(arc, list) or len(arc) < 2:
                    arc = [0.0, 360.0]
                is_open_arc = abs(float(arc[1]) - float(arc[0]) - 360.0) > 1e-2
                angles = np.linspace(float(arc[0]), float(arc[1]), rc, endpoint=is_open_arc)
                for ang in angles:
                    expanded.append((w / rc, idx, c, "radial", (float(ang), c.get("radial_center", [0.0, 0.0]))))
            elif mx:
                axis_x = float(c.get("mirror_axis_x", 0.0))
                expanded.append((w / 2.0, idx, c, None, None))
                expanded.append((w / 2.0, idx, c, "mirror_x", axis_x))
            else:
                expanded.append((w, idx, c, None, None))

        M = len(expanded)
        n_bridge = 20 if M > 1 else 0
        total_bridge = M * n_bridge
        n_active = max(M * 24, N - total_bridge)
        total_w = sum(item[0] for item in expanded)

        allocs = [max(24, int(round((item[0] / total_w) * n_active))) for item in expanded]
        diff = (N - total_bridge) - sum(allocs)
        allocs[-1] = max(24, allocs[-1] + diff)

        comp_packs = []
        for (w, idx, c, sym_mode, sym_param), n_pts in zip(expanded, allocs):
            comp_packs.append(_compute_component_time_harmonics(c, idx, n_pts, sym_mode, sym_param))

        # Nearest-Neighbor TSP ordering
        ordered = [comp_packs[0]]
        remaining = list(comp_packs[1:])
        while remaining:
            last_x = 0.5 * (ordered[-1][0][-1] + ordered[-1][2][-1])
            last_y = 0.5 * (ordered[-1][1][-1] + ordered[-1][3][-1])
            best_idx, best_dist, reverse_best = 0, 1e9, False
            for idx_r, cur in enumerate(remaining):
                sx, sy = 0.5 * (cur[0][0] + cur[2][0]), 0.5 * (cur[1][0] + cur[3][0])
                ex, ey = 0.5 * (cur[0][-1] + cur[2][-1]), 0.5 * (cur[1][-1] + cur[3][-1])
                d_start = math.hypot(sx - last_x, sy - last_y)
                d_end = math.hypot(ex - last_x, ey - last_y)
                if d_start < best_dist:
                    best_dist, best_idx, reverse_best = d_start, idx_r, False
                if d_end < best_dist:
                    best_dist, best_idx, reverse_best = d_end, idx_r, True
            chosen = remaining.pop(best_idx)
            if reverse_best:
                chosen = tuple(arr[::-1] for arr in chosen)
            ordered.append(chosen)

        lists = [[] for _ in range(12)]
        s = np.linspace(0.0, 1.0, n_bridge + 2)[1:-1]
        sigma = 10.0 * (s ** 3) - 15.0 * (s ** 4) + 6.0 * (s ** 5)
        zeros_br = np.zeros(n_bridge, dtype=np.float64)

        for i, cur in enumerate(ordered):
            for j in range(12):
                lists[j].append(cur[j])
            if n_bridge > 0:
                nxt = ordered[(i + 1) % len(ordered)]
                m1_x, m1_y = 0.5 * (cur[0][-1] + cur[2][-1]), 0.5 * (cur[1][-1] + cur[3][-1])
                m2_x, m2_y = 0.5 * (nxt[0][0] + nxt[2][0]), 0.5 * (nxt[1][0] + nxt[3][0])
                bx = (1.0 - sigma) * m1_x + sigma * m2_x
                by = (1.0 - sigma) * m1_y + sigma * m2_y
                lists[0].append(bx)
                lists[1].append(by)
                lists[2].append(bx)
                lists[3].append(by)
                for j in range(4, 12):
                    lists[j].append(zeros_br)

        arrays = [np.concatenate(lst)[:N] for lst in lists]
        x1_c, y1_c, x2_c, y2_c, ax1_c, bx1_c, ay1_c, by1_c, ax2_c, bx2_c, ay2_c, by2_c = arrays
        effective_harmonics = max(num_harmonics, 84)
    else:
        t = np.linspace(0.0, 2.0 * np.pi, N, endpoint=False)
        if x1_raw is None or y1_raw is None:
            x1_c = 1.5 * (np.sin(t + np.pi / 3.0) ** 7)
            y1_c = 0.25 * (np.cos(3.0 * t) ** 2)
            x2_c = 0.2 * np.sin(3.0 * t + np.pi / 5.0)
            y2_c = -(2.0 / 3.0) * (np.sin(2.0 * t - np.pi / 3.0) ** 2)
        else:
            x1_c = np.asarray(x1_raw, dtype=np.float64)[:N]
            y1_c = np.asarray(y1_raw, dtype=np.float64)[:N]
            if dual_raw is not None:
                x2_c = np.asarray(dual_raw[0], dtype=np.float64)[:N]
                y2_c = np.asarray(dual_raw[1], dtype=np.float64)[:N]
            else:
                delta = max(2, N // 45)
                x2_c = 0.85 * np.roll(x1_c, -delta)
                y2_c = 0.85 * np.roll(y1_c, -delta)

        arch_low = (archetype or title or "").lower()
        if any(k in arch_low for k in ["bird", "eagle", "butterfly", "swan", "owl", "dragon", "peacock"]):
            # Wing-flapping and feather-breathing harmonic modulation
            ax1_c = -0.10 * x1_c * (np.sin(t) ** 2)
            bx1_c = 0.04 * np.sin(2.0 * t)
            ay1_c = 0.24 * np.abs(x1_c) - 0.06
            by1_c = 0.06 * np.cos(3.0 * t)
            ax2_c = -0.06 * x2_c
            bx2_c = 0.03 * np.cos(2.0 * t)
            ay2_c = -0.10 * np.abs(y2_c)
            by2_c = 0.05 * np.sin(3.0 * t)
        elif any(k in arch_low for k in ["fish", "dolphin", "seahorse", "snake", "octopus", "wave", "boat"]):
            # Travelling aquatic/serpentine wave modulation
            ax1_c = 0.06 * np.sin(3.0 * t)
            bx1_c = 0.04 * np.cos(2.0 * t)
            ay1_c = -0.14 * np.cos(4.0 * t) * (0.5 + 0.5 * np.abs(x1_c))
            by1_c = 0.14 * np.sin(4.0 * t) * (0.5 + 0.5 * np.abs(x1_c))
            ax2_c = 0.04 * np.sin(3.0 * t + 0.4)
            bx2_c = 0.04 * np.cos(2.0 * t + 0.4)
            ay2_c = -0.11 * np.cos(4.0 * t + 0.4)
            by2_c = 0.11 * np.sin(4.0 * t + 0.4)
        else:
            # Radial blooming, rotation, and harmonic oscillation
            ax1_c = 0.11 * x1_c * np.cos(2.0 * t)
            bx1_c = -0.08 * y1_c
            ay1_c = 0.11 * y1_c * np.cos(2.0 * t)
            by1_c = 0.08 * x1_c
            ax2_c = -0.09 * y2_c
            bx2_c = 0.07 * x2_c
            ay2_c = 0.09 * x2_c
            by2_c = 0.07 * y2_c
        effective_harmonics = max(num_harmonics, 64)

    max_abs = max(
        float(np.max(np.abs(x1_c))),
        float(np.max(np.abs(y1_c))),
        float(np.max(np.abs(x2_c))),
        float(np.max(np.abs(y2_c))),
        1e-5,
    )
    x1_c, y1_c, x2_c, y2_c = x1_c / max_abs, y1_c / max_abs, x2_c / max_abs, y2_c / max_abs
    ax1_c, bx1_c, ay1_c, by1_c = ax1_c / max_abs, bx1_c / max_abs, ay1_c / max_abs, by1_c / max_abs
    ax2_c, bx2_c, ay2_c, by2_c = ax2_c / max_abs, bx2_c / max_abs, ay2_c / max_abs, by2_c / max_abs

    k = np.arange(1, N + 1, dtype=np.float64)
    base_signals = {"X1": x1_c, "Y1": y1_c, "X2": x2_c, "Y2": y2_c}
    mod_signals = {
        "X1": (ax1_c, bx1_c),
        "Y1": (ay1_c, by1_c),
        "X2": (ax2_c, bx2_c),
        "Y2": (ay2_c, by2_c),
    }

    latex_dict: Dict[str, str] = {}
    py_formulas: Dict[str, str] = {}
    harmonics_meta: Dict[str, Any] = {}
    eval_base: Dict[str, np.ndarray] = {}
    eval_sin: Dict[str, np.ndarray] = {}
    eval_cos: Dict[str, np.ndarray] = {}

    mod_harmonics_count = max(16, min(36, effective_harmonics // 2))

    for var_name, sig in base_signals.items():
        a0_b, h_b = fit_fourier_series_1d(sig, num_harmonics=effective_harmonics)
        sig_sin, sig_cos = mod_signals[var_name]
        a0_s, h_s = fit_fourier_series_1d(sig_sin, num_harmonics=mod_harmonics_count)
        a0_c, h_c = fit_fourier_series_1d(sig_cos, num_harmonics=mod_harmonics_count)

        latex_dict[var_name] = format_4d_video_latex(var_name, a0_b, h_b, a0_s, h_s, a0_c, h_c, N)
        py_formulas[var_name] = format_series_to_py_expr(a0_b, h_b)
        py_formulas[f"A{var_name}"] = format_series_to_py_expr(a0_s, h_s, max_terms=16)
        py_formulas[f"B{var_name}"] = format_series_to_py_expr(a0_c, h_c, max_terms=16)
        harmonics_meta[var_name] = {"a0": a0_b, "terms": h_b}

        eval_base[var_name] = eval_fourier_series_1d(a0_b, h_b, k, N)
        eval_sin[var_name] = eval_fourier_series_1d(a0_s, h_s, k, N)
        eval_cos[var_name] = eval_fourier_series_1d(a0_c, h_c, k, N)

    primitives = np.column_stack([
        eval_base["X1"],
        eval_base["Y1"],
        eval_base["X2"],
        eval_base["Y2"],
    ]).round(4).tolist()

    time_harmonics = np.column_stack([
        eval_sin["X1"],
        eval_cos["X1"],
        eval_sin["Y1"],
        eval_cos["Y1"],
        eval_sin["X2"],
        eval_cos["X2"],
        eval_sin["Y2"],
        eval_cos["Y2"],
    ]).round(4).tolist()

    raw_title = title or (ir or {}).get("title") or "4D Time-Harmonic Mathematical Video"
    clean_title = sanitize_math_text(raw_title, "4D Time-Harmonic Mathematical Video")
    subtitle = (
        f"4D Space-Time Trigonometric System X(k,t) = X₀(k) + A(k)sin(ωt) + B(k)cos(ωt) — "
        f"{effective_harmonics} Spatial × {mod_harmonics_count} Temporal Harmonics (60 FPS)"
    )

    res: Dict[str, Any] = {
        "title": clean_title,
        "subtitle": subtitle,
        "author": "4D Space-Time Fourier-Hamid Differential Compiler",
        "mode": "math_video",
        "N": N,
        "harmonics_count": f"{effective_harmonics} Spatial × {mod_harmonics_count} Temporal",
        "latex": latex_dict,
        "formulas": {
            "X1": py_formulas["X1"],
            "Y1": py_formulas["Y1"],
            "X2": py_formulas["X2"],
            "Y2": py_formulas["Y2"],
        },
        "harmonics": harmonics_meta,
        "primitives": primitives,
        "time_harmonics": time_harmonics,
        "python_code": build_video_python_export_script(clean_title, N, py_formulas),
    }
    if raw_comps:
        res["ir_components"] = [
            {"name": sanitize_math_text(c.get("name", "Sub-Manifold")), "type": c.get("type", "ribbon_spine"), "mirror_x": bool(c.get("mirror_x", False))}
            for c in raw_comps
        ]
    return res


# Backwards-compatible alias
compile_to_math_video = compile_ir_to_math_video


def compile_animated_pixel_field_frames(
    ir: Dict[str, Any],
    num_frames: int = 8,
    width: int = 480,
    height: int = 288,
) -> Dict[str, Any]:
    """
    Renders a loopable multi-frame animated 2D pixel-field sequence across
    t_phase = 2*pi*m / num_frames using pixel_field_engine.compile_ir_to_pixel_field.
    """
    from pixel_field_engine import compile_ir_to_pixel_field

    num_frames = max(2, min(24, int(num_frames)))
    frames_b64: List[str] = []
    first_res: Optional[Dict[str, Any]] = None
    for m in range(num_frames):
        t_m = 2.0 * math.pi * (float(m) / float(num_frames))
        frame_res = compile_ir_to_pixel_field(
            ir,
            width=width,
            height=height,
            t_phase=t_m,
            include_raw_array=False,
        )
        frame_res.pop("rgb_array", None)
        frames_b64.append(frame_res["image_base64"])
        if first_res is None:
            first_res = frame_res

    assert first_res is not None
    first_res["frames_base64"] = frames_b64
    first_res["num_frames"] = num_frames
    first_res["t_phase"] = 0.0
    first_res["title"] = sanitize_math_text(first_res.get("title", "Animated 2D Pixel-Field Painting"))
    first_res["subtitle"] = sanitize_math_text(first_res.get("subtitle", ""))
    first_res["author"] = sanitize_math_text(
        first_res.get("author", ""), "Hamid Naderi Yeganeh Double-Exponential Level-Set Engine"
    )
    return first_res


# ============================================================================
# 7. PHASE 2 SOTA: 4D COLOR VIDEO SHOWCASES, ISO-CONTOUR 2D/3D/4D & GIF/MP4
# ============================================================================

PIXEL_VIDEO_SHOWCASE_PRESETS: List[Dict[str, Any]] = [
    {
        "id": "pvid_bengal_tiger",
        "name": "4D Color Video: Bengal Tiger",
        "author": "Symbolic Differential-Geometric Math Compiler",
        "category": "3D Manifolds & 4D Math Video",
        "mode": "pixel_video",
        "N": 400,
        "promptFallback": "a roaring Bengal tiger face in lush jungle",
        "description": "4D full-color analytical SDF Bengal tiger portrait with animated fur strands, sunlit jungle bokeh, and breathing jaw.",
        "latex": {
            "RGB": r"\operatorname{rgb}\!\left(F(H_0(x,y,t)),\,F(H_1(x,y,t)),\,F(H_2(x,y,t))\right)",
        },
    },
    {
        "id": "pvid_jellyfish",
        "name": "4D Color Video: Bioluminescent Jellyfish",
        "author": "Symbolic Differential-Geometric Math Compiler",
        "category": "3D Manifolds & 4D Math Video",
        "mode": "pixel_video",
        "N": 400,
        "promptFallback": "a glowing bioluminescent jellyfish swimming in deep ocean",
        "description": "4D pulsing bioluminescent jellyfish bell and undulating deep-ocean tentacles compiled to 60 FPS GPU equations.",
        "latex": {
            "RGB": r"\operatorname{rgb}\!\left(F(H_0(x,y,t)),\,F(H_1(x,y,t)),\,F(H_2(x,y,t))\right)",
        },
    },
    {
        "id": "pvid_lighthouse_aurora",
        "name": "4D Color Video: Lighthouse & Aurora",
        "author": "Symbolic Differential-Geometric Math Compiler",
        "category": "3D Manifolds & 4D Math Video",
        "mode": "pixel_video",
        "N": 400,
        "promptFallback": "a crystal lighthouse on a rocky cliff under dancing northern lights aurora",
        "description": "4D coastal crystal lighthouse beam sweeping beneath time-evolving northern lights aurora curtains and waves.",
        "latex": {
            "RGB": r"\operatorname{rgb}\!\left(F(H_0(x,y,t)),\,F(H_1(x,y,t)),\,F(H_2(x,y,t))\right)",
        },
    },
    {
        "id": "pvid_saturn_astronaut",
        "name": "4D Color Video: Saturn & Astronaut",
        "author": "Symbolic Differential-Geometric Math Compiler",
        "category": "3D Manifolds & 4D Math Video",
        "mode": "pixel_video",
        "N": 400,
        "promptFallback": "an astronaut floating beside Saturn golden rings in cosmic space",
        "description": "4D deep-space orbital scene featuring an astronaut drifting past Saturn's golden analytical ring system.",
        "latex": {
            "RGB": r"\operatorname{rgb}\!\left(F(H_0(x,y,t)),\,F(H_1(x,y,t)),\,F(H_2(x,y,t))\right)",
        },
    },
]


def compile_math_program_to_2d_3d_4d(
    prog: Dict[str, Any],
    mode: str,
    N: int = 1800,
    num_harmonics: int = 64,
    t_phase: float = 0.0,
    refinement_passes: int = 1,
    visual_score: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Compiles a validated symbolic `math_program` into crisp, unmistakably recognizable
    2D chord envelopes (`lines`), 3D relief manifolds (`3d_surface`), or 4D time-evolving
    wireframe animations (`math_video`) via analytical multi-threshold SDF iso-contours
    and Farneback optical-flow temporal harmonics.
    """
    from symbolic_math_compiler import (
        compile_math_program_to_glsl,
        extract_iso_contours_from_math_program,
    )

    clean_prog = sanitize_math_program(prog)
    clean_title = clean_prog["title"]
    N = max(100, min(5000, int(N)))
    eff_harmonics = max(24, min(120, int(num_harmonics)))

    iso = extract_iso_contours_from_math_program(clean_prog, N=N, t_phase=float(t_phase))
    x1 = np.asarray(iso["x1"], dtype=np.float64)
    y1 = np.asarray(iso["y1"], dtype=np.float64)
    z1 = np.asarray(iso["z1"], dtype=np.float64)
    x2 = np.asarray(iso["x2"], dtype=np.float64)
    y2 = np.asarray(iso["y2"], dtype=np.float64)
    z2 = np.asarray(iso["z2"], dtype=np.float64)
    primitives = iso["primitives"]
    primitives_3d = iso["primitives_3d"]
    time_harmonics = iso["time_harmonics"]
    contour_count = int(iso.get("contour_count", 0))
    motion_energy = float(iso.get("motion_energy", 0.0))

    # Guarantee rich 3D depth relief if flat
    if mode == "3d_surface":
        z_std = max(float(np.std(z1)), float(np.std(z2)))
        if z_std < 0.03:
            ang = np.linspace(0.0, 2.0 * np.pi, N, endpoint=False)
            z1 = np.clip(z1 * 2.5 + 0.16 * np.sin(2.0 * ang + x1 * 2.0), -0.95, 0.95)
            z2 = np.clip(z2 * 2.5 + 0.16 * np.cos(2.0 * ang - y2 * 2.0), -0.95, 0.95)
            primitives_3d = np.column_stack([x1, y1, z1, x2, y2, z2]).round(5).tolist()

    glsl_src = compile_math_program_to_glsl(clean_prog)

    if mode == "3d_surface":
        signals = {"X1": x1, "Y1": y1, "Z1": z1, "X2": x2, "Y2": y2, "Z2": z2}
        latex_dict: Dict[str, str] = {}
        py_formulas: Dict[str, str] = {}
        harmonics_meta: Dict[str, Any] = {}
        for var_name, sig in signals.items():
            a0, harms = fit_fourier_series_1d(sig, num_harmonics=eff_harmonics)
            latex_dict[var_name] = format_3d_series_to_latex(var_name, a0, harms, N)
            py_formulas[var_name] = format_series_to_py_expr(a0, harms)
            harmonics_meta[var_name] = {"a0": a0, "terms": harms}

        return {
            "title": clean_title,
            "subtitle": f"3D Iso-Contour Relief Manifold ({contour_count} Analytical Contours, N={N})",
            "author": "3D Differential-Geometric Iso-Contour Compiler",
            "mode": "3d_surface",
            "N": N,
            "harmonics_count": eff_harmonics,
            "latex": latex_dict,
            "latex_equations": latex_dict,
            "formulas": py_formulas,
            "harmonics": harmonics_meta,
            "primitives_3d": primitives_3d,
            "primitives": primitives,
            "glsl_fragment_shader": glsl_src,
            "math_program": clean_prog,
            "refinement_passes": int(refinement_passes),
            "visual_score": visual_score,
            "contour_count": contour_count,
            "python_code": build_3d_python_export_script(clean_title, N, py_formulas),
        }

    if mode == "math_video":
        th_arr = np.asarray(time_harmonics, dtype=np.float64)
        if float(np.max(np.abs(th_arr))) <= 0.008:
            ang = np.linspace(0.0, 2.0 * np.pi, N, endpoint=False)
            th_arr[:, 0] = 0.035 * np.sin(2.0 * y1 + ang * 0.25)
            th_arr[:, 2] = 0.035 * np.cos(2.0 * x1 - ang * 0.25)
            th_arr[:, 4] = 0.035 * np.sin(2.0 * y2 + ang * 0.25)
            th_arr[:, 6] = 0.035 * np.cos(2.0 * x2 - ang * 0.25)
            time_harmonics = np.round(th_arr, 5).tolist()

        base_signals = {"X1": x1, "Y1": y1, "X2": x2, "Y2": y2}
        mod_signals = {
            "X1": (th_arr[:, 0], th_arr[:, 1]),
            "Y1": (th_arr[:, 2], th_arr[:, 3]),
            "X2": (th_arr[:, 4], th_arr[:, 5]),
            "Y2": (th_arr[:, 6], th_arr[:, 7]),
        }
        latex_dict = {}
        py_formulas = {}
        harmonics_meta = {}
        mod_h = max(12, min(32, eff_harmonics // 2))
        for var_name, sig in base_signals.items():
            a0_b, h_b = fit_fourier_series_1d(sig, num_harmonics=eff_harmonics)
            a0_s, h_s = fit_fourier_series_1d(mod_signals[var_name][0], num_harmonics=mod_h)
            a0_c, h_c = fit_fourier_series_1d(mod_signals[var_name][1], num_harmonics=mod_h)
            latex_dict[var_name] = format_4d_video_latex(var_name, a0_b, h_b, a0_s, h_s, a0_c, h_c, N)
            py_formulas[var_name] = format_series_to_py_expr(a0_b, h_b)
            py_formulas[f"A{var_name}"] = format_series_to_py_expr(a0_s, h_s, max_terms=16)
            py_formulas[f"B{var_name}"] = format_series_to_py_expr(a0_c, h_c, max_terms=16)
            harmonics_meta[var_name] = {"a0": a0_b, "terms": h_b}

        return {
            "title": clean_title,
            "subtitle": f"4D Time-Harmonic Iso-Contour System ({contour_count} Analytical Contours, 60 FPS)",
            "author": "4D Space-Time Iso-Contour Differential Compiler",
            "mode": "math_video",
            "N": N,
            "harmonics_count": f"{eff_harmonics} Spatial × {mod_h} Temporal",
            "latex": latex_dict,
            "latex_equations": latex_dict,
            "formulas": {
                "X1": py_formulas["X1"],
                "Y1": py_formulas["Y1"],
                "X2": py_formulas["X2"],
                "Y2": py_formulas["Y2"],
            },
            "harmonics": harmonics_meta,
            "primitives": primitives,
            "time_harmonics": time_harmonics,
            "glsl_fragment_shader": glsl_src,
            "math_program": clean_prog,
            "refinement_passes": int(refinement_passes),
            "visual_score": visual_score,
            "motion_energy": motion_energy,
            "contour_count": contour_count,
            "python_code": build_video_python_export_script(clean_title, N, py_formulas),
        }

    # Default: mode == "lines" (crisp 2D chord envelopes)
    signals_2d = {"X1": x1, "Y1": y1, "X2": x2, "Y2": y2}
    latex_dict = {}
    py_formulas = {}
    harmonics_meta = {}
    for var_name, sig in signals_2d.items():
        a0, harms = fit_fourier_series_1d(sig, num_harmonics=eff_harmonics)
        latex_dict[var_name] = format_3d_series_to_latex(var_name, a0, harms, N)
        py_formulas[var_name] = format_series_to_py_expr(a0, harms)
        harmonics_meta[var_name] = {"a0": a0, "terms": harms}

    py_script = f'''"""
Mathematical Art: {clean_title}
Style: Analytical SDF Iso-Contour Chord Envelope (k = 1..{N})
Generated by DesignWithMath
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

N = {N}
k = np.arange(1, N + 1, dtype=np.float64)

X1 = {py_formulas["X1"]}
Y1 = {py_formulas["Y1"]}
X2 = {py_formulas["X2"]}
Y2 = {py_formulas["Y2"]}

fig, ax = plt.subplots(figsize=(8, 8), facecolor="#050811")
ax.set_facecolor("#050811")
segments = np.stack([np.column_stack([X1, Y1]), np.column_stack([X2, Y2])], axis=1)
lc = LineCollection(segments, linewidths=0.55, colors="#38bdf8", alpha=0.50)
ax.add_collection(lc)
ax.autoscale()
ax.set_aspect("equal")
ax.axis("off")
plt.tight_layout()
plt.show()
'''

    return {
        "title": clean_title,
        "subtitle": f"Analytical SDF Iso-Contour Chord Envelope ({contour_count} Contours, {eff_harmonics}-Harmonic Series, N={N})",
        "author": "Fourier-Hamid Iso-Contour Differential Compiler",
        "mode": "lines",
        "N": N,
        "harmonics_count": eff_harmonics,
        "latex": latex_dict,
        "latex_equations": latex_dict,
        "formulas": py_formulas,
        "harmonics": harmonics_meta,
        "primitives": primitives,
        "glsl_fragment_shader": glsl_src,
        "math_program": clean_prog,
        "refinement_passes": int(refinement_passes),
        "visual_score": visual_score,
        "contour_count": contour_count,
        "python_code": py_script,
    }


def render_math_video_response(
    prog: Dict[str, Any],
    width: int = 480,
    height: int = 300,
    num_frames: int = 32,
    fps: int = 16,
    include_mp4: bool = True,
    refinement_passes: int = 1,
    visual_score: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Renders a real looping animated GIF and optional MP4 directly from a validated
    symbolic `math_program` using `symbolic_math_compiler.render_math_video_file`.
    """
    from symbolic_math_compiler import (
        compile_math_program_to_glsl,
        compile_math_program_to_latex,
        render_math_video_file,
    )

    clean_prog = sanitize_math_program(prog)
    W = max(120, min(960, int(width)))
    H = max(80, min(640, int(height)))
    n_fr = max(4, min(64, int(num_frames)))
    fps_i = max(4, min(30, int(fps)))

    tmp_mp4_path: Optional[str] = None
    if include_mp4:
        fd, tmp_mp4_path = tempfile.mkstemp(suffix=".mp4", prefix="dwm_vid_")
        os.close(fd)

    try:
        vid_res = render_math_video_file(
            clean_prog,
            out_gif_path=None,
            out_mp4_path=tmp_mp4_path,
            width=W,
            height=H,
            num_frames=n_fr,
            fps=fps_i,
        )
    finally:
        if tmp_mp4_path and os.path.exists(tmp_mp4_path):
            try:
                os.remove(tmp_mp4_path)
            except OSError:
                pass

    latex_eqs = compile_math_program_to_latex(clean_prog)
    glsl_src = compile_math_program_to_glsl(clean_prog)

    return {
        "title": clean_prog["title"],
        "subtitle": clean_prog["subtitle"],
        "author": "Symbolic Differential-Geometric Math Compiler",
        "mode": "pixel_video",
        "width": vid_res["width"],
        "height": vid_res["height"],
        "num_frames": vid_res["num_frames"],
        "fps": vid_res["fps"],
        "motion_energy": vid_res["motion_energy"],
        "gif_base64": vid_res["gif_base64"],
        "mp4_base64": vid_res.get("mp4_base64"),
        "first_frame_stats": vid_res.get("first_frame_stats"),
        "glsl_fragment_shader": glsl_src,
        "latex": latex_eqs,
        "latex_equations": latex_eqs,
        "animation_capable": True,
        "refinement_passes": int(refinement_passes),
        "visual_score": visual_score,
        "math_program": clean_prog,
    }


# ============================================================================
# 8. STAGE 1 & STAGE 2: 3D RAYMARCH CINEMA SHOWCASES & VIDEO RENDERER
# ============================================================================

RAYMARCH_CINEMA_SHOWCASE_PRESETS: List[Dict[str, Any]] = [
    {
        "id": "rcin_golden_phoenix",
        "name": "3D Raymarch Cinema: Golden Phoenix & Ocean",
        "author": "3D Implicit SDF & Differential-Geometry Cinema Engine",
        "category": "3D Manifolds & 4D Math Video",
        "mode": "raymarch_cinema",
        "N": 1600,
        "promptFallback": "a golden phoenix rising over turbulent sapphire ocean waves",
        "description": "True 3D sphere-traced golden phoenix with Floraform ruffled wings ascending over analytical sapphire ocean waves.",
        "latex": {
            "Scene_SDF_3D": r"d(\mathbf{p}, t) = \operatorname{smin}_k\!\left(d_{\text{phoenix}}(\mathbf{p}, t),\, d_{\text{ocean}}(\mathbf{p}, t)\right) = 0",
        },
    },
    {
        "id": "rcin_gyroid_cathedral",
        "name": "3D Raymarch Cinema: Hyperbolic Gyroid Cathedral",
        "author": "3D Implicit SDF & Differential-Geometry Cinema Engine",
        "category": "3D Manifolds & 4D Math Video",
        "mode": "raymarch_cinema",
        "N": 1600,
        "promptFallback": "a hyperbolic bronze cathedral with glowing gyroid pillars and god rays",
        "description": "True 3D raymarched hyperbolic bronze cathedral with Schoen gyroid minimal-surface pillars and volumetric god-rays.",
        "latex": {
            "Scene_SDF_3D": r"d(\mathbf{p}, t) = \operatorname{smin}_k\!\left(d_{\text{cathedral}}(\mathbf{p}, t),\, |\sin x\cos y + \sin y\cos z + \sin z\cos x| - \delta\right) = 0",
        },
    },
]


def sanitize_raymarch_scene(scene: Dict[str, Any]) -> Dict[str, Any]:
    """Returns a clean, branding-free copy of a canonical `raymarch_3d_scene` dict."""
    clean_nodes: List[Dict[str, Any]] = []
    for idx, node in enumerate(scene.get("nodes", [])):
        if not isinstance(node, dict):
            continue
        nd = dict(node)
        if "id" in nd:
            nd["id"] = sanitize_math_text(str(nd["id"]), f"node_{idx + 1}")
        if "name" in nd:
            nd["name"] = sanitize_math_text(str(nd["name"]), f"Manifold {idx + 1}")
        clean_nodes.append(nd)

    return {
        "title": sanitize_math_text(scene.get("title", ""), "3D Raymarched Implicit Manifold"),
        "subtitle": sanitize_math_text(
            scene.get("subtitle", ""),
            "3D Sphere-Traced Signed Distance Field & Differential-Geometry Cinema",
        ),
        "environment": dict(scene.get("environment", {})),
        "nodes": clean_nodes,
    }


def render_raymarch_video_response(
    scene_or_preset: Any,
    width: int = 420,
    height: int = 262,
    num_frames: int = 24,
    fps: int = 12,
    include_mp4: bool = True,
    refinement_passes: int = 2,
    visual_score: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Renders a true 3D raymarched looping animated GIF & optional MP4 directly from
    either a Stage-1 Master Paradigm preset ID (`mp_*`) or a Stage-2 `raymarch_3d_scene` dict.
    """
    from master_paradigms_engine import (
        MASTER_PARADIGM_PRESETS,
        compile_raymarch_scene_to_glsl,
        compile_raymarch_scene_to_latex,
        evaluate_master_paradigm_preset,
        render_raymarch_cinema_video,
    )

    W = max(120, min(800, int(width)))
    H = max(80, min(500, int(height)))
    n_fr = max(4, min(48, int(num_frames)))
    fps_i = max(4, min(30, int(fps)))

    is_preset_id = isinstance(scene_or_preset, str)
    target = scene_or_preset if is_preset_id else sanitize_raymarch_scene(scene_or_preset)

    tmp_mp4_path: Optional[str] = None
    if include_mp4:
        fd, tmp_mp4_path = tempfile.mkstemp(suffix=".mp4", prefix="dwm_rmv_")
        os.close(fd)

    try:
        vid_res = render_raymarch_cinema_video(
            target,
            out_gif_path=None,
            out_mp4_path=tmp_mp4_path,
            width=W,
            height=H,
            num_frames=n_fr,
            fps=fps_i,
        )
    finally:
        if tmp_mp4_path and os.path.exists(tmp_mp4_path):
            try:
                os.remove(tmp_mp4_path)
            except OSError:
                pass

    if is_preset_id:
        p_meta = next((p for p in MASTER_PARADIGM_PRESETS if p["id"] == scene_or_preset), None)
        p_eval = evaluate_master_paradigm_preset(scene_or_preset, width=160, height=100, t_phase=0.0, N=200)
        title = sanitize_math_text(p_eval.get("title") or (p_meta or {}).get("name", "3D Master Paradigm Cinema"))
        subtitle = sanitize_math_text(p_eval.get("subtitle") or (p_meta or {}).get("description", ""))
        author = sanitize_math_text(p_eval.get("author", "Master Mathematical Paradigms Engine"))
        glsl_src = p_eval.get("glsl_fragment_shader", "")
        latex_eqs = p_eval.get("latex_equations") or p_eval.get("latex") or {}
        clean_scene = None
    else:
        title = target["title"]
        subtitle = target["subtitle"]
        author = "3D Implicit SDF & Differential-Geometry Cinema Engine"
        glsl_src = compile_raymarch_scene_to_glsl(target)
        latex_eqs = compile_raymarch_scene_to_latex(target)
        clean_scene = target

    res: Dict[str, Any] = {
        "title": title,
        "subtitle": subtitle,
        "author": author,
        "mode": "raymarch_cinema",
        "width": vid_res["width"],
        "height": vid_res["height"],
        "num_frames": vid_res["num_frames"],
        "fps": vid_res["fps"],
        "motion_energy": vid_res["motion_energy"],
        "gif_base64": vid_res["gif_base64"],
        "mp4_base64": vid_res.get("mp4_base64"),
        "first_frame_stats": vid_res.get("first_frame_stats"),
        "glsl_fragment_shader": glsl_src,
        "latex": latex_eqs,
        "latex_equations": latex_eqs,
        "animation_capable": True,
        "refinement_passes": int(refinement_passes),
        "visual_score": visual_score,
    }
    if clean_scene is not None:
        res["raymarch_scene"] = clean_scene
    if is_preset_id:
        res["preset_id"] = scene_or_preset
    return res
