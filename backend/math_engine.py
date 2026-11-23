import math
import os
import re
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

import ai_architect
from ai_architect import (
    call_cloudflare_ai_for_ir,
    compile_hamid_ir_to_chords,
    decompose_long_script_to_storyboard,
    generate_3d_raymarch_scene_with_critique,
    generate_math_program,
    generate_math_program_with_passes,
    get_curated_complex_ir,
)
from cinema_director_engine import (
    SAMPLE_EPIC_5_ACT_SCRIPT,
    compile_and_render_long_script_film,
)
from master_paradigms_engine import (
    MASTER_PARADIGM_PRESETS,
    evaluate_master_paradigm_preset,
    evaluate_raymarch_scene_numpy,
    validate_raymarch_scene,
)
from pixel_field_engine import (
    PIXEL_FIELD_PRESETS,
    compile_ir_to_pixel_field,
    evaluate_pixel_field_preset,
)
from symbolic_math_compiler import (
    evaluate_math_program_numpy,
    validate_math_program,
)
from math_3d_video_engine import (
    MATH_3D_PRESETS,
    MATH_VIDEO_PRESETS,
    PIXEL_VIDEO_SHOWCASE_PRESETS,
    RAYMARCH_CINEMA_SHOWCASE_PRESETS,
    build_3d_python_export_script,
    build_video_python_export_script,
    compile_animated_pixel_field_frames,
    compile_ir_to_3d_surface,
    compile_ir_to_math_video,
    compile_math_program_to_2d_3d_4d,
    compile_to_math_video,
    evaluate_3d_preset,
    evaluate_video_preset,
    render_math_video_response,
    render_raymarch_video_response,
    sanitize_math_program,
    sanitize_math_text,
    sanitize_raymarch_scene,
)


# ============================================================================
# 1. AUTHENTIC HAMID NADERI YEGANEH ICONIC & SCIENTIFIC AMERICAN PRESETS
# ============================================================================

HAMID_PRESETS: List[Dict[str, Any]] = [
    {
        "id": "bird_in_flight",
        "name": "A Bird in Flight (Parinda)",
        "author": "Hamid Naderi Yeganeh",
        "category": "Hamid Originals",
        "mode": "lines",
        "N": 2000,
        "description": "Hamid Naderi Yeganeh's iconic 2000-line-segment bird in flight using powers of sine and cosine.",
        "latex": {
            "X1": r"X_1(k) = \frac{3}{2}\sin^7\left(\frac{2\pi k}{2000} + \frac{\pi}{3}\right)",
            "Y1": r"Y_1(k) = \frac{1}{4}\cos^2\left(\frac{6\pi k}{2000}\right)",
            "X2": r"X_2(k) = \frac{1}{5}\sin\left(\frac{6\pi k}{2000} + \frac{\pi}{5}\right)",
            "Y2": r"Y_2(k) = -\frac{2}{3}\sin^2\left(\frac{4\pi k}{2000} - \frac{\pi}{3}\right)",
        },
        "formulas": {
            "X1": "1.5 * (np.sin(2*np.pi*k/N + np.pi/3)**7)",
            "Y1": "0.25 * (np.cos(6*np.pi*k/N)**2)",
            "X2": "0.2 * np.sin(6*np.pi*k/N + np.pi/5)",
            "Y2": "-(2.0/3.0) * (np.sin(4*np.pi*k/N - np.pi/3)**2)",
        },
    },
    {
        "id": "japanese_maple_sciam",
        "name": "Japanese Maple Leaf (7-Step SciAm)",
        "author": "Hamid Naderi Yeganeh (Scientific American)",
        "category": "Scientific American",
        "mode": "lines",
        "N": 3000,
        "description": "Hamid Naderi Yeganeh's flagship 7-step Japanese Maple leaf from 'How to Draw with Math' (Scientific American).",
        "latex": {
            "X1": r"X(t) = \left(\frac{4}{15} + \cos^2(8t) + \frac{1}{5}(\cos(8t)\cos(24t))^{10} + \left(\frac{1}{6}\cos^2(80t) + \frac{1}{15}\cos^{10}(80t)\right)(1 - \sin^{10}(8t))\right)\sin(2t)(1 + \sin(t))(1 - \cos^{10}(t)\cos^2(3t)) + \frac{1}{30}\cos^9(t)\cos^{10}(5t)",
            "Y1": r"Y(t) = -\left(\frac{8}{30} + \cos^2(8t) + \frac{1}{5}(\cos(8t)\cos(24t))^{10} + \left(\frac{1}{6}\cos^2(80t) + \frac{1}{15}\cos^{10}(80t)\right)(1 - \cos^{10}(t)\cos^2(3t))(1 - \sin^{10}(8t))\right)\cos(2t)\left(1 + \sin(t) + (\cos(t)\cos(3t))^{10}\right)",
            "X2": r"X_2(k) = X\left(\frac{\pi k}{N}\right),\quad 0 \le t \le \pi",
            "Y2": r"Y_2(k) = Y\left(\frac{\pi k}{N}\right),\quad 0 \le t \le \pi",
        },
        "formulas": {
            "X1": "((4/15 + np.cos(8*np.pi*(k-1)/N)**2 + (1/5)*(np.cos(8*np.pi*(k-1)/N)*np.cos(24*np.pi*(k-1)/N))**10 + ((1/6)*np.cos(80*np.pi*(k-1)/N)**2 + (1/15)*np.cos(80*np.pi*(k-1)/N)**10)*(1 - np.sin(8*np.pi*(k-1)/N)**10)) * np.sin(2*np.pi*(k-1)/N) * (1 + np.sin(np.pi*(k-1)/N)) * (1 - (np.cos(np.pi*(k-1)/N)**10)*(np.cos(3*np.pi*(k-1)/N)**2)) + (1/30)*(np.cos(np.pi*(k-1)/N)**9)*(np.cos(5*np.pi*(k-1)/N)**10))",
            "Y1": "(-(8/30 + np.cos(8*np.pi*(k-1)/N)**2 + (1/5)*(np.cos(8*np.pi*(k-1)/N)*np.cos(24*np.pi*(k-1)/N))**10 + ((1/6)*np.cos(80*np.pi*(k-1)/N)**2 + (1/15)*np.cos(80*np.pi*(k-1)/N)**10)*(1 - (np.cos(np.pi*(k-1)/N)**10)*(np.cos(3*np.pi*(k-1)/N)**2))*(1 - np.sin(8*np.pi*(k-1)/N)**10)) * np.cos(2*np.pi*(k-1)/N) * (1 + np.sin(np.pi*(k-1)/N) + (np.cos(np.pi*(k-1)/N)*np.cos(3*np.pi*(k-1)/N))**10))",
            "X2": "((4/15 + np.cos(8*np.pi*k/N)**2 + (1/5)*(np.cos(8*np.pi*k/N)*np.cos(24*np.pi*k/N))**10 + ((1/6)*np.cos(80*np.pi*k/N)**2 + (1/15)*np.cos(80*np.pi*k/N)**10)*(1 - np.sin(8*np.pi*k/N)**10)) * np.sin(2*np.pi*k/N) * (1 + np.sin(np.pi*k/N)) * (1 - (np.cos(np.pi*k/N)**10)*(np.cos(3*np.pi*k/N)**2)) + (1/30)*(np.cos(np.pi*k/N)**9)*(np.cos(5*np.pi*k/N)**10))",
            "Y2": "(-(8/30 + np.cos(8*np.pi*k/N)**2 + (1/5)*(np.cos(8*np.pi*k/N)*np.cos(24*np.pi*k/N))**10 + ((1/6)*np.cos(80*np.pi*k/N)**2 + (1/15)*np.cos(80*np.pi*k/N)**10)*(1 - (np.cos(np.pi*k/N)**10)*(np.cos(3*np.pi*k/N)**2))*(1 - np.sin(8*np.pi*k/N)**10)) * np.cos(2*np.pi*k/N) * (1 + np.sin(np.pi*k/N) + (np.cos(np.pi*k/N)*np.cos(3*np.pi*k/N))**10))",
        },
    },
    {
        "id": "maple_leaf_sciam",
        "name": "Autumn Maple Leaf (Barg-e-Chinar)",
        "author": "Hamid Naderi Yeganeh (Scientific American)",
        "category": "Scientific American",
        "mode": "lines",
        "N": 2800,
        "description": "Exact Scientific American parametric Maple Leaf with 5 palmate lobes, acuminate tip sharpeners, and marginal serrations.",
        "latex": {
            "X1": r"X(t) = \left(1 + \cos^2(6t) + \frac{1}{5}(\cos(6t)\cos(24t))^{10} + \left(\frac{1}{4}\cos^2(30t) + \frac{1}{9}\cos^{12}(30t)\right)(1 - \sin^{10}(6t))\right)\sin(2t)(1 - \cos^4(t))(1 - \cos^{10}(t)\cos^2(3t)) + \frac{1}{70}\cos^9(t)",
            "Y1": r"Y(t) = -\frac{21}{20}\cos(2t)\left(1 - \cos^4(t) + \frac{1}{2}(\cos(t)\cos(3t))^{10}\right)\left(1 + \cos^2(6t) + \frac{1}{5}(\cos(6t)\cos(18t))^{10} + \left(\frac{1}{4}\cos^4(30t) + \frac{1}{10}\cos^{12}(30t)\right)(1 - \cos^{10}(t)\cos^2(3t))(1 - \sin^{10}(6t))\right)",
            "X2": r"X_2(k) = X\left(\frac{\pi k}{N}\right),\quad 0 \le t \le \pi",
            "Y2": r"Y_2(k) = Y\left(\frac{\pi k}{N}\right),\quad 0 \le t \le \pi",
        },
        "formulas": {
            "X1": "((1 + np.cos(6*np.pi*(k-1)/N)**2 + (1/5)*(np.cos(6*np.pi*(k-1)/N)*np.cos(24*np.pi*(k-1)/N))**10 + ((1/4)*np.cos(30*np.pi*(k-1)/N)**2 + (1/9)*np.cos(30*np.pi*(k-1)/N)**12)*(1 - np.sin(6*np.pi*(k-1)/N)**10)) * np.sin(2*np.pi*(k-1)/N) * (1 - np.cos(np.pi*(k-1)/N)**4) * (1 - (np.cos(np.pi*(k-1)/N)**10)*(np.cos(3*np.pi*(k-1)/N)**2)) + (1/70)*(np.cos(np.pi*(k-1)/N)**9))",
            "Y1": "(-(21/20) * np.cos(2*np.pi*(k-1)/N) * (1 - np.cos(np.pi*(k-1)/N)**4 + 0.5*(np.cos(np.pi*(k-1)/N)*np.cos(3*np.pi*(k-1)/N))**10) * (1 + np.cos(6*np.pi*(k-1)/N)**2 + (1/5)*(np.cos(6*np.pi*(k-1)/N)*np.cos(18*np.pi*(k-1)/N))**10 + ((1/4)*np.cos(30*np.pi*(k-1)/N)**4 + (1/10)*np.cos(30*np.pi*(k-1)/N)**12)*(1 - (np.cos(np.pi*(k-1)/N)**10)*(np.cos(3*np.pi*(k-1)/N)**2))*(1 - np.sin(6*np.pi*(k-1)/N)**10)))",
            "X2": "((1 + np.cos(6*np.pi*k/N)**2 + (1/5)*(np.cos(6*np.pi*k/N)*np.cos(24*np.pi*k/N))**10 + ((1/4)*np.cos(30*np.pi*k/N)**2 + (1/9)*np.cos(30*np.pi*k/N)**12)*(1 - np.sin(6*np.pi*k/N)**10)) * np.sin(2*np.pi*k/N) * (1 - np.cos(np.pi*k/N)**4) * (1 - (np.cos(np.pi*k/N)**10)*(np.cos(3*np.pi*k/N)**2)) + (1/70)*(np.cos(np.pi*k/N)**9))",
            "Y2": "(-(21/20) * np.cos(2*np.pi*k/N) * (1 - np.cos(np.pi*k/N)**4 + 0.5*(np.cos(np.pi*k/N)*np.cos(3*np.pi*k/N))**10) * (1 + np.cos(6*np.pi*k/N)**2 + (1/5)*(np.cos(6*np.pi*k/N)*np.cos(18*np.pi*k/N))**10 + ((1/4)*np.cos(30*np.pi*k/N)**4 + (1/10)*np.cos(30*np.pi*k/N)**12)*(1 - (np.cos(np.pi*k/N)**10)*(np.cos(3*np.pi*k/N)**2))*(1 - np.sin(6*np.pi*k/N)**10)))",
        },
    },
    {
        "id": "oak_leaf_sciam",
        "name": "White Oak Leaf (Barg-e-Baloot)",
        "author": "Hamid Naderi Yeganeh (Scientific American)",
        "category": "Scientific American",
        "mode": "lines",
        "N": 2000,
        "description": "Exact Scientific American White Oak Leaf with rounded pinnate lobes and tapered petiole.",
        "latex": {
            "X1": r"X(t) = \frac{1}{100}\cos^9(t)\cos^{10}(5t) + \frac{1}{4}\sin(2t)\left(1 - \frac{1}{2}\sin^2(10t)\right)\left(1 - (\cos(t)\cos(3t))^8\right)",
            "Y1": r"Y(t) = \sin(t)\left(1 - \frac{1}{5}\sin^2(10t)\left(\frac{1}{2} + \sin^2(2t)\right)\right)",
            "X2": r"X_2(k) = X\left(\frac{\pi k}{N}\right),\quad 0 \le t \le \pi",
            "Y2": r"Y_2(k) = Y\left(\frac{\pi k}{N}\right),\quad 0 \le t \le \pi",
        },
        "formulas": {
            "X1": "((1/100)*(np.cos(np.pi*(k-1)/N)**9)*(np.cos(5*np.pi*(k-1)/N)**10) + 0.25*np.sin(2*np.pi*(k-1)/N)*(1 - 0.5*(np.sin(10*np.pi*(k-1)/N)**2))*(1 - (np.cos(np.pi*(k-1)/N)*np.cos(3*np.pi*(k-1)/N))**8))",
            "Y1": "(np.sin(np.pi*(k-1)/N)*(1 - (1/5)*(np.sin(10*np.pi*(k-1)/N)**2)*(0.5 + np.sin(2*np.pi*(k-1)/N)**2)))",
            "X2": "((1/100)*(np.cos(np.pi*k/N)**9)*(np.cos(5*np.pi*k/N)**10) + 0.25*np.sin(2*np.pi*k/N)*(1 - 0.5*(np.sin(10*np.pi*k/N)**2))*(1 - (np.cos(np.pi*k/N)*np.cos(3*np.pi*k/N))**8))",
            "Y2": "(np.sin(np.pi*k/N)*(1 - (1/5)*(np.sin(10*np.pi*k/N)**2)*(0.5 + np.sin(2*np.pi*k/N)**2)))",
        },
    },
    {
        "id": "olive_branch_sciam",
        "name": "Olive Branch (Shakh-e-Zaitoon)",
        "author": "Hamid Naderi Yeganeh (Scientific American)",
        "category": "Scientific American",
        "mode": "circles",
        "N": 2600,
        "description": "2600 circles tracing a slender olive branch with 21 lanceolate leaflets via even/odd harmonic parity and radius pinching.",
        "latex": {
            "X": r"X(k) = -\cos\left(\frac{\pi k}{N}\right) + \frac{1}{5}\sin^8\left(\frac{21\pi k}{N}\right)\left(1 - \frac{1}{2}\cos^{10}\left(\frac{\pi k}{N}\right)\right) + \frac{1}{75}\sin\left(\frac{42\pi k}{N}\right)\sin^4\left(\frac{21\pi k}{N}\right)",
            "Y": r"Y(k) = -\frac{1}{4}\cos\left(\frac{\pi k}{N}\right) + \frac{1}{12}\sin\left(\frac{\pi k}{N}\right) + \frac{3}{10}\sin^7\left(\frac{21\pi k}{N}\right)\left(1 - \frac{1}{2}\cos^{10}\left(\frac{\pi k}{N}\right)\right)",
            "R": r"R(k) = \frac{1}{130} + \frac{1}{14}\sin^8\left(\frac{21\pi k}{N}\right)\cos^2\left(\frac{21\pi k}{N}\right)\left(1 - \frac{1}{2}\cos^{10}\left(\frac{\pi k}{N}\right)\right)",
        },
        "formulas": {
            "X": "-np.cos(np.pi*k/N) + 0.2*(np.sin(21*np.pi*k/N)**8)*(1 - 0.5*(np.cos(np.pi*k/N)**10)) + (1/75)*np.sin(42*np.pi*k/N)*(np.sin(21*np.pi*k/N)**4)",
            "Y": "-0.25*np.cos(np.pi*k/N) + (1/12)*np.sin(np.pi*k/N) + 0.3*(np.sin(21*np.pi*k/N)**7)*(1 - 0.5*(np.cos(np.pi*k/N)**10))",
            "R": "(1/130) + (1/14)*(np.sin(21*np.pi*k/N)**8)*(np.cos(21*np.pi*k/N)**2)*(1 - 0.5*(np.cos(np.pi*k/N)**10))",
        },
    },
    {
        "id": "palm_branch_sciam",
        "name": "Pinnate Palm Frond (Shakh-e-Khajoor)",
        "author": "Hamid Naderi Yeganeh (Scientific American)",
        "category": "Scientific American",
        "mode": "circles",
        "N": 2800,
        "description": "2800 circles forming an arched palm rachis and 64 sweeping needle-like pinnae via quartic-cubic harmonic deflection.",
        "latex": {
            "X": r"X(k) = -\cos\left(\frac{\pi k}{N}\right) + \frac{2}{7}\sin^4\left(\frac{64\pi k}{N}\right)\sin\left(\frac{\pi k}{N}\right)\left(1 - \frac{1}{4}\cos\left(\frac{\pi k}{N}\right)\right)",
            "Y": r"Y(k) = -\frac{1}{5}\cos\left(\frac{\pi k}{N}\right) + \frac{1}{8}\sin\left(\frac{\pi k}{N}\right) + \frac{2}{7}\sin^3\left(\frac{64\pi k}{N}\right)\sin\left(\frac{\pi k}{N}\right)\left(1 - \frac{1}{4}\cos\left(\frac{\pi k}{N}\right)\right)",
            "R": r"R(k) = \frac{1}{500} + \frac{1}{90}\left(1 + \cos^2\left(\frac{\pi k}{2N}\right)\right)\cos^{10}\left(\frac{64\pi k}{N}\right) + \frac{1}{60}\sin^2\left(\frac{64\pi k}{N}\right)\cos^2\left(\frac{64\pi k}{N}\right)\sin\left(\frac{\pi k}{N}\right)",
        },
        "formulas": {
            "X": "-np.cos(np.pi*k/N) + (2/7)*(np.sin(64*np.pi*k/N)**4)*np.sin(np.pi*k/N)*(1 - 0.25*np.cos(np.pi*k/N))",
            "Y": "-0.2*np.cos(np.pi*k/N) + 0.125*np.sin(np.pi*k/N) + (2/7)*(np.sin(64*np.pi*k/N)**3)*np.sin(np.pi*k/N)*(1 - 0.25*np.cos(np.pi*k/N))",
            "R": "0.002 + (1/90)*(1 + np.cos(np.pi*k/(2*N))**2)*(np.cos(64*np.pi*k/N)**10) + (1/60)*(np.sin(64*np.pi*k/N)**2)*(np.cos(64*np.pi*k/N)**2)*np.sin(np.pi*k/N)",
        },
    },
    {
        "id": "sailing_boat",
        "name": "The Sailing Boat (Kashti)",
        "author": "Hamid Naderi Yeganeh",
        "category": "Hamid Originals",
        "mode": "lines",
        "N": 2000,
        "description": "Delicate sailboat crafted from 2000 harmonic chords connecting cubic sine waves.",
        "latex": {
            "X1": r"X_1(k) = \frac{2}{3}\sin^3\left(\frac{2\pi k}{2000}\right) + \frac{1}{3}\sin\left(\frac{6\pi k}{2000}\right)",
            "Y1": r"Y_1(k) = \frac{1}{2}\cos\left(\frac{2\pi k}{2000}\right) - \frac{1}{3}\cos\left(\frac{4\pi k}{2000}\right)",
            "X2": r"X_2(k) = \frac{1}{4}\sin\left(\frac{4\pi k}{2000} + \frac{\pi}{6}\right)",
            "Y2": r"Y_2(k) = -\frac{1}{2}\sin^2\left(\frac{2\pi k}{2000}\right)",
        },
        "formulas": {
            "X1": "(2/3)*(np.sin(2*np.pi*k/N)**3) + (1/3)*np.sin(6*np.pi*k/N)",
            "Y1": "0.5*np.cos(2*np.pi*k/N) - (1/3)*np.cos(4*np.pi*k/N)",
            "X2": "0.25*np.sin(4*np.pi*k/N + np.pi/6)",
            "Y2": "-0.5*(np.sin(2*np.pi*k/N)**2)",
        },
    },
    {
        "id": "butterfly_wings",
        "name": "Harmonic Butterfly (Titli)",
        "author": "Hamid Style / Temple-Fay",
        "category": "Hamid Originals",
        "mode": "lines",
        "N": 2400,
        "description": "Exponential-trigonometric butterfly curve woven with internal harmonic wing chords.",
        "latex": {
            "X1": r"X_1(k) = \sin\left(\frac{2\pi k}{N}\right)\left(e^{\cos(2\pi k/N)} - 2\cos\left(\frac{8\pi k}{N}\right) - \sin^5\left(\frac{\pi k}{6N}\right)\right)",
            "Y1": r"Y_1(k) = \cos\left(\frac{2\pi k}{N}\right)\left(e^{\cos(2\pi k/N)} - 2\cos\left(\frac{8\pi k}{N}\right) - \sin^5\left(\frac{\pi k}{6N}\right)\right)",
            "X2": r"X_2(k) = \frac{2}{5}\sin\left(\frac{6\pi k}{N}\right)\cos\left(\frac{2\pi k}{N}\right)",
            "Y2": r"Y_2(k) = \frac{2}{5}\cos\left(\frac{4\pi k}{N}\right)",
        },
        "formulas": {
            "X1": "np.sin(2*np.pi*k/N)*(np.exp(np.cos(2*np.pi*k/N)) - 2*np.cos(8*np.pi*k/N) - np.sin(np.pi*k/(6*N))**5)",
            "Y1": "np.cos(2*np.pi*k/N)*(np.exp(np.cos(2*np.pi*k/N)) - 2*np.cos(8*np.pi*k/N) - np.sin(np.pi*k/(6*N))**5)",
            "X2": "0.4*np.sin(6*np.pi*k/N)*np.cos(2*np.pi*k/N)",
            "Y2": "0.4*np.cos(4*np.pi*k/N)",
        },
    },
    {
        "id": "tropical_fish",
        "name": "Angelfish in Motion (Machli)",
        "author": "Hamid Naderi Yeganeh",
        "category": "Hamid Originals",
        "mode": "lines",
        "N": 2000,
        "description": "Graceful fish form emerging from phase-locked trigonometric endpoints.",
        "latex": {
            "X1": r"X_1(k) = \cos\left(\frac{2\pi k}{2000}\right) - \frac{3}{5}\sin^2\left(\frac{2\pi k}{2000}\right)",
            "Y1": r"Y_1(k) = \frac{7}{10}\sin\left(\frac{4\pi k}{2000}\right)\cos\left(\frac{2\pi k}{2000}\right)",
            "X2": r"X_2(k) = -\frac{1}{3}\cos\left(\frac{6\pi k}{2000}\right) + \frac{2}{5}\cos\left(\frac{2\pi k}{2000}\right)",
            "Y2": r"Y_2(k) = \frac{1}{3}\sin^3\left(\frac{2\pi k}{2000}\right)",
        },
        "formulas": {
            "X1": "np.cos(2*np.pi*k/N) - 0.6*(np.sin(2*np.pi*k/N)**2)",
            "Y1": "0.7*np.sin(4*np.pi*k/N)*np.cos(2*np.pi*k/N)",
            "X2": "-(1/3)*np.cos(6*np.pi*k/N) + 0.4*np.cos(2*np.pi*k/N)",
            "Y2": "(1/3)*(np.sin(2*np.pi*k/N)**3)",
        },
    },
    {
        "id": "lotus_circles",
        "name": "Sacred Lotus Mandala (Kanwal Phool)",
        "author": "Hamid Circle Family",
        "category": "Hamid Originals",
        "mode": "circles",
        "N": 1600,
        "description": "1600 concentric and petal-modulated circles governed by harmonic radius oscillation.",
        "latex": {
            "X": r"X(k) = \cos\left(\frac{14\pi k}{1600}\right)\left(1 - \frac{1}{2}\sin^2\left(\frac{2\pi k}{1600}\right)\right)",
            "Y": r"Y(k) = \sin\left(\frac{14\pi k}{1600}\right)\left(1 - \frac{1}{2}\cos^2\left(\frac{2\pi k}{1600}\right)\right)",
            "R": r"R(k) = \frac{1}{25} + \frac{1}{8}\sin^4\left(\frac{28\pi k}{1600}\right)",
        },
        "formulas": {
            "X": "np.cos(14*np.pi*k/N)*(1 - 0.5*(np.sin(2*np.pi*k/N)**2))",
            "Y": "np.sin(14*np.pi*k/N)*(1 - 0.5*(np.cos(2*np.pi*k/N)**2))",
            "R": "0.04 + 0.125*(np.sin(28*np.pi*k/N)**4)",
        },
    },
    {
        "id": "heart_caustic",
        "name": "Caustic Heart (Dil)",
        "author": "Hamid Style",
        "category": "Hamid Originals",
        "mode": "lines",
        "N": 1800,
        "description": "Cardioid-harmonic heart envelope woven from 1800 trigonometric chords.",
        "latex": {
            "X1": r"X_1(k) = \sin^3\left(\frac{2\pi k}{1800}\right)",
            "Y1": r"Y_1(k) = \frac{13\cos\left(\frac{2\pi k}{1800}\right) - 5\cos\left(\frac{4\pi k}{1800}\right) - 2\cos\left(\frac{6\pi k}{1800}\right) - \cos\left(\frac{8\pi k}{1800}\right)}{16}",
            "X2": r"X_2(k) = \frac{2}{5}\sin^3\left(\frac{6\pi k}{1800} + \frac{\pi}{4}\right)",
            "Y2": r"Y_2(k) = \frac{1}{3}\cos\left(\frac{4\pi k}{1800}\right) - \frac{1}{6}\cos\left(\frac{8\pi k}{1800}\right)",
        },
        "formulas": {
            "X1": "np.sin(2*np.pi*k/N)**3",
            "Y1": "(13*np.cos(2*np.pi*k/N) - 5*np.cos(4*np.pi*k/N) - 2*np.cos(6*np.pi*k/N) - np.cos(8*np.pi*k/N))/16.0",
            "X2": "0.4*(np.sin(6*np.pi*k/N + np.pi/4)**3)",
            "Y2": "(1/3)*np.cos(4*np.pi*k/N) - (1/6)*np.cos(8*np.pi*k/N)",
        },
    },
    {
        "id": "seahorse_spiral",
        "name": "Golden Seahorse (Samandari Ghora)",
        "author": "Hamid Style",
        "category": "Hamid Originals",
        "mode": "circles",
        "N": 1500,
        "description": "Parametric circle family tracing an undulating seahorse spine and curled tail.",
        "latex": {
            "X": r"X(k) = \frac{2}{5}\sin\left(\frac{3\pi k}{N}\right)\cos\left(\frac{\pi k}{N}\right) + \frac{1}{4}\cos\left(\frac{10\pi k}{N}\right)\sin^3\left(\frac{\pi k}{N}\right)",
            "Y": r"Y(k) = \cos\left(\frac{\pi k}{N}\right) - \frac{1}{5}\sin\left(\frac{8\pi k}{N}\right)\sin^2\left(\frac{\pi k}{N}\right)",
            "R": r"R(k) = \frac{1}{50} + \frac{1}{7}\sin^3\left(\frac{\pi k}{N}\right)\left(1 + \frac{1}{2}\cos\left(\frac{24\pi k}{N}\right)\right)",
        },
        "formulas": {
            "X": "0.4*np.sin(3*np.pi*k/N)*np.cos(np.pi*k/N) + 0.25*np.cos(10*np.pi*k/N)*(np.sin(np.pi*k/N)**3)",
            "Y": "np.cos(np.pi*k/N) - 0.2*np.sin(8*np.pi*k/N)*(np.sin(np.pi*k/N)**2)",
            "R": "0.02 + (1/7)*(np.sin(np.pi*k/N)**3)*(1 + 0.5*np.cos(24*np.pi*k/N))",
        },
    },
]

ALL_PRESETS: List[Dict[str, Any]] = (
    MASTER_PARADIGM_PRESETS + PIXEL_FIELD_PRESETS + HAMID_PRESETS + MATH_3D_PRESETS + MATH_VIDEO_PRESETS
)


# ============================================================================
# 2. SEMANTIC SHAPE ARCHETYPES & 7-STEP HAMID CONTOUR GENERATORS
# ============================================================================

SYNONYM_MAP: Dict[str, str] = {
    # Scientific American Botanical Masterpieces
    "japanese maple": "japanese_maple", "acer": "japanese_maple", "japanese leaf": "japanese_maple",
    "maple": "leaf", "maple leaf": "leaf", "chinar": "leaf", "leaf": "leaf", "patta": "leaf",
    "oak": "oak", "oak leaf": "oak", "baloot": "oak",
    "olive": "olive", "zaitoon": "olive", "olive branch": "olive",
    "palm": "palm", "khajoor": "palm", "palm tree": "palm", "palm branch": "palm", "palm frond": "palm",
    # Birds
    "bird": "bird", "parinda": "bird", "chiriya": "bird", "sparrow": "bird", "pigeon": "bird", "kabootar": "bird",
    "eagle": "eagle", "shaheen": "eagle", "hawk": "eagle", "falcon": "eagle", "baaz": "eagle", "uqab": "eagle",
    "swan": "swan", "hans": "swan", "duck": "swan", "batakh": "swan",
    "peacock": "peacock", "mor": "peacock",
    "owl": "owl", "ullu": "owl",
    # Marine
    "fish": "fish", "machli": "fish", "machhli": "fish", "goldfish": "fish", "shark": "fish",
    "dolphin": "dolphin", "whale": "dolphin",
    "seahorse": "seahorse", "octopus": "octopus", "starfish": "starfish",
    # Animals & Insects
    "butterfly": "butterfly", "titli": "butterfly", "moth": "butterfly",
    "cat": "cat", "billi": "cat", "kitten": "cat",
    "dog": "dog", "kutta": "dog", "puppy": "dog", "wolf": "dog", "bheriya": "dog",
    "lion": "lion", "sher": "lion", "tiger": "lion",
    "horse": "horse", "ghora": "horse", "unicorn": "horse",
    "dragon": "dragon", "azdaha": "dragon",
    "snake": "snake", "saanp": "snake", "cobra": "snake",
    "spider": "spider", "makri": "spider", "web": "spider",
    "deer": "deer", "hiran": "deer",
    "elephant": "elephant", "haathi": "elephant",
    "rabbit": "rabbit", "khargosh": "rabbit", "bunny": "rabbit",
    # Nature & Plants
    "flower": "flower", "phool": "flower", "daisy": "flower", "sunflower": "flower", "suraj mukhi": "flower",
    "rose": "rose", "gulab": "rose", "gulaab": "rose",
    "lotus": "lotus", "kanwal": "lotus", "mandala": "lotus",
    "tree": "tree", "darakht": "tree", "ped": "tree",
    "mountain": "mountain", "pahar": "mountain", "pahaar": "mountain", "hill": "mountain",
    "wave": "wave", "lehar": "wave", "ocean": "wave", "samandar": "wave",
    # Celestial & Symbols
    "heart": "heart", "dil": "heart", "love": "heart", "pyar": "heart",
    "star": "star", "tara": "star", "sitara": "star",
    "moon": "crescent", "chand": "crescent", "crescent": "crescent", "hilal": "crescent",
    "pakistan": "crescent_star", "chand tara": "crescent_star", "islamic": "crescent_star",
    "sun": "sun", "suraj": "sun", "sooraj": "sun",
    "infinity": "infinity", "anant": "infinity", "loop": "infinity",
    "spiral": "spiral", "galaxy": "spiral", "kehkashan": "spiral", "vortex": "spiral",
    "diamond": "diamond", "heera": "diamond", "gem": "diamond",
    "crown": "crown", "taaj": "crown", "taj": "crown", "king": "crown",
    # Objects & Vehicles & Architecture
    "boat": "boat", "kashti": "boat", "ship": "boat", "jahaz": "boat", "sailboat": "boat", "yacht": "boat",
    "airplane": "airplane", "plane": "airplane", "hawai jahaz": "airplane", "jet": "airplane",
    "rocket": "rocket", "missile": "rocket", "spaceship": "rocket",
    "car": "car", "gaari": "car", "gari": "car", "sports car": "car",
    "guitar": "guitar", "sitar": "guitar", "violin": "guitar", "music": "guitar",
    "sword": "sword", "talwar": "sword", "dagger": "sword", "khanjar": "sword",
    "mosque": "mosque", "masjid": "mosque", "taj mahal": "mosque", "dome": "mosque", "minar": "mosque",
    "eiffel": "eiffel", "eiffel tower": "eiffel", "tower": "eiffel", "minar e pakistan": "eiffel",
    "house": "house", "ghar": "house", "home": "house", "castle": "house",
    "cup": "cup", "coffee": "cup", "chai": "cup", "tea": "cup", "mug": "cup",
    "eye": "eye", "aankh": "eye", "ankh": "eye",
    "face": "face", "chehra": "face", "portrait": "face", "human": "face",
    "umbrella": "umbrella", "chatri": "umbrella",
    "apple": "apple", "seb": "apple",
    "lamp": "lamp", "chiragh": "lamp", "diya": "lamp", "lantern": "lamp",
}


def compute_unit_inward_normal(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    dx = np.gradient(x)
    dy = np.gradient(y)
    norm = np.hypot(dx, dy) + 1e-8
    signed_area = 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)
    sign = 1.0 if signed_area >= 0 else -1.0
    return sign * (-dy / norm), sign * (dx / norm)


def local_contour_dual(x1: np.ndarray, y1: np.ndarray, t: np.ndarray, depth: float = 0.14) -> Tuple[np.ndarray, np.ndarray]:
    """
    Constructs a smooth contour-following inward ribbon spine so non-convex shapes
    never suffer from origin-crossing chords flooding their concavities.
    """
    n = len(x1)
    delta = max(2, n // 45)
    nx, ny = compute_unit_inward_normal(x1, y1)
    x2 = 0.88 * np.roll(x1, -delta) + depth * nx
    y2 = 0.88 * np.roll(y1, -delta) + depth * ny
    return x2, y2


def generate_parametric_contour(archetype: str, num_points: int = 2048) -> Optional[Tuple[np.ndarray, np.ndarray, Tuple[np.ndarray, np.ndarray]]]:
    """
    Returns (x_outer, y_outer, (x_inner, y_inner)) parameterized by t in [0, 2*pi).
    Upgraded with Hamid Naderi Yeganeh's 7-Step Scientific American constructive methodology.
    """
    t = np.linspace(0, 2 * np.pi, num_points, endpoint=False)

    if archetype == "bird":
        x1 = 1.5 * (np.sin(t + np.pi / 3) ** 7)
        y1 = 0.25 * (np.cos(3 * t) ** 2)
        x2 = 0.2 * np.sin(3 * t + np.pi / 5)
        y2 = -(2.0 / 3.0) * (np.sin(2 * t - np.pi / 3) ** 2)
        return x1, y1, (x2, y2)

    elif archetype == "leaf":
        # Exact Scientific American Maple Leaf (maple.jpg)
        u = t / 2.0
        lobe_x = (
            1.0
            + np.cos(6 * u) ** 2
            + 0.2 * (np.cos(6 * u) * np.cos(24 * u)) ** 10
            + (0.25 * (np.cos(30 * u) ** 2) + (1.0 / 9.0) * (np.cos(30 * u) ** 12)) * (1.0 - np.sin(6 * u) ** 10)
        )
        lobe_y = (
            1.0
            + np.cos(6 * u) ** 2
            + 0.2 * (np.cos(6 * u) * np.cos(18 * u)) ** 10
            + (0.25 * (np.cos(30 * u) ** 4) + 0.1 * (np.cos(30 * u) ** 12))
            * (1.0 - (np.cos(u) ** 10) * (np.cos(3 * u) ** 2))
            * (1.0 - np.sin(6 * u) ** 10)
        )
        pinch = 1.0 - (np.cos(u) ** 10) * (np.cos(3 * u) ** 2)
        x1 = lobe_x * np.sin(2 * u) * (1.0 - np.cos(u) ** 4) * pinch + (1.0 / 70.0) * (np.cos(u) ** 9)
        y1 = -(21.0 / 20.0) * np.cos(2 * u) * (1.0 - np.cos(u) ** 4 + 0.5 * (np.cos(u) * np.cos(3 * u)) ** 10) * lobe_y
        vein_r = 0.55 * (np.cos(6 * u) ** 6) * (1.0 - np.cos(u) ** 4) * pinch
        x2 = vein_r * np.sin(2 * u)
        y2 = -0.35 - vein_r * np.cos(2 * u)
        return x1, y1, (x2, y2)

    elif archetype == "japanese_maple":
        # Exact Scientific American 7-Step Japanese Maple Leaf (japanese_maple.jpg)
        u = t / 2.0
        pinch = 1.0 - (np.cos(u) ** 10) * (np.cos(3 * u) ** 2)
        serr = ((1.0 / 6.0) * (np.cos(80 * u) ** 2) + (1.0 / 15.0) * (np.cos(80 * u) ** 10)) * (1.0 - np.sin(8 * u) ** 10)
        base_lobe = (4.0 / 15.0) + (np.cos(8 * u) ** 2) + 0.2 * ((np.cos(8 * u) * np.cos(24 * u)) ** 10)
        x1 = (base_lobe + serr) * np.sin(2 * u) * (1.0 + np.sin(u)) * pinch + (1.0 / 30.0) * (np.cos(u) ** 9) * (np.cos(5 * u) ** 10)
        y1 = -(base_lobe + serr * pinch) * np.cos(2 * u) * (1.0 + np.sin(u) + (np.cos(u) * np.cos(3 * u)) ** 10)
        vein_r = 0.52 * (np.cos(8 * u) ** 8) * (1.0 + np.sin(u)) * pinch
        x2 = vein_r * np.sin(2 * u)
        y2 = -0.25 - vein_r * np.cos(2 * u)
        return x1, y1, (x2, y2)

    elif archetype == "oak":
        # Exact Scientific American White Oak Leaf (oak.jpg)
        u = t / 2.0
        x1 = (1.0 / 100.0) * (np.cos(u) ** 9) * (np.cos(5 * u) ** 10) + 0.25 * np.sin(2 * u) * (1.0 - 0.5 * (np.sin(10 * u) ** 2)) * (1.0 - (np.cos(u) * np.cos(3 * u)) ** 8)
        y1 = np.sin(u) * (1.0 - 0.2 * (np.sin(10 * u) ** 2) * (0.5 + np.sin(2 * u) ** 2))
        x2 = 0.06 * np.sin(2 * u) * (np.cos(10 * u) ** 4) * (1.0 - (np.cos(u) * np.cos(3 * u)) ** 8)
        y2 = 0.88 * np.sin(u)
        return x1, y1, (x2, y2)

    elif archetype == "olive":
        # Scientific American Olive Branch (olive.jpg)
        u = t / 2.0
        env = 1.0 - 0.5 * (np.cos(u) ** 10)
        x_stem = -np.cos(u)
        y_stem = -0.25 * np.cos(u) + (1.0 / 12.0) * np.sin(u)
        x1 = x_stem + 0.2 * (np.sin(21 * u) ** 4) * env + (1.0 / 60.0) * np.sin(42 * u) * (np.sin(21 * u) ** 2)
        y1 = y_stem + 0.32 * (np.sin(21 * u) ** 3) * env
        x2 = x_stem + 0.06 * (np.sin(21 * u) ** 2) * env
        y2 = y_stem + 0.08 * np.sin(21 * u) * env
        return x1, y1, (x2, y2)

    elif archetype == "palm":
        # Scientific American Pinnate Palm Frond (palm.jpg)
        u = t / 2.0
        env = np.sin(u) * (1.0 - 0.25 * np.cos(u))
        x_stem = -np.cos(u)
        y_stem = -0.2 * np.cos(u) + 0.125 * np.sin(u)
        x1 = x_stem + (2.0 / 7.0) * (np.sin(64 * u) ** 4) * env
        y1 = y_stem + (2.0 / 7.0) * (np.sin(64 * u) ** 3) * env
        return x1, y1, (x_stem, y_stem)

    elif archetype == "eagle":
        ir = {
            "components": [
                {
                    "name": "Swept Raptor Wing Leading Edge",
                    "type": "ribbon_spine",
                    "weight": 0.34,
                    "mirror_x": True,
                    "points": [[0.06, 0.12], [0.38, 0.42], [0.76, 0.28], [0.95, 0.08]],
                    "w_start": 0.08,
                    "w_end": 0.02,
                    "w_bulge": 0.11,
                    "w_mod_amp": 0.025,
                    "w_mod_freq": 5.0,
                },
                {
                    "name": "Secondary Wing Plumage Layer",
                    "type": "ribbon_spine",
                    "weight": 0.24,
                    "mirror_x": True,
                    "points": [[0.08, -0.04], [0.36, 0.22], [0.72, 0.12], [0.88, -0.04]],
                    "w_start": 0.07,
                    "w_end": 0.015,
                    "w_bulge": 0.08,
                    "w_mod_amp": 0.02,
                    "w_mod_freq": 6.0,
                },
                {
                    "name": "Eagle Head & Hooked Beak",
                    "type": "polar_harmonic",
                    "weight": 0.12,
                    "center": [0.0, 0.30],
                    "scale": [0.10, 0.13],
                    "r0": 1.0,
                    "harmonics": [{"m": 2, "cos_amp": 0.22}, {"m": 3, "sin_amp": 0.18}],
                    "inner_scale": 0.5,
                },
                {
                    "name": "Muscular Torso",
                    "type": "polar_harmonic",
                    "weight": 0.14,
                    "center": [0.0, 0.02],
                    "scale": [0.13, 0.22],
                    "r0": 1.0,
                    "harmonics": [{"m": 2, "cos_amp": -0.25}],
                    "inner_scale": 0.55,
                },
                {
                    "name": "Fanned Tail Feathers",
                    "type": "polar_harmonic",
                    "weight": 0.16,
                    "center": [0.0, -0.20],
                    "scale": [0.28, 0.32],
                    "theta_range_deg": [220.0, 320.0],
                    "r0": 1.0,
                    "harmonics": [{"m": 7, "cos_amp": 0.22}],
                    "inner_scale": 0.35,
                },
            ]
        }
        return compile_hamid_ir_to_chords(ir, N=num_points)

    elif archetype == "swan":
        x1 = 0.9 * np.cos(t) + 0.35 * np.sin(2 * t) - 0.2 * np.cos(3 * t) + 0.08 * np.sin(9 * t) * (np.sin(t) ** 2)
        y1 = 0.7 * np.sin(t) + 0.35 * np.cos(2 * t) + 0.25 * np.sin(3 * t) - 0.1 * np.cos(5 * t)
        return x1, y1, local_contour_dual(x1, y1, t, 0.14)

    elif archetype == "peacock":
        ir = {
            "components": [
                {
                    "name": "Majestic Fanned Feather Train",
                    "type": "polar_harmonic",
                    "weight": 0.38,
                    "center": [0.0, -0.10],
                    "scale": [0.85, 0.85],
                    "theta_range_deg": [10.0, 170.0],
                    "r0": 0.78,
                    "harmonics": [
                        {"m": 18, "cos_amp": 0.22, "power": 2},
                        {"m": 9, "sin_amp": 0.08, "power": 1},
                    ],
                    "inner_scale": 0.22,
                },
                {
                    "name": "Inner Train Ocelli Feathers",
                    "type": "polar_harmonic",
                    "weight": 0.24,
                    "center": [0.0, -0.10],
                    "scale": [0.54, 0.54],
                    "theta_range_deg": [15.0, 165.0],
                    "r0": 0.78,
                    "harmonics": [
                        {"m": 14, "cos_amp": 0.20, "power": 2},
                    ],
                    "inner_scale": 0.25,
                },
                {
                    "name": "Peacock Body & Breast",
                    "type": "polar_harmonic",
                    "weight": 0.16,
                    "center": [0.0, -0.22],
                    "scale": [0.14, 0.24],
                    "r0": 1.0,
                    "harmonics": [{"m": 2, "cos_amp": -0.22}],
                    "inner_scale": 0.4,
                },
                {
                    "name": "Slender Neck & Head",
                    "type": "ribbon_spine",
                    "weight": 0.14,
                    "points": [[0.0, -0.05], [0.0, 0.20], [0.0, 0.38]],
                    "w_start": 0.06,
                    "w_end": 0.045,
                    "w_bulge": -0.015,
                },
                {
                    "name": "Royal Crown Crest",
                    "type": "polar_harmonic",
                    "weight": 0.08,
                    "center": [0.0, 0.44],
                    "scale": [0.10, 0.10],
                    "theta_range_deg": [35.0, 145.0],
                    "r0": 0.8,
                    "harmonics": [{"m": 10, "cos_amp": 0.35, "power": 2}],
                    "inner_scale": 0.2,
                },
            ]
        }
        return compile_hamid_ir_to_chords(ir, N=num_points)

    elif archetype == "owl":
        r = 0.85 + 0.25 * np.cos(2 * t) - 0.18 * np.cos(4 * t) + 0.15 * np.sin(3 * t)
        x1 = r * np.sin(t) * (1 + 0.15 * np.cos(6 * t))
        y1 = r * np.cos(t) + 0.2 * np.cos(3 * t)
        return x1, y1, local_contour_dual(x1, y1, t, 0.14)

    elif archetype == "boat":
        x1 = (2 / 3) * (np.sin(t) ** 3) + (1 / 3) * np.sin(3 * t)
        y1 = 0.5 * np.cos(t) - (1 / 3) * np.cos(2 * t)
        x2 = 0.25 * np.sin(2 * t + np.pi / 6)
        y2 = -0.5 * (np.sin(t) ** 2)
        return x1, y1, (x2, y2)

    elif archetype == "fish":
        x1 = np.cos(t) - 0.6 * (np.sin(t) ** 2)
        y1 = 0.7 * np.sin(2 * t) * np.cos(t)
        x2 = -(1 / 3) * np.cos(3 * t) + 0.4 * np.cos(t)
        y2 = (1 / 3) * (np.sin(t) ** 3)
        return x1, y1, (x2, y2)

    elif archetype == "dolphin":
        x1 = 1.1 * np.cos(t) - 0.35 * np.sin(2 * t) + 0.18 * np.cos(3 * t)
        y1 = 0.65 * np.sin(t) + 0.4 * np.cos(t) * np.sin(t) + 0.18 * np.sin(3 * t) - 0.12 * np.cos(4 * t)
        return x1, y1, local_contour_dual(x1, y1, t, 0.12)

    elif archetype == "butterfly":
        r = np.exp(np.cos(t)) - 2 * np.cos(4 * t) - (np.sin(t / 12) ** 5)
        x1 = np.sin(t) * r
        y1 = np.cos(t) * r
        x2 = 0.4 * np.sin(3 * t) * np.cos(t)
        y2 = 0.4 * np.cos(2 * t)
        return x1, y1, (x2, y2)

    elif archetype == "heart":
        x1 = np.sin(t) ** 3
        y1 = (13 * np.cos(t) - 5 * np.cos(2 * t) - 2 * np.cos(3 * t) - np.cos(4 * t)) / 16.0
        x2 = 0.4 * (np.sin(3 * t + np.pi / 4) ** 3)
        y2 = (1 / 3) * np.cos(2 * t) - (1 / 6) * np.cos(4 * t)
        return x1, y1, (x2, y2)

    elif archetype == "rose":
        k_rose = 5
        cusp = 0.14 * ((np.cos(k_rose * t) * np.cos(3 * k_rose * t)) ** 8)
        serr = 0.03 * (np.cos(10 * k_rose * t) ** 2) * (1.0 - np.sin(k_rose * t) ** 10)
        r1 = 0.72 + 0.34 * (np.cos(k_rose * t) ** 2) + cusp + serr + 0.12 * np.sin(2 * k_rose * t)
        x1 = r1 * np.cos(t) * (1.0 + 0.12 * np.sin(3 * t))
        y1 = r1 * np.sin(t) * (1.0 + 0.12 * np.cos(3 * t))
        r2 = 0.36 * (0.6 + 0.5 * (np.sin(k_rose * t + np.pi / 3) ** 2) + 0.15 * ((np.sin(k_rose * t) * np.sin(3 * k_rose * t)) ** 6))
        x2 = r2 * np.cos(2 * t + 0.4)
        y2 = r2 * np.sin(2 * t + 0.4)
        return x1, y1, (x2, y2)

    elif archetype == "flower" or archetype == "lotus":
        petals = 7 if archetype == "lotus" else 8
        lobe = np.cos(petals * t / 2.0) ** 2
        sharp_tip = 0.18 * ((np.cos(petals * t / 2.0) * np.cos(3 * petals * t / 2.0)) ** 10)
        r1 = 0.52 + 0.48 * lobe + sharp_tip
        x1 = r1 * np.cos(t)
        y1 = r1 * np.sin(t)
        r2 = 0.22 + 0.16 * (np.cos(petals * t) ** 2)
        x2 = r2 * np.cos(t + np.pi / petals)
        y2 = r2 * np.sin(t + np.pi / petals)
        return x1, y1, (x2, y2)

    elif archetype == "star":
        r1 = 0.75 + 0.42 * np.cos(5 * t) + 0.12 * np.cos(10 * t)
        x1 = r1 * np.sin(t)
        y1 = r1 * np.cos(t)
        r2 = 0.32 - 0.12 * np.cos(5 * t)
        x2 = r2 * np.sin(2 * t + np.pi / 5)
        y2 = r2 * np.cos(2 * t + np.pi / 5)
        return x1, y1, (x2, y2)

    elif archetype == "crescent":
        x1 = -0.75 * np.sin(t / 2) + 0.85 * np.cos(t)
        y1 = 1.05 * np.sin(t)
        x2 = -0.35 + 0.65 * np.cos(t)
        y2 = 0.78 * np.sin(t)
        return x1, y1, (x2, y2)

    elif archetype == "crescent_star":
        ir = {
            "components": [
                {
                    "name": "Islamic Crescent Moon",
                    "type": "crescent_lobe",
                    "weight": 0.65,
                    "center": [-0.10, -0.02],
                    "outer_radius": 0.58,
                    "thickness": 0.28,
                    "arc_deg": 240.0,
                    "rotation_deg": 145.0,
                },
                {
                    "name": "Five-Pointed Star",
                    "type": "polar_harmonic",
                    "weight": 0.35,
                    "center": [0.28, 0.06],
                    "scale": [0.20, 0.20],
                    "r0": 0.72,
                    "harmonics": [
                        {"m": 5, "cos_amp": 0.38, "power": 1},
                        {"m": 10, "cos_amp": 0.10, "power": 1},
                    ],
                    "inner_scale": 0.45,
                },
            ]
        }
        return compile_hamid_ir_to_chords(ir, N=num_points)

    elif archetype == "sun":
        r1 = 0.85 + 0.35 * (np.sin(8 * t) ** 3) + 0.12 * np.cos(16 * t)
        x1 = r1 * np.cos(t)
        y1 = r1 * np.sin(t)
        r2 = 0.42 + 0.08 * np.cos(8 * t)
        x2 = r2 * np.cos(t + np.pi / 8)
        y2 = r2 * np.sin(t + np.pi / 8)
        return x1, y1, (x2, y2)

    elif archetype == "tree":
        crown = (0.75 + 0.32 * np.cos(5 * t) + 0.15 * np.sin(11 * t)) * (np.sin(t / 2) ** 1.5)
        x1 = crown * np.sin(t)
        y1 = 0.25 + crown * np.cos(t * 0.7)
        return x1, y1, local_contour_dual(x1, y1, t, 0.15)

    elif archetype == "guitar":
        waist = 0.72 - 0.32 * np.cos(2 * t) + 0.18 * np.cos(t) - 0.12 * np.cos(4 * t)
        x1 = waist * np.sin(t) * (1 - 0.55 * (np.sin(t / 2) ** 12))
        y1 = -0.2 + 0.95 * np.cos(t) + 0.45 * (np.cos(t) ** 5)
        return x1, y1, local_contour_dual(x1, y1, t, 0.12)

    elif archetype == "infinity":
        denom = 1 + np.sin(t) ** 2
        x1 = 1.25 * np.cos(t) / denom
        y1 = 1.25 * np.sin(t) * np.cos(t) / denom
        x2 = 0.85 * np.cos(t + 0.15) / (1 + np.sin(t + 0.15) ** 2)
        y2 = 0.85 * np.sin(t + 0.15) * np.cos(t + 0.15) / (1 + np.sin(t + 0.15) ** 2)
        return x1, y1, (x2, y2)

    elif archetype == "spiral":
        theta = 3 * t
        r1 = 0.15 + 0.85 * (t / (2 * np.pi))
        x1 = r1 * np.cos(theta)
        y1 = r1 * np.sin(theta)
        x2 = (0.82 * r1) * np.cos(theta - 0.25)
        y2 = (0.82 * r1) * np.sin(theta - 0.25)
        return x1, y1, (x2, y2)

    elif archetype == "crown":
        x1 = np.sin(t) * (0.85 + 0.25 * np.sin(t / 2) ** 2)
        y1 = -0.35 + 0.75 * (np.sin(t / 2) ** 2) * (0.5 + 0.65 * np.abs(np.cos(2.5 * t)))
        return x1, y1, local_contour_dual(x1, y1, t, 0.1)

    elif archetype == "diamond":
        x1 = 0.95 * np.sin(t) * (0.75 + 0.35 * np.sin(t / 2) ** 2)
        y1 = 0.55 * np.cos(t) - 0.35 * (np.sin(t) ** 4)
        return x1, y1, local_contour_dual(x1, y1, t, 0.16)

    elif archetype == "eye":
        x1 = 1.15 * np.sin(t)
        y1 = 0.55 * np.sin(t) * np.cos(t) + 0.35 * np.cos(t) * (1 - 0.4 * np.sin(t) ** 2)
        x2 = 0.42 * np.cos(3 * t)
        y2 = 0.42 * np.sin(3 * t)
        return x1, y1, (x2, y2)

    elif archetype == "cat":
        ears = 0.35 * np.exp(-((np.sin(t - 0.55)) ** 2) / 0.04) + 0.35 * np.exp(-((np.sin(t + 0.55)) ** 2) / 0.04)
        r1 = 0.72 + 0.18 * np.cos(2 * t) + ears
        x1 = r1 * np.sin(t)
        y1 = r1 * np.cos(t)
        return x1, y1, local_contour_dual(x1, y1, t, 0.12)

    elif archetype == "mosque":
        dome = 0.75 * np.sin(t) * (1 + 0.25 * np.cos(t) - 0.2 * np.cos(2 * t))
        spire = 0.35 * (np.maximum(0, np.cos(t)) ** 8)
        minarets = 0.55 * (np.exp(-((t - 1.35) ** 2) / 0.015) + np.exp(-((t - (2 * np.pi - 1.35)) ** 2) / 0.015))
        x1 = dome + 0.35 * np.sin(t)
        y1 = 0.65 * np.cos(t) + spire + minarets - 0.15
        return x1, y1, local_contour_dual(x1, y1, t, 0.1)

    elif archetype == "eiffel":
        s = np.sin(t)
        c = np.cos(t)
        width = 0.12 + 0.72 * ((1 - (c + 1) / 2) ** 2.4)
        x1 = np.sign(s) * width * (np.abs(s) ** 0.5)
        y1 = 1.05 * c
        return x1, y1, local_contour_dual(x1, y1, t, 0.11)

    elif archetype == "airplane":
        fuselage_w = 0.18 * np.sin(t)
        wings = 1.05 * np.sin(t) * np.exp(-((np.cos(t) - 0.1) ** 2) / 0.06)
        tail = 0.45 * np.sin(t) * np.exp(-((np.cos(t) + 0.82) ** 2) / 0.03)
        x1 = fuselage_w + wings + tail
        y1 = 1.05 * np.cos(t) - 0.25 * (np.abs(wings) > 0.2) * np.abs(np.sin(t))
        return x1, y1, local_contour_dual(x1, y1, t, 0.1)

    elif archetype == "rocket":
        body = 0.32 * np.sin(t) * (1 - 0.35 * np.cos(t))
        fins = 0.45 * np.sin(t) * np.exp(-((np.cos(t) + 0.65) ** 2) / 0.04)
        x1 = body + fins
        y1 = 1.05 * np.cos(t)
        return x1, y1, local_contour_dual(x1, y1, t, 0.1)

    elif archetype == "sword":
        blade_w = 0.09 * np.sin(t) * (1 + 0.5 * np.cos(t))
        guard = 0.48 * np.sin(t) * np.exp(-((np.cos(t) + 0.45) ** 2) / 0.015)
        curve = 0.18 * ((np.cos(t) - 0.2) ** 2) * (np.cos(t) > -0.4)
        x1 = blade_w + guard + curve
        y1 = 1.15 * np.cos(t)
        return x1, y1, local_contour_dual(x1, y1, t, 0.06)

    elif archetype == "mountain":
        x1 = 1.2 * np.sin(t)
        peaks = (
            0.75 * np.exp(-((x1 + 0.38) ** 2) / 0.18)
            + 0.95 * np.exp(-((x1 - 0.28) ** 2) / 0.15)
            + 0.45 * np.exp(-((x1 - 0.82) ** 2) / 0.09)
        )
        y1 = np.where(np.cos(t) >= 0, peaks - 0.35, -0.35 + 0.15 * np.cos(3 * t))
        return x1, y1, local_contour_dual(x1, y1, t, 0.12)

    elif archetype == "wave":
        x1 = 1.15 * np.cos(t) + 0.28 * np.sin(2 * t)
        y1 = 0.55 * np.sin(t) + 0.35 * np.cos(2 * t) - 0.2 * np.sin(3 * t)
        return x1, y1, local_contour_dual(x1, y1, t, 0.14)

    elif archetype == "dragon":
        wings = 1.15 * np.sin(t) * (np.cos(2 * t) ** 2) + 0.25 * np.sin(7 * t) * (np.sin(t) ** 2)
        x1 = wings + 0.35 * np.cos(3 * t)
        y1 = 0.85 * np.cos(t) + 0.35 * np.sin(2 * t) - 0.2 * np.cos(5 * t)
        return x1, y1, local_contour_dual(x1, y1, t, 0.12)

    elif archetype == "lion":
        mane = 0.88 + 0.22 * np.cos(14 * t) + 0.15 * np.cos(2 * t)
        x1 = mane * np.sin(t)
        y1 = mane * np.cos(t)
        return x1, y1, local_contour_dual(x1, y1, t, 0.15)

    elif archetype == "horse":
        x1 = 0.65 * np.sin(t) + 0.35 * np.cos(2 * t) - 0.22 * np.sin(3 * t) + 0.15 * np.cos(5 * t)
        y1 = 0.95 * np.cos(t) + 0.28 * np.sin(2 * t) + 0.2 * np.cos(3 * t)
        return x1, y1, local_contour_dual(x1, y1, t, 0.12)

    elif archetype == "umbrella":
        is_top = np.cos(t) >= -0.1
        x1 = np.where(is_top, 1.05 * np.sin(t), 0.08 * np.sin(3 * t))
        scallop = 0.15 * np.abs(np.cos(4 * t))
        y1 = np.where(is_top, 0.15 + 0.75 * np.cos(t) + scallop, -0.75 + 0.45 * np.cos(2 * t))
        return x1, y1, local_contour_dual(x1, y1, t, 0.1)

    elif archetype == "apple":
        r = 0.75 * (1 - 0.25 * np.cos(2 * t)) + 0.18 * np.sin(t)
        x1 = r * np.sin(t) * (1 + 0.15 * np.cos(t))
        y1 = r * np.cos(t) - 0.22 * (np.cos(t) ** 4)
        return x1, y1, local_contour_dual(x1, y1, t, 0.15)

    elif archetype == "cup":
        x1 = 0.72 * np.sin(t) * (0.75 + 0.35 * np.cos(t)) + 0.25 * np.exp(-((t - 1.4) ** 2) / 0.08)
        y1 = 0.65 * np.cos(t) + 0.22 * np.sin(5 * t) * (np.cos(t) > 0.65)
        return x1, y1, local_contour_dual(x1, y1, t, 0.12)

    elif archetype == "lamp":
        x1 = 0.95 * np.sin(t) + 0.35 * np.cos(2 * t)
        y1 = -0.15 + 0.45 * np.cos(t) + 0.45 * np.exp(-((t - 0.85) ** 2) / 0.05)
        return x1, y1, local_contour_dual(x1, y1, t, 0.12)

    elif archetype == "seahorse":
        x1 = 0.42 * np.sin(1.5 * t) * np.cos(0.5 * t) + 0.28 * np.cos(5 * t) * (np.sin(0.5 * t) ** 3)
        y1 = np.cos(0.5 * t) - 0.22 * np.sin(4 * t) * (np.sin(0.5 * t) ** 2)
        return x1, y1, local_contour_dual(x1, y1, t, 0.1)

    elif archetype == "octopus":
        tentacles = 0.45 * np.cos(8 * t) * (np.sin(t / 2) ** 3)
        r = 0.65 + tentacles + 0.25 * (np.cos(t / 2) ** 4)
        x1 = r * np.sin(t) + 0.12 * np.sin(16 * t) * (np.sin(t / 2) ** 2)
        y1 = r * np.cos(t)
        return x1, y1, local_contour_dual(x1, y1, t, 0.1)

    elif archetype == "starfish":
        r = 0.72 + 0.42 * np.cos(5 * t) + 0.08 * np.sin(15 * t)
        x1 = r * np.sin(t + 0.1 * np.cos(5 * t))
        y1 = r * np.cos(t + 0.1 * np.cos(5 * t))
        return x1, y1, local_contour_dual(x1, y1, t, 0.12)

    elif archetype == "dog":
        ears = 0.32 * np.exp(-((t - 0.55) ** 2) / 0.03) + 0.32 * np.exp(-((t - (2 * np.pi - 0.55)) ** 2) / 0.03)
        snout = 0.25 * np.exp(-((t - np.pi) ** 2) / 0.12)
        r = 0.68 + ears + snout + 0.12 * np.cos(2 * t)
        x1 = r * np.sin(t)
        y1 = r * np.cos(t)
        return x1, y1, local_contour_dual(x1, y1, t, 0.12)

    elif archetype == "snake":
        hood = 0.42 * np.exp(-((t - np.pi) ** 2) / 0.18)
        x1 = (0.55 * np.sin(3 * t) + hood * np.sin(t)) * (0.4 + 0.6 * np.sin(t / 2))
        y1 = -0.85 + 1.7 * np.sin(t / 2) + 0.18 * np.cos(3 * t)
        x2 = 0.36 * np.sin(3 * t) * np.sin(t / 2)
        y2 = -0.85 + 1.7 * np.sin(t / 2)
        return x1, y1, (x2, y2)

    elif archetype == "spider":
        r = (0.75 + 0.22 * np.abs(np.cos(4 * t))) * (0.65 + 0.35 * np.cos(8 * t))
        x1 = r * np.sin(t)
        y1 = r * np.cos(t)
        return x1, y1, local_contour_dual(x1, y1, t, 0.14)

    elif archetype == "deer":
        antlers = 0.55 * (np.abs(np.sin(6 * t)) ** 2) * (np.cos(t / 2) ** 4)
        r = 0.58 + antlers - 0.15 * np.cos(2 * t)
        x1 = r * np.sin(t) * (1 + 0.25 * np.cos(t))
        y1 = r * np.cos(t)
        return x1, y1, local_contour_dual(x1, y1, t, 0.1)

    elif archetype == "elephant":
        ears = 0.48 * (np.sin(t) ** 2) * (np.cos(t) > -0.3)
        trunk = 0.55 * np.exp(-((t - np.pi) ** 2) / 0.08)
        r = 0.62 + ears + trunk
        x1 = r * np.sin(t) + 0.18 * np.sin(3 * t) * (t > 2.6) * (t < 3.6)
        y1 = r * np.cos(t)
        return x1, y1, local_contour_dual(x1, y1, t, 0.12)

    elif archetype == "rabbit":
        ears = 0.62 * (np.exp(-((t - 0.38) ** 2) / 0.025) + np.exp(-((t - (2 * np.pi - 0.38)) ** 2) / 0.025))
        r = 0.58 + ears + 0.12 * np.sin(t / 2)
        x1 = r * np.sin(t)
        y1 = r * np.cos(t) - 0.2
        return x1, y1, local_contour_dual(x1, y1, t, 0.1)

    elif archetype == "car":
        x1 = 1.15 * np.cos(t)
        cabin = 0.48 * np.exp(-((x1 + 0.05) ** 2) / 0.28)
        wheels = -0.22 * (np.exp(-((x1 - 0.62) ** 2) / 0.03) + np.exp(-((x1 + 0.62) ** 2) / 0.03))
        y1 = np.where(np.sin(t) >= 0, cabin + 0.12, -0.25 + wheels)
        return x1, y1, local_contour_dual(x1, y1, t, 0.1)

    elif archetype == "house":
        x1 = 0.95 * np.sin(t)
        y1 = np.where(np.cos(t) >= 0, 0.25 + 0.75 * (1 - np.abs(x1)), -0.65 + 0.15 * np.cos(4 * t))
        return x1, y1, local_contour_dual(x1, y1, t, 0.12)

    elif archetype == "face":
        r = 0.85 * (1 + 0.12 * np.cos(t) - 0.08 * np.cos(2 * t))
        x1 = r * np.sin(t) * (0.82 + 0.15 * np.cos(t))
        y1 = r * np.cos(t)
        return x1, y1, local_contour_dual(x1, y1, t, 0.14)

    return None


def generate_text_or_custom_contour(prompt_text: str, num_points: int = 2048) -> Tuple[np.ndarray, np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Renders a crisp typographic emblem and extracts calligraphic ribbon chords
    with zero-length inter-letter bridges.
    """
    clean_text = prompt_text.strip()[:14].upper() or "MATH"
    img_size = 512
    img = Image.new("L", (img_size, img_size), 0)
    draw = ImageDraw.Draw(img)

    font_size = max(56, min(180, int(420 / max(1, len(clean_text) * 0.75))))
    try:
        font = ImageFont.truetype("arialbd.ttf", font_size)
    except Exception:
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except Exception:
            font = ImageFont.load_default()

    bbox = draw.textbbox((0, 0), clean_text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    tx = (img_size - tw) // 2 - bbox[0]
    ty = (img_size - th) // 2 - bbox[1]
    draw.text((tx, ty), clean_text, fill=255, font=font)

    arr = np.array(img)
    return extract_chords_from_binary_mask(arr, num_points=num_points)


def extract_chords_from_binary_mask(mask: np.ndarray, num_points: int = 2048) -> Tuple[np.ndarray, np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Extracts contours and applies Prof. Fourier's Calligraphic Tangent-Ribbon
    with Zero-Length Degenerate Bridges between disconnected sub-contours.
    """
    h, w = mask.shape[:2]
    if np.max(mask) == 0:
        t = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
        return np.cos(t), np.sin(t), (0.85 * np.cos(t + 0.05), 0.85 * np.sin(t + 0.05))

    _, thresh = cv2.threshold(mask, 40, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    valid = [c[:, 0, :].astype(np.float64) for c in contours if len(c) >= 15]
    if not valid:
        t = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
        return np.cos(t), np.sin(t), (0.85 * np.cos(t + 0.05), 0.85 * np.sin(t + 0.05))

    valid.sort(key=lambda pts: np.mean(pts[:, 0]))

    cx, cy = w / 2.0, h / 2.0
    scale = max(w, h) / 2.2

    # Build IR components from each contour so compile_hamid_ir_to_chords inserts C^2 zero-length bridges
    M = len(valid)
    n_bridge = 6 if M > 1 else 0
    total_bridge = M * n_bridge
    n_active = max(M * 16, num_points - total_bridge)
    lens = [float(np.sum(np.hypot(np.diff(c[:, 0], append=c[0, 0]), np.diff(c[:, 1], append=c[0, 1])))) for c in valid]
    total_len = sum(lens) + 1e-6

    allocs = [max(16, int(round((l / total_len) * n_active))) for l in lens]
    allocs[-1] = max(16, allocs[-1] + ((num_points - total_bridge) - sum(allocs)))

    X1_list, Y1_list, X2_list, Y2_list = [], [], [], []
    s_br = np.linspace(0.0, 1.0, n_bridge + 2)[1:-1]
    sigma = 10.0 * (s_br ** 3) - 15.0 * (s_br ** 4) + 6.0 * (s_br ** 5)

    resampled_curves = []
    for pts, n_pts in zip(valid, allocs):
        closed = np.vstack([pts, pts[:1]])
        diffs = np.diff(closed, axis=0, prepend=closed[:1])
        seg = np.hypot(diffs[:, 0], diffs[:, 1])
        seg[0] = 0.0
        cum = np.cumsum(seg)
        if cum[-1] < 1e-6:
            continue
        target_s = np.linspace(0, cum[-1], n_pts, endpoint=False)
        px = (np.interp(target_s, cum, closed[:, 0]) - cx) / scale
        py = -(np.interp(target_s, cum, closed[:, 1]) - cy) / scale
        nx, ny = compute_unit_inward_normal(px, py)
        t_loc = np.linspace(0, 2 * np.pi, n_pts, endpoint=False)
        delta = max(1, n_pts // 50)
        qx = np.roll(px, -delta) + 0.032 * (np.sin(24 * t_loc) ** 2) * nx
        qy = np.roll(py, -delta) + 0.032 * (np.sin(24 * t_loc) ** 2) * ny
        resampled_curves.append((px, py, qx, qy))

    for i, cur in enumerate(resampled_curves):
        X1_list.append(cur[0])
        Y1_list.append(cur[1])
        X2_list.append(cur[2])
        Y2_list.append(cur[3])
        if n_bridge > 0:
            nxt = resampled_curves[(i + 1) % len(resampled_curves)]
            bx = (1.0 - sigma) * cur[0][-1] + sigma * nxt[0][0]
            by = (1.0 - sigma) * cur[1][-1] + sigma * nxt[1][0]
            X1_list.append(bx)
            Y1_list.append(by)
            X2_list.append(bx)
            Y2_list.append(by)

    return (
        np.concatenate(X1_list)[:num_points],
        np.concatenate(Y1_list)[:num_points],
        (np.concatenate(X2_list)[:num_points], np.concatenate(Y2_list)[:num_points]),
    )


# ============================================================================
# 3. FOURIER TRIGONOMETRIC EQUATION SYNTHESIZER WITH LANCZOS SIGMA SMOOTHING
# ============================================================================

def fit_trigonometric_series(
    signal: np.ndarray,
    num_harmonics: int = 64,
    lanczos_power: float = 0.0,
) -> Tuple[float, List[Dict[str, float]]]:
    """
    Decomposes a periodic 1D array f[k] (k = 0..N-1) into its exact
    top-M dominant Fourier trigonometric harmonics with Lanczos sigma-smoothing
    to suppress Gibbs phenomenon at sharp corners and multi-component bridges.
    """
    n = len(signal)
    fft_vals = np.fft.rfft(signal) / n
    a0 = float(np.real(fft_vals[0]))

    raw_harmonics = []
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
                "a": a_m,
                "b": b_m,
                "amp": amp,
            })

    raw_harmonics.sort(key=lambda h: h["amp"], reverse=True)
    selected = raw_harmonics[: max(1, min(num_harmonics, len(raw_harmonics)))]
    selected.sort(key=lambda h: h["m"])

    if selected and lanczos_power > 0:
        m_cutoff = max(h["m"] for h in selected) + 1.0
        for h in selected:
            x = math.pi * h["m"] / m_cutoff
            sigma = (math.sin(x) / x) ** lanczos_power if x > 1e-8 else 1.0
            h["a"] = round(h["a"] * sigma, 5)
            h["b"] = round(h["b"] * sigma, 5)
            h["amp"] = round(h["amp"] * sigma, 5)
    else:
        for h in selected:
            h["a"] = round(h["a"], 5)
            h["b"] = round(h["b"], 5)
            h["amp"] = round(h["amp"], 5)

    return round(a0, 5), selected


def evaluate_trigonometric_series(a0: float, harmonics: List[Dict[str, float]], k: np.ndarray, N: int) -> np.ndarray:
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


def format_series_to_latex(var_name: str, a0: float, harmonics: List[Dict[str, float]], N: int, max_display_terms: int = 7) -> str:
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

    pretty_var = {"X1": "X_1", "Y1": "Y_1", "X2": "X_2", "Y2": "Y_2"}.get(var_name, var_name)
    return rf"{pretty_var}(k) = {expr}"


def format_series_to_python_expr(a0: float, harmonics: List[Dict[str, float]]) -> str:
    parts = []
    if abs(a0) >= 5e-5:
        parts.append(f"{a0:.4f}")
    for h in harmonics:
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
# 4. CALLIGRAPHIC TANGENT-CHORD & CAUSTIC WEAVE MODES (Prof. Fourier)
# ============================================================================

def apply_weave_style(
    x1: np.ndarray,
    y1: np.ndarray,
    dual: Optional[Tuple[np.ndarray, np.ndarray]],
    mode: str,
    weave_style: str,
    is_prewoven_ir: bool = False,
) -> Dict[str, np.ndarray]:
    """
    Applies Hamid Naderi Yeganeh chord-weaving or circle-packing transformations.
    When is_prewoven_ir=True, the ADG-IR compiler has already woven local Frenet-Serret
    caustics and inserted C^2 zero-length degenerate bridges (P1 == P2).
    """
    n = len(x1)
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)

    if dual is not None:
        x2_base, y2_base = dual
    else:
        x2_base, y2_base = local_contour_dual(x1, y1, t, 0.14)

    max_val = max(
        float(np.max(np.abs(x1))),
        float(np.max(np.abs(y1))),
        float(np.max(np.abs(x2_base))),
        float(np.max(np.abs(y2_base))),
        1e-5,
    )
    x1 = x1 / max_val
    y1 = y1 / max_val
    x2_base = x2_base / max_val
    y2_base = y2_base / max_val

    if is_prewoven_ir:
        if mode == "circles":
            cx = 0.5 * (x1 + x2_base)
            cy = 0.5 * (y1 + y2_base)
            r = np.maximum(0.002, 0.5 * np.hypot(x1 - x2_base, y1 - y2_base))
            return {"X": cx, "Y": cy, "R": r}
        return {"X1": x1, "Y1": y1, "X2": x2_base, "Y2": y2_base}

    if mode == "circles":
        cx = 0.76 * x1 + 0.24 * x2_base
        cy = 0.76 * y1 + 0.24 * y2_base
        dist = np.hypot(x1 - x2_base, y1 - y2_base)
        r = 0.008 + 0.18 * dist * (0.55 + 0.45 * (np.sin(24 * t) ** 2))
        return {"X": cx, "Y": cy, "R": r}

    # Mode == "lines" — Smooth Calligraphic Tangent-Chord Caustic Envelopes
    if weave_style == "harmonic_web":
        delta = max(2, int(round(0.015 * n)))
        x1_shift = np.roll(x1, -delta)
        y1_shift = np.roll(y1, -delta)
        alpha = 0.38
        x2 = (1.0 - alpha) * x1_shift + alpha * x2_base
        y2 = (1.0 - alpha) * y1_shift + alpha * y2_base
    elif weave_style == "phase_chord":
        shift = max(3, int(round(0.022 * n)))
        nx, ny = compute_unit_inward_normal(x1, y1)
        x2 = np.roll(x1, -shift) + 0.03 * nx
        y2 = np.roll(y1, -shift) + 0.03 * ny
    elif weave_style == "radial_spine":
        x2, y2 = x2_base, y2_base
    else:
        x2, y2 = x2_base, y2_base

    return {"X1": x1, "Y1": y1, "X2": x2, "Y2": y2}


# ============================================================================
# 5. MAIN PROMPT / AI ARCHITECT / PRESET ORCHESTRATOR
# ============================================================================

def resolve_prompt_to_archetypes(prompt: str) -> List[Tuple[str, str]]:
    p = prompt.lower().strip()
    has_conjunction = bool(re.search(r"\b(and|aur|with|over|&|\+)\b", p))
    max_matches = 2 if has_conjunction else 1
    found: List[str] = []

    for phrase, arch in sorted(SYNONYM_MAP.items(), key=lambda item: -len(item[0])):
        if re.search(rf"\b{re.escape(phrase)}\b", p):
            if arch not in found:
                found.append(arch)
                p = re.sub(rf"\b{re.escape(phrase)}\b", " ", p)
            if len(found) >= max_matches:
                break

    if not found:
        for phrase, arch in sorted(SYNONYM_MAP.items(), key=lambda item: -len(item[0])):
            if phrase in p and len(phrase) >= 3:
                if arch not in found:
                    found.append(arch)
                if len(found) >= max_matches:
                    break

    if found:
        return [("archetype", a) for a in found]

    return [("ai_query", prompt.strip())]


def is_complex_query(prompt: str, matches: List[Tuple[str, str]]) -> bool:
    """
    Determines whether a prompt should be routed to the Multi-Tier Differential-Geometric
    Anatomical IR Compiler. Any unseen concept or multi-word descriptive scene routes
    intelligently to the compiler instead of simple templates or ASCII text.
    """
    if matches[0][0] == "ai_query":
        return True
    p = prompt.lower().strip()
    words = [w for w in re.split(r"\s+", p) if w]
    if len(words) >= 4:
        return True
    complex_triggers = [
        "faisal", "badshahi", "minar", "bicycle", "cycle", "bike", "watch", "clock",
        "teapot", "violin", "phoenix", "astronaut", "helicopter", "train", "locomotive",
        "bridge", "castle", "palace", "lantern", "chariot", "dancing", "riding", "over",
        "with", "under", "near", "scene", "landscape", "city", "robot", "skull", "anchor",
        "compass", "hourglass", "telescope", "microscope", "windmill", "lighthouse",
        "chandelier", "scorpion", "crab", "turtle", "tortoise", "penguin", "flamingo",
        "giraffe", "camel", "oont", "kangaroo", "panda", "koala",
        "jellyfish", "saturn", "rings", "space", "floating", "glowing", "roaring",
        "bengal", "tiger", "jungle", "snowy", "sunset", "sunrise", "pine", "forest",
        "volcano", "aurora", "waterfall", "nebula", "cosmos", "underwater", "reef",
        "mermaid", "samurai", "knight", "wizard", "ship", "pirate", "cathedral",
    ]
    if any(kw in p for kw in complex_triggers):
        return True
    return False


def _match_exact_preset_id(prompt: str, preset_list: List[Dict[str, Any]]) -> Optional[str]:
    p = prompt.strip().lower()
    for pr in preset_list:
        if p == pr["id"].lower():
            return pr["id"]
    return None


def _build_fallback_ir_for_prompt(prompt: str, matches: List[Tuple[str, str]]) -> Dict[str, Any]:
    kind, target = matches[0]
    label = target.replace("_", " ").title() if kind == "archetype" else prompt.strip().title()
    p_low = prompt.lower()
    if any(k in p_low for k in ["bird", "eagle", "shaheen", "owl", "swan", "butterfly"]):
        return {
            "title": f"{label} — Double-Exponential Level-Set Synthesis",
            "bg_style": "bokeh_forest",
            "components": [
                {
                    "name": "Swept Primary Wing Manifold",
                    "type": "ribbon_spine",
                    "z_order": 2,
                    "weight": 0.35,
                    "mirror_x": True,
                    "points": [[0.05, 0.08], [0.42, 0.44], [0.84, 0.22]],
                    "w_start": 0.11,
                    "w_end": 0.03,
                    "w_bulge": 0.12,
                    "color_rgb": [0.16, 0.62, 0.85],
                    "accent_rgb": [0.95, 0.74, 0.24],
                    "texture_type": "feathers",
                },
                {
                    "name": "Secondary Plumage Layer",
                    "type": "ribbon_spine",
                    "z_order": 1,
                    "weight": 0.25,
                    "mirror_x": True,
                    "points": [[0.06, -0.04], [0.38, 0.22], [0.76, 0.02]],
                    "w_start": 0.09,
                    "w_end": 0.025,
                    "w_bulge": 0.09,
                    "color_rgb": [0.12, 0.48, 0.74],
                    "accent_rgb": [0.42, 0.88, 0.95],
                    "texture_type": "feathers",
                },
                {
                    "name": "Central Torso & Breast",
                    "type": "polar_harmonic",
                    "z_order": 3,
                    "weight": 0.18,
                    "center": [0.0, 0.02],
                    "scale": [0.16, 0.26],
                    "r0": 1.0,
                    "harmonics": [{"m": 2, "cos_amp": -0.22}],
                    "color_rgb": [0.72, 0.46, 0.20],
                    "accent_rgb": [0.96, 0.82, 0.42],
                    "texture_type": "feathers",
                },
                {
                    "name": "Fanned Tail Plume",
                    "type": "polar_harmonic",
                    "z_order": 1,
                    "weight": 0.14,
                    "center": [0.0, -0.24],
                    "scale": [0.30, 0.34],
                    "theta_range_deg": [210.0, 330.0],
                    "r0": 1.0,
                    "harmonics": [{"m": 7, "cos_amp": 0.24}],
                    "color_rgb": [0.86, 0.56, 0.18],
                    "accent_rgb": [0.98, 0.88, 0.52],
                    "texture_type": "feathers",
                },
                {
                    "name": "Crested Head",
                    "type": "polar_harmonic",
                    "z_order": 4,
                    "weight": 0.08,
                    "center": [0.0, 0.32],
                    "scale": [0.11, 0.13],
                    "r0": 1.0,
                    "harmonics": [{"m": 3, "sin_amp": 0.18}],
                    "color_rgb": [0.94, 0.86, 0.62],
                    "accent_rgb": [0.98, 0.68, 0.16],
                    "texture_type": "feathers",
                },
            ],
        }
    return {
        "title": f"{label} — Double-Exponential Level-Set Synthesis",
        "bg_style": "golden_bokeh",
        "components": [
            {
                "name": "Outer Harmonic Corolla Manifold",
                "type": "polar_harmonic",
                "z_order": 1,
                "weight": 0.42,
                "center": [0.0, 0.02],
                "scale": [0.56, 0.56],
                "r0": 0.85,
                "harmonics": [{"m": 7, "cos_amp": 0.26, "power": 2}, {"m": 14, "cos_amp": 0.08, "power": 2}],
                "color_rgb": [0.86, 0.24, 0.28],
                "accent_rgb": [0.98, 0.76, 0.22],
                "texture_type": "petals",
            },
            {
                "name": "Inner Harmonic Rosette",
                "type": "polar_harmonic",
                "z_order": 2,
                "weight": 0.32,
                "center": [0.0, 0.02],
                "scale": [0.34, 0.34],
                "r0": 0.82,
                "harmonics": [{"m": 7, "sin_amp": 0.24, "power": 2}],
                "color_rgb": [0.95, 0.52, 0.18],
                "accent_rgb": [1.0, 0.92, 0.55],
                "texture_type": "petals",
            },
            {
                "name": "Central Stamen Wheel",
                "type": "wheel_rosette",
                "z_order": 3,
                "weight": 0.14,
                "center": [0.0, 0.02],
                "outer_radius": 0.16,
                "hub_radius": 0.04,
                "spokes": 12,
                "color_rgb": [0.96, 0.82, 0.24],
                "accent_rgb": [1.0, 0.96, 0.78],
                "texture_type": "metallic",
            },
            {
                "name": "Botanical Stem & Calyx",
                "type": "ribbon_spine",
                "z_order": 0,
                "weight": 0.12,
                "points": [[0.0, -0.05], [-0.04, -0.45], [0.02, -0.82]],
                "w_start": 0.055,
                "w_end": 0.035,
                "w_bulge": 0.02,
                "color_rgb": [0.18, 0.68, 0.38],
                "accent_rgb": [0.62, 0.88, 0.32],
                "texture_type": "foliage",
            },
        ],
    }


def build_python_export_script(title: str, mode: str, N: int, py_formulas: Dict[str, str]) -> str:
    clean_title = sanitize_math_text(title)
    if mode == "lines":
        script = f'''"""
Mathematical Art: {clean_title}
Style: Hamid Naderi Yeganeh Parametric Line Segments (k = 1..{N})
Generated by DesignWithMath
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

N = {N}
k = np.arange(1, N + 1, dtype=np.float64)

# Exact Trigonometric Equations for Endpoints (X1(k), Y1(k)) and (X2(k), Y2(k))
X1 = {py_formulas["X1"]}
Y1 = {py_formulas["Y1"]}
X2 = {py_formulas["X2"]}
Y2 = {py_formulas["Y2"]}

fig, ax = plt.subplots(figsize=(8, 8), facecolor="#07090e")
ax.set_facecolor("#07090e")
segments = np.stack([np.column_stack([X1, Y1]), np.column_stack([X2, Y2])], axis=1)
lc = LineCollection(segments, linewidths=0.5, colors="#38bdf8", alpha=0.45)
ax.add_collection(lc)
ax.autoscale()
ax.set_aspect("equal")
ax.axis("off")
plt.tight_layout()
plt.show()
'''
    else:
        script = f'''"""
Mathematical Art: {clean_title}
Style: Hamid Naderi Yeganeh Parametric Circle Family (k = 1..{N})
Generated by DesignWithMath
"""
import numpy as np
import matplotlib.pyplot as plt

N = {N}
k = np.arange(1, N + 1, dtype=np.float64)

# Exact Trigonometric Equations for Center (X(k), Y(k)) and Radius R(k)
X = {py_formulas["X"]}
Y = {py_formulas["Y"]}
R = np.abs({py_formulas["R"]})

fig, ax = plt.subplots(figsize=(8, 8), facecolor="#07090e")
ax.set_facecolor("#07090e")
for i in range(N):
    circle = plt.Circle((X[i], Y[i]), R[i], fill=False, edgecolor="#38bdf8", linewidth=0.5, alpha=0.4)
    ax.add_patch(circle)
ax.autoscale()
ax.set_aspect("equal")
ax.axis("off")
plt.tight_layout()
plt.show()
'''
    return script


def evaluate_preset(
    preset_id: str,
    custom_N: Optional[int] = None,
    t_phase: float = 0.0,
) -> Dict[str, Any]:
    if any(p["id"] == preset_id for p in MASTER_PARADIGM_PRESETS):
        res = evaluate_master_paradigm_preset(
            preset_id,
            t_phase=float(t_phase),
            N=int(custom_N) if custom_N else 1800,
            include_raw_array=False,
        )
        res.pop("rgb_array", None)
        res["title"] = sanitize_math_text(res.get("title", ""))
        res["subtitle"] = sanitize_math_text(res.get("subtitle", ""))
        res["author"] = sanitize_math_text(res.get("author", "Master Mathematical Paradigms Engine"))
        res["t_phase"] = float(t_phase)
        return res

    rcin_match = next((p for p in RAYMARCH_CINEMA_SHOWCASE_PRESETS if p["id"] == preset_id), None)
    if rcin_match is not None:
        return generate_math_art_from_prompt(
            prompt=rcin_match["promptFallback"],
            N=custom_N or 1600,
            mode="raymarch_cinema",
            use_ai=True,
            t_phase=float(t_phase),
            refine=True,
        )

    if any(p["id"] == preset_id for p in PIXEL_FIELD_PRESETS):
        res = evaluate_pixel_field_preset(preset_id)
        res.pop("rgb_array", None)
        res["title"] = sanitize_math_text(res.get("title", ""))
        res["subtitle"] = sanitize_math_text(res.get("subtitle", ""))
        res["author"] = sanitize_math_text(res.get("author", "Hamid Naderi Yeganeh"))
        return res

    pvid_match = next((p for p in PIXEL_VIDEO_SHOWCASE_PRESETS if p["id"] == preset_id), None)
    if pvid_match is not None:
        return generate_math_art_from_prompt(
            prompt=pvid_match["promptFallback"],
            N=custom_N or 1600,
            mode="pixel_video",
            use_ai=True,
            refine=True,
        )

    if preset_id == "video_mechanical_bicycle":
        return generate_math_art_from_prompt(
            prompt="A detailed mechanical bicycle",
            N=custom_N or 2000,
            mode="math_video",
            use_ai=True,
        )

    if any(p["id"] == preset_id for p in MATH_3D_PRESETS):
        return evaluate_3d_preset(preset_id, custom_N=custom_N)

    if any(p["id"] == preset_id for p in MATH_VIDEO_PRESETS):
        return evaluate_video_preset(preset_id, custom_N=custom_N)

    preset = next((p for p in HAMID_PRESETS if p["id"] == preset_id), HAMID_PRESETS[0])
    N = custom_N or preset["N"]
    k = np.arange(1, N + 1, dtype=np.float64)
    env = {"np": np, "k": k, "N": float(N)}

    mode = preset["mode"]
    formulas = preset["formulas"]

    if mode == "lines":
        x1 = eval(formulas["X1"], {"__builtins__": {}}, env)
        y1 = eval(formulas["Y1"], {"__builtins__": {}}, env)
        x2 = eval(formulas["X2"], {"__builtins__": {}}, env)
        y2 = eval(formulas["Y2"], {"__builtins__": {}}, env)
        max_abs = max(float(np.max(np.abs(x1))), float(np.max(np.abs(y1))), float(np.max(np.abs(x2))), float(np.max(np.abs(y2))), 1e-5)
        primitives = np.column_stack([x1 / max_abs, y1 / max_abs, x2 / max_abs, y2 / max_abs]).round(4).tolist()
    else:
        x = eval(formulas["X"], {"__builtins__": {}}, env)
        y = eval(formulas["Y"], {"__builtins__": {}}, env)
        r = np.abs(eval(formulas["R"], {"__builtins__": {}}, env))
        max_abs = max(float(np.max(np.abs(x) + r)), float(np.max(np.abs(y) + r)), 1e-5)
        primitives = np.column_stack([x / max_abs, y / max_abs, r / max_abs]).round(4).tolist()

    return {
        "title": sanitize_math_text(preset["name"]),
        "subtitle": sanitize_math_text(preset["description"]),
        "author": sanitize_math_text(preset["author"]),
        "mode": mode,
        "N": N,
        "harmonics_count": "Closed-Form Exact",
        "latex": preset["latex"],
        "formulas": formulas,
        "primitives": primitives,
        "python_code": build_python_export_script(preset["name"], mode, N, formulas),
    }


def _should_use_symbolic_math_program(prompt: str, matches: List[Tuple[str, str]], use_ai: bool) -> bool:
    """
    Determines whether a prompt in wireframe modes (`lines`, `3d_surface`, `math_video`)
    should compile via the Phase 2 Direct Symbolic Math Program + Iso-Contour Extractor.
    Always True when already cached in `math_program_cache.json` (instant cross-mode switch)
    or for unseen multi-word / descriptive scenes not covered by curated 2D chord blueprints.
    """
    p_key = prompt.strip().lower()
    if p_key in getattr(ai_architect, "_MATH_PROGRAM_CACHE", {}):
        return True
    if not use_ai:
        return False
    if get_curated_complex_ir(p_key) is not None:
        return False
    words = [w for w in re.split(r"\s+", p_key) if w]
    if len(words) <= 2 and matches and matches[0][0] == "archetype":
        return False
    return True


def _evaluate_math_program_pixel_response(
    prompt: str,
    mode: str,
    t_phase: float = 0.0,
    num_frames: int = 0,
    refine: bool = True,
    bypass_cache: bool = False,
) -> Optional[Dict[str, Any]]:
    """
    Evaluates an unseen prompt via Phase 2 Direct Symbolic Math Compiler
    (`generate_math_program_with_passes` -> `evaluate_math_program_numpy`)
    for both `"pixel_field"` and `"pixel_video"` (4D Full-Color GPU Shader Video).
    """
    try:
        pass_res = generate_math_program_with_passes(
            prompt,
            refine=bool(refine),
            bypass_cache=bool(bypass_cache),
        )
        prog = pass_res.get("program")
        if not isinstance(prog, dict):
            return None
        ok, _ = validate_math_program(prog)
        if not ok:
            return None

        clean_prog = sanitize_math_program(prog)
        eval_res = evaluate_math_program_numpy(
            clean_prog,
            width=640,
            height=400,
            t_phase=float(t_phase),
            include_raw_array=False,
        )
        eval_res.pop("rgb_array", None)

        out_mode = "pixel_video" if mode == "pixel_video" else "pixel_field"
        latex_eqs = eval_res.get("latex_equations") or eval_res.get("latex") or {}
        clean_title = sanitize_math_text(
            eval_res.get("title") or clean_prog.get("title") or f"{prompt.strip().title()} — Symbolic Math Painting"
        )
        clean_subtitle = sanitize_math_text(
            eval_res.get("subtitle") or clean_prog.get("subtitle") or "Unified Pure-Math Analytical SDF & Double-Exponential Field"
        )

        res: Dict[str, Any] = {
            "title": clean_title,
            "subtitle": clean_subtitle,
            "author": "Symbolic Differential-Geometric Math Compiler",
            "mode": out_mode,
            "N": eval_res["height"],
            "width": eval_res["width"],
            "height": eval_res["height"],
            "aspect": eval_res.get("aspect", [16, 10]),
            "nominal_aspect": eval_res.get("nominal_aspect", [2000, 1250]),
            "harmonics_count": "4D Pure-Math GPU Shader" if out_mode == "pixel_video" else "2D Pure-Math AST Exact",
            "image_base64": eval_res["image_base64"],
            "glsl_fragment_shader": eval_res["glsl_fragment_shader"],
            "latex": latex_eqs,
            "latex_equations": latex_eqs,
            "primitives": [],
            "python_code": eval_res["python_code"],
            "animation_capable": True,
            "refinement_passes": int(pass_res.get("refinement_passes", 1)),
            "visual_score": pass_res.get("visual_score"),
            "motion_energy": pass_res.get("motion_energy"),
            "stats": eval_res.get("stats", {}),
            "math_program": clean_prog,
            "t_phase": float(t_phase),
        }

        if int(num_frames) > 1:
            n_fr = max(2, min(24, int(num_frames)))
            frames_b64: List[str] = []
            for m in range(n_fr):
                tm = 2.0 * math.pi * (float(m) / float(n_fr))
                f_res = evaluate_math_program_numpy(
                    clean_prog,
                    width=480,
                    height=300,
                    t_phase=tm,
                    include_raw_array=False,
                )
                frames_b64.append(f_res["image_base64"])
            res["frames_base64"] = frames_b64
            res["num_frames"] = n_fr

        return res
    except Exception as exc:
        print(f"[MathEngine] Symbolic math_program pixel fallback triggered: {exc}")
        return None


def _evaluate_raymarch_cinema_response(
    prompt: str,
    N: int = 1600,
    t_phase: float = 0.0,
    num_frames: int = 0,
    refine: bool = True,
    bypass_cache: bool = False,
) -> Dict[str, Any]:
    """
    Stage-2 Universal 3D Raymarched Implicit SDF + Differential-Geometry + PDE Cinema:
    Synthesizes a 12-22 node 3D raymarched scene via `generate_3d_raymarch_scene_with_critique`
    and compiles it deterministically into a true 3D sphere-traced NumPy render,
    WebGL1 60 FPS GPU fragment shader, exact 3D surface manifold (`primitives_3d`,
    `primitives`, `time_harmonics`), and closed-form LaTeX equations.
    """
    pass_res = generate_3d_raymarch_scene_with_critique(
        prompt,
        refine=bool(refine),
        bypass_cache=bool(bypass_cache),
    )
    scene = pass_res.get("scene") or pass_res.get("program")
    if not isinstance(scene, dict):
        raise ValueError("Unable to synthesize 3D raymarch scene specification.")
    ok, err = validate_raymarch_scene(scene)
    if not ok:
        raise ValueError(f"Invalid 3D raymarch scene: {err}")

    clean_scene = sanitize_raymarch_scene(scene)
    eval_res = evaluate_raymarch_scene_numpy(
        clean_scene,
        width=520,
        height=325,
        t_phase=float(t_phase),
        N=max(400, min(3000, int(N))),
        include_raw_array=False,
    )
    eval_res.pop("rgb_array", None)

    latex_eqs = eval_res.get("latex_equations") or eval_res.get("latex") or {}
    clean_title = sanitize_math_text(
        eval_res.get("title") or clean_scene.get("title") or f"{prompt.strip().title()} — 3D Raymarch Cinema"
    )
    clean_subtitle = sanitize_math_text(
        eval_res.get("subtitle") or clean_scene.get("subtitle") or "3D Sphere-Traced Signed Distance Field & Differential-Geometry Cinema"
    )

    res: Dict[str, Any] = {
        "title": clean_title,
        "subtitle": clean_subtitle,
        "author": "3D Implicit SDF & Differential-Geometry Cinema Engine",
        "mode": "raymarch_cinema",
        "N": int(eval_res.get("N", N)),
        "width": int(eval_res["width"]),
        "height": int(eval_res["height"]),
        "aspect": eval_res.get("aspect", [16, 10]),
        "nominal_aspect": eval_res.get("nominal_aspect", [1920, 1200]),
        "harmonics_count": eval_res.get("harmonics_count", f"{len(clean_scene.get('nodes', []))} Coupled 3D SDF Manifolds"),
        "image_base64": eval_res["image_base64"],
        "glsl_fragment_shader": eval_res["glsl_fragment_shader"],
        "latex": latex_eqs,
        "latex_equations": latex_eqs,
        "primitives": eval_res.get("primitives", []),
        "primitives_3d": eval_res.get("primitives_3d", []),
        "time_harmonics": eval_res.get("time_harmonics", []),
        "python_code": eval_res["python_code"],
        "animation_capable": True,
        "refinement_passes": int(pass_res.get("refinement_passes", 1)),
        "visual_score": pass_res.get("visual_score"),
        "motion_energy": pass_res.get("motion_energy"),
        "stats": eval_res.get("stats", {}),
        "raymarch_scene": clean_scene,
        "t_phase": float(t_phase),
    }

    if int(num_frames) > 1:
        n_fr = max(2, min(24, int(num_frames)))
        frames_b64: List[str] = []
        for m in range(n_fr):
            tm = 2.0 * math.pi * (float(m) / float(n_fr))
            f_res = evaluate_raymarch_scene_numpy(
                clean_scene,
                width=420,
                height=262,
                t_phase=tm,
                N=400,
                include_raw_array=False,
            )
            frames_b64.append(f_res["image_base64"])
        res["frames_base64"] = frames_b64
        res["num_frames"] = n_fr

    return res


def direct_long_script_cinema_film(
    script: str,
    width: int = 520,
    height: int = 325,
    fps: int = 12,
    frames_per_act: int = 16,
    transition_frames: int = 6,
    max_acts: int = 6,
    refine_acts: bool = True,
    save_latest_artifacts: bool = True,
) -> Dict[str, Any]:
    """
    End-to-End Long-Script Multi-Act Mathematical Cinema Director:
    Decomposes ANY long multi-paragraph / multi-act script into M sequential mathematical
    acts across the 8 Master Paradigms & Stage-2 3D Raymarching Engine, applies analytical
    C^2 Homotopy Morph Transitions between consecutive acts, burns in Act badges +
    script narration subtitles + live equation tickers, saves complete MP4/GIF/storyboard
    sheet, and returns a 100% JSON-serializable branding-free response.
    """
    clean_script = (script or "").strip() or SAMPLE_EPIC_5_ACT_SCRIPT
    W = max(160, min(960, int(width)))
    H = max(100, min(600, int(height)))
    fps_i = max(4, min(30, int(fps)))
    fpa = max(4, min(48, int(frames_per_act)))
    tf = max(2, min(16, int(transition_frames)))
    m_max = max(3, min(8, int(max_acts)))

    storyboard_acts = decompose_long_script_to_storyboard(
        clean_script,
        max_acts=m_max,
        refine_acts=bool(refine_acts),
    )

    backend_dir = os.path.dirname(os.path.abspath(__file__))
    out_mp4 = os.path.join(backend_dir, "out_of_box_videos", "complete_script_film.mp4") if save_latest_artifacts else None
    out_gif = os.path.join(backend_dir, "out_of_box_videos", "complete_script_film.gif") if save_latest_artifacts else None
    out_sheet = os.path.join(backend_dir, "sheet_complete_script_film.png") if save_latest_artifacts else None

    film = compile_and_render_long_script_film(
        storyboard_acts=storyboard_acts,
        out_mp4_path=out_mp4,
        out_gif_path=out_gif,
        out_sheet_path=out_sheet,
        width=W,
        height=H,
        fps=fps_i,
        frames_per_act=fpa,
        transition_frames=tf,
    )

    # Ensure strictly sanitized, frontend-compatible fields (dict `latex`, top-level `python_code` & `N`)
    film["title"] = sanitize_math_text(film.get("title", ""), "Multi-Act Mathematical Cinema Film")
    film["subtitle"] = sanitize_math_text(
        film.get("subtitle", ""),
        f"{film.get('num_acts', len(storyboard_acts))}-Act Continuous Mathematical Cinema Film with C² Homotopy Transitions",
    )
    film["author"] = sanitize_math_text(
        film.get("author", ""), "Multi-Act Homotopy Mathematical Cinema Director"
    )
    film["mode"] = "script_cinema"
    film["N"] = int(film.get("height", H))

    acts = film.get("acts") or []
    for idx, act in enumerate(acts):
        act["title"] = sanitize_math_text(act.get("title", ""), f"Mathematical Act {idx + 1}")
        act["narration"] = sanitize_math_text(act.get("narration", ""), act["title"])
        act["paradigm_badge"] = sanitize_math_text(act.get("paradigm_badge", ""), "3D Raymarch Cinema")
        if "subtitle" in act:
            act["subtitle"] = sanitize_math_text(act["subtitle"], act["narration"])
        if isinstance(act.get("latex_equations"), dict):
            act["latex"] = act["latex_equations"]
        elif not isinstance(act.get("latex"), dict):
            act["latex"] = {}

    transitions = film.get("transitions") or []
    for tr in transitions:
        if "label" in tr:
            tr["label"] = sanitize_math_text(tr["label"], "C² Homotopy Morph")

    latex_eqs = film.get("latex_equations")
    if not isinstance(latex_eqs, dict) or not latex_eqs:
        latex_eqs = (acts[0].get("latex_equations") if acts else None) or {
            "Homotopy": r"d_\tau(\mathbf{p}, t) = (1 - \sigma(\tau))\,d_m(\mathbf{p}, t) + \sigma(\tau)\,d_{m+1}(\mathbf{p}, t),\quad \sigma(\tau) = 6\tau^5 - 15\tau^4 + 10\tau^3"
        }
    film["latex_equations"] = latex_eqs
    film["latex"] = latex_eqs

    first_act = acts[0] if acts else {}
    film["python_code"] = (
        first_act.get("python_code")
        or film.get("python_code")
        or (
            '"""\nMulti-Act Mathematical Cinema Film — DesignWithMath\n"""\n'
            'from ai_architect import decompose_long_script_to_storyboard\n'
            'from cinema_director_engine import SAMPLE_EPIC_5_ACT_SCRIPT, compile_and_render_long_script_film\n\n'
            'acts = decompose_long_script_to_storyboard(SAMPLE_EPIC_5_ACT_SCRIPT)\n'
            'film = compile_and_render_long_script_film(acts, out_mp4_path="complete_script_film.mp4")\n'
            'print(film["title"], film["total_frames"], "frames")\n'
        )
    )
    film["refinement_passes"] = 2 if refine_acts else 1
    return film


def generate_math_art_from_prompt(
    prompt: str,
    N: int = 2000,
    num_harmonics: int = 48,
    mode: str = "lines",
    weave_style: str = "harmonic_web",
    use_ai: bool = True,
    t_phase: float = 0.0,
    num_frames: int = 0,
    refine: bool = True,
    bypass_cache: bool = False,
) -> Dict[str, Any]:
    """
    Converts ANY user prompt or multi-act script into a genuine mathematical artwork
    or continuous multi-act mathematical film across all 8 modes:
    - "script_cinema": Multi-Act Script -> Homotopy-Morphed Complete Mathematical Film (MP4 + GIF)
    - "raymarch_cinema": 3D Implicit SDF Raymarching & Differential-Geometry PDE Cinema f(x,y,z,t) = 0
    - "lines": 2D Parametric Line Segments (X1(k), Y1(k)) -> (X2(k), Y2(k))
    - "circles": 2D Parametric Circle Families (X(k), Y(k), R(k))
    - "pixel_field": 2D Full-Color Pixel-Field Painting rgb(F(H0(x,y)), F(H1(x,y)), F(H2(x,y)))
                     with optional t_phase, WebGL1 GLSL shader, and 2-pass refinement
    - "pixel_video": 4D Full-Color Real-Time 60 FPS GPU Shader Math Video H_v(x, y, t)
    - "3d_surface": 3D Parametric Manifold (X1(k), Y1(k), Z1(k)) -> (X2(k), Y2(k), Z2(k))
    - "math_video": 4D Time-Evolving Wireframe Mathematical Video (X1(k,t), Y1(k,t)) -> (X2(k,t), Y2(k,t))
    """
    # MODE -1: Long-Script Multi-Act Mathematical Cinema Director
    if mode == "script_cinema":
        return direct_long_script_cinema_film(
            script=prompt,
            width=520,
            height=325,
            fps=12,
            frames_per_act=16,
            transition_frames=6,
            max_acts=6,
            refine_acts=bool(refine),
            save_latest_artifacts=True,
        )

    matches = resolve_prompt_to_archetypes(prompt)
    kind, target = matches[0]
    should_use_ir = use_ai or is_complex_query(prompt, matches)

    # MODE 0: Stage 1 & Stage 2 True 3D Raymarched Implicit SDF + PDE Cinema
    if mode == "raymarch_cinema":
        exact_mp = _match_exact_preset_id(prompt, MASTER_PARADIGM_PRESETS)
        if exact_mp is not None:
            return evaluate_preset(exact_mp, custom_N=N, t_phase=float(t_phase))
        return _evaluate_raymarch_cinema_response(
            prompt=prompt,
            N=N,
            t_phase=float(t_phase),
            num_frames=int(num_frames),
            refine=bool(refine),
            bypass_cache=bool(bypass_cache),
        )

    # MODE A & F: 2D Full-Color Pixel-Field Painting & 4D Full-Color GPU Math Video
    if mode in ("pixel_field", "pixel_video"):
        if mode == "pixel_field":
            exact_pf = _match_exact_preset_id(prompt, PIXEL_FIELD_PRESETS)
            if exact_pf is not None:
                return evaluate_preset(exact_pf)

        # Primary Phase 2 SOTA Direct Symbolic Math Compiler + 2-Pass Visual Refinement
        sym_res = _evaluate_math_program_pixel_response(
            prompt=prompt,
            mode=mode,
            t_phase=float(t_phase),
            num_frames=int(num_frames),
            refine=bool(refine),
            bypass_cache=bool(bypass_cache),
        )
        if sym_res is not None:
            return sym_res

        # Safe fallback to Phase 1 layered IR pixel-field renderer if offline
        ir = call_cloudflare_ai_for_ir(prompt)
        if ir is None or not ir.get("components"):
            ir = _build_fallback_ir_for_prompt(prompt, matches)

        if int(num_frames) > 1:
            return compile_animated_pixel_field_frames(ir, num_frames=int(num_frames))

        res = compile_ir_to_pixel_field(ir, t_phase=float(t_phase), include_raw_array=False)
        res.pop("rgb_array", None)
        res["mode"] = "pixel_video" if mode == "pixel_video" else "pixel_field"
        res["t_phase"] = float(t_phase)
        res["title"] = sanitize_math_text(
            res.get("title") or f"{prompt.strip().title()} — 2D Pixel-Field Painting"
        )
        res["subtitle"] = sanitize_math_text(res.get("subtitle", ""))
        res["author"] = sanitize_math_text(
            res.get("author", ""), "Hamid Naderi Yeganeh Double-Exponential Level-Set Engine"
        )
        if ir.get("components"):
            res["ir_components"] = [
                {"name": sanitize_math_text(c.get("name", "Sub-Manifold")), "type": c.get("type", "ribbon_spine"), "mirror_x": bool(c.get("mirror_x", False))}
                for c in ir.get("components", [])
            ]
        return res

    # MODE B: 3D Parametric Manifold Surface
    if mode == "3d_surface":
        exact_3d = _match_exact_preset_id(prompt, MATH_3D_PRESETS)
        if exact_3d is not None:
            return evaluate_preset(exact_3d, custom_N=N)

        if _should_use_symbolic_math_program(prompt, matches, use_ai=should_use_ir):
            try:
                pass_res = generate_math_program_with_passes(
                    prompt, refine=bool(refine), bypass_cache=bool(bypass_cache)
                )
                prog = pass_res.get("program")
                if isinstance(prog, dict) and validate_math_program(prog)[0]:
                    return compile_math_program_to_2d_3d_4d(
                        prog=prog,
                        mode="3d_surface",
                        N=N,
                        num_harmonics=max(num_harmonics, 64),
                        t_phase=float(t_phase),
                        refinement_passes=int(pass_res.get("refinement_passes", 1)),
                        visual_score=pass_res.get("visual_score"),
                    )
            except Exception as exc:
                print(f"[MathEngine] 3D iso-contour fallback triggered: {exc}")

        ir = call_cloudflare_ai_for_ir(prompt)
        if ir is not None and ir.get("components"):
            return compile_ir_to_3d_surface(
                ir=ir,
                N=N,
                num_harmonics=max(num_harmonics, 84),
                title=ir.get("title") or f"{prompt.strip().title()} — 3D Parametric Manifold",
            )
        if kind == "archetype":
            c_data = generate_parametric_contour(target, num_points=N)
            if c_data is not None:
                return compile_ir_to_3d_surface(
                    N=N,
                    num_harmonics=num_harmonics,
                    x1_raw=c_data[0],
                    y1_raw=c_data[1],
                    dual_raw=c_data[2],
                    title=f"{prompt.strip().title()} — 3D Parametric Manifold",
                )
        ir_fallback = _build_fallback_ir_for_prompt(prompt, matches)
        return compile_ir_to_3d_surface(
            ir=ir_fallback,
            N=N,
            num_harmonics=max(num_harmonics, 72),
            title=f"{prompt.strip().title()} — 3D Parametric Manifold",
        )

    # MODE C: 4D Time-Evolving Wireframe Mathematical Video
    if mode == "math_video":
        exact_vid = _match_exact_preset_id(prompt, MATH_VIDEO_PRESETS)
        if exact_vid is not None:
            return evaluate_preset(exact_vid, custom_N=N)

        if _should_use_symbolic_math_program(prompt, matches, use_ai=should_use_ir):
            try:
                pass_res = generate_math_program_with_passes(
                    prompt, refine=bool(refine), bypass_cache=bool(bypass_cache)
                )
                prog = pass_res.get("program")
                if isinstance(prog, dict) and validate_math_program(prog)[0]:
                    return compile_math_program_to_2d_3d_4d(
                        prog=prog,
                        mode="math_video",
                        N=N,
                        num_harmonics=max(num_harmonics, 64),
                        t_phase=float(t_phase),
                        refinement_passes=int(pass_res.get("refinement_passes", 1)),
                        visual_score=pass_res.get("visual_score"),
                    )
            except Exception as exc:
                print(f"[MathEngine] 4D wireframe iso-contour fallback triggered: {exc}")

        ir = call_cloudflare_ai_for_ir(prompt)
        if ir is not None and ir.get("components"):
            return compile_ir_to_math_video(
                ir=ir,
                N=N,
                num_harmonics=max(num_harmonics, 84),
                title=ir.get("title") or f"{prompt.strip().title()} — 4D Time-Harmonic Video",
            )
        if kind == "archetype":
            c_data = generate_parametric_contour(target, num_points=N)
            if c_data is not None:
                return compile_ir_to_math_video(
                    N=N,
                    num_harmonics=num_harmonics,
                    x1_raw=c_data[0],
                    y1_raw=c_data[1],
                    dual_raw=c_data[2],
                    title=f"{prompt.strip().title()} — 4D Time-Harmonic Video",
                    archetype=target,
                )
        ir_fallback = _build_fallback_ir_for_prompt(prompt, matches)
        return compile_ir_to_math_video(
            ir=ir_fallback,
            N=N,
            num_harmonics=max(num_harmonics, 72),
            title=f"{prompt.strip().title()} — 4D Time-Harmonic Video",
        )

    # MODES D & E: 2D Lines and 2D Circles
    # 1. Exact published Hamid formula shortcut ONLY when explicitly requested via weave_style="hamid_exact"
    if not use_ai and len(matches) == 1 and kind == "archetype" and weave_style == "hamid_exact":
        exact_map = {
            "bird": "bird_in_flight",
            "boat": "sailing_boat",
            "butterfly": "butterfly_wings",
            "fish": "tropical_fish",
            "lotus": "lotus_circles",
            "heart": "heart_caustic",
            "seahorse": "seahorse_spiral",
            "leaf": "maple_leaf_sciam",
            "japanese_maple": "japanese_maple_sciam",
            "oak": "oak_leaf_sciam",
            "olive": "olive_branch_sciam",
            "palm": "palm_branch_sciam",
        }
        if target in exact_map:
            return evaluate_preset(exact_map[target], custom_N=N)

    # 2. Crisp SDF Iso-Contour Extraction from Phase 2 Symbolic Math Program for unseen `lines` prompts
    if mode == "lines" and _should_use_symbolic_math_program(prompt, matches, use_ai=should_use_ir):
        try:
            pass_res = generate_math_program_with_passes(
                prompt, refine=bool(refine), bypass_cache=bool(bypass_cache)
            )
            prog = pass_res.get("program")
            if isinstance(prog, dict) and validate_math_program(prog)[0]:
                return compile_math_program_to_2d_3d_4d(
                    prog=prog,
                    mode="lines",
                    N=N,
                    num_harmonics=max(num_harmonics, 64),
                    t_phase=float(t_phase),
                    refinement_passes=int(pass_res.get("refinement_passes", 1)),
                    visual_score=pass_res.get("visual_score"),
                )
        except Exception as exc:
            print(f"[MathEngine] 2D lines iso-contour fallback triggered: {exc}")

    # 3. Multi-Tier Differential-Geometric IR Decomposition for curated/complex prompts or circles
    if should_use_ir:
        ir = call_cloudflare_ai_for_ir(prompt)
        if ir is not None and ir.get("components"):
            x1_raw, y1_raw, dual_raw = compile_hamid_ir_to_chords(ir, N=N)
            comp_names = ", ".join(c.get("name", "part") for c in ir.get("components", [])[:5])
            effective_harmonics = max(num_harmonics, 96)
            clean_title = sanitize_math_text(
                ir.get("title") or f"{prompt.strip().title()} — Harmonic Differential-Geometric Synthesis"
            )
            clean_subtitle = sanitize_math_text(
                f"Differential-Geometric Decomposition → {len(ir.get('components', []))} Coupled Parametric Manifolds ({comp_names}) → {effective_harmonics}-Harmonic Fourier Series"
            )
            return synthesize_fourier_hamid_art(
                title=clean_title,
                subtitle=clean_subtitle,
                x1_raw=x1_raw,
                y1_raw=y1_raw,
                dual_raw=dual_raw,
                N=N,
                num_harmonics=effective_harmonics,
                mode=mode,
                weave_style=weave_style,
                is_prewoven_ir=True,
                author="Fourier-Hamid Differential-Geometric Compiler",
                ir_blueprint=ir,
            )

    # 3. Multi-Archetype Composition with C^2 Zero-Length Degenerate Chord Bridges
    PREWOVEN_ARCHETYPES = {
        "bird", "boat", "fish", "butterfly", "heart", "eagle",
        "crescent_star", "olive", "palm", "peacock", "rose",
        "flower", "lotus", "star", "sun", "eye", "infinity", "spiral",
    }
    is_prewoven = False
    if len(matches) == 2 and all(m[0] == "archetype" for m in matches):
        n_bridge = 8
        n1 = (N - 2 * n_bridge) // 2
        n2 = N - 2 * n_bridge - n1
        a1_name, a2_name = matches[0][1], matches[1][1]
        c1 = generate_parametric_contour(a1_name, num_points=n1)
        c2 = generate_parametric_contour(a2_name, num_points=n2)
        if c1 is not None and c2 is not None:
            w1 = apply_weave_style(c1[0], c1[1], c1[2], mode="lines", weave_style=weave_style, is_prewoven_ir=(a1_name in PREWOVEN_ARCHETYPES))
            w2 = apply_weave_style(c2[0], c2[1], c2[2], mode="lines", weave_style=weave_style, is_prewoven_ir=(a2_name in PREWOVEN_ARCHETYPES))
            ax1, ay1, ax2, ay2 = 0.62 * w1["X1"] - 0.45, 0.62 * w1["Y1"] - 0.12, 0.62 * w1["X2"] - 0.45, 0.62 * w1["Y2"] - 0.12
            bx1, by1, bx2, by2 = 0.46 * w2["X1"] + 0.50, 0.46 * w2["Y1"] + 0.36, 0.46 * w2["X2"] + 0.50, 0.46 * w2["Y2"] + 0.36

            s = np.linspace(0.0, 1.0, n_bridge + 2)[1:-1]
            sigma = 10.0 * (s ** 3) - 15.0 * (s ** 4) + 6.0 * (s ** 5)
            m1_x, m1_y = 0.5 * (ax1[-1] + ax2[-1]), 0.5 * (ay1[-1] + ay2[-1])
            m2_x, m2_y = 0.5 * (bx1[0] + bx2[0]), 0.5 * (by1[0] + by2[0])
            br1_x = (1.0 - sigma) * m1_x + sigma * m2_x
            br1_y = (1.0 - sigma) * m1_y + sigma * m2_y

            m3_x, m3_y = 0.5 * (bx1[-1] + bx2[-1]), 0.5 * (by1[-1] + by2[-1])
            m4_x, m4_y = 0.5 * (ax1[0] + ax2[0]), 0.5 * (ay1[0] + ay2[0])
            br2_x = (1.0 - sigma) * m3_x + sigma * m4_x
            br2_y = (1.0 - sigma) * m3_y + sigma * m4_y

            x1_raw = np.concatenate([ax1, br1_x, bx1, br2_x])
            y1_raw = np.concatenate([ay1, br1_y, by1, br2_y])
            dual_raw = (
                np.concatenate([ax2, br1_x, bx2, br2_x]),
                np.concatenate([ay2, br1_y, by2, br2_y]),
            )
            is_prewoven = True
            matched_label = f"{a1_name.title()} + {a2_name.title()}"
        else:
            x1_raw, y1_raw, dual_raw = generate_text_or_custom_contour(prompt, num_points=N)
            is_prewoven = True
            matched_label = f"Custom ({prompt})"
    elif kind == "archetype":
        contour_data = generate_parametric_contour(target, num_points=N)
        if contour_data is not None:
            x1_raw, y1_raw, dual_raw = contour_data
            if target in PREWOVEN_ARCHETYPES:
                is_prewoven = True
            matched_label = target.replace("_", " ").title()
        else:
            fallback_ir = _build_fallback_ir_for_prompt(prompt, matches)
            x1_raw, y1_raw, dual_raw = compile_hamid_ir_to_chords(fallback_ir, N=N)
            is_prewoven = True
            matched_label = f"Parametric Manifold ({prompt})"
    else:
        fallback_ir = _build_fallback_ir_for_prompt(prompt, matches)
        x1_raw, y1_raw, dual_raw = compile_hamid_ir_to_chords(fallback_ir, N=N)
        is_prewoven = True
        matched_label = f"Parametric Manifold ({prompt})"

    return synthesize_fourier_hamid_art(
        title=sanitize_math_text(f"{prompt.strip().title()} — Trigonometric Synthesis"),
        subtitle=sanitize_math_text(f"Synthesized via {num_harmonics}-Harmonic Lanczos-Fourier Series ({matched_label}, N={N})"),
        x1_raw=x1_raw,
        y1_raw=y1_raw,
        dual_raw=dual_raw,
        N=N,
        num_harmonics=num_harmonics,
        mode=mode,
        weave_style=weave_style,
        is_prewoven_ir=is_prewoven,
    )


def synthesize_fourier_hamid_art(
    title: str,
    subtitle: str,
    x1_raw: np.ndarray,
    y1_raw: np.ndarray,
    dual_raw: Optional[Tuple[np.ndarray, np.ndarray]],
    N: int,
    num_harmonics: int,
    mode: str,
    weave_style: str,
    is_prewoven_ir: bool = False,
    author: str = "Fourier-Hamid Differential-Geometric Engine",
    ir_blueprint: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    clean_title = sanitize_math_text(title)
    clean_subtitle = sanitize_math_text(subtitle)
    clean_author = sanitize_math_text(author, "Fourier-Hamid Differential-Geometric Engine")

    woven = apply_weave_style(
        x1_raw,
        y1_raw,
        dual_raw,
        mode=mode,
        weave_style=weave_style,
        is_prewoven_ir=is_prewoven_ir,
    )
    k = np.arange(1, N + 1, dtype=np.float64)

    latex_dict: Dict[str, str] = {}
    py_formulas: Dict[str, str] = {}
    harmonics_meta: Dict[str, Any] = {}
    evaluated: Dict[str, np.ndarray] = {}

    for var_name, signal in woven.items():
        a0, harms = fit_trigonometric_series(signal, num_harmonics=num_harmonics)
        latex_dict[var_name] = format_series_to_latex(var_name, a0, harms, N)
        py_formulas[var_name] = format_series_to_python_expr(a0, harms)
        harmonics_meta[var_name] = {"a0": a0, "terms": harms}
        evaluated[var_name] = evaluate_trigonometric_series(a0, harms, k, N)

    if mode == "lines":
        primitives = np.column_stack([
            evaluated["X1"],
            evaluated["Y1"],
            evaluated["X2"],
            evaluated["Y2"],
        ]).round(4).tolist()
    else:
        primitives = np.column_stack([
            evaluated["X"],
            evaluated["Y"],
            np.maximum(0.003, np.abs(evaluated["R"])),
        ]).round(4).tolist()

    res = {
        "title": clean_title,
        "subtitle": clean_subtitle,
        "author": clean_author,
        "mode": mode,
        "N": N,
        "harmonics_count": num_harmonics,
        "latex": latex_dict,
        "formulas": py_formulas,
        "harmonics": harmonics_meta,
        "primitives": primitives,
        "python_code": build_python_export_script(clean_title, mode, N, py_formulas),
    }
    if ir_blueprint is not None:
        res["ir_components"] = [
            {"name": sanitize_math_text(c.get("name", "Sub-Manifold")), "type": c.get("type"), "mirror_x": c.get("mirror_x", False)}
            for c in ir_blueprint.get("components", [])
        ]
    return res


def evaluate_custom_formulas(
    mode: str,
    N: int,
    formulas: Dict[str, str],
) -> Dict[str, Any]:
    N = max(50, min(6000, int(N)))
    k = np.arange(1, N + 1, dtype=np.float64)
    safe_env = {
        "np": np,
        "sin": np.sin,
        "cos": np.cos,
        "tan": np.tan,
        "exp": np.exp,
        "sqrt": np.sqrt,
        "abs": np.abs,
        "pi": np.pi,
        "k": k,
        "N": float(N),
        "t": 0.0,
    }

    for key, expr in formulas.items():
        if re.search(r"(__|\b(import|exec|eval|open|os|sys)\b)", str(expr)):
            raise ValueError(f"Disallowed token in formula {key}")

    if mode == "pixel_field":
        return evaluate_preset("pf_peacock")

    if mode == "raymarch_cinema":
        return evaluate_preset("mp_iq_mandelbulb", custom_N=N)

    if mode == "3d_surface":
        f_3d = {
            "X1": formulas.get("X1", "np.sin(2*np.pi*k/N)"),
            "Y1": formulas.get("Y1", "np.cos(2*np.pi*k/N)"),
            "Z1": formulas.get("Z1", "0.5*np.sin(6*np.pi*k/N)"),
            "X2": formulas.get("X2", "0.5*np.cos(4*np.pi*k/N)"),
            "Y2": formulas.get("Y2", "0.5*np.sin(4*np.pi*k/N)"),
            "Z2": formulas.get("Z2", "-0.5*np.cos(6*np.pi*k/N)"),
        }
        x1 = np.broadcast_to(eval(f_3d["X1"], {"__builtins__": {}}, safe_env), (N,)).astype(np.float64)
        y1 = np.broadcast_to(eval(f_3d["Y1"], {"__builtins__": {}}, safe_env), (N,)).astype(np.float64)
        z1 = np.broadcast_to(eval(f_3d["Z1"], {"__builtins__": {}}, safe_env), (N,)).astype(np.float64)
        x2 = np.broadcast_to(eval(f_3d["X2"], {"__builtins__": {}}, safe_env), (N,)).astype(np.float64)
        y2 = np.broadcast_to(eval(f_3d["Y2"], {"__builtins__": {}}, safe_env), (N,)).astype(np.float64)
        z2 = np.broadcast_to(eval(f_3d["Z2"], {"__builtins__": {}}, safe_env), (N,)).astype(np.float64)
        max_abs = max(
            float(np.max(np.abs(x1))), float(np.max(np.abs(y1))), float(np.max(np.abs(z1))),
            float(np.max(np.abs(x2))), float(np.max(np.abs(y2))), float(np.max(np.abs(z2))), 1e-5,
        )
        x1, y1, z1 = x1 / max_abs, y1 / max_abs, z1 / max_abs
        x2, y2, z2 = x2 / max_abs, y2 / max_abs, z2 / max_abs
        primitives_3d = np.column_stack([x1, y1, z1, x2, y2, z2]).round(4).tolist()
        primitives = np.column_stack([x1, y1, x2, y2]).round(4).tolist()
        latex = {k_var: f"{k_var[0]}_{k_var[1]}(k) = {expr}" for k_var, expr in f_3d.items()}
        return {
            "title": "Custom 3D Parametric Manifold",
            "subtitle": f"Evaluated directly from custom 3D spatial formulas for k = 1..{N}",
            "author": "Custom Formula Studio",
            "mode": "3d_surface",
            "N": N,
            "harmonics_count": "Custom 3D Exact",
            "latex": latex,
            "formulas": f_3d,
            "primitives_3d": primitives_3d,
            "primitives": primitives,
            "python_code": build_3d_python_export_script("Custom 3D Manifold", N, f_3d),
        }

    if mode == "math_video":
        f_vid = {
            "X1": formulas.get("X1", "1.5*(np.sin(2*np.pi*k/N + np.pi/3)**7)"),
            "Y1": formulas.get("Y1", "0.25*(np.cos(6*np.pi*k/N)**2)"),
            "X2": formulas.get("X2", "0.2*np.sin(6*np.pi*k/N + np.pi/5)"),
            "Y2": formulas.get("Y2", "-(2.0/3.0)*(np.sin(4*np.pi*k/N - np.pi/3)**2)"),
        }
        x1 = np.broadcast_to(eval(f_vid["X1"], {"__builtins__": {}}, safe_env), (N,)).astype(np.float64)
        y1 = np.broadcast_to(eval(f_vid["Y1"], {"__builtins__": {}}, safe_env), (N,)).astype(np.float64)
        x2 = np.broadcast_to(eval(f_vid["X2"], {"__builtins__": {}}, safe_env), (N,)).astype(np.float64)
        y2 = np.broadcast_to(eval(f_vid["Y2"], {"__builtins__": {}}, safe_env), (N,)).astype(np.float64)
        max_abs = max(float(np.max(np.abs(x1))), float(np.max(np.abs(y1))), float(np.max(np.abs(x2))), float(np.max(np.abs(y2))), 1e-5)
        x1, y1, x2, y2 = x1 / max_abs, y1 / max_abs, x2 / max_abs, y2 / max_abs
        primitives = np.column_stack([x1, y1, x2, y2]).round(4).tolist()
        ax1 = -0.08 * x1
        bx1 = 0.04 * np.sin(2.0 * np.pi * k / N)
        ay1 = 0.22 * np.abs(x1)
        by1 = 0.05 * np.cos(4.0 * np.pi * k / N)
        ax2 = -0.06 * x2
        bx2 = 0.03 * np.cos(2.0 * np.pi * k / N)
        ay2 = -0.10 * np.abs(y2)
        by2 = 0.04 * np.sin(4.0 * np.pi * k / N)
        time_harmonics = np.column_stack([ax1, bx1, ay1, by1, ax2, bx2, ay2, by2]).round(4).tolist()
        latex = {
            "X1": f"X_1(k, t) = ({f_vid['X1']}) + A_1(k)\\sin(t) + B_1(k)\\cos(t)",
            "Y1": f"Y_1(k, t) = ({f_vid['Y1']}) + C_1(k)\\sin(t) + D_1(k)\\cos(t)",
            "X2": f"X_2(k, t) = ({f_vid['X2']}) + A_2(k)\\sin(t) + B_2(k)\\cos(t)",
            "Y2": f"Y_2(k, t) = ({f_vid['Y2']}) + C_2(k)\\sin(t) + D_2(k)\\cos(t)",
        }
        return {
            "title": "Custom 4D Time-Harmonic Equations",
            "subtitle": f"Evaluated from custom 4D space-time formulas for k = 1..{N}",
            "author": "Custom Formula Studio",
            "mode": "math_video",
            "N": N,
            "harmonics_count": "Custom 4D Exact",
            "latex": latex,
            "formulas": f_vid,
            "primitives": primitives,
            "time_harmonics": time_harmonics,
            "python_code": build_video_python_export_script("Custom 4D Video", N, f_vid),
        }

    if mode == "lines":
        x1 = np.broadcast_to(eval(formulas["X1"], {"__builtins__": {}}, safe_env), (N,)).astype(np.float64)
        y1 = np.broadcast_to(eval(formulas["Y1"], {"__builtins__": {}}, safe_env), (N,)).astype(np.float64)
        x2 = np.broadcast_to(eval(formulas["X2"], {"__builtins__": {}}, safe_env), (N,)).astype(np.float64)
        y2 = np.broadcast_to(eval(formulas["Y2"], {"__builtins__": {}}, safe_env), (N,)).astype(np.float64)
        max_abs = max(float(np.max(np.abs(x1))), float(np.max(np.abs(y1))), float(np.max(np.abs(x2))), float(np.max(np.abs(y2))), 1e-5)
        primitives = np.column_stack([x1 / max_abs, y1 / max_abs, x2 / max_abs, y2 / max_abs]).round(4).tolist()
        latex = {
            "X1": f"X_1(k) = {formulas['X1']}",
            "Y1": f"Y_1(k) = {formulas['Y1']}",
            "X2": f"X_2(k) = {formulas['X2']}",
            "Y2": f"Y_2(k) = {formulas['Y2']}",
        }
    else:
        x = np.broadcast_to(eval(formulas["X"], {"__builtins__": {}}, safe_env), (N,)).astype(np.float64)
        y = np.broadcast_to(eval(formulas["Y"], {"__builtins__": {}}, safe_env), (N,)).astype(np.float64)
        r = np.abs(np.broadcast_to(eval(formulas["R"], {"__builtins__": {}}, safe_env), (N,)).astype(np.float64))
        max_abs = max(float(np.max(np.abs(x) + r)), float(np.max(np.abs(y) + r)), 1e-5)
        primitives = np.column_stack([x / max_abs, y / max_abs, r / max_abs]).round(4).tolist()
        latex = {
            "X": f"X(k) = {formulas['X']}",
            "Y": f"Y(k) = {formulas['Y']}",
            "R": f"R(k) = {formulas['R']}",
        }

    return {
        "title": "Custom Parametric Equations",
        "subtitle": f"Evaluated directly from custom trigonometric formulas for k = 1..{N}",
        "author": "Custom Formula Studio",
        "mode": mode,
        "N": N,
        "harmonics_count": "Custom Exact",
        "latex": latex,
        "formulas": formulas,
        "primitives": primitives,
        "python_code": build_python_export_script("Custom Formula", mode, N, formulas),
    }
