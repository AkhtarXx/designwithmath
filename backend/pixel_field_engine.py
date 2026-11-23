import ast
import base64
import io
import math
import re
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from PIL import Image

try:
    from ai_architect import safe_eval_sdf_expr as _ai_safe_eval_sdf_expr
except Exception:
    _ai_safe_eval_sdf_expr = None


def _ee(z: np.ndarray) -> np.ndarray:
    """Analytic double-exponential e^(-e^z) with overflow protection."""
    return np.exp(-np.exp(np.clip(z, -45.0, 45.0)))


def hamid_F_quantize(h: np.ndarray) -> np.ndarray:
    """
    Exact Hamid Naderi Yeganeh pixel quantizer:
        F(x) = floor( 255 * exp(-exp(-1000*x)) * |x|^(exp(-exp(1000*(x - 1)))) )
    """
    step_0 = _ee(-1000.0 * h)
    exp_1 = _ee(1000.0 * (h - 1.0))
    val = 255.0 * step_0 * (np.maximum(np.abs(h), 1e-8) ** exp_1)
    return np.floor(np.clip(val, 0.0, 255.0)).astype(np.uint8)


def encode_rgb_to_base64_png(rgb_arr: np.ndarray) -> str:
    pil_img = Image.fromarray(rgb_arr, mode="RGB")
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


# ============================================================================
# 1. AUTHENTIC HAMID NADERI YEGANEH 2D PIXEL-FIELD MASTERPIECES
# ============================================================================

PIXEL_FIELD_PRESETS: List[Dict[str, Any]] = [
    {
        "id": "pf_peacock",
        "name": "Peacock in Full Feather Display (2D Pixel-Field)",
        "author": "Hamid Naderi Yeganeh",
        "category": "Pixel-Field Paintings",
        "mode": "pixel_field",
        "aspect": (2000, 1100),
        "N": 1100,
        "description": "5 concentric rings of 20 iridescent train feathers (100 ocelli eye-spots W_{u,r} and herringbone barb fields A_{u,r}), royal cobalt body Q(x,y), and golden bokeh K(x,y).",
        "latex": {
            "RGB": r"\operatorname{rgb}\!\left(F\!\left(H_0\!\left(\frac{m-1000}{1300},\frac{1021-n}{1300}\right)\right),\,F\!\left(H_1\!\left(\frac{m-1000}{1300},\frac{1021-n}{1300}\right)\right),\,F\!\left(H_2\!\left(\frac{m-1000}{1300},\frac{1021-n}{1300}\right)\right)\right),\quad F(x)=\left\lfloor 255e^{-e^{-1000x}}|x|^{e^{-e^{1000(x-1)}}}\right\rfloor",
            "H_s": r"H_s(x,y) = \left(\frac{3-s}{10}(1+K(x,y))\frac{y+1}{2}\left(1-E(x,y)e^{-e^{-500(y+\frac{3}{50}-\frac{3}{25}|x|)}}\right)\prod_{u=1}^{5}\prod_{r=1}^{20}(1-J_{u,r}(x,y)) + e^{-e^{-500(y+\frac{3}{50}-\frac{3}{25}|x|)}}\big(U_s(1-E)+L_s E\big)\right)\left(1-e^{-e^{-1000(x-\frac{y}{5}+\frac{1}{10})-e^{50Q(x,y)}}}\right) + \frac{1-3y}{5^{2-s}}e^{-e^{-100(x-\frac{y}{5}+\frac{1}{10})-e^{50Q}-e^{-50Q}}}",
            "W_ur": r"W_{u,r}(x,y) = \left(R_{u,r}(x,y)+\frac{9}{10}\left(\frac{2u}{15}+\frac{(-1)^r}{30}\right)\right)^2 + \frac{7}{5}(P_{u,r}(x,y))^2,\quad T_{u,r}(x,y)=\left(\frac{2}{25}-\frac{x^2+y^2}{15}\right)\cos(u^2)+\frac{\pi r}{20}",
            "A_ur": r"A_{u,r}(x,y) = e^{-e^{-1000\cos\left(r+u+6\cos(18R_{u,r}+ur+14P_{u,r})+600R_{u,r}+600|P_{u,r}|^{1-2(\frac{7u}{30})^4(N_{u,r})^4}-4800(P_{u,r})^2\right)+400-700(\frac{7}{10}N_{u,r})^4}}",
        },
    },
    {
        "id": "pf_autumn_trees",
        "name": "Autumn Trees Forest Avenue (2D Pixel-Field)",
        "author": "Hamid Naderi Yeganeh",
        "category": "Pixel-Field Paintings",
        "mode": "pixel_field",
        "aspect": (2000, 1200),
        "N": 1200,
        "description": "8 perspective-receding rows of branching tree trunks O_s(x,y), 30 occluded layers of crimson/gold autumn foliage W_v(x,y), and leaf-strewn forest path I(x,y).",
        "latex": {
            "RGB": r"\operatorname{rgb}\!\left(F\!\left(H_0\!\left(\frac{m-1000}{600},\frac{1001-n}{600}\right)\right),\,F\!\left(H_1\!\left(\frac{m-1000}{600},\frac{1001-n}{600}\right)\right),\,F\!\left(H_2\!\left(\frac{m-1000}{600},\frac{1001-n}{600}\right)\right)\right),\quad F(x)=\left\lfloor 255e^{-e^{-1000x}}|x|^{e^{-e^{1000(x-1)}}}\right\rfloor",
            "H_v": r"H_v(x,y) = \frac{3-v}{20} + \frac{1}{30}(18+6v^2-6v+5y)(1-G(x,y))Z_8(x,y) + \frac{1}{400}W_v(x,y)\left(460C(x,y)+(13-5(v-1)^2)G(x,y)Z_8(x,y)(3+20I(x,y))\right)",
            "W_v": r"W_v(x,y) = \sum_{s=1}^{30}\left(\prod_{u=0}^{s-1}1-e^{-e^{-100(u-\frac{1}{2})}}J_u(x,y)\right)J_s(x,y)\left(\frac{7-3v}{5}+\frac{3+10v^2-20v}{10}\cos^2(3(K(x,y))^2)+\frac{7-4v}{10}\sin^4(4(K(x,y))^2)+\frac{1+G(x,y)}{4}\cos((4+v)s^2+(K(x,y))^2)\right)",
            "E": r"E(x,y) = \sum_{s=1}^{25}\frac{5}{4}\left(\frac{9}{10}\right)^s D_s(x,y),\quad G(x,y)=e^{-e^{500\left(y-\frac{1}{50}+\frac{E(x,y)}{300}\right)}},\quad I(x,y)=e^{-e^{-50y+2-30|x|-\frac{1}{4}E(x,y)}}",
        },
    },
    {
        "id": "pf_kiwi",
        "name": "Kiwi Bird in Forest Bokeh (2D Pixel-Field)",
        "author": "Hamid Naderi Yeganeh",
        "category": "Pixel-Field Paintings",
        "mode": "pixel_field",
        "aspect": (2000, 1060),
        "N": 1060,
        "description": "100 occluded shaggy brown feather-hair tufts J_r(x,y), ivory beak A(x,y), legs B(x,y), and 40-harmonic forest bokeh background K_s(x,y).",
        "latex": {
            "RGB": r"\operatorname{rgb}\!\left(F\!\left(H_0\!\left(\frac{m-1300}{540},\frac{531-n}{540}\right)\right),\,F\!\left(H_1\!\left(\frac{m-1300}{540},\frac{531-n}{540}\right)\right),\,F\!\left(H_2\!\left(\frac{m-1300}{540},\frac{531-n}{540}\right)\right)\right),\quad F(x)=\left\lfloor 255e^{-e^{-1000x}}|x|^{e^{-e^{1000(x-1)}}}\right\rfloor",
            "H_s": r"H_s(x,y) = \left(\frac{16-2s^2+s}{20}+4y-\frac{36}{25}+\frac{2x^2}{5}-e^{-e^{1000\left(\left|y-\frac{9}{25}+\frac{x^2}{10}\right|-\frac{1}{2000}\right)}}\right)A(x,y)Z(x,y) + Z(x,y)\frac{B(x,y)}{10}\left(\frac{3}{2}-\frac{4s^2+5s}{30}\right)(\cdots) + \left(\frac{2-s}{10}+\left(\frac{x}{5}+\frac{53}{50}\right)K_s(x,y)e^{-\cdots}\right)(1-A-B)Z + L_s(x,y)e^{-e^{-2000\left((x+1)^2+\frac{64}{1000}\left|y-\frac{1}{4}\right|-\frac{1}{500}\right)}}",
            "J_r": r"J_r(x,y) = e^{-e^{-1000(r-\frac{1}{2})}-e^{-5\left(\sin(U_r(x,y))-\frac{2}{5}-\frac{2}{5\pi}\arctan\left(\frac{2}{5}(r-15)\right)-4\left(x+\frac{60-r}{40}\right)^2\right)}-e^{-100\left(x+\frac{60-r}{40}\right)}}\left(1-e^{-e^{-200(\cdots)}-e^{-40(\cdots)}}\right)",
            "K_s": r"K_s(x,y) = \sum_{r=1}^{40}\frac{9}{54-30(s-1)+32(s^2-s)}\left(\frac{3}{5}+\frac{\sin(2r+4s)}{5}+\frac{\sin((2+5s)r)}{5}\right)e^{-e^{-15\left(\sin^{10}(\cdots)\sin^{10}(\cdots)-\frac{1}{10}\right)}}",
        },
    },
    {
        "id": "pf_strawberries",
        "name": "Fresh Strawberries (2D Pixel-Field)",
        "author": "Hamid Naderi Yeganeh",
        "category": "Pixel-Field Paintings",
        "mode": "pixel_field",
        "aspect": (2000, 1200),
        "N": 1200,
        "description": "30 front-to-back occluded strawberries with golden achenes (seeds), glossy dimple specular highlights, and green leafy calyxes.",
        "latex": {
            "RGB": r"\operatorname{rgb}\!\left(F\!\left(H_0\!\left(\frac{m-1000}{600},\frac{601-n}{600}\right)\right),\,F\!\left(H_1\!\left(\frac{m-1000}{600},\frac{601-n}{600}\right)\right),\,F\!\left(H_2\!\left(\frac{m-1000}{600},\frac{601-n}{600}\right)\right)\right),\quad F(x)=\left\lfloor 255e^{-e^{-1000x}}|x|^{e^{-e^{1000(x-1)}}}\right\rfloor",
            "H_v": r"H_v(x,y) = \sum_{s=1}^{30}\left(\prod_{r=0}^{s-1}(1-A_{1000,r}(x,y))\left(1-e^{-e^{-1000(r-\frac{1}{2})}}U_r(x,y)\right)\left(1-\frac{5}{4}A_{4,r}(x,y)\right)\right)\left(\frac{5-3(v-1)^2+W(x,y)}{10}e^{-e^{-\left|\frac{71}{10}-10P_s(x,y)\right|}}U_s(x,y) + A_{1000,s}(x,y)(1-U_s(x,y))L_{v,s}(x,y)\right)",
            "L_vs": r"L_{v,s}(x,y) = \left(\frac{1}{10}-\frac{1}{40}\cos(20\arccos(R_{0,s}(x,y)))\cos(25P_s(x,y))\right)\left(4v^2-13v+11+\cos(7s+vs)+20e^{-e^{-70(C_{20,s}-\frac{1}{2})}}+20e^{-e^{-10(C_{10,s}-\frac{1}{2})}}\right)A_{4,s}A_{1000,s}+B_s(x,y)",
            "P_s": r"P_s(x,y)=\arctan\!\left(\tan\!\left(2\sin(5s)x-2\cos(5s)y+3\cos(5s)+\frac{3\cos(14x-19y+5s)}{200}\right)\right),\quad Q_s(x,y)=\arctan\!\left(\tan\!\left(2(\cos(5s)x+\sin(5s)y+2\cos(4s))+\frac{3\cos(18x+15y+4s)}{200}\right)\right)",
        },
    },
    {
        "id": "pf_corals",
        "name": "Corals & Reef Fish (2D Pixel-Field)",
        "author": "Hamid Naderi Yeganeh",
        "category": "Pixel-Field Paintings",
        "mode": "pixel_field",
        "aspect": (2000, 1200),
        "N": 1200,
        "description": "150 front-to-back occluded magenta sea-anemone tentacles J_s(x,y) with a striped golden reef fish K(x,y) swimming in deep turquoise water.",
        "latex": {
            "RGB": r"\operatorname{rgb}\!\left(F\!\left(H_0\!\left(\frac{m-1000}{400},\frac{801-n}{400}\right)\right),\,F\!\left(H_1\!\left(\frac{m-1000}{400},\frac{801-n}{400}\right)\right),\,F\!\left(H_2\!\left(\frac{m-1000}{400},\frac{801-n}{400}\right)\right)\right),\quad F(x)=\left\lfloor 255e^{-e^{-1000x}}|x|^{e^{-e^{1000(x-1)}}}\right\rfloor",
            "H_s": r"H_s(x,y) = \left(1+(-1)^s\frac{s}{2}\right)U_s(x,y) + \left(\frac{1}{10}+\frac{s}{4}+\left(y-\frac{x^2}{3}-\frac{1}{2}\right)\left(\frac{1}{25}+\frac{17}{300}(s^2-s)\right)\right)\prod_{r=1}^{150}(1-J_r(x,y)) + \left(\frac{5}{2}-\frac{7}{4}s\right)K(x,y) - e^{-e^{100\left(\sqrt{(x-\frac{41}{50})^2+(y-\frac{3}{5})^2}-\frac{1}{100}\right)}}",
            "J_s": r"J_s(x,y) = e^{-e^{50\left(30|P_s(x,y)|^3+(Q_s(x,y))^4-\frac{3}{5}\right)}},\quad P_s(x,y)=\sin\!\left(5\!\left(x+\frac{1}{5}(\sin(7s^2)+\sin(x+s^2)-3)y+\sin(s^2)-\frac{2}{25}\sin(5y+\sin(4s^2))\right)\right)",
            "K": r"K(x,y) = \left(\frac{1}{2}+2\left(y-\frac{3}{5}\right)\right)\left(1-\frac{3}{10}\sin^8\!\left(120\left(y-\frac{3}{5}\right)e^{10(x-1)^2}\right)\right)e^{-e^{\left(170-160e^{-500(x-\frac{19}{20})^4}\right)\left(\frac{3}{5}(x-1)^2+14\left(\frac{100}{100-99e^{-10(x-\frac{23}{20})^2}}\left|y-\frac{3}{5}\right|\right)^{2+3(x-\frac{3}{5})}-\frac{1}{20}\right)}}",
        },
    },
    {
        "id": "pf_bipolar_nebula",
        "name": "Bipolar Planetary Nebula (2D Pixel-Field)",
        "author": "Hamid Naderi Yeganeh",
        "category": "Pixel-Field Paintings",
        "mode": "pixel_field",
        "aspect": (2000, 1200),
        "N": 1200,
        "description": "27 nested hourglass nebula shells L_s(x,y), 50-octave domain-warped ionized dust filaments K_v(x,y), central protostar W(x,y), and starfield T_v(x,y).",
        "latex": {
            "RGB": r"\operatorname{rgb}\!\left(F\!\left(H_0\!\left(\frac{m-1000}{420},\frac{601-n}{420}\right)\right),\,F\!\left(H_1\!\left(\frac{m-1000}{420},\frac{601-n}{420}\right)\right),\,F\!\left(H_2\!\left(\frac{m-1000}{420},\frac{601-n}{420}\right)\right)\right),\quad F(x)=\left\lfloor 255e^{-e^{-1000x}}|x|^{e^{-e^{1000(x-1)}}}\right\rfloor",
            "H_v": r"H_v(x,y) = \frac{11}{10}(1-W(x,y))K_v(x,y)A(x,y) + \frac{4+v^2-v}{2}W(x,y) + T_v(x,y)",
            "L_s": r"L_s(x,y) = \sqrt{(U_s(x,y))^2 + \left(2(R(s))^{\frac{3}{10}}|U_s(x,y)|^{-\frac{3}{10}}\left(y-\left(\frac{3}{20}+\frac{\cos(4s^2)}{5}\right)x\right)\right)^2} - R(s),\quad J_s(x,y)=e^{-e^{25-50s-e^{10L_s(x,y)}}}",
            "K_v": r"K_v(x,y) = \sum_{s=1}^{50}I_s(x,y)\left(\frac{19}{20}\right)^s\frac{12-4v+v^2+(v-1)\cos(2s^2)+8\cos((7+v)s^2)}{50}",
        },
    },
    {
        "id": "pf_moon_footprint",
        "name": "Astronaut's Footprint on the Moon (2D Pixel-Field)",
        "author": "Hamid Naderi Yeganeh",
        "category": "Pixel-Field Paintings",
        "mode": "pixel_field",
        "aspect": (2000, 1200),
        "N": 1200,
        "description": "Apollo lunar bootprint rendered via 30-octave trigonometric regolith fractal C_v(x,y), super-elliptical tread depression B_v(x,y), and directional self-shadowing.",
        "latex": {
            "RGB": r"\operatorname{rgb}\!\left(F\!\left(H_0\!\left(\frac{m-1000}{600},\frac{601-n}{600}\right)\right),\,F\!\left(H_1\!\left(\frac{m-1000}{600},\frac{601-n}{600}\right)\right),\,F\!\left(H_2\!\left(\frac{m-1000}{600},\frac{601-n}{600}\right)\right)\right),\quad F(x)=\left\lfloor 255e^{-e^{-1000x}}|x|^{e^{-e^{1000(x-1)}}}\right\rfloor",
            "H_v": r"H_v(x,y) = \left(\frac{3}{2}-\frac{v}{10}\right)e^{-e^{8(A_{20}(x,y)-A_0(x,y))}}\left(1+\frac{1}{4}e^{-e^{20(A_5(x,y)-A_0(x,y))}}\right),\quad A_v(x,y)=B_v(x,y)C_v(x,y)",
            "B_v": r"B_v(x,y) = \frac{1}{20} + e^{-e^{50\left(\left(U_v(x)+\frac{V_v(y)}{10}\right)^2+\left(10-4\left(U_v(x)+\frac{V_v(y)}{10}\right)\right)\left|V_v(y)-\frac{U_v(x)}{10}\right|^3-\frac{4}{5}\right)-e^{2\sin\left(30\left(U_v(x)+\frac{V_v(y)}{10}\right)\right)}}} + \frac{5}{4}e^{-e^{-15\left(\left(U_v(x)+\frac{V_v(y)}{10}\right)^2+\left(10-4\left(U_v(x)+\frac{V_v(y)}{10}\right)\right)\left|V_v(y)-\frac{U_v(x)}{10}\right|^3-1\right)}}",
            "C_v": r"C_v(x,y) = \sum_{s=1}^{30}\left(\frac{87}{100}\right)^s e^{-e^{-10\left(\cos\left(\left(\frac{6}{5}\right)^s(\cos(4s^2)U_v(x)+\sin(4s^2)V_v(y))+2\cos(2s^2)\right)\cos\left(\left(\frac{6}{5}\right)^s(\cos(4s^2)V_v(y)-\sin(4s^2)U_v(x))+2\cos(3s^2)\right)-\frac{3}{4}\right)}}",
        },
    },
]


def evaluate_pixel_field_preset(
    preset_id: str,
    width: int = 580,
    include_raw_array: bool = False,
) -> Dict[str, Any]:
    """
    Evaluates one of Hamid Naderi Yeganeh's 7 iconic 2D Pixel-Field paintings
    directly from its exact mathematical equations in vectorized float32 NumPy.
    Uses e^(-e^A - e^B) = _ee(A) * _ee(B) exact first-exponent product rule.
    """
    preset = next((p for p in PIXEL_FIELD_PRESETS if p["id"] == preset_id), PIXEL_FIELD_PRESETS[0])
    w_ref, h_ref = preset["aspect"]
    W = int(width)
    H = int(round(W * h_ref / w_ref))

    m = np.linspace(1.0, float(w_ref), W, dtype=np.float32)
    n = np.linspace(1.0, float(h_ref), H, dtype=np.float32)
    M_grid, N_grid = np.meshgrid(m, n)

    if preset_id == "pf_moon_footprint":
        x = (M_grid - 1000.0) / 600.0
        y = (601.0 - N_grid) / 600.0

        def eval_A(v_val: float) -> np.ndarray:
            uv = x + v_val / 500.0
            vv = y + v_val / 1000.0
            xi = uv + vv / 10.0
            eta = vv - uv / 10.0
            boot_core = (xi ** 2) + (10.0 - 4.0 * xi) * (np.abs(eta) ** 3)
            # Exact product: e^(-e^(50*(boot_core - 4/5)) - e^(2*sin(30*xi))) = _ee(50*(...)) * _ee(2*sin(30*xi))
            Bv = (
                0.05
                + _ee(50.0 * (boot_core - 0.8)) * _ee(2.0 * np.sin(30.0 * xi))
                + 1.25 * _ee(-15.0 * (boot_core - 1.0))
            )
            Cv = np.zeros_like(x)
            for s in range(1, 31):
                s2 = float(s * s)
                freq = 1.2 ** s
                cs, sn = math.cos(4.0 * s2), math.sin(4.0 * s2)
                c1 = np.cos(freq * (cs * uv + sn * vv) + 2.0 * math.cos(2.0 * s2))
                c2 = np.cos(freq * (cs * vv - sn * uv) + 2.0 * math.cos(3.0 * s2))
                Cv += (0.87 ** s) * _ee(-10.0 * (c1 * c2 - 0.75))
            return Bv * Cv

        A0 = eval_A(0.0)
        A5 = eval_A(5.0)
        A20 = eval_A(20.0)
        base_light = _ee(8.0 * (A20 - A0)) * (1.0 + 0.25 * _ee(20.0 * (A5 - A0)))
        H_channels = [(1.5 - 0.1 * v) * base_light for v in (0.0, 1.0, 2.0)]

    elif preset_id == "pf_corals":
        x = (M_grid - 1000.0) / 400.0
        y = (801.0 - N_grid) / 400.0
        trans = np.ones_like(x)
        U0 = np.zeros_like(x)
        U1 = np.zeros_like(x)
        U2 = np.zeros_like(x)

        for r in range(1, 151):
            r2 = float(r * r)
            r3 = float(r * r * r)
            Pr = np.sin(
                5.0 * (
                    x
                    + 0.2 * (math.sin(7.0 * r2) + np.sin(x + r2) - 3.0) * y
                    + math.sin(r2)
                    - 0.08 * np.sin(5.0 * y + math.sin(4.0 * r2))
                )
            )
            Qr = (
                y
                + 0.6 * x
                + ((x + 1.0) ** 2) / 10.0
                - 1.8 * (1.0 - ((1.0 - r / 150.0) ** 1.6))
                + 1.7
                + 0.3 * math.sin(r3)
            )
            p3_q4 = 30.0 * (np.abs(Pr) ** 3) + (Qr ** 4) - 0.6
            Jr = _ee(50.0 * p3_q4)
            vis = (1.0 + Pr) * Jr * trans
            # Exact printed U_s(x,y) formula: 4 * e^(-e^(100*...) - e^(-5*(...)) - e^(-1000*Qr))
            rim = 4.0 * (
                _ee(100.0 * p3_q4)
                * _ee(-5.0 * (20.0 * (np.abs(Pr) ** 3) + ((0.02 + Qr) ** 4) - 0.6))
                * _ee(-1000.0 * Qr)
            )
            body_base = 0.3 * Qr + _ee(p3_q4)
            U0 += vis * (rim + (1.20 + 0.20 * math.sin(5.0 * r)) * body_base)
            U1 += vis * (rim + (1.05 + 0.20 * math.sin(10.0 * r)) * body_base)
            U2 += vis * (rim + (1.00 + 0.20 * math.sin(15.0 * r)) * body_base)
            trans *= (1.0 - Jr)

        ym = y - 0.6
        fish_env = (170.0 - 160.0 * np.exp(-500.0 * ((x - 0.95) ** 4))) * (
            0.6 * ((x - 1.0) ** 2)
            + 14.0 * ((100.0 / (100.0 - 95.5 * np.exp(-10.0 * ((x - 1.15) ** 2)))) * np.abs(ym)) ** np.clip(2.0 + 3.0 * (x - 0.6), 0.5, 8.0)
            - 0.05
        )
        K_xy = (0.5 + 2.0 * ym) * (1.0 - 0.3 * (np.sin(120.0 * ym * np.exp(np.clip(10.0 * ((x - 1.0) ** 2), -20.0, 20.0))) ** 8)) * _ee(fish_env)
        eye = _ee(100.0 * (np.sqrt((x - 0.82) ** 2 + ym ** 2) - 0.01))
        U_list = [U0, U1, U2]
        H_channels = []
        for s in range(3):
            sf = float(s)
            bg = (0.1 + 0.25 * sf + (y - (x ** 2) / 3.0 - 0.5) * (0.04 + (17.0 / 300.0) * (sf * sf - sf))) * trans
            H_s = (1.0 + ((-1.0) ** s) * (sf / 2.0)) * U_list[s] + bg + (2.5 - 1.75 * sf) * K_xy - eye
            H_channels.append(H_s)

    elif preset_id == "pf_kiwi":
        x = (M_grid - 1300.0) / 540.0
        y = (531.0 - N_grid) / 540.0
        # Exact printed Beak A(x,y) from Kiwi formula plate
        A_xy = (
            _ee(100.0 * (np.abs(y - 0.36 + (x ** 2) / 10.0) - 0.05 * (np.abs(x + 2.1) ** 0.3) - 0.14 * ((x + 2.1) ** 4)))
            * _ee(20.0 * (x + 1.23))
            * _ee(-1000.0 * (x + 2.1))
        )
        leg_profile = np.abs(y + 4.0 * np.abs(x - 0.2 + 0.5 * (np.abs(y + 0.8) ** 1.3))) - 0.9 + 0.005 * np.sin(200.0 * y + 5.0 * x + 5.0 * np.sin(34.0 * y + 4.0 * x)) - (10.0 * y + 2.0) / 11.0
        B_xy = _ee(20.0 * leg_profile) * _ee(50.0 * (y + 0.5 - x ** 2))

        trans = np.ones_like(x)
        L0, L1, L2 = np.zeros_like(x), np.zeros_like(x), np.zeros_like(x)
        for r in range(1, 101):
            rf = float(r)
            xr = x + (60.0 - rf) / 40.0
            Tr = y + 0.8 * (3.0 + np.cos((12.0 + 5.0 * math.cos(2.0 * rf * rf)) * y + 5.0 * math.cos(14.0 * rf))) * (xr ** 3)
            Ur = (140.0 - 30.0 * math.atan(3.0 * (rf - 20.0) / 10.0) - 30.0 * np.exp(-(y ** 2))) * Tr + np.cos(10.0 * Tr + 5.0 * rf) + 10.0 * math.cos(rf * rf)
            # Exact printed body_mask: |Tr| - (29/80)*(1 + sqrt(r/100)) + (3/2)*(|r-50|/45)^3
            bm1 = _ee(-200.0 * (np.abs(Tr) - 0.3625 * (1.0 + math.sqrt(rf / 100.0)) + 1.5 * ((abs(rf - 50.0) / 45.0) ** 3)))
            bm2 = _ee(-40.0 * ((x + 1.0 - (70.0 * np.abs(y - 0.2) + 4.0) ** (-1.0)) ** 2 + 2.0 * (np.abs(y - 0.25) ** np.clip(3.0 + x, 0.5, 6.0)) - 0.11))
            body_mask = 1.0 - bm1 * bm2
            Jr = (
                _ee(-1000.0 * (rf - 0.5))
                * _ee(-5.0 * (np.sin(Ur) - 0.4 - (0.4 / math.pi) * math.atan(0.4 * (rf - 15.0)) - 4.0 * (xr ** 2)))
                * _ee(-100.0 * xr)
                * body_mask
            )
            vis = Jr * trans
            sh1 = 1.0 + _ee(-30.0 * (y - 0.64 + (5.0 + 100.0 * ((x + 0.58) ** 2)) ** (-1.0)))
            sh2 = 0.4 + 0.5 * _ee(-1000.0 * np.sin(0.5 * Ur)) + 0.6 * Tr
            sh3 = 0.1 + 0.3 * (0.5 - (1.0 / math.pi) * math.atan(3.0 * (rf - 25.0) / 10.0)) + (3.5 + (7.0 / math.pi) * math.atan((rf - 10.0) / 10.0)) * xr
            base_L = vis * sh1 * sh2 * sh3
            at_r = 0.5 - (1.0 / math.pi) * math.atan((rf - 10.0) / 10.0)
            L0 += base_L * (0.40 + 0.20 * at_r)
            L1 += base_L * (0.20 + 0.20 * at_r)
            L2 += base_L * (0.10 + (1.0 / 6.0) * at_r)
            trans *= (1.0 - Jr)

        Z_xy = trans
        K_list = [np.zeros_like(x) for _ in range(3)]
        for r in range(1, 41):
            rf = float(r)
            s_r2 = 3.0 + math.sin(rf * rf)
            s2_r2 = 3.0 + 2.0 * math.sin(rf * rf)
            ang = math.sin(7.0 * rf) * x / 2.0 + math.cos(7.0 * rf) * (y - 8.0) / 2.0
            ang2 = math.sin(7.0 * rf) * (y - 8.0) / 2.0 - math.cos(7.0 * rf) * x / 2.0
            bokeh = _ee(-15.0 * ((np.sin(math.sin(2.0 * rf) + s_r2 * ang) ** 10) * (np.sin(math.sin(3.0 * rf) + s2_r2 * ang2) ** 10) - 0.1))
            for s_idx in range(3):
                sf = float(s_idx)
                denom = 54.0 - 30.0 * (sf - 1.0) + 32.0 * (sf * sf - sf)
                K_list[s_idx] += (9.0 / denom) * (0.6 + 0.2 * math.sin(2.0 * rf + 4.0 * sf) + 0.2 * math.sin((2.0 + 5.0 * sf) * rf)) * bokeh

        beak_slit = _ee(1000.0 * (np.abs(y - 0.36 + (x ** 2) / 10.0) - 0.0005))
        leg_shade = 1.0 + 8.0 * (_ee(-50.0 * (x - 0.15)) - 0.5) * (y + 4.0 * np.abs(x - 0.2 + 0.5 * (np.abs(y + 0.8) ** 1.3)))
        eye_mask = _ee(-2000.0 * ((x + 1.0) ** 2 + 0.064 * np.abs(y - 0.25) - 0.002))
        halo = (
            _ee(-40.0 * ((x + 1.0) ** 2 + 2.0 * (np.abs(y - 0.25) ** np.clip(3.0 + x, 0.5, 6.0)) - 0.1))
            * _ee(-17.0 * ((x + 0.24) ** 2 + 1.6 * (np.abs(y) ** np.clip(2.0 + x, 0.5, 6.0)) - 0.7))
        )
        L_list = [L0, L1, L2]
        H_channels = []
        for s_idx in range(3):
            sf = float(s_idx)
            t_beak = ((16.0 - 2.0 * sf * sf + sf) / 20.0 + 4.0 * y - 1.44 + 0.4 * (x ** 2) - beak_slit) * A_xy * Z_xy
            t_legs = Z_xy * (B_xy / 10.0) * (1.5 - (4.0 * sf * sf + 5.0 * sf) / 30.0) * leg_shade
            t_bg = ((2.0 - sf) / 10.0 + (0.2 * x + 1.06) * K_list[s_idx] * halo) * (1.0 - A_xy - B_xy) * Z_xy
            H_channels.append(t_beak + t_legs + t_bg + L_list[s_idx] * eye_mask)

    elif preset_id == "pf_strawberries":
        x = (M_grid - 1000.0) / 600.0
        y = (601.0 - N_grid) / 600.0
        W_xy = np.zeros_like(x)
        for s in range(1, 41):
            sf = float(s)
            freq = 28.0 * (0.85 ** s)
            cs, sn = math.cos(2.0 * sf), math.sin(2.0 * sf)
            c1 = np.cos(freq * (cs * x + sn * y) + 2.0 * math.sin(5.0 * sf)) ** 2
            c2 = np.cos(freq * (cs * y - sn * x) + 2.0 * math.sin(6.0 * sf)) ** 2
            W_xy += _ee(-3.0 * (c1 * c2 - 0.92))

        trans = np.ones_like(x)
        H0, H1, H2 = np.zeros_like(x), np.zeros_like(x), np.zeros_like(x)
        for s in range(1, 31):
            sf = float(s)
            Ps = np.arctan(np.tan(2.0 * math.sin(5.0 * sf) * x - 2.0 * math.cos(5.0 * sf) * y + 3.0 * math.cos(5.0 * sf) + 0.015 * np.cos(14.0 * x - 19.0 * y + 5.0 * sf)))
            Qs = np.arctan(np.tan(2.0 * (math.cos(5.0 * sf) * x + math.sin(5.0 * sf) * y + 2.0 * math.cos(4.0 * sf)) + 0.015 * np.cos(18.0 * x + 15.0 * y + 4.0 * sf)))

            def eval_Rt(t_val: float) -> np.ndarray:
                num = 1000.0 * Qs * np.sqrt(5.0 * np.abs(20.0 - 20.0 * (1.0 - 2.0 * t_val) * Ps - 27.0 * t_val))
                den = math.sqrt(20.0) * (1.0 + 50.0 * np.sqrt(np.abs(4.0 * (200.0 - ((20.0 * (1.0 - 2.0 * t_val) * Ps + 27.0 * t_val) ** 2)))))
                Et = num / np.maximum(den, 1e-5)
                return np.clip(Et * _ee(1000.0 * (np.abs(Et) - 1.0)), -1.0, 1.0)

            ac0 = np.arccos(eval_Rt(0.0))
            ac1 = np.arccos(eval_Rt(1.0))
            # Exact first-level exponent product: _ee(A) * _ee(B)
            Ms = (
                _ee(-100.0 * (Ps - 0.57 - (0.15 + 0.1 * np.cos(7.0 * Qs + 2.0 * sf)) * np.cos((10.0 + 3.0 * math.cos(14.0 * sf)) * ac0 + 0.3 * np.cos(45.0 * x + 47.0 * y + np.cos(17.0 * x) + 2.0 * math.cos(5.0 * sf)))))
                * _ee(1000.0 * (Ps - 0.72 + ((1.5 * Qs) ** 8)))
            )
            Ns = (
                _ee(100.0 * (Ps - 0.74 - (0.15 + 0.1 * np.cos(8.0 * Qs + 5.0 * sf)) * np.cos((10.0 + 3.0 * math.cos(16.0 * sf)) * ac1 + 0.3 * np.cos(38.0 * x - 47.0 * y + np.cos(19.0 * x) + 2.0 * math.cos(4.0 * sf)))))
                * _ee(-1000.0 * (Ps - 0.71 - ((1.5 * Qs) ** 8)))
            )
            Us = 1.0 - (1.0 - Ms) * (1.0 - Ns)

            def eval_Av(v_val: float) -> np.ndarray:
                g1 = 1.25 * (1.0 - Ps) * (Qs ** 2) + (Ps ** 2) - 0.55 + math.atan(100.0 * (v_val - 100.0)) / (10.0 * math.pi)
                g2 = (Qs ** 2) + (Ps ** 2) - 1.0
                return _ee(-1000.0 * (sf - 0.5)) * _ee(v_val * g1) * _ee(v_val * g2)

            A4 = eval_Av(4.0)
            A1000 = eval_Av(1000.0)
            c10 = np.cos(10.0 * ac0)
            s10 = np.sin(10.0 * ac0)
            delta_p = 0.82 + W_xy / 5.0
            c_pminus = np.cos(12.5 * Ps - delta_p)
            c_pplus = np.cos(12.5 * Ps + delta_p)
            s_pminus = np.sin(12.5 * Ps - delta_p)
            s_pplus = np.sin(12.5 * Ps + delta_p)

            def eval_Cv(v_val: float) -> np.ndarray:
                env = _ee(1.5 * ((Qs ** 2) + ((Ps - 0.25) ** 2) - 0.42 + W_xy / 5.0))
                return (
                    _ee(v_val * (c10 * c_pminus))
                    * _ee(-v_val * (c10 * c_pplus))
                    * _ee(v_val * (s10 * s_pminus))
                    * _ee(-v_val * (s10 * s_pplus))
                    * env
                )

            C10 = eval_Cv(10.0)
            C20 = eval_Cv(20.0)
            Bs = _ee(-70.0 * (np.cos(20.0 * ac0) * np.cos(25.0 * Ps) - 0.94))
            dimple = (0.1 - 0.025 * np.cos(20.0 * ac0) * np.cos(25.0 * Ps)) * A4 * A1000
            spec = 20.0 * _ee(-70.0 * (C20 - 0.5)) + 20.0 * _ee(-10.0 * (C10 - 0.5))
            calyx_e = _ee(-np.abs(7.1 - 10.0 * Ps)) * Us

            for v_idx, H_acc in enumerate((H0, H1, H2)):
                vf = float(v_idx)
                calyx_col = ((5.0 - 3.0 * ((vf - 1.0) ** 2) + W_xy) / 10.0) * calyx_e
                L_vs = dimple * (4.0 * vf * vf - 13.0 * vf + 11.0 + math.cos(7.0 * sf + vf * sf) + spec) + Bs
                H_acc += trans * (calyx_col + A1000 * (1.0 - Us) * L_vs)

            trans *= (1.0 - A1000) * (1.0 - Us) * np.clip(1.0 - 1.25 * A4, 0.0, 1.0)
        H_channels = [H0, H1, H2]

    elif preset_id == "pf_bipolar_nebula":
        x = (M_grid - 1000.0) / 420.0
        y = (601.0 - N_grid) / 420.0
        trans = np.ones_like(x)
        A_xy = np.zeros_like(x)
        S_xy = np.zeros_like(x)
        for s in range(1, 28):
            sf = float(s)
            s2 = sf * sf
            Rs = 0.1 * sf + 0.06 * math.cos(5.0 * s2)
            Us = x + (0.15 + 0.2 * math.cos(3.0 * s2)) * y + 1e-4
            transverse = 2.0 * (Rs ** 0.3) * (np.maximum(np.abs(Us), 1e-4) ** (-0.3)) * (y - (0.15 + 0.2 * math.cos(4.0 * s2)) * x)
            Ls = np.sqrt(Us ** 2 + transverse ** 2) - Rs
            # J_s(x,y) = e^(-e^(25 - 50s) - e^(10*L_s))
            Js = _ee(10.0 * Ls) * 0.25
            vis = trans * Js
            A_xy += 0.52 * vis * _ee(0.15 * (sf - 23.0)) * _ee(-3.0 * Ls)
            S_xy += 2.0 * vis * Ls
            trans *= (1.0 - Js)

        Q5 = x * math.cos(375.0) + y * math.sin(375.0)
        E_xy = np.zeros_like(x)
        for s in range(1, 41):
            sf = float(s)
            freq = 1.25 ** min(s, 28)
            c7, s7 = math.cos(7.0 * sf), math.sin(7.0 * sf)
            c4, s4 = math.cos(4.0 * sf), math.sin(4.0 * sf)
            c8, s8 = math.cos(8.0 * sf), math.sin(8.0 * sf)
            d1 = np.cos(freq * (c7 * S_xy + s7 * Q5 + 2.0 * math.cos(17.0 * sf)) + 4.0 * np.cos(freq * (c4 * S_xy + s4 * Q5)) + 2.0 * math.cos(5.0 * sf))
            d2 = np.cos(freq * (c7 * Q5 - s7 * S_xy + 2.0 * math.cos(15.0 * sf)) + 4.0 * np.cos(freq * (c8 * S_xy + s8 * Q5)) + 2.0 * math.cos(7.0 * sf))
            E_xy += (0.95 ** s) * d1 * d2

        W_xy = _ee(10.0 * np.sqrt(x ** 2 + y ** 2) - 1.0 + 0.25 * E_xy)
        K0, K1, K2 = np.zeros_like(x), np.zeros_like(x), np.zeros_like(x)
        for s in range(1, 51):
            sf = float(s)
            s2 = sf * sf
            freq = (23.0 / 20.0) ** s
            Qs = x * math.cos(15.0 * s2) + y * math.sin(15.0 * s2)
            cs, sn = math.cos(15.0 * s2), math.sin(15.0 * s2)
            g = np.cos(freq * (cs * S_xy + sn * Qs) + 2.0 * math.cos(27.0 * s2)) * np.cos(freq * (cs * Qs - sn * S_xy) + 2.0 * math.cos(28.0 * s2)) - 1.25 + 2.0 * A_xy + E_xy / 7.0
            C0s = _ee(-0.25 * g)
            C1s = _ee(-4.0 * g)
            Is = (45.0 * C1s + 6.0 * C0s) * (0.95 ** s)
            for v_idx, K_acc in enumerate((K0, K1, K2)):
                vf = float(v_idx)
                K_acc += Is * (12.0 - 4.0 * vf + vf * vf + (vf - 1.0) * math.cos(2.0 * s2) + 8.0 * math.cos((7.0 + vf) * s2)) / 50.0

        # Exact pinprick diffraction-spiked starfield T_v(x,y) with freq = 2*(6/5)^s
        T0, T1, T2 = np.zeros_like(x), np.zeros_like(x), np.zeros_like(x)
        for s in range(1, 31):
            sf = float(s)
            s2 = sf * sf
            Ps = y * math.cos(15.0 * s2) - x * math.sin(15.0 * s2)
            Qs = x * math.cos(15.0 * s2) + y * math.sin(15.0 * s2)
            freq = 2.0 * (1.2 ** s)
            c19, s19 = math.cos(19.0 * s2), math.sin(19.0 * s2)
            Ms = np.arccos(np.cos(freq * (c19 * Ps + s19 * Qs) + 2.0 * math.cos(27.0 * s2)))
            Ns = np.arccos(np.cos(freq * (c19 * Qs - s19 * Ps) + 2.0 * math.cos(28.0 * s2)))
            ang = np.arctan2(Ms, np.maximum(Ns, 1e-4))
            Bs = _ee(5.0 * np.cos(20.0 * ang + 2.0 * np.cos(9.0 * ang + s2)) + 3.75)
            r2_star = Ms ** 2 + Ns ** 2
            star_profile = 4.0 * _ee(200.0 * (r2_star - 1.0 / 800.0 - Bs / 200.0)) + _ee(20.0 * r2_star - 0.14)
            for v_idx, T_acc in enumerate((T0, T1, T2)):
                vf = float(v_idx)
                T_acc += ((vf * vf - 2.0 * vf + 4.0 + (vf - 1.0) * ((-1.0) ** s)) / 4.0) * star_profile

        H_channels = [
            1.1 * (1.0 - W_xy) * K_acc * A_xy + ((4.0 + vf * vf - vf) / 2.0) * W_xy + T_acc
            for vf, K_acc, T_acc in zip((0.0, 1.0, 2.0), (K0, K1, K2), (T0, T1, T2))
        ]

    elif preset_id == "pf_autumn_trees":
        x = (M_grid - 1000.0) / 600.0
        y = (1001.0 - N_grid) / 600.0
        E_xy = np.zeros_like(x)
        for s in range(1, 26):
            sf = float(s)
            s2 = sf * sf
            freq = 1.25 ** s
            c2, s2_sin = math.cos(2.0 * s2), math.sin(2.0 * s2)
            c7, s7 = math.cos(7.0 * s2), math.sin(7.0 * s2)
            c8, s8 = math.cos(8.0 * s2), math.sin(8.0 * s2)
            # Exact printed D_s(x,y): cos(freq*(...) + 4*cos(freq*(...)) + 2*cos(5s)) * cos(...)
            d1 = np.cos(freq * (c2 * x + s2_sin * y) + 4.0 * np.cos(freq * (c7 * x + s7 * y)) + 2.0 * math.cos(5.0 * sf))
            d2 = np.cos(freq * (c2 * y - s2_sin * x) + 4.0 * np.cos(freq * (c8 * x + s8 * y)) + 2.0 * math.cos(5.0 * sf))
            E_xy += 1.25 * (0.9 ** s) * d1 * d2

        G_xy = _ee(500.0 * (y - 0.02 + E_xy / 300.0))
        I_xy = _ee(-50.0 * y + 2.0 - 30.0 * np.abs(x) - 0.25 * E_xy)

        Z_prev = np.ones_like(x)
        B_xy = np.zeros_like(x)
        K_xy = np.zeros_like(x)
        C_xy = np.zeros_like(x)
        for s in range(1, 9):
            sf = float(s)
            s2 = sf * sf
            scale_s = (15.0 + math.cos(3.0 * s2)) / 10.0
            Us = np.arctan(np.tan(0.7 * (scale_s ** s) * x - 1.5 + 0.5 * math.cos(9.0 * s2)))
            Vs = 0.5 * (scale_s ** s) * y
            # Exact product _ee(A) * _ee(B)
            Ps = (
                _ee(200.0 * (10.0 * np.abs(Us) - 22.0 / 30.0 + Vs / 3.0 + ((0.2 * (Vs - 0.2)) ** 8) - 0.75 - 3.0 * np.arctan(-30.0 * Vs - 6.0) / (2.0 * math.pi) + E_xy / 20.0))
                * _ee(-100.0 * (Vs + 0.2 + E_xy / 100.0))
            )
            Ls = Vs - 0.1 - np.abs(Us) / 5.0 + np.cos(3.0 * x + 2.0 * y + 1.0) / 20.0
            Rs = 20.0 * Ls - np.arctan(np.tan(20.0 * Ls))
            Rs2 = Rs ** 2

            A0s = np.ones_like(x)
            A1s = np.ones_like(x)
            for u in range(3):
                uf = float(u)
                Tus = 5.0 * np.abs(Us) - 0.5 + np.cos(uf * uf + Rs2)
                Bus = 0.2 * math.cos((4.0 + uf) ** 2) * Tus * _ee(-20.0 * Tus)
                base_ang = (
                    np.abs(np.arctan(np.tan(20.0 * Ls + 5.0 * Bus)))
                    - 0.4
                    + np.cos(3.0 * x - 2.0 * y + 1.0 + 2.0 * uf) / 20.0
                    + np.cos(5.0 * x + 3.0 * y + 2.0 + 3.0 * uf) / 20.0
                    + ((3.0 + 9.0 * uf) / 20.0) * (Us ** 2)
                )
                e_term = (1.25 ** s) * (3.0 ** (-u)) * E_xy
                q0 = base_ang + 0.1 * (e_term / 25.0) + (1.0 / 400.0) * ((10.0 * Vs - 12.0) ** 2)
                # Exact printed Q_{1,u,s} includes -8/5*v = -1.6 and (1+19v)/10 * (5/4)^s * 3^(-u) * E(x,y)
                q1 = base_ang - 1.6 + 0.5 * e_term + (8.0 / 400.0) * ((10.0 * Vs - 15.0) ** 2)
                A0s *= _ee(-200.0 * q0)
                A1s *= _ee(-10.0 * q1)

            Os = 1.0 - A0s * A1s * (1.0 - Ps)
            vis_s = Z_prev * Os
            B_xy += vis_s * (1.5 ** s)
            K_xy += vis_s * sf
            C_xy += vis_s * (0.5 + 0.3 * ((Vs - 1.3) ** 2)) * (1.0 - A1s)
            Z_prev *= (1.0 - Os)

        Z8 = Z_prev
        # Preserve signed 1/(10000*y + 1) for true 3D perspective on the fallen leaf ground!
        denom_s = 10000.0 * y + 1.0
        denom_s = np.where(np.abs(denom_s) < 1e-1, np.where(denom_s >= 0.0, 1e-1, -1e-1), denom_s)
        S_xy = 4000.0 * Z8 / denom_s
        M_xy = x * B_xy + x * S_xy
        N_xy = y * B_xy + S_xy

        trans_J = np.ones_like(x)
        W0, W1, W2 = np.zeros_like(x), np.zeros_like(x), np.zeros_like(x)
        for s in range(1, 31):
            sf = float(s)
            s2 = sf * sf
            c2, s2_sin = math.cos(2.0 * s2), math.sin(2.0 * s2)
            # Exact printed J_s(x,y): e^(-e^(-20*cos(...)*cos(...) + 4 + 14*G - (s - 4.5*G)*E))
            c_m = np.cos(30.0 * (c2 * M_xy + s2_sin * N_xy) + 2.0 * math.cos(5.0 * s2))
            c_n = np.cos(30.0 * (c2 * N_xy - s2_sin * M_xy) + 2.0 * math.cos(7.0 * s2))
            Js = _ee(-20.0 * c_m * c_n + 4.0 + 14.0 * G_xy - (sf - 4.5 * G_xy) * E_xy)
            vis_j = trans_J * Js
            k2 = K_xy ** 2
            c3 = np.cos(3.0 * k2) ** 2
            s4 = np.sin(4.0 * k2) ** 4
            g_f = (1.0 + G_xy) / 4.0
            for v_idx, W_acc in enumerate((W0, W1, W2)):
                vf = float(v_idx)
                col = (7.0 - 3.0 * vf) / 5.0 + ((3.0 + 10.0 * vf * vf - 20.0 * vf) / 10.0) * c3 + ((7.0 - 4.0 * vf) / 10.0) * s4 + g_f * np.cos((4.0 + vf) * s2 + k2)
                W_acc += vis_j * col
            trans_J *= (1.0 - Js)

        H_channels = [
            (3.0 - vf) / 20.0
            + (18.0 + 6.0 * vf * vf - 6.0 * vf + 5.0 * y) * (1.0 - G_xy) * Z8 / 30.0
            + (W_acc / 400.0) * (460.0 * C_xy + (13.0 - 5.0 * ((vf - 1.0) ** 2)) * G_xy * Z8 * (3.0 + 20.0 * I_xy))
            for vf, W_acc in zip((0.0, 1.0, 2.0), (W0, W1, W2))
        ]

    else:  # pf_peacock
        x = (M_grid - 1000.0) / 1300.0
        y = (1021.0 - N_grid) / 1300.0
        r2_xy = x ** 2 + y ** 2

        # Golden bokeh background K(x,y)
        K_xy = np.zeros_like(x)
        for r in range(1, 41):
            rf = float(r)
            s7, c7 = math.sin(7.0 * rf), math.cos(7.0 * rf)
            p1 = np.sin(math.sin(2.0 * rf) + 10.0 * (s7 * x + c7 * y)) ** 10
            p2 = np.sin(math.sin(3.0 * rf) + 10.0 * (s7 * y - c7 * x)) ** 10
            K_xy += _ee(3.0 - 30.0 * p1 * p2)

        trans = np.ones_like(x)
        E_xy = np.zeros_like(x)
        L0, L1, L2 = np.zeros_like(x), np.zeros_like(x), np.zeros_like(x)
        U0, U1, U2 = np.zeros_like(x), np.zeros_like(x), np.zeros_like(x)

        x_rot = 0.94 * x + 0.07 * y
        y_rot = 0.94 * y - 0.07 * x

        for u in range(1, 6):
            uf = float(u)
            cu2 = math.cos(uf * uf)
            for r in range(1, 21):
                rf = float(r)
                Tur = (0.08 - r2_xy / 15.0) * cu2 + math.pi * rf / 20.0
                sT, cT = np.sin(Tur), np.cos(Tur)
                Pur = sT * x_rot + cT * y_rot
                Rur = cT * x_rot - sT * y_rot
                denom_ur = 4.0 * uf + ((-1.0) ** r)
                Nur = 30.0 * Rur / denom_ur
                offset_ur = 0.9 * (2.0 * uf / 15.0 + ((-1.0) ** r) / 30.0)
                Wur = ((Rur + offset_ur) ** 2) + 1.4 * (Pur ** 2)
                sqrt_W = np.sqrt(Wur)
                scale_u = 0.67 + uf / 15.0

                # Exact non-overlapping iridescent eye-spot (ocellus) mask E_xy and rings L_s
                oc_outer = _ee(1000.0 * (sqrt_W - 0.023 * scale_u))
                E_xy += oc_outer
                oc_inner = _ee(1000.0 * (sqrt_W - 0.020 * scale_u))
                oc_ring = _ee(1000.0 * (np.abs(sqrt_W - 0.015 * scale_u) - 0.0018 * scale_u))
                L0 += oc_inner * (2.0 / 3.0) * oc_ring
                L1 += oc_inner * 2.0 * oc_ring
                L2 += oc_inner * 2.0 * oc_ring

                # Exact feather barb herringbone wave A_ur & rachis shaft B_ur from equation plate
                abs_Nur = np.abs(Nur)
                pow_p = np.clip(1.0 - 2.0 * ((7.0 * uf / 30.0) ** 4) * (abs_Nur ** 4), 0.25, 1.5)
                barb_cos = np.cos(
                    rf + uf
                    + 6.0 * np.cos(18.0 * Rur + uf * rf + 14.0 * Pur)
                    + 600.0 * Rur
                    + 600.0 * (np.abs(Pur) ** pow_p)
                    - 4800.0 * (Pur ** 2)
                )
                Aur = _ee(-1000.0 * barb_cos + 400.0 - 700.0 * ((0.7 * abs_Nur) ** 4))
                Bur = _ee(1000.0 * np.abs(Pur) - 3.0 * (1.0 - np.cbrt(abs_Nur / 12.0)))
                vane_env = _ee(-100.0 * (y + 0.02 - 0.08 * np.abs(x))) * _ee(-40.0 * (Rur + 2.0 * uf / 15.0 + ((-1.0) ** r) / 30.0 - 12.0 * (Pur ** 2)))
                a_gate = Aur * _ee(1000.0 * (np.abs(Pur) - 0.05))
                Jur = vane_env * (1.0 - (1.0 - Bur) * (1.0 - a_gate))

                vis = Jur * trans
                a_shade = Aur * _ee(-1000.0 * (np.abs(Pur) - 0.006))
                radial_ratio = 30.0 * r2_xy / denom_ur
                for s_idx, U_acc in enumerate((U0, U1, U2)):
                    sf = float(s_idx)
                    c_vane = (0.7 - ((2.0 - sf) ** 2) / 10.0 - (0.1 + sf / 5.0) * radial_ratio) * a_shade + 1.5 * Bur
                    U_acc += vis * np.clip(c_vane, 0.0, 1.5)
                trans *= (1.0 - Jur)

        E_xy = np.clip(E_xy, 0.0, 1.0)
        # Exact Hamid Naderi Yeganeh Peacock Body Q(x,y) with 3D royal cobalt limb shading
        q_den = (
            3.0
            + 150.0 * np.exp(-500.0 * np.abs(y - 0.3))
            + 100.0 * np.exp(-90.0 * ((y + 0.08) ** 2))
            + 10.0 * np.exp(-200.0 * ((x + y - 0.28) ** 2))
        )
        Q_xy = 100.0 * (q_den ** (-1.0)) * np.abs(6.0 * x + 3.0 * (y ** 2) * np.exp(np.clip(100.0 * (y - 0.25), -30.0, 30.0))) - np.sqrt(np.abs(6.0 * y + 0.48))
        body_cut = 1.0 - _ee(-1000.0 * (x - y / 5.0 + 0.1)) * _ee(50.0 * Q_xy)
        body_fill = _ee(-100.0 * (x - y / 5.0 + 0.1)) * _ee(50.0 * Q_xy) * _ee(5.0 * Q_xy)
        face_mark = 0.3 * _ee(500.0 * (np.abs(x) - 30.0 * x * ((y - 0.08) ** 6))) * _ee(500.0 * (np.abs(((x + 0.01) ** 2) + (y ** 2) - 0.08) - 0.014))

        fan_gate = _ee(-500.0 * (y + 0.06 - 0.12 * np.abs(x)))
        bg_weight = trans * (1.0 - E_xy * fan_gate)
        H_channels = []
        for s_idx, (U_acc, L_acc) in enumerate(zip((U0, U1, U2), (L0, L1, L2))):
            sf = float(s_idx)
            bg_col = ((3.0 - sf) / 10.0) * (1.0 + K_xy) * ((y + 1.0) / 2.0)
            feathers_col = bg_col * bg_weight + fan_gate * (U_acc * (1.0 - E_xy) + L_acc * E_xy)
            H_s = feathers_col * body_cut + ((1.0 - 3.0 * y) / (5.0 ** (2.0 - sf))) * body_fill - face_mark
            H_channels.append(H_s)

    rgb = np.stack([hamid_F_quantize(H_c) for H_c in H_channels], axis=-1)
    result: Dict[str, Any] = {
        "title": preset["name"],
        "subtitle": preset["description"],
        "author": preset["author"],
        "mode": "pixel_field",
        "N": H,
        "width": W,
        "height": H,
        "nominal_aspect": list(preset["aspect"]),
        "harmonics_count": "2D Pixel-Field Exact",
        "latex": preset["latex"],
        "primitives": [],
        "image_base64": encode_rgb_to_base64_png(rgb),
        "python_code": build_pixel_field_python_script(preset["name"], preset["aspect"], preset["latex"]),
    }
    if include_raw_array:
        result["rgb_array"] = rgb
    return result


def build_pixel_field_python_script(title: str, aspect: Tuple[int, int], latex_dict: Dict[str, str]) -> str:
    W_nom, H_nom = aspect
    m0 = W_nom // 2
    n0 = (H_nom // 2) + 1
    S = H_nom // 2
    return f'''"""
2D Pixel-Field Mathematical Painting: {title}
Style: Hamid Naderi Yeganeh Double-Exponential Level-Set Equations
For m = 1..{W_nom} and n = 1..{H_nom}:
    pixel(m, n) = rgb(F(H_0(x, y)), F(H_1(x, y)), F(H_2(x, y)))
where F(x) = floor(255 * exp(-exp(-1000*x)) * |x|**exp(-exp(1000*(x-1))))
"""
import numpy as np
import matplotlib.pyplot as plt

def ee(z):
    return np.exp(-np.exp(np.clip(z, -45.0, 45.0)))

def F(x):
    return np.floor(255.0 * ee(-1000.0 * x) * (np.maximum(np.abs(x), 1e-8) ** ee(1000.0 * (x - 1.0)))).astype(np.uint8)

W, H = 800, int(800 * {H_nom} / {W_nom})
m, n = np.meshgrid(np.linspace(1, {W_nom}, W), np.linspace(1, {H_nom}, H))
x, y = (m - {m0}) / {S}.0, ({n0} - n) / {S}.0

# Multi-Octave Trigonometric fBm Field E(x, y)
E = sum(1.25 * (0.88**s) * np.cos((1.25**s)*(np.cos(2*s*s)*x + np.sin(2*s*s)*y) + 2*np.cos(5*s))
                         * np.cos((1.25**s)*(np.cos(2*s*s)*y - np.sin(2*s*s)*x) + 2*np.cos(5*s))
        for s in range(1, 20))

# Evaluate H_0(x,y), H_1(x,y), H_2(x,y) and quantize via F(x)
H0 = 0.5 + 0.35 * np.cos(3.0 * x + E)
H1 = 0.4 + 0.35 * np.sin(3.0 * y + E)
H2 = 0.6 - 0.30 * np.cos(2.0 * (x - y) + E)
img = np.stack([F(H0), F(H1), F(H2)], axis=-1)

plt.figure(figsize=(10, 6), facecolor="#050811")
plt.imshow(img)
plt.axis("off")
plt.tight_layout()
plt.show()
'''


# ============================================================================
# 2. UNIVERSAL GEOMETRIC IR -> 2D PIXEL-FIELD PAINTING COMPILER
# ============================================================================

_SAFE_MATH_FUNCS = {
    "sin": np.sin,
    "cos": np.cos,
    "tan": lambda a: np.tan(np.clip(a, -1.55, 1.55)),
    "arcsin": lambda a: np.arcsin(np.clip(a, -1.0, 1.0)),
    "arccos": lambda a: np.arccos(np.clip(a, -1.0, 1.0)),
    "arctan": np.arctan,
    "atan": np.arctan,
    "atan2": np.arctan2,
    "arctan2": np.arctan2,
    "sinh": lambda a: np.sinh(np.clip(a, -20.0, 20.0)),
    "cosh": lambda a: np.cosh(np.clip(a, -20.0, 20.0)),
    "tanh": np.tanh,
    "exp": lambda a: np.exp(np.clip(a, -40.0, 40.0)),
    "log": lambda a: np.log(np.maximum(np.abs(a), 1e-6)),
    "log10": lambda a: np.log10(np.maximum(np.abs(a), 1e-6)),
    "sqrt": lambda a: np.sqrt(np.maximum(a, 0.0)),
    "abs": np.abs,
    "fabs": np.abs,
    "sign": np.sign,
    "floor": np.floor,
    "ceil": np.ceil,
    "clip": np.clip,
    "maximum": np.maximum,
    "minimum": np.minimum,
    "max": np.maximum,
    "min": np.minimum,
    "hypot": np.hypot,
    "pow": lambda a, b: np.sign(a) * (np.abs(a) ** np.clip(b, -8.0, 8.0)) if isinstance(b, (int, float)) and int(b) == b and int(b) % 2 == 1 else (np.abs(a) ** np.clip(b, -8.0, 8.0)),
    "ee": _ee,
    "mod": lambda a, b: np.mod(a, np.where(np.abs(b) < 1e-6, 1e-6, b)),
}


def _safe_eval_math_expr(expr: str, env: Dict[str, Any]) -> np.ndarray:
    """
    Sandboxed AST-validated evaluator for analytical mathematical expressions
    in (x, y, E, r, theta, t). Strictly forbids attribute access, imports,
    subscripts, or non-whitelisted calls.
    """
    ref_arr = None
    for v in env.values():
        if isinstance(v, np.ndarray):
            ref_arr = v
            break
    if ref_arr is None:
        ref_arr = np.zeros((1,), dtype=np.float32)

    if not isinstance(expr, str) or not expr.strip():
        return np.zeros_like(ref_arr, dtype=np.float32)

    cleaned = expr.strip().replace("^", "**")
    cleaned = re.sub(r"\b(?:np|numpy|math)\.", "", cleaned)
    tree = ast.parse(cleaned, mode="eval")

    consts = {"pi": math.pi, "e": math.e, "tau": 2.0 * math.pi}

    def _eval(node: ast.AST) -> Any:
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return float(node.value)
            raise ValueError("Unsupported constant type")
        if isinstance(node, ast.Name):
            if node.id in env:
                return env[node.id]
            if node.id in consts:
                return consts[node.id]
            raise ValueError(f"Unknown mathematical symbol: {node.id}")
        if isinstance(node, ast.UnaryOp):
            operand = _eval(node.operand)
            if isinstance(node.op, ast.UAdd):
                return operand
            if isinstance(node.op, ast.USub):
                return -operand
            raise ValueError("Unsupported unary operator")
        if isinstance(node, ast.BinOp):
            left = _eval(node.left)
            right = _eval(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                safe_r = np.where(np.abs(right) < 1e-6, np.where(right >= 0, 1e-6, -1e-6), right) if isinstance(right, np.ndarray) else (right if abs(right) >= 1e-6 else 1e-6)
                return left / safe_r
            if isinstance(node.op, ast.Mod):
                safe_r = np.where(np.abs(right) < 1e-6, 1e-6, right) if isinstance(right, np.ndarray) else (right if abs(right) >= 1e-6 else 1e-6)
                return np.mod(left, safe_r)
            if isinstance(node.op, ast.Pow):
                if isinstance(right, (int, float)):
                    p = float(np.clip(right, -8.0, 8.0))
                    if abs(p - round(p)) < 1e-6:
                        ip = int(round(p))
                        if ip < 0:
                            base = np.where(np.abs(left) < 1e-4, 1e-4, left) if isinstance(left, np.ndarray) else (left if abs(left) >= 1e-4 else 1e-4)
                            return base ** ip
                        return left ** ip
                    return np.maximum(np.abs(left), 1e-6) ** p
                return np.maximum(np.abs(left), 1e-6) ** np.clip(right, -8.0, 8.0)
            raise ValueError("Unsupported binary operator")
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in _SAFE_MATH_FUNCS:
                raise ValueError("Disallowed function call in SDF expression")
            if node.keywords:
                raise ValueError("Keyword arguments not allowed")
            fn = _SAFE_MATH_FUNCS[node.func.id]
            args = [_eval(a) for a in node.args]
            return fn(*args)
        raise ValueError(f"Unsupported AST construct: {type(node).__name__}")

    val = _eval(tree)
    if isinstance(val, np.ndarray):
        arr = np.broadcast_to(val, ref_arr.shape)
        return np.nan_to_num(arr, nan=1.0, posinf=10.0, neginf=-10.0).astype(np.float32)
    return np.full_like(ref_arr, float(val), dtype=np.float32)


def compute_hamid_trigonometric_fbm(X: np.ndarray, Y: np.ndarray, octaves: int = 7) -> np.ndarray:
    E = np.zeros_like(X, dtype=np.float32)
    total_amp = 0.0
    for s in range(1, octaves + 1):
        amp = 1.25 * (0.85 ** s)
        freq = 3.2 * (1.28 ** s)
        a2 = 2.0 * (s * s)
        a7 = 7.0 * (s * s)
        a8 = 8.0 * (s * s)
        u1 = math.cos(a2) * X + math.sin(a2) * Y
        v1 = math.cos(a2) * Y - math.sin(a2) * X
        w1 = math.cos(a7) * X + math.sin(a7) * Y
        w2 = math.cos(a8) * X + math.sin(a8) * Y
        phase = 2.0 * math.cos(5.0 * s)
        D_s = np.cos(freq * u1 + 3.2 * np.cos(freq * w1) + phase) * np.cos(freq * v1 + 3.2 * np.cos(freq * w2) + phase)
        E += amp * D_s
        total_amp += amp
    return E / max(total_amp, 1e-6)


def _eval_hamid_starfield(
    X: np.ndarray,
    Y: np.ndarray,
    num_octaves: int = 14,
    t_phase: float = 0.0,
    density_scale: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Multi-octave Hamid Bipolar-Nebula-style pinprick diffraction starfield."""
    T0 = np.zeros_like(X, dtype=np.float32)
    T1 = np.zeros_like(X, dtype=np.float32)
    T2 = np.zeros_like(X, dtype=np.float32)
    for s in range(1, num_octaves + 1):
        sf = float(s)
        s2 = sf * sf
        c15, s15 = math.cos(15.0 * s2), math.sin(15.0 * s2)
        Ps = Y * c15 - X * s15
        Qs = X * c15 + Y * s15
        freq = 2.2 * density_scale * (1.24 ** s)
        c19, s19 = math.cos(19.0 * s2), math.sin(19.0 * s2)
        Ms = np.arccos(np.cos(freq * (c19 * Ps + s19 * Qs) + 2.0 * math.cos(27.0 * s2)))
        Ns = np.arccos(np.cos(freq * (c19 * Qs - s19 * Ps) + 2.0 * math.cos(28.0 * s2)))
        r2_star = Ms * Ms + Ns * Ns
        twinkle = 0.85 + 0.15 * math.sin(t_phase + 2.3 * sf)
        if s <= 4:
            ang = np.arctan2(Ms, np.maximum(Ns, 1e-4))
            Bs = _ee(5.0 * np.cos(20.0 * ang + 2.0 * np.cos(9.0 * ang + s2)) + 3.75)
            sp = twinkle * (2.2 * _ee(240.0 * (r2_star - 1.0 / 900.0 - Bs / 240.0)) + 0.45 * _ee(32.0 * r2_star - 0.12))
        else:
            sp = twinkle * (1.6 * _ee(260.0 * (r2_star - 1.0 / 1100.0)) + 0.25 * _ee(48.0 * r2_star - 0.10))
        sign_s = (-1.0) ** s
        T0 += ((4.0 - sign_s) / 4.0) * sp
        T1 += (3.0 / 4.0) * sp
        T2 += ((4.0 + sign_s) / 4.0) * sp
    return np.clip(T0, 0.0, 1.4), np.clip(T1, 0.0, 1.4), np.clip(T2, 0.0, 1.4)


def evaluate_background_field(
    bg_style: str,
    X: np.ndarray,
    Y: np.ndarray,
    E: np.ndarray,
    primary_rgb: Optional[List[float]] = None,
    secondary_rgb: Optional[List[float]] = None,
    t_phase: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    def bokeh_waves(freq_base: float, num_waves: int = 18) -> np.ndarray:
        acc = np.zeros_like(X, dtype=np.float32)
        for r in range(1, num_waves + 1):
            rf = float(r)
            ang = 1.73 * rf
            ca, sa = math.cos(ang), math.sin(ang)
            xr = ca * X + sa * Y
            yr = -sa * X + ca * Y
            wf = freq_base * (1.0 + 0.11 * rf)
            drift = 0.12 * math.sin(t_phase + rf)
            lobe = (np.sin(wf * xr + 1.3 * rf + drift) ** 8) * (np.sin(wf * yr + 2.1 * rf - drift) ** 8)
            acc += (0.93 ** r) * _ee(-16.0 * (lobe - 0.09))
        return np.clip(acc / 4.2, 0.0, 1.0)

    style = (bg_style or "golden_bokeh").strip().lower()

    if style == "golden_bokeh":
        bk = bokeh_waves(2.5, 18)
        glow_y = np.clip((Y + 1.15) / 1.9, 0.35, 1.15)
        B0 = (0.68 + 0.34 * bk + 0.06 * E) * glow_y
        B1 = (0.44 + 0.46 * bk + 0.05 * E) * glow_y
        B2 = (0.10 + 0.36 * (bk ** 1.4)) * glow_y
    elif style in ("bokeh_forest", "jungle_canopy"):
        bk = bokeh_waves(2.3, 18)
        vig = np.clip(0.95 - 0.22 * (X ** 2 + Y ** 2) + 0.14 * X, 0.28, 1.12)
        B0 = (0.20 + 0.50 * bk + 0.06 * E) * vig
        B1 = (0.38 + 0.56 * bk + 0.06 * E) * vig
        B2 = (0.10 + 0.32 * (bk ** 1.5)) * vig
    elif style == "cosmic_nebula":
        r2 = X ** 2 + Y ** 2
        swirl_x = X * np.cos(0.6 * E + 0.08 * t_phase) - Y * np.sin(0.6 * E + 0.08 * t_phase)
        swirl_y = X * np.sin(0.6 * E + 0.08 * t_phase) + Y * np.cos(0.6 * E + 0.08 * t_phase)
        filament = np.clip(0.5 + 0.5 * np.cos(3.2 * swirl_x + 2.8 * E) * np.cos(3.0 * swirl_y - 2.4 * E), 0.0, 1.0)
        glow = np.exp(-0.95 * r2) * (0.35 + 0.85 * (filament ** 1.4))
        band = np.exp(-2.8 * ((Y - 0.28 * X + 0.22 * E) ** 2)) * (0.4 + 0.6 * filament)
        S0, S1, S2 = _eval_hamid_starfield(X, Y, num_octaves=16, t_phase=t_phase, density_scale=1.05)
        B0 = 0.02 + 0.50 * glow + 0.32 * band + S0
        B1 = 0.015 + 0.20 * (glow ** 1.2) + 0.22 * band + S1
        B2 = 0.05 + 0.58 * glow + 0.48 * band + S2
    elif style in ("sunset_sky", "alpine_sunset"):
        horizon = _ee(-32.0 * (Y + 0.34 + 0.03 * E))
        y_sky = np.clip((Y + 0.34) / 1.35, 0.0, 1.0)
        sun_r2 = (X - 0.12) ** 2 + (Y + 0.06) ** 2
        sun_halo = np.exp(-3.8 * sun_r2) + 0.65 * np.exp(-16.0 * sun_r2)
        clouds = np.clip(0.5 + 0.5 * np.sin(3.2 * X + 4.0 * E - 0.2 * t_phase) * np.cos(5.5 * Y + 2.5 * E), 0.0, 1.0) * np.exp(-3.2 * ((Y - 0.22) ** 2))
        sky0 = 0.96 - 0.52 * (y_sky ** 1.1) + 0.38 * sun_halo + 0.12 * clouds
        sky1 = 0.64 - 0.42 * (y_sky ** 0.95) + 0.44 * sun_halo + 0.08 * clouds
        sky2 = 0.38 - 0.04 * y_sky + 0.22 * (y_sky ** 1.5) + 0.22 * sun_halo + 0.08 * clouds
        g_depth = np.clip((-0.34 - Y) / 0.70, 0.0, 1.0)
        # Painterly alpine valley/meadow dusk floor with warm sunlit glow near horizon
        meadow = 0.5 + 0.5 * np.sin(5.5 * E + 3.0 * X)
        sun_refl = np.exp(-4.0 * ((X - 0.12) ** 2)) * np.exp(-3.5 * g_depth)
        g0 = 0.24 * (1.0 - 0.45 * g_depth) + 0.10 * meadow + 0.36 * sun_refl
        g1 = 0.15 * (1.0 - 0.45 * g_depth) + 0.08 * meadow + 0.22 * sun_refl
        g2 = 0.14 * (1.0 - 0.35 * g_depth) + 0.06 * meadow + 0.10 * sun_refl
        B0 = horizon * sky0 + (1.0 - horizon) * g0
        B1 = horizon * sky1 + (1.0 - horizon) * g1
        B2 = horizon * sky2 + (1.0 - horizon) * g2
    elif style == "deep_ocean":
        depth = np.clip(0.5 * (Y + 1.05), 0.0, 1.0)
        rays = (0.5 + 0.5 * np.cos(3.8 * X - 1.5 * Y + 0.85 * E - 0.35 * t_phase)) ** 5
        rays2 = (0.5 + 0.5 * np.cos(7.2 * X - 2.6 * Y - 0.65 * E + 0.2 * t_phase)) ** 8
        caustic = (0.5 + 0.5 * np.cos(16.0 * X + 4.0 * E) * np.cos(16.0 * Y - 4.0 * E + t_phase)) ** 6 * (depth ** 1.5)
        B0 = 0.015 + 0.09 * (depth ** 1.6) + 0.10 * rays * depth + 0.08 * caustic
        B1 = 0.10 + 0.40 * (depth ** 1.1) + 0.22 * (rays + 0.5 * rays2) * depth + 0.18 * caustic
        B2 = 0.20 + 0.48 * (depth ** 0.9) + 0.25 * (rays + 0.5 * rays2) * depth + 0.22 * caustic
    elif style == "lunar_dust":
        E_sh = compute_hamid_trigonometric_fbm(X + 0.035, Y + 0.025, octaves=5)
        relief = _ee(6.0 * (E_sh - E))
        regolith = np.clip(0.32 + 0.55 * relief + 0.14 * E, 0.04, 0.98)
        B0, B1, B2 = regolith * 1.02, regolith * 0.99, regolith * 0.95
    else:  # night_sky (luminous twilight + Hamid pinprick starfield + perspective water)
        sky_grad = np.clip(0.5 * (Y + 1.0), 0.0, 1.0)
        horizon_glow = np.exp(-7.0 * ((Y + 0.44) ** 2))
        neb_wisp = 0.05 * np.clip(E + 0.3, 0.0, 1.0) * (1.0 - 0.4 * sky_grad)
        S0, S1, S2 = _eval_hamid_starfield(X, Y, num_octaves=14, t_phase=t_phase, density_scale=1.1)
        sky_mask = _ee(-45.0 * (Y + 0.44))
        water_mask = 1.0 - sky_mask

        # Perspective reflective water plane below y = -0.44
        w_depth = np.maximum(0.02, -0.44 - Y)
        persp_y = 1.8 / (w_depth + 0.07)
        ripple = np.cos(7.5 * persp_y + 2.4 * np.sin(5.5 * X + 1.8 * E) - 1.4 * t_phase)
        shimmer = np.exp(-3.2 * (X ** 2) / (0.18 + 2.0 * w_depth)) * (0.45 + 0.55 * np.clip(ripple, 0.0, 1.0))

        sky0 = 0.025 + 0.09 * horizon_glow + 0.04 * (1.0 - sky_grad) + neb_wisp + S0
        sky1 = 0.050 + 0.14 * horizon_glow + 0.08 * (1.0 - sky_grad) + 1.2 * neb_wisp + S1
        sky2 = 0.130 + 0.24 * horizon_glow + 0.16 * (1.0 - sky_grad) + 2.0 * neb_wisp + S2

        wat0 = 0.02 + 0.10 * shimmer + 0.025 * ripple * np.exp(-3.0 * w_depth)
        wat1 = 0.05 + 0.16 * shimmer + 0.040 * ripple * np.exp(-3.0 * w_depth)
        wat2 = 0.12 + 0.22 * shimmer + 0.060 * ripple * np.exp(-3.0 * w_depth)

        B0 = sky_mask * sky0 + water_mask * wat0
        B1 = sky_mask * sky1 + water_mask * wat1
        B2 = sky_mask * sky2 + water_mask * wat2

    if primary_rgb and len(primary_rgb) == 3 and secondary_rgb and len(secondary_rgb) == 3:
        mix = np.clip(0.5 * (Y + 1.0) + 0.15 * E, 0.0, 1.0)
        B0 = 0.65 * B0 + 0.35 * (float(primary_rgb[0]) * mix + float(secondary_rgb[0]) * (1.0 - mix))
        B1 = 0.65 * B1 + 0.35 * (float(primary_rgb[1]) * mix + float(secondary_rgb[1]) * (1.0 - mix))
        B2 = 0.65 * B2 + 0.35 * (float(primary_rgb[2]) * mix + float(secondary_rgb[2]) * (1.0 - mix))

    return np.clip(B0, 0.0, 1.0), np.clip(B1, 0.0, 1.0), np.clip(B2, 0.0, 1.0)


def _infer_anim_role(comp: Dict[str, Any]) -> str:
    role = (comp.get("anim_role") or "").strip().lower()
    if role in {"flap", "spin", "sway", "wave", "pulse", "orbit", "static"}:
        return role
    name_low = (comp.get("name") or "").lower()
    ctype = (comp.get("type") or "").lower()
    if any(k in name_low for k in ["wing", "plume", "feather", "fin"]):
        return "flap"
    if ctype == "wheel_rosette" or any(k in name_low for k in ["wheel", "gear", "star", "sun", "rosette"]):
        return "spin"
    if any(k in name_low for k in ["water", "pool", "wave", "coil", "ocean", "ripple", "tentacle", "tendril", "jellyfish"]):
        return "wave"
    if ctype == "crescent_lobe" or any(k in name_low for k in ["moon", "hilal", "crown", "halo", "eye", "bell", "ring"]):
        return "pulse"
    if any(k in name_low for k in ["neck", "hood", "flame", "tail", "tree", "petal", "branch", "leaf", "pine", "grass"]):
        return "sway"
    return "static"


def _resolve_component_palette(comp: Dict[str, Any], idx: int) -> Tuple[List[float], List[float], str]:
    name_low = (comp.get("name") or "").lower()
    default_palettes = [
        ([0.88, 0.72, 0.28], [0.96, 0.90, 0.65], "marble"),
        ([0.16, 0.62, 0.85], [0.42, 0.88, 0.95], "feathers"),
        ([0.85, 0.25, 0.22], [0.96, 0.68, 0.18], "petals"),
        ([0.22, 0.74, 0.48], [0.75, 0.92, 0.35], "foliage"),
        ([0.65, 0.38, 0.88], [0.95, 0.65, 0.92], "cosmic"),
    ]
    def_c1, def_c2, def_tex = default_palettes[idx % len(default_palettes)]

    if any(k in name_low for k in ["moon", "hilal", "crescent", "star", "crown", "gold", "sun", "saturn", "ring"]):
        def_c1, def_c2, def_tex = [0.94, 0.80, 0.34], [1.0, 0.96, 0.76], "metallic"
    elif any(k in name_low for k in ["jelly", "biolum", "tentacle", "nebula", "aurora", "glow", "plasma", "cosmic"]):
        def_c1, def_c2, def_tex = [0.22, 0.78, 0.98], [0.88, 0.35, 0.96], "cosmic"
    elif any(k in name_low for k in ["minaret", "roof", "dome", "gable", "marble", "arch", "mosque", "tower", "snow", "peak", "mountain", "helmet", "astronaut"]):
        def_c1, def_c2, def_tex = [0.90, 0.88, 0.85], [0.68, 0.78, 0.92], "marble"
    elif any(k in name_low for k in ["wing", "feather", "plume", "tail", "peacock", "eagle", "phoenix"]):
        def_c1, def_c2, def_tex = [0.14, 0.58, 0.76], [0.92, 0.68, 0.20], "feathers"
    elif any(k in name_low for k in ["wheel", "gear", "tube", "frame", "fork", "anchor", "shaft", "fluke", "visor"]):
        def_c1, def_c2, def_tex = [0.25, 0.68, 0.88], [0.88, 0.94, 0.98], "metallic"
    elif any(k in name_low for k in ["tiger", "cobra", "hood", "snake", "coil", "neck", "body", "fur", "lion", "cat", "fox", "wolf"]):
        def_c1, def_c2, def_tex = [0.88, 0.46, 0.12], [0.18, 0.10, 0.06], "fur"
    elif any(k in name_low for k in ["pine", "tree", "forest", "leaf", "foliage", "jungle", "bush", "grass", "palm"]):
        def_c1, def_c2, def_tex = [0.14, 0.48, 0.24], [0.42, 0.74, 0.32], "foliage"
    elif any(k in name_low for k in ["pool", "water", "wave", "ocean", "reflection", "river"]):
        def_c1, def_c2, def_tex = [0.12, 0.48, 0.75], [0.55, 0.88, 0.96], "water"

    c1 = comp.get("color_rgb", def_c1)
    c2 = comp.get("accent_rgb", def_c2)
    if not (isinstance(c1, (list, tuple)) and len(c1) == 3):
        c1 = def_c1
    if not (isinstance(c2, (list, tuple)) and len(c2) == 3):
        c2 = def_c2
    c1 = [float(np.clip(v, 0.0, 1.0)) for v in c1]
    c2 = [float(np.clip(v, 0.0, 1.0)) for v in c2]
    tex_type = str(comp.get("texture_type") or def_tex).strip().lower()
    return c1, c2, tex_type


def _parse_bounds(raw_bounds: Any) -> Tuple[float, float, float, float]:
    """Parses bounds from [xmin, xmax, ymin, ymax], [[xmin, ymin], [xmax, ymax]], or dict."""
    if isinstance(raw_bounds, dict):
        bx = raw_bounds.get("x", [-1.1, 1.1])
        by = raw_bounds.get("y", [-0.75, 0.75])
        return float(bx[0]), float(bx[1]), float(by[0]), float(by[1])
    if isinstance(raw_bounds, (list, tuple)):
        if len(raw_bounds) == 4 and all(isinstance(v, (int, float)) for v in raw_bounds):
            return float(raw_bounds[0]), float(raw_bounds[1]), float(raw_bounds[2]), float(raw_bounds[3])
        if len(raw_bounds) == 2 and all(isinstance(v, (list, tuple)) and len(v) >= 2 for v in raw_bounds):
            return float(raw_bounds[0][0]), float(raw_bounds[1][0]), float(raw_bounds[0][1]), float(raw_bounds[1][1])
    return -1.1, 1.1, -0.75, 0.75


def _eval_repeated_base_shape(
    shape: str,
    u: np.ndarray,
    v: np.ndarray,
    E_loc: np.ndarray,
    k_idx: int,
    t_phase: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Evaluates local unit-scale SDF d(u, v) and surface texture coordinates (u_par, u_perp, extra_mask)
    for Hamid-inspired repeated_field base shapes:
      "lobe", "tree", "tentacle", "feather", "petal", "sphere", "star"
    """
    sh = (shape or "lobe").strip().lower()
    if sh in ("pine", "cypress", "oak", "maple", "palm"):
        sh = "tree"
    elif sh in ("tendril", "coral", "flame", "fire", "grass", "vine", "seaweed"):
        sh = "tentacle"
    elif sh in ("plume", "wing", "quill"):
        sh = "feather"
    elif sh in ("leaf", "blossom", "flower"):
        sh = "petal"
    elif sh in ("orb", "bubble", "berry", "fruit", "planet", "pearl", "droplet"):
        sh = "sphere"
    elif sh in ("spark", "snowflake", "crystal", "sparkle"):
        sh = "star"

    kf = float(k_idx + 1)

    if sh == "tree":
        # Trunk from v = -1.0 to +0.55 + multi-tier pine/autumn foliage crown
        trunk_w = 0.11 * np.clip(1.0 - 0.55 * (v + 0.9), 0.25, 1.15)
        d_trunk = np.maximum(np.abs(u + 0.04 * np.sin(3.5 * v + kf)) - trunk_w, np.maximum(-1.0 - v, v - 0.52))
        # Tapered multi-tiered canopy silhouette with fBm noise edge (fits pines, maples, blossoms!)
        tier_wave = 0.18 * np.cos(10.0 * v + 1.4 * kf)
        crown_w = np.clip(0.88 * (1.0 - 0.68 * np.clip((v + 0.45) / 1.55, 0.0, 1.1)) + tier_wave, 0.08, 1.15)
        d_crown = ((u / np.maximum(crown_w, 0.08)) ** 2) + (((v - 0.22) / 0.84) ** 2) - 1.0 + 0.26 * E_loc
        d_crown = d_crown * 0.48
        d_total = np.minimum(d_trunk, d_crown)
        is_trunk = _ee(60.0 * (d_trunk - d_crown + 0.01))
        u_par = np.clip(0.5 * (v + 1.0), 0.0, 1.0)
        u_perp = np.clip(u, -1.0, 1.0)
        return d_total, u_par, u_perp, is_trunk

    if sh == "tentacle":
        # Undulating glowing jellyfish tentacle / coral polyp / swaying flame
        sway = 0.26 * np.sin(3.8 * v + 1.7 * kf - t_phase) + 0.08 * np.cos(7.2 * v + kf)
        u_bent = u - sway * np.clip(0.5 * (1.0 - v), 0.0, 1.2)
        taper = 0.20 * np.clip(0.35 + 0.65 * (0.5 * (v + 1.0)), 0.18, 1.0)
        d_tube = np.abs(u_bent) - taper
        d_ends = np.maximum(-1.05 - v, v - 1.0)
        d_total = np.maximum(d_tube, d_ends)
        u_par = np.clip(0.5 * (1.0 - v), 0.0, 1.0)
        u_perp = np.clip(u_bent / np.maximum(taper, 0.05), -1.2, 1.2)
        return d_total, u_par, u_perp, None

    if sh == "feather":
        # Hamid Peacock-inspired feather with herringbone barbs & slender central rachis shaft
        v_shift = v - 0.1
        d_vane = ((u / 0.38) ** 2) + ((v_shift / 0.96) ** 2) - 1.0
        barb_wave = np.cos(34.0 * v_shift + 38.0 * np.abs(u) - 36.0 * (u ** 2) + 2.5 * np.sin(9.0 * v_shift + kf))
        d_total = d_vane * 0.42 + 0.035 * (1.0 - barb_wave)
        shaft = _ee(140.0 * (np.abs(u) - 0.025)) * _ee(40.0 * (np.abs(v_shift) - 0.90))
        u_par = np.clip(0.5 * (v + 1.0), 0.0, 1.0)
        u_perp = np.clip(u / 0.42, -1.2, 1.2)
        return d_total, u_par, u_perp, shaft

    if sh == "petal":
        # Tapered botanical petal / blossom / leaf with delicate venation
        w_v = 0.54 * np.sin(np.clip(0.5 * (v + 1.0), 0.01, 0.99) * math.pi) ** 0.72
        d_total = (np.hypot(u / np.maximum(w_v, 0.06), (v - 0.04) / 0.96) - 1.0) * 0.44 + 0.04 * E_loc
        u_par = np.clip(0.5 * (v + 1.0), 0.0, 1.0)
        u_perp = np.clip(u / np.maximum(w_v, 0.08), -1.2, 1.2)
        return d_total, u_par, u_perp, None

    if sh == "sphere":
        r = np.hypot(u, v)
        d_total = r - 0.92
        u_par = np.clip(r / 0.92, 0.0, 1.2)
        u_perp = np.arctan2(v, u) / math.pi
        return d_total, u_par, u_perp, None

    if sh == "star":
        r = np.hypot(u, v)
        th = np.arctan2(v, u)
        spikes = 0.24 + 0.76 * (np.abs(np.cos(2.0 * th)) ** 5 + 0.45 * (np.abs(np.cos(2.0 * th + 0.785)) ** 6))
        d_total = (r - np.minimum(spikes, 1.05)) * 0.65
        u_par = np.clip(r, 0.0, 1.2)
        u_perp = th / math.pi
        return d_total, u_par, u_perp, None

    # Default "lobe": organic cloud / coral / fruit / mountain ridge lobe
    r = np.hypot(u, v)
    th = np.arctan2(v, u)
    r_lobe = 0.88 + 0.14 * np.cos(3.0 * th + kf) + 0.08 * np.sin(5.0 * th - 2.0 * kf) + 0.12 * E_loc
    d_total = (r - r_lobe) * 0.72
    u_par = np.clip(r / np.maximum(r_lobe, 0.15), 0.0, 1.2)
    u_perp = th / math.pi
    return d_total, u_par, u_perp, None


def evaluate_repeated_field_layer(
    comp: Dict[str, Any],
    idx: int,
    X: np.ndarray,
    Y: np.ndarray,
    E: np.ndarray,
    t_phase: float = 0.0,
    bg_style: str = "golden_bokeh",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Vectorized Hamid-inspired multi-instance Porter-Duff generator for
    comp["type"] == "repeated_field" supporting 4 spatial layouts:
      - perspective_corridor (with atmospheric depth haze)
      - radial_fan (concentric staggered rings with iridescent shifts)
      - scattered_cluster (front-to-back Porter-Duff 3D shading + contact shadow)
      - wave_array (swaying translucent tendrils / flames / ridges animated by t_phase)
    Uses local pixel bounding-box slicing for blazing-fast rendering even at count=60.
    Returns (alpha_total, premult_W0, premult_W1, premult_W2).
    """
    H, W = X.shape
    layout = str(comp.get("layout", "scattered_cluster")).strip().lower()
    base_shape = str(comp.get("base_shape", "lobe")).strip().lower()
    count = max(3, min(64, int(comp.get("count", 18))))
    xmin, xmax, ymin, ymax = _parse_bounds(comp.get("bounds"))
    sr = comp.get("scale_range", [0.12, 0.30])
    if isinstance(sr, (list, tuple)) and len(sr) >= 2:
        s_min, s_max = max(0.03, float(sr[0])), max(0.05, float(sr[1]))
    elif isinstance(sr, (int, float)):
        s_min = s_max = max(0.04, float(sr))
    else:
        s_min, s_max = 0.12, 0.30
    if s_min > s_max:
        s_min, s_max = s_max, s_min

    c1, c2, tex_type = _resolve_component_palette(comp, idx)
    K_s = float(comp.get("edge_sharpness", 75.0))
    seed_off = float(comp.get("seed", idx * 3 + 1))

    # Build instance list: (cx, cy, sx, sy, rot_rad, depth_01, color_mix)
    # Ordered FRONT-TO-BACK so Porter-Duff transmittance naturally occludes distant instances!
    instances: List[Tuple[float, float, float, float, float, float, float]] = []

    if layout == "perspective_corridor":
        horizon_y = float(comp.get("horizon_y", min(ymax, max(ymin, ymin + 0.58 * (ymax - ymin)))))
        center_x = 0.5 * (xmin + xmax)
        half_w = max(0.35, 0.5 * (xmax - xmin))
        for k in range(count):
            depth = float(k) / max(1.0, float(count - 1))  # 0=front, 1=far
            side = -1.0 if (k % 2 == 0) else 1.0
            persp = 1.0 / (1.0 + 2.5 * depth)
            jitter_x = 0.14 * math.sin(5.0 * (k + seed_off) ** 2)
            cx = center_x + side * (0.28 + 0.62 * abs(math.cos(2.3 * k + seed_off))) * half_w * persp + jitter_x * persp
            cy = ymin + (horizon_y - ymin) * (1.0 - persp) + 0.05 * math.sin(7.0 * k + seed_off) * persp
            sc = (s_min + (s_max - s_min) * persp) * (0.85 + 0.30 * abs(math.sin(3.1 * k + seed_off)))
            sx = sc * (0.72 if base_shape == "tree" else 0.95)
            sy = sc * (1.35 if base_shape == "tree" else 1.05)
            rot = 0.05 * math.sin(4.0 * k + t_phase)
            cmix = 0.5 + 0.5 * math.sin(3.7 * k + seed_off)
            instances.append((cx, cy, sx, sy, rot, depth, cmix))

    elif layout == "radial_fan":
        rc = comp.get("center", [0.5 * (xmin + xmax), 0.5 * (ymin + ymax)])
        rcx, rcy = float(rc[0]), float(rc[1])
        arc = comp.get("radial_arc_deg")
        if not (isinstance(arc, (list, tuple)) and len(arc) >= 2):
            arc = [-18.0, 198.0] if base_shape == "feather" else [0.0, 360.0]
        a0, a1 = float(arc[0]), float(arc[1])
        full_circle = abs((a1 - a0) - 360.0) < 5.0
        rings = 3 if count >= 15 else (2 if count >= 8 else 1)
        per_ring = max(3, int(math.ceil(count / rings)))
        span_r = max(0.20, 0.42 * min(xmax - xmin, ymax - ymin))
        idx_k = 0
        for r_idx in range(rings):
            # Inner ring first (front), outer rings behind (back)
            ring_frac = (r_idx + 1.0) / float(rings)
            n_this = min(per_ring, count - idx_k)
            if n_this <= 0:
                break
            stagger = (0.5 / max(1, n_this)) * (a1 - a0) if (r_idx % 2 == 1) else 0.0
            for j in range(n_this):
                frac = float(j) / max(1.0, float(n_this if full_circle else max(1, n_this - 1)))
                ang_deg = a0 + frac * (a1 - a0) + stagger
                ang_rad = math.radians(ang_deg) + 0.06 * math.sin(t_phase + r_idx)
                rad_dist = span_r * (0.28 + 0.72 * ring_frac)
                cx = rcx + rad_dist * math.cos(ang_rad)
                cy = rcy + rad_dist * math.sin(ang_rad)
                sc = s_min + (s_max - s_min) * (0.55 + 0.45 * ring_frac)
                sx = sc * (0.65 if base_shape in ("feather", "petal", "tentacle") else 0.90)
                sy = sc * (1.28 if base_shape in ("feather", "petal", "tentacle") else 0.95)
                rot = ang_rad - math.pi / 2.0
                cmix = 0.5 + 0.5 * math.sin(2.5 * r_idx + 1.3 * j + seed_off)
                instances.append((cx, cy, sx, sy, rot, ring_frac * 0.65, cmix))
                idx_k += 1

    elif layout == "wave_array":
        for k in range(count):
            frac = float(k) / max(1.0, float(count - 1))
            depth = abs(2.0 * frac - 1.0) * 0.55 + 0.15 * (k % 3)
            cx = xmin + frac * (xmax - xmin) + 0.04 * math.sin(5.0 * k + seed_off)
            wave_y = 0.16 * (ymax - ymin) * math.sin(4.2 * frac + 0.8 * seed_off - t_phase)
            cy = 0.5 * (ymin + ymax) + wave_y + 0.12 * (ymax - ymin) * math.cos(7.0 * k)
            sc = s_min + (s_max - s_min) * (0.45 + 0.55 * (1.0 - abs(2.0 * frac - 1.0) * 0.5))
            if base_shape == "tentacle":
                sx = sc * 0.55
                sy = sc * 1.95
            else:
                sx = sc * 0.95
                sy = sc * 1.15
            rot = 0.16 * math.sin(3.0 * frac + t_phase + k * 0.5)
            cmix = 0.5 + 0.5 * math.sin(2.1 * k + seed_off)
            instances.append((cx, cy, sx, sy, rot, depth, cmix))
        instances.sort(key=lambda it: it[5])

    else:  # scattered_cluster (golden-ratio low-discrepancy + Hamid trigonometric jitter)
        golden = 0.61803398875
        for k in range(count):
            kf = float(k + 1)
            depth = float(k) / max(1.0, float(count - 1))
            gx = (kf * golden + 0.19 * math.sin(5.0 * kf * kf + seed_off)) % 1.0
            gy = ((kf + 0.5) / float(count) + 0.16 * math.cos(7.0 * kf * kf + seed_off)) % 1.0
            cx = xmin + gx * (xmax - xmin)
            cy = ymin + gy * (ymax - ymin)
            sc = s_min + (s_max - s_min) * (1.0 - 0.45 * depth) * (0.8 + 0.4 * abs(math.sin(4.0 * kf + seed_off)))
            sx = sc * (0.72 if base_shape == "tree" else (0.85 + 0.25 * math.cos(3.0 * kf)))
            sy = sc * (1.35 if base_shape == "tree" else (0.85 + 0.25 * math.sin(3.0 * kf)))
            if base_shape == "tree":
                rot = 0.07 * math.sin(4.0 * kf + t_phase)
            else:
                rot = math.pi * math.sin(9.0 * kf * kf + seed_off) + 0.12 * math.sin(t_phase + kf)
            cmix = 0.5 + 0.5 * math.sin(5.0 * kf + seed_off)
            instances.append((cx, cy, sx, sy, rot, depth, cmix))

    # Map world X, Y back to pixel indices for fast bounding-box slicing
    x0_val, x1_val = float(X[0, 0]), float(X[0, -1])
    y0_val, y1_val = float(Y[0, 0]), float(Y[-1, 0])
    dx_pix = (x1_val - x0_val) / max(1, W - 1)
    dy_pix = (y1_val - y0_val) / max(1, H - 1)  # negative since Y decreases downward

    W0 = np.zeros_like(X, dtype=np.float32)
    W1 = np.zeros_like(X, dtype=np.float32)
    W2 = np.zeros_like(X, dtype=np.float32)
    local_trans = np.ones_like(X, dtype=np.float32)
    contact_ao = np.ones_like(X, dtype=np.float32)

    haze_rgb = (
        [0.72, 0.70, 0.84] if bg_style in ("sunset_sky", "alpine_sunset")
        else [0.06, 0.26, 0.38] if bg_style == "deep_ocean"
        else [0.10, 0.14, 0.26] if bg_style in ("night_sky", "cosmic_nebula")
        else [0.58, 0.64, 0.46]
    )

    is_luminous = tex_type in ("cosmic", "water") or base_shape in ("star", "tentacle")

    for k, (cx, cy, sx, sy, rot, depth, cmix) in enumerate(instances):
        rad_bb = 1.45 * max(sx, sy)
        ix0 = max(0, int(math.floor((cx - rad_bb - x0_val) / dx_pix)))
        ix1 = min(W, int(math.ceil((cx + rad_bb - x0_val) / dx_pix)) + 1)
        iy0 = max(0, int(math.floor((cy + rad_bb - y0_val) / dy_pix)))
        iy1 = min(H, int(math.ceil((cy - rad_bb - y0_val) / dy_pix)) + 1)
        if ix1 <= ix0 + 1 or iy1 <= iy0 + 1:
            continue

        Xs = X[iy0:iy1, ix0:ix1]
        Ys = Y[iy0:iy1, ix0:ix1]
        Es = E[iy0:iy1, ix0:ix1]

        ca, sa = math.cos(rot), math.sin(rot)
        dx_s = Xs - cx
        dy_s = Ys - cy
        u_loc = (ca * dx_s + sa * dy_s) / max(sx, 0.015)
        v_loc = (-sa * dx_s + ca * dy_s) / max(sy, 0.015)

        d_loc, u_par, u_perp, extra_mask = _eval_repeated_base_shape(base_shape, u_loc, v_loc, Es, k, t_phase)
        d_world = d_loc * min(sx, sy)

        # Double-exponential Hamid mask J_k
        J_k = _ee(K_s * d_world)
        if float(np.max(J_k)) < 1e-3:
            continue

        # Fast 3D normal & directional self-shadowing within local slice
        w_eff = max(0.02, 0.45 * min(sx, sy))
        rho = np.clip(-d_world / w_eff, 0.0, 1.0)
        Nz = np.sqrt(np.clip(2.0 * rho - rho * rho, 0.05, 1.0))
        gy, gx = np.gradient(d_world)
        gnorm = np.hypot(gx, gy) + 1e-6
        xy_sc = np.sqrt(np.maximum(0.0, 1.0 - Nz * Nz))
        Nx = (-gx / gnorm) * xy_sc
        Ny = (-gy / gnorm) * xy_sc
        diff = np.clip(-0.40 * Nx + 0.60 * Ny + 0.68 * Nz, 0.0, 1.0)
        spec = (0.42 if tex_type in ("metallic", "petals", "water", "cosmic") else 0.18) * (
            np.clip(0.25 * Nx + 0.45 * Ny + 0.85 * Nz, 0.0, 1.0) ** 16.0
        )
        fresnel = ((1.0 - Nz) ** 2.2) if is_luminous else 0.0

        # Instance color mix + texture detail
        inst_mix = np.clip(0.35 * cmix + 0.35 * u_par + 0.30 * (0.5 + 0.5 * np.sin(5.0 * Es + 3.0 * v_loc)), 0.0, 1.0)
        r_c = c1[0] * (1.0 - inst_mix) + c2[0] * inst_mix
        g_c = c1[1] * (1.0 - inst_mix) + c2[1] * inst_mix
        b_c = c1[2] * (1.0 - inst_mix) + c2[2] * inst_mix

        if base_shape == "tree" and extra_mask is not None:
            # Dark furrowed bark trunk + organic layered evergreen/autumn boughs (no checkerboard!)
            bark_shade = 0.26 + 0.36 * diff
            bough_wave = 0.78 + 0.22 * np.sin(18.0 * (v_loc + 0.55 * np.abs(u_loc)) + 3.2 * Es)
            crown_sun = 0.56 + 0.44 * np.clip(0.55 + 0.45 * v_loc - 0.30 * u_loc + 0.25 * Es, 0.0, 1.0)
            foliage_lit = (0.42 + 0.58 * diff) * bough_wave * crown_sun
            r_c = extra_mask * (0.22 * bark_shade) + (1.0 - extra_mask) * (r_c * foliage_lit)
            g_c = extra_mask * (0.14 * bark_shade) + (1.0 - extra_mask) * (g_c * foliage_lit)
            b_c = extra_mask * (0.09 * bark_shade) + (1.0 - extra_mask) * (b_c * foliage_lit)
        elif base_shape == "feather" and extra_mask is not None:
            # Glowing central rachis shaft + iridescent tip ocellus
            tip_d = np.hypot(u_loc, (v_loc - 0.52) * 1.15)
            oc_ring = _ee(35.0 * (np.abs(tip_d - 0.16) - 0.04))
            oc_core = _ee(40.0 * (tip_d - 0.11))
            shade_f = 0.42 + 0.58 * diff
            r_c = r_c * shade_f * (1.0 - oc_core) + 0.72 * extra_mask + 0.15 * oc_ring + 0.04 * oc_core
            g_c = g_c * shade_f * (1.0 - oc_core) + 0.78 * extra_mask + 0.90 * oc_ring + 0.08 * oc_core
            b_c = b_c * shade_f * (1.0 - oc_core) + 0.55 * extra_mask + 0.94 * oc_ring + 0.28 * oc_core
        elif base_shape == "tentacle":
            # Hamid Corals-inspired translucent glowing rim highlight!
            rim_glow = 0.85 * _ee(65.0 * d_world) * _ee(-8.0 * (d_world + 0.30 * w_eff))
            shade_t = 0.48 + 0.52 * diff
            r_c = r_c * shade_t + (0.45 + 0.55 * c2[0]) * (rim_glow + 0.45 * fresnel) + spec
            g_c = g_c * shade_t + (0.45 + 0.55 * c2[1]) * (rim_glow + 0.45 * fresnel) + spec
            b_c = b_c * shade_t + (0.45 + 0.55 * c2[2]) * (rim_glow + 0.45 * fresnel) + spec
        elif base_shape == "star":
            core_glow = np.exp(-5.0 * (u_loc * u_loc + v_loc * v_loc))
            r_c = r_c * 0.65 + 0.95 * core_glow
            g_c = g_c * 0.65 + 0.95 * core_glow
            b_c = b_c * 0.65 + 0.98 * core_glow
        else:
            shade_o = 0.36 + 0.64 * diff
            r_c = r_c * shade_o + spec + 0.30 * fresnel * c2[0]
            g_c = g_c * shade_o + spec + 0.30 * fresnel * c2[1]
            b_c = b_c * shade_o + spec + 0.30 * fresnel * c2[2]

        # Apply contact AO shadow from foreground instances + atmospheric perspective haze
        ao_s = contact_ao[iy0:iy1, ix0:ix1]
        r_c = r_c * ao_s
        g_c = g_c * ao_s
        b_c = b_c * ao_s

        if layout == "perspective_corridor":
            haze = float(np.clip(0.55 * (depth ** 1.3), 0.0, 0.58))
            r_c = r_c * (1.0 - haze) + haze_rgb[0] * haze
            g_c = g_c * (1.0 - haze) + haze_rgb[1] * haze
            b_c = b_c * (1.0 - haze) + haze_rgb[2] * haze

        # Porter-Duff front-to-back compositing in local slice
        t_s = local_trans[iy0:iy1, ix0:ix1]
        opacity = J_k * (0.82 if base_shape == "tentacle" else 1.0)
        w_k = t_s * opacity
        W0[iy0:iy1, ix0:ix1] += w_k * np.clip(r_c, 0.0, 1.15)
        W1[iy0:iy1, ix0:ix1] += w_k * np.clip(g_c, 0.0, 1.15)
        W2[iy0:iy1, ix0:ix1] += w_k * np.clip(b_c, 0.0, 1.15)
        local_trans[iy0:iy1, ix0:ix1] = t_s * (1.0 - opacity)

        # Soft contact shadow for deeper overlapping instances (Hamid Strawberries A_4 trick!)
        soft_shadow = _ee(14.0 * (d_world - 0.015))
        contact_ao[iy0:iy1, ix0:ix1] = np.clip(ao_s * (1.0 - 0.32 * soft_shadow), 0.42, 1.0)

    alpha_total = np.clip(1.0 - local_trans, 0.0, 1.0)
    return alpha_total, W0, W1, W2


def evaluate_implicit_sdf_layer(
    comp: Dict[str, Any],
    idx: int,
    X: np.ndarray,
    Y: np.ndarray,
    E: np.ndarray,
    t_phase: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Evaluates comp["type"] == "implicit_sdf_expr" via sandboxed AST math evaluation
    with Hamid Moon-Footprint-inspired directional finite-difference 3D relief
    self-shadowing: e^(-e^(8*(d_far - d_0))) * (1 + 0.25*e^(-e^(20*(d_near - d_0)))).
    Returns (J_s, C0, C1, C2).
    """
    expr = str(comp.get("sdf_expr") or comp.get("expr") or "sqrt(x**2 + y**2) - 0.35 + 0.06*E")
    cx0, cy0 = comp.get("center", [0.0, 0.0])
    sc = comp.get("scale", [1.0, 1.0])
    if isinstance(sc, (int, float)):
        sx = sy = max(0.05, float(sc))
    else:
        sx, sy = max(0.05, float(sc[0])), max(0.05, float(sc[1]))
    rot = math.radians(float(comp.get("rotation_deg", 0.0)))
    if abs(t_phase) > 1e-6:
        role = _infer_anim_role(comp)
        if role == "spin":
            rot += t_phase
        elif role == "pulse":
            p_sc = 1.0 + 0.07 * math.sin(t_phase)
            sx *= p_sc
            sy *= p_sc
        elif role == "sway":
            cx0 = float(cx0) + 0.04 * math.sin(t_phase)

    ca, sa = math.cos(rot), math.sin(rot)
    dx = X - float(cx0)
    dy = Y - float(cy0)
    x_loc = (ca * dx + sa * dy) / sx
    y_loc = (-sa * dx + ca * dy) / sy

    def _eval_at(xl: np.ndarray, yl: np.ndarray, el: np.ndarray) -> np.ndarray:
        if _ai_safe_eval_sdf_expr is not None and "r" not in expr and "theta" not in expr and "t" not in expr:
            try:
                return _ai_safe_eval_sdf_expr(expr, xl, yl, el).astype(np.float32)
            except Exception:
                pass
        r_l = np.hypot(xl, yl)
        th_l = np.arctan2(yl, xl)
        env = {"x": xl, "y": yl, "E": el, "r": r_l, "theta": th_l, "t": float(t_phase)}
        return _safe_eval_math_expr(expr, env)

    try:
        d_0 = _eval_at(x_loc, y_loc, E) * min(sx, sy)
        # Directional finite-difference relief self-shadowing (inspired by Moon Footprint A_5, A_20)
        d_near = _eval_at(x_loc + 0.012 / sx, y_loc + 0.015 / sy, E) * min(sx, sy)
        d_far = _eval_at(x_loc + 0.042 / sx, y_loc + 0.052 / sy, E) * min(sx, sy)
    except Exception:
        # Graceful fallback so malformed expressions never crash rendering
        r_fb = np.hypot(x_loc, y_loc)
        d_0 = (r_fb - 0.35 + 0.05 * E) * min(sx, sy)
        d_near = (np.hypot(x_loc + 0.012, y_loc + 0.015) - 0.35 + 0.05 * E) * min(sx, sy)
        d_far = (np.hypot(x_loc + 0.042, y_loc + 0.052) - 0.35 + 0.05 * E) * min(sx, sy)

    r_norm = np.hypot(x_loc, y_loc)
    u_par = np.clip(r_norm, 0.0, 1.5)
    u_perp = np.arctan2(y_loc, x_loc) / math.pi
    w_eff = max(0.04, 0.25 * min(sx, sy))

    # Shade strictly from pure smooth d_0 so bounding box never creates artificial 45-deg box bevels
    J_s, C0, C1, C2 = shade_component_layer(comp, idx, d_0, u_par, u_perp, w_eff, X, Y, E)

    # Apply soft spatial window ONLY for true sub-region bounds (do not clip full-canvas [-1, 1] horizons)
    if comp.get("bounds") is not None:
        bxmin, bxmax, bymin, bymax = _parse_bounds(comp.get("bounds"))
        win = np.ones_like(X, dtype=np.float32)
        if bxmin > -0.92:
            win *= _ee(22.0 * (bxmin - X))
        if bxmax < 0.92:
            win *= _ee(22.0 * (X - bxmax))
        if bymin > -0.92:
            win *= _ee(22.0 * (bymin - Y))
        if bymax < 0.92:
            win *= _ee(22.0 * (Y - bymax))
        J_s = J_s * win

    # Blend Hamid Moon-style directional self-shadowing relief onto shaded surface
    use_relief = bool(comp.get("relief_shadow", True))
    relief_strength = float(np.clip(comp.get("relief_strength", 0.38 if use_relief else 0.0), 0.0, 0.85))
    if relief_strength > 1e-4:
        self_shadow = _ee(14.0 * (d_far - d_0)) * (0.82 + 0.28 * _ee(28.0 * (d_near - d_0)))
        relief_mult = (1.0 - relief_strength) + relief_strength * np.clip(self_shadow, 0.25, 1.20)
        C0, C1, C2 = C0 * relief_mult, C1 * relief_mult, C2 * relief_mult
    return J_s, np.clip(C0, 0.0, 1.0), np.clip(C1, 0.0, 1.0), np.clip(C2, 0.0, 1.0)


def evaluate_primitive_level_set(
    comp: Dict[str, Any],
    X: np.ndarray,
    Y: np.ndarray,
    t_phase: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    ctype = comp.get("type", "ribbon_spine")
    if abs(t_phase) > 1e-6:
        role = _infer_anim_role(comp)
        if role == "sway":
            X = X - 0.045 * math.sin(t_phase) * np.clip(Y + 0.5, 0.0, 1.5)
        elif role == "wave":
            Y = Y - 0.025 * np.sin(5.0 * X - t_phase)
        elif role == "orbit":
            X = X - 0.05 * math.cos(t_phase)
            Y = Y - 0.05 * math.sin(t_phase)
        elif role == "flap":
            ang_f = 0.18 * math.sin(t_phase) * np.sign(X + 1e-6)
            ca_f, sa_f = np.cos(ang_f), np.sin(ang_f)
            X, Y = ca_f * X - sa_f * Y, sa_f * X + ca_f * Y
        elif role == "pulse":
            s_p = 1.0 / (1.0 + 0.07 * math.sin(t_phase))
            X, Y = X * s_p, Y * s_p
        elif role == "spin":
            cx_s, cy_s = comp.get("center", [0.0, 0.0])
            ca_s, sa_s = math.cos(t_phase), math.sin(t_phase)
            dx_s, dy_s = X - float(cx_s), Y - float(cy_s)
            X = float(cx_s) + ca_s * dx_s + sa_s * dy_s
            Y = float(cy_s) - sa_s * dx_s + ca_s * dy_s

    if ctype == "ribbon_spine":
        raw_pts = comp.get("points", [[-0.4, 0.0], [0.0, 0.2], [0.4, 0.0]])
        if not raw_pts or len(raw_pts) < 2:
            raw_pts = [[-0.4, 0.0], [0.4, 0.0]]
        pts = np.asarray(raw_pts, dtype=np.float32)
        K = len(pts) - 1
        L = 32
        u_s = np.linspace(0.0, 1.0, L, dtype=np.float32)
        cx = np.zeros(L, dtype=np.float32)
        cy = np.zeros(L, dtype=np.float32)
        for j in range(K + 1):
            bern = math.comb(K, j) * ((1.0 - u_s) ** (K - j)) * (u_s ** j)
            cx += bern * pts[j, 0]
            cy += bern * pts[j, 1]

        dx = np.gradient(cx)
        dy = np.gradient(cy)
        norm = np.hypot(dx, dy) + 1e-8
        nx, ny = -dy / norm, dx / norm

        w_start = float(comp.get("w_start", 0.06))
        w_end = float(comp.get("w_end", 0.02))
        w_bulge = float(comp.get("w_bulge", 0.0))
        w_mod_amp = float(comp.get("w_mod_amp", 0.0))
        w_mod_freq = float(comp.get("w_mod_freq", 4.0))
        w_s = np.maximum(
            0.006,
            w_start * (1.0 - u_s)
            + w_end * u_s
            + w_bulge * np.sin(np.pi * u_s)
            + w_mod_amp * np.cos(2.0 * np.pi * w_mod_freq * u_s),
        )
        w_eff = float(np.mean(w_s))

        d_best = np.full_like(X, 1e6, dtype=np.float32)
        u_par_best = np.zeros_like(X, dtype=np.float32)
        u_perp_best = np.zeros_like(X, dtype=np.float32)

        for k in range(L - 1):
            ax, ay = cx[k], cy[k]
            bx, by = cx[k + 1], cy[k + 1]
            abx, aby = bx - ax, by - ay
            len2 = abx * abx + aby * aby + 1e-10
            tau = np.clip(((X - ax) * abx + (Y - ay) * aby) / len2, 0.0, 1.0)
            qx = ax + tau * abx
            qy = ay + tau * aby
            rx, ry = X - qx, Y - qy
            dist = np.hypot(rx, ry)
            wk = (1.0 - tau) * w_s[k] + tau * w_s[k + 1]
            d_k = dist - wk
            mask = d_k < d_best
            if np.any(mask):
                d_best = np.where(mask, d_k, d_best)
                u_par_best = np.where(mask, (k + tau) / (L - 1.0), u_par_best)
                nk_x = (1.0 - tau) * nx[k] + tau * nx[k + 1]
                nk_y = (1.0 - tau) * ny[k] + tau * ny[k + 1]
                signed_perp = (rx * nk_x + ry * nk_y) / (wk + 1e-5)
                u_perp_best = np.where(mask, np.clip(signed_perp, -1.5, 1.5), u_perp_best)

        return d_best, u_par_best, u_perp_best, w_eff

    elif ctype == "parametric_equation":
        cx0, cy0 = comp.get("center", [0.0, 0.0])
        sx, sy = comp.get("scale", [0.4, 0.4])
        sx, sy = max(0.03, float(sx)), max(0.03, float(sy))
        L = 36
        t_arr = np.linspace(0.0, 2.0 * math.pi, L, dtype=np.float32)
        env_t = {"t": t_arr, "x": t_arr}
        try:
            x1 = float(cx0) + sx * _safe_eval_math_expr(str(comp.get("x1_expr", "cos(t)")), env_t)
            y1 = float(cy0) + sy * _safe_eval_math_expr(str(comp.get("y1_expr", "sin(t)")), env_t)
            x2 = float(cx0) + sx * _safe_eval_math_expr(str(comp.get("x2_expr", "0.75*cos(t)")), env_t)
            y2 = float(cy0) + sy * _safe_eval_math_expr(str(comp.get("y2_expr", "0.75*sin(t)")), env_t)
        except Exception:
            x1 = float(cx0) + sx * np.cos(t_arr)
            y1 = float(cy0) + sy * np.sin(t_arr)
            x2 = float(cx0) + 0.72 * sx * np.cos(t_arr)
            y2 = float(cy0) + 0.72 * sy * np.sin(t_arr)

        cx = 0.5 * (x1 + x2)
        cy = 0.5 * (y1 + y2)
        w_s = np.maximum(0.015, 0.5 * np.hypot(x1 - x2, y1 - y2))
        w_eff = float(np.mean(w_s))

        # If a closed parametric curve represents a solid disc/sun/planet/helmet/head, fill its interior
        name_low = (comp.get("name") or "").lower()
        is_closed = float(np.hypot(x1[0] - x1[-1], y1[0] - y1[-1])) < 0.22 * max(sx, sy)
        is_ring_like = any(k in name_low for k in ("ring", "halo", "hoop", "orbit", "torus", "wheel", "arch", "crescent"))
        is_solid_like = any(k in name_low for k in ("sun", "moon", "planet", "helmet", "head", "eye", "body", "disc", "orb", "bell", "core", "sphere", "face"))
        if is_closed and is_solid_like and not is_ring_like:
            r_outer = float(np.mean(np.hypot(x1 - float(cx0), y1 - float(cy0))))
            dx_p = (X - float(cx0)) / max(sx, 0.03)
            dy_p = (Y - float(cy0)) / max(sy, 0.03)
            r_p = np.hypot(dx_p, dy_p)
            r_ref = max(0.25, r_outer / max(min(sx, sy), 0.03))
            d_solid = (r_p - r_ref) * min(sx, sy)
            u_par_s = np.clip(r_p / r_ref, 0.0, 1.5)
            u_perp_s = np.arctan2(dy_p, dx_p) / math.pi
            return d_solid, u_par_s, u_perp_s, max(0.05, 0.5 * r_outer)

        d_best = np.full_like(X, 1e6, dtype=np.float32)
        u_par_best = np.zeros_like(X, dtype=np.float32)
        u_perp_best = np.zeros_like(X, dtype=np.float32)
        for k in range(L - 1):
            ax, ay = cx[k], cy[k]
            bx, by = cx[k + 1], cy[k + 1]
            abx, aby = bx - ax, by - ay
            len2 = abx * abx + aby * aby + 1e-10
            tau = np.clip(((X - ax) * abx + (Y - ay) * aby) / len2, 0.0, 1.0)
            rx, ry = X - (ax + tau * abx), Y - (ay + tau * aby)
            dist = np.hypot(rx, ry)
            wk = (1.0 - tau) * w_s[k] + tau * w_s[k + 1]
            d_k = dist - wk
            mask = d_k < d_best
            if np.any(mask):
                d_best = np.where(mask, d_k, d_best)
                u_par_best = np.where(mask, (k + tau) / (L - 1.0), u_par_best)
                u_perp_best = np.where(mask, np.clip(dist / (wk + 1e-5) - 0.5, -1.0, 1.0), u_perp_best)
        return d_best, u_par_best, u_perp_best, w_eff

    elif ctype == "polar_harmonic":
        cx0, cy0 = comp.get("center", [0.0, 0.0])
        sx, sy = comp.get("scale", [0.35, 0.35])
        sx, sy = max(0.02, float(sx)), max(0.02, float(sy))
        rot = np.deg2rad(float(comp.get("rotation_deg", 0.0)))
        ca, sa = math.cos(rot), math.sin(rot)
        dx, dy = X - float(cx0), Y - float(cy0)
        x_loc = (ca * dx + sa * dy) / sx
        y_loc = (-sa * dx + ca * dy) / sy
        r = np.hypot(x_loc, y_loc)
        theta = np.arctan2(y_loc, x_loc)

        R_th = np.full_like(theta, float(comp.get("r0", 1.0)))
        for h in comp.get("harmonics", []):
            m = float(h.get("m", 1.0))
            p = float(h.get("power", 1.0))
            c_amp = float(h.get("cos_amp", 0.0))
            s_amp = float(h.get("sin_amp", 0.0))
            ac_amp = float(h.get("abs_cos_amp", 0.0))
            if abs(c_amp) > 1e-4:
                R_th += c_amp * (np.cos(m * theta) ** int(max(1, round(p))))
            if abs(s_amp) > 1e-4:
                R_th += s_amp * (np.sin(m * theta) ** int(max(1, round(p))))
            if abs(ac_amp) > 1e-4:
                R_th += ac_amp * (np.abs(np.cos(m * theta)) ** p)
        R_th = np.maximum(0.05, R_th)

        scale_min = min(sx, sy)
        d_s = (r - R_th) * scale_min

        t_deg = comp.get("theta_range_deg", [0.0, 360.0])
        if isinstance(t_deg, list) and len(t_deg) >= 2 and abs(float(t_deg[1]) - float(t_deg[0])) < 355.0:
            th0 = np.deg2rad(float(t_deg[0]))
            th1 = np.deg2rad(float(t_deg[1]))
            th_mid = 0.5 * (th0 + th1)
            half_span = 0.5 * abs(th1 - th0)
            ang_diff = np.abs(np.arctan2(np.sin(theta - th_mid), np.cos(theta - th_mid)))
            d_ang = (ang_diff - half_span) * r * scale_min
            d_s = np.maximum(d_s, d_ang)

        u_par = np.clip(r / (R_th + 1e-5), 0.0, 1.5)
        u_perp = theta / math.pi
        return d_s, u_par, u_perp, scale_min * float(np.mean(R_th)) * 0.5

    elif ctype == "crescent_lobe":
        cx0, cy0 = comp.get("center", [0.0, 0.0])
        R = max(0.03, float(comp.get("outer_radius", 0.25)))
        w_belly = np.clip(float(comp.get("thickness", 0.30)), 0.12, 0.65)
        beta = np.deg2rad(np.clip(float(comp.get("arc_deg", 235.0)), 80.0, 300.0) / 2.0)
        rot = np.deg2rad(float(comp.get("rotation_deg", 0.0)))
        ca, sa = math.cos(rot), math.sin(rot)
        dx, dy = X - float(cx0), Y - float(cy0)
        x_loc = ca * dx + sa * dy
        y_loc = -sa * dx + ca * dy

        x0 = (1.0 - w_belly) * R
        xc = R * math.cos(beta)
        denom = 2.0 * (xc - x0)
        if abs(denom) < 1e-5:
            denom = -1e-5
        d_shift = (R * R - x0 * x0) / denom
        r_in = abs(x0 - d_shift)

        d_out = np.hypot(x_loc, y_loc) - R
        d_in = np.hypot(x_loc - d_shift, y_loc) - r_in
        d_s = np.maximum(d_out, -d_in)
        u_par = (np.arctan2(y_loc, x_loc) + math.pi) / (2.0 * math.pi)
        u_perp = np.clip((d_out + d_in) / (R * w_belly + 1e-4), -1.0, 1.0)
        return d_s, u_par, u_perp, R * w_belly * 0.5

    else:  # wheel_rosette or default
        cx0, cy0 = comp.get("center", [0.0, 0.0])
        R_out = max(0.04, float(comp.get("outer_radius", 0.3)))
        R_hub = max(0.01, float(comp.get("hub_radius", 0.04)))
        spokes = max(3, int(comp.get("spokes", 10)))
        rim_wave = float(comp.get("rim_wave", 0.0))

        dx, dy = X - float(cx0), Y - float(cy0)
        r = np.hypot(dx, dy)
        theta = np.arctan2(dy, dx)
        r_rim = R_out + rim_wave * np.cos(spokes * theta)
        w_rim = max(0.014, 0.09 * R_out)
        w_spoke = max(0.007, 0.04 * R_out)

        d_rim = np.abs(r - r_rim) - w_rim
        d_hub = r - R_hub
        d_spokes = np.maximum(r * np.abs(np.sin(0.5 * spokes * theta)) - w_spoke, np.maximum(r - r_rim, R_hub - r))
        d_s = np.minimum(d_rim, np.minimum(d_hub, d_spokes))
        u_par = np.clip(r / (R_out + 1e-4), 0.0, 1.2)
        u_perp = theta / math.pi
        return d_s, u_par, u_perp, w_rim


def shade_component_layer(
    comp: Dict[str, Any],
    idx: int,
    d_s: np.ndarray,
    u_par: np.ndarray,
    u_perp: np.ndarray,
    w_eff: float,
    X: np.ndarray,
    Y: np.ndarray,
    E: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ctype = comp.get("type", "ribbon_spine")
    c1, c2, tex_type = _resolve_component_palette(comp, idx)
    K_s = float(comp.get("edge_sharpness", 110.0))

    d_mod = d_s.copy()
    if tex_type == "feathers":
        barb_phase = 75.0 * (u_par - 0.36 * (np.abs(u_perp) ** 1.35)) + 3.5 * np.sin(14.0 * u_par + 9.0 * u_perp)
        tex_pattern = 0.5 + 0.5 * np.cos(barb_phase)
        d_mod += 0.005 * (1.0 - tex_pattern)
    elif tex_type == "fur":
        fur_wave = np.sin(95.0 * u_perp + 5.0 * np.sin(18.0 * u_par + 4.0 * u_perp) + 3.5 * E)
        stripes = 0.5 + 0.5 * np.cos(14.0 * u_par + 2.8 * np.sin(6.0 * u_perp + 2.0 * E))
        # High-contrast fur palettes (e.g. Bengal tiger orange/black) show bold organic stripes
        col_dist = abs(c1[0] - c2[0]) + abs(c1[1] - c2[1]) + abs(c1[2] - c2[2])
        if col_dist > 0.75:
            stripe_mask = _ee(-12.0 * (stripes - 0.68))
            tex_pattern = np.clip(0.28 * (0.5 + 0.5 * fur_wave) + 0.72 * stripe_mask, 0.0, 1.0)
        else:
            tex_pattern = 0.5 + 0.5 * fur_wave
        edge_hairs = np.sin(42.0 * u_par + 3.5 * np.sin(14.0 * u_perp)) ** 8
        d_mod -= 0.012 * edge_hairs
    elif tex_type == "foliage":
        leaf_speckle = np.cos(48.0 * X + 6.0 * E) * np.cos(48.0 * Y - 6.0 * E)
        d_mod += 0.025 * E + 0.010 * leaf_speckle
        tex_pattern = np.clip(0.5 + 0.45 * np.sin(3.5 * E + 5.0 * Y) + 0.25 * leaf_speckle, 0.0, 1.0)
    elif tex_type == "petals":
        seeds = _ee(-22.0 * (np.cos(24.0 * u_par) * np.cos(24.0 * u_perp) - 0.62))
        veins = 0.5 + 0.5 * np.cos(18.0 * u_perp + 2.0 * np.sin(6.0 * u_par))
        tex_pattern = np.clip(0.65 * veins + 0.35 * seeds, 0.0, 1.0)
    elif tex_type == "marble":
        tex_pattern = 0.5 + 0.5 * np.cos(12.0 * X + 8.0 * Y + 4.0 * E)
    elif tex_type == "metallic":
        tex_pattern = 0.5 + 0.5 * np.cos(28.0 * u_par - 14.0 * u_perp)
    elif tex_type == "water":
        tex_pattern = 0.5 + 0.5 * np.cos(42.0 * Y + 5.0 * np.sin(12.0 * X + 3.0 * E))
    elif tex_type == "cosmic":
        plasma = np.cos(18.0 * X + 5.0 * E) * np.cos(18.0 * Y - 5.0 * E)
        tex_pattern = np.clip(0.5 + 0.35 * plasma + 0.25 * np.sin(10.0 * u_par + 3.0 * E), 0.0, 1.0)
    else:
        tex_pattern = np.clip(0.5 + 0.3 * (u_par - 0.5) + 0.2 * E, 0.0, 1.0)

    J_s = _ee(K_s * d_mod)

    rho = np.clip(-d_mod / max(w_eff, 0.015), 0.0, 1.0)
    Nz = np.sqrt(np.clip(2.0 * rho - rho * rho, 0.04, 1.0))
    gy, gx = np.gradient(d_mod)
    gnorm = np.hypot(gx, gy) + 1e-6
    xy_scale = np.sqrt(np.maximum(0.0, 1.0 - Nz * Nz))
    Nx = (-gx / gnorm) * xy_scale
    Ny = (-gy / gnorm) * xy_scale

    diff = np.clip(-0.42 * Nx + 0.62 * Ny + 0.66 * Nz, 0.0, 1.0)
    spec_power = 22.0 if tex_type in ("metallic", "petals", "water", "cosmic") else 10.0
    spec_amp = 0.45 if tex_type in ("metallic", "petals", "water", "cosmic") else 0.18
    spec = spec_amp * (np.clip(0.25 * Nx + 0.45 * Ny + 0.85 * Nz, 0.0, 1.0) ** spec_power)
    fresnel = ((1.0 - Nz) ** 2.4) if tex_type in ("cosmic", "water", "metallic") else 0.0

    if tex_type == "feathers" and ctype == "polar_harmonic":
        z_pupil = _ee(35.0 * (u_par - 0.26))
        z_cyan = _ee(35.0 * (u_par - 0.46)) * (1.0 - z_pupil)
        z_gold = _ee(30.0 * (u_par - 0.72)) * (1.0 - z_pupil - z_cyan)
        z_outer = np.clip(1.0 - z_pupil - z_cyan - z_gold, 0.0, 1.0)
        barb_shade = 0.55 + 0.45 * tex_pattern
        C0 = (0.05 * z_pupil + 0.10 * z_cyan + 0.82 * z_gold + c1[0] * z_outer) * barb_shade + 0.5 * spec
        C1 = (0.06 * z_pupil + 0.85 * z_cyan + 0.60 * z_gold + c1[1] * z_outer) * barb_shade + 0.5 * spec
        C2 = (0.22 * z_pupil + 0.85 * z_cyan + 0.18 * z_gold + c1[2] * z_outer) * barb_shade + 0.5 * spec
    else:
        mix = np.clip(0.60 * tex_pattern + 0.40 * np.clip(u_par, 0.0, 1.0), 0.0, 1.0)
        base0 = c1[0] * (1.0 - mix) + c2[0] * mix
        base1 = c1[1] * (1.0 - mix) + c2[1] * mix
        base2 = c1[2] * (1.0 - mix) + c2[2] * mix
        shade = (0.48 + 0.52 * diff) if tex_type == "cosmic" else (0.35 + 0.65 * diff)
        C0 = base0 * shade + spec + 0.45 * fresnel * c2[0]
        C1 = base1 * shade + spec + 0.45 * fresnel * c2[1]
        C2 = base2 * shade + spec + 0.45 * fresnel * c2[2]

    return J_s, np.clip(C0, 0.0, 1.0), np.clip(C1, 0.0, 1.0), np.clip(C2, 0.0, 1.0)


def _format_lagrange_rgb_poly(rgb: List[float]) -> str:
    """
    Formats the exact Hamid-style Lagrange quadratic polynomial P(v) in v in {0, 1, 2}
    evaluating to rgb[0] at v=0, rgb[1] at v=1, rgb[2] at v=2:
      P(v) = a0 + a1*v + a2*v^2
    """
    r, g, b = float(rgb[0]), float(rgb[1]), float(rgb[2])
    a0 = r
    a1 = 0.5 * (-3.0 * r + 4.0 * g - b)
    a2 = 0.5 * (r - 2.0 * g + b)
    terms = [f"{a0:.2f}"]
    if abs(a1) >= 0.005:
        terms.append(f"{'+' if a1 >= 0 else '-'} {abs(a1):.2f}v")
    if abs(a2) >= 0.005:
        terms.append(f"{'+' if a2 >= 0 else '-'} {abs(a2):.2f}v^2")
    return " ".join(terms)


def _expr_to_latex(expr: str) -> str:
    """Converts a safe analytical SDF expression string into readable KaTeX LaTeX."""
    s = str(expr or "").strip()
    s = re.sub(r"\b(?:np|numpy|math)\.", "", s)
    s = s.replace("**", "^").replace("*", r"\cdot ")
    for fn, tex in (
        ("arctan2", r"\operatorname{atan2}"),
        ("atan2", r"\operatorname{atan2}"),
        ("arctan", r"\arctan"),
        ("arcsin", r"\arcsin"),
        ("arccos", r"\arccos"),
        ("sqrt", r"\sqrt"),
        ("sin", r"\sin"),
        ("cos", r"\cos"),
        ("tan", r"\tan"),
        ("exp", r"\exp"),
        ("abs", r"\operatorname{abs}"),
        ("pi", r"\pi"),
    ):
        s = re.sub(rf"\b{fn}\b", lambda _m, t=tex: t, s)
    if len(s) > 140:
        s = s[:137] + r"\dots"
    return s


def build_hamid_pixel_field_latex(ir: Dict[str, Any], W_nom: int = 2000, H_nom: int = 1200) -> Dict[str, str]:
    """
    Generates prompt-specific Hamid Naderi Yeganeh double-exponential LaTeX equations
    featuring exact Lagrange quadratic RGB polynomials C_s(v) in v in {0,1,2},
    real component/layer counts, real repeated_field nested summations, and real
    implicit_sdf_expr level-set equations with directional relief self-shadowing.
    """
    comps = ir.get("components", [])
    M = max(1, len(comps))
    m0 = W_nom // 2
    n0 = (H_nom // 2) + 1
    S = H_nom // 2

    bg_style = str(ir.get("bg_style") or "golden_bokeh").strip().lower()
    bg_eq_map = {
        "night_sky": r"B_v(x,y) = T_v(x,y)\,e^{-e^{-45(y+0.44)}} + \left(\frac{2+5v}{100} + \frac{10+6v}{100}\cos\!\left(\frac{13.5}{-y-0.37}\right)\right)\left(1-e^{-e^{-45(y+0.44)}}\right)",
        "cosmic_nebula": r"B_v(x,y) = T_v(x,y) + \left(0.50 - 0.64v + 0.34v^2\right)e^{-0.95(x^2+y^2)}\cos^2(3.2x + 2.8E(x,y))",
        "deep_ocean": r"B_v(x,y) = (0.02 + 0.08v + 0.01v^2) + (0.09 + 0.43v - 0.12v^2)\frac{y+1}{2} + (0.10 + 0.16v - 0.04v^2)\cos^{5}\!\left(3.8x - 1.5y + 0.85E(x,y)\right)",
        "sunset_sky": r"B_v(x,y) = G(x,y)\left((0.96 - 0.35v + 0.03v^2) + 0.38e^{-4.2((x-0.12)^2+(y+0.10)^2)}\right) + (1-G(x,y))(0.26 - 0.17v + 0.05v^2)",
        "bokeh_forest": r"B_v(x,y) = \left(0.20 + 0.41v - 0.23v^2 + (0.50 + 0.21v - 0.15v^2)K(x,y)\right)\left(0.95 - 0.22(x^2+y^2)\right)",
        "lunar_dust": r"B_v(x,y) = \left(1.02 - 0.03v - 0.01v^2\right)\left(0.32 + 0.55\,e^{-e^{6(E(x+0.035,y+0.025)-E(x,y))}}\right)",
        "golden_bokeh": r"B_v(x,y) = \left(0.68 - 0.19v - 0.05v^2 + (0.34 + 0.23v - 0.11v^2)K(x,y)\right)\frac{y+1.15}{1.9}",
    }
    bg_tex = bg_eq_map.get(bg_style, bg_eq_map["golden_bokeh"])

    # Extract actual component Lagrange quadratic RGB polynomials & special primitives
    poly_lines: List[str] = []
    rep_comps: List[Tuple[int, Dict[str, Any]]] = []
    sdf_comps: List[Tuple[int, Dict[str, Any]]] = []
    K_vals: List[int] = []

    for idx, c in enumerate(comps):
        c1, c2, _ = _resolve_component_palette(c, idx)
        K_vals.append(int(round(float(c.get("edge_sharpness", 110.0)))))
        ctype = str(c.get("type", "ribbon_spine")).strip().lower()
        if ctype == "repeated_field":
            rep_comps.append((idx + 1, c))
        elif ctype == "implicit_sdf_expr":
            sdf_comps.append((idx + 1, c))
        if idx < 3:
            p1 = _format_lagrange_rgb_poly(c1)
            p2 = _format_lagrange_rgb_poly(c2)
            poly_lines.append(rf"C_{{{idx+1}}}(v) = \left({p1}\right)(1-\mu_{{{idx+1}}}) + \left({p2}\right)\mu_{{{idx+1}}}")

    K0 = K_vals[0] if K_vals else 110
    latex: Dict[str, str] = {
        "RGB": (
            rf"\operatorname{{rgb}}\!\left(F\!\left(H_0\!\left(\frac{{m-{m0}}}{{{S}}}, \frac{{{n0}-n}}{{{S}}}\right)\right),\, "
            rf"F\!\left(H_1\!\left(\frac{{m-{m0}}}{{{S}}}, \frac{{{n0}-n}}{{{S}}}\right)\right),\, "
            rf"F\!\left(H_2\!\left(\frac{{m-{m0}}}{{{S}}}, \frac{{{n0}-n}}{{{S}}}\right)\right)\right), \quad "
            r"F(x) = \left\lfloor 255\,e^{-e^{-1000x}}\,|x|^{e^{-e^{1000(x-1)}}} \right\rfloor"
        ),
        "H_v": (
            rf"H_v(x, y) = B_v(x, y)\prod_{{s=1}}^{{{M}}}\big(1 - J_s(x, y)\big) + W_v(x, y), \quad v \in \{{0, 1, 2\}}, \quad {bg_tex}"
        ),
        "W_v": (
            rf"W_v(x, y) = \sum_{{s=1}}^{{{M}}}\left(\prod_{{u=0}}^{{s-1}}\left(1 - e^{{-e^{{-100(u-1/2)}}}}J_u(x, y)\right)\right)"
            rf"J_s(x, y)\,C_s(v)\,\Phi_s(x,y), \quad J_s(x, y) = e^{{-e^{{{K0}\,d_s(x, y)}}}}"
        ),
    }

    if rep_comps:
        s_idx, rc = rep_comps[0]
        n_rep = max(3, min(64, int(rc.get("count", 18))))
        layout_name = str(rc.get("layout", "scattered_cluster")).replace("_", r"\text{-}")
        K_rep = int(round(float(rc.get("edge_sharpness", 75.0))))
        latex["R_s"] = (
            rf"J_{{{s_idx}}}(x, y) = 1 - \prod_{{j=1}}^{{{n_rep}}}\left(1 - e^{{-e^{{{K_rep}\,d_{{j}}^{{(\mathrm{{{layout_name}}})}}(x, y)}}}}\right), \quad "
            rf"W_{{v,{s_idx}}}(x, y) = \sum_{{j=1}}^{{{n_rep}}}\left(\prod_{{q=1}}^{{j-1}}(1 - J_{{{s_idx},q}})\right) J_{{{s_idx},j}}\,C_{{{s_idx},j}}(v)"
        )

    if sdf_comps:
        s_idx, sc = sdf_comps[0]
        K_sdf = int(round(float(sc.get("edge_sharpness", 110.0))))
        expr_tex = _expr_to_latex(str(sc.get("sdf_expr") or sc.get("expr") or "sqrt(x^2+y^2)-0.35"))
        latex["D_s"] = (
            rf"d_{{{s_idx}}}(x, y) = {expr_tex}, \quad J_{{{s_idx}}}(x, y) = e^{{-e^{{{K_sdf}\,d_{{{s_idx}}}(x, y)}}}}, \quad "
            rf"\Lambda_{{{s_idx}}}(x, y) = e^{{-e^{{14\left(d_{{{s_idx}}}(x+\delta_x, y+\delta_y) - d_{{{s_idx}}}(x, y)\right)}}}}"
        )

    if poly_lines:
        latex["C_v"] = r", \quad ".join(poly_lines)

    latex["E"] = (
        r"E(x, y) = \sum_{s=1}^{7}\frac{5}{4}\left(\frac{17}{20}\right)^s "
        r"\cos\!\Big(3.2(1.28)^s(\cos(2s^2)x + \sin(2s^2)y) + 3.2\cos(3.2(1.28)^s(\cos(7s^2)x + \sin(7s^2)y)) + 2\cos(5s)\Big)"
        r"\cos\!\Big(3.2(1.28)^s(\cos(2s^2)y - \sin(2s^2)x) + 3.2\cos(3.2(1.28)^s(\cos(8s^2)x + \sin(8s^2)y)) + 2\cos(5s)\Big)"
    )
    return latex


def compile_ir_to_pixel_field(
    ir: Dict[str, Any],
    width: int = 600,
    height: int = 360,
    t_phase: float = 0.0,
    include_raw_array: bool = False,
) -> Dict[str, Any]:
    """
    Compiles ANY HamidMathematicalArtIR specification (including direct symbolic
    math_program ASTs, repeated_field, implicit_sdf_expr, ribbon_spine,
    polar_harmonic, crescent_lobe, wheel_rosette, and parametric_equation) into
    a full-color 2D pixel-field painting using Hamid Naderi Yeganeh's
    Porter-Duff double-exponential level-set equations H_v(x, y) and F(x).
    """
    if isinstance(ir, dict):
        math_prog = ir.get("math_program")
        if math_prog is None and all(k in ir for k in ("steps", "H0", "H1", "H2")):
            math_prog = ir
        if isinstance(math_prog, dict) and all(k in math_prog for k in ("steps", "H0", "H1", "H2")):
            from symbolic_math_compiler import evaluate_math_program_numpy
            out = evaluate_math_program_numpy(
                math_prog,
                width=width,
                height=height,
                t_phase=t_phase,
                include_raw_array=include_raw_array,
            )
            if "refinement_passes" in ir:
                out["refinement_passes"] = ir["refinement_passes"]
            if "visual_score" in ir:
                out["visual_score"] = ir["visual_score"]
            return out

    m0 = width / 2.0
    n0 = height / 2.0
    S = min(width, height) / 2.0
    m_coords = np.arange(width, dtype=np.float32)
    n_coords = np.arange(height, dtype=np.float32)
    M_grid, N_grid = np.meshgrid(m_coords, n_coords)
    X = (M_grid - m0) / S
    Y = (n0 - N_grid) / S

    E = compute_hamid_trigonometric_fbm(X, Y, octaves=6)

    title_low = (ir.get("title") or "").lower()
    default_bg = (
        "deep_ocean" if any(k in title_low for k in ["jelly", "coral", "ocean", "sea", "fish", "whale", "reef"]) else
        "cosmic_nebula" if any(k in title_low for k in ["saturn", "space", "astronaut", "nebula", "galaxy", "planet", "cosmos"]) else
        "sunset_sky" if any(k in title_low for k in ["sunset", "mountain", "peak", "autumn", "desert", "phoenix"]) else
        "night_sky" if any(k in title_low for k in ["mosque", "faisal", "hilal", "moon", "night", "castle"]) else
        "bokeh_forest" if any(k in title_low for k in ["tiger", "jungle", "cobra", "snake", "eagle", "bird", "kiwi", "leaf", "tree", "forest"]) else
        "golden_bokeh"
    )
    bg_style = ir.get("bg_style") or default_bg
    B0, B1, B2 = evaluate_background_field(
        bg_style, X, Y, E,
        primary_rgb=ir.get("bg_primary_rgb"),
        secondary_rgb=ir.get("bg_secondary_rgb"),
        t_phase=t_phase,
    )

    raw_comps = ir.get("components", [])
    expanded_layers: List[Tuple[int, int, Dict[str, Any], Optional[str], Any]] = []
    for idx, c in enumerate(raw_comps):
        z = int(c.get("z_order", idx + 1))
        ctype = str(c.get("type", "ribbon_spine")).strip().lower()
        if ctype == "repeated_field":
            expanded_layers.append((z, idx, c, None, None))
            continue
        rc = max(1, min(16, int(c.get("radial_copies", 1))))
        mx = bool(c.get("mirror_x", False))
        if rc > 1:
            arc = c.get("radial_arc_deg", [0.0, 360.0])
            if not isinstance(arc, list) or len(arc) < 2:
                arc = [0.0, 360.0]
            angles = np.linspace(float(arc[0]), float(arc[1]), rc, endpoint=(abs(float(arc[1]) - float(arc[0]) - 360.0) > 1e-2))
            for ang in angles:
                expanded_layers.append((z, idx, c, "radial", (float(ang), c.get("radial_center", [0.0, 0.0]))))
        elif mx:
            axis_x = float(c.get("mirror_axis_x", 0.0))
            expanded_layers.append((z, idx, c, None, None))
            expanded_layers.append((z, idx, c, "mirror_x", axis_x))
        else:
            expanded_layers.append((z, idx, c, None, None))

    # Front-to-back order (smallest z_order in foreground, larger z_order in background)
    expanded_layers.sort(key=lambda item: item[0])

    W0 = np.zeros_like(X, dtype=np.float32)
    W1 = np.zeros_like(X, dtype=np.float32)
    W2 = np.zeros_like(X, dtype=np.float32)
    transmittance = np.ones_like(X, dtype=np.float32)
    cast_shadow = np.ones_like(X, dtype=np.float32)

    for z, idx, comp, sym_mode, sym_param in expanded_layers:
        ctype = str(comp.get("type", "ribbon_spine")).strip().lower()

        if ctype == "repeated_field":
            alpha_rep, rw0, rw1, rw2 = evaluate_repeated_field_layer(
                comp, idx, X, Y, E, t_phase=t_phase, bg_style=bg_style
            )
            W0 += transmittance * rw0 * cast_shadow
            W1 += transmittance * rw1 * cast_shadow
            W2 += transmittance * rw2 * cast_shadow
            transmittance *= (1.0 - alpha_rep)
            continue

        if sym_mode == "mirror_x":
            ax = float(sym_param)
            X_loc, Y_loc = 2.0 * ax - X, Y
        elif sym_mode == "radial":
            ang_deg, (rcx, rcy) = sym_param
            rad = np.deg2rad(-ang_deg)
            ca, sa = math.cos(rad), math.sin(rad)
            dx, dy = X - float(rcx), Y - float(rcy)
            X_loc = float(rcx) + ca * dx - sa * dy
            Y_loc = float(rcy) + sa * dx + ca * dy
        else:
            X_loc, Y_loc = X, Y

        if ctype == "implicit_sdf_expr":
            J_s, C0, C1, C2 = evaluate_implicit_sdf_layer(comp, idx, X_loc, Y_loc, E, t_phase=t_phase)
        else:
            d_s, u_par, u_perp, w_eff = evaluate_primitive_level_set(comp, X_loc, Y_loc, t_phase=t_phase)
            J_s, C0, C1, C2 = shade_component_layer(
                comp, idx, d_s, u_par, u_perp, w_eff, X_loc, Y_loc, E,
            )

        weight = transmittance * J_s
        W0 += weight * C0 * cast_shadow
        W1 += weight * C1 * cast_shadow
        W2 += weight * C2 * cast_shadow
        transmittance *= (1.0 - J_s)

        # Subtle soft cast shadow on deeper background layers (zero-padded so edges never wrap!)
        sh_y = max(1, height // 80)
        sh_x = max(1, width // 110)
        J_shift = np.zeros_like(J_s)
        J_shift[sh_y:, sh_x:] = J_s[:-sh_y, :-sh_x]
        cast_shadow = np.clip(cast_shadow * (1.0 - 0.16 * J_shift), 0.55, 1.0)

    H0 = B0 * transmittance * cast_shadow + W0
    H1 = B1 * transmittance * cast_shadow + W1
    H2 = B2 * transmittance * cast_shadow + W2

    rgb_img = np.stack([
        hamid_F_quantize(H0),
        hamid_F_quantize(H1),
        hamid_F_quantize(H2),
    ], axis=-1)

    latex_dict = build_hamid_pixel_field_latex(ir, 2000, 1200)
    title = ir.get("title") or "2D Pixel-Field Mathematical Painting"
    rep_count = sum(max(3, min(64, int(c.get("count", 18)))) for c in raw_comps if str(c.get("type", "")).lower() == "repeated_field")
    total_layers = len(raw_comps) + rep_count
    result: Dict[str, Any] = {
        "title": title,
        "subtitle": f"2D Pixel-Field Painting rgb(F(H₀(x,y)), F(H₁(x,y)), F(H₂(x,y))) — {total_layers} Occluded Double-Exponential Layers",
        "author": "Hamid Naderi Yeganeh Double-Exponential Level-Set Engine",
        "mode": "pixel_field",
        "N": height,
        "width": width,
        "height": height,
        "nominal_aspect": [2000, 1200],
        "harmonics_count": "2D Pixel-Field Exact",
        "latex": latex_dict,
        "primitives": [],
        "image_base64": encode_rgb_to_base64_png(rgb_img),
        "python_code": build_pixel_field_python_script(title, (2000, 1200), latex_dict),
    }
    if include_raw_array:
        result["rgb_array"] = rgb_img
    return result
