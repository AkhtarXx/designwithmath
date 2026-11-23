"""
H-DGES Core Calculus Engine: Vectorized Pure-NumPy Reverse-Mode Automatic
Differentiation over Hamid Naderi Yeganeh's C-infinity Double-Exponential
Gompertz-Harmonic Algebra (H_kappa).

Zero external ML frameworks (no PyTorch / JAX / TensorFlow). Every forward
evaluation, Porter-Duff occlusion adjoint, Gompertz survival derivative,
superellipse-warp Jacobian, 3D directional relief gradient, harmonic texture
gradient, and exact spatial partial derivative (d/dx, d/dy) is computed in
exact closed-form vectorized NumPy calculus.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

EPS_ABS = 1e-8
EPS_RAD = 1e-10
EXP_CLIP = 22.0


@dataclass
class GompertzHarmonicLayer:
    """Single closed-form differentiable Gompertz-Harmonic analytical layer."""

    cx: float = 0.0
    cy: float = 0.0
    rx: float = 0.35
    ry: float = 0.35
    theta: float = 0.0
    p_exp: float = 2.0
    warp_a: float = 0.0
    warp_f: float = 3.0
    warp_p: float = 0.0
    k_mul: float = 1.0
    alpha: float = 0.92
    r: float = 0.8
    g: float = 0.5
    b: float = 0.3
    lx: float = -0.45
    ly: float = 0.55
    shade_amp: float = 0.22
    spec_amp: float = 0.15
    tex_amp: float = 0.0
    tex_wx: float = 12.0
    tex_wy: float = 4.0
    tex_phi: float = 0.0
    z_depth: float = 0.5
    t_amp_x: float = 0.0
    t_amp_y: float = 0.0
    t_amp_rot: float = 0.0
    t_freq: float = 1.0
    t_phase0: float = 0.0
    label: str = "manifold"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cx": float(self.cx),
            "cy": float(self.cy),
            "rx": float(self.rx),
            "ry": float(self.ry),
            "theta": float(self.theta),
            "p_exp": float(self.p_exp),
            "warp_a": float(self.warp_a),
            "warp_f": float(self.warp_f),
            "warp_p": float(self.warp_p),
            "k_mul": float(self.k_mul),
            "alpha": float(self.alpha),
            "r": float(self.r),
            "g": float(self.g),
            "b": float(self.b),
            "lx": float(self.lx),
            "ly": float(self.ly),
            "shade_amp": float(self.shade_amp),
            "spec_amp": float(self.spec_amp),
            "tex_amp": float(self.tex_amp),
            "tex_wx": float(self.tex_wx),
            "tex_wy": float(self.tex_wy),
            "tex_phi": float(self.tex_phi),
            "z_depth": float(self.z_depth),
            "t_amp_x": float(self.t_amp_x),
            "t_amp_y": float(self.t_amp_y),
            "t_amp_rot": float(self.t_amp_rot),
            "t_freq": float(self.t_freq),
            "t_phase0": float(self.t_phase0),
            "label": str(self.label),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "GompertzHarmonicLayer":
        valid = {k: d[k] for k in cls.__dataclass_fields__ if k in d}
        return cls(**valid)


# Parameter ordering for flat vectorized gradient updates per layer
LAYER_PARAM_NAMES: Tuple[str, ...] = (
    "cx", "cy", "rx", "ry", "theta", "p_exp",
    "warp_a", "warp_f", "warp_p", "k_mul", "alpha",
    "r", "g", "b",
    "lx", "ly", "shade_amp", "spec_amp",
    "tex_amp", "tex_wx", "tex_wy", "tex_phi",
)

LAYER_BOUNDS: Dict[str, Tuple[float, float]] = {
    "cx": (-1.75, 1.75),
    "cy": (-1.15, 1.15),
    "rx": (0.015, 1.85),
    "ry": (0.015, 1.35),
    "theta": (-math.pi, math.pi),
    "p_exp": (1.25, 6.0),
    "warp_a": (-0.65, 0.65),
    "warp_f": (0.5, 18.0),
    "warp_p": (-math.pi, math.pi),
    "k_mul": (0.25, 3.0),
    "alpha": (0.08, 1.0),
    "r": (0.0, 1.0),
    "g": (0.0, 1.0),
    "b": (0.0, 1.0),
    "lx": (-1.0, 1.0),
    "ly": (-1.0, 1.0),
    "shade_amp": (-0.65, 0.65),
    "spec_amp": (0.0, 0.85),
    "tex_amp": (-0.45, 0.45),
    "tex_wx": (-45.0, 45.0),
    "tex_wy": (-45.0, 45.0),
    "tex_phi": (-math.pi, math.pi),
}


@dataclass
class HDGESBackground:
    """Differentiable multi-frequency closed-form trigonometric background B_v(x,y)."""

    c0: np.ndarray = field(default_factory=lambda: np.array([0.08, 0.12, 0.22], dtype=np.float64))
    cx: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0], dtype=np.float64))
    cy: np.ndarray = field(default_factory=lambda: np.array([0.04, 0.06, 0.10], dtype=np.float64))
    amp: np.ndarray = field(default_factory=lambda: np.zeros((3, 4), dtype=np.float64))
    wx: np.ndarray = field(default_factory=lambda: np.array([1.8, 3.5, 6.2, 10.5], dtype=np.float64))
    wy: np.ndarray = field(default_factory=lambda: np.array([2.2, -2.8, 4.7, -8.3], dtype=np.float64))
    phi: np.ndarray = field(default_factory=lambda: np.array([0.3, 1.2, 2.4, 0.8], dtype=np.float64))

    def copy(self) -> "HDGESBackground":
        return HDGESBackground(
            c0=self.c0.copy(),
            cx=self.cx.copy(),
            cy=self.cy.copy(),
            amp=self.amp.copy(),
            wx=self.wx.copy(),
            wy=self.wy.copy(),
            phi=self.phi.copy(),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "c0": self.c0.tolist(),
            "cx": self.cx.tolist(),
            "cy": self.cy.tolist(),
            "amp": self.amp.tolist(),
            "wx": self.wx.tolist(),
            "wy": self.wy.tolist(),
            "phi": self.phi.tolist(),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "HDGESBackground":
        return cls(
            c0=np.array(d["c0"], dtype=np.float64),
            cx=np.array(d["cx"], dtype=np.float64),
            cy=np.array(d["cy"], dtype=np.float64),
            amp=np.array(d["amp"], dtype=np.float64),
            wx=np.array(d["wx"], dtype=np.float64),
            wy=np.array(d["wy"], dtype=np.float64),
            phi=np.array(d["phi"], dtype=np.float64),
        )


@dataclass
class HDGESModel:
    """Complete closed-form H-DGES mathematical field H_v(x, y, t)."""

    title: str = "H-DGES Analytical Field"
    subtitle: str = "Homotopy-Annealed Differentiable Gompertz-Harmonic Equation"
    aspect: Tuple[int, int] = (16, 10)
    x_min: float = -1.6
    x_max: float = 1.6
    y_min: float = -1.0
    y_max: float = 1.0
    kappa_global: float = 48.0
    background: HDGESBackground = field(default_factory=HDGESBackground)
    layers: List[GompertzHarmonicLayer] = field(default_factory=list)
    symmetry_pairs: List[Tuple[int, int]] = field(default_factory=list)

    def copy(self) -> "HDGESModel":
        return HDGESModel(
            title=self.title,
            subtitle=self.subtitle,
            aspect=self.aspect,
            x_min=self.x_min,
            x_max=self.x_max,
            y_min=self.y_min,
            y_max=self.y_max,
            kappa_global=self.kappa_global,
            background=self.background.copy(),
            layers=[GompertzHarmonicLayer.from_dict(L.to_dict()) for L in self.layers],
            symmetry_pairs=list(self.symmetry_pairs),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "subtitle": self.subtitle,
            "aspect": list(self.aspect),
            "domain": {
                "x_min": self.x_min,
                "x_max": self.x_max,
                "y_min": self.y_min,
                "y_max": self.y_max,
            },
            "kappa_global": float(self.kappa_global),
            "background": self.background.to_dict(),
            "layers": [L.to_dict() for L in self.layers],
            "symmetry_pairs": [list(p) for p in self.symmetry_pairs],
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "HDGESModel":
        dom = d.get("domain", {})
        return cls(
            title=str(d.get("title", "H-DGES Analytical Field")),
            subtitle=str(d.get("subtitle", "")),
            aspect=tuple(d.get("aspect", [16, 10])),
            x_min=float(dom.get("x_min", -1.6)),
            x_max=float(dom.get("x_max", 1.6)),
            y_min=float(dom.get("y_min", -1.0)),
            y_max=float(dom.get("y_max", 1.0)),
            kappa_global=float(d.get("kappa_global", 48.0)),
            background=HDGESBackground.from_dict(d["background"]),
            layers=[GompertzHarmonicLayer.from_dict(L) for L in d.get("layers", [])],
            symmetry_pairs=[tuple(p) for p in d.get("symmetry_pairs", [])],
        )


def make_grid(
    width: int,
    height: int,
    x_min: float = -1.6,
    x_max: float = 1.6,
    y_min: float = -1.0,
    y_max: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Create Cartesian (X, Y) evaluation grid with +y upward (Hamid convention)."""
    xs = np.linspace(x_min, x_max, width, dtype=np.float64)
    ys = np.linspace(y_max, y_min, height, dtype=np.float64)
    return np.meshgrid(xs, ys)


def hamid_F_quantize_smooth(h: np.ndarray) -> np.ndarray:
    """Hamid Naderi Yeganeh exact uint8 quantization F(x)."""
    h_clip = np.clip(h, 0.0, 1.0)
    return np.floor(255.0 * h_clip + 1e-9).astype(np.uint8)


def _eval_background(
    bg: HDGESBackground, X: np.ndarray, Y: np.ndarray, t_phase: float = 0.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate B_v(x, y, t) and return (B[3,H,W], phases[Q,H,W], cos_phases[Q,H,W])."""
    Q = len(bg.wx)
    # Periodic time drift on higher-frequency background waves for subtle living motion
    t_shift = 0.35 * math.sin(t_phase) if t_phase != 0.0 else 0.0
    phases = (
        bg.wx[:, None, None] * X[None, :, :]
        + bg.wy[:, None, None] * Y[None, :, :]
        + bg.phi[:, None, None]
        + t_shift
    )
    cos_p = np.cos(phases)  # (Q, H, W)
    B = (
        bg.c0[:, None, None]
        + bg.cx[:, None, None] * X[None, :, :]
        + bg.cy[:, None, None] * Y[None, :, :]
        + np.tensordot(bg.amp, cos_p, axes=([1], [0]))
    )
    return B, phases, cos_p


def _layer_effective_pose(L: GompertzHarmonicLayer, t_phase: float) -> Tuple[float, float, float]:
    """Evaluate smooth periodic 4D temporal flow (cx(t), cy(t), theta(t))."""
    if t_phase == 0.0 or (L.t_amp_x == 0.0 and L.t_amp_y == 0.0 and L.t_amp_rot == 0.0):
        return L.cx, L.cy, L.theta
    arg = L.t_freq * t_phase + L.t_phase0
    return (
        L.cx + L.t_amp_x * math.sin(arg),
        L.cy + L.t_amp_y * math.cos(arg),
        L.theta + L.t_amp_rot * math.sin(arg),
    )


def forward_hdges(
    model: HDGESModel,
    X: np.ndarray,
    Y: np.ndarray,
    kappa_global: Optional[float] = None,
    t_phase: float = 0.0,
    return_cache: bool = False,
) -> Tuple[np.ndarray, Optional[Dict[str, Any]]]:
    """
    Exact forward evaluation of H_v(x, y, t) in H_kappa double-exponential algebra.
    Returns H of shape (3, H, W) in R (typically [0, 1]) and optional autodiff cache.
    """
    k_glob = model.kappa_global if kappa_global is None else float(kappa_global)
    H_grid, W_grid = X.shape
    B, bg_phases, bg_cos = _eval_background(model.background, X, Y, t_phase=t_phase)

    M = len(model.layers)
    P = np.ones((H_grid, W_grid), dtype=np.float64)
    H_accum = np.zeros((3, H_grid, W_grid), dtype=np.float64)
    layer_caches: List[Dict[str, Any]] = []

    for L in model.layers:
        cx_t, cy_t, th_t = _layer_effective_pose(L, t_phase)
        ct = math.cos(th_t)
        st = math.sin(th_t)
        dx = X - cx_t
        dy = Y - cy_t
        u = dx * ct + dy * st
        v = -dx * st + dy * ct

        # Trigonometric curvature warp
        w_arg = L.warp_f * u + L.warp_p
        sin_w = np.sin(w_arg)
        vw = v + L.warp_a * sin_w

        rx = max(L.rx, 1e-4)
        ry = max(L.ry, 1e-4)
        un = u / rx
        vn = vw / ry

        # Smooth C-infinity absolute values for exact differentiability at axes
        au = np.sqrt(un * un + EPS_ABS)
        av = np.sqrt(vn * vn + EPS_ABS)
        p = max(L.p_exp, 1.05)

        au_p = np.power(au, p)
        av_p = np.power(av, p)
        S_pow = au_p + av_p + EPS_RAD
        rad = np.power(S_pow, 1.0 / p)
        g_val = rad - 1.0

        # Hamid Gompertz Double-Exponential Kernel J_s = alpha_s * Phi_kappa(g)
        k_eff = k_glob * max(L.k_mul, 0.1)
        z_raw = k_eff * g_val
        z_clamped = np.clip(z_raw, -EXP_CLIP, EXP_CLIP)
        ez = np.exp(z_clamped)
        Phi = np.exp(-ez)
        alpha_eff = float(max(0.05, min(1.0, L.alpha)))
        J = alpha_eff * Phi

        # Harmonic trigonometric micro-texture T_s(x, y)
        tex_arg = L.tex_wx * un + L.tex_wy * vn + L.tex_phi
        cos_tex = np.cos(tex_arg)
        T = 1.0 + L.tex_amp * cos_tex

        # Analytical 3D directional relief shading + specular highlight S_s(x, y)
        dot_l = L.lx * un + L.ly * vn
        du_l = un - L.lx
        dv_l = vn - L.ly
        spec_d2 = du_l * du_l + dv_l * dv_l
        spec_e = np.exp(-3.5 * spec_d2)
        S_light = 1.0 + L.shade_amp * dot_l + L.spec_amp * spec_e

        # Combined layer radiance F_{s,v} = C_{s,v} * T_s * S_s
        C_vec = np.array([L.r, L.g, L.b], dtype=np.float64)
        TS = T * S_light
        F = C_vec[:, None, None] * TS[None, :, :]

        # Porter-Duff front-to-back analytic occlusion
        W_s = P * J
        H_accum += W_s[None, :, :] * F
        P_next = P * (1.0 - J)

        if return_cache:
            layer_caches.append({
                "dx": dx, "dy": dy, "ct": ct, "st": st,
                "u": u, "v": v, "w_arg": w_arg, "sin_w": sin_w, "vw": vw,
                "rx": rx, "ry": ry, "un": un, "vn": vn,
                "au": au, "av": av, "p": p,
                "au_p": au_p, "av_p": av_p, "S_pow": S_pow, "rad": rad, "g_val": g_val,
                "k_eff": k_eff, "z_raw": z_raw, "ez": ez, "Phi": Phi, "alpha_eff": alpha_eff, "J": J,
                "tex_arg": tex_arg, "cos_tex": cos_tex, "T": T,
                "dot_l": dot_l, "du_l": du_l, "dv_l": dv_l, "spec_e": spec_e,
                "S_light": S_light, "TS": TS, "C_vec": C_vec, "F": F,
                "P": P, "W_s": W_s,
            })
        P = P_next

    H_out = H_accum + P[None, :, :] * B
    if not return_cache:
        return H_out, None

    cache = {
        "X": X,
        "Y": Y,
        "k_glob": k_glob,
        "B": B,
        "bg_phases": bg_phases,
        "bg_cos": bg_cos,
        "P_final": P,
        "layers": layer_caches,
    }
    return H_out, cache


def backward_hdges(
    model: HDGESModel,
    cache: Dict[str, Any],
    dL_dH: np.ndarray,
) -> Dict[str, Any]:
    """
    Exact reverse-mode analytical backpropagation through the entire H_kappa DAG.
    Given dL_dH of shape (3, H, W), returns exact gradients w.r.t. every background
    and layer parameter in O(M * H * W) time with zero numerical finite differences.
    """
    X = cache["X"]
    Y = cache["Y"]
    k_glob = cache["k_glob"]
    B = cache["B"]
    bg_phases = cache["bg_phases"]
    bg_cos = cache["bg_cos"]
    P_final = cache["P_final"]
    layer_caches = cache["layers"]

    # 1. Background adjoints: H = H_accum + P_final * B
    dB = dL_dH * P_final[None, :, :]  # (3, H, W)
    grad_bg_c0 = np.sum(dB, axis=(1, 2))
    grad_bg_cx = np.sum(dB * X[None, :, :], axis=(1, 2))
    grad_bg_cy = np.sum(dB * Y[None, :, :], axis=(1, 2))
    # B_v += sum_q amp[v, q] * cos(phase[q])
    grad_bg_amp = np.tensordot(dB, bg_cos, axes=([1, 2], [1, 2]))  # (3, Q)
    d_cos_p = np.tensordot(model.background.amp, dB, axes=([0], [0]))  # (Q, H, W)
    d_phase = -d_cos_p * np.sin(bg_phases)
    grad_bg_wx = np.sum(d_phase * X[None, :, :], axis=(1, 2))
    grad_bg_wy = np.sum(d_phase * Y[None, :, :], axis=(1, 2))
    grad_bg_phi = np.sum(d_phase, axis=(1, 2))

    # Adjoint w.r.t. final occlusion transmittance P_M
    dP = np.sum(dL_dH * B, axis=0)  # (H, W)

    # 2. Reverse Porter-Duff & Gompertz-Harmonic sweep: s = M-1 ... 0
    M = len(model.layers)
    layer_grads: List[Dict[str, float]] = [{} for _ in range(M)]

    for s in range(M - 1, -1, -1):
        L = model.layers[s]
        c = layer_caches[s]

        P_s = c["P"]
        J = c["J"]
        W_s = c["W_s"]
        F = c["F"]

        # Exact Porter-Duff occlusion recurrence
        dot_H_F = np.sum(dL_dH * F, axis=0)  # (H, W)
        dJ = P_s * (dot_H_F - dP)
        dP = dP * (1.0 - J) + J * dot_H_F

        # Adjoint w.r.t. F_{s,v} = C_v * T * S_light
        dF = dL_dH * W_s[None, :, :]  # (3, H, W)
        TS = c["TS"]
        C_vec = c["C_vec"]

        g_r = float(np.sum(dF[0] * TS))
        g_g = float(np.sum(dF[1] * TS))
        g_b = float(np.sum(dF[2] * TS))

        dTS = dF[0] * C_vec[0] + dF[1] * C_vec[1] + dF[2] * C_vec[2]
        dT = dTS * c["S_light"]
        dS_light = dTS * c["T"]

        # Texture T = 1 + tex_amp * cos(tex_wx * un + tex_wy * vn + tex_phi)
        un = c["un"]
        vn = c["vn"]
        sin_tex = np.sin(c["tex_arg"])
        g_tex_amp = float(np.sum(dT * c["cos_tex"]))
        d_tex_arg = -dT * L.tex_amp * sin_tex
        g_tex_wx = float(np.sum(d_tex_arg * un))
        g_tex_wy = float(np.sum(d_tex_arg * vn))
        g_tex_phi = float(np.sum(d_tex_arg))

        # 3D Relief Lighting S_light = 1 + shade_amp*(lx*un + ly*vn) + spec_amp*exp(-3.5*((un-lx)^2+(vn-ly)^2))
        spec_e = c["spec_e"]
        du_l = c["du_l"]
        dv_l = c["dv_l"]
        g_shade_amp = float(np.sum(dS_light * c["dot_l"]))
        g_spec_amp = float(np.sum(dS_light * spec_e))
        d_spec_d2 = dS_light * L.spec_amp * spec_e * (-3.5)
        g_lx = float(np.sum(dS_light * L.shade_amp * un + d_spec_d2 * (-2.0 * du_l)))
        g_ly = float(np.sum(dS_light * L.shade_amp * vn + d_spec_d2 * (-2.0 * dv_l)))

        # Accumulate indirect spatial gradients into (un, vn) from texture + lighting
        d_un = d_tex_arg * L.tex_wx + dS_light * (L.shade_amp * L.lx) + d_spec_d2 * (2.0 * du_l)
        d_vn = d_tex_arg * L.tex_wy + dS_light * (L.shade_amp * L.ly) + d_spec_d2 * (2.0 * dv_l)

        # Gompertz Double-Exponential Kernel J = alpha * Phi, Phi = exp(-exp(clip(k_eff * g_val)))
        Phi = c["Phi"]
        g_alpha = float(np.sum(dJ * Phi))
        dPhi = dJ * c["alpha_eff"]
        mask_active = (c["z_raw"] > -EXP_CLIP) & (c["z_raw"] < EXP_CLIP)
        dPhi_dz = np.where(mask_active, -c["ez"] * Phi, 0.0)
        dz = dPhi * dPhi_dz
        dg = dz * c["k_eff"]
        g_k_mul = float(np.sum(dz * (k_glob * c["g_val"])))

        # Superellipse g_val = (au^p + av^p + eps)^(1/p) - 1
        p = c["p"]
        rad = c["rad"]
        S_pow = c["S_pow"]
        au = c["au"]
        av = c["av"]
        au_p = c["au_p"]
        av_p = c["av_p"]

        # d(rad)/d(S_pow) = (1/p) * rad / S_pow
        dS_pow = dg * (rad / (p * S_pow))
        # d(au^p)/d(un) = p * au^(p-2) * un
        d_un += dS_pow * p * (au_p / (au * au)) * un
        d_vn += dS_pow * p * (av_p / (av * av)) * vn

        # Exact derivative w.r.t. superellipse exponent p_exp:
        # rad = exp((1/p) * log(S_pow)) => drad/dp = rad * ( -log(S_pow)/p^2 + (au^p*log(au) + av^p*log(av)) / (p * S_pow) )
        log_S = np.log(S_pow)
        dS_dp = au_p * np.log(au) + av_p * np.log(av)
        drad_dp = rad * (-log_S / (p * p) + dS_dp / (p * S_pow))
        g_p_exp = float(np.sum(dg * drad_dp))

        # Scaling un = u / rx, vn = vw / ry
        rx = c["rx"]
        ry = c["ry"]
        u = c["u"]
        vw = c["vw"]
        g_rx = float(np.sum(d_un * (-u / (rx * rx))))
        g_ry = float(np.sum(d_vn * (-vw / (ry * ry))))
        d_u = d_un / rx
        d_vw = d_vn / ry

        # Trigonometric curvature warp vw = v + warp_a * sin(warp_f * u + warp_p)
        cos_w = np.cos(c["w_arg"])
        g_warp_a = float(np.sum(d_vw * c["sin_w"]))
        d_w_arg = d_vw * L.warp_a * cos_w
        g_warp_f = float(np.sum(d_w_arg * u))
        g_warp_p = float(np.sum(d_w_arg))
        d_u += d_w_arg * L.warp_f
        d_v = d_vw

        # Rotation u = dx*ct + dy*st, v = -dx*st + dy*ct
        ct = c["ct"]
        st = c["st"]
        dx = c["dx"]
        dy = c["dy"]
        # du/dtheta = -dx*st + dy*ct = v; dv/dtheta = -dx*ct - dy*st = -u
        g_theta = float(np.sum(d_u * c["v"] - d_v * u))
        # du/dcx = -ct, dv/dcx = st; du/dcy = -st, dv/dcy = -ct
        g_cx = float(np.sum(-d_u * ct + d_v * st))
        g_cy = float(np.sum(-d_u * st - d_v * ct))

        layer_grads[s] = {
            "cx": g_cx, "cy": g_cy, "rx": g_rx, "ry": g_ry,
            "theta": g_theta, "p_exp": g_p_exp,
            "warp_a": g_warp_a, "warp_f": g_warp_f, "warp_p": g_warp_p,
            "k_mul": g_k_mul, "alpha": g_alpha,
            "r": g_r, "g": g_g, "b": g_b,
            "lx": g_lx, "ly": g_ly, "shade_amp": g_shade_amp, "spec_amp": g_spec_amp,
            "tex_amp": g_tex_amp, "tex_wx": g_tex_wx, "tex_wy": g_tex_wy, "tex_phi": g_tex_phi,
        }

    return {
        "background": {
            "c0": grad_bg_c0,
            "cx": grad_bg_cx,
            "cy": grad_bg_cy,
            "amp": grad_bg_amp,
            "wx": grad_bg_wx,
            "wy": grad_bg_wy,
            "phi": grad_bg_phi,
        },
        "layers": layer_grads,
    }


def compute_exact_layer_spatial_gradient(
    L: GompertzHarmonicLayer,
    X: np.ndarray,
    Y: np.ndarray,
    t_phase: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Exact analytical implicit field g_s(x, y) and its exact closed-form partial
    derivatives (dg_s/dx, dg_s/dy) — ZERO finite differences, ZERO OpenCV!
    """
    cx_t, cy_t, th_t = _layer_effective_pose(L, t_phase)
    ct = math.cos(th_t)
    st = math.sin(th_t)
    dx = X - cx_t
    dy = Y - cy_t
    u = dx * ct + dy * st
    v = -dx * st + dy * ct

    w_arg = L.warp_f * u + L.warp_p
    vw = v + L.warp_a * np.sin(w_arg)
    dvw_du = L.warp_a * L.warp_f * np.cos(w_arg)

    rx = max(L.rx, 1e-4)
    ry = max(L.ry, 1e-4)
    un = u / rx
    vn = vw / ry
    au = np.sqrt(un * un + EPS_ABS)
    av = np.sqrt(vn * vn + EPS_ABS)
    p = max(L.p_exp, 1.05)

    au_p = np.power(au, p)
    av_p = np.power(av, p)
    S_pow = au_p + av_p + EPS_RAD
    rad = np.power(S_pow, 1.0 / p)
    g_val = rad - 1.0

    # dg/dun and dg/dvn
    coef = rad / S_pow
    dg_dun = coef * (au_p / (au * au)) * un
    dg_dvn = coef * (av_p / (av * av)) * vn

    dg_du = dg_dun / rx + (dg_dvn / ry) * dvw_du
    dg_dv = dg_dvn / ry

    # Chain rule through rotation (u = dx*ct + dy*st, v = -dx*st + dy*ct)
    dg_dx = dg_du * ct - dg_dv * st
    dg_dy = dg_du * st + dg_dv * ct
    return g_val, dg_dx, dg_dy


def compute_exact_analytical_relief_normals(
    model: HDGESModel,
    X: np.ndarray,
    Y: np.ndarray,
    kappa_global: Optional[float] = None,
    t_phase: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute exact analytical 3D heightfield Z(x, y) and exact analytical unit
    surface normals (nx, ny, nz) directly from symbolic partial derivatives.
    """
    k_glob = model.kappa_global if kappa_global is None else float(kappa_global)
    H_grid, W_grid = X.shape
    Z = np.zeros((H_grid, W_grid), dtype=np.float64)
    dZ_dx = np.zeros((H_grid, W_grid), dtype=np.float64)
    dZ_dy = np.zeros((H_grid, W_grid), dtype=np.float64)
    P = np.ones((H_grid, W_grid), dtype=np.float64)
    dP_dx = np.zeros((H_grid, W_grid), dtype=np.float64)
    dP_dy = np.zeros((H_grid, W_grid), dtype=np.float64)

    for idx, L in enumerate(model.layers):
        g_val, dg_dx, dg_dy = compute_exact_layer_spatial_gradient(L, X, Y, t_phase=t_phase)
        k_eff = k_glob * max(L.k_mul, 0.1)
        z_raw = k_eff * g_val
        z_clamped = np.clip(z_raw, -EXP_CLIP, EXP_CLIP)
        ez = np.exp(z_clamped)
        J = np.exp(-ez)
        dJ_dg = np.where((z_raw > -EXP_CLIP) & (z_raw < EXP_CLIP), -k_eff * ez * J, 0.0)
        dJ_dx = dJ_dg * dg_dx
        dJ_dy = dJ_dg * dg_dy

        # Smooth dome height inside each Gompertz layer: z_s(x,y) = z_base + 0.25 * clamp(1 - (g+1)^2, 0, 1)
        r_sq = np.clip((g_val + 1.0) ** 2, 0.0, 1.5)
        dome = np.exp(-2.0 * r_sq)
        ddome_dg = -4.0 * (g_val + 1.0) * dome
        z_base = L.z_depth * (1.0 - 0.35 * idx / max(len(model.layers), 1))
        h_layer = z_base + 0.28 * dome
        dh_dx = 0.28 * ddome_dg * dg_dx
        dh_dy = 0.28 * ddome_dg * dg_dy

        W_s = P * J
        dW_dx = dP_dx * J + P * dJ_dx
        dW_dy = dP_dy * J + P * dJ_dy

        Z += W_s * h_layer
        dZ_dx += dW_dx * h_layer + W_s * dh_dx
        dZ_dy += dW_dy * h_layer + W_s * dh_dy

        dP_dx = dP_dx * (1.0 - J) - P * dJ_dx
        dP_dy = dP_dy * (1.0 - J) - P * dJ_dy
        P = P * (1.0 - J)

    norm = np.sqrt(dZ_dx * dZ_dx + dZ_dy * dZ_dy + 1.0)
    nx = -dZ_dx / norm
    ny = -dZ_dy / norm
    nz = 1.0 / norm
    return Z, nx, ny, nz


def verify_autodiff_gradients(seed: int = 42, eps: float = 1e-5) -> Dict[str, float]:
    """
    Self-verification test: compares exact reverse-mode analytical gradients
    against central finite differences across ALL 21 layer parameters and ALL
    background parameters. Must achieve max relative error < 1e-5.
    """
    rng = np.random.default_rng(seed)
    X, Y = make_grid(20, 14)
    target = rng.uniform(0.1, 0.9, size=(3, 14, 20))

    bg = HDGESBackground(
        c0=rng.uniform(0.1, 0.5, 3),
        cx=rng.uniform(-0.1, 0.1, 3),
        cy=rng.uniform(-0.1, 0.1, 3),
        amp=rng.uniform(-0.15, 0.15, (3, 4)),
        wx=rng.uniform(1.0, 5.0, 4),
        wy=rng.uniform(-5.0, 5.0, 4),
        phi=rng.uniform(-2.0, 2.0, 4),
    )
    layers = [
        GompertzHarmonicLayer(
            cx=float(rng.uniform(-0.5, 0.5)),
            cy=float(rng.uniform(-0.4, 0.4)),
            rx=float(rng.uniform(0.25, 0.6)),
            ry=float(rng.uniform(0.2, 0.5)),
            theta=float(rng.uniform(-1.0, 1.0)),
            p_exp=float(rng.uniform(1.6, 3.5)),
            warp_a=float(rng.uniform(-0.15, 0.15)),
            warp_f=float(rng.uniform(2.0, 6.0)),
            warp_p=float(rng.uniform(-1.0, 1.0)),
            k_mul=float(rng.uniform(0.6, 1.5)),
            alpha=float(rng.uniform(0.35, 0.95)),
            r=float(rng.uniform(0.2, 0.9)),
            g=float(rng.uniform(0.2, 0.9)),
            b=float(rng.uniform(0.2, 0.9)),
            lx=float(rng.uniform(-0.6, 0.6)),
            ly=float(rng.uniform(-0.6, 0.6)),
            shade_amp=float(rng.uniform(-0.3, 0.3)),
            spec_amp=float(rng.uniform(0.05, 0.4)),
            tex_amp=float(rng.uniform(-0.25, 0.25)),
            tex_wx=float(rng.uniform(-8.0, 8.0)),
            tex_wy=float(rng.uniform(-8.0, 8.0)),
            tex_phi=float(rng.uniform(-1.5, 1.5)),
        )
        for _ in range(3)
    ]
    model = HDGESModel(kappa_global=6.0, background=bg, layers=layers)

    H_out, cache = forward_hdges(model, X, Y, kappa_global=6.0, return_cache=True)
    diff = H_out - target
    dL_dH = (2.0 / diff.size) * diff
    grads = backward_hdges(model, cache, dL_dH)

    def eval_loss(m: HDGESModel) -> float:
        h, _ = forward_hdges(m, X, Y, kappa_global=6.0, return_cache=False)
        return float(np.mean((h - target) ** 2))

    max_rel_err = 0.0
    worst_param = ""
    # Check every layer parameter
    for s in range(len(layers)):
        for pname in LAYER_PARAM_NAMES:
            orig = getattr(model.layers[s], pname)
            setattr(model.layers[s], pname, orig + eps)
            lp = eval_loss(model)
            setattr(model.layers[s], pname, orig - eps)
            lm = eval_loss(model)
            setattr(model.layers[s], pname, orig)
            num_g = (lp - lm) / (2.0 * eps)
            ana_g = grads["layers"][s][pname]
            rel = abs(ana_g - num_g) / max(abs(ana_g), abs(num_g), 1e-8)
            if rel > max_rel_err:
                max_rel_err = rel
                worst_param = f"layer[{s}].{pname}"

    # Check every background array entry
    for bname in ("c0", "cx", "cy", "amp", "wx", "wy", "phi"):
        arr = getattr(model.background, bname)
        g_arr = grads["background"][bname]
        for idx in np.ndindex(arr.shape):
            orig = arr[idx]
            arr[idx] = orig + eps
            lp = eval_loss(model)
            arr[idx] = orig - eps
            lm = eval_loss(model)
            arr[idx] = orig
            num_g = (lp - lm) / (2.0 * eps)
            ana_g = float(g_arr[idx])
            rel = abs(ana_g - num_g) / max(abs(ana_g), abs(num_g), 1e-8)
            if rel > max_rel_err:
                max_rel_err = rel
                worst_param = f"bg.{bname}{idx}"

    # Also verify exact spatial partial derivatives (dg/dx, dg/dy)
    g0, dgx, dgy = compute_exact_layer_spatial_gradient(model.layers[0], X, Y)
    gp_x, _, _ = compute_exact_layer_spatial_gradient(model.layers[0], X + eps, Y)
    gm_x, _, _ = compute_exact_layer_spatial_gradient(model.layers[0], X - eps, Y)
    gp_y, _, _ = compute_exact_layer_spatial_gradient(model.layers[0], X, Y + eps)
    gm_y, _, _ = compute_exact_layer_spatial_gradient(model.layers[0], X, Y - eps)
    num_dgx = (gp_x - gm_x) / (2.0 * eps)
    num_dgy = (gp_y - gm_y) / (2.0 * eps)
    spatial_err = float(
        max(
            np.max(np.abs(dgx - num_dgx) / np.maximum(np.abs(dgx), 1e-6)),
            np.max(np.abs(dgy - num_dgy) / np.maximum(np.abs(dgy), 1e-6)),
        )
    )

    return {
        "max_param_rel_err": float(max_rel_err),
        "worst_param": worst_param,
        "max_spatial_grad_rel_err": spatial_err,
        "passed": bool(max_rel_err < 1e-5 and spatial_err < 1e-5),
    }


if __name__ == "__main__":
    res = verify_autodiff_gradients()
    print("H-DGES Exact Autodiff Gradient Check:", res)
    assert res["passed"], f"Autodiff gradient check failed: {res}"
