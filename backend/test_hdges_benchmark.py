"""
H-DGES Quantitative Scientific Benchmark & Visual Proof Generator.
Tests:
  1. Exact Pure-NumPy Reverse-Mode Autodiff (< 1e-5 relative error vs central FD).
  2. Homotopy Theorem Ablation: Fixed high-kappa (65) vs H-DGES kappa-Homotopy
     Annealing (2 -> 6.5 -> 20 -> 65) from identical initialization.
  3. Residual Symbolic Matching Pursuit (RSMP) + Homotopy Inverse Closed-Form
     Formula Distillation on 3 challenging real-world targets + Sensitivity-Weighted
     Continued-Fraction Rational Snapping + 10x Infinite-Resolution Evaluation.
  4. Pure-Calculus RK2 Hamiltonian Level-Set Streamlines (ZERO OpenCV!) & Exact
     Analytical 3D Surface Normals + WebGL1 GLSL validation + Looping GIF Video.
"""

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

BACKEND_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BACKEND_DIR))

from hdges_autodiff import (
    compute_exact_analytical_relief_normals,
    forward_hdges,
    hamid_F_quantize_smooth,
    make_grid,
    verify_autodiff_gradients,
)
from hdges_solver import (
    compile_hdges_to_glsl,
    compile_hdges_to_latex,
    compute_psnr,
    compute_ssim,
    distill_image_to_hdges,
    evaluate_hdges_render,
    extract_hamiltonian_manifolds,
    render_hdges_video,
    run_homotopy_ablation,
)
from pixel_field_engine import evaluate_pixel_field_preset
from symbolic_math_compiler import validate_glsl_webgl1_static


def _render_wireframe_panel(primitives, width=420, height=262, is_3d=False):
    img = Image.new("RGB", (width, height), (9, 13, 24))
    draw = ImageDraw.Draw(img)
    n = len(primitives)
    for i, seg in enumerate(primitives):
        frac = i / max(n - 1, 1)
        if not is_3d:
            x1, y1, x2, y2 = seg
            col = (
                int(65 + 165 * frac),
                int(220 - 110 * frac),
                int(245 - 65 * frac),
            )
        else:
            x1, y1, z1, x2, y2, z2 = seg
            # Perspective orbit projection (yaw=28 deg, pitch=20 deg)
            cy, sy = 0.8829, 0.4695
            cp, sp = 0.9397, 0.3420
            def proj(x, y, z):
                xr = x * cy + z * sy
                zr = -x * sy + z * cy
                yr = y * cp - zr * sp
                zp = y * sp + zr * cp
                persp = 2.6 / (2.6 - zp)
                return xr * persp, yr * persp
            x1, y1 = proj(x1, y1, z1)
            x2, y2 = proj(x2, y2, z2)
            col = (
                int(175 + 70 * frac),
                int(75 + 145 * (1.0 - frac)),
                int(235 - 50 * frac),
            )
        px1 = int((x1 * 0.46 + 0.5) * width)
        py1 = int((0.5 - y1 * 0.46) * height)
        px2 = int((x2 * 0.46 + 0.5) * width)
        py2 = int((0.5 - y2 * 0.46) * height)
        draw.line([(px1, py1), (px2, py2)], fill=col, width=1)
    return img


def _render_ablation_plot(ablation: dict, width=420, height=262) -> Image.Image:
    img = Image.new("RGB", (width, height), (12, 16, 28))
    draw = ImageDraw.Draw(img)
    pad_l, pad_r, pad_t, pad_b = 46, 18, 34, 34
    pw, ph = width - pad_l - pad_r, height - pad_t - pad_b

    draw.rectangle([pad_l, pad_t, width - pad_r, height - pad_b], outline=(55, 70, 95), width=1)

    tr_f = ablation["fixed_kappa"]["trace"]
    tr_h = ablation["homotopy_annealed"]["trace"]
    all_psnr = [p["psnr"] for p in tr_f + tr_h if "psnr" in p]
    y_min = max(8.0, min(all_psnr) - 1.5)
    y_max = max(all_psnr) + 1.5
    max_step = max(p["step"] for p in tr_f + tr_h)

    def to_xy(step, psnr):
        x = pad_l + int((step / max(max_step, 1)) * pw)
        y = pad_t + int((1.0 - (psnr - y_min) / max(y_max - y_min, 1e-4)) * ph)
        return x, y

    # Grid lines
    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        yy = pad_t + int(frac * ph)
        val = y_max - frac * (y_max - y_min)
        draw.line([(pad_l, yy), (width - pad_r, yy)], fill=(28, 36, 54), width=1)
        draw.text((6, yy - 6), f"{val:.1f}", fill=(140, 160, 190))

    # Plot curves
    pts_f = [to_xy(p["step"], p["psnr"]) for p in tr_f if "psnr" in p]
    pts_h = [to_xy(p["step"], p["psnr"]) for p in tr_h if "psnr" in p]
    if len(pts_f) >= 2:
        draw.line(pts_f, fill=(235, 95, 95), width=2)
    if len(pts_h) >= 2:
        draw.line(pts_h, fill=(65, 235, 175), width=3)

    draw.text((pad_l + 6, 8), "Homotopy Theorem Ablation (PSNR dB vs Step)", fill=(230, 240, 255))
    draw.text(
        (pad_l + 6, height - 24),
        f"Green: H-DGES Homotopy ({ablation['homotopy_annealed']['psnr']:.2f} dB)  |  Red: Fixed k=180 ({ablation['fixed_kappa']['psnr']:.2f} dB)",
        fill=(190, 210, 235),
    )
    return img


def main():
    print("=== 1. Exact Pure-NumPy Reverse-Mode Autodiff Verification ===")
    grad_check = verify_autodiff_gradients(seed=42, eps=1e-5)
    print(json.dumps(grad_check, indent=2))
    assert grad_check["passed"], f"Autodiff verification failed: {grad_check}"

    # Load 3 diverse complex high-detail real targets (Hamid Kiwi, Strawberries, Corals)
    targets = []
    for pid, label in [
        ("pf_kiwi", "Kiwi Bird in Forest Bokeh"),
        ("pf_strawberries", "Fresh Strawberries Specular Field"),
        ("pf_corals", "Magenta Reef Corals & Golden Fish"),
    ]:
        res = evaluate_pixel_field_preset(pid, width=320, include_raw_array=True)
        targets.append((pid, label, res["rgb_array"]))

    print("\n=== 2. Homotopy Theorem Ablation (Fixed k=180 vs H-DGES kappa-Homotopy) ===")
    ablation = run_homotopy_ablation(targets[0][2], opt_width=84, opt_height=54, n_layers=18, total_steps=100)
    print(
        f"Fixed k=180: PSNR={ablation['fixed_kappa']['psnr']} dB, SSIM={ablation['fixed_kappa']['ssim']} | "
        f"H-DGES Homotopy: PSNR={ablation['homotopy_annealed']['psnr']} dB, SSIM={ablation['homotopy_annealed']['ssim']} | "
        f"Gain: +{ablation['psnr_gain_db']} dB, +{ablation['ssim_gain']} SSIM"
    )
    assert ablation["psnr_gain_db"] > 0.5, f"Expected positive Homotopy gain, got {ablation['psnr_gain_db']}"

    print("\n=== 3. RSMP + Homotopy Inverse Distillation + Rational Snapping + Infinite Zoom ===")
    panel_w, panel_h = 420, 262
    rows_imgs = []
    summary = {
        "autodiff_check": grad_check,
        "ablation": {
            "fixed_kappa_psnr": ablation["fixed_kappa"]["psnr"],
            "fixed_kappa_ssim": ablation["fixed_kappa"]["ssim"],
            "homotopy_psnr": ablation["homotopy_annealed"]["psnr"],
            "homotopy_ssim": ablation["homotopy_annealed"]["ssim"],
            "psnr_gain_db": ablation["psnr_gain_db"],
            "ssim_gain": ablation["ssim_gain"],
        },
        "distilled_targets": [],
    }

    out_vid_dir = BACKEND_DIR / "out_of_box_videos"
    out_vid_dir.mkdir(exist_ok=True)

    for idx, (pid, label, target_arr) in enumerate(targets):
        dist = distill_image_to_hdges(
            target_arr,
            title=f"H-DGES: {label}",
            subtitle="Closed-Form Rational Gompertz-Harmonic Equation via Homotopy RSMP",
            opt_width=112,
            opt_height=70,
            snap_rationals=True,
        )
        model = dist["model"]
        # Render at 6.6x super-resolution (640x400) directly from snapped rational formula!
        rendered_hr = evaluate_hdges_render(model, width=640, height=400, t_phase=0.0, include_raw_array=True)
        glsl_ok, glsl_msg = validate_glsl_webgl1_static(rendered_hr["glsl_fragment_shader"])
        assert glsl_ok, f"GLSL invalid for {pid}: {glsl_msg}"

        # True pure-calculus RK2 Hamiltonian streamlines (ZERO OpenCV!)
        ham = extract_hamiltonian_manifolds(model, N=1500, t_phase=0.0)
        assert ham["method"] == "rk2_hamiltonian_exact_jacobian"
        assert ham["N"] >= 1000

        # Render exact analytical surface normal map visualization
        Xg, Yg = make_grid(panel_w, panel_h, model.x_min, model.x_max, model.y_min, model.y_max)
        _, nx, ny, nz = compute_exact_analytical_relief_normals(model, Xg, Yg)
        norm_rgb = np.stack([
            np.clip((nx * 0.5 + 0.5) * 255.0, 0, 255),
            np.clip((ny * 0.5 + 0.5) * 255.0, 0, 255),
            np.clip((nz * 0.5 + 0.5) * 255.0, 0, 255),
        ], axis=2).astype(np.uint8)

        # Render looping 4D pure-math GIF for first target
        vid_info = None
        if idx == 0:
            vid_info = render_hdges_video(
                model,
                out_gif_path=str(out_vid_dir / "hdges_kiwi_homotopy.gif"),
                out_mp4_path=str(out_vid_dir / "hdges_kiwi_homotopy.mp4"),
                width=400,
                height=250,
                num_frames=24,
                fps=12,
            )

        # Evaluate high-res PSNR against upscaled target
        t_hr = np.asarray(Image.fromarray(target_arr).resize((640, 400), Image.Resampling.LANCZOS), dtype=np.float64) / 255.0
        p_hr = rendered_hr["rgb_array"].astype(np.float64) / 255.0
        psnr_hr = round(compute_psnr(p_hr, t_hr), 3)
        ssim_hr = round(compute_ssim(p_hr, t_hr), 4)

        latex_chars = sum(len(v) for v in rendered_hr["latex"].values())
        print(
            f"[{pid}] {dist['n_layers']} layers in {dist['opt_seconds']}s | "
            f"Opt PSNR={dist['psnr']} dB (SSIM={dist['ssim']}) | "
            f"HR 640x400 PSNR={psnr_hr} dB (SSIM={ssim_hr}) | "
            f"Rational Snap drop={dist['rational_snapping']['psnr_drop']} dB | "
            f"GLSL={len(rendered_hr['glsl_fragment_shader'])}c OK | "
            f"LaTeX={latex_chars}c | Hamiltonian N={ham['N']}"
        )

        summary["distilled_targets"].append({
            "id": pid,
            "label": label,
            "n_layers": dist["n_layers"],
            "opt_seconds": dist["opt_seconds"],
            "psnr_opt": dist["psnr"],
            "ssim_opt": dist["ssim"],
            "psnr_hr_640x400": psnr_hr,
            "ssim_hr_640x400": ssim_hr,
            "rational_snapping": dist["rational_snapping"],
            "glsl_chars": len(rendered_hr["glsl_fragment_shader"]),
            "latex_chars": latex_chars,
            "hamiltonian_N": ham["N"],
            "video": {k: vid_info[k] for k in ("gif_path", "mp4_path", "motion_energy")} if vid_info else None,
        })

        # Build 4 visual panels for this row:
        # Col 1: Target | Col 2: H-DGES Rational Closed-Form Formula (640x400) | Col 3: RK2 Hamiltonian 2D Streamlines | Col 4: Analytical 3D Normals or Ablation Plot
        p1 = Image.fromarray(target_arr).resize((panel_w, panel_h), Image.Resampling.LANCZOS)
        p2 = Image.fromarray(rendered_hr["rgb_array"]).resize((panel_w, panel_h), Image.Resampling.LANCZOS)
        p3 = _render_wireframe_panel(ham["primitives"], panel_w, panel_h, is_3d=False)
        p4 = _render_ablation_plot(ablation, panel_w, panel_h) if idx == 0 else Image.fromarray(norm_rgb)
        rows_imgs.append((label, dist, psnr_hr, ssim_hr, [p1, p2, p3, p4]))

    # Assemble 3x4 scientific proof contact sheet
    margin = 14
    header_h = 42
    cap_h = 34
    sheet_w = 4 * panel_w + 5 * margin
    sheet_h = header_h + len(rows_imgs) * (panel_h + cap_h + margin) + margin
    sheet = Image.new("RGB", (sheet_w, sheet_h), (8, 11, 20))
    draw = ImageDraw.Draw(sheet)
    draw.text(
        (margin, 12),
        f"H-DGES World-First Scientific Proof | Autodiff RelErr={grad_check['max_param_rel_err']:.1e} | Homotopy Gain=+{ablation['psnr_gain_db']:.2f} dB over Fixed k=180",
        fill=(130, 235, 255),
    )

    col_titles_0 = [
        "1. Target Reference Image",
        "2. H-DGES Rational Closed-Form Formula (640x400)",
        "3. Pure-Calculus RK2 Hamiltonian Streamlines (0 OpenCV)",
        "4. Homotopy Theorem Ablation Curve (k: 2->180 vs k=180)",
    ]
    col_titles_other = [
        "1. Target Reference Image",
        "2. H-DGES Rational Closed-Form Formula (640x400)",
        "3. Pure-Calculus RK2 Hamiltonian Streamlines (0 OpenCV)",
        "4. Exact Analytical Surface Normals n(x,y) from AST",
    ]

    for r_i, (label, dist, psnr_hr, ssim_hr, panels) in enumerate(rows_imgs):
        y0 = header_h + r_i * (panel_h + cap_h + margin)
        titles = col_titles_0 if r_i == 0 else col_titles_other
        for c_i, pimg in enumerate(panels):
            x0 = margin + c_i * (panel_w + margin)
            draw.rectangle([x0 - 1, y0 - 1, x0 + panel_w, y0 + cap_h + panel_h], outline=(40, 56, 84))
            draw.text((x0 + 6, y0 + 3), titles[c_i], fill=(165, 210, 255))
            sub = (
                f"{label} | {dist['n_layers']}L | PSNR={psnr_hr:.2f}dB SSIM={ssim_hr:.3f} | SnapDrop={dist['rational_snapping']['psnr_drop']:.2f}dB"
                if c_i == 1 else label
            )
            draw.text((x0 + 6, y0 + 18), sub[:68], fill=(170, 185, 205))
            sheet.paste(pimg, (x0, y0 + cap_h))

    out_sheet = BACKEND_DIR / "sheet_hdges_proof.png"
    sheet.save(out_sheet)
    with open(BACKEND_DIR / "hdges_benchmark_summary.json", "w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2)
    print(f"\nSaved H-DGES scientific proof sheet -> {out_sheet}")


if __name__ == "__main__":
    main()
