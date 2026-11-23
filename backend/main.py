from typing import Any, Dict, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from ai_architect import (
    generate_3d_raymarch_scene_with_critique,
    generate_hdges_topology_for_prompt,
    generate_math_program_with_passes,
)
from hdges_autodiff import verify_autodiff_gradients
from hdges_solver import (
    distill_image_to_hdges,
    evaluate_hdges_render,
    extract_hamiltonian_manifolds,
    optimize_text_hdges_zero_diffusion,
    run_homotopy_ablation,
)
from master_paradigms_engine import (
    MASTER_PARADIGM_PRESETS,
    validate_raymarch_scene,
)
from math_3d_video_engine import (
    PIXEL_VIDEO_SHOWCASE_PRESETS,
    RAYMARCH_CINEMA_SHOWCASE_PRESETS,
    render_math_video_response,
    render_raymarch_video_response,
    sanitize_math_text,
)
from cinema_director_engine import SAMPLE_EPIC_5_ACT_SCRIPT
from math_engine import (
    ALL_PRESETS,
    HAMID_PRESETS,
    SYNONYM_MAP,
    direct_long_script_cinema_film,
    evaluate_custom_formulas,
    evaluate_preset,
    generate_math_art_from_prompt,
)
from pixel_field_engine import evaluate_pixel_field_preset
from symbolic_math_compiler import validate_math_program

app = FastAPI(
    title="DesignWithMath — 3D Raymarch Cinema, H-DGES & Hamid Naderi Yeganeh Engine",
    version="6.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PromptRequest(BaseModel):
    prompt: str
    N: int = 2000
    harmonics: int = 54
    mode: str = "lines"
    weave_style: str = "harmonic_web"
    use_ai: bool = True
    t_phase: float = 0.0
    num_frames: int = 0
    refine: bool = True
    bypass_cache: bool = False


class PresetRequest(BaseModel):
    preset_id: str
    N: Optional[int] = None
    t_phase: float = 0.0


class CustomFormulaRequest(BaseModel):
    mode: str = "lines"
    N: int = 2000
    formulas: Dict[str, str]


class RenderMathVideoRequest(BaseModel):
    prompt: str = ""
    preset_id: Optional[str] = None
    mode: str = "pixel_video"
    width: int = 480
    height: int = 300
    num_frames: int = 32
    fps: int = 16
    refine: bool = True
    bypass_cache: bool = False
    include_mp4: bool = True
    math_program: Optional[Dict[str, Any]] = None
    raymarch_scene: Optional[Dict[str, Any]] = None


@app.get("/api/health")
def health_check():
    return {
        "status": "ok",
        "engine": "DesignWithMath 3D Raymarch Cinema, Symbolic Fourier-Hamid & Differential-Geometric Engine",
    }


@app.get("/api/presets")
def get_presets():
    return {
        "presets": [
            {
                "id": p["id"],
                "name": p["name"],
                "author": p["author"],
                "category": p["category"],
                "mode": p["mode"],
                "N": p["N"],
                "description": p["description"],
                "latex": p["latex"],
            }
            for p in ALL_PRESETS
        ],
        "master_paradigm_presets": [
            {
                "id": p["id"],
                "name": p["name"],
                "author": p["author"],
                "category": p["category"],
                "mode": p["mode"],
                "N": p["N"],
                "description": p["description"],
                "latex": p["latex"],
            }
            for p in MASTER_PARADIGM_PRESETS
        ],
        "pixel_video_presets": [
            {
                "id": p["id"],
                "name": p["name"],
                "author": p["author"],
                "category": p["category"],
                "mode": p["mode"],
                "N": p["N"],
                "promptFallback": p["promptFallback"],
                "description": p["description"],
                "latex": p["latex"],
            }
            for p in PIXEL_VIDEO_SHOWCASE_PRESETS
        ],
        "sample_prompts": [
            {"label": "Golden Phoenix & Sapphire Ocean (3D Cinema)", "prompt": "a golden phoenix rising over turbulent sapphire ocean waves", "mode": "raymarch_cinema", "use_ai": True},
            {"label": "Hyperbolic Gyroid Cathedral (3D Cinema)", "prompt": "a hyperbolic bronze cathedral with glowing gyroid pillars and god rays", "mode": "raymarch_cinema", "use_ai": True},
            {"label": "Bioluminescent Dragonfish & Coral (3D Cinema)", "prompt": "a bioluminescent deep-sea dragonfish surrounded by differential-growth coral", "mode": "raymarch_cinema", "use_ai": True},
            {"label": "Non-Euclidean Astrolabe & Saturn (3D Cinema)", "prompt": "a non-Euclidean celestial astrolabe with orbiting Saturn rings", "mode": "raymarch_cinema", "use_ai": True},
            {"label": "Roaring Bengal Tiger (4D Video)", "prompt": "a roaring Bengal tiger face in lush jungle", "mode": "pixel_video", "use_ai": True},
            {"label": "Bioluminescent Jellyfish (4D Video)", "prompt": "a glowing bioluminescent jellyfish swimming in deep ocean", "mode": "pixel_video", "use_ai": True},
            {"label": "Lighthouse & Aurora (4D Video)", "prompt": "a crystal lighthouse on a rocky cliff under dancing northern lights aurora", "mode": "pixel_video", "use_ai": True},
            {"label": "Astronaut & Saturn Rings (4D Video)", "prompt": "an astronaut floating beside Saturn golden rings in cosmic space", "mode": "pixel_video", "use_ai": True},
            {"label": "Snowy Peak at Golden Sunset", "prompt": "snowy mountain peak at golden sunset with pine trees", "mode": "pixel_field", "use_ai": True},
            {"label": "Faisal Mosque & Crescent", "prompt": "Faisal Mosque Islamabad with crescent moon", "mode": "lines", "use_ai": True},
            {"label": "3D Rose Corolla", "prompt": "Blooming rose flower corolla", "mode": "3d_surface", "use_ai": True},
            {"label": "4D Flapping Shaheen", "prompt": "Shaheen eagle in flight", "mode": "math_video", "use_ai": True},
        ],
    }


@app.post("/api/preset")
def load_preset(req: PresetRequest):
    try:
        return evaluate_preset(req.preset_id, custom_N=req.N, t_phase=float(req.t_phase))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/generate-from-prompt")
def api_generate_from_prompt(req: PromptRequest):
    if not req.prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt cannot be empty.")
    try:
        N = max(100, min(5000, req.N))
        harmonics = max(2, min(120, req.harmonics))
        num_frames = max(0, min(24, int(req.num_frames)))
        return generate_math_art_from_prompt(
            prompt=req.prompt,
            N=N,
            num_harmonics=harmonics,
            mode=req.mode,
            weave_style=req.weave_style,
            use_ai=req.use_ai,
            t_phase=float(req.t_phase),
            num_frames=num_frames,
            refine=bool(req.refine),
            bypass_cache=bool(req.bypass_cache),
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/render-math-video")
def api_render_math_video(req: RenderMathVideoRequest):
    try:
        pid = (req.preset_id or "").strip()

        # 1. Stage-1 Master Paradigm preset (`mp_*`) looping 3D raymarch GIF/MP4
        if pid and any(p["id"] == pid for p in MASTER_PARADIGM_PRESETS):
            return render_raymarch_video_response(
                scene_or_preset=pid,
                width=req.width,
                height=req.height,
                num_frames=req.num_frames,
                fps=req.fps,
                include_mp4=bool(req.include_mp4),
                refinement_passes=2,
            )

        # 2. Stage-2 Raymarch Cinema showcase preset (`rcin_*`)
        rcin_match = next((p for p in RAYMARCH_CINEMA_SHOWCASE_PRESETS if p["id"] == pid), None) if pid else None
        if rcin_match is not None:
            pass_res = generate_3d_raymarch_scene_with_critique(
                rcin_match["promptFallback"],
                refine=bool(req.refine),
                bypass_cache=bool(req.bypass_cache),
            )
            scene = pass_res.get("scene") or pass_res.get("program")
            return render_raymarch_video_response(
                scene_or_preset=scene,
                width=req.width,
                height=req.height,
                num_frames=req.num_frames,
                fps=req.fps,
                include_mp4=bool(req.include_mp4),
                refinement_passes=int(pass_res.get("refinement_passes", 2)),
                visual_score=pass_res.get("visual_score"),
            )

        # 3. Direct 3D raymarch scene or `mode == "raymarch_cinema"` prompt
        if isinstance(req.raymarch_scene, dict) and req.raymarch_scene:
            ok, err = validate_raymarch_scene(req.raymarch_scene)
            if not ok:
                raise HTTPException(status_code=400, detail=f"Invalid raymarch_scene: {err}")
            return render_raymarch_video_response(
                scene_or_preset=req.raymarch_scene,
                width=req.width,
                height=req.height,
                num_frames=req.num_frames,
                fps=req.fps,
                include_mp4=bool(req.include_mp4),
                refinement_passes=2,
            )

        if req.mode == "raymarch_cinema":
            prompt_text = req.prompt.strip()
            if not prompt_text:
                raise HTTPException(status_code=400, detail="Either prompt, preset_id, or raymarch_scene is required for raymarch_cinema.")
            if any(p["id"] == prompt_text for p in MASTER_PARADIGM_PRESETS):
                return render_raymarch_video_response(
                    scene_or_preset=prompt_text,
                    width=req.width,
                    height=req.height,
                    num_frames=req.num_frames,
                    fps=req.fps,
                    include_mp4=bool(req.include_mp4),
                    refinement_passes=2,
                )
            pass_res = generate_3d_raymarch_scene_with_critique(
                prompt_text,
                refine=bool(req.refine),
                bypass_cache=bool(req.bypass_cache),
            )
            scene = pass_res.get("scene") or pass_res.get("program")
            if not isinstance(scene, dict):
                raise HTTPException(status_code=400, detail="Unable to synthesize 3D raymarch scene for video.")
            return render_raymarch_video_response(
                scene_or_preset=scene,
                width=req.width,
                height=req.height,
                num_frames=req.num_frames,
                fps=req.fps,
                include_mp4=bool(req.include_mp4),
                refinement_passes=int(pass_res.get("refinement_passes", 1)),
                visual_score=pass_res.get("visual_score"),
            )

        # 4. Existing Phase-2 4D Color Video (`pixel_video` / `pvid_*` / `math_program`)
        pvid_match = next((p for p in PIXEL_VIDEO_SHOWCASE_PRESETS if p["id"] == pid), None) if pid else None
        effective_prompt = pvid_match["promptFallback"] if pvid_match is not None else req.prompt.strip()

        prog: Optional[Dict[str, Any]] = None
        refinement_passes = 1
        visual_score: Optional[float] = None

        if isinstance(req.math_program, dict) and req.math_program:
            ok, err = validate_math_program(req.math_program)
            if not ok:
                raise HTTPException(status_code=400, detail=f"Invalid math_program: {err}")
            prog = req.math_program
        else:
            if not effective_prompt:
                raise HTTPException(status_code=400, detail="Either prompt, preset_id, or math_program is required.")
            pass_res = generate_math_program_with_passes(
                effective_prompt,
                refine=bool(req.refine),
                bypass_cache=bool(req.bypass_cache),
            )
            prog = pass_res.get("program")
            refinement_passes = int(pass_res.get("refinement_passes", 1))
            visual_score = pass_res.get("visual_score")

        if not isinstance(prog, dict):
            raise HTTPException(status_code=400, detail="Unable to compile symbolic math program for video.")

        return render_math_video_response(
            prog=prog,
            width=req.width,
            height=req.height,
            num_frames=req.num_frames,
            fps=req.fps,
            include_mp4=bool(req.include_mp4),
            refinement_passes=refinement_passes,
            visual_score=visual_score,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/evaluate-custom")
def api_evaluate_custom(req: CustomFormulaRequest):
    try:
        return evaluate_custom_formulas(
            mode=req.mode,
            N=req.N,
            formulas=req.formulas,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ============================================================================
# PHASE 3 WORLD-FIRST H-DGES ENDPOINTS (ZERO-DIFFUSION TEXT + INVERSE DISTILL)
# ============================================================================

class HDGESTextRequest(BaseModel):
    prompt: str
    width: int = 640
    height: int = 400
    N: int = 1600
    snap_rationals: bool = True


class HDGESDistillRequest(BaseModel):
    target_preset_id: str = "pf_kiwi"
    width: int = 640
    height: int = 400
    N: int = 1500
    snap_rationals: bool = True
    run_ablation: bool = False


@app.post("/api/hdges/text-synthesize")
def api_hdges_text_synthesize(req: HDGESTextRequest):
    if not req.prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt cannot be empty.")
    try:
        spec = generate_hdges_topology_for_prompt(req.prompt)
        opt = optimize_text_hdges_zero_diffusion(
            spec, opt_width=84, opt_height=52, snap_rationals=bool(req.snap_rationals)
        )
        model = opt["model"]
        model.title = sanitize_math_text(model.title, "H-DGES Analytical Field")
        model.subtitle = sanitize_math_text(
            model.subtitle, "Homotopy-Annealed Gompertz-Harmonic Equation H_v(x, y, t)"
        )
        render = evaluate_hdges_render(model, width=min(960, max(320, req.width)), height=min(600, max(200, req.height)))
        ham = extract_hamiltonian_manifolds(model, N=min(2400, max(400, req.N)))
        return {
            **render,
            "mode": "hdges",
            "hdges_submode": "zero_diffusion_text",
            "primitives": ham["primitives"],
            "primitives_3d": ham["primitives_3d"],
            "time_harmonics": ham["time_harmonics"],
            "hamiltonian_method": ham["method"],
            "relief_normal_mean_nz": ham["relief_normal_mean_nz"],
            "homotopy_trace": opt["trace"],
            "rational_snapping": opt["rational_snapping"],
            "opt_seconds": opt["opt_seconds"],
            "refinement_passes": 4,
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/hdges/distill")
def api_hdges_distill(req: HDGESDistillRequest):
    try:
        target_res = evaluate_pixel_field_preset(
            req.target_preset_id, width=320, include_raw_array=True
        )
        target_arr = target_res["rgb_array"]
        clean_name = sanitize_math_text(str(target_res.get("title") or req.target_preset_id), "Analytical Target")
        dist = distill_image_to_hdges(
            target_arr,
            title=f"H-DGES Formula: {clean_name}",
            subtitle="Closed-Form Rational Gompertz-Harmonic Equation via Homotopy RSMP",
            opt_width=104,
            opt_height=65,
            snap_rationals=bool(req.snap_rationals),
        )
        model = dist["model"]
        render = evaluate_hdges_render(
            model,
            width=min(960, max(320, req.width)),
            height=min(600, max(200, req.height)),
        )
        ham = extract_hamiltonian_manifolds(model, N=min(2000, max(400, req.N)))
        ablation_out = None
        if req.run_ablation:
            ab = run_homotopy_ablation(target_arr, opt_width=80, opt_height=50, n_layers=18, total_steps=96)
            ablation_out = {
                "fixed_kappa_psnr": ab["fixed_kappa"]["psnr"],
                "fixed_kappa_ssim": ab["fixed_kappa"]["ssim"],
                "homotopy_psnr": ab["homotopy_annealed"]["psnr"],
                "homotopy_ssim": ab["homotopy_annealed"]["ssim"],
                "psnr_gain_db": ab["psnr_gain_db"],
                "ssim_gain": ab["ssim_gain"],
                "fixed_trace": ab["fixed_kappa"]["trace"],
                "homotopy_trace": ab["homotopy_annealed"]["trace"],
            }
        return {
            **render,
            "mode": "hdges",
            "hdges_submode": "inverse_distillation",
            "target_preset_id": req.target_preset_id,
            "target_image_base64": target_res["image_base64"],
            "psnr": dist["psnr"],
            "ssim": dist["ssim"],
            "psnr_float": dist["psnr_float"],
            "rational_snapping": dist["rational_snapping"],
            "homotopy_trace": dist["trace"],
            "opt_seconds": dist["opt_seconds"],
            "primitives": ham["primitives"],
            "primitives_3d": ham["primitives_3d"],
            "time_harmonics": ham["time_harmonics"],
            "hamiltonian_method": ham["method"],
            "relief_normal_mean_nz": ham["relief_normal_mean_nz"],
            "ablation": ablation_out,
            "refinement_passes": 4,
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/hdges/verify-calculus")
def api_hdges_verify_calculus():
    """Live mathematical proof endpoint verifying exact analytical autodiff gradients."""
    return verify_autodiff_gradients(seed=42, eps=1e-5)


# ============================================================================
# STAGE 3: LONG-SCRIPT MULTI-ACT MATHEMATICAL CINEMA DIRECTOR ENDPOINTS
# ============================================================================

class CinemaScriptRequest(BaseModel):
    script: str
    width: int = 520
    height: int = 325
    fps: int = 12
    frames_per_act: int = 16
    transition_frames: int = 6
    max_acts: int = 6
    refine_acts: bool = True


@app.get("/api/cinema/sample-script")
def api_cinema_sample_script():
    return {
        "title": "The Geometry of Creation — 5-Act Mathematical Documentary",
        "script": SAMPLE_EPIC_5_ACT_SCRIPT,
    }


@app.post("/api/cinema/generate-from-script")
def api_cinema_generate_from_script(req: CinemaScriptRequest):
    if not req.script or not req.script.strip():
        raise HTTPException(status_code=400, detail="Script cannot be empty.")
    try:
        return direct_long_script_cinema_film(
            script=req.script,
            width=req.width,
            height=req.height,
            fps=req.fps,
            frames_per_act=req.frames_per_act,
            transition_frames=req.transition_frames,
            max_acts=req.max_acts,
            refine_acts=bool(req.refine_acts),
            save_latest_artifacts=True,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
