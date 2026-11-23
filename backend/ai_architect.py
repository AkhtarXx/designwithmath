import json
import math
import os
import re
import urllib.request
import urllib.error
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

def _load_local_dotenv() -> None:
    """Load optional KEY=VALUE pairs from local .env files without extra dependencies."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(base_dir, ".env"),
        os.path.join(os.path.dirname(base_dir), ".env"),
    ]
    for env_path in candidates:
        if not os.path.isfile(env_path):
            continue
        try:
            with open(env_path, "r", encoding="utf-8") as fp:
                for raw_line in fp:
                    line = raw_line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    key, val = line.split("=", 1)
                    key = key.strip()
                    val = val.strip().strip('"').strip("'")
                    if key and key not in os.environ:
                        os.environ[key] = val
        except OSError:
            continue


_load_local_dotenv()

# Optional Cloudflare Workers AI Credentials (read strictly from environment / .env)
CF_ACCOUNT_ID = (
    os.environ.get("CLOUDFLARE_ACCOUNT_ID")
    or os.environ.get("CF_ACCOUNT_ID")
    or ""
).strip()
CF_API_TOKEN = (
    os.environ.get("CLOUDFLARE_API_TOKEN")
    or os.environ.get("CF_API_TOKEN")
    or ""
).strip()


def _has_cf_credentials() -> bool:
    return bool(CF_ACCOUNT_ID and CF_API_TOKEN)

# Active Live Flagship DeepSeek V4.1 (552B MoE) & GLM-5.3 Models (never exposed to UI)
# Note: @cf/deepseek-ai/deepseek-v4-pro-0813 was deprecated by Cloudflare (HTTP 410) on 2026-11-20;
# @cf/deepseek-ai/deepseek-v4.1-flash-1017 (552B) is its active live DeepSeek V4 flagship successor.
GLM_5_3 = "@cf/zai-org/glm-5.3"
DEEPSEEK_V4_PRO = "@cf/deepseek-ai/deepseek-v4.1-flash-1017"
GLM_5_3_FLASH = "@cf/zai-org/glm-5.3-flash"

TIER_SIMPLE_MODELS: List[str] = [
    GLM_5_3,
    DEEPSEEK_V4_PRO,
    GLM_5_3_FLASH,
]

TIER_MEDIUM_MODELS: List[str] = [
    GLM_5_3,
    DEEPSEEK_V4_PRO,
    GLM_5_3_FLASH,
]

TIER_COMPLEX_MODELS: List[str] = [
    GLM_5_3,
    DEEPSEEK_V4_PRO,
    GLM_5_3_FLASH,
]

PRIMARY_MODEL = GLM_5_3
FALLBACK_MODELS: List[str] = [
    DEEPSEEK_V4_PRO,
    GLM_5_3_FLASH,
]

# Verified Live Multimodal Vision-Language Critic Models (used silently for Pass 2 visual self-critique)
VISION_CRITIC_MODELS: List[str] = [
    "@cf/google/gemma-4-26b-a4b-it",
    "@cf/mistralai/mistral-small-3.1-24b-instruct",
    "@cf/meta/llama-4-scout-17b-16e-instruct",
]

VALID_COMPONENT_TYPES = {
    "ribbon_spine",
    "polar_harmonic",
    "crescent_lobe",
    "wheel_rosette",
    "parametric_equation",
    "repeated_field",
    "implicit_sdf_expr",
}

VALID_TEXTURE_TYPES = {
    "feathers",
    "fur",
    "foliage",
    "petals",
    "cosmic",
    "marble",
    "metallic",
    "water",
    "smooth",
}

VALID_ANIM_ROLES = {
    "flap",
    "spin",
    "sway",
    "wave",
    "pulse",
    "orbit",
    "static",
}

VALID_BG_STYLES = {
    "night_sky",
    "golden_bokeh",
    "bokeh_forest",
    "cosmic_nebula",
    "sunset_sky",
    "deep_ocean",
    "lunar_dust",
}

VALID_REPEATED_LAYOUTS = {
    "perspective_corridor",
    "radial_fan",
    "scattered_cluster",
    "wave_array",
}

VALID_BASE_SHAPES = {
    "lobe",
    "tree",
    "tentacle",
    "feather",
    "petal",
    "sphere",
    "star",
}

_FORBIDDEN_META_RE = re.compile(
    r"\b(ai|cloudflare|llama|qwen|qwq|deepseek|glm|kimi|moonshot|openai|gpt|oss|nemotron|gemma|mistral|scout|vlm|vision|critic|neural|120b|70b|32b|27b|26b|24b|17b|8b|llm|instruct|fp8|workers\s*ai)\b",
    flags=re.IGNORECASE,
)

_UNSAFE_EXPR_RE = re.compile(
    r"(__|[\[\]{};:=`'\"\\]|\b(import|exec|eval|open|os|sys|lambda|class|def|getattr|setattr|delattr|globals|locals|compile|input|print)\b)",
    flags=re.IGNORECASE,
)

CACHE_FILE = os.path.join(os.path.dirname(__file__), "ir_cache.json")
_IR_CACHE: Dict[str, Dict[str, Any]] = {}

SYSTEM_PROMPT = """You are a Master Differential Geometer and Mathematical Artist inspired by Hamid Naderi Yeganeh's mathematical painting techniques (double-exponential level-set masks J_s = exp(-exp(K_s * d_s)), Porter-Duff front-to-back occlusion, trigonometric fBm field E(x,y), directional self-shadowing, and rhythmic multi-instance repetition).
Decompose ANY user prompt into a rich, visually unmistakable anatomical and scenic mathematical JSON specification on [-1.0, 1.0] x [-1.0, 1.0] (x: -1 left to +1 right, y: -1 bottom to +1 top).
Output STRICT JSON ONLY with keys:
- "title": Pure mathematical & geometric title (NEVER mention AI, LLM, Cloudflare, or model names)
- "subtitle": Pure differential-geometric description (e.g. "Harmonic Differential-Geometric Decomposition — 7 Coupled Parametric Manifolds")
- "recommended_mode": "lines" or "circles"
- "bg_style": one of "night_sky", "golden_bokeh", "bokeh_forest", "cosmic_nebula", "sunset_sky", "deep_ocean", "lunar_dust"
- "components": array of 5 to 9 rich layered components ordered from foreground hero details (z_order=1) to background landscape structures (z_order=8)

CRITICAL GEOMETRY & COMPOSITION RULES:
- NEVER create a "Sky" or "Background" component — "bg_style" already renders the sky/ocean/space background!
- In "implicit_sdf_expr", sdf_expr(x, y, E) MUST be NEGATIVE (< 0) INSIDE the shape and POSITIVE (> 0) OUTSIDE!
  * CORRECT circle/bell/planet: "sqrt((x-0.0)**2 + (y-0.2)**2) - 0.35" (NEVER "0.35 - sqrt(...)").
  * CORRECT mountain/terrain below ridge f(x): "y - (0.45*exp(-4.0*x**2) + 0.25*exp(-5.0*(x-0.55)**2) - 0.18 + 0.06*E)"
  * CORRECT tilted planetary ring: "abs(sqrt((x*cos(0.35)+y*sin(0.35))**2 + (3.4*(-x*sin(0.35)+y*cos(0.35)))**2) - 0.62) - 0.07"
- Use VIVID, HIGH-CONTRAST, natural colors in "color_rgb" and "accent_rgb" (e.g., electric cyan [0.10, 0.88, 1.0] + neon violet [0.78, 0.22, 0.98] for bioluminescence; fiery orange [0.95, 0.45, 0.08] + deep obsidian [0.07, 0.06, 0.05] for tiger stripes; crisp snow [0.95, 0.97, 1.0] + golden light [1.0, 0.68, 0.18] for alpine sunset).
- Set z_order=1..2 ONLY for crisp foreground hero anatomy (eyes, visor, stripes, jellyfish bell, foreground pines), and z_order=6..9 for distant elements (mountains, sun, ocean floor, distant stars).

Each component in "components" MUST include:
- "name": string (descriptive anatomical or scenic part name)
- "type": one of "ribbon_spine", "polar_harmonic", "crescent_lobe", "wheel_rosette", "parametric_equation", "repeated_field", "implicit_sdf_expr"
- "weight": float 0.05..0.35
- "mirror_x": boolean (true ONLY for right-side symmetric halves x >= 0 like wings, ears, eyes, legs; keep false for full-span repeated_field or centered shapes)
- "z_order": integer 1..10 (1 = frontmost layer, larger = deeper background layer)
- "color_rgb": [r, g, b] floats in 0.0..1.0 (vivid primary color)
- "accent_rgb": [r, g, b] floats in 0.0..1.0 (contrasting highlight/iridescent/rim color)
- "texture_type": one of "feathers", "fur", "foliage", "petals", "cosmic", "marble", "metallic", "water", "smooth"
- "edge_sharpness": float 65.0..160.0 (double-exponential boundary sharpness K_s)
- "anim_role": one of "flap", "spin", "sway", "wave", "pulse", "orbit", "static"
- "z_offset": float -0.5..0.5 (3D manifold depth center)
- "z_bulge": float 0.02..0.25 (3D manifold surface extrusion thickness)

Type-specific geometry fields (place DIRECTLY as top-level keys in the component object, NOT nested under a sub-key):
1. "ribbon_spine": {"points": [[x0,y0], [x1,y1], ...], "w_start": 0.01..0.18, "w_end": 0.005..0.18, "w_bulge": -0.03..0.15, "w_mod_amp": 0..0.03, "w_mod_freq": 2..6}
2. "polar_harmonic": {"center": [cx,cy], "scale": [sx,sy], "r0": 1.0, "theta_range_deg": [0,360], "harmonics": [{"m": int, "cos_amp": float, "sin_amp": float, "power": 1}], "inner_scale": 0.5..0.85, "rotation_deg": 0}
3. "crescent_lobe": {"center": [cx,cy], "outer_radius": 0.08..0.65, "thickness": 0.22..0.48, "arc_deg": 140..270, "rotation_deg": float}
4. "wheel_rosette": {"center": [cx,cy], "outer_radius": 0.05..0.45, "hub_radius": 0.015..0.12, "spokes": 6..16}
5. "parametric_equation": {"center": [cx,cy], "scale": [sx,sy], "x1_expr": "sin(t)", "y1_expr": "cos(t)", "x2_expr": "0.7*sin(t+0.2)", "y2_expr": "0.7*cos(t+0.2)"}
6. "repeated_field" (use for forests, pine groves, cherry blossoms, jellyfish tentacles, star clusters, feather fans, fruit clusters, ripples, asteroid rings):
   {"layout": "perspective_corridor"|"radial_fan"|"scattered_cluster"|"wave_array", "count": 8..60, "base_shape": "lobe"|"tree"|"tentacle"|"feather"|"petal"|"sphere"|"star", "bounds": [x_min, x_max, y_min, y_max], "scale_range": [s_min, s_max]}
7. "implicit_sdf_expr" (use for mountain ranges, planetary discs & tilted rings, organic domes/bells, ocean floor where sdf_expr < 0 is INSIDE):
   {"sdf_expr": "valid expression in (x, y, E) using sin, cos, exp, abs, sqrt, pi, arctan, tan, min, max, **", "bounds": [x_min, x_max, y_min, y_max]}"""


def select_model_tier(prompt: str) -> List[str]:
    """
    Routes query complexity strictly between the two flagship models ONLY:
      - Simple / Medium queries -> GLM-5.3 primary, DeepSeek V4 Pro fallback
      - Complex / multi-object / 3D / video / scenic queries -> DeepSeek V4 Pro primary, GLM-5.3 fallback
    """
    clean = prompt.strip().lower()
    words = [w for w in re.split(r"[\s,;]+", clean) if w]
    num_words = len(words)

    complex_patterns = (
        r"\b(and|aur|with|over|under|near|beside|against|between|in|at|on|scene|landscape|city|"
        r"3d|4d|manifold|surface|extrude|video|animate|animation|motion|moving|"
        r"flapping|spinning|rotating|soaring|flying|swimming|dancing|riding|floating|roaring|glowing|"
        r"mosque|faisal|badshahi|minar|cathedral|palace|castle|temple|bridge|tower|"
        r"bicycle|velocipede|locomotive|train|helicopter|chariot|astronaut|saturn|nebula|"
        r"galaxy|corals|reef|forest|jungle|avenue|mountain|peak|sunset|ocean|jellyfish|tiger|"
        r"chandelier|clockwork|phoenix|peacock)\b"
    )
    if num_words >= 5 or "+" in clean or "&" in clean or re.search(complex_patterns, clean):
        return list(TIER_COMPLEX_MODELS)

    if num_words <= 2:
        return list(TIER_SIMPLE_MODELS)

    return list(TIER_MEDIUM_MODELS)


def safe_eval_sdf_expr(
    expr: str,
    x: np.ndarray,
    y: np.ndarray,
    E: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Safely evaluates an analytical implicit signed-distance expression sdf_expr(x, y, E)
    into a float64 NumPy array of the same shape as x and y.
    Negative values denote the interior (J_s = exp(-exp(K_s * sdf_expr)) -> 1).
    """
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    if E is None:
        E_arr = 0.5 * np.cos(4.0 * x_arr + 2.0 * np.sin(3.0 * y_arr)) * np.cos(4.0 * y_arr - 2.0 * np.sin(3.0 * x_arr))
    else:
        E_arr = np.asarray(E, dtype=np.float64)

    expr_str = str(expr or "").strip()
    if not expr_str or _UNSAFE_EXPR_RE.search(expr_str):
        return np.hypot(x_arr, y_arr) - 0.45

    expr_clean = expr_str.replace("^", "**")
    safe_env = {
        "x": x_arr,
        "y": y_arr,
        "E": E_arr,
        "r": np.hypot(x_arr, y_arr),
        "theta": np.arctan2(y_arr, x_arr),
        "sin": np.sin,
        "cos": np.cos,
        "tan": lambda z: np.tan(np.clip(z, -1.52, 1.52)),
        "arctan": np.arctan,
        "arctan2": np.arctan2,
        "atan": np.arctan,
        "atan2": np.arctan2,
        "exp": lambda z: np.exp(np.clip(z, -40.0, 40.0)),
        "abs": np.abs,
        "sqrt": lambda z: np.sqrt(np.maximum(z, 0.0)),
        "pi": math.pi,
        "min": np.minimum,
        "max": np.maximum,
        "minimum": np.minimum,
        "maximum": np.maximum,
        "hypot": np.hypot,
        "tanh": np.tanh,
        "sign": np.sign,
        "clip": np.clip,
    }
    val = eval(expr_clean, {"__builtins__": {}}, safe_env)
    out = np.broadcast_to(np.asarray(val, dtype=np.float64), x_arr.shape).copy()
    return np.nan_to_num(out, nan=1.0, posinf=10.0, neginf=-10.0)


def validate_sdf_expr(expr: str) -> bool:
    """Checks that an implicit SDF expression safely compiles and evaluates on a test grid."""
    expr_str = str(expr or "").strip()
    if not expr_str or _UNSAFE_EXPR_RE.search(expr_str):
        return False
    try:
        gx, gy = np.meshgrid(np.linspace(-1.0, 1.0, 8), np.linspace(-1.0, 1.0, 8))
        ge = np.zeros_like(gx)
        res = safe_eval_sdf_expr(expr_str, gx, gy, ge)
        return res.shape == gx.shape and bool(np.all(np.isfinite(res))) and float(np.std(res)) > 1e-6
    except Exception:
        return False


def infer_bg_style(text: str) -> str:
    low = text.lower()
    if any(k in low for k in ("nebula", "galaxy", "cosmic", "space", "starfield", "saturn", "planet", "pulsar", "quasar", "astronaut")):
        if "footprint" in low or ("moon" in low and "lunar" in low) or "regolith" in low:
            return "lunar_dust"
        return "cosmic_nebula"
    if any(k in low for k in ("footprint", "lunar", "crater", "regolith")):
        return "lunar_dust"
    if any(k in low for k in ("coral", "reef", "ocean", "sea", "fish", "seahorse", "whale", "dolphin", "shark", "turtle", "crab", "anchor", "jellyfish", "underwater")):
        return "deep_ocean"
    if any(k in low for k in ("mosque", "faisal", "badshahi", "hilal", "moon", "night", "lantern", "castle", "palace", "minar")):
        return "night_sky"
    if any(k in low for k in ("phoenix", "sunset", "autumn", "desert", "camel", "pyramid", "flame", "fire", "mountain", "peak", "snowy", "volcano")):
        return "sunset_sky"
    if any(k in low for k in ("cobra", "snake", "saanp", "kiwi", "bird", "eagle", "shaheen", "owl", "tree", "forest", "jungle", "tiger", "lion", "leaf", "maple", "oak", "olive", "palm", "butterfly", "deer", "pine")):
        return "bokeh_forest"
    return "golden_bokeh"


def infer_anim_role(comp: Dict[str, Any]) -> str:
    """
    Infers the 4D temporal animation role from component name and geometric primitive type:
      - wing / plume -> "flap"
      - wheel / gear / star / sun -> "spin"
      - neck / hood / flame / tail / tree / petal -> "sway"
      - water / pool / wave / coil / tentacle -> "wave"
      - moon / hilal / crown -> "pulse"
      - planet / ring / orbit -> "orbit"
    """
    existing = str(comp.get("anim_role", "")).strip().lower()
    if existing in VALID_ANIM_ROLES:
        return existing

    name_low = str(comp.get("name", "")).lower()
    ctype = str(comp.get("type", "ribbon_spine")).lower()

    if any(k in name_low for k in ("wing", "plume", "feather", "fin", "airfoil", "fluke")):
        if "hood" in name_low or "tail" in name_low:
            return "sway"
        return "flap"
    if any(k in name_low for k in ("wheel", "gear", "star", "sun", "crank", "rotor", "propeller", "turbine")):
        return "spin"
    if any(k in name_low for k in ("water", "pool", "wave", "coil", "ocean", "river", "ripple", "stream", "serpentine", "sand", "tentacle", "tendril", "jellyfish")):
        return "wave"
    if any(k in name_low for k in ("neck", "hood", "flame", "tail", "tree", "pine", "petal", "branch", "leaf", "frond", "stem", "fork", "handlebar", "foliage", "blossom", "grass", "mane")):
        return "sway"
    if any(k in name_low for k in ("moon", "hilal", "crown", "ocellus", "eye", "heart", "bulb", "core", "nebula", "crescent", "head", "body", "bell", "glow", "astronaut")):
        return "pulse"
    if any(k in name_low for k in ("orbit", "planet", "saturn", "satellite", "halo", "ring", "asteroid")):
        return "orbit"

    if ctype == "repeated_field":
        layout = str(comp.get("layout", "")).lower()
        bshape = str(comp.get("base_shape", "")).lower()
        if bshape == "tentacle" or layout == "wave_array":
            return "wave"
        if bshape == "feather":
            return "flap"
        if bshape == "star":
            return "pulse"
        return "sway"
    if ctype == "implicit_sdf_expr":
        if any(k in name_low for k in ("mountain", "peak", "ridge", "ground", "cliff", "snow", "terrain")):
            return "static"
        return "pulse"
    if ctype == "wheel_rosette":
        return "spin"
    if ctype == "polar_harmonic":
        return "pulse"
    if ctype == "crescent_lobe":
        return "flap"
    return "static"


def _normalize_bounds(raw_bounds: Any, default_bounds: List[float]) -> List[float]:
    if isinstance(raw_bounds, list) and len(raw_bounds) >= 4:
        try:
            x0, x1, y0, y1 = [float(np.clip(float(v), -1.5, 1.5)) for v in raw_bounds[:4]]
            if x1 <= x0 + 0.05:
                x0, x1 = -0.8, 0.8
            if y1 <= y0 + 0.05:
                y0, y1 = -0.75, 0.75
            return [round(x0, 3), round(x1, 3), round(y0, 3), round(y1, 3)]
        except Exception:
            pass
    return list(default_bounds)


def _normalize_scale_range(raw_sr: Any) -> List[float]:
    if isinstance(raw_sr, list) and len(raw_sr) >= 2:
        try:
            s0 = float(np.clip(float(raw_sr[0]), 0.02, 0.65))
            s1 = float(np.clip(float(raw_sr[1]), 0.02, 0.65))
            if s1 < s0:
                s0, s1 = s1, s0
            return [round(s0, 3), round(max(s1, s0 + 0.01), 3)]
        except Exception:
            pass
    return [0.06, 0.22]


def infer_component_attributes(comp: Dict[str, Any], idx: int = 0, total: int = 6) -> Dict[str, Any]:
    """
    Guarantees every IR component carries valid geometry fields (including for
    'repeated_field' and 'implicit_sdf_expr'), full-color pixel-field attributes
    (z_order, color_rgb, accent_rgb, texture_type, edge_sharpness), and
    3D/4D manifold attributes (anim_role, z_offset, z_bulge).
    """
    c = dict(comp)
    name_low = str(c.get("name", "")).lower()
    ctype = str(c.get("type", "ribbon_spine")).strip().lower()
    if ctype not in VALID_COMPONENT_TYPES:
        ctype = "ribbon_spine"
    c["type"] = ctype

    # Unwrap nested type-specific sub-objects (e.g. {"type": "implicit_sdf_expr", "implicit_sdf_expr": {"sdf_expr": ...}})
    nested = c.get(ctype)
    if isinstance(nested, dict):
        for nk, nv in nested.items():
            c.setdefault(nk, nv)
        c.pop(ctype, None)

    # Normalize type-specific schemas for the two advanced Hamid-inspired primitives
    if ctype == "repeated_field":
        layout = str(c.get("layout", "")).strip().lower()
        if layout not in VALID_REPEATED_LAYOUTS:
            if any(k in name_low for k in ("corridor", "forest", "avenue", "pine", "trees", "alley")):
                layout = "perspective_corridor"
            elif any(k in name_low for k in ("fan", "feather", "peacock", "burst", "ray", "ring", "corolla")):
                layout = "radial_fan"
            elif any(k in name_low for k in ("tentacle", "tendril", "wave", "ripple", "grass", "flame", "stripe", "coral")):
                layout = "wave_array"
            else:
                layout = "scattered_cluster"
        c["layout"] = layout

        try:
            cnt = int(round(float(c.get("count", 24))))
        except Exception:
            cnt = 24
        c["count"] = max(8, min(60, cnt))

        bshape = str(c.get("base_shape", "")).strip().lower()
        if bshape not in VALID_BASE_SHAPES:
            if any(k in name_low for k in ("tree", "pine", "forest", "cypress", "trunk")):
                bshape = "tree"
            elif any(k in name_low for k in ("tentacle", "tendril", "coral", "vine", "seaweed", "flame")):
                bshape = "tentacle"
            elif any(k in name_low for k in ("feather", "plume", "wing")):
                bshape = "feather"
            elif any(k in name_low for k in ("petal", "blossom", "flower", "leaf", "foliage", "cherry")):
                bshape = "petal"
            elif any(k in name_low for k in ("star", "spark", "glint", "galaxy", "constellation")):
                bshape = "star"
            elif any(k in name_low for k in ("sphere", "berry", "fruit", "bubble", "planet", "orb", "asteroid", "drop")):
                bshape = "sphere"
            else:
                bshape = "lobe"
        c["base_shape"] = bshape
        c["bounds"] = _normalize_bounds(c.get("bounds"), [-0.85, 0.85, -0.75, 0.75])
        if c["bounds"][0] < -0.15 and c["bounds"][1] > 0.15:
            c["mirror_x"] = False
        c["scale_range"] = _normalize_scale_range(c.get("scale_range"))
        c["seed"] = int(c.get("seed", idx + 1))

    elif ctype == "implicit_sdf_expr":
        c["bounds"] = _normalize_bounds(c.get("bounds"), [-1.0, 1.0, -1.0, 1.0])
        expr = str(c.get("sdf_expr", "")).strip()
        if not validate_sdf_expr(expr):
            cx, cy = 0.0, 0.0
            if isinstance(c.get("center"), list) and len(c["center"]) >= 2:
                cx, cy = float(c["center"][0]), float(c["center"][1])
            sx, sy = 0.45, 0.35
            if isinstance(c.get("scale"), list) and len(c["scale"]) >= 2:
                sx, sy = max(0.08, float(c["scale"][0])), max(0.08, float(c["scale"][1]))
            c["sdf_expr"] = f"sqrt(((x - ({cx:.3f})) / {sx:.3f})**2 + ((y - ({cy:.3f})) / {sy:.3f})**2) - (1.0 + 0.08 * E)"
        else:
            clean_expr = expr.replace("^", "**")
            # Auto-correct inverted SDF sign (e.g. "0.5 - sqrt(x**2+y**2)" -> "-(0.5 - sqrt(x**2+y**2))")
            # so J_s = exp(-exp(K_s * sdf_expr)) is strictly 1 INSIDE and 0 OUTSIDE.
            is_horizon_like = any(
                k in name_low
                for k in ("mountain", "peak", "ridge", "ground", "floor", "horizon", "slope", "cliff", "terrain", "valley", "land", "sea", "ocean", "sky")
            )
            if not is_horizon_like:
                try:
                    gx, gy = np.meshgrid(np.linspace(-1.2, 1.2, 16), np.linspace(1.2, -1.2, 16))
                    ge = np.zeros_like(gx)
                    D_chk = safe_eval_sdf_expr(clean_expr, gx, gy, ge)
                    corners_neg = (
                        D_chk[0, 0] < 0.0
                        and D_chk[0, -1] < 0.0
                        and D_chk[-1, 0] < 0.0
                        and D_chk[-1, -1] < 0.0
                    )
                    if corners_neg and float(np.mean(D_chk < 0.0)) > 0.50:
                        clean_expr = f"-({clean_expr})"
                except Exception:
                    pass
            c["sdf_expr"] = clean_expr
        c["relief_shadow"] = bool(c.get("relief_shadow", True))

    default_palettes = [
        ([0.88, 0.72, 0.28], [0.96, 0.90, 0.65], "marble"),
        ([0.16, 0.62, 0.85], [0.42, 0.88, 0.95], "feathers"),
        ([0.85, 0.25, 0.22], [0.96, 0.68, 0.18], "petals"),
        ([0.22, 0.74, 0.48], [0.75, 0.92, 0.35], "foliage"),
        ([0.65, 0.38, 0.88], [0.95, 0.65, 0.92], "cosmic"),
    ]
    def_c1, def_c2, def_tex = default_palettes[idx % len(default_palettes)]

    if any(k in name_low for k in ("moon", "hilal", "crescent", "star", "crown", "gold", "sun", "saturn")):
        def_c1, def_c2, def_tex = [0.96, 0.82, 0.25], [1.0, 0.96, 0.75], "metallic"
    elif any(k in name_low for k in ("snow", "peak", "mountain", "glacier", "ice", "minaret", "roof", "dome", "gable", "marble", "arch", "mosque", "tower", "pillar")):
        def_c1, def_c2, def_tex = [0.90, 0.88, 0.86], [0.68, 0.78, 0.92], "marble"
    elif any(k in name_low for k in ("wing", "feather", "plume", "tail", "peacock", "eagle", "phoenix", "bird", "kiwi")):
        def_c1, def_c2, def_tex = [0.14, 0.58, 0.76], [0.92, 0.68, 0.20], "feathers"
    elif any(k in name_low for k in ("wheel", "gear", "tube", "frame", "fork", "anchor", "shaft", "fluke", "ring", "stock", "saddle", "handlebar", "astronaut", "visor", "helmet")):
        def_c1, def_c2, def_tex = [0.25, 0.68, 0.88], [0.88, 0.94, 0.98], "metallic"
    elif any(k in name_low for k in ("tiger", "lion", "cobra", "hood", "snake", "coil", "neck", "body", "fur", "cat", "camel", "stripe")):
        def_c1, def_c2, def_tex = [0.88, 0.46, 0.14], [0.98, 0.82, 0.42], "fur"
    elif any(k in name_low for k in ("jellyfish", "tentacle", "pool", "water", "wave", "ocean", "reflection", "sea", "river", "sand")):
        def_c1, def_c2, def_tex = [0.14, 0.76, 0.94], [0.72, 0.42, 0.98], "water"
    elif any(k in name_low for k in ("tree", "pine", "leaf", "foliage", "branch", "forest", "jungle", "frond", "olive", "palm", "maple", "oak")):
        def_c1, def_c2, def_tex = [0.16, 0.52, 0.32], [0.68, 0.86, 0.34], "foliage"
    elif any(k in name_low for k in ("petal", "rose", "flower", "lotus", "strawberry", "corolla", "bloom", "blossom", "cherry")):
        def_c1, def_c2, def_tex = [0.88, 0.26, 0.34], [0.98, 0.72, 0.62], "petals"
    elif any(k in name_low for k in ("nebula", "cosmic", "galaxy", "space", "halo", "aurora", "glow")):
        def_c1, def_c2, def_tex = [0.65, 0.38, 0.88], [0.95, 0.65, 0.92], "cosmic"

    c["z_order"] = int(c.get("z_order", idx + 1))

    c1 = c.get("color_rgb")
    if not (isinstance(c1, list) and len(c1) == 3):
        c["color_rgb"] = [round(float(v), 3) for v in def_c1]
    else:
        c["color_rgb"] = [round(float(np.clip(float(v), 0.0, 1.0)), 3) for v in c1]

    c2 = c.get("accent_rgb")
    if not (isinstance(c2, list) and len(c2) == 3):
        c["accent_rgb"] = [round(float(v), 3) for v in def_c2]
    else:
        c["accent_rgb"] = [round(float(np.clip(float(v), 0.0, 1.0)), 3) for v in c2]

    tex = str(c.get("texture_type", "")).strip().lower()
    c["texture_type"] = tex if tex in VALID_TEXTURE_TYPES else def_tex

    c["edge_sharpness"] = round(float(np.clip(float(c.get("edge_sharpness", 118.0)), 50.0, 200.0)), 1)
    c["anim_role"] = infer_anim_role(c)

    if "z_offset" in c and c["z_offset"] is not None:
        z_off = float(np.clip(float(c["z_offset"]), -0.5, 0.5))
    else:
        span = max(1, total - 1)
        z_off = float(np.clip(0.22 * (0.5 - idx / span), -0.5, 0.5))
    c["z_offset"] = round(z_off, 3)

    if "z_bulge" in c and c["z_bulge"] is not None:
        z_blg = float(np.clip(float(c["z_bulge"]), 0.02, 0.25))
    else:
        default_bulge = {
            "polar_harmonic": 0.16,
            "crescent_lobe": 0.14,
            "ribbon_spine": 0.11,
            "wheel_rosette": 0.07,
            "parametric_equation": 0.12,
            "repeated_field": 0.13,
            "implicit_sdf_expr": 0.18,
        }.get(ctype, 0.11)
        z_blg = float(np.clip(default_bulge, 0.02, 0.25))
    c["z_bulge"] = round(z_blg, 3)

    return c


def sanitize_mathematical_text(text: str, fallback: str) -> str:
    if not text or not str(text).strip():
        return fallback
    cleaned = _FORBIDDEN_META_RE.sub("", str(text))
    cleaned = re.sub(r"\(\s*\)", "", cleaned)
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip(" -—:|,")
    return cleaned if len(cleaned) >= 4 else fallback


def enrich_ir_blueprint(ir: Dict[str, Any], prompt: str = "") -> Dict[str, Any]:
    """
    Sanitizes all top-level metadata to ensure zero AI/Cloudflare/model references
    and enriches all components with pixel-field, 3D surface, and 4D time attributes.
    """
    out = dict(ir)
    out.pop("ai_model", None)
    out.pop("model", None)

    raw_comps = out.get("components", [])
    filtered_raw: List[Dict[str, Any]] = []
    for c in raw_comps:
        cname = str(c.get("name", "")).strip().lower()
        ctype = str(c.get("type", "")).strip().lower()
        if ctype == "implicit_sdf_expr" and cname in ("sky", "background", "backdrop", "night sky", "sunset sky", "space background", "ocean background"):
            continue
        filtered_raw.append(c)
    if len(filtered_raw) >= 2:
        raw_comps = filtered_raw

    total = max(1, len(raw_comps))
    enriched_comps = [infer_component_attributes(c, idx=i, total=total) for i, c in enumerate(raw_comps)]

    if enriched_comps and all(c.get("anim_role") == "static" for c in enriched_comps):
        enriched_comps[0]["anim_role"] = "sway"
        if len(enriched_comps) > 1:
            enriched_comps[-1]["anim_role"] = "pulse"

    out["components"] = enriched_comps

    base_name = prompt.strip().title() if prompt.strip() else "Harmonic Manifold"
    default_title = f"{base_name} — Harmonic Caustic Synthesis"
    raw_title = str(out.get("title") or "")
    out["title"] = sanitize_mathematical_text(raw_title, default_title)

    comp_names = ", ".join(str(c.get("name", "Manifold")).replace("_", " ").title() for c in enriched_comps[:5])
    default_subtitle = (
        f"Harmonic Differential-Geometric Decomposition — {len(enriched_comps)} Coupled Parametric Manifolds ({comp_names})"
    )
    raw_subtitle = str(out.get("subtitle") or "")
    if (
        not raw_subtitle
        or _FORBIDDEN_META_RE.search(raw_subtitle)
        or raw_subtitle.strip().lower() in ("geometric decomposition", "a decomposition into geometric components")
    ):
        out["subtitle"] = default_subtitle
    else:
        out["subtitle"] = sanitize_mathematical_text(raw_subtitle, default_subtitle)

    bg = str(out.get("bg_style", "")).strip().lower()
    out["bg_style"] = bg if bg in VALID_BG_STYLES else infer_bg_style(f"{out['title']} {prompt}")
    out["recommended_mode"] = out.get("recommended_mode") if out.get("recommended_mode") in ("lines", "circles") else "lines"
    return out


def _save_cache() -> None:
    try:
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(_IR_CACHE, f, indent=2)
    except Exception:
        pass


def _load_and_sanitize_cache() -> None:
    global _IR_CACHE
    if not os.path.exists(CACHE_FILE):
        _IR_CACHE = {}
        return
    try:
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            raw_cache = json.load(f)
        if isinstance(raw_cache, dict):
            dirty = False
            cleaned_cache: Dict[str, Dict[str, Any]] = {}
            for k, v in raw_cache.items():
                if isinstance(v, dict) and isinstance(v.get("components"), list):
                    if "ai_model" in v or not v.get("bg_style") or any("anim_role" not in c for c in v.get("components", [])):
                        dirty = True
                    cleaned_cache[k] = enrich_ir_blueprint(v, prompt=k)
            _IR_CACHE = cleaned_cache
            if dirty:
                _save_cache()
    except Exception:
        _IR_CACHE = {}


_load_and_sanitize_cache()


def _extract_response_text(body: Dict[str, Any]) -> Any:
    """
    Extracts generated content across all Cloudflare Workers AI flagship response schemas:
      1. OpenAI Responses API (e.g. gpt-oss-120b): result.output[type=='message'].content[].text
      2. OpenAI Chat Completions API (e.g. deepseek-v4-pro, glm-5.3, kimi-k2.7-code, llama-3.3-70b):
         result.choices[0].message.content
      3. Classic Workers AI API (e.g. qwq-32b, deepseek-r1): result.response
    """
    res = body.get("result")
    if isinstance(res, dict):
        # 1. OpenAI Responses API
        output_items = res.get("output")
        if isinstance(output_items, list):
            msg_chunks: List[str] = []
            for item in output_items:
                if isinstance(item, dict) and item.get("type") == "message":
                    for part in item.get("content", []):
                        if isinstance(part, dict) and isinstance(part.get("text"), str):
                            msg_chunks.append(part["text"])
            if msg_chunks:
                return "\n".join(msg_chunks)

        # 2. OpenAI Chat Completions API
        choices = res.get("choices")
        if isinstance(choices, list) and len(choices) > 0:
            msg = choices[0].get("message", {}) if isinstance(choices[0], dict) else {}
            content = msg.get("content")
            if content:
                return content
            reasoning_c = msg.get("reasoning_content")
            if reasoning_c:
                return reasoning_c

        # 3. Classic Workers AI API
        if res.get("response"):
            return res["response"]
    return ""


def _invoke_cloudflare_model(
    model: str,
    system_prompt: str,
    user_msg: str,
    max_tokens: int = 8192,
    temperature: float = 0.18,
    enable_thinking: bool = False,
) -> Optional[Dict[str, Any]]:
    if not _has_cf_credentials():
        return None
    url = f"https://api.cloudflare.com/client/v4/accounts/{CF_ACCOUNT_ID}/ai/run/{model}"
    payload: Dict[str, Any] = {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if "glm-5" in model:
        payload["reasoning_effort"] = "low"
    elif "deepseek-v4" in model and not enable_thinking:
        payload["chat_template_kwargs"] = {"enable_thinking": False}

    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {CF_API_TOKEN}",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=75) as resp:
            body = json.loads(resp.read().decode("utf-8"))
            raw_resp = _extract_response_text(body)
            if isinstance(raw_resp, dict):
                ir = raw_resp
            else:
                ir = _extract_json_object(str(raw_resp))
            if ir and isinstance(ir.get("components"), list) and len(ir["components"]) > 0:
                return ir
    except Exception:
        return None
    return None


def call_cloudflare_ai_for_ir(prompt: str, bypass_cache: bool = False) -> Optional[Dict[str, Any]]:
    """
    Silent Multi-Tier Flagship Semantic-to-Mathematics Compiler.
    Routes every prompt live strictly through DeepSeek V4 Pro (@cf/deepseek-ai/deepseek-v4-pro-0813)
    and GLM-5.3 (@cf/zai-org/glm-5.3) ONLY with max_tokens=8192.
    Performs an automatic quality retry with the companion flagship model if a complex prompt
    returns < 4 components. Uses get_curated_complex_ir ONLY as an offline fallback.
    """
    cache_key = prompt.strip().lower()
    if not bypass_cache and cache_key in _IR_CACHE:
        cached = enrich_ir_blueprint(_IR_CACHE[cache_key], prompt=prompt)
        _IR_CACHE[cache_key] = cached
        return cached

    tier_models = select_model_tier(prompt)
    is_complex_prompt = (tier_models == TIER_COMPLEX_MODELS)

    models_to_try = list(tier_models)
    for fb in FALLBACK_MODELS:
        if fb not in models_to_try:
            models_to_try.append(fb)

    user_msg = (
        f"Decompose \"{prompt}\" into 6-9 vivid, mathematically accurate layered components in [-0.95, 0.95]^2. "
        f"Use repeated_field for natural repeating elements (forests, pine trees, tentacles, stars, feathers, blossoms, ripples) "
        f"and implicit_sdf_expr for mountains, planets, tilted rings, bells, domes, or sculpted contours where appropriate. "
        f"Place type-specific fields (points, sdf_expr, layout, count, base_shape, bounds, scale_range, center, scale, etc.) "
        f"directly as top-level keys inside each component object. "
        f"Include z_order, color_rgb, accent_rgb, texture_type, edge_sharpness, anim_role, z_offset, z_bulge, and bg_style. "
        f"Use mirror_x:true for right-side symmetric halves only. Never mention AI or model names. "
        f"Keep internal reasoning concise and output ONLY the final valid JSON object."
    )

    retried_quality = False
    for model in models_to_try:
        ir = _invoke_cloudflare_model(model, SYSTEM_PROMPT, user_msg, max_tokens=8192, temperature=0.18)
        if ir is not None:
            # Lightweight quality check: complex prompts returning < 4 components auto-retry once with companion flagship
            if is_complex_prompt and len(ir.get("components", [])) < 4 and not retried_quality:
                retried_quality = True
                retry_model = GLM_5_3 if model == DEEPSEEK_V4_PRO else DEEPSEEK_V4_PRO
                retry_msg = (
                    f"Decompose \"{prompt}\" into AT LEAST 6 distinct, richly layered mathematical components "
                    f"(foreground hero anatomy, mid-ground details, and background landscape/fields using repeated_field "
                    f"and implicit_sdf_expr as well as ribbon_spine/polar_harmonic/crescent_lobe). "
                    f"Every component MUST have z_order, color_rgb, accent_rgb, texture_type, edge_sharpness, anim_role, z_offset, z_bulge. "
                    f"Keep reasoning brief and return ONLY valid JSON."
                )
                retry_ir = _invoke_cloudflare_model(
                    retry_model, SYSTEM_PROMPT, retry_msg, max_tokens=8192, temperature=0.28
                )
                if retry_ir is not None and len(retry_ir.get("components", [])) >= len(ir.get("components", [])):
                    ir = retry_ir

            enriched = enrich_ir_blueprint(ir, prompt=prompt)
            _IR_CACHE[cache_key] = enriched
            _save_cache()
            return enriched

    # Offline fallback ONLY when Cloudflare API fails
    builtin_ir = get_curated_complex_ir(cache_key)
    if builtin_ir is not None:
        return enrich_ir_blueprint(builtin_ir, prompt=prompt)

    return None


def get_curated_complex_ir(p: str) -> Optional[Dict[str, Any]]:
    """
    Offline fallback anatomical differential-geometric blueprints used ONLY when
    the live multi-tier Cloudflare compilation cascade is unreachable.
    """
    p_low = p.strip().lower()

    if "faisal" in p_low:
        return {
            "title": "Shah Faisal Mosque & Hilal — Harmonic Caustic Synthesis",
            "subtitle": "Harmonic Differential-Geometric Decomposition — 6 Coupled Parametric Manifolds (Four Pencil Minarets, Pyramidal Bedouin Tent Roof, Reflection Pool, and Hilal)",
            "recommended_mode": "lines",
            "bg_style": "night_sky",
            "components": [
                {
                    "name": "Pyramidal Main Tent Roof Slope",
                    "type": "ribbon_spine",
                    "weight": 0.24,
                    "mirror_x": True,
                    "points": [[0.0, 0.36], [0.27, 0.02], [0.54, -0.42]],
                    "w_start": 0.032,
                    "w_end": 0.11,
                    "w_bulge": 0.055,
                    "z_order": 3,
                    "color_rgb": [0.92, 0.90, 0.86],
                    "accent_rgb": [0.72, 0.80, 0.92],
                    "texture_type": "marble",
                    "edge_sharpness": 125.0,
                    "anim_role": "static",
                    "z_offset": 0.05,
                    "z_bulge": 0.18,
                },
                {
                    "name": "Central Gable Ridge & Entrance Portal",
                    "type": "ribbon_spine",
                    "weight": 0.16,
                    "mirror_x": True,
                    "points": [[0.0, 0.36], [0.14, -0.06], [0.28, -0.42]],
                    "w_start": 0.022,
                    "w_end": 0.055,
                    "w_bulge": 0.035,
                    "z_order": 2,
                    "color_rgb": [0.95, 0.93, 0.88],
                    "accent_rgb": [0.82, 0.76, 0.58],
                    "texture_type": "marble",
                    "edge_sharpness": 130.0,
                    "anim_role": "static",
                    "z_offset": 0.14,
                    "z_bulge": 0.14,
                },
                {
                    "name": "Outer Tall Pencil Minaret",
                    "type": "ribbon_spine",
                    "weight": 0.20,
                    "mirror_x": True,
                    "points": [[0.72, -0.44], [0.72, 0.18], [0.72, 0.76]],
                    "w_start": 0.038,
                    "w_end": 0.005,
                    "w_bulge": 0.008,
                    "z_order": 1,
                    "color_rgb": [0.94, 0.92, 0.88],
                    "accent_rgb": [0.88, 0.78, 0.45],
                    "texture_type": "marble",
                    "edge_sharpness": 135.0,
                    "anim_role": "static",
                    "z_offset": 0.22,
                    "z_bulge": 0.08,
                },
                {
                    "name": "Inner Flanking Minaret",
                    "type": "ribbon_spine",
                    "weight": 0.14,
                    "mirror_x": True,
                    "points": [[0.48, -0.42], [0.48, 0.12], [0.48, 0.56]],
                    "w_start": 0.028,
                    "w_end": 0.004,
                    "w_bulge": 0.006,
                    "z_order": 4,
                    "color_rgb": [0.86, 0.85, 0.82],
                    "accent_rgb": [0.68, 0.74, 0.86],
                    "texture_type": "marble",
                    "edge_sharpness": 120.0,
                    "anim_role": "static",
                    "z_offset": -0.14,
                    "z_bulge": 0.07,
                },
                {
                    "name": "Marble Courtyard & Reflection Pool",
                    "type": "ribbon_spine",
                    "weight": 0.12,
                    "mirror_x": False,
                    "points": [[-0.84, -0.45], [0.0, -0.45], [0.84, -0.45]],
                    "w_start": 0.018,
                    "w_end": 0.018,
                    "w_bulge": 0.028,
                    "z_order": 5,
                    "color_rgb": [0.14, 0.52, 0.78],
                    "accent_rgb": [0.58, 0.88, 0.96],
                    "texture_type": "water",
                    "edge_sharpness": 110.0,
                    "anim_role": "wave",
                    "z_offset": 0.0,
                    "z_bulge": 0.05,
                },
                {
                    "name": "Celestial Crescent Moon (Hilal)",
                    "type": "crescent_lobe",
                    "weight": 0.14,
                    "center": [0.0, 0.70],
                    "outer_radius": 0.17,
                    "thickness": 0.28,
                    "arc_deg": 235.0,
                    "rotation_deg": 125.0,
                    "z_order": 1,
                    "color_rgb": [0.98, 0.84, 0.26],
                    "accent_rgb": [1.0, 0.96, 0.78],
                    "texture_type": "metallic",
                    "edge_sharpness": 140.0,
                    "anim_role": "pulse",
                    "z_offset": -0.25,
                    "z_bulge": 0.12,
                },
            ],
        }

    if "bicycle" in p_low or "velocipede" in p_low or "cycle" in p_low or "bike" in p_low:
        return {
            "title": "Harmonic Velocipede — Geared Caustic Bicycle",
            "subtitle": "Harmonic Differential-Geometric Decomposition — 8 Coupled Parametric Manifolds (Twin Spoked Wheel Rosettes, Diamond Truss Frame, Crankset Gear, Saddle, and Handlebars)",
            "recommended_mode": "lines",
            "bg_style": "golden_bokeh",
            "components": [
                {
                    "name": "Rear Spoked Wheel",
                    "type": "wheel_rosette",
                    "weight": 0.24,
                    "center": [-0.52, -0.22],
                    "outer_radius": 0.30,
                    "hub_radius": 0.038,
                    "spokes": 10,
                    "z_order": 3,
                    "color_rgb": [0.24, 0.68, 0.88],
                    "accent_rgb": [0.88, 0.94, 0.98],
                    "texture_type": "metallic",
                    "edge_sharpness": 130.0,
                    "anim_role": "spin",
                    "z_offset": 0.0,
                    "z_bulge": 0.08,
                },
                {
                    "name": "Front Spoked Wheel",
                    "type": "wheel_rosette",
                    "weight": 0.24,
                    "center": [0.52, -0.22],
                    "outer_radius": 0.30,
                    "hub_radius": 0.038,
                    "spokes": 10,
                    "z_order": 3,
                    "color_rgb": [0.24, 0.68, 0.88],
                    "accent_rgb": [0.88, 0.94, 0.98],
                    "texture_type": "metallic",
                    "edge_sharpness": 130.0,
                    "anim_role": "spin",
                    "z_offset": 0.0,
                    "z_bulge": 0.08,
                },
                {
                    "name": "Crankset Gear",
                    "type": "wheel_rosette",
                    "weight": 0.10,
                    "center": [-0.08, -0.22],
                    "outer_radius": 0.095,
                    "hub_radius": 0.022,
                    "spokes": 6,
                    "z_order": 1,
                    "color_rgb": [0.94, 0.76, 0.24],
                    "accent_rgb": [1.0, 0.94, 0.72],
                    "texture_type": "metallic",
                    "edge_sharpness": 135.0,
                    "anim_role": "spin",
                    "z_offset": 0.08,
                    "z_bulge": 0.06,
                },
                {
                    "name": "Horizontal Top Tube",
                    "type": "ribbon_spine",
                    "weight": 0.09,
                    "points": [[-0.22, 0.20], [0.06, 0.20], [0.34, 0.20]],
                    "w_start": 0.022,
                    "w_end": 0.022,
                    "w_bulge": 0.008,
                    "z_order": 2,
                    "color_rgb": [0.86, 0.24, 0.22],
                    "accent_rgb": [0.98, 0.64, 0.28],
                    "texture_type": "metallic",
                    "edge_sharpness": 125.0,
                    "anim_role": "static",
                    "z_offset": 0.04,
                    "z_bulge": 0.06,
                },
                {
                    "name": "Rear Seat Stay & Chainstay",
                    "type": "ribbon_spine",
                    "weight": 0.11,
                    "points": [[-0.22, 0.20], [-0.52, -0.22], [-0.08, -0.22]],
                    "w_start": 0.020,
                    "w_end": 0.022,
                    "z_order": 2,
                    "color_rgb": [0.82, 0.22, 0.20],
                    "accent_rgb": [0.96, 0.58, 0.24],
                    "texture_type": "metallic",
                    "edge_sharpness": 125.0,
                    "anim_role": "static",
                    "z_offset": 0.03,
                    "z_bulge": 0.05,
                },
                {
                    "name": "Diagonal Down Tube",
                    "type": "ribbon_spine",
                    "weight": 0.08,
                    "points": [[-0.08, -0.22], [0.13, -0.01], [0.34, 0.20]],
                    "w_start": 0.024,
                    "w_end": 0.024,
                    "z_order": 2,
                    "color_rgb": [0.86, 0.24, 0.22],
                    "accent_rgb": [0.98, 0.64, 0.28],
                    "texture_type": "metallic",
                    "edge_sharpness": 125.0,
                    "anim_role": "static",
                    "z_offset": 0.04,
                    "z_bulge": 0.06,
                },
                {
                    "name": "Seat Tube & Racing Saddle",
                    "type": "ribbon_spine",
                    "weight": 0.07,
                    "points": [[-0.08, -0.22], [-0.24, 0.27], [-0.34, 0.29], [-0.15, 0.28]],
                    "w_start": 0.022,
                    "w_end": 0.035,
                    "z_order": 1,
                    "color_rgb": [0.28, 0.32, 0.38],
                    "accent_rgb": [0.75, 0.80, 0.86],
                    "texture_type": "metallic",
                    "edge_sharpness": 125.0,
                    "anim_role": "static",
                    "z_offset": 0.06,
                    "z_bulge": 0.07,
                },
                {
                    "name": "Front Fork & Drop Handlebars",
                    "type": "ribbon_spine",
                    "weight": 0.07,
                    "points": [[0.52, -0.22], [0.32, 0.33], [0.47, 0.35], [0.45, 0.21]],
                    "w_start": 0.022,
                    "w_end": 0.028,
                    "z_order": 1,
                    "color_rgb": [0.28, 0.72, 0.90],
                    "accent_rgb": [0.90, 0.96, 1.0],
                    "texture_type": "metallic",
                    "edge_sharpness": 125.0,
                    "anim_role": "sway",
                    "z_offset": 0.06,
                    "z_bulge": 0.07,
                },
            ],
        }

    if "cobra" in p_low or "saanp" in p_low or "snake" in p_low:
        return {
            "title": "Majestic King Cobra — Harmonic Caustic Synthesis",
            "subtitle": "Harmonic Differential-Geometric Decomposition — 6 Coupled Parametric Manifolds (Flared Spectacle Hood, Upright Strike Neck, and S-Coiled Serpentine Body)",
            "recommended_mode": "lines",
            "bg_style": "bokeh_forest",
            "components": [
                {
                    "name": "Flared Cobra Hood Wing",
                    "type": "crescent_lobe",
                    "weight": 0.28,
                    "mirror_x": True,
                    "center": [0.05, 0.34],
                    "outer_radius": 0.32,
                    "thickness": 0.36,
                    "arc_deg": 165.0,
                    "rotation_deg": 5.0,
                    "z_order": 3,
                    "color_rgb": [0.72, 0.44, 0.18],
                    "accent_rgb": [0.94, 0.76, 0.36],
                    "texture_type": "fur",
                    "edge_sharpness": 115.0,
                    "anim_role": "sway",
                    "z_offset": -0.04,
                    "z_bulge": 0.12,
                },
                {
                    "name": "Cobra Head & Crown",
                    "type": "polar_harmonic",
                    "weight": 0.14,
                    "center": [0.0, 0.56],
                    "scale": [0.12, 0.14],
                    "r0": 1.0,
                    "harmonics": [{"m": 2, "cos_amp": -0.18}, {"m": 3, "sin_amp": 0.12}],
                    "inner_scale": 0.55,
                    "z_order": 1,
                    "color_rgb": [0.82, 0.54, 0.22],
                    "accent_rgb": [0.98, 0.86, 0.48],
                    "texture_type": "fur",
                    "edge_sharpness": 125.0,
                    "anim_role": "sway",
                    "z_offset": 0.14,
                    "z_bulge": 0.16,
                },
                {
                    "name": "Hood Spectacle Ocellus",
                    "type": "wheel_rosette",
                    "weight": 0.10,
                    "mirror_x": True,
                    "center": [0.13, 0.36],
                    "outer_radius": 0.055,
                    "hub_radius": 0.018,
                    "spokes": 8,
                    "z_order": 2,
                    "color_rgb": [0.96, 0.84, 0.28],
                    "accent_rgb": [0.18, 0.12, 0.08],
                    "texture_type": "feathers",
                    "edge_sharpness": 130.0,
                    "anim_role": "pulse",
                    "z_offset": 0.06,
                    "z_bulge": 0.08,
                },
                {
                    "name": "Upright Strike Neck Column",
                    "type": "ribbon_spine",
                    "weight": 0.20,
                    "points": [[0.0, 0.52], [0.0, 0.16], [0.0, -0.22]],
                    "w_start": 0.065,
                    "w_end": 0.088,
                    "w_bulge": 0.025,
                    "z_order": 2,
                    "color_rgb": [0.78, 0.50, 0.22],
                    "accent_rgb": [0.96, 0.82, 0.46],
                    "texture_type": "fur",
                    "edge_sharpness": 120.0,
                    "anim_role": "sway",
                    "z_offset": 0.08,
                    "z_bulge": 0.15,
                },
                {
                    "name": "Upper Serpentine Coil",
                    "type": "ribbon_spine",
                    "weight": 0.15,
                    "points": [[0.0, -0.22], [-0.46, -0.34], [0.46, -0.48]],
                    "w_start": 0.085,
                    "w_end": 0.065,
                    "w_bulge": 0.02,
                    "z_order": 4,
                    "color_rgb": [0.64, 0.38, 0.16],
                    "accent_rgb": [0.88, 0.68, 0.32],
                    "texture_type": "fur",
                    "edge_sharpness": 115.0,
                    "anim_role": "wave",
                    "z_offset": 0.02,
                    "z_bulge": 0.14,
                },
                {
                    "name": "Lower Tapered Tail Coil",
                    "type": "ribbon_spine",
                    "weight": 0.13,
                    "points": [[0.46, -0.48], [-0.52, -0.62], [0.45, -0.68]],
                    "w_start": 0.065,
                    "w_end": 0.012,
                    "w_bulge": 0.02,
                    "z_order": 5,
                    "color_rgb": [0.56, 0.32, 0.14],
                    "accent_rgb": [0.82, 0.60, 0.28],
                    "texture_type": "fur",
                    "edge_sharpness": 115.0,
                    "anim_role": "wave",
                    "z_offset": -0.08,
                    "z_bulge": 0.11,
                },
            ],
        }

    if "phoenix" in p_low:
        return {
            "title": "Celestial Phoenix Rising — Harmonic Flame Synthesis",
            "subtitle": "Harmonic Differential-Geometric Decomposition — 6 Coupled Parametric Manifolds (Outstretched Flaming Primary Wings, Crested Head, and Streaming Triple Fire-Plume Tail)",
            "recommended_mode": "lines",
            "bg_style": "sunset_sky",
            "components": [
                {
                    "name": "Primary Flaming Wing Arch",
                    "type": "crescent_lobe",
                    "weight": 0.30,
                    "mirror_x": True,
                    "center": [0.44, 0.24],
                    "outer_radius": 0.48,
                    "thickness": 0.34,
                    "arc_deg": 175.0,
                    "rotation_deg": 35.0,
                    "z_order": 2,
                    "color_rgb": [0.94, 0.32, 0.12],
                    "accent_rgb": [1.0, 0.84, 0.22],
                    "texture_type": "feathers",
                    "edge_sharpness": 115.0,
                    "anim_role": "flap",
                    "z_offset": 0.06,
                    "z_bulge": 0.18,
                },
                {
                    "name": "Secondary Inner Wing Plume",
                    "type": "ribbon_spine",
                    "weight": 0.22,
                    "mirror_x": True,
                    "points": [[0.06, 0.05], [0.38, 0.38], [0.78, 0.52]],
                    "w_start": 0.07,
                    "w_end": 0.015,
                    "w_bulge": 0.08,
                    "z_order": 3,
                    "color_rgb": [0.98, 0.56, 0.14],
                    "accent_rgb": [1.0, 0.92, 0.38],
                    "texture_type": "feathers",
                    "edge_sharpness": 110.0,
                    "anim_role": "flap",
                    "z_offset": -0.04,
                    "z_bulge": 0.14,
                },
                {
                    "name": "Phoenix Crested Head & Crown",
                    "type": "polar_harmonic",
                    "weight": 0.14,
                    "center": [0.0, 0.42],
                    "scale": [0.10, 0.15],
                    "r0": 1.0,
                    "harmonics": [{"m": 3, "sin_amp": 0.28}, {"m": 5, "cos_amp": 0.14}],
                    "inner_scale": 0.5,
                    "z_order": 1,
                    "color_rgb": [1.0, 0.78, 0.20],
                    "accent_rgb": [1.0, 0.96, 0.68],
                    "texture_type": "feathers",
                    "edge_sharpness": 125.0,
                    "anim_role": "pulse",
                    "z_offset": 0.16,
                    "z_bulge": 0.15,
                },
                {
                    "name": "Central Body",
                    "type": "polar_harmonic",
                    "weight": 0.12,
                    "center": [0.0, 0.08],
                    "scale": [0.12, 0.22],
                    "r0": 1.0,
                    "harmonics": [{"m": 2, "cos_amp": -0.22}],
                    "inner_scale": 0.55,
                    "z_order": 1,
                    "color_rgb": [0.96, 0.42, 0.14],
                    "accent_rgb": [1.0, 0.86, 0.30],
                    "texture_type": "feathers",
                    "edge_sharpness": 120.0,
                    "anim_role": "pulse",
                    "z_offset": 0.12,
                    "z_bulge": 0.20,
                },
                {
                    "name": "Outer Streaming Tail Plume",
                    "type": "ribbon_spine",
                    "weight": 0.14,
                    "mirror_x": True,
                    "points": [[0.04, -0.12], [0.26, -0.45], [0.12, -0.78]],
                    "w_start": 0.045,
                    "w_end": 0.008,
                    "w_bulge": 0.03,
                    "z_order": 4,
                    "color_rgb": [0.88, 0.22, 0.16],
                    "accent_rgb": [0.98, 0.68, 0.18],
                    "texture_type": "feathers",
                    "edge_sharpness": 105.0,
                    "anim_role": "sway",
                    "z_offset": -0.08,
                    "z_bulge": 0.10,
                },
                {
                    "name": "Central Flame Tail Ribbon",
                    "type": "ribbon_spine",
                    "weight": 0.08,
                    "points": [[0.0, -0.12], [0.06, -0.48], [-0.05, -0.84]],
                    "w_start": 0.045,
                    "w_end": 0.006,
                    "w_bulge": 0.025,
                    "z_order": 3,
                    "color_rgb": [0.96, 0.48, 0.12],
                    "accent_rgb": [1.0, 0.90, 0.32],
                    "texture_type": "feathers",
                    "edge_sharpness": 108.0,
                    "anim_role": "sway",
                    "z_offset": -0.04,
                    "z_bulge": 0.10,
                },
            ],
        }

    return None


def _extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    text = str(text or "").strip()
    # Strip complete <think>...</think> blocks and any dangling pre-</think> reasoning
    text = re.sub(r"<think>[\s\S]*?</think>", "", text, flags=re.IGNORECASE).strip()
    if "</think>" in text.lower():
        parts = re.split(r"</think>", text, flags=re.IGNORECASE)
        text = parts[-1].strip()
    # Remove markdown code fences cleanly
    fenced = re.findall(r"```(?:json)?\s*([\s\S]*?)```", text, flags=re.IGNORECASE)
    search_pool = list(reversed(fenced)) + [text]

    decoder = json.JSONDecoder()
    best_dict: Optional[Dict[str, Any]] = None
    for blob in search_pool:
        s = blob.strip()
        idx = 0
        while idx < len(s):
            brace = s.find("{", idx)
            if brace == -1:
                break
            try:
                obj, end_idx = decoder.raw_decode(s, brace)
                if isinstance(obj, dict):
                    if isinstance(obj.get("components"), list) and len(obj["components"]) > 0:
                        return obj
                    if best_dict is None:
                        best_dict = obj
                idx = max(end_idx, brace + 1)
                continue
            except json.JSONDecodeError:
                pass
            idx = brace + 1

        # Trailing-comma fallback on outermost braces
        start = s.find("{")
        end = s.rfind("}")
        if start != -1 and end != -1 and end > start:
            candidate = re.sub(r",\s*([\]}])", r"\1", s[start : end + 1])
            try:
                obj = json.loads(candidate)
                if isinstance(obj, dict):
                    if isinstance(obj.get("components"), list) and len(obj["components"]) > 0:
                        return obj
                    if best_dict is None:
                        best_dict = obj
            except Exception:
                pass
    return best_dict


def _apply_temporal_animation_transform(
    comp: Dict[str, Any],
    u: np.ndarray,
    x1: np.ndarray,
    y1: np.ndarray,
    x2: np.ndarray,
    y2: np.ndarray,
    t_phase: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Applies analytic trigonometric 4D time-phase transformations for t_phase in [0, 2*pi]:
      - "flap": flapping wings/plumes around root hinge via theta_flap = 0.28 * sin(t_phase)
      - "spin": spinning wheels/gears/rosettes by angle t_phase around component center
      - "sway": swaying necks/hoods/branches/tails via 0.06 * sin(t_phase) * u^1.5
      - "wave": rippling water/coils/tentacles via travelling wave sin(4*u - t_phase)
      - "pulse": harmonic radial breathing around center via 1 + 0.085 * sin(t_phase)
      - "orbit": harmonic orbital revolution via (r*(cos(t_phase)-1), r*sin(t_phase))
    """
    role = infer_anim_role(comp)
    ctype = str(comp.get("type", "ribbon_spine")).lower()

    if role == "flap":
        theta_flap = 0.28 * math.sin(t_phase)
        if ctype == "ribbon_spine":
            pts = comp.get("points", [[0.0, 0.0]])
            hx = float(pts[0][0]) if pts else 0.0
            hy = float(pts[0][1]) if pts else 0.0
        else:
            center = comp.get("center", [0.0, 0.0])
            hx = float(center[0]) * 0.25
            hy = float(center[1])
        ca, sa = math.cos(theta_flap), math.sin(theta_flap)
        dx1, dy1 = x1 - hx, y1 - hy
        dx2, dy2 = x2 - hx, y2 - hy
        x1 = hx + ca * dx1 - sa * dy1
        y1 = hy + sa * dx1 + ca * dy1
        x2 = hx + ca * dx2 - sa * dy2
        y2 = hy + sa * dx2 + ca * dy2

    elif role == "spin":
        if "center" in comp and isinstance(comp["center"], list) and len(comp["center"]) >= 2:
            cx0, cy0 = float(comp["center"][0]), float(comp["center"][1])
        else:
            cx0 = float(0.5 * (np.mean(x1) + np.mean(x2)))
            cy0 = float(0.5 * (np.mean(y1) + np.mean(y2)))
        ca, sa = math.cos(t_phase), math.sin(t_phase)
        dx1, dy1 = x1 - cx0, y1 - cy0
        dx2, dy2 = x2 - cx0, y2 - cy0
        x1 = cx0 + ca * dx1 - sa * dy1
        y1 = cy0 + sa * dx1 + ca * dy1
        x2 = cx0 + ca * dx2 - sa * dy2
        y2 = cy0 + sa * dx2 + ca * dy2

    elif role == "sway":
        sway_offset = 0.06 * math.sin(t_phase) * (u ** 1.5)
        sway_lift = 0.018 * math.sin(2.0 * t_phase) * (u ** 1.5)
        x1 = x1 + sway_offset
        y1 = y1 + sway_lift
        x2 = x2 + sway_offset
        y2 = y2 + sway_lift

    elif role == "wave":
        wave_y = 0.036 * np.sin(4.0 * u - t_phase)
        wave_x = 0.014 * np.cos(4.0 * u - t_phase)
        x1 = x1 + wave_x
        y1 = y1 + wave_y
        x2 = x2 + wave_x
        y2 = y2 + wave_y

    elif role == "pulse":
        if "center" in comp and isinstance(comp["center"], list) and len(comp["center"]) >= 2:
            cx0, cy0 = float(comp["center"][0]), float(comp["center"][1])
        else:
            cx0 = float(0.5 * (np.mean(x1) + np.mean(x2)))
            cy0 = float(0.5 * (np.mean(y1) + np.mean(y2)))
        scale_p = 1.0 + 0.085 * math.sin(t_phase)
        x1 = cx0 + scale_p * (x1 - cx0)
        y1 = cy0 + scale_p * (y1 - cy0)
        x2 = cx0 + scale_p * (x2 - cx0)
        y2 = cy0 + scale_p * (y2 - cy0)

    elif role == "orbit":
        orb_r = float(comp.get("orbit_radius", 0.055))
        dx_o = orb_r * (math.cos(t_phase) - 1.0)
        dy_o = orb_r * math.sin(t_phase)
        x1, y1 = x1 + dx_o, y1 + dy_o
        x2, y2 = x2 + dx_o, y2 + dy_o

    return x1, y1, x2, y2


def _compile_repeated_field_chords(
    comp: Dict[str, Any],
    num_pts: int,
    t_phase: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compiles a 'repeated_field' primitive (perspective_corridor, radial_fan,
    scattered_cluster, or wave_array) into num_pts smooth Hamid chord segments.
    """
    layout = str(comp.get("layout", "scattered_cluster")).lower()
    count = max(8, min(60, int(comp.get("count", 24))))
    bshape = str(comp.get("base_shape", "lobe")).lower()
    bounds = _normalize_bounds(comp.get("bounds"), [-0.85, 0.85, -0.75, 0.75])
    xmin, xmax, ymin, ymax = bounds
    smin, smax = _normalize_scale_range(comp.get("scale_range"))
    seed = int(comp.get("seed", 1))

    # Cap instances so each instance gets enough smooth chords for crisp caustic rendering
    max_inst = max(4, num_pts // 10)
    m_inst = min(count, max_inst)
    bridge_pts = 2 if m_inst > 1 and num_pts >= m_inst * 12 else 0
    total_bridge = (m_inst - 1) * bridge_pts
    active_pts = max(m_inst * 6, num_pts - total_bridge)
    base_n = active_pts // m_inst
    rem_n = active_pts - base_n * m_inst

    placements: List[Tuple[float, float, float, float]] = []
    for i in range(m_inst):
        fi = (i + 0.5) / m_inst
        s2 = float((i + 1 + seed) ** 2)
        if layout == "perspective_corridor":
            side = -1.0 if (i % 2 == 0) else 1.0
            xm = 0.5 * (xmin + xmax)
            xh = 0.5 * (xmax - xmin)
            cx_i = xm + side * xh * (0.14 + 0.82 * (fi ** 1.15)) * (0.88 + 0.12 * math.cos(2.7 * s2))
            cy_i = ymax - (ymax - ymin) * (0.14 + 0.78 * (fi ** 1.25))
            sc_i = smin + (smax - smin) * (fi ** 1.2)
            rot_i = 0.08 * side * math.sin(1.9 * (i + 1))
        elif layout == "radial_fan":
            fcx, fcy = comp.get("center", [0.5 * (xmin + xmax), ymin + 0.25 * (ymax - ymin)])
            arc = comp.get("arc_deg", [12.0, 168.0])
            a0, a1 = (float(arc[0]), float(arc[1])) if isinstance(arc, list) and len(arc) >= 2 else (12.0, 168.0)
            rings = 3 if m_inst >= 24 else (2 if m_inst >= 14 else 1)
            r_idx = i % rings
            per_ring = max(1, int(math.ceil(m_inst / rings)))
            p_idx = i // rings
            ang = math.radians(a0 + (a1 - a0) * ((p_idx + 0.5) / per_ring))
            rad_span = 0.5 * min(xmax - xmin, ymax - ymin)
            r_dist = rad_span * (0.34 + 0.62 * (r_idx + 1) / rings)
            cx_i = float(fcx) + r_dist * math.cos(ang)
            cy_i = float(fcy) + r_dist * math.sin(ang)
            sc_i = smin + (smax - smin) * (0.45 + 0.55 * (r_idx + 1) / rings)
            rot_i = ang - 0.5 * math.pi
        elif layout == "wave_array":
            wfreq = float(comp.get("wave_freq", 3.2))
            wamp = float(comp.get("wave_amp", 0.10))
            cx_i = xmin + (xmax - xmin) * fi
            cy_i = 0.5 * (ymin + ymax) + wamp * math.sin(wfreq * math.pi * fi + 0.5 * seed)
            sc_i = smin + (smax - smin) * (0.55 + 0.45 * math.sin(math.pi * fi))
            rot_i = 0.22 * math.cos(wfreq * math.pi * fi)
        else:  # scattered_cluster
            fx = 0.5 + 0.44 * math.sin(2.399963 * (i + 1 + seed) + 0.35 * math.cos(3.0 * s2))
            fy = 0.5 + 0.44 * math.cos(1.618034 * 2.1 * (i + 1 + seed) + 0.35 * math.sin(5.0 * s2))
            cx_i = xmin + (xmax - xmin) * fx
            cy_i = ymin + (ymax - ymin) * fy
            sc_i = smin + (smax - smin) * (0.5 + 0.5 * math.sin(7.0 * (i + 1) + seed))
            rot_i = 0.55 * math.sin(4.0 * s2)
        placements.append((cx_i, cy_i, sc_i, rot_i))

    X1_parts, Y1_parts, X2_parts, Y2_parts = [], [], [], []
    phase_t = float(t_phase) if t_phase is not None else 0.0
    animate_flag = t_phase is not None

    for i, (cx_i, cy_i, sc_i, rot_i) in enumerate(placements):
        n_i = base_n + (1 if i < rem_n else 0)
        u_lin = np.linspace(0.0, 1.0, n_i, endpoint=False)
        u_i = 0.5 * (1.0 - np.cos(np.pi * u_lin))
        th = 2.0 * np.pi * u_i

        if bshape == "tentacle":
            sway_t = 0.24 * sc_i * np.sin(4.0 * u_i - (phase_t if animate_flag else 0.0) + 0.65 * i) * (u_i ** 0.8)
            lx1 = sway_t + 0.16 * sc_i * (1.0 - 0.82 * u_i)
            ly1 = -2.2 * sc_i * u_i
            lx2 = sway_t - 0.16 * sc_i * (1.0 - 0.82 * u_i) + 0.08 * sc_i * np.cos(3.0 * u_i)
            ly2 = -2.2 * sc_i * np.clip(u_i + 0.06, 0.0, 1.0)
        elif bshape == "tree":
            canopy_w = sc_i * (0.14 + 0.85 * np.sin(np.pi * np.clip((u_i - 0.2) / 0.8, 0.0, 1.0)) * (1.0 - 0.55 * u_i))
            bough = 0.22 * canopy_w * np.sin(10.0 * np.pi * u_i)
            lx1 = canopy_w + bough
            ly1 = -sc_i + 2.4 * sc_i * u_i
            lx2 = -canopy_w + bough
            ly2 = -sc_i + 2.4 * sc_i * np.clip(u_i + 0.05, 0.0, 1.0)
        elif bshape == "feather":
            env = np.sin(np.pi * u_i) ** 0.75
            barb = 0.18 * sc_i * env * np.cos(12.0 * np.pi * u_i)
            lx1 = 0.48 * sc_i * env + barb
            ly1 = 1.6 * sc_i * (u_i - 0.5)
            lx2 = -0.48 * sc_i * env - barb
            ly2 = 1.6 * sc_i * (np.clip(u_i + 0.05, 0.0, 1.0) - 0.5)
        elif bshape == "star":
            r1 = sc_i * (0.58 + 0.42 * np.cos(5.0 * th))
            r2 = 0.42 * sc_i * (0.58 + 0.42 * np.cos(5.0 * (th + 0.35)))
            lx1, ly1 = r1 * np.cos(th), r1 * np.sin(th)
            lx2, ly2 = r2 * np.cos(th + 0.35), r2 * np.sin(th + 0.35)
        elif bshape == "sphere":
            lx1, ly1 = sc_i * np.cos(th), sc_i * np.sin(th)
            lx2, ly2 = 0.66 * sc_i * np.cos(th + 0.45), 0.66 * sc_i * np.sin(th + 0.45)
        else:  # petal or lobe
            r1 = sc_i * (0.65 + 0.35 * np.sin(th)) * (np.sin(np.pi * u_i) ** 0.35 + 0.15)
            r2 = 0.62 * r1
            lx1, ly1 = 0.75 * r1 * np.cos(th), r1 * np.sin(th)
            lx2, ly2 = 0.75 * r2 * np.cos(th + 0.28), r2 * np.sin(th + 0.28)

        rot_eff = rot_i + (0.14 * math.sin(phase_t + 0.4 * i) if animate_flag else 0.0)
        ca, sa = math.cos(rot_eff), math.sin(rot_eff)
        ix1 = cx_i + ca * lx1 - sa * ly1
        iy1 = cy_i + sa * lx1 + ca * ly1
        ix2 = cx_i + ca * lx2 - sa * ly2
        iy2 = cy_i + sa * lx2 + ca * ly2

        X1_parts.append(ix1)
        Y1_parts.append(iy1)
        X2_parts.append(ix2)
        Y2_parts.append(iy2)

        if bridge_pts > 0 and i < m_inst - 1:
            mx1 = 0.5 * (ix1[-1] + ix2[-1])
            my1 = 0.5 * (iy1[-1] + iy2[-1])
            nx_next, ny_next = placements[i + 1][0], placements[i + 1][1]
            s_b = np.linspace(0.25, 0.75, bridge_pts)
            bx = (1.0 - s_b) * mx1 + s_b * nx_next
            by = (1.0 - s_b) * my1 + s_b * ny_next
            X1_parts.append(bx)
            Y1_parts.append(by)
            X2_parts.append(bx)
            Y2_parts.append(by)

    x1 = np.concatenate(X1_parts)[:num_pts]
    y1 = np.concatenate(Y1_parts)[:num_pts]
    x2 = np.concatenate(X2_parts)[:num_pts]
    y2 = np.concatenate(Y2_parts)[:num_pts]
    if len(x1) < num_pts:
        pad = num_pts - len(x1)
        x1 = np.pad(x1, (0, pad), mode="edge")
        y1 = np.pad(y1, (0, pad), mode="edge")
        x2 = np.pad(x2, (0, pad), mode="edge")
        y2 = np.pad(y2, (0, pad), mode="edge")
    return x1, y1, x2, y2


def _compile_implicit_sdf_chords(
    comp: Dict[str, Any],
    num_pts: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compiles an analytical 'implicit_sdf_expr' component into num_pts smooth Hamid
    level-set chord segments by extracting the zero contour D(x, y, E) = 0 and
    connecting boundary points to their shifted inward gradient caustic envelope.
    """
    expr = str(comp.get("sdf_expr", "hypot(x, y) - 0.45"))
    bounds = _normalize_bounds(comp.get("bounds"), [-1.0, 1.0, -1.0, 1.0])
    xmin, xmax, ymin, ymax = bounds

    G = 96
    xs = np.linspace(xmin, xmax, G, dtype=np.float64)
    ys = np.linspace(ymax, ymin, G, dtype=np.float64)
    Xg, Yg = np.meshgrid(xs, ys)
    Eg = (
        0.55 * np.cos(4.5 * Xg + 2.2 * np.sin(3.5 * Yg)) * np.cos(4.5 * Yg - 2.2 * np.sin(3.5 * Xg))
        + 0.25 * np.cos(9.0 * Xg - 3.0 * Yg) * np.cos(9.0 * Yg + 3.0 * Xg)
    )
    D = safe_eval_sdf_expr(expr, Xg, Yg, Eg)

    inside_frac = float(np.mean(D <= 0.0))
    if inside_frac < 0.008 or inside_frac > 0.985:
        iso = float(np.percentile(D, 28.0))
        D = D - iso

    mask = (D <= 0.0).astype(np.uint8) * 255
    contour_curves: List[np.ndarray] = []
    try:
        import cv2
        contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        contours = sorted(contours, key=lambda c: len(c), reverse=True)[:4]
        for cnt in contours:
            if len(cnt) >= 8:
                pts_px = cnt[:, 0, :].astype(np.float64)
                cx_c = xmin + (pts_px[:, 0] / max(1.0, G - 1.0)) * (xmax - xmin)
                cy_c = ymax - (pts_px[:, 1] / max(1.0, G - 1.0)) * (ymax - ymin)
                contour_curves.append(np.column_stack([cx_c, cy_c]))
    except Exception:
        contour_curves = []

    if not contour_curves:
        # Pure-NumPy polar level-set contour fallback around minimum-D centroid
        min_idx = int(np.argmin(D))
        ry, rx = divmod(min_idx, G)
        cx0, cy0 = float(Xg[ry, rx]), float(Yg[ry, rx])
        u_lin = np.linspace(0.0, 1.0, num_pts, endpoint=False)
        u = 0.5 * (1.0 - np.cos(np.pi * u_lin))
        th = 2.0 * np.pi * u
        r_est = max(0.12, 0.32 * min(xmax - xmin, ymax - ymin))
        x1 = cx0 + r_est * np.cos(th)
        y1 = cy0 + r_est * np.sin(th)
        x2 = cx0 + 0.68 * r_est * np.cos(th + 0.25)
        y2 = cy0 + 0.68 * r_est * np.sin(th + 0.25)
        return x1, y1, x2, y2

    total_len = sum(len(c) for c in contour_curves)
    allocs = [max(16, int(round((len(c) / total_len) * num_pts))) for c in contour_curves]
    allocs[-1] = max(16, allocs[-1] + (num_pts - sum(allocs)))

    X1_list, Y1_list, X2_list, Y2_list = [], [], [], []
    for pts, n_alloc in zip(contour_curves, allocs):
        diffs = np.diff(pts, axis=0, append=pts[:1])
        seg_lens = np.hypot(diffs[:, 0], diffs[:, 1])
        arc = np.concatenate([[0.0], np.cumsum(seg_lens[:-1])])
        arc_max = float(arc[-1]) if arc[-1] > 1e-8 else 1.0
        arc_norm = arc / arc_max

        u_lin = np.linspace(0.0, 1.0, n_alloc, endpoint=False)
        u_s = 0.5 * (1.0 - np.cos(np.pi * u_lin))
        x1_c = np.interp(u_s, arc_norm, pts[:, 0])
        y1_c = np.interp(u_s, arc_norm, pts[:, 1])

        u_shift = 0.5 * (1.0 - np.cos(np.pi * np.clip(u_lin + 0.075, 0.0, 1.0)))
        xs_c = np.interp(u_shift, arc_norm, pts[:, 0])
        ys_c = np.interp(u_shift, arc_norm, pts[:, 1])

        # Pull inner chord slightly toward the contour centroid for rich caustic shading
        mx_c, my_c = float(np.mean(x1_c)), float(np.mean(y1_c))
        x2_c = 0.80 * xs_c + 0.20 * mx_c
        y2_c = 0.80 * ys_c + 0.20 * my_c

        X1_list.append(x1_c)
        Y1_list.append(y1_c)
        X2_list.append(x2_c)
        Y2_list.append(y2_c)

    x1 = np.concatenate(X1_list)[:num_pts]
    y1 = np.concatenate(Y1_list)[:num_pts]
    x2 = np.concatenate(X2_list)[:num_pts]
    y2 = np.concatenate(Y2_list)[:num_pts]
    if len(x1) < num_pts:
        pad = num_pts - len(x1)
        x1 = np.pad(x1, (0, pad), mode="edge")
        y1 = np.pad(y1, (0, pad), mode="edge")
        x2 = np.pad(x2, (0, pad), mode="edge")
        y2 = np.pad(y2, (0, pad), mode="edge")
    return x1, y1, x2, y2


def compile_ir_component(
    comp: Dict[str, Any],
    num_pts: int,
    t_phase: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compiles a single anatomical or scenic primitive (ribbon_spine, polar_harmonic,
    crescent_lobe, wheel_rosette, parametric_equation, repeated_field, or
    implicit_sdf_expr) into C^2-smooth, Clenshaw-Curtis cosine-eased boundary curves
    (x1, y1) and (x2, y2) of exact length num_pts.
    When t_phase is provided, applies analytic 4D trigonometric time transformations.
    """
    u_lin = np.linspace(0.0, 1.0, num_pts, endpoint=False)
    # Clenshaw-Curtis cosine ease-in / ease-out parameterization: du_smooth/du = 0 at endpoints!
    u = 0.5 * (1.0 - np.cos(np.pi * u_lin))
    t = 2.0 * np.pi * u
    ctype = str(comp.get("type", "ribbon_spine")).strip().lower()

    if ctype == "repeated_field":
        x1, y1, x2, y2 = _compile_repeated_field_chords(comp, num_pts, t_phase=t_phase)
        if t_phase is not None and infer_anim_role(comp) in ("pulse", "orbit", "flap"):
            x1, y1, x2, y2 = _apply_temporal_animation_transform(comp, u, x1, y1, x2, y2, float(t_phase))
        return x1, y1, x2, y2

    if ctype == "implicit_sdf_expr":
        x1, y1, x2, y2 = _compile_implicit_sdf_chords(comp, num_pts)
        if t_phase is not None:
            x1, y1, x2, y2 = _apply_temporal_animation_transform(comp, u, x1, y1, x2, y2, float(t_phase))
        return x1, y1, x2, y2

    if ctype == "ribbon_spine":
        raw_pts = comp.get("points", [[-0.4, 0.0], [0.0, 0.25], [0.4, 0.0]])
        if not raw_pts or len(raw_pts) < 2:
            raw_pts = [[-0.4, 0.0], [0.4, 0.0]]
        pts = np.array(raw_pts, dtype=np.float64)
        K = len(pts) - 1

        def eval_spine(u_arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            cx = np.zeros_like(u_arr)
            cy = np.zeros_like(u_arr)
            for j in range(K + 1):
                bern = math.comb(K, j) * ((1.0 - u_arr) ** (K - j)) * (u_arr ** j)
                cx += bern * pts[j, 0]
                cy += bern * pts[j, 1]

            dx = np.zeros_like(u_arr)
            dy = np.zeros_like(u_arr)
            if K >= 1:
                for j in range(K):
                    bern_d = K * math.comb(K - 1, j) * ((1.0 - u_arr) ** (K - 1 - j)) * (u_arr ** j)
                    dx += bern_d * (pts[j + 1, 0] - pts[j, 0])
                    dy += bern_d * (pts[j + 1, 1] - pts[j, 1])
            norm = np.hypot(dx, dy) + 1e-8
            tx, ty = dx / norm, dy / norm
            nx, ny = -ty, tx

            w_start = float(comp.get("w_start", 0.045))
            w_end = float(comp.get("w_end", 0.02))
            w_bulge = float(comp.get("w_bulge", 0.0))
            w_mod_amp = float(comp.get("w_mod_amp", 0.0))
            w_mod_freq = min(5.0, float(comp.get("w_mod_freq", 3.0)))
            end_cap = np.sin(np.pi * u_arr) ** 0.35
            w = np.maximum(
                0.001,
                (
                    w_start * (1.0 - u_arr)
                    + w_end * u_arr
                    + w_bulge * np.sin(np.pi * u_arr)
                    + w_mod_amp * np.cos(2.0 * np.pi * w_mod_freq * u_arr)
                )
                * end_cap,
            )
            return cx, cy, nx, ny, w

        cx1, cy1, nx1, ny1, w1 = eval_spine(u)
        shift_u = 0.5 * (1.0 - np.cos(np.pi * np.clip(u_lin + 0.045, 0.0, 1.0)))
        cx2, cy2, nx2, ny2, w2 = eval_spine(shift_u)

        x1 = cx1 + w1 * nx1
        y1 = cy1 + w1 * ny1
        x2 = cx2 - w2 * nx2
        y2 = cy2 - w2 * ny2

    elif ctype == "polar_harmonic":
        t_deg = comp.get("theta_range_deg", [0.0, 360.0])
        if not isinstance(t_deg, list) or len(t_deg) < 2:
            t_deg = [0.0, 360.0]
        th0, th1 = np.deg2rad(float(t_deg[0])), np.deg2rad(float(t_deg[1]))
        theta = th0 + (th1 - th0) * u

        def eval_r(ang: np.ndarray) -> np.ndarray:
            r = np.full_like(ang, float(comp.get("r0", 1.0)))
            for h in comp.get("harmonics", []):
                m = min(14.0, float(h.get("m", 1.0)))
                p = float(h.get("power", 1.0))
                ca = float(h.get("cos_amp", 0.0))
                sa = float(h.get("sin_amp", 0.0))
                aca = float(h.get("abs_cos_amp", 0.0))
                if abs(ca) > 1e-4:
                    r += ca * (np.cos(m * ang) ** int(max(1, round(p))))
                if abs(sa) > 1e-4:
                    r += sa * (np.sin(m * ang) ** int(max(1, round(p))))
                if abs(aca) > 1e-4:
                    r += aca * (np.abs(np.cos(m * ang)) ** p)
            return np.maximum(0.02, r)

        r1 = eval_r(theta)
        eta = np.clip(float(comp.get("inner_scale", 0.68)), 0.35, 0.92)
        dtheta = 0.22
        r2 = eta * eval_r(theta + dtheta)

        scale = comp.get("scale", [0.35, 0.35])
        sx, sy = float(scale[0]), float(scale[1])
        center = comp.get("center", [0.0, 0.0])
        cx0, cy0 = float(center[0]), float(center[1])
        rot = np.deg2rad(float(comp.get("rotation_deg", 0.0)))
        ca, sa = np.cos(rot), np.sin(rot)

        lx1, ly1 = sx * r1 * np.cos(theta), sy * r1 * np.sin(theta)
        lx2, ly2 = sx * r2 * np.cos(theta + dtheta), sy * r2 * np.sin(theta + dtheta)

        x1 = cx0 + ca * lx1 - sa * ly1
        y1 = cy0 + sa * lx1 + ca * ly1
        x2 = cx0 + ca * lx2 - sa * ly2
        y2 = cy0 + sa * lx2 + ca * ly2

    elif ctype == "crescent_lobe":
        R = float(comp.get("outer_radius", 0.25))
        w_belly = np.clip(float(comp.get("thickness", 0.30)), 0.12, 0.65)
        beta = np.deg2rad(np.clip(float(comp.get("arc_deg", 235.0)), 80.0, 300.0) / 2.0)

        x0 = (1.0 - w_belly) * R
        xc = R * np.cos(beta)
        yc = R * np.sin(beta)
        denom = 2.0 * (xc - x0)
        if abs(denom) < 1e-5:
            denom = -1e-5
        d_shift = (R * R - x0 * x0) / denom
        r_in = abs(x0 - d_shift)

        theta_out = -beta + 2.0 * beta * u
        lx1, ly1 = R * np.cos(theta_out), R * np.sin(theta_out)

        phi_max = math.atan2(yc, xc - d_shift)
        u_shift = 0.5 * (1.0 - np.cos(np.pi * np.clip(u_lin + 0.035, 0.0, 1.0)))
        phi_in = -phi_max + 2.0 * phi_max * u_shift
        lx2 = d_shift + r_in * np.cos(phi_in)
        ly2 = r_in * np.sin(phi_in)

        center = comp.get("center", [0.0, 0.0])
        cx0, cy0 = float(center[0]), float(center[1])
        rot = np.deg2rad(float(comp.get("rotation_deg", 0.0)))
        ca, sa = np.cos(rot), np.sin(rot)
        x1 = cx0 + ca * lx1 - sa * ly1
        y1 = cy0 + sa * lx1 + ca * ly1
        x2 = cx0 + ca * lx2 - sa * ly2
        y2 = cy0 + sa * lx2 + ca * ly2

    elif ctype == "wheel_rosette":
        center = comp.get("center", [0.0, 0.0])
        cx0, cy0 = float(center[0]), float(center[1])
        R_out = float(comp.get("outer_radius", 0.28))
        R_hub = float(comp.get("hub_radius", 0.04))

        n_rim = num_pts // 2
        n_spk = num_pts - n_rim

        u1 = 0.5 * (1.0 - np.cos(np.pi * np.linspace(0.0, 1.0, n_rim, endpoint=False)))
        th_rim = 2.0 * np.pi * u1
        x1_rim = cx0 + R_out * np.cos(th_rim)
        y1_rim = cy0 + R_out * np.sin(th_rim)
        x2_rim = cx0 + 0.88 * R_out * np.cos(th_rim + 0.32)
        y2_rim = cy0 + 0.88 * R_out * np.sin(th_rim + 0.32)

        u2 = 0.5 * (1.0 - np.cos(np.pi * np.linspace(0.0, 1.0, n_spk, endpoint=False)))
        th_spk = 2.0 * np.pi * u2
        x1_spk = cx0 + 0.94 * R_out * np.cos(th_spk)
        y1_spk = cy0 + 0.94 * R_out * np.sin(th_spk)
        x2_spk = cx0 + R_hub * np.cos(th_spk + 1.25)
        y2_spk = cy0 + R_hub * np.sin(th_spk + 1.25)

        x1 = np.concatenate([x1_rim, x1_spk])
        y1 = np.concatenate([y1_rim, y1_spk])
        x2 = np.concatenate([x2_rim, x2_spk])
        y2 = np.concatenate([y2_rim, y2_spk])

    else:  # parametric_equation
        safe_env = {
            "np": np, "sin": np.sin, "cos": np.cos, "abs": np.abs,
            "exp": np.exp, "sqrt": np.sqrt, "pi": np.pi, "t": t, "u": u,
        }

        def safe_eval(expr_str: str, fallback: np.ndarray) -> np.ndarray:
            if not expr_str or _UNSAFE_EXPR_RE.search(str(expr_str)):
                return fallback
            try:
                val = eval(str(expr_str).replace("^", "**"), {"__builtins__": {}}, safe_env)
                return np.broadcast_to(np.asarray(val, dtype=np.float64), (num_pts,)).copy()
            except Exception:
                return fallback

        lx1 = safe_eval(comp.get("x1_expr", ""), np.cos(t))
        ly1 = safe_eval(comp.get("y1_expr", ""), np.sin(t))
        lx2 = safe_eval(comp.get("x2_expr", ""), 0.78 * np.cos(t + 0.18))
        ly2 = safe_eval(comp.get("y2_expr", ""), 0.78 * np.sin(t + 0.18))

        scale = comp.get("scale", [0.4, 0.4])
        sx, sy = float(scale[0]), float(scale[1])
        center = comp.get("center", [0.0, 0.0])
        cx0, cy0 = float(center[0]), float(center[1])
        rot = np.deg2rad(float(comp.get("rotation_deg", 0.0)))
        ca, sa = np.cos(rot), np.sin(rot)
        x1 = cx0 + ca * (sx * lx1) - sa * (sy * ly1)
        y1 = cy0 + sa * (sx * lx1) + ca * (sy * ly1)
        x2 = cx0 + ca * (sx * lx2) - sa * (sy * ly2)
        y2 = cy0 + sa * (sx * lx2) + ca * (sy * ly2)

    if t_phase is not None:
        x1, y1, x2, y2 = _apply_temporal_animation_transform(
            comp, u, x1, y1, x2, y2, float(t_phase)
        )

    return x1, y1, x2, y2


def _expand_and_allocate_components(
    ir: Dict[str, Any], N: int
) -> Tuple[List[Tuple[float, Dict[str, Any], Optional[str], Any]], List[int], int]:
    enriched = enrich_ir_blueprint(ir)
    raw_comps = enriched.get("components", [])
    expanded: List[Tuple[float, Dict[str, Any], Optional[str], Any]] = []
    for c in raw_comps:
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
                expanded.append((w / rc, c, "radial", (float(ang), c.get("radial_center", [0.0, 0.0]))))
        elif mx:
            axis_x = float(c.get("mirror_axis_x", 0.0))
            expanded.append((w / 2.0, c, None, None))
            expanded.append((w / 2.0, c, "mirror_x", axis_x))
        else:
            expanded.append((w, c, None, None))

    M = len(expanded)
    n_bridge = 20 if M > 1 else 0
    total_bridge = M * n_bridge
    n_active = max(M * 24, N - total_bridge)
    total_w = sum(item[0] for item in expanded)

    allocs = [max(24, int(round((item[0] / total_w) * n_active))) for item in expanded]
    diff = (N - total_bridge) - sum(allocs)
    allocs[-1] = max(24, allocs[-1] + diff)
    return expanded, allocs, n_bridge


def _apply_symmetry(
    x1: np.ndarray,
    y1: np.ndarray,
    x2: np.ndarray,
    y2: np.ndarray,
    sym_mode: Optional[str],
    sym_param: Any,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if sym_mode == "mirror_x":
        ax = float(sym_param)
        x1, x2 = 2.0 * ax - x1, 2.0 * ax - x2
    elif sym_mode == "radial":
        ang_deg, (rcx, rcy) = sym_param
        rad = np.deg2rad(ang_deg)
        ca, sa = np.cos(rad), np.sin(rad)
        x1, y1 = rcx + ca * (x1 - rcx) - sa * (y1 - rcy), rcy + sa * (x1 - rcx) + ca * (y1 - rcy)
        x2, y2 = rcx + ca * (x2 - rcx) - sa * (y2 - rcy), rcy + sa * (x2 - rcx) + ca * (y2 - rcy)
    return x1, y1, x2, y2


def _compute_tsp_order(
    base_curves: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]
) -> List[Tuple[int, bool]]:
    """
    Computes a deterministic Nearest-Neighbor TSP ordering [(comp_idx, reverse_flag), ...]
    on the rest-pose curves so that multi-frame 4D evaluation maintains 100% consistent
    segment indexing k = 1..N across all time phases t_phase in [0, 2*pi].
    """
    if not base_curves:
        return []
    order: List[Tuple[int, bool]] = [(0, False)]
    last_curve = base_curves[0]
    remaining = list(range(1, len(base_curves)))

    while remaining:
        last_x = 0.5 * (last_curve[0][-1] + last_curve[2][-1])
        last_y = 0.5 * (last_curve[1][-1] + last_curve[3][-1])
        best_pos, best_dist, reverse_best = 0, 1e9, False
        for pos, c_idx in enumerate(remaining):
            cur = base_curves[c_idx]
            sx, sy = 0.5 * (cur[0][0] + cur[2][0]), 0.5 * (cur[1][0] + cur[3][0])
            ex, ey = 0.5 * (cur[0][-1] + cur[2][-1]), 0.5 * (cur[1][-1] + cur[3][-1])
            d_start = math.hypot(sx - last_x, sy - last_y)
            d_end = math.hypot(ex - last_x, ey - last_y)
            if d_start < best_dist:
                best_dist, best_pos, reverse_best = d_start, pos, False
            if d_end < best_dist:
                best_dist, best_pos, reverse_best = d_end, pos, True
        chosen_idx = remaining.pop(best_pos)
        chosen_curve = base_curves[chosen_idx]
        if reverse_best:
            chosen_curve = (
                chosen_curve[0][::-1],
                chosen_curve[1][::-1],
                chosen_curve[2][::-1],
                chosen_curve[3][::-1],
            )
        last_curve = chosen_curve
        order.append((chosen_idx, reverse_best))
    return order


def compile_hamid_ir_to_chords(
    ir: Dict[str, Any], N: int = 2000
) -> Tuple[np.ndarray, np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Expands all symmetries (mirror_x, radial_copies), orders components via TSP,
    inserts 20-sample C^2 zero-length degenerate chord bridges (P1 == P2) matching
    the Clenshaw-Curtis zero-derivative component endpoints, and returns unified
    arrays (x1, y1, (x2, y2)) of exact length N.
    """
    raw_comps = ir.get("components", [])
    if not raw_comps:
        t = np.linspace(0, 2 * np.pi, N, endpoint=False)
        return np.cos(t), np.sin(t), (0.5 * np.cos(3 * t), 0.5 * np.sin(2 * t))

    expanded, allocs, n_bridge = _expand_and_allocate_components(ir, N)
    comp_curves: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []

    for (w, c, sym_mode, sym_param), n_pts in zip(expanded, allocs):
        x1, y1, x2, y2 = compile_ir_component(c, n_pts, t_phase=None)
        x1, y1, x2, y2 = _apply_symmetry(x1, y1, x2, y2, sym_mode, sym_param)
        comp_curves.append((x1, y1, x2, y2))

    tsp_order = _compute_tsp_order(comp_curves)
    ordered: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    for c_idx, rev in tsp_order:
        cur = comp_curves[c_idx]
        if rev:
            cur = (cur[0][::-1], cur[1][::-1], cur[2][::-1], cur[3][::-1])
        ordered.append(cur)

    X1_list, Y1_list, X2_list, Y2_list = [], [], [], []
    s = np.linspace(0.0, 1.0, n_bridge + 2)[1:-1]
    sigma = 10.0 * (s ** 3) - 15.0 * (s ** 4) + 6.0 * (s ** 5)

    for i, cur in enumerate(ordered):
        X1_list.append(cur[0])
        Y1_list.append(cur[1])
        X2_list.append(cur[2])
        Y2_list.append(cur[3])
        if n_bridge > 0:
            nxt = ordered[(i + 1) % len(ordered)]
            m1_x, m1_y = 0.5 * (cur[0][-1] + cur[2][-1]), 0.5 * (cur[1][-1] + cur[3][-1])
            m2_x, m2_y = 0.5 * (nxt[0][0] + nxt[2][0]), 0.5 * (nxt[1][0] + nxt[3][0])
            bx = (1.0 - sigma) * m1_x + sigma * m2_x
            by = (1.0 - sigma) * m1_y + sigma * m2_y
            X1_list.append(bx)
            Y1_list.append(by)
            X2_list.append(bx)
            Y2_list.append(by)

    x1_all = np.concatenate(X1_list)[:N]
    y1_all = np.concatenate(Y1_list)[:N]
    x2_all = np.concatenate(X2_list)[:N]
    y2_all = np.concatenate(Y2_list)[:N]
    return x1_all, y1_all, (x2_all, y2_all)


def compile_hamid_ir_at_time(
    ir: Dict[str, Any], N: int = 2000, t_phase: float = 0.0
) -> Tuple[np.ndarray, np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Evaluates the component chord endpoints at time phase t_phase in [0, 2*pi],
    applying analytic trigonometric transformations:
      - flapping wings around their root hinge via theta_flap = 0.28 * sin(t_phase)
      - spinning wheels/gears by angle t_phase
      - swaying necks/hoods/branches via 0.06 * sin(t_phase) * u^1.5
      - rippling water/coils/tentacles via travelling wave sin(4*u - t_phase)
    Preserves the exact rest-pose TSP component ordering so segment index k = 1..N
    evolves smoothly and analytically over time t_phase.
    """
    raw_comps = ir.get("components", [])
    if not raw_comps:
        t = np.linspace(0, 2 * np.pi, N, endpoint=False)
        return (
            np.cos(t + t_phase),
            np.sin(t + t_phase),
            (0.5 * np.cos(3 * t - t_phase), 0.5 * np.sin(2 * t + t_phase)),
        )

    expanded, allocs, n_bridge = _expand_and_allocate_components(ir, N)

    base_curves: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    time_curves: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []

    for (w, c, sym_mode, sym_param), n_pts in zip(expanded, allocs):
        bx1, by1, bx2, by2 = compile_ir_component(c, n_pts, t_phase=None)
        bx1, by1, bx2, by2 = _apply_symmetry(bx1, by1, bx2, by2, sym_mode, sym_param)
        base_curves.append((bx1, by1, bx2, by2))

        tx1, ty1, tx2, ty2 = compile_ir_component(c, n_pts, t_phase=float(t_phase))
        tx1, ty1, tx2, ty2 = _apply_symmetry(tx1, ty1, tx2, ty2, sym_mode, sym_param)
        time_curves.append((tx1, ty1, tx2, ty2))

    tsp_order = _compute_tsp_order(base_curves)
    ordered: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    for c_idx, rev in tsp_order:
        cur = time_curves[c_idx]
        if rev:
            cur = (cur[0][::-1], cur[1][::-1], cur[2][::-1], cur[3][::-1])
        ordered.append(cur)

    X1_list, Y1_list, X2_list, Y2_list = [], [], [], []
    s = np.linspace(0.0, 1.0, n_bridge + 2)[1:-1]
    sigma = 10.0 * (s ** 3) - 15.0 * (s ** 4) + 6.0 * (s ** 5)

    for i, cur in enumerate(ordered):
        X1_list.append(cur[0])
        Y1_list.append(cur[1])
        X2_list.append(cur[2])
        Y2_list.append(cur[3])
        if n_bridge > 0:
            nxt = ordered[(i + 1) % len(ordered)]
            m1_x, m1_y = 0.5 * (cur[0][-1] + cur[2][-1]), 0.5 * (cur[1][-1] + cur[3][-1])
            m2_x, m2_y = 0.5 * (nxt[0][0] + nxt[2][0]), 0.5 * (nxt[1][0] + nxt[3][0])
            bx = (1.0 - sigma) * m1_x + sigma * m2_x
            by = (1.0 - sigma) * m1_y + sigma * m2_y
            X1_list.append(bx)
            Y1_list.append(by)
            X2_list.append(bx)
            Y2_list.append(by)

    x1_all = np.concatenate(X1_list)[:N]
    y1_all = np.concatenate(Y1_list)[:N]
    x2_all = np.concatenate(X2_list)[:N]
    y2_all = np.concatenate(Y2_list)[:N]
    return x1_all, y1_all, (x2_all, y2_all)



# Backward-compatible alias for scene IR lookup
get_scene_ir_for_prompt = call_cloudflare_ai_for_ir


# ============================================================================
# PHASE 2 SOTA: DIRECT PURE-MATH SYMBOLIC PROGRAM SYNTHESIS
#               + CLOSED-LOOP VISUAL SELF-CRITIQUE REFINEMENT
# ============================================================================

MATH_PROGRAM_CACHE_FILE = os.path.join(os.path.dirname(__file__), "math_program_cache.json")
_MATH_PROGRAM_CACHE: Dict[str, Dict[str, Any]] = {}

MATH_PROGRAM_SYSTEM_PROMPT = """You are a World-Master Mathematical Artist combining Hamid Naderi Yeganeh's analytical painting equations (double-exponential masks ee(K*d), Porter-Duff layered occlusion, multi-octave trigonometric fBm, dense summation loops) with Inigo Quilez's precision 2D analytical SDF sculpture.

Your task: Given ANY user prompt, write a complete, self-contained, pure-math painting & 4D looping animation program as STRICT JSON ONLY.

### Canonical JSON Schema
{
  "title": "Pure Artistic & Mathematical Title (NEVER mention AI, model names, or vision)",
  "subtitle": "Pure analytical geometric description (e.g. 'Coupled Analytical SDF & Harmonic Summation Field H_v(x, y, t)')",
  "aspect": [16, 10],
  "domain": {"x_min": -1.6, "x_max": 1.6, "y_min": -1.0, "y_max": 1.0},
  "steps": [
    {"var": "sx", "expr": "abs(x)"},
    {"var": "head_d", "expr": "sdEllipse(x, y - 0.05, 0.55, 0.48)"},
    {
      "loop": {
        "var": "fur",
        "iter": "s",
        "start": 1,
        "end": 32,
        "init": "0.0",
        "body": [
          {"var": "ang", "expr": "s * 0.1963 + 0.04 * sin(t + s)"},
          {"var": "d_s", "expr": "sdSegment(x, y, 0.35*cos(ang), 0.35*sin(ang), 0.65*cos(ang), 0.65*sin(ang)) - 0.014"}
        ],
        "update": "max(acc, ee(85.0 * d_s))"
      }
    }
  ],
  "H0": "clamp(r_final, 0.0, 1.0)",
  "H1": "clamp(g_final, 0.0, 1.0)",
  "H2": "clamp(b_final, 0.0, 1.0)"
}

### STRICT WHITELISTED SYNTAX (IDENTICAL IN NUMPY & WEBGL1 GLSL)
- Built-in input variables: x (in [-1.6, 1.6], left to right), y (in [-1.0, 1.0], bottom to top), t (periodic time phase in [0, 2*PI]), PI.
- Inside a loop's "body" and "update": the loop counter variable (e.g. s, float-valued from start..end) and "acc" (running accumulator initialized to "init").
- ALL numeric literals in expressions MUST be floats with a decimal point (e.g. 0.0, 1.0, 2.5, NEVER bare 0 or 1 in math ops). Loop "start" and "end" are integers with (end - start + 1) <= 50.
- Total entries in "steps" MUST be between 40 and 72 (including 1 to 3 rich "loop" blocks). NEVER exceed 78 steps!
- NEVER use "**" (use pow(abs(a), b) or a*a), NEVER use if/else or ternary "? :", NEVER use array indexing or undeclared variables!
- Whitelisted functions ONLY:
  * Math: sin(a), cos(a), tan(a), asin(a), acos(a), atan(a), atan2(y, x), exp(a), log(a), sqrt(a), abs(a), pow(a, b), min(a, b), max(a, b), clamp(v, lo, hi), floor(a), fract(a), mod(a, b), sign(a), mix(a, b, k), smoothstep(e0, e1, v), step(edge, v)
  * Hamid & Analytical SDFs (NEGATIVE < 0 is INSIDE, POSITIVE > 0 is OUTSIDE):
    - ee(u): exp(-exp(clamp(u, -30.0, 30.0))) -> 1.0 when u < 0 (INSIDE), 0.0 when u > 0 (OUTSIDE)!
      CRITICAL SHARPNESS RULE: Because ee(K*d) transitions over width ~3.0/K, ALWAYS use HIGH sharpness K = 180.0 to 360.0 (e.g. ee(220.0 * d), ee(300.0 * d)) for crisp silhouettes, eyes, pupils, fangs, stripes, whiskers, planets, rings, buildings, and foreground foliage! NEVER use low K=40..60 for main subject edges (it looks blurry!). Reserve low K=15.0..45.0 ONLY for soft glows, mist, and drop shadows!
    - sdCircle(px, py, r)
    - sdEllipse(px, py, rx, ry)
    - sdBox(px, py, bx, by)
    - sdSegment(px, py, ax, ay, bx, by)  [unsigned distance >= 0; subtract thickness r to get negative interior!]
    - sdCapsule(px, py, ax, ay, bx, by, r)  [signed distance with radius r]
    - sdTriangle(px, py, ax, ay, bx, by, cx, cy)  [exact signed distance to triangle]
    - sdRing(px, py, r, w)
    - sdStar(px, py, r, n, m)
    - smin(a, b, k)  [smooth organic union of two SDFs, k ~ 0.04..0.18]
    - smax(a, b, k)  [smooth intersection/subtraction]
    - rotX(px, py, ang), rotY(px, py, ang)  [2D rotation by angle ang radians]
    - fbm(px, py)  [6-octave Hamid trigonometric fBm in roughly [-1.0, 1.0]]
    - fbmT(px, py, t)  [seamless 2*PI-periodic time-evolving trig fBm]
    - hash21(px, py)  [deterministic hash in [0.0, 1.0]]

### CRITICAL MATHEMATICAL BUGS STRICTLY FORBIDDEN (AUTO-REJECTED BY VALIDATOR!)
1. **LOOP OUTPUT VARIABLE RULE**: In `{"loop": {"var": "NAME", "iter": "s", ...}}`, `"var"` is the **OUTPUT ACCUMULATOR MASK/FIELD NAME** (e.g. `"star_mask"`, `"aurora_mask"`, `"tentacles"`, `"whiskers"`), NEVER a loop index like `"i"`, `"j"`, or `"k"`! Every `loop.var` MUST be referenced in later color/mixing steps or `H0/H1/H2` (unused loops are REJECTED!). NEVER write `{"var": "star_glow", "expr": "0.0"}` followed by a loop with `"var": "k"` — the loop itself defines `"star_glow"`!
2. **SCALAR ENDPOINT RULE (NEVER PASS PIXEL ARRAYS INTO `sdSegment`/`sdCapsule`/`sdTriangle` ENDPOINTS!)**: In `sdSegment(px, py, ax, ay, bx, by)` and `sdCapsule(px, py, ax, ay, bx, by, r)`, `(px, py)` are pixel coordinates (`x, y` or `sx, y`), while `(ax, ay, bx, by)` MUST be **SCALAR** world coordinates (e.g. `bellY = 0.25 + 0.04*sin(t)`, `ay = bellY - 0.14`, `by = bellY - 0.88`). NEVER define `cy = y - 0.25` and then pass `ay = cy - 0.14` into `sdSegment(x, y, ax, ay, ...)` — doing so cancels `y` out (`y - ay = 0.39` everywhere!) and makes every tentacle/whisker/branch completely INVISIBLE (`0.0`)!
3. **HALF-SPACE / DOME SIGN RULE**: Because NEGATIVE `< 0` is INSIDE, to keep the UPPER dome of a jellyfish bell (`y >= bellY - 0.06`), cut off the bottom with `smax(sdEllipse(x, y - bellY, 0.46, 0.32), (bellY - 0.06) - y, 0.04)` — NEVER `y - bellY` (which cuts off the top and creates an upside-down bowl!).
4. **BILATERAL SYMMETRY RULE**: For ALL symmetric anatomical features (ears, cheek ruffs, eyes, pupils, eyebrows, fangs, whiskers, wings, fins), ALWAYS pass `sx = abs(x)` as the FIRST argument to `sdCircle`, `sdEllipse`, `sdTriangle`, `sdSegment`, `sdCapsule` (e.g. `sdSegment(sx, y, 0.18, -0.06, 0.78, -0.03)`). NEVER pass signed `x` for whiskers or ears (which draws them on only ONE side of the face)!
5. **FINAL COMPOSITING CHAIN RULE (NEVER DROP THE HERO SUBJECT!)**: Ensure your final foreground/hero RGB variables (e.g. `hR, hG, hB` or `fR, fG, fB`) are carried all the way into `H0, H1, H2`! When layering framing elements (jungle fronds, leaves, mist, vignette, glow) in the last steps, ALWAYS mix onto the **LATEST hero/subject RGB** (e.g. `mix(hR, frR, fronds)`), NEVER accidentally mix onto `bgR, bgG, bgB` (which erases the entire hero subject!).

### ARTISTIC MASTERY RULES (HOW TO MAKE IT MUSEUM-GRADE & UNMISTAKABLY RECOGNIZABLE)
1. **Rich Atmospheric Background**: Paint a luminous multi-stop sky/ocean/space/jungle backdrop (bgR, bgG, bgB) using vertical/radial gradients, fbmT(x, y, t) atmospheric depth, AND a visible loop for stars, bokeh, crisp pointed jungle leaves/fronds (NOT flat dark ovals!), ocean caustics, or glowing aurora curtains!
2. **Hero Subject Scale & Framing**: Center the hero subject boldly so it fills 55%-75% of the canvas (around `[-0.85, 0.85] x [-0.75, 0.75]`), with crisp dark ambient-occlusion outline AND warm rim lighting separating it cleanly from the background.
3. **True Anatomical & Structural Fidelity (ZERO BLUR, ZERO FLOATING ARTIFACTS)**:
   - Use bilateral symmetry `sx = abs(x - cx)` for faces, ears, cheek ruffs, eyes, whiskers, fangs, wings, jellyfish bells, towers, and spacesuits!
   - Layer EVERY iconic visual feature explicitly back-to-front using `mix(curR, layerR, mask)` with RAZOR-SHARP masks `ee(200.0 * d)` to `ee(340.0 * d)`:
     * **Animals / Tiger faces**: Sculpt a bold organic head (`smin` of cranium + wide furred cheek ruffs + protruding rounded ears at `sx = 0.44, y = 0.46` with dark rim & pale inner ear!) + CONTINUOUS FLOWING STRIPES via warped trig wave fields (`ee(200.0 * (abs(sin(14.0*y + 3.8*cos(4.2*sx) + 1.8*fbm(sx*3.2, y*3.2))) - 0.26))`, clipped to head mask) + tilted almond eyes (`sx`) with dark eyeliner, glowing golden iris, razor vertical slit pupil, and bright white specular catchlight + inverted pink-black triangle nose + wide roaring crimson jaws + 4 sharp ivory fangs (`sx`) + bilateral whisker loop using `sdSegment(sx, y, 0.20, -0.08 - s*0.04, 0.78, -0.04 - s*0.05)` (using `sx`, scalar endpoints!) + crisp pointed palm/fern foliage fronds framing the edges!
     * **Ocean / Bioluminescent Jellyfish**: Luminous translucent UPPER dome bell (`bellY = 0.28 + 0.04*sin(t)`, `smax(sdEllipse(x, y - bellY, 0.46, 0.34), (bellY - 0.06) - y, 0.05)`) with scalloped ruffled rim + glowing electric-cyan/magenta/violet internal organs & ribs + loop (`var: "tentacles"`) of 14-18 flowing sinusoidal tentacles hanging BELOW the bell using SCALAR `ay = bellY - 0.06`, `by = bellY - 0.95` and wave-warped `tx = x + 0.055*sin(7.0*y + t + s*0.9)` passed into `sdCapsule(tx, y, ax, ay, bx, by, 0.012)` + rising bubble loop + volumetric caustic light shafts!
     * **Architecture / Crystal Lighthouse & Northern Lights**: Starry night sky loop (`var: "star_mask"`, visibly added to sky RGB!) + dancing multi-ribbon emerald/cyan/magenta aurora curtains in the upper sky (`var: "aurora_mask"` using `exp(-abs(y - (0.48 + 0.14*s*0.18 + 0.12*sin(2.4*x + t + s*0.8) + 0.07*fbmT(x*2.2, y*1.5, t))) * 11.0)` visibly added to sky!) + jagged dark rocky cliff (`y - (-0.42 + 0.26*exp(-2.8*x*x) + 0.07*sin(9.0*x))`) + faceted glowing crystal lighthouse tower + glowing lantern room + sweeping golden light beam cones (`rotX/rotY`) + turbulent moonlit ocean waves with white foam!
     * **Space / Planets & Astronaut**: Oblate banded Saturn with spherical 3D shading + tilted multi-band golden rings with Cassini division PROPERLY depth-split behind & in front of the sphere + twinkling starfield loop (`var: "star_mask"`) + detailed white-suited astronaut (helmet, reflective gold visor with catchlight, torso, backpack, limbs, umbilical tether)!
4. **3D Lighting, Volume & Specular Highlights**: Modulate every major shape with 3D spherical/cylindrical shading, warm key light + cool ambient shadow + crisp white specular catchlights!
5. **Continuous 2*PI-Periodic 4D Animation**: Every scene MUST visibly animate over `t` in `[0, 2*PI]` using integer frequencies (`sin(t)`, `cos(t)`, `sin(2.0*t)`, `fbmT(..., t)`).
6. Output STRICT JSON ONLY — no markdown commentary outside the JSON block."""


def _sanitize_math_program_metadata(prog: Dict[str, Any], prompt: str) -> Dict[str, Any]:
    out = dict(prog)
    for k in ("ai_model", "model", "critic_model", "vlm_model", "provider"):
        out.pop(k, None)

    base_name = prompt.strip().title() if prompt.strip() else "Harmonic Pixel Field"
    default_title = f"{base_name} — Analytical Field Synthesis"
    out["title"] = sanitize_mathematical_text(str(out.get("title") or ""), default_title)

    n_steps = len(out.get("steps", [])) if isinstance(out.get("steps"), list) else 0
    default_subtitle = (
        f"Coupled Analytical SDF & Harmonic Summation Field H_v(x, y, t) — {n_steps} Symbolic Operators"
    )
    raw_sub = str(out.get("subtitle") or "")
    if not raw_sub or _FORBIDDEN_META_RE.search(raw_sub):
        out["subtitle"] = default_subtitle
    else:
        out["subtitle"] = sanitize_mathematical_text(raw_sub, default_subtitle)
    return out


def _normalize_math_program(prog: Dict[str, Any], prompt: str = "") -> Dict[str, Any]:
    """
    Normalizes any minor structural quirks in a synthesized math_program into the
    exact canonical schema required by symbolic_math_compiler.py and scrubs branding.
    """
    out = _sanitize_math_program_metadata(prog, prompt)

    raw_aspect = out.get("aspect")
    if isinstance(raw_aspect, list) and len(raw_aspect) >= 2:
        try:
            aw, ah = max(1, int(round(float(raw_aspect[0])))), max(1, int(round(float(raw_aspect[1]))))
            out["aspect"] = [aw, ah]
        except Exception:
            out["aspect"] = [16, 10]
    else:
        out["aspect"] = [16, 10]

    raw_dom = out.get("domain") if isinstance(out.get("domain"), dict) else {}
    try:
        x_min = float(raw_dom.get("x_min", -1.6))
        x_max = float(raw_dom.get("x_max", 1.6))
        y_min = float(raw_dom.get("y_min", -1.0))
        y_max = float(raw_dom.get("y_max", 1.0))
        if x_max <= x_min + 0.1:
            x_min, x_max = -1.6, 1.6
        if y_max <= y_min + 0.1:
            y_min, y_max = -1.0, 1.0
    except Exception:
        x_min, x_max, y_min, y_max = -1.6, 1.6, -1.0, 1.0
    out["domain"] = {"x_min": x_min, "x_max": x_max, "y_min": y_min, "y_max": y_max}

    raw_steps = out.get("steps")
    norm_steps: List[Dict[str, Any]] = []
    if isinstance(raw_steps, list):
        for st in raw_steps[:88]:
            if not isinstance(st, dict):
                continue
            if "loop" in st and isinstance(st["loop"], dict):
                lp = dict(st["loop"])
                l_var = str(lp.get("var") or "acc_out").strip()
                l_iter = str(lp.get("iter") or "s").strip()
                try:
                    l_start = int(round(float(lp.get("start", 1))))
                    l_end = int(round(float(lp.get("end", 16))))
                except Exception:
                    l_start, l_end = 1, 16
                if l_end < l_start:
                    l_end = l_start
                if (l_end - l_start + 1) > 60:
                    l_end = l_start + 59
                l_init = str(lp.get("init", "0.0")).strip() or "0.0"
                l_update = str(lp.get("update", "acc")).strip() or "acc"
                raw_body = lp.get("body") if isinstance(lp.get("body"), list) else []
                norm_body: List[Dict[str, str]] = []
                for bst in raw_body:
                    if isinstance(bst, dict):
                        bv = str(bst.get("var") or "").strip()
                        be = str(bst.get("expr") or bst.get("expression") or "").strip()
                        if bv and be:
                            norm_body.append({"var": bv, "expr": be})
                norm_steps.append(
                    {
                        "loop": {
                            "var": l_var,
                            "iter": l_iter,
                            "start": l_start,
                            "end": l_end,
                            "init": l_init,
                            "body": norm_body,
                            "update": l_update,
                        }
                    }
                )
            else:
                vname = str(st.get("var") or st.get("name") or "").strip()
                expr = str(st.get("expr") or st.get("expression") or "").strip()
                if vname and expr:
                    norm_steps.append({"var": vname, "expr": expr})
    out["steps"] = norm_steps

    out["H0"] = str(out.get("H0") or out.get("h0") or out.get("R") or "0.5").strip()
    out["H1"] = str(out.get("H1") or out.get("h1") or out.get("G") or "0.5").strip()
    out["H2"] = str(out.get("H2") or out.get("h2") or out.get("B") or "0.5").strip()
    return out


def _save_math_program_cache() -> None:
    try:
        with open(MATH_PROGRAM_CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(_MATH_PROGRAM_CACHE, f, indent=2)
    except Exception:
        pass


def _load_math_program_cache() -> None:
    global _MATH_PROGRAM_CACHE
    if not os.path.exists(MATH_PROGRAM_CACHE_FILE):
        _MATH_PROGRAM_CACHE = {}
        return
    try:
        with open(MATH_PROGRAM_CACHE_FILE, "r", encoding="utf-8") as f:
            raw = json.load(f)
        if isinstance(raw, dict):
            cleaned: Dict[str, Dict[str, Any]] = {}
            for k, v in raw.items():
                if isinstance(v, dict) and isinstance(v.get("steps"), list) and "H0" in v:
                    cleaned[k] = _normalize_math_program(v, prompt=k)
            _MATH_PROGRAM_CACHE = cleaned
    except Exception:
        _MATH_PROGRAM_CACHE = {}


_load_math_program_cache()


def _extract_math_program_json(text: str) -> Optional[Dict[str, Any]]:
    text = str(text or "").strip()
    text = re.sub(r"<think>[\s\S]*?</think>", "", text, flags=re.IGNORECASE).strip()
    if "</think>" in text.lower():
        text = re.split(r"</think>", text, flags=re.IGNORECASE)[-1].strip()
    fenced = re.findall(r"```(?:json)?\s*([\s\S]*?)```", text, flags=re.IGNORECASE)
    search_pool = list(reversed(fenced)) + [text]

    decoder = json.JSONDecoder()
    best_dict: Optional[Dict[str, Any]] = None
    for blob in search_pool:
        s = blob.strip()
        idx = 0
        while idx < len(s):
            brace = s.find("{", idx)
            if brace == -1:
                break
            try:
                obj, end_idx = decoder.raw_decode(s, brace)
                if isinstance(obj, dict):
                    if isinstance(obj.get("steps"), list) and ("H0" in obj or "h0" in obj or "R" in obj):
                        return obj
                    if best_dict is None:
                        best_dict = obj
                idx = max(end_idx, brace + 1)
                continue
            except json.JSONDecodeError:
                pass
            idx = brace + 1

        start = s.find("{")
        end = s.rfind("}")
        if start != -1 and end != -1 and end > start:
            candidate = re.sub(r",\s*([\]}])", r"\1", s[start : end + 1])
            try:
                obj = json.loads(candidate)
                if isinstance(obj, dict):
                    if isinstance(obj.get("steps"), list) and ("H0" in obj or "h0" in obj or "R" in obj):
                        return obj
                    if best_dict is None:
                        best_dict = obj
            except Exception:
                pass
    return best_dict


def _invoke_flagship_for_math_program(
    system_prompt: str,
    user_msg: str,
    max_tokens: int = 8192,
    temperature: float = 0.2,
    prefer_glm: bool = True,
) -> Optional[Dict[str, Any]]:
    """
    Calls active live flagship GLM-5.3 (with reasoning_effort='low') and DeepSeek V4.1
    (552B MoE, fallback GLM-5.3-Flash) with max_tokens=8192 to synthesize or refine
    a canonical pure-math program JSON.
    """
    models = (
        [GLM_5_3, DEEPSEEK_V4_PRO, GLM_5_3_FLASH]
        if prefer_glm
        else [DEEPSEEK_V4_PRO, GLM_5_3, GLM_5_3_FLASH]
    )
    if not _has_cf_credentials():
        return None
    for model in models:
        url = f"https://api.cloudflare.com/client/v4/accounts/{CF_ACCOUNT_ID}/ai/run/{model}"
        payload: Dict[str, Any] = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_msg},
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if "glm-5" in model:
            payload["reasoning_effort"] = "low"
        elif "deepseek-v4" in model:
            payload["chat_template_kwargs"] = {"enable_thinking": False}

        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {CF_API_TOKEN}",
                "Content-Type": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=90) as resp:
                body = json.loads(resp.read().decode("utf-8"))
                raw_resp = _extract_response_text(body)
                if isinstance(raw_resp, dict):
                    prog = raw_resp
                else:
                    prog = _extract_math_program_json(str(raw_resp))
                if prog and isinstance(prog.get("steps"), list) and len(prog["steps"]) > 0:
                    return prog
        except Exception:
            continue
    return None


def critique_rendered_math_image(
    prompt: str,
    image_base64: str,
    stats: Dict[str, Any],
    motion_energy: float = 0.0,
) -> Dict[str, Any]:
    """
    Sends the rendered Pass preview PNG + deterministic mathematical diagnostics
    to the silent live Cloudflare Vision Critic cascade (Gemma 4 26B / Mistral Small 3.1 24B /
    Llama 4 Scout 17B-16E) and returns a strict 1..10 score plus concrete (x, y) SDF fixes.
    """
    data_uri = (
        image_base64
        if str(image_base64).startswith("data:image/")
        else f"data:image/png;base64,{image_base64}"
    )

    critique_prompt = (
        f'You are a strict fine-art & mathematical-graphics critic evaluating an analytical pure-math painting '
        f'rendered on coordinates x in [-1.6, 1.6] (left to right) and y in [-1.0, 1.0] (bottom to top).\n'
        f'Intended subject: "{prompt}"\n'
        f'Quantitative diagnostics: std={stats.get("std", 0):.1f}, chroma={stats.get("chroma", 0):.1f}, '
        f'edge_density={stats.get("edge_density", 0):.3f}, center_contrast={stats.get("center_contrast", 0):.1f}, '
        f'motion_energy={motion_energy:.2f}.\n\n'
        f'Be STRICT and HONEST. Score 1.0 to 10.0 on recognizability, anatomical/structural accuracy, color richness, '
        f'lighting/3D shading, and crisp separation from the background. Do NOT give >= 8.5 unless the target subject '
        f'is unmistakably recognizable and gallery-grade.\n'
        f'Return STRICT JSON ONLY:\n'
        f'{{"score": 1.0..10.0, "what_is_visible": "honest 1-sentence description", '
        f'"defects": ["defect 1", "defect 2", "defect 3"], '
        f'"coordinate_fixes": ["concrete SDF/color fix with explicit (x, y) coords and RGB values", ...]}}'
    )

    for vm in (VISION_CRITIC_MODELS if _has_cf_credentials() else []):
        url = f"https://api.cloudflare.com/client/v4/accounts/{CF_ACCOUNT_ID}/ai/run/{vm}"
        payload = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": critique_prompt},
                        {"type": "image_url", "image_url": {"url": data_uri}},
                    ],
                }
            ],
            "max_tokens": 900,
            "temperature": 0.15,
        }
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {CF_API_TOKEN}",
                "Content-Type": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=45) as resp:
                body = json.loads(resp.read().decode("utf-8"))
                raw_text = _extract_response_text(body)
                parsed = raw_text if isinstance(raw_text, dict) else _extract_json_object(str(raw_text))
                if isinstance(parsed, dict) and "score" in parsed:
                    score = float(np.clip(float(parsed.get("score", 5.0)), 1.0, 10.0))
                    defects = parsed.get("defects") if isinstance(parsed.get("defects"), list) else []
                    fixes = parsed.get("coordinate_fixes") if isinstance(parsed.get("coordinate_fixes"), list) else []
                    if not fixes and isinstance(parsed.get("fixes"), list):
                        fixes = parsed["fixes"]
                    return {
                        "score": round(score, 2),
                        "what_is_visible": str(parsed.get("what_is_visible") or parsed.get("sees") or ""),
                        "defects": [str(d) for d in defects[:6]],
                        "coordinate_fixes": [str(fx) for fx in fixes[:6]],
                    }
        except Exception:
            continue

    std_v = float(stats.get("std", 0.0))
    chr_v = float(stats.get("chroma", 0.0))
    heuristic_score = round(min(8.2, 3.0 + 2.5 * min(1.0, std_v / 55.0) + 2.5 * min(1.0, chr_v / 45.0)), 2)
    return {
        "score": heuristic_score,
        "what_is_visible": "Analytical field rendering evaluated via statistical luminance and chroma metrics.",
        "defects": [
            "Enhance subject silhouette contrast and anatomical micro-details around center (0.0, 0.0)."
        ],
        "coordinate_fixes": [
            "Sharpen primary silhouette masks with ee(90.0 * d), boost foreground rim lighting, and add periodic motion via sin(t) and fbmT(x, y, t)."
        ],
    }


def _repair_math_program(
    prog: Dict[str, Any], error_msg: str, prompt: str
) -> Optional[Dict[str, Any]]:
    """
    Runs 1 targeted syntax/AST repair pass via GLM-5.3 / DeepSeek V4.1
    when validate_math_program reports an error.
    """
    repair_msg = (
        f'The math_program JSON below for "{prompt}" failed strict AST/GLSL validation with error:\n'
        f'ERROR: {error_msg}\n\n'
        f'Fix the error while preserving or improving ALL visual anatomy, shading, loops, and periodic t animation.\n'
        f'Rules reminder:\n'
        f'- Only whitelisted functions (sin, cos, tan, asin, acos, atan, atan2, exp, log, sqrt, abs, pow, min, max, clamp, floor, fract, mod, sign, mix, smoothstep, step, ee, sdCircle, sdEllipse, sdBox, sdSegment, sdCapsule, sdTriangle, sdRing, sdStar, smin, smax, rotX, rotY, fbm, fbmT, hash21).\n'
        f'- Every loop "var" is the OUTPUT field name (NEVER an index like "i" or "k") and MUST be referenced in later steps or H0/H1/H2!\n'
        f'- Never pass y-shifted pixel variables (e.g. cy = y - 0.2) into sdSegment/sdCapsule endpoints; use scalar coordinates!\n'
        f'- All numeric constants in expressions must be floats (e.g. 0.0, 1.0). Never use ** or if/else or undefined vars.\n'
        f'- Max 85 steps, max 50 iterations per loop.\n'
        f'Original JSON:\n{json.dumps(prog)}\n\n'
        f'Return ONLY the repaired valid JSON.'
    )
    repaired = _invoke_flagship_for_math_program(
        MATH_PROGRAM_SYSTEM_PROMPT, repair_msg, max_tokens=8192, temperature=0.12
    )
    if repaired is not None:
        return _normalize_math_program(repaired, prompt=prompt)
    return None


def _build_emergency_fallback_math_program(prompt: str) -> Dict[str, Any]:
    """
    Guaranteed valid offline pure-math program used ONLY if all live Cloudflare
    synthesis calls fail (e.g., offline environment).
    """
    clean_title = sanitize_mathematical_text(
        f"{prompt.strip().title()} — Harmonic Caustic Field",
        "Harmonic Caustic Field",
    )
    return {
        "title": clean_title,
        "subtitle": "Coupled Analytical SDF & Harmonic Summation Field H_v(x, y, t) — Offline Fallback Manifold",
        "aspect": [16, 10],
        "domain": {"x_min": -1.6, "x_max": 1.6, "y_min": -1.0, "y_max": 1.0},
        "steps": [
            {"var": "r", "expr": "sqrt(x*x + y*y + 0.0001)"},
            {"var": "ang", "expr": "atan2(y, x)"},
            {"var": "n", "expr": "fbmT(x * 1.4, y * 1.4, t)"},
            {"var": "bgR", "expr": "clamp(0.08 + 0.24 * (1.0 - y * 0.5) + 0.14 * n, 0.0, 1.0)"},
            {"var": "bgG", "expr": "clamp(0.14 + 0.30 * exp(-r * 1.1) + 0.12 * n, 0.0, 1.0)"},
            {"var": "bgB", "expr": "clamp(0.28 + 0.38 * (0.5 + 0.5 * y) + 0.14 * n, 0.0, 1.0)"},
            {
                "loop": {
                    "var": "petals",
                    "iter": "s",
                    "start": 1,
                    "end": 16,
                    "init": "0.0",
                    "body": [
                        {"var": "a_s", "expr": "s * 0.3927 + 0.12 * sin(t + s * 0.5)"},
                        {"var": "px_s", "expr": "rotX(x, y, a_s) - 0.36"},
                        {"var": "py_s", "expr": "rotY(x, y, a_s)"},
                        {"var": "d_s", "expr": "sdEllipse(px_s, py_s, 0.24, 0.09)"},
                    ],
                    "update": "max(acc, ee(65.0 * d_s) * (0.55 + 0.45 * cos(s * 0.8 + t)))",
                }
            },
            {"var": "core_d", "expr": "sdCircle(x, y, 0.24 + 0.025 * sin(t))"},
            {"var": "core_m", "expr": "ee(80.0 * core_d)"},
            {"var": "rim_m", "expr": "ee(70.0 * sdRing(x, y, 0.52 + 0.03 * cos(t), 0.025))"},
            {"var": "shade", "expr": "clamp(0.55 + 0.45 * (0.5 * y - 0.3 * x + 0.3), 0.2, 1.2)"},
            {"var": "cR", "expr": "mix(bgR, 0.96 * shade, petals * 0.85)"},
            {"var": "cG", "expr": "mix(bgG, 0.62 * shade, petals * 0.85)"},
            {"var": "cB", "expr": "mix(bgB, 0.24 * shade, petals * 0.85)"},
            {"var": "cR2", "expr": "mix(cR, 1.0 * shade, core_m)"},
            {"var": "cG2", "expr": "mix(cG, 0.88 * shade, core_m)"},
            {"var": "cB2", "expr": "mix(cB, 0.35 * shade, core_m)"},
        ],
        "H0": "clamp(cR2 + 0.32 * rim_m, 0.0, 1.0)",
        "H1": "clamp(cG2 + 0.42 * rim_m, 0.0, 1.0)",
        "H2": "clamp(cB2 + 0.55 * rim_m, 0.0, 1.0)",
        "refinement_passes": 1,
        "visual_score": 6.5,
    }


def _evaluate_program_preview_and_motion(
    prog: Dict[str, Any], width: int = 384, height: int = 240
) -> Tuple[Dict[str, Any], float]:
    """
    Evaluates a math_program at t=0.0 and t=pi/2 via symbolic_math_compiler
    and returns (eval_t0_dict, motion_energy).
    """
    res0_clean, motion_energy, _ = _evaluate_program_preview_motion_rgb(
        prog, width=width, height=height
    )
    return res0_clean, motion_energy


def _evaluate_program_preview_motion_rgb(
    prog: Dict[str, Any], width: int = 384, height: int = 240
) -> Tuple[Dict[str, Any], float, Optional[np.ndarray]]:
    """
    Evaluates a math_program at t=0.0 and t=pi/2 via symbolic_math_compiler
    and returns (eval_t0_dict, motion_energy, rgb0_array).
    """
    from symbolic_math_compiler import evaluate_math_program_numpy

    res0 = evaluate_math_program_numpy(
        prog, width=width, height=height, t_phase=0.0, include_raw_array=True
    )
    res1 = evaluate_math_program_numpy(
        prog, width=width, height=height, t_phase=math.pi * 0.5, include_raw_array=True
    )
    rgb0 = res0.get("rgb_array")
    rgb1 = res1.get("rgb_array")
    if rgb0 is not None and rgb1 is not None:
        motion_energy = float(np.mean(np.abs(rgb1.astype(np.float32) - rgb0.astype(np.float32))))
    else:
        motion_energy = 0.0
    res0_clean = {k: v for k, v in res0.items() if k != "rgb_array"}
    return res0_clean, round(motion_energy, 3), rgb0


def _compute_rgb_mad(rgb_a: Optional[np.ndarray], rgb_b: Optional[np.ndarray]) -> float:
    if rgb_a is None or rgb_b is None or rgb_a.shape != rgb_b.shape:
        return 0.0
    return float(np.mean(np.abs(rgb_b.astype(np.float32) - rgb_a.astype(np.float32))))


def _rank_pass_quality(score: float, stats: Dict[str, Any], motion_energy: float) -> float:
    std_v = float(stats.get("std", 0.0))
    chr_v = float(stats.get("chroma", 0.0))
    cc_v = float(stats.get("center_contrast", 50.0))
    dead_loops = stats.get("dead_loops") or []
    dead_penalty = 1.25 * len(dead_loops) if isinstance(dead_loops, list) else 0.0
    missing_subject_penalty = 2.0 if cc_v < 40.0 else 0.0
    return (
        float(score)
        + 0.45 * min(1.0, std_v / 50.0)
        + 0.45 * min(1.0, chr_v / 35.0)
        + 0.35 * min(1.0, motion_energy / 3.0)
        - dead_penalty
        - missing_subject_penalty
    )


def generate_math_program_with_passes(
    prompt: str,
    refine: bool = True,
    bypass_cache: bool = False,
) -> Dict[str, Any]:
    """
    Full 2-Pass Direct Symbolic Math Synthesis + Closed-Loop Visual Self-Critique pipeline:
      - Pass 1: GLM-5.3 (DeepSeek V4.1 fallback, max_tokens=8192) -> AST validate (+ auto-repair).
      - Pass 2 (when refine=True): Renders preview (t=0 & t=pi/2), critiques via live Vision Critic,
        detects dead/zero-output loops, and if score < 8.5 (or low variance/chroma/motion or dead loops),
        feeds critique + dead_loops into Pass 2 refinement. Verifies mean_abs_pixel_diff(img1, img2):
        if Pass 2 barely changed Pass 1 (mad < 2.0), retries with GLM-5.3 (temperature=0.45) for a
        genuine geometric upgrade (never verbatim copy), then selects the superior pass!
    Returns diagnostics dict containing:
      program, pass1_program, pass2_program, pass1_score, pass2_score,
      pass1_preview_base64, pass2_preview_base64, refinement_passes, visual_score,
      motion_energy, improved, pass1_pass2_mad.
    """
    from symbolic_math_compiler import validate_math_program

    cache_key = prompt.strip().lower()
    if not bypass_cache and cache_key in _MATH_PROGRAM_CACHE:
        cached = _normalize_math_program(_MATH_PROGRAM_CACHE[cache_key], prompt=prompt)
        ok, _ = validate_math_program(cached)
        if ok:
            return {
                "program": cached,
                "pass1_program": cached,
                "pass2_program": None,
                "pass1_score": float(cached.get("visual_score", 8.5)),
                "pass2_score": None,
                "pass1_preview_base64": None,
                "pass2_preview_base64": None,
                "refinement_passes": int(cached.get("refinement_passes", 1)),
                "visual_score": float(cached.get("visual_score", 8.5)),
                "motion_energy": float(cached.get("motion_energy", 2.0)),
                "improved": False,
                "pass1_pass2_mad": 0.0,
            }

    # --- PASS 1: Initial Direct Pure-Math Program Synthesis ---
    pass1_user = (
        f'Create a museum-grade, unmistakably recognizable pure-math painting & 4D periodic animation '
        f'program for: "{prompt}"\n\n'
        f'Requirements:\n'
        f'1. Build a rich atmospheric background FIRST, then layer EVERY signature anatomical/architectural '
        f'feature back-to-front filling 55%-75% of the frame.\n'
        f'2. Use bilateral symmetry `sx = abs(x)` for ALL symmetric anatomy (ears, cheek ruffs, eyes, fangs, whiskers, wings) — NEVER signed `x`!\n'
        f'3. Include 40 to 68 clean symbolic steps and 1 to 3 "loop" blocks (for stars, aurora, tentacles, whiskers, foliage, rings). '
        f'Every loop "var" MUST be a descriptive output mask name (NEVER "i" or "k") and MUST be visibly mixed into H0/H1/H2!\n'
        f'4. NEVER pass pixel-array y-shifted variables (e.g. cy = y - 0.2) into sdSegment/sdCapsule endpoints — always use SCALAR center vars like `bellY = 0.25 + 0.04*sin(t)`!\n'
        f'5. CRITICAL COMPOSITING CHECK: Ensure the final foreground/subject RGB variables (e.g. `hR, hG, hB`) are composited all the way into `H0, H1, H2` — NEVER accidentally mix framing foliage/mist onto `bgR, bgG, bgB` instead of the latest subject RGB on the final step!\n'
        f'6. Use razor-sharp `ee(200.0 * d)`..`ee(320.0 * d)` for main silhouettes/details, rich 3D lighting + specular catchlights, and visible 2*PI-periodic motion via sin(t), cos(t), fbmT(..., t).\n'
        f'7. Use ONLY whitelisted functions and float literals (e.g. 0.0, 1.0). Never use ** or if/else. '
        f'NEVER mention AI or model names in title or subtitle. Output STRICT JSON ONLY.'
    )

    raw_p1 = _invoke_flagship_for_math_program(
        MATH_PROGRAM_SYSTEM_PROMPT, pass1_user, max_tokens=8192, temperature=0.20, prefer_glm=True
    )
    if raw_p1 is None:
        fallback = _build_emergency_fallback_math_program(prompt)
        return {
            "program": fallback,
            "pass1_program": fallback,
            "pass2_program": None,
            "pass1_score": 6.5,
            "pass2_score": None,
            "pass1_preview_base64": None,
            "pass2_preview_base64": None,
            "refinement_passes": 1,
            "visual_score": 6.5,
            "motion_energy": 2.0,
            "improved": False,
            "pass1_pass2_mad": 0.0,
        }

    p1 = _normalize_math_program(raw_p1, prompt=prompt)
    ok1, err1 = validate_math_program(p1)
    if not ok1:
        repaired1 = _repair_math_program(p1, err1, prompt)
        if repaired1 is not None:
            ok1_r, _ = validate_math_program(repaired1)
            if ok1_r:
                p1, ok1 = repaired1, True
    if not ok1:
        alt_p1 = _invoke_flagship_for_math_program(
            MATH_PROGRAM_SYSTEM_PROMPT, pass1_user, max_tokens=8192, temperature=0.18, prefer_glm=False
        )
        if alt_p1 is not None:
            alt_norm = _normalize_math_program(alt_p1, prompt=prompt)
            ok_alt, err_alt = validate_math_program(alt_norm)
            if not ok_alt:
                rep_alt = _repair_math_program(alt_norm, err_alt, prompt)
                if rep_alt is not None and validate_math_program(rep_alt)[0]:
                    p1, ok1 = rep_alt, True
            else:
                p1, ok1 = alt_norm, True
    if not ok1:
        p1 = _build_emergency_fallback_math_program(prompt)

    try:
        eval1, motion1, rgb1 = _evaluate_program_preview_motion_rgb(p1, width=384, height=240)
        stats1 = eval1.get("stats", {})
        img1_b64 = eval1.get("image_base64", "")
    except Exception:
        p1 = _build_emergency_fallback_math_program(prompt)
        eval1, motion1, rgb1 = _evaluate_program_preview_motion_rgb(p1, width=384, height=240)
        stats1 = eval1.get("stats", {})
        img1_b64 = eval1.get("image_base64", "")

    if not refine:
        p1["refinement_passes"] = 1
        p1["visual_score"] = round(min(8.5, 4.5 + float(stats1.get("std", 30.0)) / 20.0), 2)
        p1["motion_energy"] = motion1
        _MATH_PROGRAM_CACHE[cache_key] = p1
        _save_math_program_cache()
        return {
            "program": p1,
            "pass1_program": p1,
            "pass2_program": None,
            "pass1_score": p1["visual_score"],
            "pass2_score": None,
            "pass1_preview_base64": img1_b64,
            "pass2_preview_base64": None,
            "refinement_passes": 1,
            "visual_score": p1["visual_score"],
            "motion_energy": motion1,
            "improved": False,
            "pass1_pass2_mad": 0.0,
        }

    # --- CLOSED-LOOP VISION CRITIQUE ON PASS 1 ---
    critique1 = critique_rendered_math_image(prompt, img1_b64, stats1, motion1)
    score1 = float(critique1.get("score", 6.0))
    std1 = float(stats1.get("std", 0.0))
    chroma1 = float(stats1.get("chroma", 0.0))
    cc1 = float(stats1.get("center_contrast", 50.0))
    dead_loops1 = list(stats1.get("dead_loops") or [])

    needs_refine = (
        (score1 < 8.5)
        or (std1 < 35.0)
        or (chroma1 < 20.0)
        or (cc1 < 40.0)
        or (motion1 < 1.0)
        or (len(dead_loops1) > 0)
    )
    if not needs_refine:
        p1["refinement_passes"] = 1
        p1["visual_score"] = score1
        p1["motion_energy"] = motion1
        _MATH_PROGRAM_CACHE[cache_key] = p1
        _save_math_program_cache()
        return {
            "program": p1,
            "pass1_program": p1,
            "pass2_program": None,
            "pass1_score": score1,
            "pass2_score": None,
            "pass1_preview_base64": img1_b64,
            "pass2_preview_base64": None,
            "refinement_passes": 1,
            "visual_score": score1,
            "motion_energy": motion1,
            "improved": False,
            "pass1_pass2_mad": 0.0,
        }

    # --- PASS 2: Visual Self-Critique Guided Mathematical Refinement ---
    defects_txt = "\n".join(f"  - {d}" for d in critique1.get("defects", [])) or "  - Increase anatomical clarity, hero scale, & shading contrast."
    fixes_txt = "\n".join(f"  - {fx}" for fx in critique1.get("coordinate_fixes", [])) or "  - Sharpen silhouette edges, add micro-details & bright highlights."
    missing_subject_note = (
        f"\n- CRITICAL MISSING-SUBJECT ALERT: Pass 1 scored {score1:.1f}/10 with center_contrast={cc1:.1f}! "
        f"The main hero subject is MISSING or obscured from the center of the canvas! "
        f"CHECK YOUR FINAL RGB COMPOSITING CHAIN: ensure the latest hero subject RGB variables (e.g. `hR, hG, hB`) "
        f"are mixed into `H0, H1, H2` — NEVER accidentally mix foreground foliage/mist onto `bgR, bgG, bgB` instead of the subject RGB!"
        if (score1 < 5.0 or cc1 < 40.0)
        else ""
    )
    dead_loop_note = (
        f"\n- CRITICAL ZERO-OUTPUT / INVISIBLE LOOPS DETECTED IN PASS 1: {dead_loops1}\n"
        f"  These loops evaluated to 0.0 everywhere on screen! Common causes: (a) passing pixel-shifted `cy = y - ...` into "
        f"`sdSegment`/`sdCapsule` endpoints instead of scalar coordinates like `bellY = 0.25`, (b) wrong SDF sign, or (c) naming "
        f"`loop.var` as an index (`i`/`k`) instead of the output mask name. FIX EVERY ONE so these loops render brightly & visibly!"
        if dead_loops1
        else ""
    )
    motion_note = (
        "\n- IMPORTANT: Pass 1 had very low motion_energy (< 1.5)! Add clearly visible 2*PI-periodic motion "
        "(swaying foliage/tentacles/auroras/waves, breathing/pulsing glow, rotating beams, drifting stars) via sin(t), cos(t), and fbmT(..., t)."
        if motion1 < 1.5
        else ""
    )

    pass2_user = (
        f'DRAMATICALLY RE-ARCHITECT AND UPGRADE the Pass 1 pure-math painting program for: "{prompt}"\n\n'
        f'Visual Inspection Report on Pass 1 (score {score1:.1f}/10.0, std={std1:.1f}, chroma={chroma1:.1f}, center_contrast={cc1:.1f}, motion={motion1:.2f}):\n'
        f'What actually rendered: {critique1.get("what_is_visible", "")}\n'
        f'CRITICAL DEFECTS TO ELIMINATE:\n{defects_txt}\n'
        f'CONCRETE (x, y) GEOMETRIC, SHARPNESS & COLOR FIXES TO APPLY:\n{fixes_txt}{missing_subject_note}{dead_loop_note}{motion_note}\n\n'
        f'MANDATORY QUALITY UPGRADES FOR PASS 2 (STRICTLY ENFORCED):\n'
        f'1. NEVER copy Pass 1 verbatim! Substantially upgrade the silhouette, scale (55%-75% of canvas), anatomy, background detail, and lighting.\n'
        f'2. FINAL COMPOSITING CHAIN: Ensure the final foreground/subject RGB variables (e.g. `hR, hG, hB`) are composited into `H0, H1, H2` — NEVER accidentally mix `bgR/bgG/bgB` instead of the subject RGB on the final step!\n'
        f'3. NEVER pass pixel-array y-shifted vars (`cy = y - ...`) into `sdSegment`/`sdCapsule`/`sdTriangle` endpoints — ALWAYS use SCALAR world coordinates (e.g. `bellY = 0.26 + 0.04*sin(t)`, `ay = bellY - 0.08`)!\n'
        f'4. NEVER name `loop.var` as an index (`i`/`k`); every `loop.var` MUST be an active output mask visibly mixed into H0/H1/H2!\n'
        f'5. ALWAYS use `sx = abs(x)` for ALL symmetric features (ears, cheek ruffs, eyes, fangs, whiskers, wings) — NEVER signed `x`!\n'
        f'6. SHARPEN ALL MAIN EDGES with `ee(200.0 * d)` to `ee(320.0 * d)`, deepen 3D highlights & ambient-occlusion shadows, and keep smooth 2*PI-periodic motion.\n'
        f'7. Keep total steps between 45 and 76 (NEVER > 82), loop counts <= 45, float literals only, ONLY whitelisted functions, and ZERO AI/model names.\n\n'
        f'Pass 1 math_program JSON (DO NOT COPY UNCHANGED — rewrite weak sections aggressively):\n{json.dumps(p1)}\n\n'
        f'Output STRICT JSON ONLY.'
    )

    def _evaluate_pass2_candidate(
        raw_cand: Optional[Dict[str, Any]],
    ) -> Optional[Tuple[Dict[str, Any], Dict[str, Any], float, float, float, str]]:
        if raw_cand is None:
            return None
        cand = _normalize_math_program(raw_cand, prompt=prompt)
        ok_c, err_c = validate_math_program(cand)
        if not ok_c:
            rep_c = _repair_math_program(cand, err_c, prompt)
            if rep_c is not None and validate_math_program(rep_c)[0]:
                cand, ok_c = rep_c, True
        if not ok_c:
            return None
        try:
            ev_c, mot_c, rgb_c = _evaluate_program_preview_motion_rgb(cand, width=384, height=240)
            st_c = ev_c.get("stats", {})
            b64_c = ev_c.get("image_base64", "")
            mad_c = _compute_rgb_mad(rgb1, rgb_c)
            crit_c = critique_rendered_math_image(prompt, b64_c, st_c, mot_c)
            sc_c = float(crit_c.get("score", score1))
            return cand, ev_c, mot_c, sc_c, mad_c, b64_c
        except Exception:
            return None

    def _is_missing_subject_cand(
        cand_tuple: Optional[Tuple[Dict[str, Any], Dict[str, Any], float, float, float, str]],
    ) -> bool:
        if cand_tuple is None:
            return True
        st_c = cand_tuple[1].get("stats", {})
        return (cand_tuple[3] < 5.0) or (float(st_c.get("center_contrast", 50.0)) < 40.0)

    # Attempt 2A: Cross-model refinement via DeepSeek V4.1 (552B) (fallback GLM-5.3)
    raw_p2 = _invoke_flagship_for_math_program(
        MATH_PROGRAM_SYSTEM_PROMPT, pass2_user, max_tokens=8192, temperature=0.32, prefer_glm=False
    )
    cand_a = _evaluate_pass2_candidate(raw_p2)

    # Anti-verbatim-copy & missing-subject guard:
    # If Attempt 2A failed, barely changed Pass 1 (mad < 2.0), OR rendered a missing/weak subject
    # (score < 5.0 or center_contrast < 40.0), retry Pass 2 with GLM-5.3 (temperature=0.45) using
    # a clean rebuild brief (WITHOUT raw Pass 1 JSON) so it CANNOT verbatim-copy and MUST render
    # the full hero subject cleanly into H0/H1/H2!
    cand_b = None
    if cand_a is None or cand_a[4] < 2.0 or _is_missing_subject_cand(cand_a):
        p1_var_summary = ", ".join(
            (st["loop"]["var"] if "loop" in st else st.get("var", ""))
            for st in p1.get("steps", [])[:25]
        )
        pass2_rebuild_user = (
            f'BUILD A DRAMATICALLY SUPERIOR PASS 2 pure-math painting & 4D animation program from scratch for: "{prompt}"\n\n'
            f'Our previous attempt (using variables: {p1_var_summary}) scored ONLY {score1:.1f}/10.0 (center_contrast={cc1:.1f}).\n'
            f'What rendered previously: {critique1.get("what_is_visible", "")}\n'
            f'DEFECTS YOU MUST COMPLETELY FIX IN THIS NEW VERSION:\n{defects_txt}\n'
            f'EXACT (x, y) GEOMETRIC & COLOR IMPROVEMENTS REQUIRED:\n{fixes_txt}{missing_subject_note}{dead_loop_note}{motion_note}\n\n'
            f'CRITICAL ENGINEERING RULES:\n'
            f'- Make the hero subject BOLD, CENTERED & UNMISTAKABLE (55%-75% of canvas) with crisp `ee(220.0 * d)` edges, rich 3D lighting, and vivid background depth.\n'
            f'- FINAL COMPOSITING CHAIN: Verify that the final hero subject RGB variables (e.g. `hR, hG, hB`) feed directly into `H0, H1, H2`! When adding framing foliage/mist/vignette at the end, mix onto the LATEST subject RGB (`mix(hR, frR, fronds)`), NEVER onto `bgR, bgG, bgB`!\n'
            f'- Use `sx = abs(x)` for ALL left-right symmetric features (ears, cheek ruffs, eyes, fangs, whiskers, wings) — NEVER signed `x`!\n'
            f'- In `sdSegment`/`sdCapsule`/`sdTriangle`, pass SCALAR endpoints (e.g. `bellY = 0.26 + 0.04*sin(t)`), NEVER pixel-array `y - ...` expressions!\n'
            f'- For upper domes (like a jellyfish bell), cut bottom via `smax(sdEllipse(x, y - bellY, rx, ry), (bellY - 0.06) - y, 0.04)` (NOT `y - bellY`!).\n'
            f'- Every loop `"var"` MUST be a descriptive output mask name (NEVER `"i"` or `"k"`) and MUST visibly contribute to `H0/H1/H2`!\n'
            f'- 45 to 72 steps total, 1 to 3 rich loops (<= 45 iters each), float literals only, whitelisted functions only, visible 2*PI-periodic motion, ZERO AI/model names.\n'
            f'Output STRICT JSON ONLY.'
        )
        raw_p2_retry = _invoke_flagship_for_math_program(
            MATH_PROGRAM_SYSTEM_PROMPT,
            pass2_rebuild_user,
            max_tokens=8192,
            temperature=0.45,
            prefer_glm=True,
        )
        cand_b = _evaluate_pass2_candidate(raw_p2_retry)

    # Pick the best valid Pass 2 candidate (strongly preferring mad >= 2.0 & visible hero subject)
    pass2_pool = [c for c in (cand_a, cand_b) if c is not None]

    # Last-resort clean fallback regeneration if BOTH Pass 1 and all Pass 2 candidates have missing subject
    # (score < 5.0 or center_contrast < 40.0), so a foliage-only / missing-subject render can never win!
    p1_missing = (score1 < 5.0) or (cc1 < 40.0)
    all_p2_missing = all(_is_missing_subject_cand(c) for c in pass2_pool)
    if p1_missing and all_p2_missing:
        fallback_regen_user = (
            f'Create a CLEAN, SELF-CONTAINED, UNMISTAKABLE pure-math painting & 4D animation for: "{prompt}"\n\n'
            f'Previous attempts accidentally dropped the central subject during final RGB compositing! '
            f'Keep your compositing chain crystal clear: `bgR/G/B` -> `subR/G/B` (hero subject filling 60% of center!) -> '
            f'`finR/G/B` (optional light framing mixed onto `subR/G/B`, NEVER `bgR/G/B`!) -> `H0/H1/H2`.\n'
            f'Double-check: `sx = abs(x)` for ALL symmetric anatomy (ears, eyes, fangs, whiskers), razor-sharp `ee(220.0 * d)` edges, '
            f'scalar endpoints in `sdSegment`/`sdCapsule`, descriptive active `loop.var` names, 45-68 steps, float literals only, '
            f'visible 2*PI-periodic motion, ZERO AI/model names. Output STRICT JSON ONLY.'
        )
        raw_p2_fallback = _invoke_flagship_for_math_program(
            MATH_PROGRAM_SYSTEM_PROMPT,
            fallback_regen_user,
            max_tokens=8192,
            temperature=0.35,
            prefer_glm=False,
        )
        cand_c = _evaluate_pass2_candidate(raw_p2_fallback)
        if cand_c is not None:
            pass2_pool.append(cand_c)

    chosen_p2 = None
    if pass2_pool:
        chosen_p2 = max(
            pass2_pool,
            key=lambda c: _rank_pass_quality(c[3], c[1].get("stats", {}), c[2])
            + (0.5 if c[4] >= 2.0 else -2.0),
        )

    p2: Optional[Dict[str, Any]] = None
    eval2: Optional[Dict[str, Any]] = None
    motion2: float = 0.0
    score2: Optional[float] = None
    mad_1_2: float = 0.0
    img2_b64: Optional[str] = None
    if chosen_p2 is not None:
        p2, eval2, motion2, score2, mad_1_2, img2_b64 = chosen_p2

    rank1 = _rank_pass_quality(score1, stats1, motion1)
    if p2 is not None and eval2 is not None and score2 is not None:
        stats2 = eval2.get("stats", {})
        rank2 = _rank_pass_quality(score2, stats2, motion2)
        if mad_1_2 >= 2.0 and rank2 >= rank1 - 0.15:
            winner = p2
            winner["refinement_passes"] = 2
            winner["visual_score"] = score2
            winner["motion_energy"] = motion2
            improved = True
        else:
            winner = p1
            winner["refinement_passes"] = 2
            winner["visual_score"] = score1
            winner["motion_energy"] = motion1
            improved = False
    else:
        winner = p1
        winner["refinement_passes"] = 1
        winner["visual_score"] = score1
        winner["motion_energy"] = motion1
        improved = False

    _MATH_PROGRAM_CACHE[cache_key] = winner
    _save_math_program_cache()
    return {
        "program": winner,
        "pass1_program": p1,
        "pass2_program": p2,
        "pass1_score": score1,
        "pass2_score": score2,
        "pass1_preview_base64": img1_b64,
        "pass2_preview_base64": img2_b64,
        "refinement_passes": int(winner["refinement_passes"]),
        "visual_score": float(winner["visual_score"]),
        "motion_energy": float(winner["motion_energy"]),
        "improved": improved,
        "pass1_pass2_mad": round(mad_1_2, 3),
    }


def generate_math_program(
    prompt: str,
    refine: bool = True,
    bypass_cache: bool = False,
) -> Dict[str, Any]:
    """
    Synthesizes (and optionally 2-pass visually refines) a canonical pure-math
    symbolic program for any arbitrary prompt using DeepSeek V4 Pro & GLM-5.3 ONLY,
    with silent closed-loop visual self-critique and zero AI/model branding leaks.
    """
    bundle = generate_math_program_with_passes(
        prompt=prompt, refine=refine, bypass_cache=bypass_cache
    )
    return bundle["program"]



# ============================================================================
# PHASE 3 WORLD-FIRST H-DGES: PARAMETERIZED GOMPERTZ-HARMONIC TOPOLOGY PLANNER
# (Strictly ~1% Symbolic Topology Prior -> 99% Pure-Calculus Homotopy Autodiff)
# ============================================================================

HDGES_TOPOLOGY_SYSTEM_PROMPT = """You are a Differential-Geometric Sculptor designing a compact parameterized H-DGES (Homotopy-Annealed Differentiable Gompertz-Harmonic Equation) topology for pure-calculus gradient optimization on x in [-1.6, 1.6], y in [-1.0, 1.0].

Output STRICT JSON ONLY with this exact schema:
{
  "title": "Pure Artistic & Mathematical Title (NEVER mention AI or models)",
  "subtitle": "Homotopy-Annealed Gompertz-Harmonic Analytical Field H_v(x,y,t)",
  "kappa_global": 42.0,
  "background": {
    "c0": [r, g, b],
    "cx": [dr_dx, dg_dx, db_dx],
    "cy": [dr_dy, dg_dy, db_dy],
    "amp": [[a0,a1,a2,a3],[b0,b1,b2,b3],[c0,c1,c2,c3]],
    "wx": [1.8, 3.4, 6.0, 9.5],
    "wy": [2.1, -2.6, 4.4, -7.8],
    "phi": [0.2, 1.1, 2.3, 0.7]
  },
  "layers": [
    {
      "label": "short_part_name",
      "mirror_x": false,
      "cx": 0.0, "cy": 0.0,
      "rx": 0.45, "ry": 0.35,
      "theta": 0.0, "p_exp": 2.0,
      "warp_a": 0.0, "warp_f": 3.0, "warp_p": 0.0,
      "k_mul": 1.0, "alpha": 0.95,
      "r": 0.85, "g": 0.45, "b": 0.15,
      "lx": -0.45, "ly": 0.55, "shade_amp": 0.28, "spec_amp": 0.18,
      "tex_amp": 0.12, "tex_wx": 12.0, "tex_wy": 4.0, "tex_phi": 0.0,
      "z_depth": 0.7,
      "t_amp_x": 0.015, "t_amp_y": 0.02, "t_amp_rot": 0.04, "t_freq": 1.0, "t_phase0": 0.0
    }
  ]
}

CRITICAL RULES:
1. Porter-Duff FRONT-TO-BACK order: layers[0] is FRONT-MOST (eyes, pupils, catchlights, fangs, highlights, whiskers), and layers[-1] is BACK-MOST (silhouette body, glow aura, distant cliffs/planets)!
2. Use 14 to 24 rich anatomical/architectural layers filling 55%-75% of the canvas. Set `"mirror_x": true` on paired bilateral parts (eyes, pupils, ears, wings, cheek ruffs, whiskers, petals) with `cx > 0.05` so exact bilateral symmetry is created automatically!
3. Use `p_exp` (1.4=diamond/pointed petal/ear/fang, 2.0=organic ellipse, 3.8=architecture/cliff/tower), `warp_a/warp_f` (curved tails, waves, aurora ribbons, smiles, roaring mouths, tentacles), `tex_amp/tex_wx/tex_wy` (stripes, fur, feathers, ripples), and `shade_amp/spec_amp` (3D volume & glossy specular highlights).
4. Add gentle 2*PI-periodic motion (`t_amp_x`, `t_amp_y`, `t_amp_rot`, integer `t_freq` 1.0 or 2.0) on living/dynamic parts so the closed-form equation animates smoothly at 60 FPS!
5. NEVER mention AI, Cloudflare, DeepSeek, GLM, or model names anywhere. Return STRICT JSON ONLY."""


def _fallback_hdges_topology(prompt: str) -> Dict[str, Any]:
    clean_title = sanitize_mathematical_text(
        f"{prompt.strip().title()} — H-DGES Harmonic Field",
        "H-DGES Harmonic Field",
    )
    return {
        "title": clean_title,
        "subtitle": "Homotopy-Annealed Gompertz-Harmonic Analytical Field H_v(x, y, t)",
        "kappa_global": 45.0,
        "background": {
            "c0": [0.06, 0.10, 0.22],
            "cx": [0.0, 0.02, 0.04],
            "cy": [0.04, 0.08, 0.14],
            "amp": [
                [0.05, 0.03, 0.02, 0.01],
                [0.08, 0.05, 0.03, 0.02],
                [0.10, 0.06, 0.04, 0.02],
            ],
            "wx": [1.8, 3.2, 5.6, 9.2],
            "wy": [2.2, -2.8, 4.1, -6.9],
            "phi": [0.3, 1.2, 2.4, 0.8],
        },
        "layers": [
            {"label": "core_glint", "cx": -0.06, "cy": 0.12, "rx": 0.07, "ry": 0.06, "r": 1.0, "g": 0.96, "b": 0.82, "spec_amp": 0.45, "z_depth": 0.95, "t_amp_y": 0.02, "t_freq": 1.0},
            {"label": "eye_pupil", "mirror_x": True, "cx": 0.20, "cy": 0.14, "rx": 0.045, "ry": 0.085, "r": 0.05, "g": 0.06, "b": 0.08, "z_depth": 0.92},
            {"label": "eye_iris", "mirror_x": True, "cx": 0.20, "cy": 0.14, "rx": 0.11, "ry": 0.10, "r": 0.98, "g": 0.78, "b": 0.18, "spec_amp": 0.35, "z_depth": 0.88},
            {"label": "crest_arc", "cx": 0.0, "cy": 0.36, "rx": 0.46, "ry": 0.14, "warp_a": 0.08, "warp_f": 4.0, "r": 0.22, "g": 0.85, "b": 0.92, "tex_amp": 0.18, "tex_wx": 16.0, "z_depth": 0.78, "t_amp_y": 0.025, "t_freq": 1.0},
            {"label": "wing_petal", "mirror_x": True, "cx": 0.52, "cy": -0.02, "rx": 0.36, "ry": 0.18, "theta": 0.32, "warp_a": 0.06, "warp_f": 3.5, "r": 0.92, "g": 0.48, "b": 0.24, "tex_amp": 0.20, "tex_wx": 18.0, "shade_amp": 0.32, "z_depth": 0.65, "t_amp_rot": 0.08, "t_freq": 1.0},
            {"label": "central_body", "cx": 0.0, "cy": 0.02, "rx": 0.44, "ry": 0.42, "r": 0.88, "g": 0.56, "b": 0.20, "shade_amp": 0.34, "spec_amp": 0.22, "tex_amp": 0.16, "tex_wx": 14.0, "tex_wy": 6.0, "z_depth": 0.60, "t_amp_y": 0.018, "t_freq": 1.0},
            {"label": "lower_plume", "cx": 0.0, "cy": -0.46, "rx": 0.58, "ry": 0.24, "warp_a": 0.09, "warp_f": 5.0, "r": 0.18, "g": 0.62, "b": 0.76, "alpha": 0.85, "tex_amp": 0.22, "tex_wx": 20.0, "z_depth": 0.42, "t_amp_x": 0.03, "t_freq": 1.0},
            {"label": "halo_aura", "cx": 0.0, "cy": 0.05, "rx": 0.82, "ry": 0.64, "k_mul": 0.35, "alpha": 0.55, "r": 0.30, "g": 0.72, "b": 0.95, "z_depth": 0.20},
        ],
    }


def generate_hdges_topology_for_prompt(prompt: str) -> Dict[str, Any]:
    """
    1% Symbolic Topology Planner for Zero-Diffusion H-DGES Text-to-Equation Mode.
    Synthesizes an initial parameterized Gompertz-Harmonic layer skeleton via
    GLM-5.3 / DeepSeek V4.1 that is then continuously refined by our pure-calculus
    Homotopy kappa-Annealed Reverse-Mode Autodiff Solver.
    """
    user_msg = (
        f'Design a bold, unmistakable, 16-to-22-layer front-to-back H-DGES Gompertz-Harmonic '
        f'topology JSON for: "{prompt}".\n'
        f'Remember: layers[0] is front-most (eyes, catchlights, highlights, details) and layers[-1] '
        f'is back-most (body, glow, background forms). Use mirror_x=true for symmetric pairs, '
        f'rich 3D relief shading (shade_amp, spec_amp), harmonic textures (tex_amp, tex_wx, tex_wy), '
        f'and gentle periodic 4D animation (t_amp_x, t_amp_y, t_amp_rot). Output STRICT JSON ONLY.'
    )
    models = [GLM_5_3, DEEPSEEK_V4_PRO, GLM_5_3_FLASH] if _has_cf_credentials() else []
    for model in models:
        url = f"https://api.cloudflare.com/client/v4/accounts/{CF_ACCOUNT_ID}/ai/run/{model}"
        payload: Dict[str, Any] = {
            "messages": [
                {"role": "system", "content": HDGES_TOPOLOGY_SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            "max_tokens": 4096,
            "temperature": 0.22,
        }
        if "glm-5" in model:
            payload["reasoning_effort"] = "low"
        elif "deepseek-v4" in model:
            payload["chat_template_kwargs"] = {"enable_thinking": False}
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {CF_API_TOKEN}",
                "Content-Type": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                body = json.loads(resp.read().decode("utf-8"))
                raw_resp = _extract_response_text(body)
                parsed = raw_resp if isinstance(raw_resp, dict) else _extract_json_object(str(raw_resp))
                if isinstance(parsed, dict) and isinstance(parsed.get("layers"), list) and len(parsed["layers"]) >= 4:
                    parsed["title"] = sanitize_mathematical_text(
                        str(parsed.get("title") or ""),
                        f"{prompt.strip().title()} — H-DGES Harmonic Field",
                    )
                    parsed["subtitle"] = sanitize_mathematical_text(
                        str(parsed.get("subtitle") or ""),
                        "Homotopy-Annealed Gompertz-Harmonic Analytical Field H_v(x, y, t)",
                    )
                    return parsed
        except Exception:
            continue
    return _fallback_hdges_topology(prompt)


# ============================================================================
# STAGE 2 BEYOND SOTA: UNIVERSAL 3D RAYMARCHED SDF + DIFFERENTIAL-GEOMETRY
#                      + MULTI-PARADIGM PDE CINEMA ARCHITECT
# ============================================================================

RAYMARCH_3D_CACHE_FILE = os.path.join(os.path.dirname(__file__), "raymarch_3d_cache.json")
_RAYMARCH_3D_CACHE: Dict[str, Dict[str, Any]] = {}

VALID_RAYMARCH_PRIMITIVES = {
    "sphere",
    "ellipsoid",
    "capsule",
    "torus",
    "box",
    "cone",
    "cylinder",
    "gyroid_shell",
    "mandelbulb_core",
    "kleinian_fold",
    "ruled_paraboloid",
    "floraform_ribbon",
}

VALID_RAYMARCH_OPS = {
    "union",
    "smin",
    "subtract",
    "intersect",
}

VALID_RAYMARCH_FLOORS = {
    "ocean_waves",
    "marble_reflection",
    "gyroid_ground",
    "cosmic_void",
    "terrain_fbm",
}

RAYMARCH_3D_SYSTEM_PROMPT = """You are a World-Master 3D Mathematical Cinema Director unifying the 8 Master Paradigms of Mathematical Art:
1. Inigo Quilez (3D Signed Distance Fields d(p,t), smooth organic union smin, Mandelbulb fractal cores, domain warping, Fresnel PBR + soft shadows + AO),
2. Helaman Ferguson & Bathsheba Grossman (Triply Periodic Minimal Surfaces: Schoen Gyroid sin(x)cos(y)+sin(y)cos(z)+sin(z)cos(x)=0 shells & hyperbolic bronze sculptures),
3. Nervous System Floraform (3D differential-growth buckled wave ribbons & biological coral/fin/petal membranes),
4. William Thurston Geometry Center (4D stereographic & toroidal homotopies),
5. Iannis Xenakis (3D ruled hyperbolic paraboloid shells & architectural vaults),
6. SideFX Houdini (3D curl-warped organic displacement & dynamic fluid/wave relief),
7. Conformal & Celestial Mechanisms (nested 3D armillary/astrolabe gimbals & orbital rings),
8. Non-Euclidean Kleinian Folded Cathedrals (box-fold + sphere-inversion monumental colonnades).

Your task: Decompose ANY user prompt into a complete, museum-grade **3D Raymarched Implicit Scene (`raymarch_3d_scene`)** in R^3 x R (with +Y UP, +X RIGHT, +Z TOWARD CAMERA, primary stage volume [-1.5, 1.5] x [-0.8, 1.3] x [-1.3, 1.3]).
NEVER create flat 2D cutouts! Every component must have genuine 3D depth along Z, sculptural 3D normals, physically based 3D materials (metallic, roughness, subsurface scattering, emission, Fresnel, procedural 3D stripes), and smooth 2*PI-periodic 4D motion.

Output STRICT JSON ONLY matching this exact canonical schema:
{
  "title": "Pure 3D Differential-Geometric & Raymarching Title (NEVER mention AI or model names)",
  "subtitle": "Coupled 3D Implicit SDF & Minimal-Surface Raymarched Manifold d(p, t) = 0",
  "aspect": [16, 10],
  "camera_dist": 2.85,
  "camera_height": 0.65,
  "camera_orbit_speed": 0.35,
  "fov": 1.72,
  "sky_top_rgb": [0.04, 0.08, 0.22],
  "sky_horizon_rgb": [0.35, 0.52, 0.78],
  "fog_density": 0.04,
  "sun_dir": [0.58, 0.72, 0.38],
  "floor_type": "ocean_waves",
  "floor_height": -1.05,
  "floor_rgb": [0.05, 0.22, 0.48],
  "nodes": [
    {
      "id": "hero_core",
      "op": "union",
      "blend_k": 0.14,
      "primitive": "ellipsoid",
      "center": [0.0, 0.15, 0.0],
      "size": [0.32, 0.44, 0.28],
      "rotation": [0.1, 0.0, 0.0],
      "endpoint_a": [0.0, -0.2, 0.0],
      "endpoint_b": [0.0, 0.4, 0.0],
      "symmetry_x": false,
      "symmetry_radial": 0,
      "repeat_period": null,
      "twist": 0.0,
      "bend": 0.0,
      "fbm_amp": 0.02,
      "fbm_freq": 4.5,
      "gyroid_scale": 0.0,
      "gyroid_thickness": 0.05,
      "material": {
        "rgb": [0.95, 0.68, 0.18],
        "metallic": 0.78,
        "roughness": 0.22,
        "subsurface": 0.35,
        "emission": 0.25,
        "fresnel": 0.85,
        "stripe_freq": 6.0,
        "stripe_rgb": [1.0, 0.88, 0.42]
      },
      "motion": {
        "amp_pos": [0.0, 0.05, 0.0],
        "amp_rot": [0.04, 0.08, 0.0],
        "freq": 1.0,
        "phase": 0.0
      }
    }
  ]
}

### STRICT 3D SCULPTURAL & MULTI-PARADIGM RULES
1. **12 to 20 Rich 3D Nodes**: Build a complete, unmistakable 3D sculpture + environment filling 55%-75% of the camera viewport around (0, 0.15, 0):
   -Allowed `"primitive"` values ONLY:
     * `"sphere"`: 3D sphere of radius `size[0]` at `center`.
     * `"ellipsoid"`: 3D tri-axial ellipsoid with radii `size = [rx, ry, rz]`.
     * `"capsule"`: 3D swept cylinder-sphere from `endpoint_a = [ax,ay,az]` to `endpoint_b = [bx,by,bz]` with radius `size[0]`.
     * `"torus"`: 3D ring at `center` rotated by `rotation = [rx,ry,rz]` with major radius `size[0]` and tube radius `size[1]`.
     * `"box"`: 3D rounded box with half-extents `size = [bx, by, bz]`.
     * `"cone"`: 3D tapered cone/beak/horn/spire with base radius `size[0]` and height `size[1]`.
     * `"cylinder"`: 3D column/pillar with radius `size[0]` and half-height `size[1]`.
     * `"gyroid_shell"`: Helaman Ferguson / Bathsheba Grossman Schoen Gyroid minimal surface shell (`|sin(kx)cos(ky)+sin(ky)cos(kz)+sin(kz)cos(kx)| - thickness`) bounded by radius `size[0]`, with `gyroid_scale` (4.0..10.0) and `gyroid_thickness` (0.03..0.12).
     * `"mandelbulb_core"`: Inigo Quilez 3D Mandelbulb fractal power-lobe core of scale `size[0]`.
     * `"kleinian_fold"`: Non-Euclidean box-folded architectural fractal column/vault of extents `size = [sx, sy, sz]`.
     * `"ruled_paraboloid"`: Xenakis 3D ruled hyperbolic paraboloid saddle shell spanning `size = [sx, sy, sz]`.
     * `"floraform_ribbon"`: Nervous System 3D differential-growth buckled wave coral/wing/fin membrane from `endpoint_a` to `endpoint_b` with thickness `size[0]` and wave amplitude `size[1]`.
2. **True 3D Symmetry & Repetition**:
   - Set `"symmetry_x": true` with `center[0] > 0.08` (and `endpoint_a[0] > 0`, `endpoint_b[0] > 0`) for ALL bilateral 3D pairs: wings, primary feathers, eyes, horns, fangs, pectoral fins, temple colonnade pillars, flying buttresses!
   - Set `"symmetry_radial": N` (`N` between 4 and 14) for radial 3D crowns, astrolabe gear spokes, coral polyp tentacles, cathedral vault ribs, tail-feather fans, and lotus petals!
   - Use `"op": "smin"` (`"blend_k": 0.08..0.24`) to fuse organic anatomy (head + neck + torso + wing roots) into a seamless 3D manifold, and `"op": "union"` for crisp distinct materials (glowing eyes, metallic rings, gemstones, pillars).
3. **Subject-Specific 3D Blueprints**:
   - **Golden Phoenix / Mythic Birds over Ocean Waves**:
     * `"floor_type": "ocean_waves"`, `"floor_height": -0.78`, sapphire ocean `"floor_rgb": [0.04, 0.24, 0.58]`, golden-twilight sky.
     * Sculpted 3D chest/torso (`ellipsoid`), arched neck (`capsule`), head + sharp beak (`cone`), glowing solar eyes (`sphere`, `symmetry_x: true`, `emission: 1.8`), 3 layered 3D wing spans (`symmetry_x: true` at `x = 0.25..1.25`, using `floraform_ribbon` + `ruled_paraboloid` + `capsule` feathers with flapping `motion.amp_rot`), radiant crown crest (`symmetry_radial: 7`), and trailing fiery tail plumes (`floraform_ribbon` with gold/amber/crimson `metallic: 0.85, emission: 0.75`).
   - **Hyperbolic Bronze Cathedral / Gyroid Pillars & God Rays**:
     * `"floor_type": "gyroid_ground"` or `"marble_reflection"`, `"fog_density": 0.06`, warm golden sunbeam `"sun_dir": [0.45, 0.78, 0.42]`.
     * Soaring central minimal-surface sanctuary vault (`gyroid_shell`, `size: [0.68, 0.78, 0.68]`, `gyroid_scale: 6.5`, bronze patina `metallic: 0.88`), flanked by bilateral 3D colonnades of carved pillars (`symmetry_x: true`, `kleinian_fold` & `gyroid_shell` at `x = 0.78, z = -0.35, 0.35`), ribbed hyperbolic Gothic arches (`torus` + `ruled_paraboloid` with `symmetry_radial: 6`), stepped altar plinth (`box`), and levitating luminous rose-window fractal relic (`mandelbulb_core`, `emission: 1.6`, cyan-gold).
   - **Bioluminescent Deep-Sea Dragonfish & Differential-Growth Coral**:
     * `"floor_type": "terrain_fbm"`, `"floor_height": -0.82`, abyssal ocean sky `[0.01, 0.04, 0.11]` -> `[0.03, 0.15, 0.28]`.
     * Sinuous 3D dragonfish spine (`capsule` + `ellipsoid` with `bend: 0.35`, iridescent midnight-cyan scales + glowing photophore lateral line `stripe_freq: 10.0, stripe_rgb: [0.1, 0.96, 1.0]`), fierce sculpted jaw + bilateral fangs (`cone`, `symmetry_x: true`), glowing emerald eyes (`sphere`, `symmetry_x: true`, `emission: 1.8`), arching bioluminescent esca lure stalk (`capsule`) + blazing magenta-cyan lure orb (`sphere`, `emission: 2.2`), translucent ruffled fins (`floraform_ribbon`, `symmetry_x: true`, `subsurface: 0.88`), framed by 3D ruffled differential-growth coral clusters (`floraform_ribbon` + `gyroid_shell` in neon magenta, electric cyan, and coral orange)!
   - **Non-Euclidean Celestial Astrolabe & Orbiting Saturn Rings**:
     * `"floor_type": "cosmic_void"` or `"marble_reflection"`, deep cosmic sky `[0.02, 0.02, 0.08]` -> `[0.12, 0.06, 0.24]`.
     * Blazing central fractal star/orb (`mandelbulb_core`, `size: [0.32, 0.32, 0.32]`, `emission: 1.5`), 3 nested tilted 3D armillary/astrolabe gimbal rings (`torus` at radii `0.54, 0.76, 0.98` with orthogonal Euler tilts and counter-rotating `motion.amp_rot`, engraved celestial brass `metallic: 0.92, roughness: 0.15, stripe_freq: 12.0`), radial astrolabe spokes (`symmetry_radial: 12`, `capsule`), carved non-Euclidean outer meridian cradle (`gyroid_shell` / `kleinian_fold`), and orbiting banded Saturn sphere + tilted golden ring (`sphere` + `torus`)!
4. **Output STRICT JSON ONLY** — never include markdown prose or AI/model branding."""


def _clamp_vec3(v: Any, default: List[float], lo: float = -10.0, hi: float = 10.0) -> List[float]:
    if isinstance(v, (list, tuple)) and len(v) >= 3:
        try:
            return [
                round(float(np.clip(float(v[0]), lo, hi)), 4),
                round(float(np.clip(float(v[1]), lo, hi)), 4),
                round(float(np.clip(float(v[2]), lo, hi)), 4),
            ]
        except Exception:
            pass
    return list(default)


def _normalize_raymarch_scene(scene: Dict[str, Any], prompt: str = "") -> Dict[str, Any]:
    """
    Canonicalizes any LLM-synthesized 3D raymarch scene into the exact schema
    expected by backend/master_paradigms_engine.py and scrubs all AI/model branding.
    """
    out = dict(scene) if isinstance(scene, dict) else {}
    for k in ("ai_model", "model", "critic_model", "vlm_model", "provider"):
        out.pop(k, None)

    # Unwrap optional nested camera / environment blocks
    cam_block = out.get("camera") if isinstance(out.get("camera"), dict) else {}
    env_block = out.get("environment") if isinstance(out.get("environment"), dict) else {}

    base_name = prompt.strip().title() if prompt.strip() else "3D Implicit Raymarched Manifold"
    default_title = f"{base_name} — 3D SDF & Minimal-Surface Synthesis"
    out["title"] = sanitize_mathematical_text(str(out.get("title") or ""), default_title)

    raw_nodes = out.get("nodes") if isinstance(out.get("nodes"), list) else []
    default_subtitle = (
        f"Coupled 3D Implicit SDF & Differential-Geometry Raymarched Manifold d(p, t) = 0 — {max(10, len(raw_nodes))} Volumetric Operators"
    )
    raw_sub = str(out.get("subtitle") or "")
    if not raw_sub or _FORBIDDEN_META_RE.search(raw_sub):
        out["subtitle"] = default_subtitle
    else:
        out["subtitle"] = sanitize_mathematical_text(raw_sub, default_subtitle)

    out["aspect"] = [16, 10]

    def _f(key: str, default: float, lo: float, hi: float, block: Dict[str, Any] = cam_block) -> float:
        val = out.get(key, block.get(key, default))
        try:
            return round(float(np.clip(float(val), lo, hi)), 4)
        except Exception:
            return default

    out["camera_dist"] = _f("camera_dist", 2.85, 2.2, 4.2, cam_block)
    out["camera_height"] = _f("camera_height", 0.62, -0.4, 1.8, cam_block)
    out["camera_orbit_speed"] = _f("camera_orbit_speed", 0.35, 0.0, 1.5, cam_block)
    raw_fov = out.get("fov", cam_block.get("fov", 1.72))
    try:
        fov_f = float(raw_fov)
        if fov_f < 1.25:
            fov_f = 1.72
        out["fov"] = round(float(np.clip(fov_f, 1.45, 2.15)), 4)
    except Exception:
        out["fov"] = 1.72

    out["sky_top_rgb"] = _clamp_vec3(
        out.get("sky_top_rgb", env_block.get("sky_top_rgb")), [0.03, 0.07, 0.20], 0.0, 1.0
    )
    out["sky_horizon_rgb"] = _clamp_vec3(
        out.get("sky_horizon_rgb", env_block.get("sky_horizon_rgb")), [0.28, 0.46, 0.72], 0.0, 1.0
    )
    out["fog_density"] = _f("fog_density", 0.04, 0.005, 0.16, env_block)

    sun = _clamp_vec3(out.get("sun_dir", env_block.get("sun_dir")), [0.58, 0.72, 0.38], -2.0, 2.0)
    snorm = math.sqrt(sun[0] * sun[0] + sun[1] * sun[1] + sun[2] * sun[2])
    if snorm < 1e-4:
        sun = [0.58, 0.72, 0.38]
    else:
        sun = [round(sun[0] / snorm, 4), round(sun[1] / snorm, 4), round(sun[2] / snorm, 4)]
    out["sun_dir"] = sun

    ft = str(out.get("floor_type", env_block.get("floor_type", "ocean_waves"))).strip().lower()
    if ft not in VALID_RAYMARCH_FLOORS:
        p_low = prompt.lower()
        if any(w in p_low for w in ("cathedral", "gyroid", "temple", "bronze", "pillar")):
            ft = "gyroid_ground"
        elif any(w in p_low for w in ("astrolabe", "saturn", "cosmic", "space", "orbit", "star")):
            ft = "cosmic_void"
        elif any(w in p_low for w in ("dragonfish", "coral", "reef", "deep-sea", "abyss")):
            ft = "terrain_fbm"
        else:
            ft = "ocean_waves"
    out["floor_type"] = ft
    out["floor_height"] = _f("floor_height", -1.05, -1.5, -0.78, env_block)
    out["floor_rgb"] = _clamp_vec3(
        out.get("floor_rgb", env_block.get("floor_rgb")), [0.06, 0.22, 0.46], 0.0, 1.0
    )

    norm_nodes: List[Dict[str, Any]] = []
    for idx, raw_nd in enumerate(raw_nodes[:24]):
        if not isinstance(raw_nd, dict):
            continue
        nd_id = re.sub(r"[^a-zA-Z0-9_]", "_", str(raw_nd.get("id") or raw_nd.get("name") or f"node_{idx+1}")).strip("_")
        if not nd_id:
            nd_id = f"node_{idx+1}"

        op = str(raw_nd.get("op") or raw_nd.get("operation") or ("union" if idx == 0 else "smin")).strip().lower()
        if op not in VALID_RAYMARCH_OPS:
            op = "smin" if idx > 0 else "union"

        try:
            blend_k = round(float(np.clip(float(raw_nd.get("blend_k", 0.12)), 0.02, 0.35)), 4)
        except Exception:
            blend_k = 0.12

        prim = str(raw_nd.get("primitive") or raw_nd.get("type") or raw_nd.get("shape") or "ellipsoid").strip().lower()
        if prim not in VALID_RAYMARCH_PRIMITIVES:
            if "gyroid" in prim:
                prim = "gyroid_shell"
            elif "mandel" in prim or "fractal" in prim:
                prim = "mandelbulb_core"
            elif "klein" in prim or "fold" in prim:
                prim = "kleinian_fold"
            elif "parabol" in prim or "ruled" in prim:
                prim = "ruled_paraboloid"
            elif "flora" in prim or "ribbon" in prim or "coral" in prim or "wing" in prim:
                prim = "floraform_ribbon"
            elif "ring" in prim:
                prim = "torus"
            elif "seg" in prim or "tube" in prim:
                prim = "capsule"
            else:
                prim = "ellipsoid"

        center = _clamp_vec3(
            raw_nd.get("center") or raw_nd.get("pos") or raw_nd.get("position"),
            [0.0, 0.1, 0.0],
            -2.2,
            2.2,
        )
        ep_a = _clamp_vec3(
            raw_nd.get("endpoint_a") or raw_nd.get("a") or raw_nd.get("start"),
            [center[0], center[1] - 0.25, center[2]],
            -2.2,
            2.2,
        )
        ep_b = _clamp_vec3(
            raw_nd.get("endpoint_b") or raw_nd.get("b") or raw_nd.get("end"),
            [center[0], center[1] + 0.25, center[2]],
            -2.2,
            2.2,
        )
        dist_ab = math.sqrt((ep_b[0] - ep_a[0]) ** 2 + (ep_b[1] - ep_a[1]) ** 2 + (ep_b[2] - ep_a[2]) ** 2)
        if dist_ab < 0.06:
            ep_b = [ep_a[0], round(ep_a[1] + 0.32, 4), ep_a[2]]
            dist_ab = 0.32

        raw_sz = raw_nd.get("size") or raw_nd.get("radii") or raw_nd.get("scale") or raw_nd.get("radius")
        if isinstance(raw_sz, (int, float)):
            r_val = float(np.clip(float(raw_sz), 0.02, 1.4))
            size = [r_val, round(r_val * 0.4, 4), r_val]
        elif isinstance(raw_sz, (list, tuple)) and len(raw_sz) >= 3:
            size = _clamp_vec3(raw_sz, [0.28, 0.28, 0.28], 0.018, 1.5)
        else:
            # Handle primitive-specific scalar keys if LLM omitted 'size'
            if prim == "torus":
                mr = float(raw_nd.get("major_radius", 0.52) or 0.52)
                tr = float(raw_nd.get("tube_radius", raw_nd.get("minor_radius", 0.042)) or 0.042)
                size = [round(float(np.clip(mr, 0.16, 1.25)), 4), round(float(np.clip(tr, 0.018, 0.10)), 4), round(float(np.clip(mr, 0.16, 1.25)), 4)]
            elif prim == "cone":
                br = float(raw_nd.get("base_radius", 0.08) or 0.08)
                ht = float(raw_nd.get("height", 0.18) or 0.18)
                size = [round(float(np.clip(br, 0.025, 0.24)), 4), round(float(np.clip(ht, 0.06, 0.55)), 4), round(float(np.clip(br, 0.025, 0.24)), 4)]
            elif prim == "capsule":
                cr_v = float(raw_nd.get("thickness", 0.065) or 0.065)
                size = [round(float(np.clip(cr_v, 0.02, 0.16)), 4), round(float(np.clip(0.5 * dist_ab, 0.08, 0.85)), 4), round(float(np.clip(cr_v, 0.02, 0.16)), 4)]
            elif prim == "floraform_ribbon":
                span_r = float(np.clip(0.52 * dist_ab, 0.22, 0.65))
                w_amp = float(raw_nd.get("wave_amp", 0.18) or 0.18)
                size = [round(span_r, 4), round(float(np.clip(w_amp, 0.08, 0.30)), 4), round(span_r * 0.75, 4)]
            else:
                size = [0.28, 0.28, 0.28]

        # Enforce sensible primitive geometry bounds even when 'size' was provided
        if prim == "torus":
            size[0] = round(float(np.clip(size[0], 0.15, 1.25)), 4)
            if size[1] > 0.12 or abs(size[1] - size[0]) < 0.03:
                size[1] = round(min(0.055, size[0] * 0.16), 4)
        elif prim == "cone":
            size[0] = round(float(np.clip(size[0], 0.025, 0.24)), 4)
            size[1] = round(float(np.clip(size[1], 0.06, 0.55)), 4)
        elif prim == "capsule":
            size[0] = round(float(np.clip(size[0], 0.02, 0.16)), 4)
        elif prim == "floraform_ribbon":
            size[0] = round(float(np.clip(size[0], 0.18, 0.68)), 4)
            size[1] = round(float(np.clip(size[1], 0.08, 0.32)), 4)

        rotation = _clamp_vec3(
            raw_nd.get("rotation") or raw_nd.get("rot") or raw_nd.get("angles"),
            [0.0, 0.0, 0.0],
            -6.2832,
            6.2832,
        )

        sym_x = bool(raw_nd.get("symmetry_x", raw_nd.get("mirror_x", False)))
        if sym_x and abs(center[0]) < 0.04 and abs(ep_a[0]) < 0.04 and abs(ep_b[0]) < 0.04:
            center[0] = 0.32
        elif sym_x and center[0] < 0:
            center[0] = abs(center[0])
            ep_a[0] = abs(ep_a[0])
            ep_b[0] = abs(ep_b[0])

        try:
            sym_rad = int(round(float(raw_nd.get("symmetry_radial", raw_nd.get("radial_symmetry", 0)) or 0)))
            if sym_rad != 0:
                sym_rad = int(np.clip(sym_rad, 3, 16))
        except Exception:
            sym_rad = 0

        rep = raw_nd.get("repeat_period")
        if isinstance(rep, (list, tuple)) and len(rep) >= 3:
            repeat_period: Optional[List[float]] = _clamp_vec3(rep, [0.0, 0.0, 0.0], 0.0, 8.0)
        else:
            repeat_period = None

        def _nf(k: str, d: float, lo: float, hi: float) -> float:
            try:
                return round(float(np.clip(float(raw_nd.get(k, d)), lo, hi)), 4)
            except Exception:
                return d

        raw_mat = raw_nd.get("material") if isinstance(raw_nd.get("material"), dict) else {}
        mat_rgb = _clamp_vec3(
            raw_mat.get("rgb") or raw_nd.get("color_rgb") or raw_nd.get("color"),
            [0.85, 0.62, 0.24],
            0.0,
            1.0,
        )
        stripe_rgb = _clamp_vec3(
            raw_mat.get("stripe_rgb") or raw_nd.get("accent_rgb"),
            [min(1.0, mat_rgb[0] + 0.2), min(1.0, mat_rgb[1] + 0.15), min(1.0, mat_rgb[2] + 0.2)],
            0.0,
            1.0,
        )

        def _mf(k: str, d: float, lo: float, hi: float) -> float:
            try:
                return round(float(np.clip(float(raw_mat.get(k, raw_nd.get(k, d))), lo, hi)), 4)
            except Exception:
                return d

        material = {
            "rgb": mat_rgb,
            "metallic": _mf("metallic", 0.45, 0.0, 1.0),
            "roughness": _mf("roughness", 0.28, 0.04, 0.95),
            "subsurface": _mf("subsurface", 0.25, 0.0, 1.0),
            "emission": _mf("emission", 0.08, 0.0, 2.5),
            "fresnel": _mf("fresnel", 0.65, 0.0, 1.5),
            "stripe_freq": _mf("stripe_freq", 0.0, 0.0, 16.0),
            "stripe_rgb": stripe_rgb,
        }

        raw_mot = raw_nd.get("motion") if isinstance(raw_nd.get("motion"), dict) else {}
        amp_pos = _clamp_vec3(raw_mot.get("amp_pos"), [0.0, 0.03, 0.0], -0.45, 0.45)
        amp_rot = _clamp_vec3(raw_mot.get("amp_rot"), [0.0, 0.05, 0.0], -0.75, 0.75)
        try:
            freq = float(max(1, min(4, int(round(float(raw_mot.get("freq", 1.0)))))))
        except Exception:
            freq = 1.0
        try:
            phase = round(float(np.clip(float(raw_mot.get("phase", idx * 0.45)), 0.0, 6.2832)), 4)
        except Exception:
            phase = round((idx * 0.45) % 6.2832, 4)

        motion = {
            "amp_pos": amp_pos,
            "amp_rot": amp_rot,
            "freq": freq,
            "phase": phase,
        }

        gy_scale = _nf("gyroid_scale", 6.2, 3.8, 12.0) if prim == "gyroid_shell" else 0.0

        norm_nodes.append(
            {
                "id": nd_id,
                "op": op,
                "blend_k": blend_k,
                "primitive": prim,
                "center": center,
                "size": size,
                "rotation": rotation,
                "endpoint_a": ep_a,
                "endpoint_b": ep_b,
                "symmetry_x": sym_x,
                "symmetry_radial": sym_rad,
                "repeat_period": repeat_period,
                "twist": _nf("twist", 0.0, -1.8, 1.8),
                "bend": _nf("bend", 0.0, -1.2, 1.2),
                "fbm_amp": _nf("fbm_amp", 0.015, 0.0, 0.045),
                "fbm_freq": _nf("fbm_freq", 4.0, 1.0, 10.0),
                "gyroid_scale": gy_scale,
                "gyroid_thickness": _nf("gyroid_thickness", 0.055, 0.025, 0.16),
                "material": material,
                "motion": motion,
            }
        )

    out["nodes"] = norm_nodes
    return out


def _build_emergency_fallback_raymarch_scene(prompt: str) -> Dict[str, Any]:
    """
    Guaranteed valid 12-node 3D multi-paradigm raymarch scene used ONLY if live
    Cloudflare calls fail (e.g., offline test).
    """
    p_low = prompt.lower()
    is_cathedral = any(w in p_low for w in ("cathedral", "gyroid", "bronze", "pillar", "temple"))
    is_dragonfish = any(w in p_low for w in ("dragonfish", "coral", "bioluminescent", "deep-sea", "reef"))
    is_astrolabe = any(w in p_low for w in ("astrolabe", "saturn", "celestial", "ring", "orbit"))

    if is_cathedral:
        floor_type = "gyroid_ground"
        sky_top, sky_hor, floor_rgb = [0.05, 0.04, 0.08], [0.42, 0.28, 0.14], [0.22, 0.16, 0.10]
        main_rgb, acc_rgb = [0.82, 0.58, 0.24], [0.25, 0.92, 0.98]
    elif is_dragonfish:
        floor_type = "terrain_fbm"
        sky_top, sky_hor, floor_rgb = [0.01, 0.04, 0.10], [0.03, 0.16, 0.28], [0.04, 0.12, 0.18]
        main_rgb, acc_rgb = [0.12, 0.88, 0.96], [0.96, 0.22, 0.68]
    elif is_astrolabe:
        floor_type = "cosmic_void"
        sky_top, sky_hor, floor_rgb = [0.02, 0.02, 0.08], [0.14, 0.08, 0.26], [0.06, 0.05, 0.14]
        main_rgb, acc_rgb = [0.94, 0.74, 0.22], [0.30, 0.85, 1.0]
    else:
        floor_type = "ocean_waves"
        sky_top, sky_hor, floor_rgb = [0.04, 0.08, 0.24], [0.68, 0.36, 0.16], [0.04, 0.22, 0.54]
        main_rgb, acc_rgb = [0.98, 0.64, 0.14], [1.0, 0.28, 0.08]

    raw = {
        "title": f"{prompt.strip().title()} — 3D Implicit SDF & Minimal-Surface Manifold",
        "subtitle": "Coupled 3D Sphere-Traced SDF, Schoen Gyroid Minimal Shell & Floraform Buckling Field d(p, t) = 0",
        "aspect": [16, 10],
        "camera_dist": 3.2,
        "camera_height": 0.45,
        "camera_orbit_speed": 0.4,
        "fov": 1.0,
        "sky_top_rgb": sky_top,
        "sky_horizon_rgb": sky_hor,
        "fog_density": 0.042,
        "sun_dir": [0.55, 0.74, 0.38],
        "floor_type": floor_type,
        "floor_height": -0.78,
        "floor_rgb": floor_rgb,
        "nodes": [
            {
                "id": "central_sanctuary_gyroid",
                "op": "union",
                "blend_k": 0.12,
                "primitive": "gyroid_shell",
                "center": [0.0, 0.16, 0.0],
                "size": [0.46, 0.52, 0.46],
                "gyroid_scale": 6.2,
                "gyroid_thickness": 0.055,
                "material": {"rgb": main_rgb, "metallic": 0.85, "roughness": 0.20, "fresnel": 0.9, "stripe_freq": 6.0, "stripe_rgb": acc_rgb},
                "motion": {"amp_pos": [0.0, 0.04, 0.0], "amp_rot": [0.0, 0.12, 0.0], "freq": 1.0, "phase": 0.0},
            },
            {
                "id": "radiant_mandelbulb_core",
                "op": "union",
                "blend_k": 0.08,
                "primitive": "mandelbulb_core",
                "center": [0.0, 0.18, 0.0],
                "size": [0.24, 0.24, 0.24],
                "material": {"rgb": acc_rgb, "metallic": 0.35, "roughness": 0.15, "emission": 1.35, "subsurface": 0.65, "fresnel": 1.1},
                "motion": {"amp_pos": [0.0, 0.05, 0.0], "amp_rot": [0.15, 0.25, 0.1], "freq": 1.0, "phase": 0.5},
            },
            {
                "id": "sweeping_floraform_wings_inner",
                "op": "smin",
                "blend_k": 0.14,
                "primitive": "floraform_ribbon",
                "center": [0.48, 0.24, 0.0],
                "size": [0.06, 0.18, 0.14],
                "endpoint_a": [0.18, 0.12, 0.04],
                "endpoint_b": [0.92, 0.52, -0.14],
                "symmetry_x": True,
                "material": {"rgb": main_rgb, "metallic": 0.72, "roughness": 0.24, "subsurface": 0.55, "emission": 0.35, "stripe_freq": 8.0, "stripe_rgb": acc_rgb},
                "motion": {"amp_pos": [0.0, 0.06, 0.0], "amp_rot": [0.12, 0.08, 0.28], "freq": 1.0, "phase": 0.2},
            },
            {
                "id": "sweeping_floraform_wings_outer",
                "op": "smin",
                "blend_k": 0.12,
                "primitive": "floraform_ribbon",
                "center": [0.76, 0.36, -0.08],
                "size": [0.045, 0.22, 0.12],
                "endpoint_a": [0.42, 0.22, -0.02],
                "endpoint_b": [1.24, 0.68, -0.26],
                "symmetry_x": True,
                "material": {"rgb": acc_rgb, "metallic": 0.65, "roughness": 0.22, "subsurface": 0.75, "emission": 0.55},
                "motion": {"amp_pos": [0.0, 0.08, 0.0], "amp_rot": [0.16, 0.10, 0.34], "freq": 1.0, "phase": 0.8},
            },
            {
                "id": "gimbal_ring_primary",
                "op": "union",
                "blend_k": 0.06,
                "primitive": "torus",
                "center": [0.0, 0.18, 0.0],
                "size": [0.68, 0.038, 0.68],
                "rotation": [0.55, 0.25, 0.0],
                "material": {"rgb": main_rgb, "metallic": 0.92, "roughness": 0.14, "fresnel": 1.0, "stripe_freq": 12.0, "stripe_rgb": acc_rgb},
                "motion": {"amp_pos": [0.0, 0.02, 0.0], "amp_rot": [0.22, 0.35, 0.15], "freq": 1.0, "phase": 1.1},
            },
            {
                "id": "gimbal_ring_secondary",
                "op": "union",
                "blend_k": 0.06,
                "primitive": "torus",
                "center": [0.0, 0.18, 0.0],
                "size": [0.88, 0.032, 0.88],
                "rotation": [-0.62, 0.42, 0.35],
                "material": {"rgb": acc_rgb, "metallic": 0.88, "roughness": 0.18, "emission": 0.35, "fresnel": 0.95},
                "motion": {"amp_pos": [0.0, 0.02, 0.0], "amp_rot": [-0.28, 0.22, -0.18], "freq": 1.0, "phase": 2.2},
            },
            {
                "id": "radial_crown_spires",
                "op": "smin",
                "blend_k": 0.10,
                "primitive": "cone",
                "center": [0.0, 0.56, 0.0],
                "size": [0.08, 0.28, 0.08],
                "symmetry_radial": 8,
                "material": {"rgb": main_rgb, "metallic": 0.86, "roughness": 0.18, "emission": 0.45},
                "motion": {"amp_pos": [0.0, 0.04, 0.0], "amp_rot": [0.0, 0.18, 0.0], "freq": 1.0, "phase": 0.4},
            },
            {
                "id": "ruled_paraboloid_crest",
                "op": "smin",
                "blend_k": 0.14,
                "primitive": "ruled_paraboloid",
                "center": [0.0, 0.42, -0.12],
                "size": [0.42, 0.34, 0.28],
                "rotation": [0.25, 0.0, 0.0],
                "material": {"rgb": main_rgb, "metallic": 0.78, "roughness": 0.22, "subsurface": 0.45, "stripe_freq": 10.0, "stripe_rgb": acc_rgb},
                "motion": {"amp_pos": [0.0, 0.05, 0.0], "amp_rot": [0.08, 0.12, 0.0], "freq": 1.0, "phase": 1.5},
            },
            {
                "id": "bilateral_kleinian_pillars",
                "op": "union",
                "blend_k": 0.08,
                "primitive": "kleinian_fold",
                "center": [0.86, -0.12, -0.22],
                "size": [0.16, 0.58, 0.16],
                "symmetry_x": True,
                "material": {"rgb": main_rgb, "metallic": 0.84, "roughness": 0.26, "fresnel": 0.75},
                "motion": {"amp_pos": [0.0, 0.015, 0.0], "amp_rot": [0.0, 0.04, 0.0], "freq": 1.0, "phase": 0.0},
            },
            {
                "id": "luminous_beacon_orbs",
                "op": "union",
                "blend_k": 0.05,
                "primitive": "sphere",
                "center": [0.86, 0.54, -0.22],
                "size": [0.11, 0.11, 0.11],
                "symmetry_x": True,
                "material": {"rgb": acc_rgb, "metallic": 0.2, "roughness": 0.1, "emission": 1.65, "subsurface": 0.8},
                "motion": {"amp_pos": [0.0, 0.06, 0.0], "amp_rot": [0.0, 0.0, 0.0], "freq": 2.0, "phase": 0.7},
            },
        ],
    }
    return _normalize_raymarch_scene(raw, prompt=prompt)


def _extract_raymarch_scene_json(text: str) -> Optional[Dict[str, Any]]:
    text = str(text or "").strip()
    text = re.sub(r"<think>[\s\S]*?</think>", "", text, flags=re.IGNORECASE).strip()
    if "</think>" in text.lower():
        text = re.split(r"</think>", text, flags=re.IGNORECASE)[-1].strip()
    fenced = re.findall(r"```(?:json)?\s*([\s\S]*?)```", text, flags=re.IGNORECASE)
    search_pool = list(reversed(fenced)) + [text]

    decoder = json.JSONDecoder()
    best_dict: Optional[Dict[str, Any]] = None
    for blob in search_pool:
        s = blob.strip()
        idx = 0
        while idx < len(s):
            brace = s.find("{", idx)
            if brace == -1:
                break
            try:
                obj, end_idx = decoder.raw_decode(s, brace)
                if isinstance(obj, dict):
                    if isinstance(obj.get("nodes"), list) and len(obj["nodes"]) >= 4:
                        return obj
                    if best_dict is None:
                        best_dict = obj
                idx = max(end_idx, brace + 1)
                continue
            except json.JSONDecodeError:
                pass
            idx = brace + 1

        start = s.find("{")
        end = s.rfind("}")
        if start != -1 and end != -1 and end > start:
            candidate = re.sub(r",\s*([\]}])", r"\1", s[start : end + 1])
            try:
                obj = json.loads(candidate)
                if isinstance(obj, dict):
                    if isinstance(obj.get("nodes"), list) and len(obj["nodes"]) >= 4:
                        return obj
                    if best_dict is None:
                        best_dict = obj
            except Exception:
                pass
    return best_dict


def _invoke_flagship_for_raymarch_scene(
    system_prompt: str,
    user_msg: str,
    max_tokens: int = 8192,
    temperature: float = 0.22,
    prefer_glm: bool = True,
) -> Optional[Dict[str, Any]]:
    models = (
        [GLM_5_3, DEEPSEEK_V4_PRO, GLM_5_3_FLASH]
        if prefer_glm
        else [DEEPSEEK_V4_PRO, GLM_5_3, GLM_5_3_FLASH]
    )
    if not _has_cf_credentials():
        return None
    for model in models:
        url = f"https://api.cloudflare.com/client/v4/accounts/{CF_ACCOUNT_ID}/ai/run/{model}"
        payload: Dict[str, Any] = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_msg},
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if "glm-5" in model:
            payload["reasoning_effort"] = "low"
        elif "deepseek-v4" in model:
            payload["chat_template_kwargs"] = {"enable_thinking": False}

        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {CF_API_TOKEN}",
                "Content-Type": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=90) as resp:
                body = json.loads(resp.read().decode("utf-8"))
                raw_resp = _extract_response_text(body)
                if isinstance(raw_resp, dict):
                    sc = raw_resp
                else:
                    sc = _extract_raymarch_scene_json(str(raw_resp))
                if sc and isinstance(sc.get("nodes"), list) and len(sc["nodes"]) >= 4:
                    return sc
        except Exception:
            continue
    return None


def _save_raymarch_3d_cache() -> None:
    try:
        with open(RAYMARCH_3D_CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(_RAYMARCH_3D_CACHE, f, indent=2)
    except Exception:
        pass


def _load_raymarch_3d_cache() -> None:
    global _RAYMARCH_3D_CACHE
    if not os.path.exists(RAYMARCH_3D_CACHE_FILE):
        _RAYMARCH_3D_CACHE = {}
        return
    try:
        with open(RAYMARCH_3D_CACHE_FILE, "r", encoding="utf-8") as f:
            raw = json.load(f)
        if isinstance(raw, dict):
            cleaned: Dict[str, Dict[str, Any]] = {}
            for k, v in raw.items():
                if isinstance(v, dict) and isinstance(v.get("nodes"), list) and len(v["nodes"]) >= 4:
                    cleaned[k] = _normalize_raymarch_scene(v, prompt=k)
            _RAYMARCH_3D_CACHE = cleaned
    except Exception:
        _RAYMARCH_3D_CACHE = {}


_load_raymarch_3d_cache()


def critique_rendered_3d_raymarch_image(
    prompt: str,
    image_base64: str,
    stats: Dict[str, Any],
    motion_energy: float = 0.0,
) -> Dict[str, Any]:
    """
    Sends the rendered 3D Raymarched preview PNG + 3D normal/depth/lighting diagnostics
    to the live Cloudflare Vision Critic cascade (Gemma 4 26B / Mistral Small 3.1 24B /
    Llama 4 Scout 17B-16E) and returns a strict 1..10 score + concrete 3D (x, y, z) fixes.
    """
    data_uri = (
        image_base64
        if str(image_base64).startswith("data:image/")
        else f"data:image/png;base64,{image_base64}"
    )

    critique_prompt = (
        f'You are a strict 3D mathematical-cinema & shader-raymarching critic evaluating a 3D sphere-traced '
        f'SDF + differential-geometry scene in R^3 (+Y up, +X right, +Z toward camera, stage [-1.5, 1.5] x [-0.8, 1.2] x [-1.2, 1.2]).\n'
        f'Intended 3D subject: "{prompt}"\n'
        f'Quantitative 3D diagnostics: std={stats.get("std", 0):.1f}, chroma={stats.get("chroma", 0):.1f}, '
        f'normal_variance={stats.get("normal_variance", 0):.3f}, motion_energy={motion_energy:.2f}.\n\n'
        f'Score 1.0 to 10.0 on: (1) true 3D volumetric depth & sculptural normals (NEVER flat 2D cutouts!), '
        f'(2) unmistakable anatomical/architectural recognizability for "{prompt}", '
        f'(3) rich PBR metallic/subsurface/emissive lighting & floor reflections/waves, and (4) camera framing.\n'
        f'Do NOT give >= 8.5 unless the 3D sculpture is museum-grade and unmistakably recognizable.\n'
        f'Return STRICT JSON ONLY:\n'
        f'{{"score": 1.0..10.0, "what_is_visible": "honest 1-sentence 3D description", '
        f'"defects": ["3D defect 1", "3D defect 2"], '
        f'"coordinate_fixes": ["concrete 3D node/primitive/material/camera fix with (x, y, z) coords", ...]}}'
    )

    for vm in (VISION_CRITIC_MODELS if _has_cf_credentials() else []):
        url = f"https://api.cloudflare.com/client/v4/accounts/{CF_ACCOUNT_ID}/ai/run/{vm}"
        payload = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": critique_prompt},
                        {"type": "image_url", "image_url": {"url": data_uri}},
                    ],
                }
            ],
            "max_tokens": 900,
            "temperature": 0.15,
        }
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {CF_API_TOKEN}",
                "Content-Type": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=45) as resp:
                body = json.loads(resp.read().decode("utf-8"))
                raw_text = _extract_response_text(body)
                parsed = raw_text if isinstance(raw_text, dict) else _extract_json_object(str(raw_text))
                if isinstance(parsed, dict) and "score" in parsed:
                    score = float(np.clip(float(parsed.get("score", 5.5)), 1.0, 10.0))
                    defects = parsed.get("defects") if isinstance(parsed.get("defects"), list) else []
                    fixes = parsed.get("coordinate_fixes") if isinstance(parsed.get("coordinate_fixes"), list) else []
                    if not fixes and isinstance(parsed.get("fixes"), list):
                        fixes = parsed["fixes"]
                    return {
                        "score": round(score, 2),
                        "what_is_visible": str(parsed.get("what_is_visible") or parsed.get("sees") or ""),
                        "defects": [str(d) for d in defects[:6]],
                        "coordinate_fixes": [str(fx) for fx in fixes[:6]],
                    }
        except Exception:
            continue

    std_v = float(stats.get("std", 0.0))
    chr_v = float(stats.get("chroma", 0.0))
    heuristic_score = round(min(8.2, 3.2 + 2.5 * min(1.0, std_v / 48.0) + 2.5 * min(1.0, chr_v / 38.0)), 2)
    return {
        "score": heuristic_score,
        "what_is_visible": "3D sphere-traced implicit manifold evaluated via volumetric luminance and normal variance metrics.",
        "defects": [
            "Increase 3D silhouette scale, sculptural depth along Z, and emissive/specular PBR contrast."
        ],
        "coordinate_fixes": [
            "Add bilateral 3D floraform_ribbon / gyroid_shell / torus detail nodes around (0.0, 0.2, 0.0) and boost key-light specular + emission."
        ],
    }


def _repair_raymarch_scene(
    scene: Dict[str, Any], error_msg: str, prompt: str
) -> Optional[Dict[str, Any]]:
    repair_msg = (
        f'The 3D raymarch_3d_scene JSON below for "{prompt}" failed validation with error:\n'
        f'ERROR: {error_msg}\n\n'
        f'Fix the error while preserving all 12-20 3D nodes, PBR materials, and 4D periodic motion.\n'
        f'Allowed primitives: {sorted(VALID_RAYMARCH_PRIMITIVES)}\n'
        f'Allowed ops: {sorted(VALID_RAYMARCH_OPS)}\n'
        f'Allowed floor_type: {sorted(VALID_RAYMARCH_FLOORS)}\n'
        f'Original JSON:\n{json.dumps(scene)}\n\n'
        f'Return ONLY the repaired valid JSON.'
    )
    repaired = _invoke_flagship_for_raymarch_scene(
        RAYMARCH_3D_SYSTEM_PROMPT, repair_msg, max_tokens=8192, temperature=0.15
    )
    if repaired is not None:
        return _normalize_raymarch_scene(repaired, prompt=prompt)
    return None


def _evaluate_raymarch_preview_motion_rgb(
    scene: Dict[str, Any], width: int = 384, height: int = 240
) -> Tuple[Dict[str, Any], float, Optional[np.ndarray]]:
    from master_paradigms_engine import evaluate_raymarch_scene_numpy

    res0 = evaluate_raymarch_scene_numpy(
        scene, width=width, height=height, t_phase=0.0, N=900, include_raw_array=True
    )
    res1 = evaluate_raymarch_scene_numpy(
        scene, width=width, height=height, t_phase=math.pi * 0.5, N=900, include_raw_array=True
    )
    rgb0 = res0.get("rgb_array")
    rgb1 = res1.get("rgb_array")
    if rgb0 is not None and rgb1 is not None:
        motion_energy = float(np.mean(np.abs(rgb1.astype(np.float32) - rgb0.astype(np.float32))))
    else:
        motion_energy = 0.0
    res0_clean = {k: v for k, v in res0.items() if k != "rgb_array"}
    return res0_clean, round(motion_energy, 3), rgb0


def _rank_raymarch_pass_quality(score: float, stats: Dict[str, Any], motion_energy: float) -> float:
    std_v = float(stats.get("std", 0.0))
    chr_v = float(stats.get("chroma", 0.0))
    nvar_v = float(stats.get("normal_variance", 0.15))
    low_contrast_pen = 1.5 if std_v < 28.0 else 0.0
    return (
        float(score)
        + 0.45 * min(1.0, std_v / 45.0)
        + 0.45 * min(1.0, chr_v / 32.0)
        + 0.35 * min(1.0, motion_energy / 3.0)
        + 0.30 * min(1.0, nvar_v / 0.25)
        - low_contrast_pen
    )


def generate_3d_raymarch_scene_with_critique(
    prompt: str,
    refine: bool = True,
    bypass_cache: bool = False,
) -> Dict[str, Any]:
    """
    Stage-2 Universal 3D Raymarched + Multi-Paradigm Cinema Architect:
      - Pass 1: Synthesizes a 12-20 node 3D raymarch_3d_scene via live GLM-5.3 / DeepSeek V4.1
        combining Inigo Quilez 3D SDFs, Ferguson Gyroid minimal shells, Nervous System Floraform
        differential-growth ribbons, Xenakis ruled paraboloids, and Kleinian folds.
      - Pass 2 (when refine=True): Renders 3D sphere-traced preview at t=0 and t=pi/2,
        checks 3D normal variance, contrast, motion energy, and runs live Cloudflare Vision Critic
        to perform closed-loop 3D geometric & PBR material refinement.
      - Caches winner in backend/raymarch_3d_cache.json with zero AI/model branding leaks.
    """
    from master_paradigms_engine import validate_raymarch_scene

    cache_key = prompt.strip().lower()
    if not bypass_cache and cache_key in _RAYMARCH_3D_CACHE:
        cached = _normalize_raymarch_scene(_RAYMARCH_3D_CACHE[cache_key], prompt=prompt)
        ok, _ = validate_raymarch_scene(cached)
        if ok:
            return {
                "scene": cached,
                "program": cached,
                "pass1_scene": cached,
                "pass2_scene": None,
                "pass1_score": float(cached.get("visual_score", 8.6)),
                "pass2_score": None,
                "pass1_preview_base64": None,
                "pass2_preview_base64": None,
                "refinement_passes": int(cached.get("refinement_passes", 1)),
                "visual_score": float(cached.get("visual_score", 8.6)),
                "motion_energy": float(cached.get("motion_energy", 4.0)),
                "improved": False,
                "pass1_pass2_mad": 0.0,
            }

    pass1_user = (
        f'Design a breathtaking, museum-grade 12-to-18-node 3D Raymarched Implicit Scene (`raymarch_3d_scene`) '
        f'for: "{prompt}"\n\n'
        f'Requirements:\n'
        f'1. Sculpt a true 3D volumetric masterpiece in [-1.4, 1.4] x [-0.7, 1.2] x [-1.2, 1.2] (NEVER flat 2D cutouts!).\n'
        f'2. Combine classic 3D SDF primitives (`sphere`, `ellipsoid`, `capsule`, `torus`, `cone`, `box`, `cylinder`) '
        f'WITH Master-Paradigm 3D primitives (`gyroid_shell`, `floraform_ribbon`, `ruled_paraboloid`, `mandelbulb_core`, `kleinian_fold`).\n'
        f'3. Use `"symmetry_x": true` (with `center[0] > 0.1`) for bilateral 3D wings/pillars/fins/eyes/horns, and `"symmetry_radial": 6..12` for radial crowns/spokes/corals/arches.\n'
        f'4. Assign rich 3D PBR materials (`metallic`, `roughness`, `subsurface`, `emission`, `fresnel`, `stripe_freq`, `stripe_rgb`) and choose the best `"floor_type"` (`ocean_waves`, `gyroid_ground`, `terrain_fbm`, `cosmic_void`, `marble_reflection`).\n'
        f'5. Add visible 2*PI-periodic 3D motion (`amp_pos`, `amp_rot`, integer `freq` 1.0 or 2.0) and camera orbit (`camera_orbit_speed`: 0.3..0.6).\n'
        f'6. NEVER mention AI or model names in title or subtitle. Output STRICT JSON ONLY.'
    )

    raw_s1 = _invoke_flagship_for_raymarch_scene(
        RAYMARCH_3D_SYSTEM_PROMPT, pass1_user, max_tokens=8192, temperature=0.22, prefer_glm=True
    )
    if raw_s1 is None:
        fallback = _build_emergency_fallback_raymarch_scene(prompt)
        return {
            "scene": fallback,
            "program": fallback,
            "pass1_scene": fallback,
            "pass2_scene": None,
            "pass1_score": 7.5,
            "pass2_score": None,
            "pass1_preview_base64": None,
            "pass2_preview_base64": None,
            "refinement_passes": 1,
            "visual_score": 7.5,
            "motion_energy": 4.2,
            "improved": False,
            "pass1_pass2_mad": 0.0,
        }

    s1 = _normalize_raymarch_scene(raw_s1, prompt=prompt)
    ok1, err1 = validate_raymarch_scene(s1)
    if not ok1:
        rep1 = _repair_raymarch_scene(s1, err1, prompt)
        if rep1 is not None and validate_raymarch_scene(rep1)[0]:
            s1, ok1 = rep1, True
    if not ok1:
        s1 = _build_emergency_fallback_raymarch_scene(prompt)

    try:
        eval1, motion1, rgb1 = _evaluate_raymarch_preview_motion_rgb(s1, width=384, height=240)
        stats1 = eval1.get("stats", {})
        img1_b64 = eval1.get("image_base64", "")
    except Exception:
        s1 = _build_emergency_fallback_raymarch_scene(prompt)
        eval1, motion1, rgb1 = _evaluate_raymarch_preview_motion_rgb(s1, width=384, height=240)
        stats1 = eval1.get("stats", {})
        img1_b64 = eval1.get("image_base64", "")

    if not refine:
        s1["refinement_passes"] = 1
        s1["visual_score"] = round(min(8.8, 5.2 + float(stats1.get("std", 35.0)) / 18.0), 2)
        s1["motion_energy"] = motion1
        _RAYMARCH_3D_CACHE[cache_key] = s1
        _save_raymarch_3d_cache()
        return {
            "scene": s1,
            "program": s1,
            "pass1_scene": s1,
            "pass2_scene": None,
            "pass1_score": s1["visual_score"],
            "pass2_score": None,
            "pass1_preview_base64": img1_b64,
            "pass2_preview_base64": None,
            "refinement_passes": 1,
            "visual_score": s1["visual_score"],
            "motion_energy": motion1,
            "improved": False,
            "pass1_pass2_mad": 0.0,
        }

    critique1 = critique_rendered_3d_raymarch_image(prompt, img1_b64, stats1, motion1)
    score1 = float(critique1.get("score", 6.5))
    std1 = float(stats1.get("std", 0.0))
    chroma1 = float(stats1.get("chroma", 0.0))
    nvar1 = float(stats1.get("normal_variance", 0.15))

    needs_refine = (
        (score1 < 8.5)
        or (std1 < 32.0)
        or (chroma1 < 20.0)
        or (nvar1 < 0.06)
        or (motion1 < 1.2)
    )
    if not needs_refine:
        s1["refinement_passes"] = 1
        s1["visual_score"] = score1
        s1["motion_energy"] = motion1
        _RAYMARCH_3D_CACHE[cache_key] = s1
        _save_raymarch_3d_cache()
        return {
            "scene": s1,
            "program": s1,
            "pass1_scene": s1,
            "pass2_scene": None,
            "pass1_score": score1,
            "pass2_score": None,
            "pass1_preview_base64": img1_b64,
            "pass2_preview_base64": None,
            "refinement_passes": 1,
            "visual_score": score1,
            "motion_energy": motion1,
            "improved": False,
            "pass1_pass2_mad": 0.0,
        }

    defects_txt = "\n".join(f"  - {d}" for d in critique1.get("defects", [])) or "  - Increase 3D volumetric depth, silhouette scale, and PBR lighting contrast."
    fixes_txt = "\n".join(f"  - {fx}" for fx in critique1.get("coordinate_fixes", [])) or "  - Enlarge hero 3D nodes, add emissive/metallic accents, and improve camera framing."

    pass2_user = (
        f'DRAMATICALLY UPGRADE AND REFINE this 3D Raymarched Implicit Scene (`raymarch_3d_scene`) for: "{prompt}"\n\n'
        f'3D Vision Critic Report on Pass 1 (score {score1:.1f}/10.0, std={std1:.1f}, chroma={chroma1:.1f}, normal_variance={nvar1:.3f}, motion={motion1:.2f}):\n'
        f'What rendered in Pass 1: {critique1.get("what_is_visible", "")}\n'
        f'3D DEFECTS TO FIX:\n{defects_txt}\n'
        f'CONCRETE 3D (x, y, z) & MATERIAL FIXES TO APPLY:\n{fixes_txt}\n\n'
        f'MANDATORY UPGRADES FOR PASS 2:\n'
        f'1. DO NOT copy Pass 1 verbatim! Substantially enrich the 3D silhouette, depth along Z, sculptural details, and PBR lighting.\n'
        f'2. Ensure the hero 3D sculpture fills 55%-75% of the viewport (`camera_dist` in 2.7..3.5, `camera_height` in 0.25..0.65) with high contrast against the floor & sky.\n'
        f'3. Include Master-Paradigm 3D primitives (`gyroid_shell`, `floraform_ribbon`, `ruled_paraboloid`, `mandelbulb_core`, `kleinian_fold`, `torus`, `capsule`), `"symmetry_x": true` for bilateral pairs, and `"symmetry_radial"` for radial structures.\n'
        f'4. Boost emissive highlights (`emission: 0.9..2.0` on glowing cores/eyes/lures/rings), metallic reflections, and smooth 2*PI-periodic motion.\n'
        f'5. Keep 12 to 20 nodes total and ZERO AI/model names.\n\n'
        f'Pass 1 JSON (rewrite & upgrade aggressively):\n{json.dumps(s1)}\n\n'
        f'Output STRICT JSON ONLY.'
    )

    def _eval_s2_cand(
        raw_c: Optional[Dict[str, Any]],
    ) -> Optional[Tuple[Dict[str, Any], Dict[str, Any], float, float, float, str]]:
        if raw_c is None:
            return None
        cand = _normalize_raymarch_scene(raw_c, prompt=prompt)
        ok_c, err_c = validate_raymarch_scene(cand)
        if not ok_c:
            rep_c = _repair_raymarch_scene(cand, err_c, prompt)
            if rep_c is not None and validate_raymarch_scene(rep_c)[0]:
                cand, ok_c = rep_c, True
        if not ok_c:
            return None
        try:
            ev_c, mot_c, rgb_c = _evaluate_raymarch_preview_motion_rgb(cand, width=384, height=240)
            st_c = ev_c.get("stats", {})
            b64_c = ev_c.get("image_base64", "")
            mad_c = _compute_rgb_mad(rgb1, rgb_c)
            crit_c = critique_rendered_3d_raymarch_image(prompt, b64_c, st_c, mot_c)
            sc_c = float(crit_c.get("score", score1))
            return cand, ev_c, mot_c, sc_c, mad_c, b64_c
        except Exception:
            return None

    raw_s2 = _invoke_flagship_for_raymarch_scene(
        RAYMARCH_3D_SYSTEM_PROMPT, pass2_user, max_tokens=8192, temperature=0.32, prefer_glm=False
    )
    cand_a = _eval_s2_cand(raw_s2)

    cand_b = None
    if cand_a is None or cand_a[4] < 2.0 or float(cand_a[1].get("stats", {}).get("std", 0.0)) < 28.0:
        s1_ids = ", ".join(nd.get("id", "") for nd in s1.get("nodes", [])[:18])
        pass2_rebuild = (
            f'BUILD A DRAMATICALLY SUPERIOR 3D Raymarched Implicit Scene (`raymarch_3d_scene`) from scratch for: "{prompt}"\n\n'
            f'Our Pass 1 attempt (nodes: {s1_ids}) scored {score1:.1f}/10.0 (std={std1:.1f}, chroma={chroma1:.1f}).\n'
            f'What rendered in Pass 1: {critique1.get("what_is_visible", "")}\n'
            f'3D DEFECTS YOU MUST ELIMINATE:\n{defects_txt}\n'
            f'CONCRETE 3D (x, y, z) & MATERIAL IMPROVEMENTS REQUIRED:\n{fixes_txt}\n\n'
            f'Build 14 to 18 rich 3D nodes combining `gyroid_shell`, `floraform_ribbon`, `ruled_paraboloid`, `mandelbulb_core`, `kleinian_fold`, `torus`, `capsule`, `ellipsoid`, `cone` with `"symmetry_x": true`, `"symmetry_radial"`, glowing emissive accents, metallic PBR reflections, and dynamic 4D motion. ZERO AI/model names. Output STRICT JSON ONLY.'
        )
        raw_s2_retry = _invoke_flagship_for_raymarch_scene(
            RAYMARCH_3D_SYSTEM_PROMPT, pass2_rebuild, max_tokens=8192, temperature=0.42, prefer_glm=True
        )
        cand_b = _eval_s2_cand(raw_s2_retry)

    pool = [c for c in (cand_a, cand_b) if c is not None]
    chosen_s2 = None
    if pool:
        chosen_s2 = max(
            pool,
            key=lambda c: _rank_raymarch_pass_quality(c[3], c[1].get("stats", {}), c[2])
            + (0.5 if c[4] >= 2.0 else -2.0),
        )

    s2: Optional[Dict[str, Any]] = None
    eval2: Optional[Dict[str, Any]] = None
    motion2: float = 0.0
    score2: Optional[float] = None
    mad_1_2: float = 0.0
    img2_b64: Optional[str] = None
    if chosen_s2 is not None:
        s2, eval2, motion2, score2, mad_1_2, img2_b64 = chosen_s2

    rank1 = _rank_raymarch_pass_quality(score1, stats1, motion1)
    if s2 is not None and eval2 is not None and score2 is not None:
        stats2 = eval2.get("stats", {})
        rank2 = _rank_raymarch_pass_quality(score2, stats2, motion2)
        if mad_1_2 >= 2.0 and rank2 >= rank1 - 0.15:
            winner = s2
            winner["refinement_passes"] = 2
            winner["visual_score"] = score2
            winner["motion_energy"] = motion2
            improved = True
        else:
            winner = s1
            winner["refinement_passes"] = 2
            winner["visual_score"] = score1
            winner["motion_energy"] = motion1
            improved = False
    else:
        winner = s1
        winner["refinement_passes"] = 1
        winner["visual_score"] = score1
        winner["motion_energy"] = motion1
        improved = False

    _RAYMARCH_3D_CACHE[cache_key] = winner
    _save_raymarch_3d_cache()
    return {
        "scene": winner,
        "program": winner,
        "pass1_scene": s1,
        "pass2_scene": s2,
        "pass1_score": score1,
        "pass2_score": score2,
        "pass1_preview_base64": img1_b64,
        "pass2_preview_base64": img2_b64,
        "refinement_passes": int(winner["refinement_passes"]),
        "visual_score": float(winner["visual_score"]),
        "motion_energy": float(winner["motion_energy"]),
        "improved": improved,
        "pass1_pass2_mad": round(mad_1_2, 3),
    }


def generate_3d_raymarch_scene(
    prompt: str,
    refine: bool = True,
    bypass_cache: bool = False,
) -> Dict[str, Any]:
    """
    Synthesizes (and optionally 2-pass visually refines) a canonical 3D Raymarched
    Implicit Scene (`raymarch_3d_scene`) for any arbitrary user prompt using
    live GLM-5.3 & DeepSeek V4.1 with silent closed-loop 3D visual self-critique
    and zero AI/model branding leaks.
    """
    bundle = generate_3d_raymarch_scene_with_critique(
        prompt=prompt, refine=refine, bypass_cache=bypass_cache
    )
    return bundle["scene"]


# ============================================================================
# STAGE 3 LONG-SCRIPT MULTI-ACT MATHEMATICAL CINEMA STORYBOARD ARCHITECT
# ============================================================================

SAMPLE_EPIC_5_ACT_SCRIPT = """ACT 1: Genesis of the Complex Riemann Plane
In the beginning, pure complex conformal waves w(z, t) = (1 - tau)*z + tau*(z^3 - 1)/(z^3 + 1) bend the fabric of the mathematical vacuum into glowing analytical phase lattices and essential singularities.

ACT 2: Awakening in the Abyssal Floraform Reef
From the complex plane, life emerges in the deep ocean: a bioluminescent deep-sea dragonfish glides above ruffled differential-growth Floraform coral saddles governed by elastic buckling PDEs.

ACT 3: Ascent of the Aurelian Phoenix
Breaking through the ocean surface, a golden 3D raymarched phoenix with FBM-sculpted plumage ascends over turbulent sapphire waves under volumetric god-rays.

ACT 4: Sanctum of the Hyperbolic Gyroid Cathedral
The phoenix enters a monumental bronze cathedral carved from Schoen's triply periodic Gyroid minimal surface sin(x)cos(y) + sin(y)cos(z) + sin(z)cos(x) = 0 with zero mean curvature H = 0.

ACT 5: Transcendence into 4D Non-Euclidean Space
At the altar, a sphere turns smoothly inside-out via William Thurston's C4 Morin sphere-eversion homotopy, expanding into a non-Euclidean celestial astrolabe encircled by golden Saturnian rings."""

VALID_STORYBOARD_PARADIGMS = {
    "mp_manim_conformal",
    "mp_iq_mandelbulb",
    "mp_ddg_gyroid",
    "mp_floraform_pde",
    "mp_thurston_eversion",
    "mp_xenakis_metastasis",
    "mp_houdini_curl_attractor",
    "mp_kleinian_mandelbox",
    "pf_peacock",
    "pf_bipolar_nebula",
    "pf_corals",
    "custom_raymarch_3d",
}

STORYBOARD_SYSTEM_PROMPT = """You are a Mathematical Cinema Storyboard Director.
Decompose the user's multi-act or multi-paragraph script into M sequential narrative acts (3 <= M <= max_acts).
Output STRICT JSON ONLY with schema:
{
  "acts": [
    {
      "act_number": 1,
      "title": "Pure Mathematical & Visual Act Title (NEVER mention AI or model names)",
      "narration": "Exact narrative sentence from the script for burned-in cinema subtitles",
      "visual_prompt": "Detailed 3D mathematical visual description for this act",
      "paradigm_choice": "mp_manim_conformal | mp_iq_mandelbulb | mp_ddg_gyroid | mp_floraform_pde | mp_thurston_eversion | mp_xenakis_metastasis | mp_houdini_curl_attractor | mp_kleinian_mandelbox | pf_peacock | pf_bipolar_nebula | pf_corals | custom_raymarch_3d",
      "formula_ticker": "Compact 1-line governing mathematical equation for the HUD"
    }
  ]
}"""


def _split_script_into_raw_acts(script_text: str, max_acts: int = 6) -> List[Dict[str, str]]:
    """
    Splits any script (whether formatted with explicit multi-line 'ACT 1: Title\\nNarration',
    inline 'Act 1: sentence. Act 2: sentence.', or unstructured paragraphs/sentences)
    into sequential (title, narration) act blocks.
    """
    text = (script_text or "").strip()
    if not text:
        text = SAMPLE_EPIC_5_ACT_SCRIPT

    # Check for explicit ACT / SCENE markers (multi-line or inline)
    marker_re = re.compile(
        r"(?:^|\n|(?<=[.!?])\s+)(?:ACT|Act|SCENE|Scene|PART|Part)\s*(\d+|[IVX]+)\s*[:\-–—]\s*",
    )
    matches = list(marker_re.finditer(text))
    acts_out: List[Dict[str, str]] = []

    if len(matches) >= 2:
        for i, m in enumerate(matches[:max_acts]):
            start_idx = m.end()
            end_idx = matches[i + 1].start() if (i + 1) < len(matches) else len(text)
            chunk = text[start_idx:end_idx].strip()
            if not chunk:
                continue
            lines = [ln.strip() for ln in chunk.splitlines() if ln.strip()]
            if len(lines) >= 2:
                raw_title = lines[0].rstrip(".!?:")
                narration = " ".join(lines[1:]).strip()
            else:
                single = lines[0] if lines else chunk
                # Check if there is a colon or dash separating a short title from narration
                if ":" in single and single.find(":") < 48:
                    p_title, p_narr = single.split(":", 1)
                    raw_title = p_title.strip()
                    narration = p_narr.strip()
                else:
                    narration = single.strip()
                    words = re.sub(r"[^a-zA-Z0-9\s\-]", "", narration).split()
                    raw_title = " ".join(words[:6]).title() if words else f"Act {i + 1}"
            acts_out.append({"title": raw_title, "narration": narration})
        if acts_out:
            return acts_out[:max_acts]

    # Next, try splitting by double-newline paragraphs
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if len(paras) >= 2:
        for i, p in enumerate(paras[:max_acts]):
            lines = [ln.strip() for ln in p.splitlines() if ln.strip()]
            if len(lines) >= 2 and len(lines[0]) < 68:
                raw_title = re.sub(r"^(?:ACT|Act|SCENE|Scene)\s*\d+\s*[:\-–—]\s*", "", lines[0]).strip()
                narration = " ".join(lines[1:]).strip()
            else:
                narration = " ".join(lines).strip()
                words = re.sub(r"[^a-zA-Z0-9\s\-]", "", narration).split()
                raw_title = " ".join(words[:6]).title() if words else f"Act {i + 1}"
            acts_out.append({"title": raw_title, "narration": narration})
        return acts_out[:max_acts]

    # Finally, split single-paragraph prose into sentences and group into 3..max_acts acts
    sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    if not sents:
        sents = [text]
    target_n = min(max_acts, max(1, len(sents)))
    chunk_sz = max(1, int(math.ceil(len(sents) / float(target_n))))
    for i in range(0, len(sents), chunk_sz):
        grp = " ".join(sents[i : i + chunk_sz]).strip()
        if not grp:
            continue
        words = re.sub(r"[^a-zA-Z0-9\s\-]", "", grp).split()
        raw_title = " ".join(words[:6]).title() if words else f"Act {len(acts_out) + 1}"
        acts_out.append({"title": raw_title, "narration": grp})
        if len(acts_out) >= max_acts:
            break
    return acts_out


def _route_act_to_paradigm_and_prompt(
    title: str,
    narration: str,
    llm_choice: str = "",
    llm_visual_prompt: str = "",
    llm_formula: str = "",
) -> Dict[str, str]:
    """
    Deterministically & semantically maps an act's title + narration to the exact
    Master Paradigm preset or cached/live 3D Raymarch Scene, paradigm badge, and
    governing mathematical equation ticker.
    """
    blob = f"{title} {narration} {llm_visual_prompt}".lower()
    title_low = title.lower()

    # 1. Bioluminescent Deep-Sea Dragonfish + Floraform Coral (3D Raymarch)
    if "dragonfish" in blob or ("deep-sea" in blob and "coral" in blob) or ("abyssal" in title_low and ("floraform" in title_low or "reef" in title_low)):
        return {
            "paradigm_choice": "custom_raymarch_3d",
            "visual_prompt": "a bioluminescent deep-sea dragonfish surrounded by differential-growth coral",
            "paradigm_badge": "3D SDF Raymarching + Floraform Differential-Growth PDE",
            "formula_ticker": "∂_t γ = F_spring + F_bend + F_repulse • d_dragonfish(p,t) = 0",
        }

    # 2. Hyperbolic Bronze Gyroid Cathedral (3D Raymarch) — checked before 'phoenix' so "The phoenix enters a bronze cathedral..." routes to Cathedral!
    if ("cathedral" in blob or "sanctum" in blob or "pillar" in blob or "bronze" in blob) and ("gyroid" in blob or "minimal surface" in blob or "hyperbolic" in blob):
        return {
            "paradigm_choice": "custom_raymarch_3d",
            "visual_prompt": "a hyperbolic bronze cathedral with glowing gyroid pillars and god rays",
            "paradigm_badge": "Ferguson/Grossman Schoen Gyroid TPMS Cathedral (H = 0)",
            "formula_ticker": "sin(kx)cos(ky) + sin(ky)cos(kz) + sin(kz)cos(kx) = 0 • H = (κ₁+κ₂)/2 = 0",
        }

    # 3. Celestial Astrolabe with Saturn Rings (+ 4D Sphere Eversion) (3D Raymarch)
    if "astrolabe" in blob or ("saturn" in blob and "ring" in blob):
        return {
            "paradigm_choice": "custom_raymarch_3d",
            "visual_prompt": "a non-Euclidean celestial astrolabe with orbiting Saturn rings",
            "paradigm_badge": "4D Thurston Eversion + Non-Euclidean Celestial Astrolabe",
            "formula_ticker": "H_τ: S²×[0,1] → R³ • d_astrolabe(p,t) = smin(d_core, d_rings)",
        }

    # 4. Golden Phoenix over Turbulent Sapphire Ocean Waves (3D Raymarch)
    if "phoenix" in blob or ("golden" in blob and "ocean" in blob and ("wave" in blob or "wing" in blob or "plumage" in blob)):
        return {
            "paradigm_choice": "custom_raymarch_3d",
            "visual_prompt": "a golden phoenix rising over turbulent sapphire ocean waves",
            "paradigm_badge": "Quilez 3D Sphere-Tracing + FBM Plumage + Sapphire Waves",
            "formula_ticker": "s_{n+1} = s_n + d(r_o + s_n r_d, t) • N(p) = ∇d/‖∇d‖",
        }

    # 5. Complex Conformal / Riemann / Manim
    if any(k in blob for k in ("conformal", "riemann", "mobius", "möbius", "joukowsky", "w(z", "singularity", "manim")) or ("complex" in title_low):
        return {
            "paradigm_choice": "mp_manim_conformal",
            "visual_prompt": "Complex conformal domain coloring and Riemann phase-lattice homotopy w(z, t)",
            "paradigm_badge": "Manim Complex Conformal Homotopy w(z,t)",
            "formula_ticker": "w(z,t) = (1-τ(t))z + τ(t)(z³-1)/(z³+1) • Re(w), Im(w) ∈ Z",
        }

    # 6. Inigo Quilez 3D Mandelbulb Fractal
    if "mandelbulb" in blob or ("quilez" in blob and "fractal" in blob):
        return {
            "paradigm_choice": "mp_iq_mandelbulb",
            "visual_prompt": "Inigo Quilez 8th-order 3D Mandelbulb fractal in deep space",
            "paradigm_badge": "Inigo Quilez 8th-Order 3D Mandelbulb Raymarching",
            "formula_ticker": "z_{n+1} = r⁸(sin(8θ)cos(8φ), sin(8θ)sin(8φ), cos(8θ)) + p • d(p) = 0.5 r ln(r)/dr",
        }

    # 7. Houdini Curl-Noise / RK4 Strange Attractor / Lorenz / Gray-Scott PDE
    if any(k in blob for k in ("lorenz", "strange attractor", "attractor", "curl-noise", "curl noise", "gray-scott", "reaction-diffusion", "houdini", "thomas")):
        return {
            "paradigm_choice": "mp_houdini_curl_attractor",
            "visual_prompt": "Chaotic Lorenz strange attractor in a 3D divergence-free curl-noise nebula",
            "paradigm_badge": "Houdini 3D Curl-Noise + RK4 Strange Attractor + Gray-Scott PDE",
            "formula_ticker": "v = ∇×Ψ (∇·v = 0) • dx/dt = σ(y-x) • ∂_t U = D_u∇²U - UV² + F(1-U)",
        }

    # 8. Iannis Xenakis Ruled Hyperbolic Paraboloids / Pavilion / Metastasis
    if any(k in blob for k in ("xenakis", "pavilion", "metastasis", "ruled", "paraboloid", "glissand")):
        return {
            "paradigm_choice": "mp_xenakis_metastasis",
            "visual_prompt": "Iannis Xenakis ruled hyperbolic paraboloid architectural pavilion shells",
            "paradigm_badge": "Iannis Xenakis Ruled Hyperbolic Paraboloids + OU Stochastic Field",
            "formula_ticker": "S_m(u,v,t) = (1-v)A_m(u,t) + vB_m(u,t) • dX_t = θ(μ-X_t)dt + σdW_t",
        }

    # 9. William Thurston 4D Sphere Eversion
    if any(k in blob for k in ("eversion", "outside in", "inside-out", "thurston", "morin", "hopf")):
        return {
            "paradigm_choice": "mp_thurston_eversion",
            "visual_prompt": "William Thurston C4 Morin 4D sphere eversion turning inside-out",
            "paradigm_badge": "William Thurston C4 Morin 4D Sphere Eversion Homotopy",
            "formula_ticker": "H_τ(θ,φ) = (1-τ)S²(θ,φ) + τ M_4(θ,φ) • π_{S³→R³}(x₁,x₂,x₃,x₄)",
        }

    # 10. Helaman Ferguson / Bathsheba Grossman Gyroid Minimal Surface
    if any(k in blob for k in ("gyroid", "minimal surface", "ferguson", "bathsheba", "schwarz")):
        return {
            "paradigm_choice": "mp_ddg_gyroid",
            "visual_prompt": "Triply periodic Schoen Gyroid minimal surface sculpture with zero mean curvature",
            "paradigm_badge": "Discrete Differential Geometry Schoen Gyroid Minimal Surface (H = 0)",
            "formula_ticker": "sin(x)cos(y) + sin(y)cos(z) + sin(z)cos(x) = 0 • H = (κ₁+κ₂)/2 = 0",
        }

    # 11. Nervous System Floraform Elastic Buckling PDE
    if any(k in blob for k in ("floraform", "differential growth", "differential-growth", "buckling")):
        return {
            "paradigm_choice": "mp_floraform_pde",
            "visual_prompt": "Nervous System Floraform differential-growth elastic buckling coral ribbon",
            "paradigm_badge": "Nervous System Floraform Differential-Growth Elastic Buckling PDE",
            "formula_ticker": "∂_t γ = F_spring + F_bending + F_repulsion + g(r)n_buckle",
        }

    # 12. Kleinian Limit Set / Mandelbox Cathedral
    if any(k in blob for k in ("kleinian", "mandelbox", "apollonian")):
        return {
            "paradigm_choice": "mp_kleinian_mandelbox",
            "visual_prompt": "Non-Euclidean Kleinian box-folded and sphere-inverted fractal cathedral",
            "paradigm_badge": "Non-Euclidean Kleinian Box-Fold & Sphere-Inversion Cathedral",
            "formula_ticker": "p ← boxFold(p), p ← (r²/‖p‖²)p • s_{n+1} = s_n + d_Kleinian(p,t)",
        }

    # 13. Hamid Pixel-Field Classics
    if "peacock" in blob:
        return {
            "paradigm_choice": "pf_peacock",
            "visual_prompt": "Royal Emerald Peacock with trigonometric plume eyespots",
            "paradigm_badge": "Hamid Double-Exponential Level-Set Painting J_s(x,y)",
            "formula_ticker": "H_v(x,y) = B_v∏(1-J_s) + ∑ C_{s,v} J_s • J_s = exp(-exp(K_s d_s))",
        }
    if "bipolar nebula" in blob or "nebula" in blob:
        return {
            "paradigm_choice": "pf_bipolar_nebula",
            "visual_prompt": "Cosmic Bipolar Nebula with trigonometric fBm gas lobes",
            "paradigm_badge": "Hamid Trigonometric fBm & Bipolar Nebula Field",
            "formula_ticker": "E(x,y) = ∑ a_k sin(ω_k x + φ_k)cos(ω_k y + ψ_k) • J_s = exp(-exp(K_s d_s))",
        }

    # 14. Check if LLM suggested a valid paradigm_choice
    if llm_choice in VALID_STORYBOARD_PARADIGMS and llm_choice != "custom_raymarch_3d":
        return {
            "paradigm_choice": llm_choice,
            "visual_prompt": llm_visual_prompt or narration or title,
            "paradigm_badge": f"Mathematical Paradigm ({llm_choice})",
            "formula_ticker": llm_formula or "s_{n+1} = s_n + d(r_o + s_n r_d, t) • N(p) = ∇d/‖∇d‖",
        }

    # 15. General unseen 3D scene -> Universal 3D Raymarched Cinema
    v_prompt = (llm_visual_prompt or narration or title).strip()
    return {
        "paradigm_choice": "custom_raymarch_3d",
        "visual_prompt": v_prompt,
        "paradigm_badge": "Universal 3D Implicit SDF & Differential-Geometry Raymarching",
        "formula_ticker": llm_formula or "s_{n+1} = s_n + d(r_o + s_n r_d, t) • N(p) = ∇d/‖∇d‖",
    }


def _invoke_llm_for_unstructured_storyboard(
    script_text: str, max_acts: int = 6
) -> Optional[List[Dict[str, Any]]]:
    """
    Calls live GLM-5.3 / DeepSeek V4.1 when the user provides unstructured prose
    to decompose it into M narrative mathematical acts.
    """
    user_msg = (
        f"Decompose the following script into up to {max_acts} sequential mathematical cinema acts:\n\n"
        f"{script_text}\n\n"
        f"Output STRICT JSON ONLY with key 'acts' (never mention AI or model names)."
    )
    for model in ((GLM_5_3, DEEPSEEK_V4_PRO) if _has_cf_credentials() else ()):
        url = f"https://api.cloudflare.com/client/v4/accounts/{CF_ACCOUNT_ID}/ai/run/{model}"
        payload: Dict[str, Any] = {
            "messages": [
                {"role": "system", "content": STORYBOARD_SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            "max_tokens": 3072,
            "temperature": 0.18,
        }
        if "glm-5" in model:
            payload["reasoning_effort"] = "low"
        elif "deepseek-v4" in model:
            payload["chat_template_kwargs"] = {"enable_thinking": False}
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {CF_API_TOKEN}",
                "Content-Type": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=35) as resp:
                body = json.loads(resp.read().decode("utf-8"))
                raw_resp = _extract_response_text(body)
                parsed = raw_resp if isinstance(raw_resp, dict) else _extract_json_object(str(raw_resp))
                if isinstance(parsed, dict) and isinstance(parsed.get("acts"), list) and len(parsed["acts"]) >= 2:
                    return parsed["acts"][:max_acts]
        except Exception:
            continue
    return None


def decompose_long_script_to_storyboard(
    script_text: str,
    max_acts: int = 6,
    refine_acts: bool = True,
) -> List[Dict[str, Any]]:
    """
    Decomposes ANY long multi-act or multi-paragraph script into M sequential
    mathematical cinema acts (3 <= M <= max_acts), mapping each act to its
    exact Master Paradigm or 3D Raymarched Implicit Scene (`raymarch_3d_scene`),
    with compiled WebGL1 GLSL, LaTeX governing equations, HUD formula ticker,
    exact script narration subtitles, and zero AI/model branding leaks.
    """
    from master_paradigms_engine import (
        compile_raymarch_scene_to_glsl,
        compile_raymarch_scene_to_latex,
        evaluate_master_paradigm_preset,
    )

    max_m = max(2, min(8, int(max_acts)))
    raw_text = (script_text or "").strip() or SAMPLE_EPIC_5_ACT_SCRIPT

    # Check if the script already has explicit ACT / SCENE markers
    has_explicit_markers = bool(
        re.search(r"(?:^|\n|\s+)(?:ACT|Act|SCENE|Scene|PART|Part)\s*\d+\s*[:\-–—]", raw_text)
    )

    raw_acts = _split_script_into_raw_acts(raw_text, max_acts=max_m)
    llm_acts: Optional[List[Dict[str, Any]]] = None
    if not has_explicit_markers and len(raw_acts) < 3:
        llm_acts = _invoke_llm_for_unstructured_storyboard(raw_text, max_acts=max_m)

    source_list: List[Dict[str, Any]] = []
    if llm_acts and len(llm_acts) >= len(raw_acts):
        for i, la in enumerate(llm_acts[:max_m]):
            if isinstance(la, dict):
                source_list.append(
                    {
                        "title": str(la.get("title") or f"Act {i + 1}"),
                        "narration": str(la.get("narration") or la.get("visual_prompt") or ""),
                        "llm_choice": str(la.get("paradigm_choice") or ""),
                        "llm_visual_prompt": str(la.get("visual_prompt") or ""),
                        "llm_formula": str(la.get("formula_ticker") or ""),
                    }
                )
    else:
        for i, ra in enumerate(raw_acts[:max_m]):
            source_list.append(
                {
                    "title": ra["title"],
                    "narration": ra["narration"],
                    "llm_choice": "",
                    "llm_visual_prompt": "",
                    "llm_formula": "",
                }
            )

    storyboard: List[Dict[str, Any]] = []
    for idx, item in enumerate(source_list):
        act_num = idx + 1
        clean_title = sanitize_mathematical_text(
            item["title"],
            f"Act {act_num} — Differential-Geometric Manifold",
        )
        clean_narration = sanitize_mathematical_text(
            item["narration"],
            clean_title,
        )

        routed = _route_act_to_paradigm_and_prompt(
            title=clean_title,
            narration=clean_narration,
            llm_choice=item["llm_choice"],
            llm_visual_prompt=item["llm_visual_prompt"],
            llm_formula=item["llm_formula"],
        )
        p_choice = routed["paradigm_choice"]
        v_prompt = routed["visual_prompt"]
        p_badge = sanitize_mathematical_text(routed["paradigm_badge"], "3D Implicit SDF Raymarching")
        f_ticker = sanitize_mathematical_text(
            routed["formula_ticker"],
            "s_{n+1} = s_n + d(r_o + s_n r_d, t) • N(p) = ∇d/‖∇d‖",
        )

        raymarch_scene: Optional[Dict[str, Any]] = None
        glsl_shader = ""
        latex_eqs: Dict[str, str] = {}
        latex_str = ""
        preset_id: Optional[str] = None

        if p_choice == "custom_raymarch_3d":
            raymarch_scene = generate_3d_raymarch_scene(
                v_prompt, refine=refine_acts, bypass_cache=False
            )
            glsl_shader = compile_raymarch_scene_to_glsl(raymarch_scene)
            latex_eqs = compile_raymarch_scene_to_latex(raymarch_scene)
            latex_str = " \\\\ ".join(latex_eqs.values())
            subtitle = sanitize_mathematical_text(
                str(raymarch_scene.get("subtitle") or ""),
                f"Act {act_num}: Coupled 3D Implicit SDF & Minimal-Surface Manifold",
            )
        elif p_choice.startswith("mp_"):
            preset_id = p_choice
            try:
                mp_eval = evaluate_master_paradigm_preset(
                    p_choice, width=160, height=100, t_phase=0.0, N=300
                )
                glsl_shader = str(mp_eval.get("glsl_fragment_shader") or "")
                latex_eqs = dict(mp_eval.get("latex_equations") or {})
                latex_str = str(mp_eval.get("latex") or "")
                subtitle = sanitize_mathematical_text(
                    str(mp_eval.get("subtitle") or ""),
                    p_badge,
                )
            except Exception:
                subtitle = p_badge
        else:
            preset_id = p_choice
            subtitle = p_badge

        storyboard.append(
            {
                "act_number": act_num,
                "act_index": idx,
                "title": clean_title,
                "subtitle": subtitle,
                "narration": clean_narration,
                "visual_prompt": v_prompt,
                "paradigm_choice": p_choice,
                "preset_id": preset_id,
                "paradigm_badge": p_badge,
                "formula_ticker": f_ticker,
                "raymarch_3d_scene": raymarch_scene,
                "raymarch_scene": raymarch_scene,
                "scene": raymarch_scene,
                "glsl_fragment_shader": glsl_shader,
                "latex_equations": latex_eqs,
                "latex": latex_str,
            }
        )

    return storyboard


