"""
Unified Pure-Math Expression AST Compiler for DesignWithMath.

Compiles a canonical symbolic `math_program` specification into four
deterministic pure-mathematical outputs:
  1. Vectorized NumPy RGB frames at arbitrary continuous time phase t
  2. WebGL1 GLSL ES 1.00 fragment shader for real-time 60 FPS GPU rendering
  3. Authentic AST-derived KaTeX LaTeX equations
  4. Crisp multi-threshold SDF iso-contour chords, 3D relief manifolds, and
     4D motion harmonics, plus standalone looping animated GIF / MP4 videos.
"""

import ast
import base64
import io
import math
import os
import re
from typing import Any, Dict, List, Optional, Set, Tuple

import cv2
import numpy as np
from PIL import Image


MAX_STEPS = 90
MAX_LOOP_ITERS = 60
MAX_LOOPS = 8

_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_FORBIDDEN_BRAND_RE = re.compile(
    r"\b(ai|cloudflare|deepseek|glm|kimi|gpt-?oss|qwq|llama|qwen|70b|120b|32b|17b|8b|llm|workers\s*ai|vision\s*critic|vlm)\b",
    flags=re.IGNORECASE,
)

# Exact whitelisted pure-math functions with valid arity sets
# (-1 means variadic >= 2)
WHITELISTED_FUNCS: Dict[str, Set[int]] = {
    "sin": {1},
    "cos": {1},
    "tan": {1},
    "asin": {1},
    "acos": {1},
    "atan": {1, 2},
    "atan2": {2},
    "exp": {1},
    "log": {1},
    "sqrt": {1},
    "abs": {1},
    "pow": {2},
    "min": {-1},
    "max": {-1},
    "clamp": {3},
    "floor": {1},
    "fract": {1},
    "mod": {2},
    "sign": {1},
    "mix": {3},
    "smoothstep": {3},
    "step": {2},
    # Hamid & Analytical SDF primitives
    "ee": {1},
    "sdCircle": {3},
    "sdEllipse": {4},
    "sdBox": {4},
    "sdSegment": {6},
    "sdCapsule": {7},
    "sdTriangle": {8},
    "sdRing": {4},
    "sdStar": {5},
    "smin": {3},
    "smax": {3},
    "rotX": {3},
    "rotY": {3},
    "fbm": {2},
    "fbmT": {3},
    "hash21": {2},
}

BUILTIN_CONSTS = {"PI": math.pi, "pi": math.pi, "TAU": 2.0 * math.pi, "tau": 2.0 * math.pi}
BASE_VARS = {"x", "y", "t", "PI", "pi", "TAU", "tau"}


def sanitize_math_label(text: str, fallback: str = "Symbolic Mathematical Painting") -> str:
    if not isinstance(text, str) or not text.strip():
        return fallback
    cleaned = _FORBIDDEN_BRAND_RE.sub("", text)
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip(" -—|:,;")
    return cleaned if cleaned else fallback


def _preprocess_expr(expr: str) -> str:
    """Normalizes minor notation variants (^ -> **, np./math. prefixes) before AST parse."""
    s = str(expr).strip()
    s = s.replace("^", "**")
    s = re.sub(r"\b(?:np|numpy|math)\.", "", s)
    return s


def _parse_expr_ast(expr: str) -> ast.AST:
    if not isinstance(expr, str) or not expr.strip():
        raise ValueError("Expression must be a non-empty string")
    return ast.parse(_preprocess_expr(expr), mode="eval")


# ============================================================================
# 1. IDENTICAL NUMPY IMPLEMENTATIONS OF ALL WHITELISTED PURE-MATH FUNCTIONS
# ============================================================================

def _np_ee(u: Any) -> np.ndarray:
    return np.exp(-np.exp(np.clip(u, -30.0, 30.0))).astype(np.float32)


def _np_safe_div(a: Any, b: Any) -> Any:
    if isinstance(b, np.ndarray):
        safe_b = np.where(np.abs(b) < 1e-6, np.where(b >= 0.0, 1e-6, -1e-6), b)
        return a / safe_b
    bf = float(b)
    if abs(bf) < 1e-6:
        bf = 1e-6 if bf >= 0.0 else -1e-6
    return a / bf


def _np_safe_pow(a: Any, b: Any) -> Any:
    if isinstance(b, (int, float)) and not isinstance(b, bool):
        bf = float(np.clip(b, -8.0, 8.0))
        if abs(bf - round(bf)) < 1e-6:
            ip = int(round(bf))
            if ip == 2:
                return a * a
            if ip == 3:
                return a * a * a
            if ip == 4:
                a2 = a * a
                return a2 * a2
            if ip >= 0:
                return a ** ip
            return _np_safe_div(1.0, a ** (-ip))
        return np.maximum(np.abs(a), 1e-6) ** bf
    return np.maximum(np.abs(a), 1e-6) ** np.clip(b, -8.0, 8.0)


def _np_fract(u: Any) -> Any:
    return u - np.floor(u)


def _np_mod(a: Any, b: Any) -> Any:
    safe_b = np.where(np.abs(b) < 1e-6, 1e-6, b) if isinstance(b, np.ndarray) else (b if abs(b) >= 1e-6 else 1e-6)
    return a - safe_b * np.floor(a / safe_b)


def _np_mix(a: Any, b: Any, f: Any) -> Any:
    return a * (1.0 - f) + b * f


def _np_step(edge: Any, x: Any) -> Any:
    return np.where(x < edge, np.float32(0.0), np.float32(1.0))


def _np_smoothstep(e0: Any, e1: Any, x: Any) -> Any:
    span = e1 - e0
    if isinstance(span, np.ndarray):
        safe_span = np.where(np.abs(span) < 1e-6, 1e-6, span)
        u = np.clip((x - e0) / safe_span, 0.0, 1.0)
        res = u * u * (3.0 - 2.0 * u)
        return np.where(np.abs(span) < 1e-6, _np_step(e0, x), res)
    if abs(float(span)) < 1e-6:
        return _np_step(e0, x)
    u = np.clip((x - e0) / float(span), 0.0, 1.0)
    return u * u * (3.0 - 2.0 * u)


def _np_sd_circle(px: Any, py: Any, r: Any) -> Any:
    return np.sqrt(px * px + py * py) - r


def _np_sd_ellipse(px: Any, py: Any, rx: Any, ry: Any) -> Any:
    rx_s = np.maximum(np.abs(rx), 1e-4)
    ry_s = np.maximum(np.abs(ry), 1e-4)
    ux = px / rx_s
    uy = py / ry_s
    return (np.sqrt(ux * ux + uy * uy) - 1.0) * np.minimum(rx_s, ry_s)


def _np_sd_box(px: Any, py: Any, bx: Any, by: Any) -> Any:
    dx = np.abs(px) - bx
    dy = np.abs(py) - by
    ox = np.maximum(dx, 0.0)
    oy = np.maximum(dy, 0.0)
    return np.sqrt(ox * ox + oy * oy) + np.minimum(np.maximum(dx, dy), 0.0)


def _np_sd_segment(px: Any, py: Any, ax: Any, ay: Any, bx: Any, by: Any) -> Any:
    pax = px - ax
    pay = py - ay
    bax = bx - ax
    bay = by - ay
    denom = np.maximum(bax * bax + bay * bay, 1e-8)
    h = np.clip((pax * bax + pay * bay) / denom, 0.0, 1.0)
    dx = pax - bax * h
    dy = pay - bay * h
    return np.sqrt(dx * dx + dy * dy)


def _np_sd_capsule(px: Any, py: Any, ax: Any, ay: Any, bx: Any, by: Any, r: Any) -> Any:
    return _np_sd_segment(px, py, ax, ay, bx, by) - r


def _np_sd_triangle(px: Any, py: Any, ax: Any, ay: Any, bx: Any, by: Any, cx: Any, cy: Any) -> Any:
    e0x, e0y = bx - ax, by - ay
    e1x, e1y = cx - bx, cy - by
    e2x, e2y = ax - cx, ay - cy
    v0x, v0y = px - ax, py - ay
    v1x, v1y = px - bx, py - by
    v2x, v2y = px - cx, py - cy

    t0 = np.clip((v0x * e0x + v0y * e0y) / np.maximum(e0x * e0x + e0y * e0y, 1e-8), 0.0, 1.0)
    t1 = np.clip((v1x * e1x + v1y * e1y) / np.maximum(e1x * e1x + e1y * e1y, 1e-8), 0.0, 1.0)
    t2 = np.clip((v2x * e2x + v2y * e2y) / np.maximum(e2x * e2x + e2y * e2y, 1e-8), 0.0, 1.0)

    pq0x, pq0y = v0x - e0x * t0, v0y - e0y * t0
    pq1x, pq1y = v1x - e1x * t1, v1y - e1y * t1
    pq2x, pq2y = v2x - e2x * t2, v2y - e2y * t2

    s = np.sign(e0x * e2y - e0y * e2x)
    s = np.where(s == 0.0, 1.0, s)

    d0 = pq0x * pq0x + pq0y * pq0y
    d1 = pq1x * pq1x + pq1y * pq1y
    d2 = pq2x * pq2x + pq2y * pq2y
    c0 = s * (v0x * e0y - v0y * e0x)
    c1 = s * (v1x * e1y - v1y * e1x)
    c2 = s * (v2x * e2y - v2y * e2x)

    dist2 = np.minimum(np.minimum(d0, d1), d2)
    winding = np.minimum(np.minimum(c0, c1), c2)
    sgn = np.where(winding < 0.0, -1.0, 1.0)
    return -np.sqrt(np.maximum(dist2, 0.0)) * sgn


def _np_sd_ring(px: Any, py: Any, r: Any, w: Any) -> Any:
    return np.abs(np.sqrt(px * px + py * py) - r) - w


def _np_sd_star(px: Any, py: Any, r: Any, n: Any, m: Any) -> Any:
    rad = np.sqrt(px * px + py * py)
    ang = np.arctan2(py, px)
    pts = np.maximum(2.0, n)
    depth = np.clip(m, 0.0, 0.92)
    r_edge = r * (1.0 - depth * 0.5 + depth * 0.5 * np.cos(pts * ang))
    return rad - r_edge


def _np_smin(a: Any, b: Any, k: Any) -> Any:
    ks = np.maximum(k, 1e-5)
    h = np.maximum(ks - np.abs(a - b), 0.0) / ks
    return np.minimum(a, b) - h * h * ks * 0.25


def _np_smax(a: Any, b: Any, k: Any) -> Any:
    return -_np_smin(-a, -b, k)


def _np_rot_x(px: Any, py: Any, ang: Any) -> Any:
    return px * np.cos(ang) - py * np.sin(ang)


def _np_rot_y(px: Any, py: Any, ang: Any) -> Any:
    return px * np.sin(ang) + py * np.cos(ang)


def _np_fbm(px: Any, py: Any) -> Any:
    acc = np.float32(0.0)
    amp = 0.50
    freq = 2.4
    for i in range(1, 7):
        fi = float(i)
        a2 = 2.0 * fi * fi
        ca, sa = math.cos(a2), math.sin(a2)
        u1 = ca * px + sa * py
        v1 = ca * py - sa * px
        ph = 2.0 * math.cos(5.0 * fi)
        acc = acc + amp * np.cos(freq * u1 + 1.4 * np.cos(freq * v1) + ph) * np.cos(freq * v1 + 1.4 * np.sin(freq * u1) + ph)
        amp *= 0.62
        freq *= 1.55
    return acc


def _np_fbm_t(px: Any, py: Any, t: Any) -> Any:
    acc = np.float32(0.0)
    amp = 0.50
    freq = 2.4
    ox = 0.22 * np.cos(t)
    oy = 0.22 * np.sin(t)
    for i in range(1, 7):
        fi = float(i)
        a2 = 2.0 * fi * fi
        ca, sa = math.cos(a2), math.sin(a2)
        qx = px + ox / fi
        qy = py + oy / fi
        u1 = ca * qx + sa * qy
        v1 = ca * qy - sa * qx
        ph = 2.0 * math.cos(5.0 * fi) + np.sin(t + fi)
        acc = acc + amp * np.cos(freq * u1 + 1.4 * np.cos(freq * v1) + ph) * np.cos(freq * v1 + 1.4 * np.sin(freq * u1) + ph)
        amp *= 0.62
        freq *= 1.55
    return acc


def _np_hash21(px: Any, py: Any) -> Any:
    return _np_fract(np.sin(px * 127.1 + py * 311.7) * 43758.5453)


def _np_min(*args: Any) -> Any:
    res = args[0]
    for a in args[1:]:
        res = np.minimum(res, a)
    return res


def _np_max(*args: Any) -> Any:
    res = args[0]
    for a in args[1:]:
        res = np.maximum(res, a)
    return res


def _np_atan(*args: Any) -> Any:
    if len(args) == 2:
        return np.arctan2(args[0], args[1])
    return np.arctan(args[0])


_NP_FUNC_MAP = {
    "sin": np.sin,
    "cos": np.cos,
    "tan": lambda a: np.tan(np.clip(a, -1.55, 1.55)),
    "asin": lambda a: np.arcsin(np.clip(a, -1.0, 1.0)),
    "acos": lambda a: np.arccos(np.clip(a, -1.0, 1.0)),
    "atan": _np_atan,
    "atan2": np.arctan2,
    "exp": lambda a: np.exp(np.clip(a, -35.0, 35.0)),
    "log": lambda a: np.log(np.maximum(np.abs(a), 1e-6)),
    "sqrt": lambda a: np.sqrt(np.maximum(a, 0.0)),
    "abs": np.abs,
    "pow": _np_safe_pow,
    "min": _np_min,
    "max": _np_max,
    "clamp": np.clip,
    "floor": np.floor,
    "fract": _np_fract,
    "mod": _np_mod,
    "sign": np.sign,
    "mix": _np_mix,
    "smoothstep": _np_smoothstep,
    "step": _np_step,
    "ee": _np_ee,
    "sdCircle": _np_sd_circle,
    "sdEllipse": _np_sd_ellipse,
    "sdBox": _np_sd_box,
    "sdSegment": _np_sd_segment,
    "sdCapsule": _np_sd_capsule,
    "sdTriangle": _np_sd_triangle,
    "sdRing": _np_sd_ring,
    "sdStar": _np_sd_star,
    "smin": _np_smin,
    "smax": _np_smax,
    "rotX": _np_rot_x,
    "rotY": _np_rot_y,
    "fbm": _np_fbm,
    "fbmT": _np_fbm_t,
    "hash21": _np_hash21,
}


# ============================================================================
# 2. AST VALIDATION
# ============================================================================

def _validate_expr_ast(expr: str, defined_vars: Set[str], context: str) -> None:
    try:
        tree = _parse_expr_ast(expr)
    except SyntaxError as exc:
        raise ValueError(f"Syntax error in {context}: {exc.msg}") from exc

    def _walk(node: ast.AST) -> None:
        if isinstance(node, ast.Expression):
            _walk(node.body)
        elif isinstance(node, ast.Constant):
            if not isinstance(node.value, (int, float)) or isinstance(node.value, bool):
                raise ValueError(f"Unsupported constant in {context}: {node.value!r}")
            if math.isnan(float(node.value)) or math.isinf(float(node.value)):
                raise ValueError(f"Non-finite constant in {context}")
        elif isinstance(node, ast.Name):
            if node.id not in defined_vars:
                raise ValueError(f"Undefined variable '{node.id}' in {context}")
        elif isinstance(node, ast.UnaryOp):
            if not isinstance(node.op, (ast.UAdd, ast.USub)):
                raise ValueError(f"Unsupported unary operator in {context}")
            _walk(node.operand)
        elif isinstance(node, ast.BinOp):
            if not isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod)):
                raise ValueError(f"Unsupported binary operator in {context}")
            _walk(node.left)
            _walk(node.right)
        elif isinstance(node, ast.Compare):
            if len(node.ops) != 1 or not isinstance(node.ops[0], (ast.Lt, ast.LtE, ast.Gt, ast.GtE)):
                raise ValueError(f"Unsupported comparison in {context}")
            _walk(node.left)
            _walk(node.comparators[0])
        elif isinstance(node, ast.IfExp):
            _walk(node.test)
            _walk(node.body)
            _walk(node.orelse)
        elif isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise ValueError(f"Disallowed call target in {context}")
            fname = node.func.id
            if fname not in WHITELISTED_FUNCS:
                raise ValueError(f"Function '{fname}' is not whitelisted in {context}")
            if node.keywords:
                raise ValueError(f"Keyword arguments not allowed in '{fname}' ({context})")
            allowed = WHITELISTED_FUNCS[fname]
            n_args = len(node.args)
            if -1 in allowed:
                if n_args < 2:
                    raise ValueError(f"Function '{fname}' expects at least 2 args in {context}")
            elif n_args not in allowed:
                raise ValueError(f"Function '{fname}' expects {sorted(allowed)} args, got {n_args} in {context}")
            for a in node.args:
                _walk(a)
        else:
            raise ValueError(f"Disallowed AST node '{type(node).__name__}' in {context}")

    _walk(tree)


def _check_ident(name: Any, context: str) -> str:
    if not isinstance(name, str) or not _IDENT_RE.match(name):
        raise ValueError(f"Invalid identifier {name!r} in {context}")
    if name in WHITELISTED_FUNCS or name in BUILTIN_CONSTS or name in ("x", "y", "t"):
        raise ValueError(f"Identifier '{name}' conflicts with built-in symbol in {context}")
    return name


def _extract_expr_var_refs(expr: str) -> Set[str]:
    """Returns all variable names referenced on the RHS of a validated expression."""
    tree = _parse_expr_ast(expr)
    refs: Set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            refs.add(node.id)
    return refs


def validate_math_program(prog: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Strictly validates a canonical `math_program` dictionary via AST-only
    static analysis. Checks step count <= 90, loop iterations <= 60, valid
    whitelisted functions, no undefined variables, no dead/unused loop outputs,
    and float safety.
    """
    if not isinstance(prog, dict):
        return False, "math_program must be a JSON object"

    steps = prog.get("steps")
    if not isinstance(steps, list):
        return False, "'steps' must be a list"
    if len(steps) > MAX_STEPS:
        return False, f"Too many steps ({len(steps)} > {MAX_STEPS})"

    for ch in ("H0", "H1", "H2"):
        if not isinstance(prog.get(ch), str) or not prog[ch].strip():
            return False, f"Missing or empty output expression '{ch}'"

    domain = prog.get("domain")
    if domain is not None:
        if not isinstance(domain, dict):
            return False, "'domain' must be a dict"
        try:
            xmin = float(domain.get("x_min", -1.6))
            xmax = float(domain.get("x_max", 1.6))
            ymin = float(domain.get("y_min", -1.0))
            ymax = float(domain.get("y_max", 1.0))
            if not (xmax > xmin and ymax > ymin):
                return False, "Invalid domain bounds (max must exceed min)"
        except (TypeError, ValueError):
            return False, "Non-numeric domain bounds"

    defined_vars: Set[str] = set(BASE_VARS)
    loop_count = 0
    step_refs: List[Tuple[Optional[str], Set[str], bool]] = []  # (written_var, read_vars, is_loop)

    try:
        for i, step in enumerate(steps):
            if not isinstance(step, dict):
                return False, f"Step {i} must be an object"
            if "loop" in step:
                loop_count += 1
                if loop_count > MAX_LOOPS:
                    return False, f"Too many loops ({loop_count} > {MAX_LOOPS})"
                lp = step["loop"]
                if not isinstance(lp, dict):
                    return False, f"Step {i} 'loop' must be an object"
                lvar = _check_ident(lp.get("var"), f"step[{i}].loop.var")
                liter = _check_ident(lp.get("iter"), f"step[{i}].loop.iter")
                if liter == lvar:
                    return False, f"Loop iter '{liter}' cannot equal accumulator var '{lvar}' in step {i}"
                try:
                    start = int(lp.get("start", 1))
                    end = int(lp.get("end", 1))
                except (TypeError, ValueError):
                    return False, f"Loop start/end must be integers in step {i}"
                if end < start:
                    return False, f"Loop end ({end}) < start ({start}) in step {i}"
                if (end - start + 1) > MAX_LOOP_ITERS:
                    return False, f"Loop iterations ({end - start + 1}) exceed max {MAX_LOOP_ITERS} in step {i}"

                init_expr = str(lp.get("init", "0.0"))
                _validate_expr_ast(init_expr, defined_vars, f"step[{i}].loop.init")
                reads: Set[str] = set(_extract_expr_var_refs(init_expr))

                body_vars = set(defined_vars) | {liter, "acc", lvar}
                local_bound: Set[str] = {liter, "acc", lvar}
                body = lp.get("body", [])
                if not isinstance(body, list) or len(body) > 24:
                    return False, f"Invalid loop body in step {i}"
                for j, bstep in enumerate(body):
                    if not isinstance(bstep, dict):
                        return False, f"step[{i}].loop.body[{j}] must be an object"
                    bvar = _check_ident(bstep.get("var"), f"step[{i}].loop.body[{j}].var")
                    if bvar in (liter, "acc"):
                        return False, f"Cannot overwrite '{bvar}' inside loop body in step {i}"
                    bexpr = str(bstep.get("expr", ""))
                    _validate_expr_ast(bexpr, body_vars, f"step[{i}].loop.body[{j}].expr")
                    reads.update(_extract_expr_var_refs(bexpr) - local_bound)
                    body_vars.add(bvar)
                    local_bound.add(bvar)

                update_expr = str(lp.get("update", ""))
                _validate_expr_ast(update_expr, body_vars, f"step[{i}].loop.update")
                reads.update(_extract_expr_var_refs(update_expr) - local_bound)
                defined_vars.add(lvar)
                step_refs.append((lvar, reads, True))
            else:
                vname = _check_ident(step.get("var"), f"step[{i}].var")
                sexpr = str(step.get("expr", ""))
                _validate_expr_ast(sexpr, defined_vars, f"step[{i}].expr({vname})")
                defined_vars.add(vname)
                step_refs.append((vname, _extract_expr_var_refs(sexpr), False))

        out_refs: Set[str] = set()
        for ch in ("H0", "H1", "H2"):
            _validate_expr_ast(prog[ch], defined_vars, f"output.{ch}")
            out_refs.update(_extract_expr_var_refs(prog[ch]))

        # Reject dead loops whose output variable is never read in subsequent steps or H0/H1/H2
        for i, (written_var, _, is_loop) in enumerate(step_refs):
            if not is_loop or written_var is None:
                continue
            used = False
            overwritten = False
            for next_written, next_reads, _ in step_refs[i + 1:]:
                if written_var in next_reads:
                    used = True
                    break
                if next_written == written_var:
                    overwritten = True
                    break
            if not used and not overwritten and written_var in out_refs:
                used = True
            if not used:
                return (
                    False,
                    f"Dead loop in step {i}: loop output var '{written_var}' is never referenced in subsequent steps or H0/H1/H2",
                )
    except ValueError as err:
        return False, str(err)

    return True, "ok"


# ============================================================================
# 3. VECTORIZED NUMPY EVALUATOR & HAMID QUANTIZER
# ============================================================================

def _hamid_F_quantize(h: np.ndarray) -> np.ndarray:
    step_0 = _np_ee(-1000.0 * h)
    exp_1 = _np_ee(1000.0 * (h - 1.0))
    val = 255.0 * step_0 * (np.maximum(np.abs(h), 1e-8) ** exp_1)
    return np.floor(np.clip(val, 0.0, 255.0)).astype(np.uint8)


def _encode_rgb_png_base64(rgb: np.ndarray) -> str:
    pil_img = Image.fromarray(rgb, mode="RGB")
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _eval_ast_numpy(tree: ast.AST, env: Dict[str, Any]) -> Any:
    def _ev(node: ast.AST) -> Any:
        if isinstance(node, ast.Expression):
            return _ev(node.body)
        if isinstance(node, ast.Constant):
            return np.float32(node.value)
        if isinstance(node, ast.Name):
            return env[node.id]
        if isinstance(node, ast.UnaryOp):
            val = _ev(node.operand)
            return val if isinstance(node.op, ast.UAdd) else -val
        if isinstance(node, ast.BinOp):
            left = _ev(node.left)
            right = _ev(node.right)
            op = node.op
            if isinstance(op, ast.Add):
                return left + right
            if isinstance(op, ast.Sub):
                return left - right
            if isinstance(op, ast.Mult):
                return left * right
            if isinstance(op, ast.Div):
                return _np_safe_div(left, right)
            if isinstance(op, ast.Mod):
                return _np_mod(left, right)
            if isinstance(op, ast.Pow):
                return _np_safe_pow(left, right)
        if isinstance(node, ast.Compare):
            left = _ev(node.left)
            right = _ev(node.comparators[0])
            op = node.ops[0]
            if isinstance(op, ast.Lt):
                return left < right
            if isinstance(op, ast.LtE):
                return left <= right
            if isinstance(op, ast.Gt):
                return left > right
            if isinstance(op, ast.GtE):
                return left >= right
        if isinstance(node, ast.IfExp):
            cond = _ev(node.test)
            return np.where(cond, _ev(node.body), _ev(node.orelse))
        if isinstance(node, ast.Call):
            fn = _NP_FUNC_MAP[node.func.id]
            args = [_ev(a) for a in node.args]
            return fn(*args)
        raise ValueError(f"Unsupported AST node: {type(node).__name__}")

    return _ev(tree)


def _compute_image_stats(rgb: np.ndarray) -> Dict[str, float]:
    arr = rgb.astype(np.float32)
    std_val = float(np.std(arr))
    mean_val = float(np.mean(arr))
    chroma_val = float(np.mean(np.max(arr, axis=-1) - np.min(arr, axis=-1)))

    lum = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]
    gy, gx = np.gradient(lum)
    gmag = np.hypot(gx, gy)
    edge_density = float(np.mean(gmag > 14.0))

    h, w = lum.shape
    y0, y1 = int(h * 0.2), int(h * 0.8)
    x0, x1 = int(w * 0.2), int(w * 0.8)
    center_contrast = float(np.std(arr[y0:y1, x0:x1]))

    bw = max(2, int(min(h, w) * 0.10))
    border_pixels = np.concatenate([
        lum[:bw, :].ravel(),
        lum[-bw:, :].ravel(),
        lum[bw:-bw, :bw].ravel(),
        lum[bw:-bw, -bw:].ravel(),
    ])
    border_mean = float(np.mean(border_pixels)) if border_pixels.size > 0 else mean_val

    return {
        "std": round(std_val, 2),
        "mean": round(mean_val, 2),
        "chroma": round(chroma_val, 2),
        "edge_density": round(edge_density, 4),
        "center_contrast": round(center_contrast, 2),
        "border_mean": round(border_mean, 2),
    }


def _resolve_domain(prog: Dict[str, Any], width: int, height: int) -> Tuple[float, float, float, float]:
    dom = prog.get("domain") or {}
    if all(k in dom for k in ("x_min", "x_max", "y_min", "y_max")):
        return float(dom["x_min"]), float(dom["x_max"]), float(dom["y_min"]), float(dom["y_max"])
    aspect = float(width) / max(1.0, float(height))
    return -aspect, aspect, -1.0, 1.0


def _precompile_program_asts(prog: Dict[str, Any]) -> Tuple[List[Any], ast.AST, ast.AST, ast.AST]:
    compiled_steps: List[Any] = []
    for step in prog.get("steps", []):
        if "loop" in step:
            lp = step["loop"]
            compiled_steps.append((
                "loop",
                lp["var"],
                lp["iter"],
                int(lp.get("start", 1)),
                int(lp.get("end", 1)),
                _parse_expr_ast(str(lp.get("init", "0.0"))),
                [(b["var"], _parse_expr_ast(str(b["expr"]))) for b in lp.get("body", [])],
                _parse_expr_ast(str(lp["update"])),
            ))
        else:
            compiled_steps.append(("assign", step["var"], _parse_expr_ast(str(step["expr"]))))
    ast_h0 = _parse_expr_ast(prog["H0"])
    ast_h1 = _parse_expr_ast(prog["H1"])
    ast_h2 = _parse_expr_ast(prog["H2"])
    return compiled_steps, ast_h0, ast_h1, ast_h2


def _eval_compiled_frame(
    compiled_steps: List[Any],
    ast_h0: ast.AST,
    ast_h1: ast.AST,
    ast_h2: ast.AST,
    X: np.ndarray,
    Y: np.ndarray,
    t_phase: float,
    loop_diag: Optional[Dict[str, Tuple[float, float]]] = None,
) -> np.ndarray:
    env: Dict[str, Any] = {
        "x": X,
        "y": Y,
        "t": np.float32(t_phase),
        "PI": np.float32(math.pi),
        "pi": np.float32(math.pi),
        "TAU": np.float32(2.0 * math.pi),
        "tau": np.float32(2.0 * math.pi),
    }

    for item in compiled_steps:
        if item[0] == "assign":
            _, vname, tree = item
            val = _eval_ast_numpy(tree, env)
            if isinstance(val, np.ndarray):
                env[vname] = np.nan_to_num(val, nan=0.0, posinf=1e4, neginf=-1e4).astype(np.float32)
            else:
                env[vname] = np.float32(val)
        else:
            _, lvar, liter, start, end, init_tree, body_trees, update_tree = item
            acc_val = _eval_ast_numpy(init_tree, env)
            if not isinstance(acc_val, np.ndarray):
                acc_val = np.full_like(X, float(acc_val), dtype=np.float32)
            else:
                acc_val = np.nan_to_num(acc_val, nan=0.0, posinf=1e4, neginf=-1e4).astype(np.float32)

            loop_env = dict(env)
            for idx_val in range(start, end + 1):
                loop_env[liter] = np.float32(idx_val)
                loop_env["acc"] = acc_val
                loop_env[lvar] = acc_val
                for bvar, btree in body_trees:
                    bval = _eval_ast_numpy(btree, loop_env)
                    if isinstance(bval, np.ndarray):
                        loop_env[bvar] = np.nan_to_num(bval, nan=0.0, posinf=1e4, neginf=-1e4).astype(np.float32)
                    else:
                        loop_env[bvar] = np.float32(bval)
                upd = _eval_ast_numpy(update_tree, loop_env)
                if isinstance(upd, np.ndarray):
                    acc_val = np.nan_to_num(upd, nan=0.0, posinf=1e4, neginf=-1e4).astype(np.float32)
                else:
                    acc_val = np.full_like(X, float(upd), dtype=np.float32)
            env[lvar] = acc_val
            if loop_diag is not None:
                loop_diag[lvar] = (float(np.max(np.abs(acc_val))), float(np.std(acc_val)))

    def _to_grid(res: Any) -> np.ndarray:
        if isinstance(res, np.ndarray):
            arr = np.broadcast_to(res, X.shape)
            return np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0).astype(np.float32)
        return np.full_like(X, float(res), dtype=np.float32)

    H0 = _to_grid(_eval_ast_numpy(ast_h0, env))
    H1 = _to_grid(_eval_ast_numpy(ast_h1, env))
    H2 = _to_grid(_eval_ast_numpy(ast_h2, env))

    return np.stack([
        _hamid_F_quantize(H0),
        _hamid_F_quantize(H1),
        _hamid_F_quantize(H2),
    ], axis=-1)


def evaluate_math_program_numpy(
    prog: Dict[str, Any],
    width: int = 640,
    height: int = 400,
    t_phase: float = 0.0,
    include_raw_array: bool = False,
) -> Dict[str, Any]:
    """
    Validates and evaluates a canonical `math_program` in vectorized float32 NumPy
    at continuous time phase `t_phase`. Returns image_base64, diagnostics stats,
    AST-derived LaTeX equations, WebGL1 GLSL fragment shader, and Python script.
    """
    ok, err = validate_math_program(prog)
    if not ok:
        raise ValueError(f"Invalid math_program: {err}")

    W = max(32, int(width))
    H = max(32, int(height))
    x_min, x_max, y_min, y_max = _resolve_domain(prog, W, H)

    xs = np.linspace(x_min, x_max, W, dtype=np.float32)
    ys = np.linspace(y_max, y_min, H, dtype=np.float32)
    X, Y = np.meshgrid(xs, ys)

    compiled_steps, ast_h0, ast_h1, ast_h2 = _precompile_program_asts(prog)
    loop_diag: Dict[str, Tuple[float, float]] = {}
    rgb = _eval_compiled_frame(
        compiled_steps, ast_h0, ast_h1, ast_h2, X, Y, float(t_phase), loop_diag=loop_diag
    )

    stats: Dict[str, Any] = _compute_image_stats(rgb)
    stats["dead_loops"] = [
        lvar for lvar, (max_abs, std_val) in loop_diag.items()
        if max_abs < 1e-3 or std_val < 1e-4
    ]
    glsl = compile_math_program_to_glsl(prog)
    latex_eqs = compile_math_program_to_latex(prog)
    py_script = build_symbolic_python_script(prog)

    aspect_raw = prog.get("aspect", [16, 10])
    if isinstance(aspect_raw, (list, tuple)) and len(aspect_raw) == 2:
        aspect_pair = [int(aspect_raw[0]), int(aspect_raw[1])]
    else:
        aspect_pair = [16, 10]
    nom_w = 2000
    nom_h = max(400, int(round(nom_w * aspect_pair[1] / max(1, aspect_pair[0]))))

    title = sanitize_math_label(str(prog.get("title", "Symbolic Analytical Pixel-Field")))
    subtitle = sanitize_math_label(
        str(prog.get("subtitle", f"Direct Symbolic Double-Exponential Field — {len(prog.get('steps', []))} Coupled Equations"))
    )

    result: Dict[str, Any] = {
        "title": title,
        "subtitle": subtitle,
        "author": "Symbolic Double-Exponential Field Engine",
        "mode": "pixel_field",
        "N": H,
        "width": W,
        "height": H,
        "aspect": aspect_pair,
        "nominal_aspect": [nom_w, nom_h],
        "harmonics_count": f"{len(prog.get('steps', []))} Symbolic Equations",
        "image_base64": _encode_rgb_png_base64(rgb),
        "stats": stats,
        "latex": latex_eqs,
        "latex_equations": latex_eqs,
        "glsl_fragment_shader": glsl,
        "primitives": [],
        "python_code": py_script,
        "animation_capable": True,
    }
    if include_raw_array:
        result["rgb_array"] = rgb
    return result


# ============================================================================
# 4. WEBGL1 GLSL ES 1.00 FRAGMENT SHADER COMPILER (60 FPS GPU VIDEO)
# ============================================================================

_GLSL_HELPERS = """precision highp float;
uniform vec2 u_resolution;
uniform float u_time;
varying vec2 v_uv;

const float PI = 3.141592653589793;
const float TAU = 6.283185307179586;

float safeDiv(float a, float b) {
    float sb = abs(b) < 1e-6 ? (b >= 0.0 ? 1e-6 : -1e-6) : b;
    return a / sb;
}

float safePow(float a, float b) {
    return pow(max(abs(a), 1e-6), clamp(b, -8.0, 8.0));
}

float safeMod(float a, float b) {
    float sb = abs(b) < 1e-6 ? 1e-6 : b;
    return a - sb * floor(a / sb);
}

float safeTan(float a) {
    return tan(clamp(a, -1.55, 1.55));
}

float safeAsin(float a) {
    return asin(clamp(a, -1.0, 1.0));
}

float safeAcos(float a) {
    return acos(clamp(a, -1.0, 1.0));
}

float safeExp(float a) {
    return exp(clamp(a, -35.0, 35.0));
}

float safeLog(float a) {
    return log(max(abs(a), 1e-6));
}

float safeSqrt(float a) {
    return sqrt(max(a, 0.0));
}

float safeSmoothstep(float e0, float e1, float x) {
    float span = e1 - e0;
    if (abs(span) < 1e-6) return step(e0, x);
    float u = clamp((x - e0) / span, 0.0, 1.0);
    return u * u * (3.0 - 2.0 * u);
}

float ee(float u) {
    return exp(-exp(clamp(u, -30.0, 30.0)));
}

float sdCircle(float px, float py, float r) {
    return sqrt(px * px + py * py) - r;
}

float sdEllipse(float px, float py, float rx, float ry) {
    float rx_s = max(abs(rx), 1e-4);
    float ry_s = max(abs(ry), 1e-4);
    return (length(vec2(px / rx_s, py / ry_s)) - 1.0) * min(rx_s, ry_s);
}

float sdBox(float px, float py, float bx, float by) {
    vec2 d = abs(vec2(px, py)) - vec2(bx, by);
    return length(max(d, 0.0)) + min(max(d.x, d.y), 0.0);
}

float sdSegment(float px, float py, float ax, float ay, float bx, float by) {
    vec2 pa = vec2(px - ax, py - ay);
    vec2 ba = vec2(bx - ax, by - ay);
    float h = clamp(dot(pa, ba) / max(dot(ba, ba), 1e-8), 0.0, 1.0);
    return length(pa - ba * h);
}

float sdCapsule(float px, float py, float ax, float ay, float bx, float by, float r) {
    return sdSegment(px, py, ax, ay, bx, by) - r;
}

float sdTriangle(float px, float py, float ax, float ay, float bx, float by, float cx, float cy) {
    vec2 p = vec2(px, py);
    vec2 p0 = vec2(ax, ay);
    vec2 p1 = vec2(bx, by);
    vec2 p2 = vec2(cx, cy);
    vec2 e0 = p1 - p0;
    vec2 e1 = p2 - p1;
    vec2 e2 = p0 - p2;
    vec2 v0 = p - p0;
    vec2 v1 = p - p1;
    vec2 v2 = p - p2;
    vec2 pq0 = v0 - e0 * clamp(dot(v0, e0) / max(dot(e0, e0), 1e-8), 0.0, 1.0);
    vec2 pq1 = v1 - e1 * clamp(dot(v1, e1) / max(dot(e1, e1), 1e-8), 0.0, 1.0);
    vec2 pq2 = v2 - e2 * clamp(dot(v2, e2) / max(dot(e2, e2), 1e-8), 0.0, 1.0);
    float s = sign(e0.x * e2.y - e0.y * e2.x);
    if (s == 0.0) s = 1.0;
    vec2 d = min(min(
        vec2(dot(pq0, pq0), s * (v0.x * e0.y - v0.y * e0.x)),
        vec2(dot(pq1, pq1), s * (v1.x * e1.y - v1.y * e1.x))),
        vec2(dot(pq2, pq2), s * (v2.x * e2.y - v2.y * e2.x)));
    float sgn = d.y < 0.0 ? -1.0 : 1.0;
    return -sqrt(max(d.x, 0.0)) * sgn;
}

float sdRing(float px, float py, float r, float w) {
    return abs(sqrt(px * px + py * py) - r) - w;
}

float sdStar(float px, float py, float r, float n, float m) {
    float rad = sqrt(px * px + py * py);
    float ang = atan(py, px);
    float pts = max(2.0, n);
    float depth = clamp(m, 0.0, 0.92);
    float r_edge = r * (1.0 - depth * 0.5 + depth * 0.5 * cos(pts * ang));
    return rad - r_edge;
}

float smin(float a, float b, float k) {
    float ks = max(k, 1e-5);
    float h = max(ks - abs(a - b), 0.0) / ks;
    return min(a, b) - h * h * ks * 0.25;
}

float smax(float a, float b, float k) {
    return -smin(-a, -b, k);
}

float rotX(float px, float py, float ang) {
    return px * cos(ang) - py * sin(ang);
}

float rotY(float px, float py, float ang) {
    return px * sin(ang) + py * cos(ang);
}

float fbm(float px, float py) {
    float acc = 0.0;
    float amp = 0.50;
    float freq = 2.4;
    for (int i = 1; i <= 6; i++) {
        float fi = float(i);
        float a2 = 2.0 * fi * fi;
        float ca = cos(a2);
        float sa = sin(a2);
        float u1 = ca * px + sa * py;
        float v1 = ca * py - sa * px;
        float ph = 2.0 * cos(5.0 * fi);
        acc += amp * cos(freq * u1 + 1.4 * cos(freq * v1) + ph)
                   * cos(freq * v1 + 1.4 * sin(freq * u1) + ph);
        amp *= 0.62;
        freq *= 1.55;
    }
    return acc;
}

float fbmT(float px, float py, float t) {
    float acc = 0.0;
    float amp = 0.50;
    float freq = 2.4;
    float ox = 0.22 * cos(t);
    float oy = 0.22 * sin(t);
    for (int i = 1; i <= 6; i++) {
        float fi = float(i);
        float a2 = 2.0 * fi * fi;
        float ca = cos(a2);
        float sa = sin(a2);
        float qx = px + ox / fi;
        float qy = py + oy / fi;
        float u1 = ca * qx + sa * qy;
        float v1 = ca * qy - sa * qx;
        float ph = 2.0 * cos(5.0 * fi) + sin(t + fi);
        acc += amp * cos(freq * u1 + 1.4 * cos(freq * v1) + ph)
                   * cos(freq * v1 + 1.4 * sin(freq * u1) + ph);
        amp *= 0.62;
        freq *= 1.55;
    }
    return acc;
}

float hash21(float px, float py) {
    return fract(sin(px * 127.1 + py * 311.7) * 43758.5453);
}

float hamidF(float h) {
    float step0 = ee(-1000.0 * h);
    float exp1 = ee(1000.0 * (h - 1.0));
    float val = step0 * pow(max(abs(h), 1e-8), exp1);
    return clamp(floor(255.0 * clamp(val, 0.0, 1.0)) / 255.0, 0.0, 1.0);
}
"""


def _fmt_glsl_float(val: float) -> str:
    fv = float(val)
    if math.isnan(fv) or math.isinf(fv):
        return "0.0"
    s = f"{fv:.8g}"
    if "e" not in s and "E" not in s and "." not in s:
        s += ".0"
    return s


def _ast_to_glsl(expr: str, var_map: Dict[str, str]) -> str:
    tree = _parse_expr_ast(expr)

    def _emit(node: ast.AST) -> str:
        if isinstance(node, ast.Expression):
            return _emit(node.body)
        if isinstance(node, ast.Constant):
            return _fmt_glsl_float(float(node.value))
        if isinstance(node, ast.Name):
            if node.id in ("PI", "pi"):
                return "PI"
            if node.id in ("TAU", "tau"):
                return "TAU"
            return var_map.get(node.id, f"v_{node.id}")
        if isinstance(node, ast.UnaryOp):
            inner = _emit(node.operand)
            return f"(+{inner})" if isinstance(node.op, ast.UAdd) else f"(-({inner}))"
        if isinstance(node, ast.BinOp):
            left = _emit(node.left)
            right = _emit(node.right)
            op = node.op
            if isinstance(op, ast.Add):
                return f"({left} + {right})"
            if isinstance(op, ast.Sub):
                return f"({left} - {right})"
            if isinstance(op, ast.Mult):
                return f"({left} * {right})"
            if isinstance(op, ast.Div):
                if isinstance(node.right, ast.Constant) and abs(float(node.right.value)) >= 1e-6:
                    return f"({left} / {right})"
                return f"safeDiv({left}, {right})"
            if isinstance(op, ast.Mod):
                return f"safeMod({left}, {right})"
            if isinstance(op, ast.Pow):
                if isinstance(node.right, ast.Constant):
                    rv = float(node.right.value)
                    if abs(rv - 2.0) < 1e-6:
                        return f"(({left}) * ({left}))"
                    if abs(rv - 3.0) < 1e-6:
                        return f"(({left}) * ({left}) * ({left}))"
                return f"safePow({left}, {right})"
        if isinstance(node, ast.Compare):
            left = _emit(node.left)
            right = _emit(node.comparators[0])
            op = node.ops[0]
            sym = "<" if isinstance(op, ast.Lt) else "<=" if isinstance(op, ast.LtE) else ">" if isinstance(op, ast.Gt) else ">="
            return f"({left} {sym} {right})"
        if isinstance(node, ast.IfExp):
            return f"({_emit(node.test)} ? {_emit(node.body)} : {_emit(node.orelse)})"
        if isinstance(node, ast.Call):
            fn = node.func.id
            args = [_emit(a) for a in node.args]
            if fn in ("min", "max") and len(args) > 2:
                acc = f"{fn}({args[0]}, {args[1]})"
                for extra in args[2:]:
                    acc = f"{fn}({acc}, {extra})"
                return acc
            if fn == "atan2":
                return f"atan({args[0]}, {args[1]})"
            if fn == "pow":
                return f"safePow({args[0]}, {args[1]})"
            if fn == "mod":
                return f"safeMod({args[0]}, {args[1]})"
            if fn == "tan":
                return f"safeTan({args[0]})"
            if fn == "asin":
                return f"safeAsin({args[0]})"
            if fn == "acos":
                return f"safeAcos({args[0]})"
            if fn == "exp":
                return f"safeExp({args[0]})"
            if fn == "log":
                return f"safeLog({args[0]})"
            if fn == "sqrt":
                return f"safeSqrt({args[0]})"
            if fn == "smoothstep":
                return f"safeSmoothstep({args[0]}, {args[1]}, {args[2]})"
            return f"{fn}({', '.join(args)})"
        raise ValueError(f"Unsupported AST node in GLSL emission: {type(node).__name__}")

    return _emit(tree)


def compile_math_program_to_glsl(prog: Dict[str, Any]) -> str:
    """
    Compiles a validated `math_program` into a 100% WebGL1-compliant GLSL ES 1.00
    fragment shader (`precision highp float; uniform vec2 u_resolution; uniform float u_time;`).
    All numeric literals are strictly float, all loop bounds are static integer
    constants, and all user variables are safely prefixed (`v_<name>`) so they
    never collide with GLSL reserved keywords.
    """
    ok, err = validate_math_program(prog)
    if not ok:
        raise ValueError(f"Cannot compile invalid math_program to GLSL: {err}")

    dom = prog.get("domain") or {}
    x_min = float(dom.get("x_min", -1.6))
    x_max = float(dom.get("x_max", 1.6))
    y_min = float(dom.get("y_min", -1.0))
    y_max = float(dom.get("y_max", 1.0))

    lines: List[str] = [_GLSL_HELPERS.rstrip(), "", "void main() {"]
    lines.append("    vec2 uv = gl_FragCoord.xy / max(u_resolution, vec2(1.0));")
    lines.append("    // Keep v_uv active for vertex-shader linkage")
    lines.append("    uv = mix(uv, v_uv, 0.0);")
    lines.append(f"    float x = mix({_fmt_glsl_float(x_min)}, {_fmt_glsl_float(x_max)}, uv.x);")
    lines.append(f"    float y = mix({_fmt_glsl_float(y_min)}, {_fmt_glsl_float(y_max)}, uv.y);")
    lines.append("    float t = u_time;")

    var_map: Dict[str, str] = {"x": "x", "y": "y", "t": "t"}
    declared: Set[str] = {"x", "y", "t"}

    for lp_idx, step in enumerate(prog.get("steps", [])):
        if "loop" in step:
            lp = step["loop"]
            lvar = lp["var"]
            liter = lp["iter"]
            start = int(lp.get("start", 1))
            end = int(lp.get("end", 1))
            g_lvar = f"v_{lvar}"
            init_glsl = _ast_to_glsl(str(lp.get("init", "0.0")), var_map)
            if lvar in declared:
                lines.append(f"    {g_lvar} = {init_glsl};")
            else:
                lines.append(f"    float {g_lvar} = {init_glsl};")
                declared.add(lvar)
            var_map[lvar] = g_lvar

            i_name = f"i_lp{lp_idx}_{liter}"
            g_iter = f"v_lp{lp_idx}_{liter}"
            g_acc = f"v_lp{lp_idx}_acc"
            lines.append(f"    for (int {i_name} = {start}; {i_name} <= {end}; {i_name}++) {{")
            lines.append(f"        float {g_iter} = float({i_name});")
            lines.append(f"        float {g_acc} = {g_lvar};")

            inner_map = dict(var_map)
            inner_map[liter] = g_iter
            inner_map["acc"] = g_acc
            inner_map[lvar] = g_acc
            inner_declared: Set[str] = set()

            for bstep in lp.get("body", []):
                bvar = bstep["var"]
                g_bvar = f"v_lp{lp_idx}_b_{bvar}"
                b_glsl = _ast_to_glsl(str(bstep["expr"]), inner_map)
                if bvar in inner_declared:
                    lines.append(f"        {g_bvar} = {b_glsl};")
                else:
                    lines.append(f"        float {g_bvar} = {b_glsl};")
                    inner_declared.add(bvar)
                inner_map[bvar] = g_bvar

            upd_glsl = _ast_to_glsl(str(lp["update"]), inner_map)
            lines.append(f"        {g_lvar} = {upd_glsl};")
            lines.append("    }")
        else:
            vname = step["var"]
            g_var = f"v_{vname}"
            e_glsl = _ast_to_glsl(str(step["expr"]), var_map)
            if vname in declared:
                lines.append(f"    {g_var} = {e_glsl};")
            else:
                lines.append(f"    float {g_var} = {e_glsl};")
                declared.add(vname)
            var_map[vname] = g_var

    lines.append(f"    float out_H0 = {_ast_to_glsl(prog['H0'], var_map)};")
    lines.append(f"    float out_H1 = {_ast_to_glsl(prog['H1'], var_map)};")
    lines.append(f"    float out_H2 = {_ast_to_glsl(prog['H2'], var_map)};")
    lines.append("    gl_FragColor = vec4(hamidF(out_H0), hamidF(out_H1), hamidF(out_H2), 1.0);")
    lines.append("}")
    return "\n".join(lines)


def validate_glsl_webgl1_static(glsl_src: str) -> Tuple[bool, str]:
    """Static sanity validator confirming WebGL1 ES 1.00 compliance."""
    required = [
        "precision highp float;",
        "uniform vec2 u_resolution;",
        "uniform float u_time;",
        "void main()",
        "gl_FragColor",
    ]
    for token in required:
        if token not in glsl_src:
            return False, f"Missing required WebGL1 token: {token}"
    if glsl_src.count("{") != glsl_src.count("}"):
        return False, "Unbalanced curly braces in GLSL"
    if glsl_src.count("(") != glsl_src.count(")"):
        return False, "Unbalanced parentheses in GLSL"
    if "atan2(" in glsl_src:
        return False, "WebGL1 GLSL forbids atan2(); must emit atan(y, x)"
    if "**" in glsl_src:
        return False, "GLSL forbids ** operator"
    return True, "ok"


# ============================================================================
# 5. AUTHENTIC AST-DERIVED LATEX EQUATIONS & STANDALONE PYTHON SCRIPT
# ============================================================================

def _tex_ident(name: str) -> str:
    if name in ("x", "y", "t"):
        return name
    if name in ("PI", "pi"):
        return r"\pi"
    if name in ("TAU", "tau"):
        return r"2\pi"
    if "_" in name:
        base, sub = name.split("_", 1)
        sub_clean = sub.replace("_", r"\,")
        return rf"{base}_{{{sub_clean}}}"
    if len(name) == 1:
        return name
    return rf"\mathrm{{{name}}}"


def _ast_to_latex(expr: str, max_len: int = 240) -> str:
    tree = _parse_expr_ast(expr)

    def _emit(node: ast.AST, prec: int = 0) -> str:
        if isinstance(node, ast.Expression):
            return _emit(node.body, 0)
        if isinstance(node, ast.Constant):
            fv = float(node.value)
            if abs(fv - round(fv)) < 1e-6 and abs(fv) < 1e6:
                return str(int(round(fv)))
            return f"{fv:.3g}"
        if isinstance(node, ast.Name):
            return _tex_ident(node.id)
        if isinstance(node, ast.UnaryOp):
            inner = _emit(node.operand, 3)
            return f"-{inner}" if isinstance(node.op, ast.USub) else inner
        if isinstance(node, ast.BinOp):
            op = node.op
            if isinstance(op, ast.Add):
                s = f"{_emit(node.left, 1)} + {_emit(node.right, 1)}"
                return f"({s})" if prec > 1 else s
            if isinstance(op, ast.Sub):
                s = f"{_emit(node.left, 1)} - {_emit(node.right, 2)}"
                return f"({s})" if prec > 1 else s
            if isinstance(op, ast.Mult):
                s = rf"{_emit(node.left, 2)}\,{_emit(node.right, 2)}"
                return f"({s})" if prec > 2 else s
            if isinstance(op, ast.Div):
                return rf"\frac{{{_emit(node.left, 0)}}}{{{_emit(node.right, 0)}}}"
            if isinstance(op, ast.Pow):
                return rf"\left({_emit(node.left, 4)}\right)^{{{_emit(node.right, 0)}}}"
            if isinstance(op, ast.Mod):
                return rf"\operatorname{{mod}}\!\left({_emit(node.left, 0)}, {_emit(node.right, 0)}\right)"
        if isinstance(node, ast.Compare):
            op = node.ops[0]
            sym = "<" if isinstance(op, ast.Lt) else r"\le" if isinstance(op, ast.LtE) else ">" if isinstance(op, ast.Gt) else r"\ge"
            return rf"{_emit(node.left, 0)} {sym} {_emit(node.comparators[0], 0)}"
        if isinstance(node, ast.IfExp):
            return rf"\begin{{cases}}{_emit(node.body, 0)} & {_emit(node.test, 0)} \\ {_emit(node.orelse, 0)} & \text{{otherwise}}\end{{cases}}"
        if isinstance(node, ast.Call):
            fn = node.func.id
            args = [_emit(a, 0) for a in node.args]
            if fn == "ee":
                return rf"e^{{-e^{{{args[0]}}}}}"
            if fn == "exp":
                return rf"e^{{{args[0]}}}"
            if fn == "sqrt":
                return rf"\sqrt{{{args[0]}}}"
            if fn == "abs":
                return rf"\left|{args[0]}\right|"
            if fn == "floor":
                return rf"\left\lfloor {args[0]} \right\rfloor"
            if fn in ("sin", "cos", "tan", "asin", "acos", "atan", "log", "min", "max"):
                cmd = r"\arcsin" if fn == "asin" else r"\arccos" if fn == "acos" else r"\arctan" if fn == "atan" else f"\\{fn}"
                return rf"{cmd}\!\left({', '.join(args)}\right)"
            if fn == "atan2":
                return rf"\operatorname{{atan2}}\!\left({args[0]}, {args[1]}\right)"
            if fn == "pow":
                return rf"\left({args[0]}\right)^{{{args[1]}}}"
            if fn == "clamp":
                return rf"\operatorname{{clamp}}\!\left({', '.join(args)}\right)"
            return rf"\operatorname{{{fn}}}\!\left({', '.join(args)}\right)"
        return ""

    res = _emit(tree, 0)
    if len(res) > max_len:
        # Clean truncation at a balanced level or suffix ellipsis
        res = res[:max_len].rstrip(r"\, +-(")
        # Balance curly braces if cut inside braces
        diff_br = res.count("{") - res.count("}")
        if diff_br > 0:
            res += "}" * diff_br
        diff_p = res.count(r"\left(") - res.count(r"\right)")
        if diff_p > 0:
            res += r"\right)" * diff_p
        res += r" \cdots"
    return res


def compile_math_program_to_latex(prog: Dict[str, Any]) -> Dict[str, str]:
    """
    Derives clean, authentic KaTeX LaTeX equations directly from the AST of `prog`:
    RGB master formula, H_0, H_1, H_2, plus key geometric definitions & loop summations.
    """
    dom = prog.get("domain") or {}
    x_min = float(dom.get("x_min", -1.6))
    x_max = float(dom.get("x_max", 1.6))
    y_min = float(dom.get("y_min", -1.0))
    y_max = float(dom.get("y_max", 1.0))
    aspect = prog.get("aspect", [16, 10])
    W_nom = 2000
    H_nom = max(400, int(round(W_nom * aspect[1] / max(1, aspect[0])))) if isinstance(aspect, (list, tuple)) and len(aspect) == 2 else 1250
    x_span = x_max - x_min
    y_span = y_max - y_min

    latex: Dict[str, str] = {
        "RGB": (
            rf"\operatorname{{rgb}}\!\left(F(H_0(x,y,t)),\, F(H_1(x,y,t)),\, F(H_2(x,y,t))\right), \quad "
            rf"x = {x_min:.2f} + {x_span:.2f}\frac{{m-1}}{{{W_nom-1}}},\; y = {y_max:.2f} - {y_span:.2f}\frac{{n-1}}{{{H_nom-1}}}, \quad "
            r"F(u) = \left\lfloor 255\,e^{-e^{-1000u}}|u|^{e^{-e^{1000(u-1)}}}\right\rfloor"
        ),
        "H_0": rf"H_0(x, y, t) = {_ast_to_latex(prog['H0'], max_len=240)}",
        "H_1": rf"H_1(x, y, t) = {_ast_to_latex(prog['H1'], max_len=240)}",
        "H_2": rf"H_2(x, y, t) = {_ast_to_latex(prog['H2'], max_len=240)}",
    }

    steps = prog.get("steps", [])
    loop_idx = 0
    helper_eqs: List[str] = []

    for step in steps:
        if "loop" in step:
            loop_idx += 1
            lp = step["loop"]
            lvar = _tex_ident(lp["var"])
            liter = _tex_ident(lp["iter"])
            s0, s1 = int(lp.get("start", 1)), int(lp.get("end", 1))
            upd_str = str(lp.get("update", ""))
            upd_tex = _ast_to_latex(upd_str, max_len=140)
            if upd_str.strip().startswith("max("):
                latex[f"L_{loop_idx}"] = rf"{lvar}(x,y,t) = \max_{{{liter}={s0}}}^{{{s1}}}\!\Big({upd_tex}\Big)"
            elif upd_str.strip().startswith("min(") or upd_str.strip().startswith("smin("):
                latex[f"L_{loop_idx}"] = rf"{lvar}(x,y,t) = \min_{{{liter}={s0}}}^{{{s1}}}\!\Big({upd_tex}\Big)"
            else:
                latex[f"L_{loop_idx}"] = rf"{lvar}(x,y,t) = \sum_{{{liter}={s0}}}^{{{s1}}}\!\Big({upd_tex}\Big)"
        else:
            if len(helper_eqs) < 6:
                v_tex = _tex_ident(step["var"])
                e_tex = _ast_to_latex(str(step["expr"]), max_len=110)
                helper_eqs.append(rf"{v_tex} = {e_tex}")

    if helper_eqs:
        latex["S_1"] = r", \quad ".join(helper_eqs[:3])
    if len(helper_eqs) > 3:
        latex["S_2"] = r", \quad ".join(helper_eqs[3:6])

    return latex


def build_symbolic_python_script(prog: Dict[str, Any]) -> str:
    """Generates a clean, standalone, AST-valid Python script reproducing the math_program."""
    import json as _json
    title = sanitize_math_label(str(prog.get("title", "Symbolic Analytical Field")))
    clean_prog = {
        "title": title,
        "subtitle": sanitize_math_label(str(prog.get("subtitle", "Symbolic Double-Exponential Field"))),
        "aspect": prog.get("aspect", [16, 10]),
        "domain": prog.get("domain", {"x_min": -1.6, "x_max": 1.6, "y_min": -1.0, "y_max": 1.0}),
        "steps": prog.get("steps", []),
        "H0": prog.get("H0", "0.0"),
        "H1": prog.get("H1", "0.0"),
        "H2": prog.get("H2", "0.0"),
    }
    prog_json = _json.dumps(clean_prog, indent=2)
    return f'''"""
Standalone Pure-Math Symbolic Painting & 4D Video Script: {title}
Evaluates H_0(x, y, t), H_1(x, y, t), H_2(x, y, t) through Hamid's F(u) quantizer.
"""
import json
import matplotlib.pyplot as plt
from symbolic_math_compiler import evaluate_math_program_numpy

MATH_PROGRAM = json.loads(r"""{prog_json}""")

if __name__ == "__main__":
    res = evaluate_math_program_numpy(MATH_PROGRAM, width=800, height=500, t_phase=0.0, include_raw_array=True)
    plt.figure(figsize=(10, 6.25), facecolor="#050811")
    plt.imshow(res["rgb_array"])
    plt.axis("off")
    plt.tight_layout()
    plt.show()
'''


# ============================================================================
# 6. CRISP ISO-CONTOUR & 3D/4D MANIFOLD EXTRACTOR FROM MATH PROGRAM
# ============================================================================

def extract_iso_contours_from_math_program(
    prog: Dict[str, Any],
    N: int = 1800,
    t_phase: float = 0.0,
) -> Dict[str, Any]:
    """
    Extracts crisp, unmistakable 2D chord envelopes (x1, y1, x2, y2), 3D relief
    manifolds (x1, y1, z1, x2, y2, z2), and 4D temporal motion harmonics
    (ax1, bx1, ay1, by1, ax2, bx2, ay2, by2) directly from the analytical
    luminance & edge iso-contours of `prog`!
    """
    ok, err = validate_math_program(prog)
    if not ok:
        raise ValueError(f"Invalid math_program: {err}")

    N = max(100, min(5000, int(N)))
    W, H = 360, 225
    x_min, x_max, y_min, y_max = _resolve_domain(prog, W, H)

    xs = np.linspace(x_min, x_max, W, dtype=np.float32)
    ys = np.linspace(y_max, y_min, H, dtype=np.float32)
    X, Y = np.meshgrid(xs, ys)
    compiled_steps, ast_h0, ast_h1, ast_h2 = _precompile_program_asts(prog)

    rgb0 = _eval_compiled_frame(compiled_steps, ast_h0, ast_h1, ast_h2, X, Y, float(t_phase))
    rgb_q = _eval_compiled_frame(compiled_steps, ast_h0, ast_h1, ast_h2, X, Y, float(t_phase + 0.5 * math.pi))
    rgb_h = _eval_compiled_frame(compiled_steps, ast_h0, ast_h1, ast_h2, X, Y, float(t_phase + math.pi))

    gray0 = cv2.cvtColor(rgb0, cv2.COLOR_RGB2GRAY)
    gray_q = cv2.cvtColor(rgb_q, cv2.COLOR_RGB2GRAY)
    gray_h = cv2.cvtColor(rgb_h, cv2.COLOR_RGB2GRAY)

    # Combine multi-channel Canny edges + multi-quantile luminance iso-contours
    blur = cv2.GaussianBlur(gray0, (3, 3), 0.8)
    canny_lum = cv2.Canny(blur, 32, 95)
    # Also detect strong chromatic boundaries (e.g. bright orange tiger vs green forest)
    canny_r = cv2.Canny(cv2.GaussianBlur(rgb0[..., 0], (3, 3), 0.8), 38, 105)
    canny_g = cv2.Canny(cv2.GaussianBlur(rgb0[..., 1], (3, 3), 0.8), 38, 105)
    canny_b = cv2.Canny(cv2.GaussianBlur(rgb0[..., 2], (3, 3), 0.8), 38, 105)
    edge_union = cv2.bitwise_or(canny_lum, cv2.bitwise_or(canny_r, cv2.bitwise_or(canny_g, canny_b)))

    contours: List[np.ndarray] = []
    c_edges, _ = cv2.findContours(edge_union, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    for cnt in c_edges:
        if len(cnt) >= 8:
            contours.append(cnt[:, 0, :].astype(np.float32))

    # Supplement with clean iso-level luminance contours for rich shaded structure
    for q in (28, 52, 76):
        thr = int(np.percentile(blur, q))
        if 12 < thr < 245:
            _, mask = cv2.threshold(blur, thr, 255, cv2.THRESH_BINARY)
            c_iso, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
            for cnt in c_iso:
                if len(cnt) >= 14:
                    contours.append(cnt[:, 0, :].astype(np.float32))

    if not contours:
        # Fallback synthetic circle contour so extraction never fails even on flat images
        ang = np.linspace(0.0, 2.0 * math.pi, 180, dtype=np.float32)
        cx_fb = W * 0.5 + 0.35 * min(W, H) * np.cos(ang)
        cy_fb = H * 0.5 + 0.35 * min(W, H) * np.sin(ang)
        contours.append(np.stack([cx_fb, cy_fb], axis=-1))

    # Score contours by perimeter & center salience, keep top clean contours
    scored: List[Tuple[float, np.ndarray]] = []
    for pts in contours:
        diffs = np.diff(pts, axis=0)
        arc_len = float(np.sum(np.hypot(diffs[:, 0], diffs[:, 1])))
        if arc_len < 12.0:
            continue
        mx = float(np.mean(pts[:, 0])) / W - 0.5
        my = float(np.mean(pts[:, 1])) / H - 0.5
        center_bonus = 1.45 - 0.65 * min(1.0, math.hypot(mx, my))
        scored.append((arc_len * center_bonus, pts))

    scored.sort(key=lambda item: item[0], reverse=True)
    kept = scored[:140]
    total_weight = sum(w for w, _ in kept) or 1.0

    # Compute smooth 3D relief heightfield from luminance + interior distance transform
    lum_f = blur.astype(np.float32) / 255.0
    inv_edges = (edge_union == 0).astype(np.uint8)
    dist_map = cv2.distanceTransform(inv_edges, cv2.DIST_L2, 5)
    dist_norm = dist_map / max(1.0, float(np.max(dist_map)))
    z_field = 0.58 * (lum_f - 0.5) + 0.42 * (dist_norm - 0.35)
    z_field = cv2.GaussianBlur(z_field, (7, 7), 1.6)

    # Compute dense 4D motion flow fields between t=0 -> t=pi/2 and t=0 -> t=pi
    flow_q = cv2.calcOpticalFlowFarneback(gray0, gray_q, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    flow_h = cv2.calcOpticalFlowFarneback(gray0, gray_h, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    # Convert pixel displacements into normalized [-1, 1] space
    scale_x = 2.0 / max(1.0, float(W - 1))
    scale_y = -2.0 / max(1.0, float(H - 1))

    # Build N Hamid-style tangent/secant chords along the crisp contours
    seg_x1: List[float] = []
    seg_y1: List[float] = []
    seg_z1: List[float] = []
    seg_x2: List[float] = []
    seg_y2: List[float] = []
    seg_z2: List[float] = []
    harm_rows: List[List[float]] = []

    norm_scale = max(x_max - x_min, y_max - y_min) * 0.5
    cx_dom = 0.5 * (x_min + x_max)
    cy_dom = 0.5 * (y_min + y_max)

    def _pix_to_norm(px: float, py: float) -> Tuple[float, float]:
        wx = x_min + (px / max(1.0, W - 1.0)) * (x_max - x_min)
        wy = y_max - (py / max(1.0, H - 1.0)) * (y_max - y_min)
        return (
            float(np.clip((wx - cx_dom) / max(norm_scale, 1e-4), -0.98, 0.98)),
            float(np.clip((wy - cy_dom) / max(norm_scale, 1e-4), -0.98, 0.98)),
        )

    def _sample_z_flow(px: float, py: float) -> Tuple[float, float, float, float, float]:
        ix = int(np.clip(round(px), 0, W - 1))
        iy = int(np.clip(round(py), 0, H - 1))
        zv = float(z_field[iy, ix])
        # Solve harmonic coefficients for u(t) = u0 + a*sin(t) + b*(cos(t) - 1)
        # At t=pi: u(pi) - u0 = -2b => b = -0.5 * d_half
        # At t=pi/2: u(pi/2) - u0 = a - b => a = d_quarter + b
        dx_q = float(flow_q[iy, ix, 0]) * scale_x
        dy_q = float(flow_q[iy, ix, 1]) * scale_y
        dx_h = float(flow_h[iy, ix, 0]) * scale_x
        dy_h = float(flow_h[iy, ix, 1]) * scale_y
        bx = float(np.clip(-0.5 * dx_h, -0.18, 0.18))
        by = float(np.clip(-0.5 * dy_h, -0.18, 0.18))
        ax = float(np.clip(dx_q + bx, -0.18, 0.18))
        ay = float(np.clip(dy_q + by, -0.18, 0.18))
        return zv, ax, bx, ay, by

    allocs = [max(2, int(round(N * w / total_weight))) for w, _ in kept]
    diff_n = N - sum(allocs)
    if allocs:
        allocs[0] = max(2, allocs[0] + diff_n)

    for alloc, (_, pts) in zip(allocs, kept):
        M = len(pts)
        if M < 2:
            continue
        idxs = np.linspace(0, M - 1, alloc, dtype=np.int32)
        stride = max(2, min(14, M // 10))
        for p_idx in idxs:
            q_idx = (p_idx + stride) % M if np.hypot(*(pts[0] - pts[-1])) < 6.0 else min(M - 1, p_idx + stride)
            if q_idx == p_idx:
                q_idx = max(0, p_idx - 1)
            px1, py1 = float(pts[p_idx, 0]), float(pts[p_idx, 1])
            px2, py2 = float(pts[q_idx, 0]), float(pts[q_idx, 1])
            # Skip artificial outer canvas-border lines from threshold contours
            on_border1 = (px1 <= 2.5 or px1 >= W - 3.5 or py1 <= 2.5 or py1 >= H - 3.5)
            on_border2 = (px2 <= 2.5 or px2 >= W - 3.5 or py2 <= 2.5 or py2 >= H - 3.5)
            if on_border1 and on_border2:
                continue
            nx1, ny1 = _pix_to_norm(px1, py1)
            nx2, ny2 = _pix_to_norm(px2, py2)
            z1, ax1, bx1, ay1, by1 = _sample_z_flow(px1, py1)
            z2, ax2, bx2, ay2, by2 = _sample_z_flow(px2, py2)

            # Guarantee gentle breathing harmonic even on quasi-static contours
            if abs(ax1) + abs(bx1) + abs(ay1) + abs(by1) < 0.008:
                ax1 = 0.024 * math.sin(2.5 * ny1 + nx1)
                ay1 = 0.024 * math.cos(2.5 * nx1 - ny1)
                ax2 = 0.024 * math.sin(2.5 * ny2 + nx2)
                ay2 = 0.024 * math.cos(2.5 * nx2 - ny2)

            seg_x1.append(nx1)
            seg_y1.append(ny1)
            seg_z1.append(z1)
            seg_x2.append(nx2)
            seg_y2.append(ny2)
            seg_z2.append(z2)
            harm_rows.append([
                round(ax1, 5), round(bx1, 5), round(ay1, 5), round(by1, 5),
                round(ax2, 5), round(bx2, 5), round(ay2, 5), round(by2, 5),
            ])

    # Trim or pad evenly across extracted contours to exact length N
    n_base = len(seg_x1)
    if n_base == 0:
        ang = np.linspace(0.0, 2.0 * math.pi, N, endpoint=False)
        for k_i in range(N):
            a0 = float(ang[k_i])
            a1 = float(ang[(k_i + 12) % N])
            seg_x1.append(0.55 * math.cos(a0))
            seg_y1.append(0.55 * math.sin(a0))
            seg_z1.append(0.15 * math.sin(2.0 * a0))
            seg_x2.append(0.55 * math.cos(a1))
            seg_y2.append(0.55 * math.sin(a1))
            seg_z2.append(0.15 * math.sin(2.0 * a1))
            harm_rows.append([0.03, 0.0, 0.0, 0.03, 0.03, 0.0, 0.0, 0.03])
    else:
        pad_i = 0
        while len(seg_x1) < N:
            idx_r = pad_i % n_base
            pad_i += 1
            seg_x1.append(seg_x1[idx_r])
            seg_y1.append(seg_y1[idx_r])
            seg_z1.append(seg_z1[idx_r])
            seg_x2.append(seg_x2[idx_r])
            seg_y2.append(seg_y2[idx_r])
            seg_z2.append(seg_z2[idx_r])
            harm_rows.append(harm_rows[idx_r])

    seg_x1 = seg_x1[:N]
    seg_y1 = seg_y1[:N]
    seg_z1 = seg_z1[:N]
    seg_x2 = seg_x2[:N]
    seg_y2 = seg_y2[:N]
    seg_z2 = seg_z2[:N]
    harm_rows = harm_rows[:N]

    primitives = [
        [round(seg_x1[i], 5), round(seg_y1[i], 5), round(seg_x2[i], 5), round(seg_y2[i], 5)]
        for i in range(N)
    ]
    primitives_3d = [
        [
            round(seg_x1[i], 5), round(seg_y1[i], 5), round(seg_z1[i], 5),
            round(seg_x2[i], 5), round(seg_y2[i], 5), round(seg_z2[i], 5),
        ]
        for i in range(N)
    ]

    return {
        "N": N,
        "primitives": primitives,
        "primitives_3d": primitives_3d,
        "time_harmonics": harm_rows,
        "x1": np.asarray(seg_x1, dtype=np.float64),
        "y1": np.asarray(seg_y1, dtype=np.float64),
        "z1": np.asarray(seg_z1, dtype=np.float64),
        "x2": np.asarray(seg_x2, dtype=np.float64),
        "y2": np.asarray(seg_y2, dtype=np.float64),
        "z2": np.asarray(seg_z2, dtype=np.float64),
        "contour_count": len(kept),
        "motion_energy": round(float(np.mean(np.abs(rgb_q.astype(np.float32) - rgb0.astype(np.float32)))), 3),
    }


# ============================================================================
# 7. REAL STANDALONE LOOPING ANIMATED GIF & MP4 VIDEO RENDERER
# ============================================================================

def render_math_video_file(
    prog: Dict[str, Any],
    out_gif_path: Optional[str] = None,
    out_mp4_path: Optional[str] = None,
    width: int = 480,
    height: int = 300,
    num_frames: int = 32,
    fps: int = 16,
    frames_strip_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Renders seamless looping 4D mathematical animation frames t_i = 2*pi*i/num_frames
    directly from `prog` into a real animated GIF (via PIL) and optional MP4
    (via OpenCV VideoWriter 'mp4v'). Returns JSON-serializable metadata including
    `gif_base64`, optional `mp4_base64`, and `motion_energy`.
    """
    ok, err = validate_math_program(prog)
    if not ok:
        raise ValueError(f"Invalid math_program for video render: {err}")

    W = max(64, int(width))
    H = max(64, int(height))
    n_fr = max(4, min(120, int(num_frames)))
    fps_i = max(4, min(60, int(fps)))

    x_min, x_max, y_min, y_max = _resolve_domain(prog, W, H)
    xs = np.linspace(x_min, x_max, W, dtype=np.float32)
    ys = np.linspace(y_max, y_min, H, dtype=np.float32)
    X, Y = np.meshgrid(xs, ys)
    compiled_steps, ast_h0, ast_h1, ast_h2 = _precompile_program_asts(prog)

    frames_rgb: List[np.ndarray] = []
    pil_frames: List[Image.Image] = []
    for i in range(n_fr):
        t_val = 2.0 * math.pi * float(i) / float(n_fr)
        rgb = _eval_compiled_frame(compiled_steps, ast_h0, ast_h1, ast_h2, X, Y, t_val)
        frames_rgb.append(rgb)
        pil_frames.append(Image.fromarray(rgb, mode="RGB").quantize(colors=256, method=Image.Quantize.MEDIANCUT))

    q_idx = max(1, n_fr // 4) % n_fr
    motion_energy = float(np.mean(np.abs(frames_rgb[q_idx].astype(np.float32) - frames_rgb[0].astype(np.float32))))

    # Encode GIF to memory (and optionally write to disk)
    gif_buf = io.BytesIO()
    duration_ms = max(20, int(round(1000.0 / float(fps_i))))
    pil_frames[0].save(
        gif_buf,
        format="GIF",
        save_all=True,
        append_images=pil_frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=False,
    )
    gif_bytes = gif_buf.getvalue()
    if out_gif_path:
        os.makedirs(os.path.dirname(os.path.abspath(out_gif_path)), exist_ok=True)
        with open(out_gif_path, "wb") as f:
            f.write(gif_bytes)

    mp4_base64: Optional[str] = None
    saved_mp4: Optional[str] = None
    if out_mp4_path:
        os.makedirs(os.path.dirname(os.path.abspath(out_mp4_path)), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_mp4_path, fourcc, float(fps_i), (W, H))
        try:
            for rgb in frames_rgb:
                writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        finally:
            writer.release()
        if os.path.exists(out_mp4_path) and os.path.getsize(out_mp4_path) > 0:
            saved_mp4 = out_mp4_path
            with open(out_mp4_path, "rb") as f:
                mp4_base64 = "data:video/mp4;base64," + base64.b64encode(f.read()).decode("ascii")

    saved_strip: Optional[str] = None
    if frames_strip_path:
        os.makedirs(os.path.dirname(os.path.abspath(frames_strip_path)), exist_ok=True)
        strip_indices = [0, n_fr // 4, n_fr // 2, (3 * n_fr) // 4]
        strip_arr = np.concatenate([frames_rgb[idx % n_fr] for idx in strip_indices], axis=1)
        Image.fromarray(strip_arr, mode="RGB").save(frames_strip_path, format="PNG")
        saved_strip = frames_strip_path

    return {
        "gif_path": out_gif_path if out_gif_path else None,
        "mp4_path": saved_mp4,
        "frames_strip_path": saved_strip,
        "gif_base64": "data:image/gif;base64," + base64.b64encode(gif_bytes).decode("ascii"),
        "mp4_base64": mp4_base64,
        "width": W,
        "height": H,
        "num_frames": n_fr,
        "fps": fps_i,
        "motion_energy": round(motion_energy, 3),
        "first_frame_stats": _compute_image_stats(frames_rgb[0]),
    }
