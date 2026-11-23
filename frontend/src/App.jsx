import React, { useState, useEffect, useRef, useCallback, useMemo } from 'react';
import katex from 'katex';
import {
  Sparkles,
  Play,
  Pause,
  RotateCcw,
  Code2,
  Download,
  Sliders,
  Compass,
  Copy,
  Check,
  Image as ImageIcon,
  Grid,
  Wand2,
  BookOpen,
  Layers,
  Terminal,
  Box,
  Film,
  Palette,
  FileImage,
  Video,
  Orbit,
  RefreshCw,
  Cpu,
  ShieldCheck,
  Clapperboard,
  FileText,
  ArrowRight,
} from 'lucide-react';

const SAMPLE_EPIC_5_ACT_SCRIPT = `ACT 1: Genesis of the Complex Riemann Plane
In the beginning, pure complex conformal waves w(z, t) = (1 - tau)*z + tau*(z^3 - 1)/(z^3 + 1) bend the fabric of the mathematical vacuum into glowing analytical phase lattices and essential singularities.

ACT 2: Awakening in the Abyssal Floraform Reef
From the complex plane, life emerges in the deep ocean: a bioluminescent deep-sea dragonfish glides above ruffled differential-growth Floraform coral saddles governed by elastic buckling PDEs.

ACT 3: Ascent of the Aurelian Phoenix
Breaking through the ocean surface, a golden 3D raymarched phoenix with FBM-sculpted plumage ascends over turbulent sapphire waves under volumetric god-rays.

ACT 4: Sanctum of the Hyperbolic Gyroid Cathedral
The phoenix enters a monumental bronze cathedral carved from Schoen's triply periodic Gyroid minimal surface sin(x)cos(y) + sin(y)cos(z) + sin(z)cos(x) = 0 with zero mean curvature H = 0.

ACT 5: Transcendence into 4D Non-Euclidean Space
At the altar, a sphere turns smoothly inside-out via William Thurston's C4 Morin sphere-eversion homotopy, expanding into a non-Euclidean celestial astrolabe encircled by golden Saturnian rings.`;

const COLOR_THEMES = {
  cyan_glow: {
    name: 'Hamid Cyan',
    bg: '#050811',
    grid: 'rgba(56, 189, 248, 0.08)',
    axis: 'rgba(56, 189, 248, 0.24)',
    stroke: (t, alpha = 0.42) =>
      `hsla(${192 + Math.sin(t * Math.PI * 2) * 22}, 95%, 64%, ${alpha})`,
    strokeHsl: (t) => [192 + Math.sin(t * Math.PI * 2) * 22, 95, 64],
    activeLine: '#f43f5e',
    p1: '#38bdf8',
    p2: '#fbbf24',
  },
  golden_amber: {
    name: 'Golden Silk',
    bg: '#090705',
    grid: 'rgba(251, 191, 36, 0.08)',
    axis: 'rgba(251, 191, 36, 0.24)',
    stroke: (t, alpha = 0.4) =>
      `hsla(${36 + Math.cos(t * Math.PI * 2) * 18}, 95%, 62%, ${alpha})`,
    strokeHsl: (t) => [36 + Math.cos(t * Math.PI * 2) * 18, 95, 62],
    activeLine: '#38bdf8',
    p1: '#fbbf24',
    p2: '#f43f5e',
  },
  prismatic: {
    name: 'Harmonic Prism',
    bg: '#06070d',
    grid: 'rgba(168, 85, 247, 0.08)',
    axis: 'rgba(168, 85, 247, 0.24)',
    stroke: (t, alpha = 0.42) =>
      `hsla(${(t * 300 + 180) % 360}, 90%, 66%, ${alpha})`,
    strokeHsl: (t) => [(t * 300 + 180) % 360, 90, 66],
    activeLine: '#ffffff',
    p1: '#22d3ee',
    p2: '#e879f9',
  },
  ivory_classic: {
    name: 'Classic Hamid Ink',
    bg: '#07090e',
    grid: 'rgba(226, 232, 240, 0.07)',
    axis: 'rgba(226, 232, 240, 0.22)',
    stroke: (_t, alpha = 0.36) => `rgba(241, 245, 249, ${alpha})`,
    strokeHsl: () => [210, 40, 96],
    activeLine: '#38bdf8',
    p1: '#38bdf8',
    p2: '#f43f5e',
  },
  emerald_wave: {
    name: 'Emerald Caustic',
    bg: '#040a08',
    grid: 'rgba(16, 185, 129, 0.08)',
    axis: 'rgba(16, 185, 129, 0.24)',
    stroke: (t, alpha = 0.42) =>
      `hsla(${152 + Math.sin(t * Math.PI * 4) * 24}, 90%, 58%, ${alpha})`,
    strokeHsl: (t) => [152 + Math.sin(t * Math.PI * 4) * 24, 90, 58],
    activeLine: '#fbbf24',
    p1: '#34d399',
    p2: '#38bdf8',
  },
};

const MODE_DEFINITIONS = [
  {
    id: 'script_cinema',
    label: 'Long-Script Cinema Director',
    shortLabel: 'Script Cinema Director',
    badge: 'Multi-Act Script → Homotopy-Morphed Complete Mathematical Film',
  },
  {
    id: 'raymarch_cinema',
    label: '3D Raymarch Cinema',
    shortLabel: '3D Raymarch Cinema',
    badge: '3D Implicit SDF Raymarching & Differential-Geometry PDE Cinema f(x,y,z,t) = 0',
  },
  {
    id: 'lines',
    label: '2D Lines',
    shortLabel: '2D Lines',
    badge: '2D Parametric Line Segments (X₁(k), Y₁(k)) → (X₂(k), Y₂(k))',
  },
  {
    id: 'circles',
    label: '2D Circles',
    shortLabel: '2D Circles',
    badge: '2D Parametric Circle Families (X(k), Y(k), R(k))',
  },
  {
    id: 'pixel_field',
    label: '2D Pixel-Field',
    shortLabel: '2D Pixel-Field',
    badge: '2D Full-Color Pixel-Field Painting rgb(F(H₀(x,y)), F(H₁(x,y)), F(H₂(x,y)))',
  },
  {
    id: '3d_surface',
    label: '3D Manifold',
    shortLabel: '3D Manifold',
    badge: '3D Parametric Manifold (X₁(k), Y₁(k), Z₁(k)) → (X₂(k), Y₂(k), Z₂(k))',
  },
  {
    id: 'math_video',
    label: '4D Wireframe Video',
    shortLabel: '4D Wireframe Video',
    badge: '4D Time-Evolving Wireframe Equations (X(k, t), Y(k, t))',
  },
  {
    id: 'pixel_video',
    label: '4D Color Video',
    shortLabel: '4D Color Video',
    badge: '4D Full-Color GPU Shader Video rgb(F(H₀(x,y,t)), F(H₁(x,y,t)), F(H₂(x,y,t)))',
  },
  {
    id: 'hdges',
    label: 'H-DGES Homotopy',
    shortLabel: 'H-DGES Homotopy',
    badge: 'H-DGES Homotopy-Annealed Gompertz-Harmonic Autodiff Φ_κ(g) = exp(-exp(κ·g))',
  },
];

// Stage 1: 8 Master Paradigms of Real-World Mathematical Art & Cinema
const FALLBACK_MASTER_PARADIGM_PRESETS = [
  {
    id: 'mp_iq_mandelbulb',
    name: '1. Inigo Quilez — 3D SDF Mandelbulb & GPU Raymarching',
    mode: 'raymarch_cinema',
    category: '8 Master Mathematical Paradigms',
  },
  {
    id: 'mp_ddg_gyroid',
    name: '2. Ferguson & Grossman — Minimal Gyroid & Torus Knot DDG',
    mode: 'raymarch_cinema',
    category: '8 Master Mathematical Paradigms',
  },
  {
    id: 'mp_floraform_pde',
    name: '3. Nervous System (Floraform) — Differential Growth PDE',
    mode: 'raymarch_cinema',
    category: '8 Master Mathematical Paradigms',
  },
  {
    id: 'mp_thurston_eversion',
    name: '4. Thurston Geometry Center — 4D Sphere Eversion Homotopy',
    mode: 'raymarch_cinema',
    category: '8 Master Mathematical Paradigms',
  },
  {
    id: 'mp_xenakis_metastasis',
    name: '5. Iannis Xenakis (Metastasis) — Stochastic Ruled Paraboloids',
    mode: 'raymarch_cinema',
    category: '8 Master Mathematical Paradigms',
  },
  {
    id: 'mp_houdini_curl_attractor',
    name: '6. Houdini Style — 3D Curl-Noise RK4 Attractor & Gray-Scott PDE',
    mode: 'raymarch_cinema',
    category: '8 Master Mathematical Paradigms',
  },
  {
    id: 'mp_manim_conformal',
    name: '7. Manim / 3Blue1Brown — Complex Conformal Homotopy w = f(z,t)',
    mode: 'raymarch_cinema',
    category: '8 Master Mathematical Paradigms',
  },
  {
    id: 'mp_kleinian_mandelbox',
    name: '8. 4K Raymarch Cinema — Non-Euclidean Kleinian Mandelbox Cathedral',
    mode: 'raymarch_cinema',
    category: '8 Master Mathematical Paradigms',
  },
];

// Default fallback catalog definitions grouped by category
const FALLBACK_PIXEL_FIELD_PRESETS = [
  { id: 'pf_peacock', name: 'Peacock in Full Feather Display', mode: 'pixel_field', category: '2D Pixel-Field Paintings' },
  { id: 'pf_autumn_trees', name: 'Autumn Trees Forest Avenue', mode: 'pixel_field', category: '2D Pixel-Field Paintings' },
  { id: 'pf_kiwi', name: 'Kiwi Bird in Forest Bokeh', mode: 'pixel_field', category: '2D Pixel-Field Paintings' },
  { id: 'pf_strawberries', name: 'Fresh Strawberries', mode: 'pixel_field', category: '2D Pixel-Field Paintings' },
  { id: 'pf_corals', name: 'Corals & Reef Fish', mode: 'pixel_field', category: '2D Pixel-Field Paintings' },
  { id: 'pf_bipolar_nebula', name: 'Bipolar Planetary Nebula', mode: 'pixel_field', category: '2D Pixel-Field Paintings' },
  { id: 'pf_moon_footprint', name: "Astronaut's Footprint on the Moon", mode: 'pixel_field', category: '2D Pixel-Field Paintings' },
];

const FALLBACK_SCIAM_HAMID_PRESETS = [
  { id: 'bird_in_flight', name: 'Bird in Flight', mode: 'lines', category: 'Scientific American & Hamid Originals' },
  { id: 'japanese_maple_sciam', name: 'Japanese Maple 7-Step', mode: 'lines', category: 'Scientific American & Hamid Originals' },
  { id: 'maple_leaf_sciam', name: 'Autumn Maple', mode: 'lines', category: 'Scientific American & Hamid Originals' },
  { id: 'oak_leaf_sciam', name: 'Oak Leaf', mode: 'lines', category: 'Scientific American & Hamid Originals' },
  { id: 'olive_branch_sciam', name: 'Olive Branch', mode: 'circles', category: 'Scientific American & Hamid Originals' },
  { id: 'palm_branch_sciam', name: 'Palm Frond', mode: 'circles', category: 'Scientific American & Hamid Originals' },
  { id: 'sailing_boat', name: 'Boat', mode: 'lines', category: 'Scientific American & Hamid Originals' },
  { id: 'butterfly_wings', name: 'Butterfly', mode: 'lines', category: 'Scientific American & Hamid Originals' },
  { id: 'tropical_fish', name: 'Fish', mode: 'lines', category: 'Scientific American & Hamid Originals' },
  { id: 'lotus_mandala', name: 'Lotus', mode: 'circles', category: 'Scientific American & Hamid Originals' },
  { id: 'nested_heart', name: 'Heart', mode: 'lines', category: 'Scientific American & Hamid Originals' },
  { id: 'seahorse_circles', name: 'Seahorse', mode: 'circles', category: 'Scientific American & Hamid Originals' },
];

const FALLBACK_3D_VIDEO_PRESETS = [
  { id: '3d_rose_bloom', name: '3D Rose Corolla Manifold', mode: '3d_surface', promptFallback: 'Gulaab blooming rose', category: '3D Manifolds & 4D Math Video' },
  { id: '3d_torus_knot', name: '3D Harmonic Torus Caustic', mode: '3d_surface', promptFallback: 'Harmonic torus knot manifold', category: '3D Manifolds & 4D Math Video' },
  { id: '3d_bird_wings', name: '3D Soaring Wing Surface', mode: '3d_surface', promptFallback: 'Shaheen eagle in flight', category: '3D Manifolds & 4D Math Video' },
  { id: '3d_nautilus_shell', name: '3D Nautilus Helicospiral', mode: '3d_surface', promptFallback: 'Nautilus spiral shell', category: '3D Manifolds & 4D Math Video' },
  { id: '3d_klein_caustic', name: '3D Harmonic Wave Dome', mode: '3d_surface', promptFallback: 'Faisal Mosque Islamabad with crescent moon', category: '3D Manifolds & 4D Math Video' },
  { id: 'video_soaring_bird', name: '4D Soaring Eagle (Wireframe)', mode: 'math_video', promptFallback: 'Shaheen eagle in flight', category: '3D Manifolds & 4D Math Video' },
  { id: 'video_mechanical_bicycle', name: '4D Mechanical Bicycle (Spinning)', mode: 'math_video', promptFallback: 'A detailed mechanical bicycle', category: '3D Manifolds & 4D Math Video' },
  { id: 'pvid_bengal_tiger', name: '4D Color Video: Bengal Tiger', mode: 'pixel_video', promptFallback: 'a roaring Bengal tiger face in lush jungle', category: '3D Manifolds & 4D Math Video' },
  { id: 'pvid_jellyfish', name: '4D Color Video: Bioluminescent Jellyfish', mode: 'pixel_video', promptFallback: 'a glowing bioluminescent jellyfish swimming in deep ocean', category: '3D Manifolds & 4D Math Video' },
  { id: 'pvid_lighthouse_aurora', name: '4D Color Video: Lighthouse & Aurora', mode: 'pixel_video', promptFallback: 'a crystal lighthouse on a rocky cliff under dancing northern lights aurora', category: '3D Manifolds & 4D Math Video' },
  { id: 'pvid_saturn_astronaut', name: '4D Color Video: Saturn & Astronaut', mode: 'pixel_video', promptFallback: 'an astronaut floating beside Saturn golden rings in cosmic space', category: '3D Manifolds & 4D Math Video' },
  { id: 'rcin_golden_phoenix', name: '3D Raymarch Cinema: Golden Phoenix & Ocean', mode: 'raymarch_cinema', promptFallback: 'a golden phoenix rising over turbulent sapphire ocean waves', category: '3D Manifolds & 4D Math Video' },
  { id: 'rcin_gyroid_cathedral', name: '3D Raymarch Cinema: Hyperbolic Gyroid Cathedral', mode: 'raymarch_cinema', promptFallback: 'a hyperbolic bronze cathedral with glowing gyroid pillars and god rays', category: '3D Manifolds & 4D Math Video' },
];

const VERTEX_SHADER_SRC = `
attribute vec2 a_pos;
varying vec2 v_uv;
void main() {
  v_uv = a_pos * 0.5 + 0.5;
  gl_Position = vec4(a_pos, 0.0, 1.0);
}
`;

function KatexFormula({ tex, lightMode = false }) {
  const html = useMemo(() => {
    try {
      return katex.renderToString(tex || '', {
        throwOnError: false,
        displayMode: true,
      });
    } catch {
      return `<code>${tex}</code>`;
    }
  }, [tex]);

  return (
    <div
      className={`overflow-x-auto py-1.5 px-2 text-sm ${
        lightMode ? 'text-slate-900 hamid-poster-plate' : 'text-cyan-100'
      }`}
      dangerouslySetInnerHTML={{ __html: html }}
    />
  );
}

/**
 * Real-Time 60 FPS WebGL1 GPU Math Video Canvas
 * Compiles any analytical GLSL fragment shader (u_resolution, u_time, v_uv)
 * with bulletproof compile-status checks and graceful fallback.
 */
function WebGLMathVideoCanvas({
  fragmentShader,
  isPlaying,
  speed,
  timeT,
  onTimeSync,
  onShaderError,
  onShaderReady,
  glCanvasRef,
}) {
  const localCanvasRef = useRef(null);
  const effectiveCanvasRef = glCanvasRef || localCanvasRef;
  const isPlayingRef = useRef(isPlaying);
  const speedRef = useRef(speed);
  const timeRef = useRef(timeT);
  const onTimeSyncRef = useRef(onTimeSync);

  useEffect(() => {
    isPlayingRef.current = isPlaying;
  }, [isPlaying]);

  useEffect(() => {
    speedRef.current = speed;
  }, [speed]);

  useEffect(() => {
    onTimeSyncRef.current = onTimeSync;
  }, [onTimeSync]);

  // Keep internal ref synced when user scrubs slider while paused
  useEffect(() => {
    if (!isPlaying || Math.abs(timeRef.current - timeT) > 0.25) {
      timeRef.current = timeT;
    }
  }, [timeT, isPlaying]);

  useEffect(() => {
    const canvas = effectiveCanvasRef.current;
    if (!canvas || !fragmentShader) return;

    const gl =
      canvas.getContext('webgl', {
        preserveDrawingBuffer: true,
        antialias: true,
        alpha: false,
      }) ||
      canvas.getContext('experimental-webgl', {
        preserveDrawingBuffer: true,
        antialias: true,
        alpha: false,
      });

    if (!gl) {
      console.warn('WebGL1 unavailable — falling back to rendered mathematical frame.');
      onShaderError?.('WebGL1 context unavailable');
      return;
    }

    const compileShader = (type, source) => {
      const sh = gl.createShader(type);
      gl.shaderSource(sh, source);
      gl.compileShader(sh);
      if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) {
        const log = gl.getShaderInfoLog(sh) || 'Shader compilation failed';
        gl.deleteShader(sh);
        return { shader: null, error: log };
      }
      return { shader: sh, error: null };
    };

    const vsRes = compileShader(gl.VERTEX_SHADER, VERTEX_SHADER_SRC);
    if (!vsRes.shader) {
      console.warn('WebGL1 vertex shader compile error:', vsRes.error);
      onShaderError?.(vsRes.error);
      return;
    }

    const fsRes = compileShader(gl.FRAGMENT_SHADER, fragmentShader);
    if (!fsRes.shader) {
      console.warn('WebGL1 fragment shader compile error (falling back to PNG):', fsRes.error);
      gl.deleteShader(vsRes.shader);
      onShaderError?.(fsRes.error);
      return;
    }

    const program = gl.createProgram();
    gl.attachShader(program, vsRes.shader);
    gl.attachShader(program, fsRes.shader);
    gl.linkProgram(program);

    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
      const linkErr = gl.getProgramInfoLog(program) || 'Shader link failed';
      console.warn('WebGL1 program link error (falling back to PNG):', linkErr);
      gl.deleteShader(vsRes.shader);
      gl.deleteShader(fsRes.shader);
      gl.deleteProgram(program);
      onShaderError?.(linkErr);
      return;
    }

    gl.useProgram(program);
    onShaderReady?.();

    const posBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, posBuffer);
    gl.bufferData(
      gl.ARRAY_BUFFER,
      new Float32Array([-1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, 1]),
      gl.STATIC_DRAW
    );

    const posLoc = gl.getAttribLocation(program, 'a_pos');
    if (posLoc >= 0) {
      gl.enableVertexAttribArray(posLoc);
      gl.vertexAttribPointer(posLoc, 2, gl.FLOAT, false, 0, 0);
    }

    const uResLoc = gl.getUniformLocation(program, 'u_resolution');
    const uTimeLoc = gl.getUniformLocation(program, 'u_time');

    let rafId = 0;
    let lastTs = performance.now();
    let lastHudSync = lastTs;
    let disposed = false;

    const renderFrame = (now) => {
      if (disposed) return;
      const dtSec = Math.min(0.05, Math.max(0, (now - lastTs) / 1000));
      lastTs = now;

      if (isPlayingRef.current) {
        timeRef.current = (timeRef.current + dtSec * 1.45 * speedRef.current) % (Math.PI * 200);
        if (now - lastHudSync > 65) {
          lastHudSync = now;
          onTimeSyncRef.current?.(timeRef.current);
        }
      }

      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      const rect = canvas.getBoundingClientRect();
      const targetW = Math.max(1, Math.floor(rect.width * dpr));
      const targetH = Math.max(1, Math.floor(rect.height * dpr));
      if (canvas.width !== targetW || canvas.height !== targetH) {
        canvas.width = targetW;
        canvas.height = targetH;
      }

      gl.viewport(0, 0, canvas.width, canvas.height);
      if (uResLoc) gl.uniform2f(uResLoc, canvas.width, canvas.height);
      if (uTimeLoc) gl.uniform1f(uTimeLoc, timeRef.current);

      gl.drawArrays(gl.TRIANGLES, 0, 6);
      rafId = requestAnimationFrame(renderFrame);
    };

    rafId = requestAnimationFrame(renderFrame);

    return () => {
      disposed = true;
      cancelAnimationFrame(rafId);
      gl.deleteBuffer(posBuffer);
      gl.deleteShader(vsRes.shader);
      gl.deleteShader(fsRes.shader);
      gl.deleteProgram(program);
    };
  }, [fragmentShader, effectiveCanvasRef, onShaderError, onShaderReady]);

  return <canvas ref={effectiveCanvasRef} className="gpu-math-canvas" />;
}

export default function App() {
  // Catalog & active artwork data
  const [presets, setPresets] = useState([]);
  const [masterParadigmPresets, setMasterParadigmPresets] = useState([]);
  const [samplePrompts, setSamplePrompts] = useState([]);
  const [artData, setArtData] = useState(null);
  const [originCinemaMode, setOriginCinemaMode] = useState(null);
  const [loading, setLoading] = useState(false);
  const [errorMsg, setErrorMsg] = useState('');

  // Generation parameters
  const [prompt, setPrompt] = useState('Shaheen eagle in flight');
  const [activeSource, setActiveSource] = useState({ type: 'preset', id: 'bird_in_flight' });
  const [mode, setMode] = useState('lines');
  const [weaveStyle, setWeaveStyle] = useState('harmonic_web');
  const [numSegments, setNumSegments] = useState(2000);
  const [numHarmonics, setNumHarmonics] = useState(54);
  const [refineEnabled, setRefineEnabled] = useState(true);
  const [activeCategoryTab, setActiveCategoryTab] = useState('all');

  // Long-Script Multi-Act Cinema Director state
  const [showScriptDrawer, setShowScriptDrawer] = useState(false);
  const [scriptText, setScriptText] = useState(SAMPLE_EPIC_5_ACT_SCRIPT);
  const [cinemaFilmData, setCinemaFilmData] = useState(null);
  const [directingFilm, setDirectingFilm] = useState(false);
  const [selectedActIndex, setSelectedActIndex] = useState(0);
  const [cinemaPlayerView, setCinemaPlayerView] = useState('complete_film'); // 'complete_film' | 'act_gpu_explorer'

  // Animation state (k = 1 .. N)
  const [currentK, setCurrentK] = useState(2000);
  const [isPlaying, setIsPlaying] = useState(true);
  const [animSpeed, setAnimSpeed] = useState(14);
  const [showGrid, setShowGrid] = useState(true);
  const [showLaser, setShowLaser] = useState(true);
  const [themeKey, setThemeKey] = useState('cyan_glow');

  // 2D Pixel-Field, 3D Raymarch Cinema & 4D Color Video GPU Shader state
  const [showHamidPoster, setShowHamidPoster] = useState(true);
  const [liveGpuAnim, setLiveGpuAnim] = useState(true);
  const [shaderError, setShaderError] = useState(false);
  const [shaderCompiledOk, setShaderCompiledOk] = useState(false);
  const [pixelImgVersion, setPixelImgVersion] = useState(0);
  const pixelImageRef = useRef(null);
  const glCanvasRef = useRef(null);

  // GIF / MP4 Mathematical Video Export state
  const [exportingGif, setExportingGif] = useState(false);
  const [exportedGifData, setExportedGifData] = useState(null);

  // 3D Parametric Manifold Orbit state
  const [rotX, setRotX] = useState(-0.34);
  const [rotY, setRotY] = useState(0.55);
  const [zoom3D, setZoom3D] = useState(1.0);
  const [autoSpin3D, setAutoSpin3D] = useState(true);
  const drag3DRef = useRef({ active: false, lastX: 0, lastY: 0 });

  // 4D Continuous Time Video state
  const [timeT, setTimeT] = useState(0.0);
  const [isVideoPlaying, setIsVideoPlaying] = useState(true);
  const [videoSpeed, setVideoSpeed] = useState(1.0);
  const [isRecordingWebm, setIsRecordingWebm] = useState(false);
  const [recordProgress, setRecordProgress] = useState(0);
  const mediaRecorderRef = useRef(null);

  // UI tabs & custom formula lab
  const [rightTab, setRightTab] = useState('equations'); // 'equations' | 'python' | 'custom'
  const [copiedKey, setCopiedKey] = useState('');
  const [customFormulas, setCustomFormulas] = useState({
    X1: '1.5 * (np.sin(2*np.pi*k/N + np.pi/3)**7)',
    Y1: '0.25 * (np.cos(6*np.pi*k/N)**2)',
    X2: '0.2 * np.sin(6*np.pi*k/N + np.pi/5)',
    Y2: '-(2.0/3.0) * (np.sin(4*np.pi*k/N - np.pi/3)**2)',
    X: 'np.cos(14*np.pi*k/N)*(1 - 0.5*(np.sin(2*np.pi*k/N)**2))',
    Y: 'np.sin(14*np.pi*k/N)*(1 - 0.5*(np.cos(2*np.pi*k/N)**2))',
    R: '0.04 + 0.125*(np.sin(28*np.pi*k/N)**4)',
  });

  const canvasRef = useRef(null);

  // Unified LaTeX accessor supporting both `latex` and `latex_equations`
  const activeLatex = useMemo(() => {
    return artData?.latex || artData?.latex_equations || null;
  }, [artData]);

  // Determine whether the WebGL1 60 FPS GPU Shader Canvas should be active
  const hasGlslShader = Boolean(artData?.glsl_fragment_shader);
  const isShowingFullCinemaMovie =
    Boolean(cinemaFilmData) &&
    (mode === 'script_cinema' || artData?.mode === 'script_cinema') &&
    cinemaPlayerView === 'complete_film';

  const isGpuShaderActive =
    !isShowingFullCinemaMovie &&
    hasGlslShader &&
    !shaderError &&
    (artData?.mode === 'pixel_video' ||
      artData?.mode === 'raymarch_cinema' ||
      (artData?.mode === 'script_cinema' && cinemaPlayerView === 'act_gpu_explorer') ||
      ((artData?.mode === 'pixel_field' || artData?.mode === 'hdges') && liveGpuAnim));

  const isPixelBasedMode =
    artData?.mode === 'pixel_field' ||
    artData?.mode === 'pixel_video' ||
    artData?.mode === 'hdges' ||
    artData?.mode === 'raymarch_cinema' ||
    artData?.mode === 'script_cinema';

  const isTimeBasedMode =
    artData?.mode === 'math_video' ||
    artData?.mode === 'pixel_video' ||
    artData?.mode === 'raymarch_cinema' ||
    isGpuShaderActive;

  // Group presets cleanly into Master Paradigms + the 3 classic categories
  const groupedPresets = useMemo(() => {
    const byId = new Map();
    presets.forEach((p) => byId.set(p.id, p));
    masterParadigmPresets.forEach((p) => byId.set(p.id, p));

    // 1. 8 Master Mathematical Paradigms (Stage 1)
    const masterList = [];
    const seenMaster = new Set();
    [...masterParadigmPresets, ...presets]
      .filter(
        (p) =>
          p.id.startsWith('mp_') ||
          p.mode === 'raymarch_cinema' ||
          (p.category && p.category.toLowerCase().includes('paradigm'))
      )
      .forEach((p) => {
        if (!seenMaster.has(p.id)) {
          seenMaster.add(p.id);
          masterList.push(p);
        }
      });
    FALLBACK_MASTER_PARADIGM_PRESETS.forEach((fp) => {
      if (!seenMaster.has(fp.id)) {
        seenMaster.add(fp.id);
        masterList.push(byId.get(fp.id) || fp);
      }
    });

    // 2. 2D Pixel-Field Paintings
    const pixelFieldList = [];
    const seenPixel = new Set();
    presets
      .filter(
        (p) =>
          !seenMaster.has(p.id) &&
          (p.mode === 'pixel_field' ||
            p.id.startsWith('pf_') ||
            (p.category && p.category.toLowerCase().includes('pixel')))
      )
      .forEach((p) => {
        seenPixel.add(p.id);
        pixelFieldList.push(p);
      });
    FALLBACK_PIXEL_FIELD_PRESETS.forEach((fp) => {
      if (!seenPixel.has(fp.id)) {
        pixelFieldList.push(byId.get(fp.id) || fp);
      }
    });

    // 3. Scientific American & Hamid Originals
    const sciamHamidList = [];
    const seenSciam = new Set();
    presets
      .filter(
        (p) =>
          !seenMaster.has(p.id) &&
          !seenPixel.has(p.id) &&
          p.mode !== '3d_surface' &&
          p.mode !== 'math_video' &&
          p.mode !== 'pixel_video' &&
          p.mode !== 'raymarch_cinema' &&
          !p.id.startsWith('3d_') &&
          !p.id.startsWith('vid') &&
          !p.id.startsWith('pvid_') &&
          !p.id.startsWith('rcin_')
      )
      .forEach((p) => {
        seenSciam.add(p.id);
        sciamHamidList.push(p);
      });
    FALLBACK_SCIAM_HAMID_PRESETS.forEach((fp) => {
      if (!seenSciam.has(fp.id)) {
        sciamHamidList.push(byId.get(fp.id) || fp);
      }
    });

    // 4. 3D Manifolds & 4D Math Video
    const manifoldVideoList = [];
    const seen3DVideo = new Set();
    presets
      .filter(
        (p) =>
          !seenMaster.has(p.id) &&
          (p.mode === '3d_surface' ||
            p.mode === 'math_video' ||
            p.mode === 'pixel_video' ||
            p.id.startsWith('3d_') ||
            p.id.startsWith('vid') ||
            p.id.startsWith('pvid_') ||
            p.id.startsWith('rcin_'))
      )
      .forEach((p) => {
        seen3DVideo.add(p.id);
        manifoldVideoList.push(p);
      });
    FALLBACK_3D_VIDEO_PRESETS.forEach((fp) => {
      if (!seen3DVideo.has(fp.id)) {
        manifoldVideoList.push(byId.get(fp.id) || fp);
      }
    });

    return {
      masterParadigms: masterList,
      pixelField: pixelFieldList,
      sciamHamid: sciamHamidList,
      manifoldVideo: manifoldVideoList,
    };
  }, [presets, masterParadigmPresets]);

  // Initial load of presets + Hamid's Bird in Flight
  useEffect(() => {
    async function init() {
      try {
        const res = await fetch('/api/presets');
        const data = await res.json();
        setPresets(data.presets || []);
        setMasterParadigmPresets(data.master_paradigm_presets || []);
        setSamplePrompts(data.sample_prompts || []);
        await handleSelectPreset('bird_in_flight', true);
      } catch {
        setErrorMsg('Unable to connect to the mathematical synthesis server on port 8000.');
      }
    }
    init();
  }, []);

  // Reset shader compilation flags whenever artData changes
  useEffect(() => {
    setShaderError(false);
    setShaderCompiledOk(false);
    setExportedGifData(null);
    if (
      artData?.mode === 'pixel_video' ||
      artData?.mode === 'hdges' ||
      artData?.mode === 'raymarch_cinema' ||
      artData?.mode === 'script_cinema'
    ) {
      setLiveGpuAnim(true);
      setIsVideoPlaying(true);
    } else if (artData?.mode === 'pixel_field') {
      setLiveGpuAnim(Boolean(artData?.glsl_fragment_shader && artData?.animation_capable));
    }
  }, [artData]);

  // Sync custom formula editor when artData updates
  useEffect(() => {
    if (artData?.formulas) {
      setCustomFormulas((prev) => ({
        ...prev,
        ...artData.formulas,
      }));
    }
  }, [artData]);

  // Decode base64 image whenever pixel_field, pixel_video, hdges, raymarch_cinema, or script_cinema artwork loads
  useEffect(() => {
    if (
      (artData?.mode === 'pixel_field' ||
        artData?.mode === 'pixel_video' ||
        artData?.mode === 'hdges' ||
        artData?.mode === 'raymarch_cinema' ||
        artData?.mode === 'script_cinema') &&
      artData?.image_base64
    ) {
      const img = new window.Image();
      img.crossOrigin = 'anonymous';
      img.onload = () => {
        pixelImageRef.current = img;
        setPixelImgVersion((v) => v + 1);
      };
      img.src = artData.image_base64.startsWith('data:')
        ? artData.image_base64
        : `data:image/png;base64,${artData.image_base64}`;
    } else {
      pixelImageRef.current = null;
    }
  }, [artData]);

  // Determine total step count N for the active artwork
  const getTotalSteps = useCallback((data) => {
    if (!data) return 2000;
    if (
      data.mode === 'pixel_field' ||
      data.mode === 'pixel_video' ||
      data.mode === 'hdges' ||
      data.mode === 'raymarch_cinema' ||
      data.mode === 'script_cinema'
    ) {
      return data.N || data.height || 1200;
    }
    if (data.mode === '3d_surface') {
      return data.primitives_3d?.length || data.primitives?.length || data.N || 2000;
    }
    return data.primitives?.length || data.N || 2000;
  }, []);

  // Load 5-Act Sample Script from GET /api/cinema/sample-script (with built-in fallback)
  const handleLoadSampleScript = async () => {
    try {
      const res = await fetch('/api/cinema/sample-script');
      if (res.ok) {
        const data = await res.json();
        if (data?.script) {
          setScriptText(data.script);
          return;
        }
      }
    } catch {
      // Fallback to built-in 5-act epic script
    }
    setScriptText(SAMPLE_EPIC_5_ACT_SCRIPT);
  };

  // Select a specific Act from a directed multi-act film to inspect its equations & 60 FPS GPU shader
  const handleSelectStoryAct = (actObj, idx, switchToGpuExplorer = true) => {
    if (!actObj) return;
    setSelectedActIndex(idx);
    if (switchToGpuExplorer && actObj.glsl_fragment_shader) {
      setCinemaPlayerView('act_gpu_explorer');
    }
    const actLatex = actObj.latex_equations || actObj.latex || {};
    const actPreview = actObj.preview_image_base64 || actObj.image_base64 || '';
    setArtData((prev) => ({
      ...(prev || {}),
      title: `Act ${actObj.act_number || idx + 1}: ${actObj.title || 'Mathematical Act'}`,
      subtitle: actObj.narration || actObj.subtitle || '',
      mode: 'script_cinema',
      N: prev?.height || 325,
      image_base64: actPreview,
      glsl_fragment_shader: actObj.glsl_fragment_shader || prev?.glsl_fragment_shader || '',
      latex: actLatex,
      latex_equations: actLatex,
      primitives: actObj.primitives || [],
      primitives_3d: actObj.primitives_3d || [],
      python_code: actObj.python_code || prev?.python_code || '',
      refinement_passes: 2,
    }));
  };

  // Direct a Complete Multi-Act Mathematical Film via POST /api/cinema/generate-from-script
  const handleDirectLongScriptFilm = async (overrideScript) => {
    const targetScript =
      typeof overrideScript === 'string' && overrideScript.trim()
        ? overrideScript
        : scriptText.trim() || SAMPLE_EPIC_5_ACT_SCRIPT;
    setDirectingFilm(true);
    setLoading(true);
    setErrorMsg('');
    setMode('script_cinema');
    setShowScriptDrawer(true);
    try {
      const res = await fetch('/api/cinema/generate-from-script', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          script: targetScript,
          width: 520,
          height: 325,
          fps: 12,
          frames_per_act: 16,
          transition_frames: 6,
          max_acts: 6,
          refine_acts: refineEnabled,
        }),
      });
      if (!res.ok) {
        const errData = await res.json().catch(() => ({}));
        throw new Error(errData.detail || 'Long-script mathematical film generation failed');
      }
      const data = await res.json();
      setCinemaFilmData(data);
      setCinemaPlayerView('complete_film');
      setSelectedActIndex(0);
      setActiveSource({ type: 'script_cinema', script: targetScript });

      const firstAct = data.acts?.[0] || {};
      const firstLatex = firstAct.latex_equations || firstAct.latex || data.latex || {};
      setArtData({
        title: data.title || 'Multi-Act Mathematical Cinema Film',
        subtitle: `${data.num_acts || data.acts?.length || 5} Sequential Mathematical Acts • ${
          data.total_frames || 0
        } Frames (${data.total_duration_sec || 0}s @ ${data.fps || 12} FPS) • C² Homotopy Morphs`,
        mode: 'script_cinema',
        N: data.height || 325,
        width: data.width || 520,
        height: data.height || 325,
        image_base64: firstAct.preview_image_base64 || data.image_base64 || '',
        glsl_fragment_shader: firstAct.glsl_fragment_shader || data.glsl_fragment_shader || '',
        latex: firstLatex,
        latex_equations: firstLatex,
        primitives: firstAct.primitives || [],
        primitives_3d: firstAct.primitives_3d || [],
        python_code: firstAct.python_code || data.python_code || '',
        refinement_passes: 2,
      });
    } catch (err) {
      setErrorMsg(err.message);
    } finally {
      setDirectingFilm(false);
      setLoading(false);
    }
  };

  // Select a preset (with seamless fallback to prompt endpoint for dynamic 3D/4D items)
  const handleSelectPreset = async (presetOrId, autoAnimate = true) => {
    const presetObj =
      typeof presetOrId === 'object'
        ? presetOrId
        : presets.find((p) => p.id === presetOrId) ||
          masterParadigmPresets.find((p) => p.id === presetOrId) ||
          FALLBACK_MASTER_PARADIGM_PRESETS.find((p) => p.id === presetOrId) ||
          FALLBACK_PIXEL_FIELD_PRESETS.find((p) => p.id === presetOrId) ||
          FALLBACK_SCIAM_HAMID_PRESETS.find((p) => p.id === presetOrId) ||
          FALLBACK_3D_VIDEO_PRESETS.find((p) => p.id === presetOrId) || { id: presetOrId };

    const presetId = presetObj.id;
    setOriginCinemaMode(null);

    if (
      (presetObj.mode === 'pixel_video' || presetObj.mode === 'raymarch_cinema') &&
      presetObj.promptFallback
    ) {
      setPrompt(presetObj.promptFallback);
      setMode(presetObj.mode);
      await handleGenerateFromPrompt(presetObj.promptFallback, { mode: presetObj.mode });
      setActiveSource({ type: 'preset', id: presetId });
      return;
    }

    setLoading(true);
    setErrorMsg('');
    try {
      const res = await fetch('/api/preset', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ preset_id: presetId, N: numSegments }),
      });

      if (!res.ok) {
        if (presetObj.promptFallback) {
          setPrompt(presetObj.promptFallback);
          await handleGenerateFromPrompt(presetObj.promptFallback, {
            mode: presetObj.mode || '3d_surface',
          });
          setActiveSource({ type: 'preset', id: presetId });
          return;
        }
        throw new Error('Preset evaluation failed');
      }

      const data = await res.json();
      const totalN = getTotalSteps(data);
      setArtData(data);
      if (data.mode === 'raymarch_cinema') {
        setOriginCinemaMode('raymarch_cinema');
      }
      setMode(data.mode || presetObj.mode || 'lines');
      if (
        data.N &&
        data.mode !== 'pixel_field' &&
        data.mode !== 'pixel_video' &&
        data.mode !== 'raymarch_cinema'
      ) {
        setNumSegments(data.N);
      }
      setActiveSource({ type: 'preset', id: presetId });

      if (
        data.mode === 'math_video' ||
        data.mode === 'pixel_video' ||
        data.mode === 'raymarch_cinema'
      ) {
        setCurrentK(totalN);
        setIsPlaying(false);
        setTimeT(0);
        setIsVideoPlaying(true);
      } else if (autoAnimate) {
        setCurrentK(1);
        setIsPlaying(true);
      } else {
        setCurrentK(totalN);
      }
    } catch (err) {
      setErrorMsg(err.message);
    } finally {
      setLoading(false);
    }
  };

  // Generate mathematical art from any text prompt across all modes
  const handleGenerateFromPrompt = async (overridePrompt, overrideParams = {}) => {
    const targetMode = overrideParams.mode ?? mode;
    if (targetMode === 'script_cinema') {
      const scriptInput =
        overridePrompt && overridePrompt.length > 60 ? overridePrompt : scriptText;
      await handleDirectLongScriptFilm(scriptInput);
      return;
    }

    const targetPrompt = overridePrompt !== undefined ? overridePrompt : prompt;
    if (!targetPrompt.trim()) return;
    setLoading(true);
    setErrorMsg('');
    setOriginCinemaMode(null);
    const targetN = overrideParams.N ?? numSegments;
    const targetH = overrideParams.harmonics ?? numHarmonics;
    const targetWeave = overrideParams.weave_style ?? weaveStyle;
    const targetRefine = overrideParams.refine ?? refineEnabled;

    try {
      const isHdges = targetMode === 'hdges';
      const endpoint = isHdges ? '/api/hdges/text-synthesize' : '/api/generate-from-prompt';
      const payload = isHdges
        ? {
            prompt: targetPrompt,
            width: 640,
            height: 400,
            N: targetN,
            snap_rationals: true,
          }
        : {
            prompt: targetPrompt,
            N: targetN,
            harmonics: targetH,
            mode: targetMode,
            weave_style: targetWeave,
            refine: targetRefine,
            use_ai: true,
          };
      const res = await fetch(endpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      if (!res.ok) {
        const errData = await res.json().catch(() => ({}));
        throw new Error(errData.detail || 'Mathematical synthesis failed');
      }
      const data = await res.json();
      if (data.acts && Array.isArray(data.acts)) {
        setCinemaFilmData(data);
        setCinemaPlayerView('complete_film');
        setSelectedActIndex(0);
      }
      const totalN = getTotalSteps(data);
      setArtData(data);
      if (data.mode === 'raymarch_cinema') {
        setOriginCinemaMode('raymarch_cinema');
      }
      if (data.mode) setMode(data.mode);
      setActiveSource({ type: 'prompt', prompt: targetPrompt });

      if (
        data.mode === 'math_video' ||
        data.mode === 'pixel_video' ||
        data.mode === 'hdges' ||
        data.mode === 'raymarch_cinema' ||
        data.mode === 'script_cinema'
      ) {
        setCurrentK(totalN);
        setIsPlaying(false);
        setTimeT(0);
        setIsVideoPlaying(true);
      } else if (overrideParams.keepProgress) {
        setCurrentK(totalN);
        setIsPlaying(false);
      } else {
        setCurrentK(1);
        setIsPlaying(true);
      }
    } catch (err) {
      setErrorMsg(err.message);
    } finally {
      setLoading(false);
    }
  };

  // H-DGES Inverse Closed-Form Formula Distillation
  const handleHdgesDistill = async (targetPresetId) => {
    setLoading(true);
    setErrorMsg('');
    setOriginCinemaMode(null);
    try {
      const res = await fetch('/api/hdges/distill', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          target_preset_id: targetPresetId,
          width: 640,
          height: 400,
          N: 1500,
          snap_rationals: true,
          run_ablation: true,
        }),
      });
      if (!res.ok) {
        const errData = await res.json().catch(() => ({}));
        throw new Error(errData.detail || 'H-DGES inverse formula distillation failed');
      }
      const data = await res.json();
      setArtData(data);
      setMode('hdges');
      setActiveSource({ type: 'hdges_distill', id: targetPresetId });
      setCurrentK(getTotalSteps(data));
      setIsPlaying(false);
      setTimeT(0);
      setIsVideoPlaying(true);
    } catch (err) {
      setErrorMsg(err.message);
    } finally {
      setLoading(false);
    }
  };

  // Export Looping Math GIF via POST /api/render-math-video
  const handleExportLoopingGif = async () => {
    const isPresetSource = activeSource.type === 'preset' && Boolean(activeSource.id);
    const targetPrompt =
      activeSource.type === 'prompt' ? activeSource.prompt : artData?.title || prompt;
    if (!isPresetSource && (!targetPrompt || !targetPrompt.trim())) return;
    setExportingGif(true);
    setErrorMsg('');
    try {
      const payload = {
        prompt: targetPrompt || 'Mathematical Cinema',
        mode: artData?.mode || mode || 'raymarch_cinema',
        width: 480,
        height: 300,
        num_frames: 32,
        fps: 16,
      };
      if (isPresetSource) {
        payload.preset_id = activeSource.id;
      }

      const res = await fetch('/api/render-math-video', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      if (!res.ok) {
        const errData = await res.json().catch(() => ({}));
        throw new Error(errData.detail || 'Looping Math GIF rendering failed');
      }
      const data = await res.json();
      const rawGif = data.gif_base64 || '';
      if (!rawGif) {
        throw new Error('Server did not return gif_base64');
      }
      const gifUri = rawGif.startsWith('data:') ? rawGif : `data:image/gif;base64,${rawGif}`;
      setExportedGifData({
        gifUri,
        mp4Uri: data.mp4_base64
          ? data.mp4_base64.startsWith('data:')
            ? data.mp4_base64
            : `data:video/mp4;base64,${data.mp4_base64}`
          : null,
        motionEnergy: data.motion_energy,
      });

      const link = document.createElement('a');
      link.download = `${(artData?.title || targetPrompt || 'math-cinema')
        .toLowerCase()
        .replace(/[^a-z0-9]+/g, '-')}-math-loop.gif`;
      link.href = gifUri;
      link.click();
    } catch (err) {
      setErrorMsg(err.message);
    } finally {
      setExportingGif(false);
    }
  };

  // Evaluate custom user formulas
  const handleRunCustomFormulas = async () => {
    setLoading(true);
    setErrorMsg('');
    try {
      const evalMode = mode === 'circles' ? 'circles' : 'lines';
      const payloadFormulas =
        evalMode === 'lines'
          ? {
              X1: customFormulas.X1,
              Y1: customFormulas.Y1,
              X2: customFormulas.X2,
              Y2: customFormulas.Y2,
            }
          : {
              X: customFormulas.X,
              Y: customFormulas.Y,
              R: customFormulas.R,
            };
      const res = await fetch('/api/evaluate-custom', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          mode: evalMode,
          N: numSegments,
          formulas: payloadFormulas,
        }),
      });
      if (!res.ok) {
        const errData = await res.json().catch(() => ({}));
        throw new Error(errData.detail || 'Invalid formula syntax');
      }
      const data = await res.json();
      setArtData(data);
      setMode(data.mode || evalMode);
      setCurrentK(1);
      setIsPlaying(true);
    } catch (err) {
      setErrorMsg(err.message);
    } finally {
      setLoading(false);
    }
  };

  // Animation loop: increments k from 1 to N
  useEffect(() => {
    if (!isPlaying || !artData || isGpuShaderActive || isShowingFullCinemaMovie) return;
    const total = getTotalSteps(artData);
    if (currentK >= total) {
      setIsPlaying(false);
      return;
    }
    let rafId;
    const step = () => {
      setCurrentK((prev) => {
        const stepDelta =
          artData.mode === 'pixel_field' ||
          artData.mode === 'pixel_video' ||
          artData.mode === 'raymarch_cinema'
            ? Math.max(4, Math.round(animSpeed * 0.85))
            : animSpeed;
        const next = prev + stepDelta;
        if (next >= total) {
          setIsPlaying(false);
          return total;
        }
        return next;
      });
      rafId = requestAnimationFrame(step);
    };
    rafId = requestAnimationFrame(step);
    return () => cancelAnimationFrame(rafId);
  }, [
    isPlaying,
    artData,
    animSpeed,
    currentK,
    getTotalSteps,
    isGpuShaderActive,
    isShowingFullCinemaMovie,
  ]);

  // 3D Auto-Spin loop when in 3d_surface mode
  useEffect(() => {
    if (artData?.mode !== '3d_surface' || !autoSpin3D) return;
    let rafId;
    const spinStep = () => {
      if (!drag3DRef.current.active) {
        setRotY((prev) => prev + 0.0075);
      }
      rafId = requestAnimationFrame(spinStep);
    };
    rafId = requestAnimationFrame(spinStep);
    return () => cancelAnimationFrame(rafId);
  }, [artData?.mode, autoSpin3D]);

  // 4D Continuous Time t evolution at 60 FPS when in wireframe math_video mode
  useEffect(() => {
    if (artData?.mode !== 'math_video' || !isVideoPlaying) return;
    let rafId;
    let lastTs = performance.now();
    const stepTime = (now) => {
      const dtSec = Math.min(0.05, (now - lastTs) / 1000);
      lastTs = now;
      setTimeT((prev) => (prev + dtSec * 2.2 * videoSpeed) % (Math.PI * 200));
      rafId = requestAnimationFrame(stepTime);
    };
    rafId = requestAnimationFrame(stepTime);
    return () => cancelAnimationFrame(rafId);
  }, [artData?.mode, isVideoPlaying, videoSpeed]);

  // Evaluate 4D time-evolving wireframe primitive at index i and time t
  const evaluateVideoPrimitiveAtTime = useCallback((data, i, t) => {
    const basePrims = data.primitives || [];
    const p0 = basePrims[i] || [0, 0, 0, 0];
    const thList = data.time_harmonics;

    if (Array.isArray(thList) && thList[i]) {
      const h = thList[i];
      const sinT = Math.sin(t);
      const cosT = Math.cos(t);
      const sin2T = Math.sin(2 * t);

      if (Array.isArray(h)) {
        let ax1 = h[0] || 0,
          bx1 = h[1] || 0,
          ay1 = h[2] || 0,
          by1 = h[3] || 0,
          ax2 = h[4] || 0,
          bx2 = h[5] || 0,
          ay2 = h[6] || 0,
          by2 = h[7] || 0;

        if (data.time_harmonics_order === 'sin_first') {
          ax1 = h[0] || 0;
          ay1 = h[1] || 0;
          ax2 = h[2] || 0;
          ay2 = h[3] || 0;
          bx1 = h[4] || 0;
          by1 = h[5] || 0;
          bx2 = h[6] || 0;
          by2 = h[7] || 0;
        }

        const cx1 = h[8] || 0;
        const cy1 = h[9] || 0;
        const cx2 = h[10] || 0;
        const cy2 = h[11] || 0;

        return [
          p0[0] + ax1 * sinT + bx1 * cosT + cx1 * sin2T,
          p0[1] + ay1 * sinT + by1 * cosT + cy1 * sin2T,
          (p0[2] ?? 0) + ax2 * sinT + bx2 * cosT + cx2 * sin2T,
          (p0[3] ?? 0) + ay2 * sinT + by2 * cosT + cy2 * sin2T,
        ];
      } else if (typeof h === 'object') {
        return [
          p0[0] + (h.ax1 || 0) * sinT + (h.bx1 || 0) * cosT + (h.cx1 || 0) * sin2T,
          p0[1] + (h.ay1 || 0) * sinT + (h.by1 || 0) * cosT + (h.cy1 || 0) * sin2T,
          (p0[2] ?? 0) + (h.ax2 || 0) * sinT + (h.bx2 || 0) * cosT + (h.cx2 || 0) * sin2T,
          (p0[3] ?? 0) + (h.ay2 || 0) * sinT + (h.by2 || 0) * cosT + (h.cy2 || 0) * sin2T,
        ];
      }
    }

    if (Array.isArray(data.frames) && data.frames.length > 0) {
      const M = data.frames.length;
      const twoPi = Math.PI * 2;
      const normT = ((t % twoPi) + twoPi) % twoPi;
      const exactIdx = (normT / twoPi) * M;
      const idx0 = Math.floor(exactIdx) % M;
      const idx1 = (idx0 + 1) % M;
      const frac = exactIdx - Math.floor(exactIdx);
      const f0 = data.frames[idx0]?.[i] || p0;
      const f1 = data.frames[idx1]?.[i] || p0;
      return [
        f0[0] * (1 - frac) + f1[0] * frac,
        f0[1] * (1 - frac) + f1[1] * frac,
        (f0[2] ?? 0) * (1 - frac) + (f1[2] ?? 0) * frac,
        (f0[3] ?? 0) * (1 - frac) + (f1[3] ?? 0) * frac,
      ];
    }

    const total = Math.max(1, basePrims.length);
    const u = i / total;
    const wave = 0.045 * Math.sin(t + u * Math.PI * 4);
    const flap = 0.06 * Math.sin(t) * Math.abs(p0[0]);
    return [p0[0] + wave * 0.5, p0[1] + flap, (p0[2] ?? 0) - wave * 0.5, (p0[3] ?? 0) + flap * 0.8];
  }, []);

  // Render 2D canvas modes (lines, circles, pixel_field/fallback, 3d_surface, math_video)
  const renderCanvas = useCallback(() => {
    if (isGpuShaderActive || isShowingFullCinemaMovie) return;
    const canvas = canvasRef.current;
    if (!canvas || !artData) return;
    const ctx = canvas.getContext('2d');
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    const width = rect.width;
    const height = rect.height;
    if (width <= 0 || height <= 0) return;

    if (canvas.width !== Math.floor(width * dpr) || canvas.height !== Math.floor(height * dpr)) {
      canvas.width = Math.floor(width * dpr);
      canvas.height = Math.floor(height * dpr);
    }

    ctx.save();
    ctx.scale(dpr, dpr);

    const theme = COLOR_THEMES[themeKey] || COLOR_THEMES.cyan_glow;
    ctx.fillStyle = theme.bg;
    ctx.fillRect(0, 0, width, height);

    const cx = width / 2;
    const cy = height / 2;
    const scale = Math.min(width, height) * 0.41;

    // 2D PIXEL-FIELD PAINTING OR PIXEL-VIDEO / RAYMARCH-CINEMA / HDGES / SCRIPT-CINEMA FALLBACK
    if (
      artData.mode === 'pixel_field' ||
      artData.mode === 'pixel_video' ||
      artData.mode === 'hdges' ||
      artData.mode === 'raymarch_cinema' ||
      artData.mode === 'script_cinema'
    ) {
      const img = pixelImageRef.current;
      const totalRows = getTotalSteps(artData);
      const revealRatio =
        artData.mode === 'pixel_video' ||
        artData.mode === 'raymarch_cinema' ||
        artData.mode === 'hdges' ||
        artData.mode === 'script_cinema'
          ? 1.0
          : Math.min(1, Math.max(0.01, currentK / Math.max(1, totalRows)));

      if (img && img.width > 0 && img.height > 0) {
        const pad = 20;
        const availW = width - pad * 2;
        const availH = height - pad * 2;
        const imgAspect = img.width / img.height;
        let drawW = availW;
        let drawH = drawW / imgAspect;
        if (drawH > availH) {
          drawH = availH;
          drawW = drawH * imgAspect;
        }
        const drawX = (width - drawW) / 2;
        const drawY = (height - drawH) / 2;

        ctx.strokeStyle = 'rgba(56, 189, 248, 0.25)';
        ctx.lineWidth = 1;
        ctx.strokeRect(drawX - 1, drawY - 1, drawW + 2, drawH + 2);

        const visibleH = drawH * revealRatio;
        ctx.save();
        ctx.beginPath();
        ctx.rect(drawX, drawY, drawW, visibleH);
        ctx.clip();
        ctx.drawImage(img, drawX, drawY, drawW, drawH);
        ctx.restore();

        if (showLaser && revealRatio < 1.0) {
          const scanY = drawY + visibleH;
          const grad = ctx.createLinearGradient(drawX, scanY - 4, drawX, scanY + 4);
          grad.addColorStop(0, 'rgba(56, 189, 248, 0)');
          grad.addColorStop(0.5, '#38bdf8');
          grad.addColorStop(1, 'rgba(251, 191, 36, 0)');
          ctx.fillStyle = grad;
          ctx.fillRect(drawX, scanY - 3, drawW, 6);
        }
      } else {
        ctx.fillStyle = '#94a3b8';
        ctx.font = '13px monospace';
        ctx.textAlign = 'center';
        ctx.fillText('Evaluating Mathematical Field & Implicit Level Sets...', cx, cy);
      }

      ctx.restore();
      return;
    }

    // 3D PARAMETRIC MANIFOLD
    if (artData.mode === '3d_surface') {
      const rawPrims =
        artData.primitives_3d && artData.primitives_3d.length > 0
          ? artData.primitives_3d
          : artData.primitives || [];
      const total = rawPrims.length;
      const limit = Math.min(currentK, total);

      const cosY = Math.cos(rotY);
      const sinY = Math.sin(rotY);
      const cosX = Math.cos(rotX);
      const sinX = Math.sin(rotX);
      const fov = 3.2;

      const project3D = (x, y, z) => {
        const x1 = x * cosY + z * sinY;
        const z1 = -x * sinY + z * cosY;
        const y2 = y * cosX - z1 * sinX;
        const z2 = y * sinX + z1 * cosX;
        const persp = (fov / Math.max(0.45, fov - z2)) * zoom3D;
        return {
          sx: cx + x1 * scale * persp,
          sy: cy - y2 * scale * persp,
          zDepth: z2,
          persp,
        };
      };

      if (showGrid) {
        ctx.strokeStyle = theme.grid;
        ctx.lineWidth = 1;
        ctx.beginPath();
        const ringSteps = 64;
        for (let r = 0; r <= ringSteps; r++) {
          const ang = (r / ringSteps) * Math.PI * 2;
          const pt = project3D(Math.cos(ang), 0, Math.sin(ang));
          if (r === 0) ctx.moveTo(pt.sx, pt.sy);
          else ctx.lineTo(pt.sx, pt.sy);
        }
        ctx.stroke();

        const axes = [
          { from: [-1.15, 0, 0], to: [1.15, 0, 0], label: '+X', color: 'rgba(56, 189, 248, 0.45)' },
          { from: [0, -1.15, 0], to: [0, 1.15, 0], label: '+Y', color: 'rgba(251, 191, 36, 0.45)' },
          { from: [0, 0, -1.15], to: [0, 0, 1.15], label: '+Z', color: 'rgba(244, 63, 94, 0.45)' },
        ];
        ctx.font = '11px monospace';
        axes.forEach((ax) => {
          const pFrom = project3D(ax.from[0], ax.from[1], ax.from[2]);
          const pTo = project3D(ax.to[0], ax.to[1], ax.to[2]);
          ctx.strokeStyle = ax.color;
          ctx.lineWidth = 1.1;
          ctx.beginPath();
          ctx.moveTo(pFrom.sx, pFrom.sy);
          ctx.lineTo(pTo.sx, pTo.sy);
          ctx.stroke();
          ctx.fillStyle = ax.color;
          ctx.fillText(ax.label, pTo.sx + 5, pTo.sy + 4);
        });
      }

      for (let i = 0; i < limit; i++) {
        const p = rawPrims[i];
        const t = i / Math.max(1, total);
        const x1 = p[0] ?? 0;
        const y1 = p[1] ?? 0;
        const z1 = p.length >= 6 ? p[2] : 0.28 * Math.sin(t * Math.PI * 4);
        const x2 = p.length >= 6 ? p[3] : p[2] ?? 0;
        const y2 = p.length >= 6 ? p[4] : p[3] ?? 0;
        const z2 = p.length >= 6 ? p[5] : 0.28 * Math.cos(t * Math.PI * 4);

        const pt1 = project3D(x1, y1, z1);
        const pt2 = project3D(x2, y2, z2);
        const zMid = 0.5 * (pt1.zDepth + pt2.zDepth);
        const depthNorm = Math.max(0, Math.min(1, (zMid + 1.1) / 2.2));
        const alpha = 0.16 + 0.56 * depthNorm;
        const [h, s, l] = theme.strokeHsl(t);
        const lDepth = Math.min(88, Math.max(38, l + (depthNorm - 0.5) * 26));

        ctx.strokeStyle = `hsla(${h}, ${s}%, ${lDepth}%, ${alpha})`;
        ctx.lineWidth = 0.45 + 0.85 * depthNorm;
        ctx.beginPath();
        ctx.moveTo(pt1.sx, pt1.sy);
        ctx.lineTo(pt2.sx, pt2.sy);
        ctx.stroke();
      }

      if (showLaser && limit > 0 && limit < total) {
        const p = rawPrims[limit - 1];
        const t = (limit - 1) / Math.max(1, total);
        const x1 = p[0] ?? 0;
        const y1 = p[1] ?? 0;
        const z1 = p.length >= 6 ? p[2] : 0.28 * Math.sin(t * Math.PI * 4);
        const x2 = p.length >= 6 ? p[3] : p[2] ?? 0;
        const y2 = p.length >= 6 ? p[4] : p[3] ?? 0;
        const z2 = p.length >= 6 ? p[5] : 0.28 * Math.cos(t * Math.PI * 4);

        const pt1 = project3D(x1, y1, z1);
        const pt2 = project3D(x2, y2, z2);

        ctx.strokeStyle = theme.activeLine;
        ctx.lineWidth = 2.1;
        ctx.beginPath();
        ctx.moveTo(pt1.sx, pt1.sy);
        ctx.lineTo(pt2.sx, pt2.sy);
        ctx.stroke();

        ctx.fillStyle = theme.p1;
        ctx.beginPath();
        ctx.arc(pt1.sx, pt1.sy, 4.5, 0, Math.PI * 2);
        ctx.fill();

        ctx.fillStyle = theme.p2;
        ctx.beginPath();
        ctx.arc(pt2.sx, pt2.sy, 4.5, 0, Math.PI * 2);
        ctx.fill();
      }

      ctx.restore();
      return;
    }

    // 2D LINES, 2D CIRCLES, AND 4D WIREFRAME MATH VIDEO
    if (showGrid) {
      ctx.strokeStyle = theme.grid;
      ctx.lineWidth = 1;
      const gridStep = scale / 4;
      for (let gx = cx % gridStep; gx < width; gx += gridStep) {
        ctx.beginPath();
        ctx.moveTo(gx, 0);
        ctx.lineTo(gx, height);
        ctx.stroke();
      }
      for (let gy = cy % gridStep; gy < height; gy += gridStep) {
        ctx.beginPath();
        ctx.moveTo(0, gy);
        ctx.lineTo(width, gy);
        ctx.stroke();
      }
      ctx.strokeStyle = theme.axis;
      ctx.beginPath();
      ctx.moveTo(0, cy);
      ctx.lineTo(width, cy);
      ctx.moveTo(cx, 0);
      ctx.lineTo(cx, height);
      ctx.stroke();
    }

    const primitives = artData.primitives || [];
    const total = primitives.length;
    const limit = Math.min(currentK, total);
    const isCircles = artData.mode === 'circles';
    const isVideo = artData.mode === 'math_video';

    ctx.lineWidth = isCircles ? 0.85 : 0.78;

    for (let i = 0; i < limit; i++) {
      const t = i / Math.max(1, total);
      ctx.strokeStyle = theme.stroke(t);
      const p = isVideo ? evaluateVideoPrimitiveAtTime(artData, i, timeT) : primitives[i];
      if (!p) continue;

      if (!isCircles) {
        const x1 = cx + p[0] * scale;
        const y1 = cy - p[1] * scale;
        const x2 = cx + p[2] * scale;
        const y2 = cy - p[3] * scale;
        ctx.beginPath();
        ctx.moveTo(x1, y1);
        ctx.lineTo(x2, y2);
        ctx.stroke();
      } else {
        const x = cx + p[0] * scale;
        const y = cy - p[1] * scale;
        const r = Math.max(0.5, (p[2] ?? 0.02) * scale);
        ctx.beginPath();
        ctx.arc(x, y, r, 0, Math.PI * 2);
        ctx.stroke();
      }
    }

    if (showLaser && limit > 0 && limit < total) {
      const idx = limit - 1;
      const p = isVideo ? evaluateVideoPrimitiveAtTime(artData, idx, timeT) : primitives[idx];
      if (p) {
        if (!isCircles) {
          const x1 = cx + p[0] * scale;
          const y1 = cy - p[1] * scale;
          const x2 = cx + p[2] * scale;
          const y2 = cy - p[3] * scale;

          ctx.strokeStyle = theme.activeLine;
          ctx.lineWidth = 2.0;
          ctx.beginPath();
          ctx.moveTo(x1, y1);
          ctx.lineTo(x2, y2);
          ctx.stroke();

          ctx.fillStyle = theme.p1;
          ctx.beginPath();
          ctx.arc(x1, y1, 4.5, 0, Math.PI * 2);
          ctx.fill();

          ctx.fillStyle = theme.p2;
          ctx.beginPath();
          ctx.arc(x2, y2, 4.5, 0, Math.PI * 2);
          ctx.fill();
        } else {
          const x = cx + p[0] * scale;
          const y = cy - p[1] * scale;
          const r = Math.max(1, (p[2] ?? 0.02) * scale);

          ctx.strokeStyle = theme.activeLine;
          ctx.lineWidth = 2.0;
          ctx.beginPath();
          ctx.arc(x, y, r, 0, Math.PI * 2);
          ctx.stroke();

          ctx.fillStyle = theme.p1;
          ctx.beginPath();
          ctx.arc(x, y, 4, 0, Math.PI * 2);
          ctx.fill();
        }
      }
    }

    ctx.restore();
  }, [
    isGpuShaderActive,
    isShowingFullCinemaMovie,
    artData,
    currentK,
    showGrid,
    showLaser,
    themeKey,
    rotX,
    rotY,
    zoom3D,
    timeT,
    pixelImgVersion,
    getTotalSteps,
    evaluateVideoPrimitiveAtTime,
  ]);

  useEffect(() => {
    renderCanvas();
    window.addEventListener('resize', renderCanvas);
    return () => window.removeEventListener('resize', renderCanvas);
  }, [renderCanvas]);

  // Mouse handlers for 3D Orbit Rotation
  const handleCanvasMouseDown = (e) => {
    if (artData?.mode !== '3d_surface') return;
    drag3DRef.current = { active: true, lastX: e.clientX, lastY: e.clientY };
  };

  const handleCanvasMouseMove = (e) => {
    if (artData?.mode !== '3d_surface' || !drag3DRef.current.active) return;
    const dx = e.clientX - drag3DRef.current.lastX;
    const dy = e.clientY - drag3DRef.current.lastY;
    drag3DRef.current.lastX = e.clientX;
    drag3DRef.current.lastY = e.clientY;
    setRotY((prev) => prev + dx * 0.0085);
    setRotX((prev) => Math.max(-1.45, Math.min(1.45, prev + dy * 0.0085)));
  };

  const handleCanvasMouseUp = () => {
    drag3DRef.current.active = false;
  };

  // Export High-Res PNG of active Canvas (supports both WebGL1 GPU canvas and 2D canvas)
  const handleExportPNG = () => {
    const activeCanvas = isGpuShaderActive ? glCanvasRef.current : canvasRef.current;
    if (!activeCanvas) return;
    const link = document.createElement('a');
    link.download = `${(artData?.title || 'math-art').toLowerCase().replace(/[^a-z0-9]+/g, '-')}.png`;
    link.href = activeCanvas.toDataURL('image/png');
    link.click();
  };

  // 1-Click Composite & Download Full Hamid Publication Math Poster (PNG)
  const handleDownloadMathPoster = () => {
    if (!artData) return;
    const img = pixelImageRef.current;
    const liveGl = isGpuShaderActive ? glCanvasRef.current : null;
    const posterW = 1600;
    const pad = 64;
    const innerW = posterW - pad * 2;
    const srcW = liveGl?.width || img?.width || 2000;
    const srcH = liveGl?.height || img?.height || 1200;
    const imgAspect = srcW > 0 && srcH > 0 ? srcW / srcH : 2000 / 1200;
    const artH = Math.round(innerW / imgAspect);

    const latexEntries = Object.entries(activeLatex || {});
    const plateH = 340 + latexEntries.length * 95;
    const posterH = pad + artH + plateH + pad;

    const offCanvas = document.createElement('canvas');
    offCanvas.width = posterW;
    offCanvas.height = posterH;
    const ctx = offCanvas.getContext('2d');

    ctx.fillStyle = '#ffffff';
    ctx.fillRect(0, 0, posterW, posterH);

    if (liveGl) {
      ctx.drawImage(liveGl, pad, pad, innerW, artH);
    } else if (img && img.width > 0) {
      ctx.drawImage(img, pad, pad, innerW, artH);
    } else if (canvasRef.current) {
      ctx.drawImage(canvasRef.current, pad, pad, innerW, artH);
    }

    ctx.strokeStyle = '#cbd5e1';
    ctx.lineWidth = 2;
    ctx.strokeRect(pad, pad, innerW, artH);

    const nomW = artData.nominal_aspect?.[0] || 2000;
    const nomH = artData.nominal_aspect?.[1] || 1200;
    let cursorY = pad + artH + 54;

    ctx.fillStyle = '#0f172a';
    ctx.font = '24px "Georgia", "Times New Roman", serif';
    const line1 = `The above picture is a converted version of a ${nomW} × ${nomH} image. For m = 1, 2, ..., ${nomW} and n = 1, 2, ..., ${nomH},`;
    const line2 = `the color of the pixel of the row n and the column m is rgb(F(H₀((m-${Math.round(nomW / 2)})/${Math.round(nomH / 2)}, (${Math.round(nomH / 2) + 1}-n)/${Math.round(nomH / 2)})), F(H₁(...)), F(H₂(...))),`;
    const line3 = `where F(x) = ⌊ 255 · exp(-exp(-1000x)) · |x|^(exp(-exp(1000(x - 1)))) ⌋.`;

    ctx.fillText(line1, pad, cursorY);
    cursorY += 36;
    ctx.fillText(line2, pad, cursorY);
    cursorY += 36;
    ctx.fillStyle = '#334155';
    ctx.font = 'italic 22px "Georgia", "Times New Roman", serif';
    ctx.fillText(line3, pad, cursorY);
    cursorY += 44;

    ctx.strokeStyle = '#e2e8f0';
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(pad, cursorY);
    ctx.lineTo(posterW - pad, cursorY);
    ctx.stroke();
    cursorY += 40;

    latexEntries.forEach(([key, texStr]) => {
      ctx.fillStyle = '#0369a1';
      ctx.font = 'bold 19px "Courier New", monospace';
      ctx.fillText(`[${key}]`, pad, cursorY);

      ctx.fillStyle = '#0f172a';
      ctx.font = '19px "Cambria Math", "Georgia", serif';
      const cleaned = String(texStr || '')
        .replace(/\\operatorname\{rgb\}/g, 'rgb')
        .replace(/\\left/g, '')
        .replace(/\\right/g, '')
        .replace(/\\lfloor/g, '⌊')
        .replace(/\\rfloor/g, '⌋')
        .replace(/\\frac\{([^{}]+)\}\{([^{}]+)\}/g, '($1/$2)')
        .replace(/\\sum_\{([^{}]+)\}\^\{([^{}]+)\}/g, 'Σ_{$1..$2}')
        .replace(/\\prod_\{([^{}]+)\}\^\{([^{}]+)\}/g, 'Π_{$1..$2}')
        .replace(/\\cos/g, 'cos')
        .replace(/\\sin/g, 'sin')
        .replace(/\\arctan/g, 'arctan')
        .replace(/\\arccos/g, 'arccos')
        .replace(/\\sqrt/g, '√')
        .replace(/\\pi/g, 'π')
        .replace(/\\quad/g, '   ')
        .replace(/\\!/g, '')
        .replace(/\\,/g, ' ');

      const maxChars = 118;
      const lineA = cleaned.slice(0, maxChars);
      const lineB = cleaned.length > maxChars ? cleaned.slice(maxChars, maxChars * 2) : '';
      ctx.fillText(lineA, pad + 95, cursorY);
      if (lineB) {
        cursorY += 28;
        ctx.fillText(lineB + (cleaned.length > maxChars * 2 ? ' ...' : ''), pad + 95, cursorY);
      }
      cursorY += 54;
    });

    ctx.fillStyle = '#64748b';
    ctx.font = '18px "Georgia", serif';
    ctx.fillText(
      `${artData.title || 'Mathematical Pixel-Field Painting'} — Pure Double-Exponential Trigonometric Specification`,
      pad,
      posterH - 28
    );

    const link = document.createElement('a');
    link.download = `${(artData.title || 'hamid-math-poster').toLowerCase().replace(/[^a-z0-9]+/g, '-')}-poster.png`;
    link.href = offCanvas.toDataURL('image/png');
    link.click();
  };

  // 1-Click Record & Download 60 FPS Math Video (.webm) from active GPU or 2D Canvas
  const handleRecordMathVideo = () => {
    const activeCanvas = isGpuShaderActive ? glCanvasRef.current : canvasRef.current;
    if (!activeCanvas) return;

    if (isRecordingWebm && mediaRecorderRef.current) {
      mediaRecorderRef.current.stop();
      return;
    }

    if (typeof activeCanvas.captureStream !== 'function' || typeof window.MediaRecorder === 'undefined') {
      setErrorMsg('WebM MediaRecorder is not supported in this browser.');
      return;
    }

    try {
      setIsVideoPlaying(true);
      const stream = activeCanvas.captureStream(60);
      const mimeTypes = ['video/webm;codecs=vp9', 'video/webm;codecs=vp8', 'video/webm'];
      const selectedMime = mimeTypes.find((m) => window.MediaRecorder.isTypeSupported(m)) || '';
      const recorder = selectedMime
        ? new window.MediaRecorder(stream, { mimeType: selectedMime, videoBitsPerSecond: 8000000 })
        : new window.MediaRecorder(stream);

      const chunks = [];
      recorder.ondataavailable = (e) => {
        if (e.data && e.data.size > 0) chunks.push(e.data);
      };
      recorder.onstop = () => {
        setIsRecordingWebm(false);
        setRecordProgress(0);
        const blob = new Blob(chunks, { type: selectedMime || 'video/webm' });
        const url = URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.download = `${(artData?.title || '4d-math-video').toLowerCase().replace(/[^a-z0-9]+/g, '-')}.webm`;
        link.href = url;
        link.click();
        URL.revokeObjectURL(url);
      };

      mediaRecorderRef.current = recorder;
      recorder.start(100);
      setIsRecordingWebm(true);
      setRecordProgress(5);

      const durationMs = 4800;
      const startMs = performance.now();
      const tickInterval = setInterval(() => {
        const elapsed = performance.now() - startMs;
        const pct = Math.min(100, Math.round((elapsed / durationMs) * 100));
        setRecordProgress(pct);
        if (elapsed >= durationMs) {
          clearInterval(tickInterval);
          if (recorder.state === 'recording') {
            recorder.stop();
          }
        }
      }, 120);
    } catch (err) {
      setIsRecordingWebm(false);
      setErrorMsg(`Video capture failed: ${err.message}`);
    }
  };

  // Export Pure Vector SVG directly from the mathematical coordinates
  const handleExportSVG = () => {
    if (!artData) return;
    const size = 1000;
    const cx = size / 2;
    const cy = size / 2;
    const scale = size * 0.41;
    const isCircles = artData.mode === 'circles';
    const is3D = artData.mode === '3d_surface';
    const isVideo = artData.mode === 'math_video';

    const rawPrims = is3D
      ? artData.primitives_3d?.length
        ? artData.primitives_3d
        : artData.primitives || []
      : artData.primitives || [];

    const total = Math.max(1, rawPrims.length);
    const cosY = Math.cos(rotY);
    const sinY = Math.sin(rotY);
    const cosX = Math.cos(rotX);
    const sinX = Math.sin(rotX);

    const elements = rawPrims.map((rawP, idx) => {
      const p = isVideo ? evaluateVideoPrimitiveAtTime(artData, idx, timeT) : rawP;
      const hue = Math.round(192 + Math.sin((idx / total) * Math.PI * 2) * 22);
      if (is3D) {
        const x1 = p[0] ?? 0,
          y1 = p[1] ?? 0,
          z1 = p[2] ?? 0;
        const x2 = p[3] ?? 0,
          y2 = p[4] ?? 0,
          z2 = p[5] ?? 0;
        const proj = (x, y, z) => {
          const rx = x * cosY + z * sinY;
          const rz = -x * sinY + z * cosY;
          const ry = y * cosX - rz * sinX;
          const zz = y * sinX + rz * cosX;
          const persp = (3.2 / Math.max(0.45, 3.2 - zz)) * zoom3D;
          return [(cx + rx * scale * persp).toFixed(2), (cy - ry * scale * persp).toFixed(2)];
        };
        const [sx1, sy1] = proj(x1, y1, z1);
        const [sx2, sy2] = proj(x2, y2, z2);
        return `<line x1="${sx1}" y1="${sy1}" x2="${sx2}" y2="${sy2}" stroke="hsl(${hue}, 95%, 64%)" stroke-opacity="0.45" stroke-width="0.8" />`;
      }
      if (!isCircles) {
        const x1 = (cx + p[0] * scale).toFixed(2);
        const y1 = (cy - p[1] * scale).toFixed(2);
        const x2 = (cx + p[2] * scale).toFixed(2);
        const y2 = (cy - p[3] * scale).toFixed(2);
        return `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="hsl(${hue}, 95%, 64%)" stroke-opacity="0.42" stroke-width="0.8" />`;
      } else {
        const x = (cx + p[0] * scale).toFixed(2);
        const y = (cy - p[1] * scale).toFixed(2);
        const r = Math.max(0.5, p[2] * scale).toFixed(2);
        return `<circle cx="${x}" cy="${y}" r="${r}" fill="none" stroke="hsl(${hue}, 95%, 64%)" stroke-opacity="0.42" stroke-width="0.8" />`;
      }
    });

    const svgContent = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${size} ${size}" width="${size}" height="${size}">
  <rect width="100%" height="100%" fill="#050811" />
  <g>${elements.join('\n    ')}</g>
</svg>`;

    const blob = new Blob([svgContent], { type: 'image/svg+xml;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.download = `${(artData.title || 'math-art').toLowerCase().replace(/[^a-z0-9]+/g, '-')}.svg`;
    link.href = url;
    link.click();
    URL.revokeObjectURL(url);
  };

  // Copy text helper
  const handleCopy = (key, text) => {
    navigator.clipboard.writeText(text);
    setCopiedKey(key);
    setTimeout(() => setCopiedKey(''), 2000);
  };

  // Instant zero-latency mode view switch when raymarch_cinema has extracted 3D/2D manifolds
  const handleSwitchExtractedManifoldView = (targetViewMode) => {
    if (!artData) return;
    if (artData.mode === 'raymarch_cinema') {
      setOriginCinemaMode('raymarch_cinema');
    }
    const nextData = { ...artData, mode: targetViewMode };
    const nextSteps = getTotalSteps(nextData);
    setArtData(nextData);
    setCurrentK(nextSteps);
    setIsPlaying(false);
  };

  const totalSteps = getTotalSteps(artData);

  // Active primitive coordinates for live HUD overlay
  const activePrimitive = useMemo(() => {
    if (!artData || currentK <= 0) return null;
    if (artData.mode === '3d_surface') {
      const list = artData.primitives_3d?.length ? artData.primitives_3d : artData.primitives || [];
      return list.length ? list[Math.min(currentK - 1, list.length - 1)] : null;
    }
    if (artData.mode === 'math_video') {
      const list = artData.primitives || [];
      if (!list.length) return null;
      const idx = Math.min(currentK - 1, list.length - 1);
      return evaluateVideoPrimitiveAtTime(artData, idx, timeT);
    }
    const list = artData.primitives || [];
    return list.length ? list[Math.min(currentK - 1, list.length - 1)] : null;
  }, [artData, currentK, timeT, evaluateVideoPrimitiveAtTime]);

  const activeModeBadge =
    MODE_DEFINITIONS.find((m) => m.id === (artData?.mode || mode))?.badge ||
    '2D Parametric Line Segments (X₁(k), Y₁(k)) → (X₂(k), Y₂(k))';

  const cinemaGifUri = useMemo(() => {
    if (!cinemaFilmData?.gif_base64) return null;
    return cinemaFilmData.gif_base64.startsWith('data:')
      ? cinemaFilmData.gif_base64
      : `data:image/gif;base64,${cinemaFilmData.gif_base64}`;
  }, [cinemaFilmData]);

  const cinemaMp4Uri = useMemo(() => {
    if (!cinemaFilmData?.mp4_base64) return null;
    return cinemaFilmData.mp4_base64.startsWith('data:')
      ? cinemaFilmData.mp4_base64
      : `data:video/mp4;base64,${cinemaFilmData.mp4_base64}`;
  }, [cinemaFilmData]);

  return (
    <div className="min-h-screen bg-[#050811] text-slate-100 flex flex-col">
      {/* Top Navbar */}
      <header className="border-b border-slate-800/80 bg-[#080d1a]/95 backdrop-blur-md px-5 py-3.5 flex flex-wrap items-center justify-between gap-4 sticky top-0 z-30">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-cyan-500 to-blue-600 flex items-center justify-center shadow-lg shadow-cyan-500/20">
            <Compass className="w-6 h-6 text-white" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h1 className="text-lg font-bold tracking-tight bg-gradient-to-r from-cyan-300 via-sky-200 to-amber-200 bg-clip-text text-transparent">
                DesignWithMath
              </h1>
              <span className="text-[11px] font-medium px-2 py-0.5 rounded-full bg-cyan-500/15 text-cyan-300 border border-cyan-500/30">
                Universal Mathematical Art &amp; 3D Raymarched Cinema Studio
              </span>
            </div>
            <p className="text-xs text-slate-400">
              Multi-Act Long-Script Cinema Director • 3D Implicit SDF Raymarching • Differential Geometry &amp; PDEs
            </p>
          </div>
        </div>

        {/* Pure-Math Prompt Input Bar + Mode Selector + Long-Script Movie Director Button */}
        <form
          onSubmit={(e) => {
            e.preventDefault();
            handleGenerateFromPrompt();
          }}
          className="flex-1 max-w-4xl flex flex-wrap items-center gap-2"
        >
          <div className="relative flex-1 min-w-[200px]">
            <Sparkles className="w-4 h-4 text-cyan-400 absolute left-3.5 top-1/2 -translate-y-1/2" />
            <input
              type="text"
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              placeholder="Enter any 3D mathematical scene or subject: e.g. 'golden phoenix over sapphire ocean', 'hyperbolic gyroid cathedral'..."
              className="w-full pl-10 pr-4 py-2.5 bg-slate-900/90 border border-slate-700/80 rounded-xl text-sm text-slate-100 placeholder-slate-500 focus:outline-none focus:border-cyan-400 focus:ring-1 focus:ring-cyan-400/40"
            />
          </div>

          <select
            value={mode}
            onChange={(e) => {
              const nextMode = e.target.value;
              setMode(nextMode);
              if (nextMode === 'script_cinema') {
                setShowScriptDrawer(true);
              } else {
                handleGenerateFromPrompt(prompt, { mode: nextMode });
              }
            }}
            className="px-3 py-2.5 rounded-xl bg-slate-900 border border-slate-700/80 text-xs font-semibold text-cyan-200 focus:outline-none focus:border-cyan-400"
          >
            {MODE_DEFINITIONS.map((m) => (
              <option key={m.id} value={m.id}>
                {m.label}
              </option>
            ))}
          </select>

          <button
            type="button"
            onClick={() => {
              setMode('script_cinema');
              setShowScriptDrawer((v) => !v);
            }}
            className={`px-3 py-2.5 rounded-xl border text-xs font-semibold flex items-center gap-1.5 transition ${
              mode === 'script_cinema' || showScriptDrawer
                ? 'bg-amber-500/20 border-amber-400 text-amber-200 shadow-md shadow-amber-500/10'
                : 'bg-slate-900 border-slate-700 text-amber-300 hover:border-amber-400/60'
            }`}
          >
            <Clapperboard className="w-4 h-4" />
            <span>Long-Script Movie Director</span>
          </button>

          <label
            title="Enable 2-Pass Analytical Equation Refinement for higher geometric fidelity"
            className="flex items-center gap-1.5 px-2.5 py-2 rounded-xl bg-slate-900/90 border border-slate-700/80 text-xs text-slate-200 cursor-pointer select-none hover:border-emerald-500/50 transition"
          >
            <input
              type="checkbox"
              checked={refineEnabled}
              onChange={(e) => setRefineEnabled(e.target.checked)}
              className="accent-emerald-400 w-3.5 h-3.5 rounded cursor-pointer"
            />
            <span className="font-medium whitespace-nowrap">2-Pass Analytical Refinement</span>
          </label>

          <button
            type="submit"
            disabled={loading}
            className="px-4 py-2.5 rounded-xl bg-gradient-to-r from-cyan-500 to-blue-600 hover:from-cyan-400 hover:to-blue-500 text-slate-950 font-semibold text-sm flex items-center gap-2 shadow-lg shadow-cyan-500/20 disabled:opacity-50 transition"
          >
            <Wand2 className="w-4 h-4" />
            <span>{loading ? 'Synthesizing Equations...' : 'Synthesize Math'}</span>
          </button>
        </form>
      </header>

      {/* Categorized Masterpiece Presets & Prompt Ideas Catalog */}
      <div className="bg-[#080d1a]/80 border-b border-slate-800/80 px-5 py-2.5 space-y-2">
        {/* Category Filter Pills + Sample Prompts */}
        <div className="flex flex-wrap items-center justify-between gap-2">
          <div className="flex flex-wrap items-center gap-1.5">
            <button
              onClick={() => setActiveCategoryTab('all')}
              className={`px-2.5 py-1 rounded-lg text-[11px] font-semibold uppercase tracking-wider transition border ${
                activeCategoryTab === 'all'
                  ? 'bg-cyan-500/20 border-cyan-400/60 text-cyan-200'
                  : 'bg-slate-900/70 border-slate-800 text-slate-400 hover:text-slate-200'
              }`}
            >
              All Categories
            </button>
            <button
              onClick={() => setActiveCategoryTab('master_paradigms')}
              className={`px-2.5 py-1 rounded-lg text-[11px] font-semibold flex items-center gap-1 transition border ${
                activeCategoryTab === 'master_paradigms'
                  ? 'bg-rose-500/20 border-rose-400/60 text-rose-200'
                  : 'bg-slate-900/70 border-slate-800 text-rose-300/90 hover:text-rose-200'
              }`}
            >
              <Clapperboard className="w-3 h-3" />
              <span>8 Master Mathematical Paradigms (Quilez • Ferguson • Floraform • Thurston • Xenakis • Houdini • Manim)</span>
            </button>
            <button
              onClick={() => setActiveCategoryTab('pixel_field')}
              className={`px-2.5 py-1 rounded-lg text-[11px] font-semibold flex items-center gap-1 transition border ${
                activeCategoryTab === 'pixel_field'
                  ? 'bg-emerald-500/20 border-emerald-400/60 text-emerald-200'
                  : 'bg-slate-900/70 border-slate-800 text-emerald-300/80 hover:text-emerald-200'
              }`}
            >
              <Palette className="w-3 h-3" />
              <span>2D Pixel-Field Paintings</span>
            </button>
            <button
              onClick={() => setActiveCategoryTab('sciam_hamid')}
              className={`px-2.5 py-1 rounded-lg text-[11px] font-semibold flex items-center gap-1 transition border ${
                activeCategoryTab === 'sciam_hamid'
                  ? 'bg-amber-500/20 border-amber-400/60 text-amber-200'
                  : 'bg-slate-900/70 border-slate-800 text-amber-300/80 hover:text-amber-200'
              }`}
            >
              <BookOpen className="w-3 h-3" />
              <span>Scientific American &amp; Hamid Originals</span>
            </button>
            <button
              onClick={() => setActiveCategoryTab('3d_video')}
              className={`px-2.5 py-1 rounded-lg text-[11px] font-semibold flex items-center gap-1 transition border ${
                activeCategoryTab === '3d_video'
                  ? 'bg-purple-500/20 border-purple-400/60 text-purple-200'
                  : 'bg-slate-900/70 border-slate-800 text-purple-300/80 hover:text-purple-200'
              }`}
            >
              <Box className="w-3 h-3" />
              <span>3D Manifolds &amp; 4D Math Video</span>
            </button>
          </div>

          {samplePrompts.length > 0 && (
            <div className="flex items-center gap-1.5 overflow-x-auto max-w-full">
              <span className="text-[11px] font-semibold uppercase tracking-wider text-cyan-400 shrink-0">
                Mathematical Prompts:
              </span>
              {samplePrompts.map((sp) => (
                <button
                  key={sp.label}
                  onClick={() => {
                    const targetMode = sp.mode || mode || 'raymarch_cinema';
                    setPrompt(sp.prompt);
                    setMode(targetMode);
                    handleGenerateFromPrompt(sp.prompt, { mode: targetMode });
                  }}
                  className="px-2.5 py-1 rounded-lg text-xs bg-slate-900/60 hover:bg-cyan-500/15 border border-slate-800 hover:border-cyan-500/40 text-slate-300 hover:text-cyan-200 shrink-0 transition"
                >
                  {sp.label}
                </button>
              ))}
            </div>
          )}
        </div>

        {/* Category 0: 8 Master Mathematical Paradigms */}
        {(activeCategoryTab === 'all' || activeCategoryTab === 'master_paradigms') && (
          <div className="flex items-center gap-1.5 overflow-x-auto py-0.5">
            <span className="text-[11px] font-semibold uppercase tracking-wider text-rose-400 shrink-0 flex items-center gap-1 pr-1">
              <Clapperboard className="w-3.5 h-3.5" /> 8 Master Mathematical Paradigms:
            </span>
            {groupedPresets.masterParadigms.map((p) => {
              const isSelected = activeSource.type === 'preset' && activeSource.id === p.id;
              return (
                <button
                  key={p.id}
                  onClick={() => handleSelectPreset(p, true)}
                  className={`px-2.5 py-1 rounded-lg text-xs font-medium shrink-0 transition border ${
                    isSelected
                      ? 'bg-rose-500/20 border-rose-400/60 text-rose-200 shadow-sm shadow-rose-500/10'
                      : 'bg-slate-900/70 border-slate-800 text-slate-200 hover:border-rose-500/40 hover:text-white'
                  }`}
                >
                  {p.name}
                </button>
              );
            })}
          </div>
        )}

        {/* Category 1: 2D Pixel-Field Paintings */}
        {(activeCategoryTab === 'all' || activeCategoryTab === 'pixel_field') && (
          <div className="flex items-center gap-1.5 overflow-x-auto py-0.5">
            <span className="text-[11px] font-semibold uppercase tracking-wider text-emerald-400 shrink-0 flex items-center gap-1 pr-1">
              <Palette className="w-3.5 h-3.5" /> 2D Pixel-Field Paintings:
            </span>
            {groupedPresets.pixelField.map((p) => {
              const isSelected = activeSource.type === 'preset' && activeSource.id === p.id;
              return (
                <button
                  key={p.id}
                  onClick={() => handleSelectPreset(p, true)}
                  className={`px-2.5 py-1 rounded-lg text-xs font-medium shrink-0 transition border ${
                    isSelected
                      ? 'bg-emerald-500/20 border-emerald-400/60 text-emerald-200 shadow-sm shadow-emerald-500/10'
                      : 'bg-slate-900/70 border-slate-800 text-slate-300 hover:border-emerald-500/40 hover:text-white'
                  }`}
                >
                  {p.name.replace(/\s*\(2D Pixel-Field\)/i, '')}
                </button>
              );
            })}
          </div>
        )}

        {/* Category 2: Scientific American & Hamid Originals */}
        {(activeCategoryTab === 'all' || activeCategoryTab === 'sciam_hamid') && (
          <div className="flex items-center gap-1.5 overflow-x-auto py-0.5">
            <span className="text-[11px] font-semibold uppercase tracking-wider text-amber-400 shrink-0 flex items-center gap-1 pr-1">
              <BookOpen className="w-3.5 h-3.5" /> Scientific American &amp; Hamid Originals:
            </span>
            {groupedPresets.sciamHamid.map((p) => {
              const isSelected = activeSource.type === 'preset' && activeSource.id === p.id;
              return (
                <button
                  key={p.id}
                  onClick={() => handleSelectPreset(p, true)}
                  className={`px-2.5 py-1 rounded-lg text-xs font-medium shrink-0 transition border ${
                    isSelected
                      ? 'bg-amber-500/20 border-amber-400/60 text-amber-200 shadow-sm shadow-amber-500/10'
                      : 'bg-slate-900/70 border-slate-800 text-slate-300 hover:border-amber-500/40 hover:text-white'
                  }`}
                >
                  {p.name}
                </button>
              );
            })}
          </div>
        )}

        {/* Category 3: 3D Manifolds & 4D Math Video */}
        {(activeCategoryTab === 'all' || activeCategoryTab === '3d_video') && (
          <div className="flex items-center gap-1.5 overflow-x-auto py-0.5">
            <span className="text-[11px] font-semibold uppercase tracking-wider text-purple-400 shrink-0 flex items-center gap-1 pr-1">
              <Orbit className="w-3.5 h-3.5" /> 3D Manifolds &amp; 4D Math Video:
            </span>
            {groupedPresets.manifoldVideo.map((p) => {
              const isSelected = activeSource.type === 'preset' && activeSource.id === p.id;
              return (
                <button
                  key={p.id}
                  onClick={() => handleSelectPreset(p, true)}
                  className={`px-2.5 py-1 rounded-lg text-xs font-medium shrink-0 transition border ${
                    isSelected
                      ? 'bg-purple-500/20 border-purple-400/60 text-purple-200 shadow-sm shadow-purple-500/10'
                      : 'bg-slate-900/70 border-slate-800 text-slate-300 hover:border-purple-500/40 hover:text-white'
                  }`}
                >
                  {p.name}
                </button>
              );
            })}
          </div>
        )}

        {/* Category 4: H-DGES Homotopy Inverse Formula Distillation */}
        <div className="flex items-center gap-1.5 overflow-x-auto py-0.5 border-t border-slate-800/60 pt-1.5">
          <span className="text-[11px] font-semibold uppercase tracking-wider text-teal-300 shrink-0 flex items-center gap-1 pr-1">
            <Cpu className="w-3.5 h-3.5" /> H-DGES Inverse Distillation (Pure-Calculus Homotopy RSMP):
          </span>
          {[
            { id: 'pf_kiwi', label: 'Distill Kiwi → Rational Formula' },
            { id: 'pf_strawberries', label: 'Distill Strawberries → Rational Formula' },
            { id: 'pf_corals', label: 'Distill Corals → Rational Formula' },
            { id: 'pf_peacock', label: 'Distill Peacock → Rational Formula' },
            { id: 'pf_bipolar_nebula', label: 'Distill Nebula → Rational Formula' },
          ].map((item) => (
            <button
              key={item.id}
              disabled={loading}
              onClick={() => handleHdgesDistill(item.id)}
              className={`px-2.5 py-1 rounded-lg text-xs font-medium shrink-0 transition border disabled:opacity-50 ${
                activeSource.type === 'hdges_distill' && activeSource.id === item.id
                  ? 'bg-teal-500/20 border-teal-400/70 text-teal-200'
                  : 'bg-slate-900/70 border-slate-800 text-teal-200/85 hover:border-teal-400/50 hover:text-white'
              }`}
            >
              {item.label}
            </button>
          ))}
        </div>
      </div>

      {/* LONG-SCRIPT MULTI-ACT MATHEMATICAL CINEMA DIRECTOR DRAWER / PANEL */}
      {(mode === 'script_cinema' || showScriptDrawer) && (
        <div className="mx-5 mt-3 p-4 rounded-2xl bg-gradient-to-r from-[#0b1224] via-[#101830] to-[#131129] border border-amber-500/40 shadow-2xl space-y-3">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <div className="flex items-center gap-2">
              <Clapperboard className="w-5 h-5 text-amber-400" />
              <div>
                <h3 className="text-sm font-bold text-amber-200">
                  Long-Script Multi-Act Mathematical Cinema Director
                </h3>
                <p className="text-xs text-slate-400">
                  Decomposes any multi-act script into M sequential mathematical acts across the 8 Master Paradigms &amp; 3D Raymarching Engine, stitched via C² Homotopy Morph Transitions with burned-in narration subtitles &amp; live formula ticker.
                </p>
              </div>
            </div>

            <div className="flex flex-wrap items-center gap-2">
              <button
                type="button"
                onClick={handleLoadSampleScript}
                className="px-3 py-1.5 rounded-xl bg-slate-900 hover:bg-slate-800 border border-slate-700 text-xs font-medium text-cyan-200 flex items-center gap-1.5 transition"
              >
                <FileText className="w-3.5 h-3.5 text-cyan-400" />
                <span>Load 5-Act Sample Script</span>
              </button>

              <button
                type="button"
                disabled={directingFilm}
                onClick={() => handleDirectLongScriptFilm(scriptText)}
                className="px-4 py-1.5 rounded-xl bg-gradient-to-r from-amber-500 to-rose-500 hover:from-amber-400 hover:to-rose-400 text-slate-950 font-bold text-xs flex items-center gap-1.5 shadow-lg shadow-amber-500/20 disabled:opacity-50 transition"
              >
                <Film className="w-3.5 h-3.5" />
                <span>
                  {directingFilm
                    ? 'Directing & Raymarching Multi-Act Film...'
                    : 'Direct Complete Mathematical Film (MP4 + GIF)'}
                </span>
              </button>
            </div>
          </div>

          <textarea
            rows={5}
            value={scriptText}
            onChange={(e) => setScriptText(e.target.value)}
            placeholder="Write or paste a multi-act or multi-paragraph mathematical movie script (ACT 1: ... ACT 2: ...)"
            className="w-full p-3 rounded-xl bg-slate-950/90 border border-slate-800 text-xs font-mono text-slate-200 focus:outline-none focus:border-amber-400 leading-relaxed"
          />
        </div>
      )}

      {/* H-DGES Homotopy Banner */}
      {artData?.mode === 'hdges' && (
        <div className="mx-5 mt-3 px-4 py-2.5 rounded-xl bg-teal-950/35 border border-teal-500/35 text-xs flex flex-wrap items-center justify-between gap-3">
          <div className="flex flex-wrap items-center gap-2 text-teal-100">
            <span className="px-2 py-0.5 rounded bg-teal-500/20 border border-teal-400/40 font-semibold text-teal-200">
              {artData.hdges_submode === 'inverse_distillation'
                ? 'H-DGES Inverse Formula Distillation (RSMP + κ-Homotopy)'
                : 'H-DGES Zero-Diffusion Variational Homotopy'}
            </span>
            {artData.psnr != null && (
              <span className="font-mono text-emerald-300">
                PSNR: {artData.psnr} dB • SSIM: {artData.ssim}
              </span>
            )}
            {artData.rational_snapping?.snapped && (
              <span className="font-mono text-cyan-200">
                Rational Snap: {artData.rational_snapping.parameters_snapped} fracs (Δ {artData.rational_snapping.psnr_drop} dB)
              </span>
            )}
            {artData.ablation && (
              <span className="font-mono text-amber-300">
                Homotopy Gain vs Fixed κ=180: +{artData.ablation.psnr_gain_db} dB
              </span>
            )}
            <span className="text-slate-300">
              RK2 Hamiltonian Streamlines ({artData.primitives?.length || 0} segs, 0 OpenCV)
            </span>
          </div>
          {artData.primitives?.length > 0 && (
            <button
              onClick={() =>
                setArtData((prev) => ({
                  ...prev,
                  mode: prev.mode === 'hdges' ? 'lines' : 'hdges',
                }))
              }
              className="px-2.5 py-1 rounded-lg bg-slate-900 border border-teal-500/40 text-teal-200 hover:bg-teal-500/20 font-medium transition"
            >
              View RK2 Hamiltonian 2D Streamlines
            </button>
          )}
        </div>
      )}

      {errorMsg && (
        <div className="mx-5 mt-3 px-4 py-2.5 rounded-xl bg-rose-500/15 border border-rose-500/40 text-rose-200 text-xs flex items-center justify-between">
          <span>{errorMsg}</span>
          <button onClick={() => setErrorMsg('')} className="text-rose-300 hover:text-white">
            ✕
          </button>
        </div>
      )}

      {/* Main Split Workspace */}
      <main className="flex-1 grid grid-cols-1 lg:grid-cols-12 gap-4 p-4 max-w-[1800px] w-full mx-auto">
        {/* LEFT 7 COLS: Interactive Mathematical Canvas + Complete Multi-Act Film Player + Storyboard */}
        <section className="lg:col-span-7 flex flex-col bg-[#080d1a] border border-slate-800/80 rounded-2xl overflow-hidden shadow-2xl">
          {/* Canvas Top Bar */}
          <div className="px-4 py-3 border-b border-slate-800/80 flex flex-wrap items-center justify-between gap-2 bg-slate-900/40">
            <div className="space-y-1">
              <h2 className="text-sm font-semibold text-slate-100 flex flex-wrap items-center gap-2">
                <span>{artData?.title || 'Loading Mathematical Synthesis...'}</span>
                <span className="text-[11px] px-2 py-0.5 rounded-md bg-slate-800 text-cyan-300 font-mono">
                  {activeModeBadge}
                </span>

                {artData?.refinement_passes && (
                  <span className="text-[11px] px-2 py-0.5 rounded-md bg-emerald-500/15 border border-emerald-500/40 text-emerald-300 font-medium flex items-center gap-1">
                    <ShieldCheck className="w-3 h-3" />
                    <span>
                      {artData.refinement_passes >= 2
                        ? `${artData.refinement_passes}-Pass Verified Equations`
                        : '1-Pass Analytical Equations'}
                    </span>
                  </span>
                )}

                {hasGlslShader && !shaderError && (
                  <span className="text-[11px] px-2 py-0.5 rounded-md bg-fuchsia-500/15 border border-fuchsia-500/40 text-fuchsia-300 font-medium flex items-center gap-1">
                    <Cpu className="w-3 h-3" />
                    <span>60 FPS GPU Shader</span>
                  </span>
                )}
              </h2>
              <p className="text-xs text-slate-400">{artData?.subtitle}</p>
            </div>

            {/* Mode-Aware Actions & Exports */}
            <div className="flex flex-wrap items-center gap-2">
              {/* Complete Multi-Act Film Player vs Interactive 60 FPS GPU Act Explorer Toggle */}
              {cinemaFilmData && (mode === 'script_cinema' || artData?.mode === 'script_cinema') && (
                <>
                  <button
                    onClick={() =>
                      setCinemaPlayerView((v) =>
                        v === 'complete_film' ? 'act_gpu_explorer' : 'complete_film'
                      )
                    }
                    className="px-2.5 py-1.5 rounded-lg bg-amber-500/20 border border-amber-400/60 text-amber-200 text-xs font-semibold flex items-center gap-1.5 transition"
                  >
                    <Clapperboard className="w-3.5 h-3.5" />
                    <span>
                      {cinemaPlayerView === 'complete_film'
                        ? `Interactive 60 FPS GPU Act Explorer (Act ${selectedActIndex + 1})`
                        : 'Return to Complete Stitched Multi-Act Movie'}
                    </span>
                  </button>
                  {cinemaMp4Uri && (
                    <a
                      href={cinemaMp4Uri}
                      download="complete_script_film.mp4"
                      className="px-2.5 py-1.5 rounded-lg bg-gradient-to-r from-amber-500 to-orange-500 text-slate-950 font-semibold text-xs flex items-center gap-1 shadow-md transition"
                    >
                      <Download className="w-3.5 h-3.5" />
                      <span>Download Complete MP4 Movie</span>
                    </a>
                  )}
                  {cinemaGifUri && (
                    <a
                      href={cinemaGifUri}
                      download="complete_script_film.gif"
                      className="px-2.5 py-1.5 rounded-lg bg-fuchsia-500/20 border border-fuchsia-400/60 text-fuchsia-200 font-semibold text-xs flex items-center gap-1 transition"
                    >
                      <Download className="w-3.5 h-3.5" />
                      <span>Download Complete GIF</span>
                    </a>
                  )}
                </>
              )}

              {/* 1-Click 3D Raymarch Cinema <-> Extracted 3D Manifold <-> 2D Geodesic Curves Switcher */}
              {(artData?.mode === 'raymarch_cinema' || originCinemaMode === 'raymarch_cinema') && (
                <>
                  {artData?.mode !== 'raymarch_cinema' && hasGlslShader && (
                    <button
                      onClick={() => handleSwitchExtractedManifoldView('raymarch_cinema')}
                      className="px-2.5 py-1.5 rounded-lg bg-rose-500/20 border border-rose-400/60 text-rose-200 text-xs font-semibold flex items-center gap-1.5 transition"
                    >
                      <Clapperboard className="w-3.5 h-3.5" />
                      <span>Return to 3D Raymarch GPU Cinema</span>
                    </button>
                  )}
                  {artData?.primitives_3d?.length > 0 && artData?.mode !== '3d_surface' && (
                    <button
                      onClick={() => handleSwitchExtractedManifoldView('3d_surface')}
                      className="px-2.5 py-1.5 rounded-lg bg-purple-500/20 hover:bg-purple-500/30 border border-purple-400/60 text-purple-200 text-xs font-medium flex items-center gap-1.5 transition"
                    >
                      <Orbit className="w-3.5 h-3.5" />
                      <span>Orbit Extracted 3D Manifold</span>
                    </button>
                  )}
                  {artData?.primitives?.length > 0 && artData?.mode !== 'lines' && (
                    <button
                      onClick={() => handleSwitchExtractedManifoldView('lines')}
                      className="px-2.5 py-1.5 rounded-lg bg-cyan-500/15 hover:bg-cyan-500/25 border border-cyan-400/50 text-cyan-200 text-xs font-medium flex items-center gap-1.5 transition"
                    >
                      <Compass className="w-3.5 h-3.5" />
                      <span>View 2D Geodesic Curves</span>
                    </button>
                  )}
                </>
              )}

              {hasGlslShader && !shaderError && isPixelBasedMode && !isShowingFullCinemaMovie && (
                <button
                  onClick={() => setLiveGpuAnim((v) => !v)}
                  className={`px-2.5 py-1.5 rounded-lg border text-xs font-medium flex items-center gap-1.5 transition ${
                    liveGpuAnim
                      ? 'bg-fuchsia-500/20 border-fuchsia-400/60 text-fuchsia-200'
                      : 'bg-slate-900 border-slate-700 text-slate-300 hover:text-white'
                  }`}
                >
                  <Cpu className="w-3.5 h-3.5" />
                  <span>{liveGpuAnim ? 'Live 60 FPS GPU Animation: ON' : 'Live 60 FPS GPU Animation'}</span>
                </button>
              )}

              {isPixelBasedMode && (
                <>
                  <button
                    onClick={() => setShowHamidPoster((v) => !v)}
                    className={`px-2.5 py-1.5 rounded-lg border text-xs font-medium flex items-center gap-1.5 transition ${
                      showHamidPoster
                        ? 'bg-emerald-500/20 border-emerald-400/60 text-emerald-200'
                        : 'bg-slate-900 border-slate-700 text-slate-300 hover:text-white'
                    }`}
                  >
                    <FileImage className="w-3.5 h-3.5" />
                    <span>{showHamidPoster ? 'Poster View: ON' : 'Poster View'}</span>
                  </button>
                  <button
                    onClick={handleDownloadMathPoster}
                    className="px-2.5 py-1.5 rounded-lg bg-gradient-to-r from-emerald-500 to-teal-500 hover:from-emerald-400 hover:to-teal-400 text-slate-950 font-semibold text-xs flex items-center gap-1.5 shadow-md shadow-emerald-500/20 transition"
                  >
                    <Download className="w-3.5 h-3.5" />
                    <span>Download Hamid Math Poster (PNG)</span>
                  </button>
                </>
              )}

              {artData?.mode === '3d_surface' && (
                <>
                  <button
                    onClick={() => setAutoSpin3D((v) => !v)}
                    className={`px-2.5 py-1.5 rounded-lg border text-xs font-medium flex items-center gap-1.5 transition ${
                      autoSpin3D
                        ? 'bg-purple-500/20 border-purple-400/60 text-purple-200'
                        : 'bg-slate-900 border-slate-700 text-slate-300'
                    }`}
                  >
                    <Orbit className="w-3.5 h-3.5" />
                    <span>Auto-Spin: {autoSpin3D ? 'ON' : 'OFF'}</span>
                  </button>
                  <button
                    onClick={() => {
                      setRotX(-0.34);
                      setRotY(0.55);
                      setZoom3D(1.0);
                    }}
                    title="Reset 3D Orbit Camera"
                    className="px-2 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 border border-slate-700 text-xs text-slate-200 flex items-center gap-1 transition"
                  >
                    <RefreshCw className="w-3.5 h-3.5 text-cyan-400" />
                    <span>Reset 3D</span>
                  </button>
                </>
              )}

              {isTimeBasedMode && !isShowingFullCinemaMovie && (
                <>
                  <button
                    onClick={handleRecordMathVideo}
                    className={`px-3 py-1.5 rounded-lg font-semibold text-xs flex items-center gap-1.5 shadow-md transition ${
                      isRecordingWebm
                        ? 'bg-rose-500 text-white animate-pulse shadow-rose-500/30'
                        : 'bg-gradient-to-r from-amber-500 to-orange-500 hover:from-amber-400 hover:to-orange-400 text-slate-950 shadow-amber-500/20'
                    }`}
                  >
                    <Video className="w-3.5 h-3.5" />
                    <span>
                      {isRecordingWebm
                        ? `Recording 60 FPS WebM (${recordProgress}%)...`
                        : isGpuShaderActive
                        ? 'Record 60 FPS Full-Color Video (WebM)'
                        : 'Record & Download Math Video (.webm)'}
                    </span>
                  </button>

                  {isPixelBasedMode && (
                    <button
                      onClick={handleExportLoopingGif}
                      disabled={exportingGif}
                      className="px-3 py-1.5 rounded-lg bg-gradient-to-r from-fuchsia-500 to-pink-500 hover:from-fuchsia-400 hover:to-pink-400 text-slate-950 font-semibold text-xs flex items-center gap-1.5 shadow-md shadow-fuchsia-500/20 disabled:opacity-50 transition"
                    >
                      <Clapperboard className="w-3.5 h-3.5" />
                      <span>{exportingGif ? 'Rendering Math GIF...' : 'Export Looping Math GIF'}</span>
                    </button>
                  )}
                </>
              )}

              {!isPixelBasedMode && (
                <>
                  <select
                    value={themeKey}
                    onChange={(e) => setThemeKey(e.target.value)}
                    className="bg-slate-900 border border-slate-700/80 rounded-lg px-2.5 py-1.5 text-xs text-slate-200 focus:outline-none focus:border-cyan-400"
                  >
                    {Object.entries(COLOR_THEMES).map(([k, v]) => (
                      <option key={k} value={k}>
                        {v.name}
                      </option>
                    ))}
                  </select>

                  <button
                    onClick={() => setShowGrid((g) => !g)}
                    title="Toggle Coordinate Grid & Axes"
                    className={`p-1.5 rounded-lg border text-xs transition ${
                      showGrid
                        ? 'bg-cyan-500/15 border-cyan-500/40 text-cyan-300'
                        : 'bg-slate-900 border-slate-800 text-slate-400'
                    }`}
                  >
                    <Grid className="w-4 h-4" />
                  </button>

                  <button
                    onClick={handleExportSVG}
                    className="px-2.5 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 border border-slate-700 text-xs text-slate-200 flex items-center gap-1 transition"
                  >
                    <Download className="w-3.5 h-3.5 text-cyan-400" />
                    <span>SVG</span>
                  </button>
                </>
              )}

              <button
                onClick={handleExportPNG}
                className="px-2.5 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 border border-slate-700 text-xs text-slate-200 flex items-center gap-1 transition"
              >
                <ImageIcon className="w-3.5 h-3.5 text-amber-400" />
                <span>PNG</span>
              </button>
            </div>
          </div>

          {/* Interactive Mathematical Viewport (Complete Multi-Act Film Player OR WebGL1 GPU Canvas OR 2D High-DPI Canvas) */}
          <div
            className={`relative flex-1 min-h-[430px] lg:min-h-[520px] w-full bg-[#050811] ${
              artData?.mode === '3d_surface' ? 'cursor-grab active:cursor-grabbing' : ''
            }`}
            onMouseDown={handleCanvasMouseDown}
            onMouseMove={handleCanvasMouseMove}
            onMouseUp={handleCanvasMouseUp}
            onMouseLeave={handleCanvasMouseUp}
          >
            {isShowingFullCinemaMovie ? (
              <div className="w-full h-full flex flex-col items-center justify-center p-4 bg-[#050811]">
                {cinemaMp4Uri ? (
                  <video
                    src={cinemaMp4Uri}
                    controls
                    autoPlay
                    loop
                    muted
                    playsInline
                    poster={cinemaGifUri || undefined}
                    className="max-h-[470px] w-auto rounded-xl border border-amber-500/40 shadow-2xl"
                  />
                ) : cinemaGifUri ? (
                  <img
                    src={cinemaGifUri}
                    alt={cinemaFilmData.title || 'Complete Multi-Act Mathematical Film'}
                    className="max-h-[470px] w-auto rounded-xl border border-amber-500/40 shadow-2xl"
                  />
                ) : null}
              </div>
            ) : isGpuShaderActive ? (
              <WebGLMathVideoCanvas
                fragmentShader={artData.glsl_fragment_shader}
                isPlaying={isVideoPlaying}
                speed={videoSpeed}
                timeT={timeT}
                onTimeSync={(nextT) => setTimeT(nextT)}
                onShaderError={() => setShaderError(true)}
                onShaderReady={() => setShaderCompiledOk(true)}
                glCanvasRef={glCanvasRef}
              />
            ) : (
              <canvas ref={canvasRef} className="w-full h-full block" />
            )}

            {/* Live Mathematical Coordinate HUD Overlay */}
            <div className="absolute top-3 left-3 bg-slate-950/85 backdrop-blur-md border border-slate-800/90 rounded-xl px-3.5 py-2.5 text-xs font-mono space-y-1 shadow-lg pointer-events-none">
              {isShowingFullCinemaMovie ? (
                <>
                  <div className="flex items-center gap-2 text-amber-300 font-semibold">
                    <span className="w-2 h-2 rounded-full bg-amber-400 animate-pulse" />
                    <span>
                      Multi-Act Mathematical Cinema • {cinemaFilmData?.num_acts || cinemaFilmData?.acts?.length || 5} Acts
                    </span>
                  </div>
                  <div className="text-cyan-300">
                    {cinemaFilmData?.total_frames || 0} Frames ({cinemaFilmData?.total_duration_sec || 0}s @{' '}
                    {cinemaFilmData?.fps || 12} FPS) • C² Homotopy Morphs
                  </div>
                  <div className="text-emerald-300">
                    Click any Act card below to launch its 60 FPS WebGL1 GPU Shader &amp; LaTeX Equations
                  </div>
                </>
              ) : isGpuShaderActive ? (
                <>
                  <div className="flex items-center gap-2 text-slate-200 font-semibold">
                    <span className="w-2 h-2 rounded-full bg-fuchsia-400 animate-pulse" />
                    <span>
                      {artData?.mode === 'raymarch_cinema'
                        ? '3D Raymarched Implicit SDF • t = '
                        : '60 FPS GPU Shader • t = '}
                      <strong className="text-fuchsia-300">{(timeT % (Math.PI * 2)).toFixed(3)} rad</strong>
                    </span>
                  </div>
                  <div className="text-cyan-300">
                    {artData?.mode === 'raymarch_cinema'
                      ? 's_{n+1} = s_n + d(r_o + s_n·r_d, t) • N(p) = ∇d / ‖∇d‖'
                      : 'H_v(x, y, t) → rgb(F(H₀), F(H₁), F(H₂))'}{' '}
                    {shaderCompiledOk ? '• WebGL1 Active' : ''}
                  </div>
                  <div className="text-amber-300">
                    Speed: {videoSpeed.toFixed(2)}x • Phase: {(((timeT % (Math.PI * 2)) / (Math.PI * 2)) * 100).toFixed(0)}% of 2π
                  </div>
                </>
              ) : artData?.mode === 'pixel_field' ||
                artData?.mode === 'pixel_video' ||
                artData?.mode === 'raymarch_cinema' ||
                artData?.mode === 'script_cinema' ? (
                <>
                  <div className="flex items-center gap-2 text-slate-200 font-semibold">
                    <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
                    <span>
                      Scanline Row n = <strong className="text-emerald-300">{currentK}</strong> / {totalSteps}
                    </span>
                  </div>
                  <div className="text-cyan-300">
                    Domain: m = 1…{artData?.nominal_aspect?.[0] || 2000}, n = 1…{artData?.nominal_aspect?.[1] || 1200}
                  </div>
                  <div className="text-amber-300">
                    pixel(m, n) = rgb(F(H₀(x,y)), F(H₁(x,y)), F(H₂(x,y)))
                  </div>
                </>
              ) : artData?.mode === '3d_surface' ? (
                <>
                  <div className="flex items-center gap-2 text-slate-200 font-semibold">
                    <span className="w-2 h-2 rounded-full bg-purple-400 animate-pulse" />
                    <span>
                      3D Manifold Chord k = <strong className="text-purple-300">{currentK}</strong> / {totalSteps}
                    </span>
                  </div>
                  {activePrimitive && (
                    <>
                      <div className="text-sky-300">
                        P₁(k) = ({(activePrimitive[0] ?? 0).toFixed(3)}, {(activePrimitive[1] ?? 0).toFixed(3)},{' '}
                        {(activePrimitive[2] ?? 0).toFixed(3)})
                      </div>
                      <div className="text-amber-300">
                        P₂(k) = ({(activePrimitive[3] ?? 0).toFixed(3)}, {(activePrimitive[4] ?? 0).toFixed(3)},{' '}
                        {(activePrimitive[5] ?? 0).toFixed(3)})
                      </div>
                    </>
                  )}
                  <div className="text-slate-400 text-[11px]">
                    Orbit (θ_x, θ_y) = ({rotX.toFixed(2)} rad, {(rotY % (Math.PI * 2)).toFixed(2)} rad) • Drag to rotate 3D
                  </div>
                </>
              ) : artData?.mode === 'math_video' ? (
                <>
                  <div className="flex items-center gap-2 text-slate-200 font-semibold">
                    <span className="w-2 h-2 rounded-full bg-amber-400 animate-pulse" />
                    <span>
                      4D Time Phase t = <strong className="text-amber-300">{(timeT % (Math.PI * 2)).toFixed(3)} rad</strong>{' '}
                      (k = 1…{totalSteps})
                    </span>
                  </div>
                  {activePrimitive && (
                    <>
                      <div className="text-sky-300">
                        (X₁(k,t), Y₁(k,t)) = ({(activePrimitive[0] ?? 0).toFixed(3)}, {(activePrimitive[1] ?? 0).toFixed(3)})
                      </div>
                      <div className="text-amber-300">
                        (X₂(k,t), Y₂(k,t)) = ({(activePrimitive[2] ?? 0).toFixed(3)}, {(activePrimitive[3] ?? 0).toFixed(3)})
                      </div>
                    </>
                  )}
                </>
              ) : (
                activePrimitive && (
                  <>
                    <div className="flex items-center gap-2 text-slate-300 font-semibold">
                      <span className="w-2 h-2 rounded-full bg-cyan-400 animate-pulse" />
                      <span>
                        Step k = <strong className="text-cyan-300">{currentK}</strong> / {totalSteps}
                      </span>
                    </div>
                    {artData?.mode === 'circles' ? (
                      <>
                        <div className="text-sky-300">
                          Center (X(k), Y(k)) = ({(activePrimitive[0] ?? 0).toFixed(3)}, {(activePrimitive[1] ?? 0).toFixed(3)})
                        </div>
                        <div className="text-amber-300">
                          Radius R(k) = {(activePrimitive[2] ?? 0).toFixed(4)}
                        </div>
                      </>
                    ) : (
                      <>
                        <div className="text-sky-300">
                          (X₁(k), Y₁(k)) = ({(activePrimitive[0] ?? 0).toFixed(3)}, {(activePrimitive[1] ?? 0).toFixed(3)})
                        </div>
                        <div className="text-amber-300">
                          (X₂(k), Y₂(k)) = ({(activePrimitive[2] ?? 0).toFixed(3)}, {(activePrimitive[3] ?? 0).toFixed(3)})
                        </div>
                      </>
                    )}
                  </>
                )
              )}
            </div>

            {/* Exported Looping Math GIF Ready Notification */}
            {exportedGifData && (
              <div className="absolute bottom-3 right-3 bg-slate-950/90 backdrop-blur-md border border-fuchsia-500/40 rounded-xl p-2.5 flex items-center gap-3 shadow-xl">
                <img
                  src={exportedGifData.gifUri}
                  alt="Looping Math GIF Preview"
                  className="w-16 h-10 object-cover rounded border border-slate-700"
                />
                <div className="text-xs">
                  <div className="text-fuchsia-300 font-semibold">Looping Math GIF Ready</div>
                  <div className="flex items-center gap-2 mt-0.5">
                    <a
                      href={exportedGifData.gifUri}
                      download="math-loop.gif"
                      className="text-cyan-300 underline hover:text-cyan-200"
                    >
                      Save GIF
                    </a>
                    {exportedGifData.mp4Uri && (
                      <a
                        href={exportedGifData.mp4Uri}
                        download="math-loop.mp4"
                        className="text-amber-300 underline hover:text-amber-200"
                      >
                        Save MP4
                      </a>
                    )}
                    <button
                      onClick={() => setExportedGifData(null)}
                      className="text-slate-400 hover:text-white ml-1"
                    >
                      ✕
                    </button>
                  </div>
                </div>
              </div>
            )}
          </div>

          {/* INTERACTIVE MULTI-ACT STORYBOARD TIMELINE STRIP */}
          {cinemaFilmData?.acts?.length > 0 && (
            <div className="p-4 border-t border-slate-800 bg-slate-950/90 space-y-2.5">
              <div className="flex flex-wrap items-center justify-between gap-2">
                <span className="text-xs font-bold uppercase tracking-wider text-amber-300 flex items-center gap-1.5">
                  <Clapperboard className="w-3.5 h-3.5" />
                  <span>
                    Interactive Multi-Act Storyboard Timeline ({cinemaFilmData.acts.length} Acts +{' '}
                    {Math.max(0, cinemaFilmData.acts.length - 1)} C² Homotopy Morph Transitions)
                  </span>
                </span>
                <span className="text-[11px] text-emerald-300 font-mono">
                  {cinemaFilmData.verification_summary?.all_acts_verified !== false
                    ? '✓ All Acts Script-Verified & Non-Degenerate'
                    : ''}
                </span>
              </div>

              <div className="flex items-stretch gap-2 overflow-x-auto pb-1">
                {cinemaFilmData.acts.map((act, idx) => {
                  const isActSelected = selectedActIndex === idx;
                  const thumbSrc = act.preview_image_base64
                    ? act.preview_image_base64.startsWith('data:')
                      ? act.preview_image_base64
                      : `data:image/png;base64,${act.preview_image_base64}`
                    : null;
                  return (
                    <React.Fragment key={act.act_number || idx}>
                      <div
                        onClick={() => handleSelectStoryAct(act, idx, true)}
                        className={`min-w-[230px] max-w-[250px] rounded-xl p-2.5 border cursor-pointer transition flex flex-col justify-between shrink-0 ${
                          isActSelected
                            ? 'bg-amber-500/15 border-amber-400 shadow-lg shadow-amber-500/10'
                            : 'bg-slate-900/80 border-slate-800 hover:border-slate-700'
                        }`}
                      >
                        <div className="space-y-1.5">
                          <div className="flex items-center justify-between gap-1">
                            <span className="text-[11px] font-bold text-amber-300">
                              ACT {act.act_number || idx + 1}
                            </span>
                            <span className="text-[10px] px-1.5 py-0.5 rounded bg-slate-800 text-cyan-300 font-mono truncate">
                              {act.paradigm_badge || act.paradigm_choice || '3D Raymarch'}
                            </span>
                          </div>
                          {thumbSrc && (
                            <img
                              src={thumbSrc}
                              alt={act.title}
                              className="w-full h-24 object-cover rounded-lg border border-slate-700/80"
                            />
                          )}
                          <div className="text-xs font-semibold text-slate-100 truncate">
                            {act.title}
                          </div>
                          <p className="text-[11px] text-slate-400 line-clamp-2 leading-snug">
                            {act.narration}
                          </p>
                        </div>
                        <div className="pt-1.5 mt-1.5 border-t border-slate-800/80 flex items-center justify-between text-[10px] font-mono text-emerald-300">
                          <span>✓ Verified</span>
                          <span>{act.duration_sec ? `${act.duration_sec}s` : 'Explore 60 FPS GPU →'}</span>
                        </div>
                      </div>

                      {idx < cinemaFilmData.acts.length - 1 && (
                        <div className="flex flex-col items-center justify-center px-1.5 shrink-0">
                          <div className="px-2 py-1 rounded-lg bg-fuchsia-500/15 border border-fuchsia-500/40 text-[10px] font-mono text-fuchsia-300 text-center whitespace-nowrap">
                            <div>C² Homotopy Morph</div>
                            <ArrowRight className="w-3.5 h-3.5 mx-auto my-0.5 text-fuchsia-400" />
                            <div>σ(τ)=6τ⁵-15τ⁴+10τ³</div>
                          </div>
                        </div>
                      )}
                    </React.Fragment>
                  );
                })}
              </div>
            </div>
          )}

          {/* 4D Time-Evolving Video Transport Bar */}
          {isTimeBasedMode && !isShowingFullCinemaMovie && (
            <div className="px-4 py-3 border-t border-slate-800/80 bg-fuchsia-950/20 flex flex-wrap items-center gap-3">
              <button
                onClick={() => setIsVideoPlaying((p) => !p)}
                className="px-3.5 py-1.5 rounded-xl bg-amber-500 hover:bg-amber-400 text-slate-950 font-semibold text-xs flex items-center gap-1.5 shadow-md shadow-amber-500/20 transition"
              >
                {isVideoPlaying ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
                <span>{isVideoPlaying ? 'Pause' : 'Play'}</span>
              </button>

              <div className="flex-1 flex items-center gap-3 min-w-[180px]">
                <span className="text-xs font-mono text-amber-300 shrink-0">
                  t = {(timeT % (Math.PI * 2)).toFixed(2)} rad
                </span>
                <input
                  type="range"
                  min={0}
                  max={6.28318}
                  step={0.01}
                  value={timeT % (Math.PI * 2)}
                  onChange={(e) => {
                    setIsVideoPlaying(false);
                    setTimeT(Number(e.target.value));
                  }}
                  className="w-full accent-amber-400 cursor-pointer"
                />
                <span className="text-xs font-mono text-slate-400 shrink-0">2π rad</span>
              </div>

              <div className="flex items-center gap-1.5">
                <span className="text-[11px] text-slate-400">Speed:</span>
                {[0.25, 0.5, 1.0, 2.0, 3.0].map((spd) => (
                  <button
                    key={spd}
                    onClick={() => setVideoSpeed(spd)}
                    className={`px-2 py-1 rounded-md text-[11px] font-medium border transition ${
                      videoSpeed === spd
                        ? 'bg-amber-500/20 border-amber-500/50 text-amber-200'
                        : 'bg-slate-900 border-slate-800 text-slate-400 hover:text-slate-200'
                    }`}
                  >
                    {spd}x
                  </button>
                ))}
              </div>
            </div>
          )}

          {/* Step-by-Step Drawing Animation Transport Bar (k = 1 -> N) */}
          {!isGpuShaderActive && !isShowingFullCinemaMovie && (
            <div className="px-4 py-3.5 border-t border-slate-800/80 bg-slate-900/60 flex flex-wrap items-center gap-3">
              <button
                onClick={() => {
                  if (currentK >= totalSteps) {
                    setCurrentK(1);
                    setIsPlaying(true);
                  } else {
                    setIsPlaying((p) => !p);
                  }
                }}
                className="px-3.5 py-2 rounded-xl bg-cyan-500 hover:bg-cyan-400 text-slate-950 font-semibold text-xs flex items-center gap-1.5 shadow-md shadow-cyan-500/20 transition"
              >
                {isPlaying ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
                <span>
                  {isPlaying
                    ? 'Pause'
                    : currentK >= totalSteps
                    ? `Replay k=1→${totalSteps}`
                    : 'Play'}
                </span>
              </button>

              <button
                onClick={() => {
                  setCurrentK(1);
                  setIsPlaying(true);
                }}
                title="Restart from k = 1"
                className="p-2 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-300 border border-slate-700 transition"
              >
                <RotateCcw className="w-4 h-4" />
              </button>

              <div className="flex-1 flex items-center gap-3 min-w-[180px]">
                <span className="text-xs font-mono text-slate-400 shrink-0">k=1</span>
                <input
                  type="range"
                  min={1}
                  max={totalSteps}
                  value={Math.min(currentK, totalSteps)}
                  onChange={(e) => {
                    setIsPlaying(false);
                    setCurrentK(Number(e.target.value));
                  }}
                  className="w-full accent-cyan-400 cursor-pointer"
                />
                <span className="text-xs font-mono text-cyan-300 shrink-0">
                  k={Math.min(currentK, totalSteps)}/{totalSteps}
                </span>
              </div>

              <div className="flex items-center gap-1.5">
                <span className="text-[11px] text-slate-400">Speed:</span>
                {[
                  { label: 'Slow', val: 3 },
                  { label: '1x', val: 14 },
                  { label: '3x', val: 40 },
                  { label: 'Max', val: 5000 },
                ].map((s) => (
                  <button
                    key={s.label}
                    onClick={() => {
                      if (s.val >= 5000 && artData) {
                        setIsPlaying(false);
                        setCurrentK(totalSteps);
                      } else {
                        setAnimSpeed(s.val);
                      }
                    }}
                    className={`px-2 py-1 rounded-md text-[11px] font-medium border transition ${
                      animSpeed === s.val
                        ? 'bg-cyan-500/20 border-cyan-500/50 text-cyan-200'
                        : 'bg-slate-900 border-slate-800 text-slate-400 hover:text-slate-200'
                    }`}
                  >
                    {s.label}
                  </button>
                ))}
              </div>
            </div>
          )}

          {/* HAMID POSTER VIEW: Crisp White Publication-Style Mathematical Specification Plate */}
          {isPixelBasedMode && showHamidPoster && (
            <div className="p-5 bg-slate-950 border-t border-slate-800">
              <div className="bg-white text-slate-900 rounded-2xl p-6 shadow-2xl border border-slate-200 space-y-4">
                {artData.image_base64 && (
                  <div className="flex flex-col items-center">
                    <img
                      src={
                        artData.image_base64.startsWith('data:')
                          ? artData.image_base64
                          : `data:image/png;base64,${artData.image_base64}`
                      }
                      alt={artData.title}
                      className="max-h-[340px] w-auto rounded-lg shadow-md border border-slate-300"
                    />
                  </div>
                )}

                <div className="pt-2 border-t border-slate-200 text-sm leading-relaxed font-serif text-slate-800">
                  <p>
                    The above picture is a converted version of a 2000 × 1200 image. For m = 1, 2, ..., 2000 and n = 1, 2, ..., 1200, the color of the pixel of the row n and the column m is{' '}
                    <code className="font-mono text-slate-900 font-semibold">
                      rgb(F(H₀(...)), F(H₁(...)), F(H₂(...)))
                    </code>
                    {artData.nominal_aspect &&
                      (artData.nominal_aspect[0] !== 2000 || artData.nominal_aspect[1] !== 1200) && (
                        <span className="text-slate-600">
                          {' '}
                          (evaluated on canonical {artData.nominal_aspect[0]} × {artData.nominal_aspect[1]} lattice)
                        </span>
                      )}
                    :
                  </p>
                </div>

                <div className="space-y-2.5 pt-1">
                  {activeLatex &&
                    Object.entries(activeLatex).map(([eqKey, texStr]) => (
                      <div
                        key={eqKey}
                        className="bg-slate-50 border border-slate-200/90 rounded-xl px-3.5 py-2"
                      >
                        <div className="text-[11px] font-bold uppercase tracking-wider text-sky-800 font-mono">
                          {eqKey}
                        </div>
                        <KatexFormula tex={texStr} lightMode={true} />
                      </div>
                    ))}
                </div>

                <div className="flex items-center justify-between pt-2 border-t border-slate-200 text-xs text-slate-500">
                  <span>
                    Mathematical Specification Plate — Double-Exponential &amp; 3D Implicit SDF Synthesis
                  </span>
                  <button
                    onClick={handleDownloadMathPoster}
                    className="px-3 py-1.5 rounded-lg bg-slate-900 hover:bg-slate-800 text-white font-semibold text-xs flex items-center gap-1.5 transition"
                  >
                    <Download className="w-3.5 h-3.5 text-emerald-400" />
                    <span>Download Hamid Math Poster (PNG)</span>
                  </button>
                </div>
              </div>
            </div>
          )}
        </section>

        {/* RIGHT 5 COLS: 9-Mode Synthesis Parameters + Live Equations + Python Code + Formula Lab */}
        <section className="lg:col-span-5 flex flex-col gap-4">
          {/* Card 1: 9-Mode Mathematical Synthesis Parameters */}
          <div className="bg-[#080d1a] border border-slate-800/80 rounded-2xl p-4 shadow-xl space-y-4">
            <div className="flex items-center justify-between">
              <h3 className="text-xs font-bold uppercase tracking-wider text-cyan-400 flex items-center gap-1.5">
                <Sliders className="w-3.5 h-3.5" />
                <span>Mathematical Synthesis Parameters</span>
              </h3>
              <span className="text-[11px] text-slate-400 font-mono">
                Harmonics: {artData?.harmonics_count ?? numHarmonics}
              </span>
            </div>

            {/* 9-Mode Mathematical Family Pill Selector */}
            <div>
              <label className="text-[11px] text-slate-400 block mb-1.5">
                Mathematical Representation Mode (9 Synthesis Engines)
              </label>
              <div className="grid grid-cols-2 sm:grid-cols-3 gap-1.5 bg-slate-900/90 p-1.5 rounded-xl border border-slate-800">
                {MODE_DEFINITIONS.map((m) => {
                  const isActive = (artData?.mode || mode) === m.id;
                  return (
                    <button
                      key={m.id}
                      type="button"
                      onClick={() => {
                        setMode(m.id);
                        if (m.id === 'script_cinema') {
                          setShowScriptDrawer(true);
                        } else {
                          handleGenerateFromPrompt(prompt, { mode: m.id });
                        }
                      }}
                      className={`px-2.5 py-2 rounded-lg text-xs font-medium transition text-left flex items-center gap-1.5 ${
                        isActive
                          ? m.id === 'script_cinema'
                            ? 'bg-amber-500/20 text-amber-200 border border-amber-500/40 shadow-sm'
                            : m.id === 'raymarch_cinema'
                            ? 'bg-rose-500/20 text-rose-200 border border-rose-500/40 shadow-sm'
                            : m.id === 'pixel_video'
                            ? 'bg-fuchsia-500/20 text-fuchsia-200 border border-fuchsia-500/40 shadow-sm'
                            : 'bg-cyan-500/20 text-cyan-200 border border-cyan-500/40 shadow-sm'
                          : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/50'
                      }`}
                    >
                      {m.id === 'script_cinema' ? (
                        <Clapperboard className="w-3.5 h-3.5 text-amber-400 shrink-0" />
                      ) : m.id === 'raymarch_cinema' ? (
                        <Clapperboard className="w-3.5 h-3.5 text-rose-400 shrink-0" />
                      ) : m.id === 'pixel_field' ? (
                        <Palette className="w-3.5 h-3.5 text-emerald-400 shrink-0" />
                      ) : m.id === '3d_surface' ? (
                        <Box className="w-3.5 h-3.5 text-purple-400 shrink-0" />
                      ) : m.id === 'math_video' ? (
                        <Film className="w-3.5 h-3.5 text-amber-400 shrink-0" />
                      ) : m.id === 'pixel_video' ? (
                        <Cpu className="w-3.5 h-3.5 text-fuchsia-400 shrink-0" />
                      ) : (
                        <Compass className="w-3.5 h-3.5 text-cyan-400 shrink-0" />
                      )}
                      <span className="truncate">{m.shortLabel}</span>
                    </button>
                  );
                })}
              </div>
            </div>

            {/* Hamid Chord Weave Selector & Active Domain Readout */}
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
              <div>
                <label className="text-[11px] text-slate-400 block mb-1">Hamid Chord Weave</label>
                <select
                  value={weaveStyle}
                  onChange={(e) => {
                    const nextWeave = e.target.value;
                    setWeaveStyle(nextWeave);
                    handleGenerateFromPrompt(prompt, { weave_style: nextWeave });
                  }}
                  className="w-full bg-slate-900 border border-slate-800 rounded-xl px-3 py-2 text-xs text-slate-200 focus:outline-none focus:border-cyan-400"
                >
                  <option value="harmonic_web">Harmonic Web (Classic Hamid)</option>
                  <option value="phase_chord">Phase Caustic Envelope</option>
                  <option value="radial_spine">Radial Medial Spine</option>
                  <option value="hamid_exact">Exact Published Formula</option>
                </select>
              </div>

              <div className="bg-slate-900/60 border border-slate-800/80 rounded-xl px-3 py-2 flex flex-col justify-center">
                <span className="text-[11px] text-slate-400">Active Domain</span>
                <span className="text-xs font-mono text-cyan-300 truncate">
                  {artData?.mode === 'script_cinema'
                    ? 'Multi-Act Homotopy d_τ(p,t) = (1-σ(τ))d_m + σ(τ)d_{m+1}'
                    : artData?.mode === 'raymarch_cinema'
                    ? 'ℝ³ × S¹ 3D Implicit SDF Raymarching d(p, t) = 0'
                    : artData?.mode === 'pixel_video'
                    ? 'ℝ² × S¹ Full-Color GPU Field H_v(x, y, t)'
                    : artData?.mode === 'pixel_field'
                    ? `Lattice ${artData?.nominal_aspect?.[0] || 2000} × ${artData?.nominal_aspect?.[1] || 1200}`
                    : artData?.mode === '3d_surface'
                    ? `ℝ³ Manifold (k = 1…${totalSteps})`
                    : artData?.mode === 'math_video'
                    ? 'ℝ² × S¹ Time-Harmonic (t ∈ [0, 2π])'
                    : `k = 1, 2, …, ${totalSteps}`}
                </span>
              </div>
            </div>

            {/* Sliders for Harmonics (M) and Total Segments (N) */}
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 pt-1">
              <div className="bg-slate-900/60 border border-slate-800/80 rounded-xl p-3">
                <div className="flex justify-between text-xs mb-1.5">
                  <span className="text-slate-300 font-medium">Fourier Harmonics (M)</span>
                  <span className="font-mono text-cyan-300">{numHarmonics} terms</span>
                </div>
                <input
                  type="range"
                  min={2}
                  max={110}
                  value={numHarmonics}
                  onChange={(e) => setNumHarmonics(Number(e.target.value))}
                  onMouseUp={() => {
                    handleGenerateFromPrompt(prompt, { harmonics: numHarmonics, keepProgress: true });
                  }}
                  onTouchEnd={() => {
                    handleGenerateFromPrompt(prompt, { harmonics: numHarmonics, keepProgress: true });
                  }}
                  className="w-full accent-cyan-400 cursor-pointer"
                />
                <p className="text-[11px] text-slate-500 mt-1">
                  Controls trigonometric series expansion order M
                </p>
              </div>

              <div className="bg-slate-900/60 border border-slate-800/80 rounded-xl p-3">
                <div className="flex justify-between text-xs mb-1.5">
                  <span className="text-slate-300 font-medium">Total Steps (N)</span>
                  <span className="font-mono text-amber-300">N = {numSegments}</span>
                </div>
                <input
                  type="range"
                  min={400}
                  max={4000}
                  step={100}
                  value={numSegments}
                  onChange={(e) => setNumSegments(Number(e.target.value))}
                  onMouseUp={() => {
                    if (activeSource.type === 'preset') {
                      handleSelectPreset(activeSource.id, false);
                    } else {
                      handleGenerateFromPrompt(prompt, { N: numSegments, keepProgress: true });
                    }
                  }}
                  className="w-full accent-amber-400 cursor-pointer"
                />
                <p className="text-[11px] text-slate-500 mt-1">
                  k = 1, 2, …, {numSegments} geometric primitives
                </p>
              </div>
            </div>
          </div>

          {/* Card 2: Equations / Python Script / Custom Formula Studio */}
          <div className="flex-1 bg-[#080d1a] border border-slate-800/80 rounded-2xl flex flex-col overflow-hidden shadow-xl">
            {/* Tabs */}
            <div className="flex border-b border-slate-800/80 bg-slate-900/50 px-2 pt-2 gap-1">
              <button
                onClick={() => setRightTab('equations')}
                className={`px-3.5 py-2 rounded-t-xl text-xs font-semibold flex items-center gap-1.5 transition ${
                  rightTab === 'equations'
                    ? 'bg-[#080d1a] text-cyan-300 border-t border-x border-slate-800'
                    : 'text-slate-400 hover:text-slate-200'
                }`}
              >
                <Layers className="w-3.5 h-3.5" />
                <span>Exact Math Equations</span>
              </button>
              <button
                onClick={() => setRightTab('python')}
                className={`px-3.5 py-2 rounded-t-xl text-xs font-semibold flex items-center gap-1.5 transition ${
                  rightTab === 'python'
                    ? 'bg-[#080d1a] text-amber-300 border-t border-x border-slate-800'
                    : 'text-slate-400 hover:text-slate-200'
                }`}
              >
                <Terminal className="w-3.5 h-3.5" />
                <span>Python / Matplotlib Code</span>
              </button>
              <button
                onClick={() => setRightTab('custom')}
                className={`px-3.5 py-2 rounded-t-xl text-xs font-semibold flex items-center gap-1.5 transition ${
                  rightTab === 'custom'
                    ? 'bg-[#080d1a] text-emerald-300 border-t border-x border-slate-800'
                    : 'text-slate-400 hover:text-slate-200'
                }`}
              >
                <Code2 className="w-3.5 h-3.5" />
                <span>Formula Lab</span>
              </button>
            </div>

            {/* Tab 1: Live KaTeX Trigonometric, Differential-Geometric & 3D Raymarching Equations */}
            {rightTab === 'equations' && (
              <div className="p-4 flex-1 flex flex-col justify-between space-y-3 overflow-y-auto max-h-[620px]">
                <div className="space-y-3">
                  <div className="flex items-center justify-between">
                    <span className="text-xs text-slate-400">
                      {artData?.mode === 'raymarch_cinema' || artData?.mode === 'script_cinema' ? (
                        <>
                          For 3D raymarch field <code className="text-rose-300">p = (x, y, z) ∈ ℝ³</code> and time{' '}
                          <code className="text-rose-300">t ∈ [0, 2π]</code>:
                        </>
                      ) : artData?.mode === 'pixel_video' ? (
                        <>
                          For pixel lattice <code className="text-fuchsia-300">(x, y)</code> and continuous time{' '}
                          <code className="text-fuchsia-300">t ∈ [0, 2π]</code>:
                        </>
                      ) : artData?.mode === 'pixel_field' ? (
                        <>
                          For pixel lattice{' '}
                          <code className="text-emerald-300">
                            m = 1…{artData?.nominal_aspect?.[0] || 2000}, n = 1…{artData?.nominal_aspect?.[1] || 1200}
                          </code>
                          :
                        </>
                      ) : artData?.mode === 'math_video' ? (
                        <>
                          For <code className="text-amber-300">k = 1, …, {totalSteps}</code> and continuous time{' '}
                          <code className="text-amber-300">t ∈ [0, 2π]</code>:
                        </>
                      ) : (
                        <>
                          For each integer <code className="text-cyan-300">k = 1, 2, …, {totalSteps}</code>:
                        </>
                      )}
                    </span>
                    {activeLatex && (
                      <button
                        onClick={() =>
                          handleCopy('latex', Object.values(activeLatex).join('\n\n'))
                        }
                        className="px-2.5 py-1 rounded-lg bg-slate-800 hover:bg-slate-700 text-xs text-slate-300 flex items-center gap-1 transition"
                      >
                        {copiedKey === 'latex' ? (
                          <>
                            <Check className="w-3.5 h-3.5 text-emerald-400" />
                            <span>Copied LaTeX</span>
                          </>
                        ) : (
                          <>
                            <Copy className="w-3.5 h-3.5 text-cyan-400" />
                            <span>Copy LaTeX</span>
                          </>
                        )}
                      </button>
                    )}
                  </div>

                  {activeLatex &&
                    Object.entries(activeLatex).map(([varKey, texStr]) => (
                      <div
                        key={varKey}
                        className="bg-slate-900/80 border border-slate-800/90 rounded-xl p-3 hover:border-cyan-500/30 transition"
                      >
                        <div className="flex items-center justify-between mb-1">
                          <span className="text-[11px] font-semibold uppercase tracking-wider text-cyan-400">
                            {varKey === 'X1' || varKey === 'Y1' || varKey === 'Z1'
                              ? `First Endpoint ${varKey}`
                              : varKey === 'X2' || varKey === 'Y2' || varKey === 'Z2'
                              ? `Second Endpoint ${varKey}`
                              : varKey === 'R'
                              ? 'Radius Function R(k)'
                              : varKey === 'RGB'
                              ? 'Pixel Color Quantizer rgb(F(H₀), F(H₁), F(H₂))'
                              : `Mathematical Function ${varKey}`}
                          </span>
                        </div>
                        <KatexFormula tex={texStr} />
                      </div>
                    ))}
                </div>

                <div className="bg-cyan-500/10 border border-cyan-500/20 rounded-xl p-3 text-xs text-cyan-200/90 leading-relaxed">
                  <strong>Pure Mathematical Construction:</strong> Every point, chord, surface normal,
                  3D raymarched distance field, homotopy morph transition, and pixel intensity on the
                  canvas is evaluated directly from closed-form trigonometric, differential-geometric,
                  and double-exponential equations.
                </div>
              </div>
            )}

            {/* Tab 2: Standalone Reproducible Python / NumPy / Matplotlib Code */}
            {rightTab === 'python' && (
              <div className="p-4 flex-1 flex flex-col space-y-3">
                <div className="flex items-center justify-between">
                  <span className="text-xs text-slate-400">
                    100% standalone Python script containing the exact analytical equations:
                  </span>
                  <button
                    onClick={() => handleCopy('python', artData?.python_code || '')}
                    className="px-3 py-1 rounded-lg bg-amber-500/20 hover:bg-amber-500/30 border border-amber-500/40 text-amber-200 text-xs flex items-center gap-1.5 transition"
                  >
                    {copiedKey === 'python' ? (
                      <>
                        <Check className="w-3.5 h-3.5 text-emerald-400" />
                        <span>Copied!</span>
                      </>
                    ) : (
                      <>
                        <Copy className="w-3.5 h-3.5" />
                        <span>Copy Python Script</span>
                      </>
                    )}
                  </button>
                </div>
                <pre className="flex-1 bg-slate-950 border border-slate-800 rounded-xl p-3.5 text-xs font-mono text-slate-200 overflow-auto max-h-[420px] leading-relaxed">
                  {artData?.python_code}
                </pre>
              </div>
            )}

            {/* Tab 3: Interactive Custom Formula Lab */}
            {rightTab === 'custom' && (
              <div className="p-4 flex-1 flex flex-col space-y-3 overflow-y-auto">
                <p className="text-xs text-slate-400">
                  Write or modify closed-form trigonometric equations directly (using{' '}
                  <code className="text-cyan-300">np.sin</code>,{' '}
                  <code className="text-cyan-300">np.cos</code>,{' '}
                  <code className="text-cyan-300">np.pi</code>,{' '}
                  <code className="text-cyan-300">k</code>, <code className="text-cyan-300">N</code>):
                </p>

                {mode !== 'circles' ? (
                  <div className="space-y-2.5">
                    {['X1', 'Y1', 'X2', 'Y2'].map((key) => (
                      <div key={key}>
                        <label className="text-xs font-mono text-cyan-300 block mb-1">
                          {key}(k) =
                        </label>
                        <input
                          type="text"
                          value={customFormulas[key] || ''}
                          onChange={(e) =>
                            setCustomFormulas((prev) => ({ ...prev, [key]: e.target.value }))
                          }
                          className="w-full px-3 py-2 bg-slate-900 border border-slate-700 rounded-xl text-xs font-mono text-slate-100 focus:outline-none focus:border-cyan-400"
                        />
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className="space-y-2.5">
                    {['X', 'Y', 'R'].map((key) => (
                      <div key={key}>
                        <label className="text-xs font-mono text-cyan-300 block mb-1">
                          {key}(k) =
                        </label>
                        <input
                          type="text"
                          value={customFormulas[key] || ''}
                          onChange={(e) =>
                            setCustomFormulas((prev) => ({ ...prev, [key]: e.target.value }))
                          }
                          className="w-full px-3 py-2 bg-slate-900 border border-slate-700 rounded-xl text-xs font-mono text-slate-100 focus:outline-none focus:border-cyan-400"
                        />
                      </div>
                    ))}
                  </div>
                )}

                <button
                  onClick={handleRunCustomFormulas}
                  className="w-full py-2.5 rounded-xl bg-emerald-500 hover:bg-emerald-400 text-slate-950 font-semibold text-xs flex items-center justify-center gap-2 shadow-lg shadow-emerald-500/20 transition"
                >
                  <Play className="w-4 h-4" />
                  <span>Evaluate &amp; Draw Custom Equations</span>
                </button>
              </div>
            )}
          </div>
        </section>
      </main>
    </div>
  );
}
