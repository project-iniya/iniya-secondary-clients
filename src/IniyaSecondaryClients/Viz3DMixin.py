"""
Viz3DMixin.py — Text-to-3D visualization for Iniya
Plug into IniyaClient via inheritance.

Engines (auto-selected or manually overridden):
  blenderllm  →  LLM generates bpy script → Blender executes → .glb
  shap-e      →  Diffusion model → mesh → .glb  (fallback)

Viewer:
  Flask + Three.js served on localhost:5000
  Supports orbit controls, auto-centering, prompt history

    client.generate_3d("a wooden chair with four legs")
    # opens http://localhost:5000 automatically
"""

import json
import subprocess
import os
import sys
import tempfile
import threading
import time
import webbrowser
from pathlib import Path
from typing import Literal, Optional
from .utils import _check_cuda


# ─────────────────────────────────────────────────────────────────────────────
#  Paths
# ─────────────────────────────────────────────────────────────────────────────

THIS_DIR    = Path(__file__).resolve().parent
STATIC_DIR  = THIS_DIR / "viz_static"
MODELS_DIR  = THIS_DIR / "models" / "viz"
GLB_DIR     = STATIC_DIR / "models"

STATIC_DIR.mkdir(parents=True, exist_ok=True)
GLB_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# redirect HF cache before transformers is imported
os.environ["HF_HOME"] = str(MODELS_DIR)
os.environ["HUGGINGFACE_HUB_CACHE"] = str(MODELS_DIR / "hub")
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
os.environ["TRANSFORMERS_OFFLINE"] = "0"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

viewer_html = """
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8"/>
<title>Iniya 3D Viewer</title>
<style>
  *{margin:0;padding:0;box-sizing:border-box}
  body{background:#1a1a2e;color:#e0e0ff;font-family:system-ui,sans-serif;display:flex;flex-direction:column;height:100vh}
  #top{padding:10px 14px;display:flex;gap:8px;align-items:center;background:#16213e;border-bottom:1px solid #0f3460}
  #prompt{flex:1;padding:8px 12px;background:#0f3460;color:#e0e0ff;border:1px solid #533483;border-radius:6px;font-size:14px;outline:none}
  #prompt:focus{border-color:#e94560}
  #btn{padding:8px 18px;background:#e94560;border:none;color:#fff;border-radius:6px;cursor:pointer;font-size:14px}
  #btn:disabled{background:#533483;cursor:not-allowed}
  #btn:hover:not(:disabled){background:#ff6b81}
  #engine-badge{padding:4px 10px;border-radius:12px;font-size:11px;background:#0f3460;color:#74b9ff;border:1px solid #533483}
  #status{font-size:12px;color:#888;min-width:130px}
  #canvas-wrap{flex:1;position:relative}
  canvas{display:block}
  #controls-panel{position:absolute;top:10px;right:10px;display:flex;flex-direction:column;gap:6px}
  .ctrl-btn{padding:5px 12px;border-radius:6px;border:1px solid #0f3460;background:#16213ecc;color:#aaa;font-size:11px;cursor:pointer;transition:all .15s}
  .ctrl-btn.active{background:#e94560;border-color:#e94560;color:#fff}
  .ctrl-btn:hover:not(.active){background:#0f3460;color:#fff}
  #history{position:absolute;bottom:10px;left:10px;background:#16213ecc;border:1px solid #0f3460;border-radius:8px;padding:8px;max-width:200px;max-height:180px;overflow-y:auto}
  #history h4{font-size:10px;color:#555;margin-bottom:5px;text-transform:uppercase;letter-spacing:.05em}
  .hist-item{font-size:12px;color:#888;padding:3px 6px;cursor:pointer;border-radius:4px;margin-bottom:2px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
  .hist-item:hover{background:#0f3460;color:#e0e0ff}
  #info{position:absolute;bottom:10px;right:10px;font-size:10px;color:#444}
</style>
</head>
<body>
<div id="top">
  <input id="prompt" type="text" placeholder="describe a 3D object..."/>
  <button id="btn">Generate</button>
  <span id="engine-badge">loading...</span>
  <span id="status"></span>
</div>
<div id="canvas-wrap">
  <div id="controls-panel">
    <div class="ctrl-btn active" data-mode="solid">◈ Solid</div>
    <div class="ctrl-btn"        data-mode="wire">⬡ Wire</div>
    <div class="ctrl-btn"        data-mode="both">◉ Both</div>
    <div class="ctrl-btn"        id="rotate-btn">↻ Rotate</div>
  </div>
  <div id="history"><h4>History</h4><div id="hist-list"></div></div>
  <div id="info">scroll · drag · right-drag to pan</div>
</div>
 
<script type="importmap">
{"imports":{
  "three":"https://cdn.jsdelivr.net/npm/three@0.160/build/three.module.js",
  "three/addons/":"https://cdn.jsdelivr.net/npm/three@0.160/examples/jsm/"
}}
</script>
<script type="module">
import * as THREE from 'three';
import {GLTFLoader}      from 'three/addons/loaders/GLTFLoader.js';
import {OrbitControls}   from 'three/addons/controls/OrbitControls.js';
import {RoomEnvironment} from 'three/addons/environments/RoomEnvironment.js';
 
const wrap     = document.getElementById('canvas-wrap');
const renderer = new THREE.WebGLRenderer({antialias:true});
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.shadowMap.enabled = true;
renderer.shadowMap.type    = THREE.PCFSoftShadowMap;
renderer.toneMapping       = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.0;
wrap.appendChild(renderer.domElement);
 
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x1a1a2e);
scene.fog = new THREE.Fog(0x1a1a2e, 20, 60);
 
// environment map — gives PBR materials realistic reflections for free
const pmrem  = new THREE.PMREMGenerator(renderer);
scene.environment = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;
 
const camera = new THREE.PerspectiveCamera(50, 1, 0.01, 1000);
camera.position.set(5, 4, 5);
 
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping   = true;
controls.dampingFactor   = 0.06;
controls.autoRotate      = false;
controls.autoRotateSpeed = 1.2;
controls.minDistance     = 0.5;
controls.maxDistance     = 50;
 
// 3-point lighting
const key = new THREE.DirectionalLight(0xfff5e0, 2.5);
key.position.set(6, 10, 6);
key.castShadow = true;
key.shadow.mapSize.set(2048, 2048);
key.shadow.camera.left = key.shadow.camera.bottom = -8;
key.shadow.camera.right = key.shadow.camera.top   =  8;
key.shadow.bias = -0.001;
scene.add(key);
const fill = new THREE.DirectionalLight(0xc0d8ff, 0.8);
fill.position.set(-6, 4, -6);
scene.add(fill);
const rim = new THREE.DirectionalLight(0xffe0a0, 0.6);
rim.position.set(0, 6, -8);
scene.add(rim);
scene.add(new THREE.AmbientLight(0x404060, 0.4));
 
// ground
const ground = new THREE.Mesh(
  new THREE.PlaneGeometry(30, 30),
  new THREE.MeshStandardMaterial({color:0x111128, roughness:0.9, metalness:0.0})
);
ground.rotation.x = -Math.PI / 2;
ground.receiveShadow = true;
scene.add(ground);
scene.add(new THREE.GridHelper(30, 60, 0x0f3460, 0x0a1a30));
 
// model state
let currentGroup = null;
let wireOverlay  = null;
let viewMode     = 'solid';
 
function clearScene(){
  if(currentGroup){ scene.remove(currentGroup); currentGroup = null; }
  if(wireOverlay) { scene.remove(wireOverlay);  wireOverlay  = null; }
}
 
function applyMode(){
  if(!currentGroup) return;
  currentGroup.traverse(n => {
    if(!n.isMesh) return;
    n.material.wireframe = (viewMode === 'wire');
    n.visible = true;
  });
  if(wireOverlay) wireOverlay.visible = (viewMode === 'both');
}
 
const loader = new GLTFLoader();
 
function loadModel(url){
  document.getElementById('status').textContent = 'Loading...';
  clearScene();
  loader.load(url, gltf => {
    currentGroup = gltf.scene;
    currentGroup.traverse(n => {
      if(!n.isMesh) return;
      n.castShadow    = true;
      n.receiveShadow = true;
      // keep Blender-exported PBR materials, just boost env map
      if(n.material && n.material.isMeshStandardMaterial){
        n.material.envMapIntensity = 1.2;
        n.material.needsUpdate     = true;
      } else {
        n.material = new THREE.MeshStandardMaterial({
          color:0xdddddd, roughness:0.4, metalness:0.1, envMapIntensity:1.2
        });
      }
    });
 
    // wire overlay for "both" mode
    const wg = new THREE.Group();
    currentGroup.traverse(n => {
      if(!n.isMesh) return;
      const wm = new THREE.Mesh(
        n.geometry,
        new THREE.MeshBasicMaterial({color:0x6688ff, wireframe:true, transparent:true, opacity:0.12})
      );
      wm.applyMatrix4(n.matrixWorld);
      wg.add(wm);
    });
    wireOverlay = wg;
 
    // centre + fit camera
    const box    = new THREE.Box3().setFromObject(currentGroup);
    const size   = box.getSize(new THREE.Vector3()).length();
    const center = box.getCenter(new THREE.Vector3());
    currentGroup.position.sub(center);
    currentGroup.position.y += box.getSize(new THREE.Vector3()).y / 2;
    wireOverlay.position.copy(currentGroup.position);
 
    scene.add(currentGroup);
    scene.add(wireOverlay);
    applyMode();
 
    camera.position.setLength(size * 2.0);
    controls.target.set(0, size * 0.25, 0);
    controls.update();
    controls.autoRotate = true;
    setTimeout(() => controls.autoRotate = false, 5000);
    document.getElementById('status').textContent = '✓ ready';
  }, undefined, e => {
    document.getElementById('status').textContent = '✗ load failed';
    console.error(e);
  });
}
 
// mode buttons
document.querySelectorAll('.ctrl-btn[data-mode]').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.ctrl-btn[data-mode]').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    viewMode = btn.dataset.mode;
    applyMode();
  });
});
 
const rotBtn = document.getElementById('rotate-btn');
rotBtn.addEventListener('click', () => {
  controls.autoRotate = !controls.autoRotate;
  rotBtn.classList.toggle('active', controls.autoRotate);
});
 
// generate
const histList = document.getElementById('hist-list');
const history  = [];
 
function addHistory(prompt, url){
  history.unshift({prompt, url});
  histList.innerHTML = '';
  history.slice(0, 8).forEach(h => {
    const d = document.createElement('div');
    d.className = 'hist-item'; d.title = h.prompt; d.textContent = h.prompt;
    d.onclick = () => loadModel(h.url + '?t=' + Date.now());
    histList.appendChild(d);
  });
}
 
async function generate(){
  const prompt = document.getElementById('prompt').value.trim();
  if(!prompt) return;
  const btn = document.getElementById('btn');
  btn.disabled = true;
  document.getElementById('status').textContent = 'Generating...';
  try {
    const res  = await fetch('/generate', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({prompt})});
    const data = await res.json();
    if(data.error){ document.getElementById('status').textContent = '✗ ' + data.error; return; }
    loadModel(data.model_url + '?t=' + Date.now());
    addHistory(prompt, data.model_url);
  } catch(e){
    document.getElementById('status').textContent = '✗ server error';
  } finally { btn.disabled = false; }
}
 
document.getElementById('btn').addEventListener('click', generate);
document.getElementById('prompt').addEventListener('keydown', e => { if(e.key==='Enter') generate(); });
fetch('/info').then(r=>r.json()).then(d => { document.getElementById('engine-badge').textContent = d.engine; });
 
(function animate(){
  requestAnimationFrame(animate);
  controls.update();
  renderer.render(scene, camera);
})();
 
function resize(){
  const w = wrap.clientWidth, h = wrap.clientHeight;
  renderer.setSize(w, h);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
}
new ResizeObserver(resize).observe(wrap);
resize();
</script>
</body>
</html>
"""
if not (STATIC_DIR / "viewer.html").exists():
    (STATIC_DIR / "viewer.html").write_text(viewer_html, encoding="utf-8")

# ─────────────────────────────────────────────────────────────────────────────
#  Mixin
# ─────────────────────────────────────────────────────────────────────────────

class Viz3DMixin:

    # ── setup ──────────────────────────────────────────────────────────────

    def setup_3d(
        self,
        engine: Optional[Literal["blenderllm", "shap-e", "auto"]] = "auto",
        blender_path: Optional[str] = None,
        blenderllm_model: str = "FreedomIntelligence/BlenderLLM",
        use_4bit: bool = True,
    ) -> None:
        """
        Args:
            engine:           "blenderllm" | "shap-e" | "auto"
                              auto picks blenderllm if Blender is found, else shap-e
            blender_path:     path to blender.exe. If None, searches PATH + common locations
            blenderllm_model: HuggingFace model id or local path
            use_4bit:         load BlenderLLM in 4-bit (fits 8GB VRAM). set False for better quality
        """
            # CUDA check
        if not _check_cuda():
            print(
                "[3D] WARNING: CUDA not available. "
                "3D generation features are disabled."
            )

            self._viz_engine        = None
            self._blender_exe       = None
            self._blenderllm_pipe   = None
            self._shap_e_models     = None
            self._viz_server        = None
            self._viz_server_thread = None

            return


        self._viz_engine       = None
        self._blender_exe      = None
        self._blenderllm_pipe  = None
        self._shap_e_models    = None
        self._viz_server       = None
        self._viz_server_thread = None

        # find blender
        self._blender_exe = blender_path or self._find_blender()

        if engine == "auto":
            engine = "blenderllm" if self._blender_exe else "shap-e"

        if engine == "blenderllm":
            if not self._blender_exe:
                print("[3D] Blender not found — falling back to shap-e")
                engine = "shap-e"
            else:
                self._load_blenderllm(blenderllm_model, use_4bit)

        if engine == "shap-e":
            self._load_shap_e()

        if self._viz_engine is None:
            self._viz_engine = engine
        print(f"[3D] Engine ready: {self._viz_engine}")

    def _find_blender(self) -> Optional[str]:
        """Search PATH and common Windows install locations."""
        import shutil

        # check PATH first
        found = shutil.which("blender")
        if found:
            return found

        candidates = [
            r"C:\Program Files\Blender Foundation\Blender 5.1\blender.exe",
            r"C:\Program Files\Blender Foundation\Blender 4.5\blender.exe",
            r"C:\Program Files\Blender Foundation\Blender 4.2\blender.exe",
            r"C:\Program Files\Blender Foundation\Blender 4.1\blender.exe",
            r"C:\Program Files\Blender Foundation\Blender 4.0\blender.exe",
            r"C:\Program Files\Blender Foundation\Blender 3.6\blender.exe",
            r"C:\Program Files\Blender Foundation\Blender\blender.exe",
        ]
        for c in candidates:
            if Path(c).exists():
                print(f"[3D] Found Blender at {c}")
                return c

        print("[3D] Blender not found. Install from https://www.blender.org/download/")
        return None

    # ─────────────────────────────────────────────────────────────────────
    # BlenderLLM download / load
    # ─────────────────────────────────────────────────────────────────────

    def _is_blenderllm_downloaded(self, model_id: str) -> bool:
        model_dir = MODELS_DIR / "hub"
        return model_dir.exists() and any(model_dir.iterdir())


    def download_blenderllm(self, model_id: str = "FreedomIntelligence/BlenderLLM") -> None:
        if self._is_blenderllm_downloaded(model_id):
            print("[3D] BlenderLLM already downloaded.")
            return

        print(f"[3D] Downloading BlenderLLM ({model_id})...")

        try:
            from huggingface_hub import snapshot_download

            snapshot_download(
                repo_id=model_id,
                cache_dir=str(MODELS_DIR / "hub"),
                local_files_only=False,
            )

            print("[3D] BlenderLLM download complete.")

        except Exception as e:
            print(f"[3D] BlenderLLM download failed: {e}")
            raise


    def _load_blenderllm(self, model_id: str = "FreedomIntelligence/BlenderLLM", use_4bit: bool = False) -> None:

        if getattr(self, "_blenderllm_model", None) is not None:
            print("[3D] BlenderLLM already loaded.")
            self._viz_engine = "blenderllm"
            return

        self.download_blenderllm(model_id)

        print(f"[3D] Loading BlenderLLM ({model_id})...")

        try:
            from transformers import (
                AutoTokenizer,
                AutoModelForCausalLM,
                BitsAndBytesConfig,
            )

            import torch

            tokenizer = AutoTokenizer.from_pretrained(
                model_id,
                cache_dir=str(MODELS_DIR / "hub"),
                local_files_only=True,
            )

            if use_4bit:
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                )

                model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    quantization_config=bnb_config,
                    device_map="auto",
                    cache_dir=str(MODELS_DIR / "hub"),
                    local_files_only=True,
                )

            else:
                model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    cache_dir=str(MODELS_DIR / "hub"),
                    local_files_only=True,
                )

            self._blenderllm_tokenizer = tokenizer
            self._blenderllm_model = model
            self._viz_engine = "blenderllm"

            print("[3D] BlenderLLM ready.")

        except Exception as e:
            print(f"[3D] BlenderLLM load failed: {e}")
            print("[3D] Falling back to Shap-E engine.")
            self._viz_engine = "shap-e"
            self._blenderllm_tokenizer = None
            self._blenderllm_model = None
            self._load_shap_e()
            raise

    # ─────────────────────────────────────────────────────────────────────
    # Shap-E download / load
    # ─────────────────────────────────────────────────────────────────────

    def _is_shap_e_downloaded(self) -> bool:
        model_dir = MODELS_DIR / "shap_e"
        return model_dir.exists() and any(model_dir.iterdir())


    def download_shap_e(self) -> None:

        if self._is_shap_e_downloaded():
            print("[3D] Shap-E already downloaded.")
            return

        print("[3D] Downloading Shap-E models...")

        try:
            import sys

            shap_e_parent = str(THIS_DIR)

            if shap_e_parent not in sys.path:
                sys.path.insert(0, shap_e_parent)

            from IniyaSecondaryClients.shap_e.models.download import (
                load_model,
            )

            import torch

            device = torch.device("cpu")

            # force HF cache population
            load_model("transmitter", device=device)
            load_model("text300M", device=device)

            (MODELS_DIR / "shap_e").mkdir(exist_ok=True)

            print("[3D] Shap-E download complete.")

        except Exception as e:
            print(f"[3D] Shap-E download failed: {e}")
            raise


    def _load_shap_e(self) -> None:

        if self._shap_e_models is not None:
            print("[3D] Shap-E already loaded.")
            self._viz_engine = "shap-e"
            return

        self.download_shap_e()

        print("[3D] Loading Shap-E...")

        try:
            import sys
            import torch

            shap_e_parent = str(THIS_DIR)

            if shap_e_parent not in sys.path:
                sys.path.insert(0, shap_e_parent)

            from IniyaSecondaryClients.shap_e.diffusion.gaussian_diffusion import (
                diffusion_from_config,
            )

            from IniyaSecondaryClients.shap_e.models.download import (
                load_model,
                load_config,
            )

            device = torch.device("cuda")

            xm = load_model("transmitter", device=device)
            model = load_model("text300M", device=device)
            diffusion = diffusion_from_config(load_config("diffusion"))

            self._shap_e_models = {
                "device": device,
                "xm": xm,
                "model": model,
                "diffusion": diffusion,
            }

            self._viz_engine = "shap-e"

            print(f"[3D] Shap-E ready on {device}.")

        except Exception as e:
            print(f"[3D] Shap-E load failed: {e}")
            raise

    # ─────────────────────────────────────────────────────────────────────────
    #  Generate
    # ─────────────────────────────────────────────────────────────────────────

    def generate_3d(
        self,
        prompt: str,
        filename: str = "output.glb",
        guidance_scale: float = 15.0,
        steps: int = 64,
    ) -> Path:
        """
        Convert text → 3D model (.glb).

        Args:
            prompt:         text description
            filename:       output filename in viz_static/models/
            guidance_scale: shap-e only — higher = more prompt-adherent
            steps:          shap-e only — more = better quality, slower

        Returns:
            Path to the .glb file.
        """
        out_path = GLB_DIR / filename
        print(f"[3D] Generating '{prompt}' → {out_path.name} using {self._viz_engine}...")

        if self._viz_engine == "blenderllm":
            return self._generate_blenderllm(prompt, out_path)
        elif self._viz_engine == "shap-e":
            return self._generate_shap_e(prompt, out_path, guidance_scale, steps)
        else:
            raise RuntimeError("No 3D engine loaded. Call setup_3d() first.")

    # ── BlenderLLM path ───────────────────────────────────────────────────────

    def _query_blenderllm(self, prompt: str) -> str:
        """Generate a bpy Python script from a text prompt."""
        import torch

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a Blender Python (bpy) expert. "
                    "Generate ONLY executable bpy Python code to create the described 3D object. "
                    "Start with import bpy. "
                    "Clear the scene first with: bpy.ops.wm.read_factory_settings(use_empty=True). "
                    "Do NOT include any explanation, markdown fences, or comments. "
                    "Output raw Python only."
                ),
            },
            {"role": "user", "content": f"Create a 3D model of: {prompt}"},
        ]

        text = self._blenderllm_tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._blenderllm_tokenizer(text, return_tensors="pt").to(
            next(self._blenderllm_model.parameters()).device
        )

        with torch.no_grad():
            out = self._blenderllm_model.generate(
                **inputs,
                max_new_tokens=1024,
                temperature=0.2,
                do_sample=True,
                pad_token_id=self._blenderllm_tokenizer.eos_token_id,
            )

        generated = out[0][inputs["input_ids"].shape[1]:]
        script    = self._blenderllm_tokenizer.decode(generated, skip_special_tokens=True)

        # strip markdown fences if model adds them anyway
        if "```" in script:
            lines  = script.split("\n")
            script = "\n".join(
                l for l in lines if not l.strip().startswith("```")
            )

        return script.strip()

    def _generate_blenderllm(self, prompt: str, out_path: Path) -> Path:
        print(f"[3D] BlenderLLM generating: '{prompt}'")
        script = self._query_blenderllm(prompt)

        # inject GLB export at end
        export_snippet = f"""
import bpy
bpy.ops.export_scene.gltf(
    filepath=r"{out_path}",
    export_format="GLB",
    export_apply=True,
    export_materials="EXPORT",
)
print("[3D] Export done.")
"""
        full_script = script + "\n\n" + export_snippet

        # write to temp file
        with tempfile.NamedTemporaryFile(
            suffix=".py", mode="w", delete=False, encoding="utf-8"
        ) as f:
            f.write(full_script)
            script_file = f.name

        print(f"[3D] Running Blender headlessly...")
        try:
            result = subprocess.run(
                [self._blender_exe, "--background", "--python", script_file],
                capture_output=True,
                text=True,
                timeout=120,
            )
            if result.returncode != 0:
                print(f"[3D] Blender stderr:\n{result.stderr[-1000:]}")
                raise RuntimeError("Blender script failed.")
        finally:
            Path(script_file).unlink(missing_ok=True)

        if not out_path.exists():
            raise RuntimeError("Blender ran but no .glb was produced.")

        print(f"[3D] GLB saved → {out_path.name}")
        return out_path

    # ── Shap-E path ───────────────────────────────────────────────────────────

    def _generate_shap_e(
        self,
        prompt: str,
        out_path: Path,
        guidance_scale: float,
        steps: int,
    ) -> Path:
        import torch
        import sys

        shap_e_parent = str(THIS_DIR)
        if shap_e_parent not in sys.path:
            sys.path.insert(0, shap_e_parent)

        from IniyaSecondaryClients.shap_e.diffusion.sample import sample_latents   # no dot
        from IniyaSecondaryClients.shap_e.util.notebooks import decode_latent_mesh  # no dot

        print(f"[3D] Shap-E generating: '{prompt}'")
        m = self._shap_e_models

        latents = sample_latents(
            batch_size=1,
            model=m["model"],
            diffusion=m["diffusion"],
            guidance_scale=guidance_scale,
            model_kwargs={"texts": [prompt]},
            progress=True,
            clip_denoised=True,
            use_fp16=True,
            use_karras=True,
            karras_steps=steps,
            sigma_min=1e-3,
            sigma_max=160,
            s_churn=0,
        )

        tri_mesh = decode_latent_mesh(m["xm"], latents[0]).tri_mesh()

        # shap-e has no native glb export — write obj then convert via trimesh
        try:
            import trimesh
            verts = list(zip(tri_mesh.verts[:, 0], tri_mesh.verts[:, 1], tri_mesh.verts[:, 2]))
            faces = list(zip(tri_mesh.faces[:, 0], tri_mesh.faces[:, 1], tri_mesh.faces[:, 2]))
            mesh  = trimesh.Trimesh(vertices=verts, faces=faces)
            mesh.export(str(out_path))
        except ImportError:
            # fallback: export as .obj and rename (Three.js won't load it but at least it saves)
            obj_path = out_path.with_suffix(".obj")
            with open(obj_path, "w") as f:
                tri_mesh.write_obj(f)
            print("[3D] trimesh not installed — saved as .obj instead of .glb")
            print("     Run: pip install trimesh  for proper .glb output")
            return obj_path

        print(f"[3D] GLB saved → {out_path.name}")
        return out_path


    # ─────────────────────────────────────────────────────────────────────────
    #  Flask viewer server
    # ─────────────────────────────────────────────────────────────────────────

    def serve_viewer(
        self,
        port: int = 5000,
        open_browser: bool = False,
    ) -> None:
        """
        Start Flask viewer on localhost:{port} in a background thread.
        Safe to call multiple times — won't start a second server.
        """
        if self._viz_server_thread and self._viz_server_thread.is_alive():
            print(f"[3D] Viewer already running at http://localhost:{port}")
            return

        try:
            from flask import Flask, send_from_directory, jsonify, request as freq
        except ImportError:
            print("[3D] Flask not installed. Run: pip install flask")
            return

        app = Flask(__name__, static_folder=str(STATIC_DIR))
        viz = self   # reference to the mixin instance

        @app.route("/")
        def index():
            return send_from_directory(str(STATIC_DIR), "viewer.html")

        @app.route("/info")
        def info():
            return jsonify({"engine": viz._viz_engine or "none"})

        @app.route("/generate", methods=["POST"])
        def generate_endpoint():
            data   = freq.get_json()
            prompt = (data or {}).get("prompt", "").strip()
            if not prompt:
                return jsonify({"error": "empty prompt"}), 400
            try:
                glb = viz.generate_3d(prompt)
                rel = glb.relative_to(STATIC_DIR).as_posix()
                return jsonify({"model_url": "/static/" + rel})
            except Exception as e:
                import traceback
                traceback.print_exc()   # ← prints full stack trace to terminal
                return jsonify({"error": str(e)}), 500

        @app.route("/static/<path:path>")
        def static_files(path):
            return send_from_directory(str(STATIC_DIR), path)

        import logging
        log = logging.getLogger("werkzeug")
        log.setLevel(logging.ERROR)   # suppress Flask request logs

        def _run():
            app.run(port=port, threaded=True, use_reloader=False)

        self._viz_server_thread = threading.Thread(target=_run, daemon=True)
        self._viz_server_thread.start()
        time.sleep(0.8)  # give Flask a moment to bind

        print(f"[3D] Viewer running at http://localhost:{port}")

        if open_browser:
            webbrowser.open(f"http://localhost:{port}")