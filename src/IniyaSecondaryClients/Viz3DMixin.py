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

_check_cuda()  # fail early if no CUDA GPU is available

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
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"


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

    def _load_blenderllm(self, model_id: str, use_4bit: bool) -> None:
        print(f"[3D] Loading BlenderLLM ({model_id}) {'4-bit' if use_4bit else 'full'}...")
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
            import torch

            tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=str(MODELS_DIR / "hub"))

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
                )
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    cache_dir=str(MODELS_DIR / "hub"),
                )

            self._blenderllm_tokenizer = tokenizer
            self._blenderllm_model     = model
            print("[3D] BlenderLLM ready.")
        except Exception as e:
            print(f"[3D] BlenderLLM load failed: {e}")
            print("[3D] Falling back to shap-e")
            self._viz_engine = "shap-e"
            self._load_shap_e()

    def _load_shap_e(self) -> None:
        print("[3D] Loading Shap-E...")
        try:
            import sys
            import torch

            # add the directory containing the shap_e folder to sys.path
            shap_e_parent = str(THIS_DIR)
            if shap_e_parent not in sys.path:
                sys.path.insert(0, shap_e_parent)

            from IniyaSecondaryClients.shap_e.diffusion.sample import sample_latents
            from IniyaSecondaryClients.shap_e.diffusion.gaussian_diffusion import diffusion_from_config
            from IniyaSecondaryClients.shap_e.models.download import load_model, load_config

            device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            xm        = load_model("transmitter", device=device)
            model     = load_model("text300M",    device=device)
            diffusion = diffusion_from_config(load_config("diffusion"))

            self._shap_e_models = {
                "device":    device,
                "xm":        xm,
                "model":     model,
                "diffusion": diffusion,
            }
            print(f"[3D] Shap-E ready on {device}.")
        except Exception as e:
            print(f"[3D] Shap-E load failed: {e}")

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