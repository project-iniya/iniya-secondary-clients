requirements = '''
# --- Core ML ---
torch==2.11.0+cu130
torchvision==0.26.0+cu130
--extra-index-url https://download.pytorch.org/whl/cu130

# --- HuggingFace / ML ---
transformers==5.5.4
accelerate==1.13.0
safetensors==0.7.0
tokenizers==0.22.2
huggingface_hub==1.11.0

# --- Audio / Speech ---
vosk==0.3.45

# --- Optimization ---
bitsandbytes==0.49.2

# --- Utilities ---
numpy==2.4.4
requests==2.33.1
tqdm==4.67.3
pillow==12.1.1
regex==2026.4.4
PyYAML==6.0.3
rich==15.0.0
typer==0.24.1
psutil==7.2.2
keyring==25.7.0

# --- Web (if needed) ---
Flask==3.1.3

# --- CLIP (IMPORTANT) ---
git+https://github.com/openai/CLIP.git
'''


def install_requirements():
    import subprocess
    import sys
    import os

    req_path = os.path.join(os.path.dirname(__file__), "requirements.txt")

    if not os.path.exists(req_path):
        with open(req_path, 'w') as f:
            f.write(requirements.strip())

    print("[Iniya] Installing dependencies...")

    subprocess.check_call([
        sys.executable, "-m", "pip", "install",
        "--upgrade", "pip"
    ])

    subprocess.check_call([
        sys.executable, "-m", "pip", "install",
        "-r", req_path
    ])

    from .Client import VizualizerClient , AudioClient, SearchClient
    VizualizerClient()
    AudioClient()
    SearchClient()

    print("[Iniya] Installation complete.")