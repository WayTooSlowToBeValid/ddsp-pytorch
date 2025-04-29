import torch
import torchaudio
import os
import sys
from omegaconf import OmegaConf
import argparse


# Setup path for imports
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../")

# Import your models
from network.autoencoder_new.autoencoder_new import AutoEncoder

# ========== Argument parser (same as in toScript.py) ==========
parser = argparse.ArgumentParser()
parser.add_argument("--input", default=".wav")
parser.add_argument("--output", default="output.wav")
parser.add_argument("--ckpt", default=".pth")
parser.add_argument("--config", default=".yaml")
parser.add_argument("--wave_length", type=int, default=16000)
args = parser.parse_args()




# ========== Load input audio ==========
y, sr = torchaudio.load(
    args.input, num_frames=None if args.wave_length == 0 else args.wave_length
)

# ========== Load config ==========
config = OmegaConf.load(args.config)
if sr != config.sample_rate:
    resampler = torchaudio.transforms.Resample(sr, config.sample_rate)
    y = resampler(y)

print("File :", args.input, "Loaded")
device = torch.device("cpu")
print(f"Using device: {device}")

# ========== Load Normal (eager) Autoencoder ==========
model = AutoEncoder(config, device=device).to(device)
model.load_state_dict(torch.load(args.ckpt, map_location=device), strict=False)
model.eval()

# ========== Load TorchScripted Autoencoder ==========
scripted_model = torch.jit.load("weight/newNet_autoencoder_scripted.pt", map_location=device)
scripted_model.eval()

print("Both models loaded successfully!")


# ========== Create CREPE since crepe not scriptable ==========
# 1. Load CREPE separately
from components.ptcrepe.ptcrepe.crepe import CREPE

# 2. Instantiate CREPE model
crepe_model = CREPE(config.crepe).to(device)
crepe_model.eval()

# 3. Compute f0 manually
with torch.no_grad():
    time, f0, confidence, activation = crepe_model.predict(
        y,
        sr=config.sample_rate,
        viterbi=True,
        step_size=int(config.frame_resolution * 1000),
        batch_size=32,
    )
    f0 = f0.float().to(device)
    f0[confidence < 0.5] = 0.0
    f0 = f0[:-1]  # Remove last frame if needed



# ========== Forward pass ==========
with torch.no_grad():
    # (If needed, use model.reconstruction instead of model.forward)
    recon_old = model.reconstruction(y)
    recon_new = scripted_model.forward(y,f0.unsqueeze(0), True)


# ========== Compare outputs ==========
print("\n=== Comparing outputs ===")
for k in recon_old:
    if k in recon_new:
        diff = (recon_old[k] - recon_new[k]).abs().max()
        print(f"Difference in '{k}': {diff.item()}")
    else:
        print(f"Warning: Key '{k}' not found in scripted model output.")

print("\nComparison complete.")

