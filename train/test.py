"""
args : 
--input : input wav
--output : output wav path
--ckpt : pretrained weight file
--config : network-corresponding yaml config file
--wave_length : wave length in format 
    (default : 0, which means all)
    WARNING : gpu memory might be not enough.
"""

import torch
import torchaudio
import os, sys

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../")
from network.autoencoder.autoencoder import AutoEncoder
from omegaconf import OmegaConf

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--input", default=".wav")
parser.add_argument("--output", default="output.wav")
parser.add_argument("--ckpt", default=".pth")
parser.add_argument("--config", default=".yaml")
parser.add_argument("--wave_length", type=int, default=16000)
args = parser.parse_args()

y, sr = torchaudio.load(args.input, num_frames=None if args.wave_length == 0 else args.wave_length)

config = OmegaConf.load(args.config)
if sr != config.sample_rate:
    # Resample if sampling rate is not equal to model's
    resampler = torchaudio.transforms.Resample(sr, config.sample_rate)
    y = resampler(y)

print("File :", args.input, "Loaded")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = AutoEncoder(config, device=device).to(device)
print(f"Using device: {device}")
net.load_state_dict(torch.load(args.ckpt, map_location=device), strict=False)  # also important!
#net = AutoEncoder(config).cuda() this was before. Since use supposed to be on CPU I changed - By Fabio
#net.load_state_dict(torch.load(args.ckpt))
net.eval()

print("Network Loaded")

recon = net.reconstruction(y)

dereverb = recon["audio_synth"].cpu()
# Ensure the shape is (1, num_samples)
if dereverb.ndim == 1:
    dereverb = dereverb.unsqueeze(0)  # Now it's (1, 16000)

torchaudio.save(os.path.splitext(args.output)[0] + "_synth.wav", dereverb, sample_rate=config.sample_rate)

if config.use_reverb:
    recon_add_reverb = recon["audio_reverb"].cpu()
    print("recon_add_reverb shape before save:", recon_add_reverb.shape)
    print("recon_add_reverb ndim: ", recon_add_reverb.ndim)
    if recon_add_reverb.ndim == 1:
        recon_add_reverb = recon_add_reverb.unsqueeze(0)
    torchaudio.save(
        os.path.splitext(args.output)[0] + "_reverb.wav",
        recon_add_reverb,
        sample_rate=config.sample_rate,
    )
