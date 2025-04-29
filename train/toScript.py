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
from network.autoencoder.autoencoder import AutoEncoder as AutoEncoderOld
from network.autoencoder_new.autoencoder_new import AutoEncoder
from omegaconf import OmegaConf

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--input", default=".wav")
parser.add_argument("--output", default="output.wav")
parser.add_argument("--script_output", default="output.pt")
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
device = torch.device("cpu")
print(f"Using device: {device}") #Quick Check if CPU is used (should since hard coded)
newNet = AutoEncoder(config, device=device).to(device)


newNet.load_state_dict(torch.load(args.ckpt, map_location=device), strict=False)  # also important!



newNet.eval()
#newNet = AutoEncoder(config).cuda() this was before. Since use supposed to be on CPU I changed - By Fabio
#newNet.load_state_dict(torch.load(args.ckpt))

print("Network Loaded")

#-----
# In toScript.py (modify the example_input)
# Calculate hop_length based on config
try:
    print("Trying scripting instead...")
    scripted_net = torch.jit.script(newNet)  # Script the entire module, not just `reconstruction`
    scripted_net.save(args.script_output)
except Exception as e:
    print(e)

#-------

recon = newNet.reconstruction(y)

dereverb = recon["audio_synth"].to(device)
#dereverb = recon["audio_synth"].cpu
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
