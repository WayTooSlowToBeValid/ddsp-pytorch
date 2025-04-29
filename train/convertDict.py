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


device = torch.device("cpu")
oldNet = AutoEncoderOld(config, device=device).to(device)
print(f"Using device: {device}") #Quick Check if CPU is used (should since hard coded)

#old_state_dict = torch.load(args.ckpt, map_location=device)
#print(state_dict)
#print("Keys in the state_dict:", old_state_dict.keys())

oldNet.load_state_dict(torch.load(args.ckpt, map_location=device), strict=False)  # also important!

#net = AutoEncoder(config).cuda() this was before. Since use supposed to be on CPU I changed - By Fabio
#net.load_state_dict(torch.load(args.ckpt))

oldNet.eval()

newNet = AutoEncoder(config, device=device).to(device)

old_dict = oldNet.state_dict()

for k in oldNet.state_dict().keys():
    print(k)


new_dict = newNet.state_dict()

remapped_sd = {}

for new_key in new_dict.keys():
    if new_key.startswith("decoder.mlp_f0.layers") or new_key.startswith("decoder.mlp_loudness.layers") or new_key.startswith("decoder.mlp_gru.layers"):
        parts = new_key.split('.')
        module = parts[0]  # decoder
        block = parts[1]   # mlp_f0 or mlp_loudness or mlp_gru
        layers_str = parts[2]  # layers
        layer_idx = int(parts[3])  # 0, 1, 2
        sub_layer_idx = parts[4]   # 0 or 1
        param = parts[5]  # weight or bias

        # Map to old format
        old_layer_idx = layer_idx + 1  # because old model starts from 1
        old_key = f"{module}.{block}.mlp_layer{old_layer_idx}.{sub_layer_idx}.{param}"

        if old_key in old_dict:
            remapped_sd[new_key] = old_dict[old_key]
        else:
            print(f"Warning: {old_key} not found in old checkpoint.")
    else:
        # For everything else (gru weights, dense layers, reverb, etc)
        if new_key in old_dict:
            remapped_sd[new_key] = old_dict[new_key]
        else:
            print(f"Warning: {new_key} not found in old checkpoint.")
            
load_info = newNet.load_state_dict(remapped_sd, strict=False)

print(load_info)
print("Missing keys:", load_info.missing_keys)
print("Unexpected keys:", load_info.unexpected_keys)

torch.save(newNet.state_dict(), 'new_checkpoint(2).pth')


