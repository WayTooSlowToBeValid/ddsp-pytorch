import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional
from torch import nn
class FilteredNoise(nn.Module):
    def __init__(
        self,
        frame_length: int,
        filter_coeff_length: int,
        attenuate_gain: float = 1e-2,
        device: Optional[torch.device] = None
    ):
        super().__init__()
        self.frame_length = frame_length
        self.filter_coeff_length = filter_coeff_length
        self.attenuate_gain = attenuate_gain
        self.device = device

        # Precompute your window once in init
        win_len = filter_coeff_length * 2 - 1
        window = torch.hann_window(win_len, dtype=torch.float32)
        self.register_buffer("filter_window", window)  # buffer, not a param

        self.upsampler = nn.Upsample(scale_factor=self.frame_length, mode="linear", align_corners=False)


    def forward(self, H: Tensor) -> Tensor:

        batch_size, n_frames, n_coeff = H.shape

        # Upsample H
        H = self.upsampler(H.transpose(1,2)).transpose(1,2)  # Now H: (batch_size, upsampled_frames, n_coeff)

        upsampled_frames = H.shape[1]  # now the true frame number
        noise_length = upsampled_frames + self.frame_length

        # Build zero-phase spectrum
        spec = H.unsqueeze(-1).expand(batch_size, upsampled_frames, n_coeff, 2)
        spec = spec.contiguous().view(-1, n_coeff, 2).to(torch.complex64)

        ir = torch.fft.irfft(spec, n=n_coeff * 2 - 1)
        ir = ir.roll(n_coeff - 1, dims=1)
        ir = ir * self.filter_window  # broadcast correctly

        padded = nn.functional.pad(ir, (0, self.frame_length - 1))
        fr = torch.fft.rfft(padded)

        fr = fr.mean(dim=1)
        
        # Generate white noise
        noise = torch.rand(batch_size * upsampled_frames, self.frame_length, dtype=torch.float32, device=self.device) * 2 - 1
        noise = nn.functional.pad(noise, (0, n_coeff * 2 - 2))
        noise_fr = torch.fft.rfft(noise)

        assert fr.shape == noise_fr.shape, f"Shape mismatch: fr {fr.shape} vs noise_fr {noise_fr.shape}"

        filtered_fr = noise_fr * fr

        filtered = torch.fft.irfft(filtered_fr, n=noise.shape[1])
        filtered = filtered.view(batch_size, upsampled_frames, -1) * self.attenuate_gain

        return filtered

