import torch
import torch.nn as nn
import torchaudio
from torch import Tensor
from typing import Dict, Optional
from components.loudness_extractor_new import LoudnessExtractor


class Z_Encoder(nn.Module):
    def __init__(
        self,
        sample_rate: int,
        n_mfcc: int,
        n_fft: int,
        frame_resolution: float,
        n_mels: int,
        gru_units: int,
        bidirectional: bool,
        z_units: int
    ):
        super().__init__()
        self.mfcc = torchaudio.transforms.MFCC(
            sample_rate=sample_rate,
            n_mfcc=n_mfcc,
            log_mels=True,
            melkwargs=dict(
                n_fft=n_fft,
                hop_length=int(sample_rate * frame_resolution),
                n_mels=n_mels,
                f_min=20.0,
                f_max=8000.0,
            ),
        )

        self.norm = nn.InstanceNorm1d(n_mfcc, affine=True)
        self.gru = nn.GRU(
            input_size=n_mfcc,
            hidden_size=gru_units,
            num_layers=1,
            batch_first=True,
            bidirectional=bidirectional,
        )
        self.dense = nn.Linear(
            gru_units * 2 if bidirectional else gru_units,
            z_units
        )

    def forward(self, audio: Tensor) -> Tensor:
        x = self.mfcc(audio)
        x = x[:, :, :-1]  # Remove last frame to match dimensions
        x = self.norm(x)
        x = x.permute(0, 2, 1)  # (B, Time, Features)
        x, _ = self.gru(x)
        x = self.dense(x)
        return x


class Encoder(nn.Module):
    def __init__(
        self,
        sample_rate: int,
        use_z: bool,
        hop_length: int,
        device: Optional[torch.device] = None,
        #no dict allowed due to TorchScript - pass manually
        z_n_fft: Optional[int] = None,
        z_frame_resolution: Optional[float] =None,
        z_n_mels: Optional[int] = None,
        z_n_mfcc: Optional[int] = None,
        z_gru_units: Optional[int] = None,
        z_z_units: Optional[int] = None,
        z_bidirectional: Optional[bool] = None,
    ):
        super().__init__()

        self.device = device
        self.hop_length = hop_length
        self.use_z = use_z

        self.loudness_extractor = LoudnessExtractor(
            sr=sample_rate,
            frame_length=hop_length,
            device=device,
        )

        if self.use_z:
            assert sample_rate is not None, "z_sample_rate must be provided if use_z=True"
            assert z_n_fft is not None, "z_n_fft must be provided if use_z=True"
            assert hop_length is not None, "z_hop_length must be provided if use_z=True"
            assert z_n_mels is not None, "z_n_mels must be provided if use_z=True"
            assert z_n_mfcc is not None, "z_n_mfcc must be provided if use_z=True"
            assert z_gru_units is not None, "z_gru_units must be provided if use_z=True"
            assert z_z_units is not None, "z_z_units must be provided if use_z=True"
            assert z_bidirectional is not None, "z_bidirectional must be provided if use_z=True"
            assert z_frame_resolution is not None

            self.z_encoder = Z_Encoder(
                sample_rate=sample_rate,
                n_mfcc=z_n_mfcc,
                n_fft=z_n_fft,
                frame_resolution=z_frame_resolution,
                n_mels=z_n_mels,
                gru_units=z_gru_units,
                bidirectional=z_bidirectional,
                z_units=z_z_units,
                
            )
        else:
            self.z_encoder = nn.Identity()

    def forward(self, audio: Tensor, f0: Tensor) -> Dict[str, Tensor]:
        loudness = self.loudness_extractor(audio)
        outputs: Dict[str, Tensor] = {"f0": f0, "loudness": loudness}

        if self.use_z:
            z = self.z_encoder(audio)
            outputs["z"] = z

        # Align lengths if necessary
        frame_len = f0.shape[-1]
        loudness = outputs["loudness"]
        if loudness.shape[-1] > frame_len:
            outputs["loudness"] = loudness[:, :frame_len]
            if self.use_z:
                outputs["z"] = outputs["z"][:, :frame_len]

        return outputs
