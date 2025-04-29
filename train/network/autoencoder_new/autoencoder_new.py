import torch
import torch.nn as nn
from torch import Tensor
from typing import Dict
from components.harmonic_oscillator_new import HarmonicOscillator
from components.reverb_new import TrainableFIRReverb
from components.filtered_noise_new import FilteredNoise
from network.autoencoder_new.decoder_new import Decoder
from network.autoencoder_new.encoder_new import Encoder


class AutoEncoder(nn.Module):
    def __init__(self, config, device=None):
        """
        encoder_config
                use_z=False, 
                sample_rate=16000,
                z_units=16,
                n_fft=2048,
                hop_length=64,
                n_mels=128,
                n_mfcc=30,
                gru_units=512
        
        decoder_config
                mlp_units=512,
                mlp_layers=3,
                use_z=False,
                z_units=16,
                n_harmonics=101,
                n_freq=65,
                gru_units=512,

        components_config
                sample_rate
                hop_length
        """
        super().__init__()
        self.config = config
        self.device = device
        self.use_z = config.use_z
        self.use_reverb = config.use_reverb
        hop_length  = int(config.sample_rate * config.frame_resolution)
        self.decoder = Decoder(use_z=config.use_z,
                               mlp_layers=config.mlp_layers,
                               mlp_units=config.mlp_units,
                               gru_units=config.gru_units,
                               bidirectional = config.bidirectional,
                               n_harmonics=config.n_harmonics,
                               n_freq = config.n_freq,
                               z_units=config.z_units,
                               device=device                             
        )   

        self.encoder = Encoder( sample_rate=config.sample_rate,
                                use_z=config.use_z,
                                hop_length = hop_length,
                                device=device,
                                z_n_fft= config.n_fft,
                                z_frame_resolution= config.frame_resolution,
                                z_n_mels= config.n_mels,
                                z_n_mfcc= config.n_mfcc,
                                z_gru_units= config.gru_units,
                                z_z_units= config.z_units,
                                z_bidirectional= config.bidirectional,)

        

        self.harmonic_oscillator = HarmonicOscillator(
            sr=config.sample_rate, frame_length=hop_length, device=device
        )

        #self.filtered_noise = FilteredNoise(frame_length=hop_length, device=device)
        
        self.filtered_noise = FilteredNoise(
                                frame_length=hop_length,
                                filter_coeff_length=config.n_freq,
                                device=device
                            )

        self.reverb = TrainableFIRReverb(reverb_length=config.sample_rate * 3, device=device)

        self.crepe = None
        #self.config = config

    def forward(self, audio: Tensor, f0: Tensor, add_reverb: bool = True) -> Dict[str, Tensor]:
        """
        z

        input(dict(f0, z(optional), l)) : a dict object which contains key-values below
                f0 : fundamental frequency for each frame. torch.tensor w/ shape(B, time)
                z : (optional) residual information. torch.tensor w/ shape(B, time, z_units)
                loudness : torch.tensor w/ shape(B, time)
        """

        encoded = self.encoder(audio,f0)
        #print(encoded)
        if self.use_z:
            z = encoded["z"]
            latent = self.decoder(encoded["f0"], encoded["loudness"], z)
        else:
            latent = self.decoder(encoded["f0"], encoded["loudness"], None)

        print("Checkpoint 1 --- ")
        harmonic = self.harmonic_oscillator(latent)
        noise = self.filtered_noise(latent["H"])

        # Reshape noise to fut harmonic
        noise = noise.reshape(noise.shape[0], -1)  # (batch_size, total_samples)
        noise = noise[:, :harmonic.shape[-1]]  # Crop if necessary

        print("Checkpoint 2 --- ")
       
        output = {
        "harmonic": harmonic,
        "noise": noise,
        "audio_synth": harmonic + noise[:, :harmonic.shape[-1]],
        "a": latent["a"],
        "c": latent["c"],
        }

        if self.use_reverb and add_reverb:
            output["audio_reverb"] = self.reverb(output["audio_synth"])

        return output



    def get_f0(self, x, sample_rate=16000, f0_threshold=0.5):
        """
        input:
            x = torch.tensor((1), wave sample)
        
        output:
            f0 : (n_frames, ). fundamental frequencies
        """
        if self.crepe is None:
            from components.ptcrepe.ptcrepe.crepe import CREPE

            self.crepe = CREPE(self.config.crepe)
            """for param in self.parameters():
                self.device = param.device
                break"""
            self.crepe = self.crepe.to(self.device)
        self.eval()

        with torch.no_grad():
            time, f0, confidence, activation = self.crepe.predict(
                x,
                sr=sample_rate,
                viterbi=True,
                step_size=int(self.config.frame_resolution * 1000),
                batch_size=32,
            )

            f0 = f0.float().to(self.device)
            f0[confidence < f0_threshold] = 0.0
            f0 = f0[:-1]

        return f0
    
    def reconstruction(self, x, sample_rate=16000, add_reverb=True, f0_threshold=0.5, f0=None):
        """
        input:
            x = torch.tensor((1), wave sample)
            f0 (if exists) = (num_frames, )

        output(dict):
            f0 : (n_frames, ). fundamental frequencies
            a : (n_frames, ). amplitudes
            c : (n_harmonics, n_frames). harmonic constants
            sig : (n_samples)
            audio_reverb : (n_samples + reverb, ). reconstructed signal
        """
        #self.eval()

        with torch.no_grad():
            if f0 is None:
                f0 = self.get_f0(x, sample_rate=sample_rate, f0_threshold=f0_threshold)

            batch = dict(f0=f0.unsqueeze(0), audio=x.to(self.device),)

            recon = self.forward(batch["audio"], batch["f0"], add_reverb=add_reverb)

            # make shape consistent(removing batch dim)
            for k, v in recon.items():
                recon[k] = v[0]

            recon["f0"] = f0

            return recon
