"""
Implementation of decoder network architecture of DDSP.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor #Tensor now defined
from typing import Optional #z is optional and COULD be left out. 


class MLP(nn.Module):
    """
    MLP (Multi-layer Perception). 

    One layer consists of what as below:
        - 1 Dense Layer
        - 1 Layer Norm
        - 1 ReLU

    constructor arguments :
        n_input : dimension of input
        n_units : dimension of hidden unit
        n_layer : depth of MLP (the number of layers)
        relu : relu (default : nn.ReLU, can be changed to nn.LeakyReLU, nn.PReLU for example.)

    input(x): torch.tensor w/ shape(B, ... , n_input)
    output(x): torch.tensor w/ (B, ..., n_units)
    """

    def __init__(self, n_input, n_units, n_layer, relu=nn.ReLU, inplace=True):
        super().__init__()
        self.n_layer = n_layer
        self.n_input = n_input
        self.n_units = n_units
        self.inplace = inplace

        self.layers = nn.ModuleList()

        self.layers.append(nn.Sequential(
            nn.Linear(n_input, n_units),
            nn.LayerNorm(n_units),
            relu(inplace = self.inplace)
        ))
        for _ in range(n_layer - 1):
            self.layers.append(nn.Sequential(
                nn.Linear(n_units, n_units),
                nn.LayerNorm(n_units),
                relu(inplace = self.inplace)
            ))
  

        
    def forward(self, x):

        for layer in self.layers:
            x = layer(x)
        return x
    

class Decoder(nn.Module):
    """
    Decoder.

    Constructor arguments: 
        use_z : (Bool), if True, Decoder will use z as input.
        mlp_units: 512
        mlp_layers: 3
        z_units: 16
        n_harmonics: 101
        n_freq: 65
        gru_units: 512
        bidirectional: False

    input(dict(f0, z(optional), l)) : a dict object which contains key-values below
        f0 : fundamental frequency for each frame. torch.tensor w/ shape(B, time)
        z : (optional) residual information. torch.tensor w/ shape(B, time, z_units)
        loudness : torch.tensor w/ shape(B, time)

        *note dimension of z is not specified in the paper.

    output : a dict object which contains key-values below
        f0 : same as input
        c : torch.tensor w/ shape(B, time, n_harmonics) which satisfies sum(c) == 1
        a : torch.tensor w/ shape(B, time) which satisfies a > 0
        H : noise filter in frequency domain. torch.tensor w/ shape(B, frame_num, filter_coeff_length)
    """

    def __init__(self, use_z: bool, mlp_layers: int, mlp_units: int, gru_units: int, bidirectional: bool, n_harmonics: int, n_freq: int, z_units: int, device):
        super().__init__()

        self.device = device
        self.use_z = use_z
        self.mlp_f0 = MLP(n_input=1, n_units=mlp_units, n_layer=mlp_layers)
        self.mlp_loudness = MLP(n_input=1, n_units=mlp_units, n_layer=mlp_layers)
        if use_z:
            self.mlp_z = MLP(
                n_input=z_units, n_units=mlp_units, n_layer=mlp_layers
            )
            self.num_mlp = 3
        else:
            self.num_mlp = 2
            self.mlp_z = nn.Identity()

        self.gru = nn.GRU(
            input_size=self.num_mlp * mlp_units,
            hidden_size=gru_units,
            num_layers=1,
            batch_first=True,
            bidirectional=bidirectional,
        )

        self.mlp_gru = MLP(
            n_input=gru_units * 2 if bidirectional else gru_units,
            n_units=mlp_units,
            n_layer=mlp_layers,
            inplace=True,
        )

        # one element for overall loudness
        self.dense_harmonic = nn.Linear(mlp_units, n_harmonics + 1)
        self.dense_filter = nn.Linear(mlp_units, n_freq)

    """def forward(self, batch):
        f0 = batch.get("f0")  # Use .get() instead of direct indexing
        loudness = batch.get("loudness")#
    
        if f0 is None or loudness is None:#
            raise ValueError("Missing 'f0' or 'loudness' in batch")#
    
        f0 = f0.unsqueeze(-1)#
        loudness = loudness.unsqueeze(-1)#

        if self.use_z:
            z = batch["z"]
            latent_z = self.mlp_z(z)

        latent_f0 = self.mlp_f0(f0)
        latent_loudness = self.mlp_loudness(loudness)

        if self.use_z:
            latent = torch.cat((latent_f0, latent_z, latent_loudness), dim=-1)
        else:
            latent = torch.cat((latent_f0, latent_loudness), dim=-1)

        latent, (h) = self.gru(latent)
        latent = self.mlp_gru(latent)

        amplitude = self.dense_harmonic(latent)

        a = amplitude[..., 0]
        a = Decoder.modified_sigmoid(a)

        # a = torch.sigmoid(amplitude[..., 0])
        c = F.softmax(amplitude[..., 1:], dim=-1)

        H = self.dense_filter(latent)
        H = Decoder.modified_sigmoid(H)

        c = c.permute(0, 2, 1)  # to match the shape of harmonic oscillator's input.

        return dict(f0=batch["f0"], a=a, c=c, H=H)"""
    
    def forward(self, f0: Tensor, loudness: Tensor, z: Optional[Tensor] = None):
           
        if f0 is None or loudness is None:#
            raise ValueError("Missing 'f0' or 'loudness' in batch")#
    
        f0 = f0.unsqueeze(-1)#
        loudness = loudness.unsqueeze(-1)          

        latent_f0 = self.mlp_f0(f0)
        latent_loudness = self.mlp_loudness(loudness)

        if self.use_z and z is not None:
            latent_z = self.mlp_z(z)
            latent = torch.cat((latent_f0, latent_z, latent_loudness), dim=-1)
        else:
            latent = torch.cat((latent_f0, latent_loudness), dim=-1)

        latent, (h) = self.gru(latent)
        latent = self.mlp_gru(latent)

        amplitude = self.dense_harmonic(latent)

        a = amplitude[..., 0]
        a = Decoder.modified_sigmoid(a)

        # a = torch.sigmoid(amplitude[..., 0])
        c = F.softmax(amplitude[..., 1:], dim=-1)

        H = self.dense_filter(latent)
        H = Decoder.modified_sigmoid(H)

        c = c.permute(0, 2, 1)  # to match the shape of harmonic oscillator's input.

        return dict(f0=f0, a=a, c=c, H=H)

    @staticmethod
    def modified_sigmoid(a):
        a = a.sigmoid()
        a = a.pow(2.3026)  # log10
        a = a.mul(2.0)
        a.add_(1e-7)
        return a

