import torch
from torch import nn
from torch.nn import Module, GRU, Identity
import torch.nn.functional as F

# einops

import einx
from einops import einsum
from einops.layers.torch import Rearrange

# external modules

from x_transformers import Decoder
from x_mlps_pytorch import Feedforwards

from discrete_continuous_embed_readout import Embed, Readout

# helper functions

def exists(v):
    return v is not None

def identity(t):
    return t

def default(v, d):
    return v if exists(v) else d

# meta controller

class MetaController(Module):
    def __init__(
        self,
        dim_latent,
        *,
        decoder_expansion_factor = 2.,
        decoder_depth = 1,
        hypernetwork_low_rank = 16
    ):
        super().__init__()

        dim_decoder_hidden = int(dim_latent * decoder_expansion_factor)

        assert hypernetwork_low_rank < dim_latent

        self.decoder = Feedforwards(
            dim_in = dim_latent,
            dim = dim_decoder_hidden,
            depth = decoder_depth,
            dim_out = 2 * hypernetwork_low_rank * dim_latent
        )

        self.to_hyper_network_weights = Rearrange('... (two d r) -> two ... d r', two = 2, r = hypernetwork_low_rank)

    def forward(
        self,
        latent
    ):

        decoder_out = self.decoder(latent)

        w1, w2 = self.to_hyper_network_weights(decoder_out)
        hypernetwork_weight = einsum(w1, w2, '... i r, ... j r -> ... i j')

        control_signal = einsum(latent, hypernetwork_weight, '... d1, ... d1 d2 -> ... d1')

        return latent + control_signal

# main transformer, which is subsumed into the environment after behavioral cloning

class Transformer(Module):
    def __init__(
        self,
        dim,
        *,
        embed: Embed | dict,
        lower_body: Decoder | dict,
        upper_body: Decoder | dict,
        readout: Readout | dict
    ):
        super().__init__()

        if isinstance(embed, dict):
            embed = Embed(dim = dim, **embed)

        if isinstance(lower_body, dict):
            lower_body = Decoder(dim = dim, **lower_body)

        if isinstance(upper_body, dict):
            upper_body = Decoder(dim = dim, **upper_body)

        if isinstance(readout, dict):
            readout = Readout(dim = dim, **readout)

        self.embed = embed
        self.lower_body = lower_body
        self.upper_body = upper_body
        self.readout = readout

    def forward(
        self,
        ids,
        meta_controller: Module = Identity(),
        return_latents = False
    ):
        embed = self.embed(ids)

        latents = self.lower_body(embed)

        # meta controller acts on latents here

        modified_latents = meta_controller(latents)

        # modified latents sent back

        attended = self.upper_body(latents)

        dist_params = self.readout(attended)

        if not return_latents:
            return dist_params

        return dist_params, latents