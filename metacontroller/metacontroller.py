import torch
from torch import nn
from torch.nn import Module, GRU
import torch.nn.functional as F

# external modules

from x_transformers import Decoder
from discrete_continuous_embed_readout import Embed, Readout

# helper functions

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

# modules

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
        return_latents = False
    ):
        embed = self.embed(ids)

        latents = self.lower_body(embed)

        # meta controller acts on latents here

        # modified latents sent back

        attended = self.upper_body(latents)

        dist_params = self.readout(attended)

        if not return_latents:
            return dist_params

        return dist_params, latents