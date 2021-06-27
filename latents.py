from numpy.core.arrayprint import repr_format
from numpy.lib.function_base import place
import torch
import dnnlib
import legacy
import pickle
import gc
import numpy as np
import helper
import audio_processing

# from: https://github.com/NVlabs/stylegan2-ada-pytorch
# and: https://github.com/JCBrouwer/maua-stylegan2

def generate_random_wlatents(G, n_latents):

    """
    Generates n_latents random w latents

    old_pkl_file: True if pkl is for StyleGAN1, False if for StyleGAN2

    return torch tensor: [n_latents, 18, 512]
    """
    device = torch.device('cuda')
    z_s = torch.randn([n_latents, G.z_dim]).cuda()
    w_s = G.mapping(z_s, None).cpu()
    del z_s
    gc.collect()
    torch.cuda.empty_cache()
    return w_s

def chromagram_weight_latents2(chromagram, latents):
    """Creates chromagram weighted latent sequence

    Args:
        chromagram (th.tensor): Chromagram
        latents (th.tensor): Latents (must have same number as number of notes in chromagram)

    Returns:
        th.tensor: Chromagram weighted latent sequence
    """
    weighted_latents = (chromagram[..., None, None] * latents[None, ...]).sum(1)
    return weighted_latents

def chromagram_weight_latents(chromagram, latents):
    """Creates chromagram weighted latent sequence

    Args:
        chromagram (th.tensor): Chromagram (2d tensor: [N, NOTES])
        latents (th.tensor): Latents (4d tensor: [N, NOTES, 18, 512])

    Returns:
        th.tensor: Chromagram weighted latent sequence
    """
    latents = latents[:chromagram.size()[0]]
    weighted_latents = (chromagram[..., None, None] * latents).sum(1)
    return weighted_latents

def interpolate_dlatents(start, end, steps):
    start = start.detach().clone().cpu()
    end = end.detach().clone().cpu()
    dlatents = np.vstack([(1 - i) * start + i * end for i in np.linspace(0, 1, steps)])
    return torch.from_numpy(dlatents)


def fade_dlatent_sequence2(dlatents: torch.tensor):
    """
    fades/interpolates sequence of dlatents
    input torch.tensor must have size: [X, 18, 512]
    X = sequence length (number of frames)
    """
    result = dlatents.detach().clone()
    copy = dlatents.detach().clone()
    total_frames = result.size()[0]
    for index, step in enumerate(torch.linspace(0, 1, steps=total_frames)):
        p_left = 1-step.item()
        p_right = step.item()
        l_left = copy[index].detach().clone()
        l_right = copy[total_frames-index-1].detach().clone()

        if index > total_frames//2 - 1:
            tmp = p_left
            p_left = p_right
            p_right = tmp

        w = p_left * l_left + p_right * l_right
        result[index] = w

    return result
        
