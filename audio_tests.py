import os
import re
from typing import List

import click
import dnnlib
import numpy as np
import PIL.Image
import torch
import pickle

import legacy

import librosa as rosa
from scipy import signal
import soundfile as sf
import audioreactive as ar
import torch.nn.functional as F

# audio has -1 to 1 floats
SMF = 1

def load_audio(file):
    audio, sampling_rate = rosa.load(file)
    return audio, sampling_rate


def low_pass_filter(audio_signal, sampling_rate):
    freq = 150
    w = freq / (sampling_rate / 2)
    b, a = signal.butter(5, w, 'low')
    audio_filtered_low = signal.filtfilt(b, a, audio_signal)
    return audio_filtered_low

def chromagram(audio_signal, sampling_rate, frames, notes):
    chromagram = ar.chroma(audio_signal, sampling_rate, frames, notes=notes)
    return chromagram

def chroma_weight_latents(chroma, latents):
    """Creates chromagram weighted latent sequence

    Args:
        chroma (th.tensor): Chromagram (frames, notes)
        latents (th.tensor): Latents (must have same number as number of notes in chromagram)

    Returns:
        th.tensor: Chromagram weighted latent sequence
    """
    base_latents = (chroma[..., None, None] * latents[None, ...]).sum(1)
    return base_latents

def get_latents(audio_signal, sampling_rate, frames, latents):
    chroma = chromagram(audio_signal, sampling_rate, frames, 12)
    chroma_latents = chroma_weight_latents(chroma, latents)
    #latents = gaussian_filter(chroma_latents, 2)

    return chroma_latents

def gaussian_filter(x, sigma, causal=None):
    """Smooth 3 or 4 dimensional tensors along time (first) axis with gaussian kernel.

    Args:
        x (th.tensor): Tensor to be smoothed
        sigma (float): Standard deviation for gaussian kernel (higher value gives smoother result)
        causal (float, optional): Factor to multiply right side of gaussian kernel with. Lower value decreases effect of "future" values. Defaults to None.

    Returns:
        th.tensor: Smoothed tensor
    """
    dim = len(x.shape)
    while len(x.shape) < 3:
        x = x[:, None]

    # radius =  min(int(sigma * 4 * SMF), int(len(x) / 2) - 1)  # prevent padding errors on short sequences
    radius = int(sigma * 4 * SMF)
    channels = x.shape[1]

    kernel = torch.arange(-radius, radius + 1, dtype=torch.float32, device=x.device)
    kernel = torch.exp(-0.5 / sigma ** 2 * kernel ** 2)
    if causal is not None:
        kernel[radius + 1 :] *= 0 if not isinstance(causal, float) else causal
    kernel = kernel / kernel.sum()
    kernel = kernel.view(1, 1, len(kernel)).repeat(channels, 1, 1)

    if dim == 4:
        t, c, h, w = x.shape
        x = x.view(t, c, h * w)
    x = x.transpose(0, 2)

    x = F.pad(x, (radius, radius), mode="circular")
    x = F.conv1d(x, weight=kernel, groups=channels)

    x = x.transpose(0, 2)
    if dim == 4:
        x = x.view(t, c, h, w)

    if len(x.shape) > dim:
        x = x.squeeze()

    return x