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
import audioreactive as ar
import audio_tests as at

seed = 999
seed2 = 1000
network_pkl = "WikiArt.pkl"
outdir = "test_out"


print('Loading networks from "%s"...' % network_pkl)
device = torch.device('cuda')
with dnnlib.util.open_url(network_pkl) as f:
    G = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore
'''
random_z = np.stack([np.random.RandomState(seed).randn(G.z_dim)])
random_z2 = np.stack([np.random.RandomState(seed2).randn(G.z_dim)])

random_w = G.mapping(torch.from_numpy(random_z).to(device), None)
random_w2 = G.mapping(torch.from_numpy(random_z2).to(device), None)


image = G.synthesis(random_w, noise_mode="const")
image = (image.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8).cpu().numpy()

image2 = G.synthesis(random_w2, noise_mode="const")
image2 = (image2.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8).cpu().numpy()

# Save image from array
os.makedirs(outdir, exist_ok=True)
PIL.Image.fromarray(image[0], 'RGB').save(f'{outdir}/{seed}.png')
PIL.Image.fromarray(image2[0], 'RGB').save(f'{outdir}/{seed2}.png')
'''
audio, sr = at.load_audio("audio.wav")
frames = 240*30

random_latents = np.random.RandomState(seed).randn(12, 18, 512)

latents_seq = at.get_latents(audio, sr, frames, random_latents)

print(latents_seq.size())
frames = 7200
for frame in range(frames):
    w = latents_seq[frame]
    print(w.size(), torch.max(w))
    img = G.synthesis(torch.unsqueeze(w.to(device), 0), noise_mode="const")
    img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8).cpu().numpy()
    PIL.Image.fromarray(img[0], 'RGB').save(f'{outdir}/{frame}.png')
