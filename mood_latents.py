import os
import argparse

import torch
import torchvision
import clip
import numpy as np
from PIL import Image
import legacy
import dnnlib
from tqdm import tqdm

#from stylegan_models import g_all, g_synthesis, g_mapping

device = 'cuda' if torch.cuda.is_available() else 'cpu'
clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)

#g_synthesis.eval()
#g_synthesis.to(device)

def tensor_to_pil_img(img):
    img = (img.clamp(-1, 1) + 1) / 2.0
    img = img[0].permute(1, 2, 0).detach().cpu().numpy() * 255
    img = Image.fromarray(img.astype('uint8'))
    return img

def compute_clip_loss(img, text):
    img = torch.nn.functional.interpolate(img, (224, 224))
    tokenized_text = clip.tokenize([text]).to(device)

    img_logits, _text_logits = clip_model(img, tokenized_text)

    return 1/img_logits * 100

LR = 0.01
LOG_FREQ = 25
STEPS = 450

def find_latent(G, annotation, steps):

    g_mapping = G.mapping
    g_synthesis = G.synthesis

    ### experimental
    latents_init = g_mapping.w_avg.detach().clone().unsqueeze(0)
    std = torch.std(latents_init.cpu())
    r = torch.rand(latents_init.size()) * 2 - 1
    r = r * std
    latents_init = latents_init.cuda() + r.cuda()
    ###

    ### experimental
    best_init_w = None
    best_init_loss = 1000
    for i in range(100):
        latents_init = torch.randn([1, 512]).to(device)
        w = g_mapping(latents_init, None)
        img = g_synthesis(w)
        loss = compute_clip_loss(img, annotation)
        if loss < best_init_loss:
            best_init_loss = loss
            best_init_w = w
    latents_init = best_init_w[0][0].detach().clone().unsqueeze(0)
    ###

    ### experimental
    #latents_init = torch.randn([1, 512]).to(device)
    #w = g_mapping(latents_init, None)
    #latents_init = w[0][0].detach().clone().unsqueeze(0)
    ###

    latents = torch.nn.Parameter(latents_init, requires_grad=True)
    optimizer = torch.optim.Adam(
        params=[latents],
        lr=LR,
        betas=(0.9, 0.999),
    )

    best_loss = 1000
    best_latent = None
    for i in tqdm(range(steps)):
        dlatents = latents.repeat(1,18,1)
        img = g_synthesis(dlatents)

        loss = compute_clip_loss(img, annotation)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        num_loss = loss.data.cpu().numpy()[0][0]
        if num_loss < best_loss:
            best_loss = num_loss
            best_latent = dlatents.clone().detach()

    return best_latent.cpu()
