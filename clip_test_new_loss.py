import os
import argparse
import math

import torch
from torch import optim
import torchvision
import clip
import numpy as np
from PIL import Image
import legacy
import dnnlib
from criteria.clip_loss import CLIPLoss

#from stylegan_models import g_all, g_synthesis, g_mapping
#from utils import GetFeatureMaps, transform_img, compute_loss

#torch.manual_seed(20)

network_pkl = "WikiArt_uncond_new.pkl"
#network_pkl = "faces_new.pkl"
outdir = "test_out"

parser = argparse.ArgumentParser()
parser.add_argument(
    '--output_path',
    type=str,
    default='./generations',
    help='',
)
parser.add_argument(
    '--ref_img_path',
    type=str,
    default=None,
    help='',
)
parser.add_argument(
    '--batch_size',
    type=int,
    default=1,
    help='Batch Size',
)
parser.add_argument(
    '--prompt',
    type=str,
    default='Painting of trees and mountains',
    #"An image with..."
    #"The image of a..."
    help='',
)
parser.add_argument(
    '--lr',
    type=float,
    default=0.01,
    help='',
)
parser.add_argument(
    '--img_save_freq',
    type=int,
    default=50,
    help='',
)

def get_lr(t, initial_lr, rampdown=0.25, rampup=0.05):
    lr_ramp = min(1, (1 - t) / rampdown)
    lr_ramp = 0.5 - 0.5 * math.cos(lr_ramp * math.pi)
    lr_ramp = lr_ramp * min(1, t / rampup)

    return initial_lr * lr_ramp

args = parser.parse_args()

output_path = args.output_path
batch_size = args.batch_size
prompt = args.prompt
lr = args.lr
#args.lr_ramp = 1e-3
img_save_freq = args.img_save_freq
ref_img_path = args.ref_img_path

output_dir = os.path.join(output_path, f'"{prompt}"')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("USING ", device)

if not os.path.exists(outdir):
    os.makedirs(outdir)



#g_synthesis.eval()
#g_synthesis.to(device)

print('Loading networks from "%s"...' % network_pkl)
with dnnlib.util.open_url(network_pkl) as f:
    G = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore
    g_synthesis = G.synthesis
    g_mapping = G.mapping

latent_shape = (batch_size, 1, 512)

normal_generator = torch.distributions.normal.Normal(
    torch.tensor([0.0]),
    torch.tensor([1.]),
)

#latents_init = torch.randn(latent_shape).squeeze(-1).to(device)
#latents_init = normal_generator.sample(latent_shape).squeeze(-1).to(device)

args.description = "A painting of tension"
text_inputs = torch.cat([clip.tokenize(args.description)]).cuda()

#latent_code_init = torch.zeros([1, 512])
#latent_code_init = torch.load("mean_latent.pt")
latent_code_init = G.mapping.w_avg.detach().clone().unsqueeze(0)
latent_code_init = latent_code_init.detach().clone().repeat(1, 18, 1)
latent_code_init = latent_code_init.cuda()

latent = latent_code_init.detach().clone()
latent.requires_grad = True

args.stylegan_size = 1024
clip_loss = CLIPLoss(args)

#latents_init = torch.zeros(latent_shape).squeeze(-1).to(device)
#latents = torch.nn.Parameter(latents_init, requires_grad=True)

optimizer = optim.Adam([latent], lr=lr, betas=(0.9, 0.999))

def tensor_to_pil_img(img):
    img = (img.clamp(-1, 1) + 1) / 2.0
    img = img[0].permute(1, 2, 0).detach().cpu().numpy() * 255
    img = Image.fromarray(img.astype('uint8'))
    return img

counter = 0
best_loss = 1000
best_latent = None
args.step = 5000
for i in range(args.step):
    #t = i / args.step
    #lr = get_lr(t, args.lr)
    #optimizer.param_groups[0]["lr"] = lr

    #img = g_synthesis(latent, noise_mode='const', force_fp32=True)
    img = g_synthesis(latent)
    loss = clip_loss(img, text_inputs)

    #dlatents = latents.repeat(1,18,1)
    #dlatents = g_mapping(latents.squeeze(0), None)
    #img = G(latents[0], None)
    #z = latents.squeeze(0)
    #img = G(z, None)

    #loss = compute_clip_loss(img, args.prompt)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    #num_loss = loss.data.cpu().numpy()[0][0]
    #if num_loss < best_loss:
    #    best_loss = num_loss
    #    best_latent = dlatents.clone().detach()

    if counter % args.img_save_freq == 0:
        img = tensor_to_pil_img(img)
        img.save(os.path.join(outdir, f'{counter}.png'))

        print(f'Step {counter}')
        print(f'Loss {loss.data.cpu().numpy()[0][0]}')
        if counter > 1000:
            #img = g_synthesis(best_latent)
            #img = tensor_to_pil_img(img)
            #img.save(os.path.join(outdir, f'best_new_loss.png'))
            exit()

    counter += 1
