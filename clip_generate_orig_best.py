import os
import argparse

import torch
import torchvision
import clip
import numpy as np
from PIL import Image
import legacy
import dnnlib

from stylegan_models import g_all, g_synthesis, g_mapping
#from utils import GetFeatureMaps, transform_img, compute_loss

#torch.manual_seed(20)

network_pkl = "faces.pkl"
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
    default='A painting of a calm place',
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

args = parser.parse_args()

output_path = args.output_path
batch_size = args.batch_size
prompt = args.prompt
lr = args.lr
img_save_freq = args.img_save_freq
ref_img_path = args.ref_img_path

output_dir = os.path.join(output_path, f'"{prompt}"')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("USING ", device)

if not os.path.exists(outdir):
    os.makedirs(outdir)

clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
vgg16 = torchvision.models.vgg16(pretrained=True).to(device)
vgg_layers = vgg16.features

vgg_layer_name_mapping = {
    '1': "relu1_1",
    '3': "relu1_2",
    '6': "relu2_1",
    '8': "relu2_2",
    # '15': "relu3_3",
    # '22': "relu4_3"
}

g_synthesis.eval()
g_synthesis.to(device)

latent_shape = (batch_size, 1, 512)

normal_generator = torch.distributions.normal.Normal(
    torch.tensor([0.0]),
    torch.tensor([1.]),
)

def truncation(x, threshold=0.7, max_layer=8):
    avg_latent = torch.zeros(batch_size, x.size(1), 512).to(device)
    interp = torch.lerp(avg_latent, x, threshold)
    do_trunc = (torch.arange(x.size(1)) < max_layer).view(1, -1, 1).to(device)
    return torch.where(do_trunc, interp, x)

def tensor_to_pil_img(img):
    img = (img.clamp(-1, 1) + 1) / 2.0
    img = img[0].permute(1, 2, 0).detach().cpu().numpy() * 255
    img = Image.fromarray(img.astype('uint8'))
    return img


clip_transform = torchvision.transforms.Compose([
    # clip_preprocess.transforms[2],
    clip_preprocess.transforms[4],
])

if ref_img_path is None:
    ref_img = None
else:
    ref_img = clip_preprocess(Image.open(ref_img_path)).unsqueeze(0).to(device)

clip_normalize = torchvision.transforms.Normalize(
    mean=(0.48145466, 0.4578275, 0.40821073),
    std=(0.26862954, 0.26130258, 0.27577711),
)

def compute_clip_loss(img, text):
    # img = clip_transform(img)
    #img = torch.nn.functional.upsample_bilinear(img, (224, 224))
    img = torch.nn.functional.interpolate(img, (224, 224))
    tokenized_text = clip.tokenize([text]).to(device)

    img_logits, _text_logits = clip_model(img, tokenized_text)

    return 1/img_logits * 100

def compute_perceptual_loss(gen_img, ref_img):
    gen_img = torch.nn.functional.upsample_bilinear(img, (224, 224))
    loss = 0
    len_vgg_layer_mappings = int(max(vgg_layer_name_mapping.keys()))

    ref_feats = ref_img
    gen_feats = gen_img

    for idx, (name, module) in enumerate(vgg_layers._modules.items()):
        ref_feats = module(ref_feats)
        gen_feats = module(gen_feats)
        if name in vgg_layer_name_mapping.keys():
            loss += torch.nn.functional.mse_loss(ref_feats, gen_feats)
        
        if idx >= len_vgg_layer_mappings:
            break
    
    return loss/len_vgg_layer_mappings

#latents_init = torch.randn(latent_shape).squeeze(-1).to(device)
#latents_init = normal_generator.sample(latent_shape).squeeze(-1).to(device)
#latents_init = torch.zeros(latent_shape).squeeze(-1).to(device)
#latents_init = torch.load("mean_latent.pt")

best_init_w = None
best_init_loss = 1000
for i in range(100):
    latents_init = torch.randn([1, 512]).to(device)
    w = g_mapping(latents_init, None)
    img = g_synthesis(w)
    loss = compute_clip_loss(img, args.prompt)
    if loss < best_init_loss:
        best_init_loss = loss
        best_init_w = w
        img = tensor_to_pil_img(img)
        img.save(os.path.join(outdir, f'init_best.png'))
print(f"Best init loss: {best_init_loss}")

latents_init = best_init_w[0][0].detach().clone().unsqueeze(0)
#latents_init = w[0][0].detach().clone().unsqueeze(0)
#latents_init = g_mapping.w_avg.detach().clone().unsqueeze(0)
latents = torch.nn.Parameter(latents_init, requires_grad=True)

optimizer = torch.optim.Adam(
    params=[latents],
    lr=lr,
    betas=(0.9, 0.999),
)


counter = 0
best_loss = 1000
best_latent = None
while True:
    dlatents = latents.repeat(1,18,1)
    img = g_synthesis(dlatents)
    loss = compute_clip_loss(img, args.prompt)

    # NOTE: uncomment to use perceptual loos. Still WIP. You will need to define
    # the `ref_img_path` to use it. The image referenced will be the one 
    # used to condition the generation.
    # perceptual_loss = compute_perceptual_loss(img, ref_img)
    # loss = loss + perceptual_loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    num_loss = loss.data.cpu().numpy()[0][0]
    if num_loss < best_loss:
        best_loss = num_loss
        best_latent = dlatents.clone().detach()

    if counter % args.img_save_freq == 0:
        img = tensor_to_pil_img(img)
        img.save(os.path.join(outdir, f'{counter}.png'))

        print(f'Step {counter}')
        print(f'Loss {loss.data.cpu().numpy()[0][0]}')
        if counter > 1000:
            img = g_synthesis(best_latent)
            img = tensor_to_pil_img(img)
            img.save(os.path.join(outdir, f'best_orig_loss.png'))
            exit()

    counter += 1
