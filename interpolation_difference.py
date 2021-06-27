import torch
import dnnlib
import legacy
import pickle
import PIL.Image
import latents

def get_generator(network_pkl):
    with open(f"{network_pkl}.pkl", 'rb') as f:
        G = pickle.load(f)['G_ema'].cuda()  # torch.nn.Module
    return G


def save_img(img, name, path):
    img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    img = PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB').save(f'{path}/gen{name}.png')

G = get_generator("WikiArt_uncond_new")
z1 = torch.randn([1, G.z_dim]).cuda()
z2 = torch.randn([1, G.z_dim]).cuda()
w1 = G.mapping(z1, None)
w2 = G.mapping(z2, None)
c = None

dlatents = latents.interpolate_dlatents(w1, w2, 12)
qlatents = latents.interpolate_dlatents(z1, z2, 12)
for i, latent in enumerate(dlatents):
    img = G.synthesis(latent.unsqueeze(0).cuda(), noise_mode="const")
    save_img(img, f"d{i}", "out_gen")

for i, latent in enumerate(qlatents):
    img = G(latent.unsqueeze(0).cuda(), None, noise_mode="const")
    save_img(img, f"q{i}", "out_gen")