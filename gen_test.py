import torch
import dnnlib
import legacy
import pickle
import PIL.Image
import latents
import copy
import audio_processing

def get_generator(network_pkl):
    with open(f"{network_pkl}.pkl", 'rb') as f:
        G = pickle.load(f)['G_ema'].cuda()  # torch.nn.Module
    return G


def save_img(img, name, path):
    img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    img = PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB').save(f'{path}/gen{name}.png')

G = get_generator("WikiArt_uncond_new")

z1 = torch.randn([1, G.z_dim]).cuda()
w1 = G.mapping(z1, None)
z2 = torch.randn([1, G.z_dim]).cuda()
w2 = G.mapping(z2, None)
z3 = torch.randn([1, G.z_dim]).cuda()
w3 = G.mapping(z3, None)
ws1 = w1.repeat(7, 1, 1)
ws2 = w2.repeat(7, 1, 1)
ws3 = w3.repeat(7, 1, 1)
ws = torch.cat((ws1, ws2, ws3), 0)
#ws = latents.fade_dlatent_sequence(ws)
ws = audio_processing.gaussian_filter(ws, 2)

for i, w in enumerate(ws):
    img = G.synthesis(w.unsqueeze(0), noise_mode="const")
    save_img(img, f"d{i}", "out_gen")
