import torch
import dnnlib
import legacy
import pickle
import PIL.Image

def get_generator(network_pkl, generator_name="G_ema", old_pkl_file=True):
    assert generator_name in ["G", "G_ema"]

    print(f"Loading networks from {network_pkl}...")
    device = torch.device('cuda')

    G = None
    if old_pkl_file:
        with dnnlib.util.open_url(network_pkl) as f:
            G = legacy.load_network_pkl(f)[generator_name].to(device)
    else:
        with open(network_pkl, 'rb') as f:
            G = pickle.load(f)[generator_name].cuda()
    assert G is not None
    return G

def generate_from_dlatent(G, w, noise_mode="const"):
    img = G.synthesis(w.unsqueeze(0).cuda(), noise_mode=noise_mode)
    return img

def save_img(img, name, path):
    img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    img = PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB').save(f'{path}/gen{name}.png')