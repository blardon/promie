import librosa as rosa
import numpy as np
import scipy
import scipy.signal as signal
import torch as th
import torch.nn.functional as F

# https://github.com/JCBrouwer/maua-stylegan2

def load_audio(file):
    audio, sampling_rate = rosa.load(file)
    return audio, sampling_rate

def chroma_raw(audio, sr, type="cens", nearest_neighbor=True):
    """Creates chromagram

    Args:
        audio (np.array): Audio signal
        sr (int): Sampling rate of the audio
        type (str, optional): ["cens", "cqt", "stft", "deep", "clp"]. Which strategy to use to calculate the chromagram. Defaults to "cens".
        nearest_neighbor (bool, optional): Whether to post process using nearest neighbor smoothing. Defaults to True.

    Returns:
        np.array, shape=(12, n_frames): Chromagram
    """
    if type == "cens":
        ch = rosa.feature.chroma_cens(y=audio, sr=sr)
    elif type == "cqt":
        ch = rosa.feature.chroma_cqt(y=audio, sr=sr)
    elif type == "stft":
        ch = rosa.feature.chroma_stft(y=audio, sr=sr)
    else:
        print("chroma type not recognized, options are: [cens, cqt, deep, clp, or stft]. defaulting to cens...")
        ch = rosa.feature.chroma_cens(y=audio, sr=sr)

    if nearest_neighbor:
        ch = np.minimum(ch, rosa.decompose.nn_filter(ch, aggregate=np.median, metric="cosine"))

    return ch

def chromagram_harmonic(audio, sr, n_frames, margin=16, type="cens", notes=12):
    """Creates chromagram for the harmonic component of the audio

    Args:
        audio (np.array): Audio signal
        sr (int): Sampling rate of the audio
        n_frames (int): Total number of frames to resample envelope to
        margin (int, optional): For harmonic source separation, higher values create more extreme separations. Defaults to 16.
        type (str, optional): ["cens", "cqt", "stft", "deep", "clp"]. Which strategy to use to calculate the chromagram. Defaults to "cens".
        notes (int, optional): Number of notes to use in output chromagram (e.g. 5 for pentatonic scale, 7 for standard western scales). Defaults to 12.

    Returns:
        th.tensor, shape=(n_frames, 12): Chromagram
    """
    y_harm = rosa.effects.harmonic(y=audio, margin=margin)
    chroma = chroma_raw(y_harm, sr, type=type).T
    chroma = signal.resample(chroma, n_frames)
    notes_indices = np.argsort(np.median(chroma, axis=0))[:notes]
    chroma = chroma[:, notes_indices]
    chroma = th.from_numpy(chroma / chroma.sum(1)[:, None]).float() # normalize
    return chroma

def onsets(audio, sr, n_frames, margin=8, fmin=20, fmax=8000, smooth=1, power=1):
    """Creates onset envelope from audio
    Args:
        audio (np.array): Audio signal
        sr (int): Sampling rate of the audio
        n_frames (int): Total number of frames to resample envelope to
        margin (int, optional): For percussive source separation, higher values create more extreme separations. Defaults to 8.
        fmin (int, optional): Minimum frequency for onset analysis. Defaults to 20.
        fmax (int, optional): Maximum frequency for onset analysis. Defaults to 8000.
        smooth (int, optional): Standard deviation of gaussian kernel to smooth with. Defaults to 1.
        power (int, optional): Exponent to raise onset signal to. Defaults to 1.
        type (str, optional): ["rosa", "mm"]. Whether to use librosa or madmom for onset analysis. Madmom is slower but often more accurate. Defaults to "mm".
    Returns:
        th.tensor, shape=(n_frames,): Onset envelope
    """
    y_perc = rosa.effects.percussive(y=audio, margin=margin)
    onset = rosa.onset.onset_strength(y=y_perc, sr=sr, fmin=fmin, fmax=fmax)
    onset = np.clip(signal.resample(onset, n_frames), onset.min(), onset.max())
    onset = th.from_numpy(onset).float()
    onset = gaussian_filter(onset, smooth, causal=0)
    onset = onset ** power
    onset = onset / onset.max()
    return onset

SMF = 1 # FPS / 30
def gaussian_filter(x, sigma, causal=None):
    """Smooth tensors along time (first) axis with gaussian kernel.
    Args:
        x (th.tensor): Tensor to be smoothed
        sigma (float): Standard deviation for gaussian kernel (higher value gives smoother result)
        causal (float, optional): Factor to multiply right side of gaussian kernel with. Lower value decreases effect of "future" values. Defaults to None.
    Returns:
        th.tensor: Smoothed tensor
    """
    dim = len(x.shape)
    n_frames = x.shape[0]
    while len(x.shape) < 3:
        x = x[:, None]

    radius = min(int(sigma * 4 * SMF), 3 * len(x))
    channels = x.shape[1]

    kernel = th.arange(-radius, radius + 1, dtype=th.float32, device=x.device)
    kernel = th.exp(-0.5 / sigma ** 2 * kernel ** 2)
    if causal is not None:
        kernel[radius + 1 :] *= 0 if not isinstance(causal, float) else causal
    kernel = kernel / kernel.sum()
    kernel = kernel.view(1, 1, len(kernel)).repeat(channels, 1, 1)

    if dim == 4:
        t, c, h, w = x.shape
        x = x.view(t, c, h * w)
    x = x.transpose(0, 2)

    if radius > n_frames:  # prevent padding errors on short sequences
        x = F.pad(x, (n_frames, n_frames), mode="circular")
        print(
            f"WARNING: Gaussian filter radius ({int(sigma * 4 * SMF)}) is larger than number of frames ({n_frames}).\n\t Filter size has been lowered to ({radius}). You might want to consider lowering sigma ({sigma})."
        )
        x = F.pad(x, (radius - n_frames, radius - n_frames), mode="constant")
    else:
        x = F.pad(x, (radius, radius), mode="circular")

    x = F.conv1d(x, weight=kernel, groups=channels)

    x = x.transpose(0, 2)
    if dim == 4:
        x = x.view(t, c, h, w)

    if len(x.shape) > dim:
        x = x.squeeze()

    return x