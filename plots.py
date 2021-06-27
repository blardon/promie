import librosa as rosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np


def plot_simple_signal(signal, title, xlabel, ylabel):

    fig, ax = plt.subplots()

    ax.plot(signal.cpu().numpy())
    ax.set_title(title)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    plt.tight_layout()
    plt.show()

def plot_chromagram(chromagram, num_notes):

    if chromagram.shape[1] == 12:
        chromagram = chromagram.T

    #chromagram = chromagram.numpy()

    fig, ax = plt.subplots()

    librosa.display.specshow(chromagram, y_axis="chroma", x_axis="time", ax=ax)
    ax.set_title(f'Chromagram {num_notes} notes')

    plt.tight_layout()
    plt.show()

def plot_chromagram_multiple_lines(chromagram, num_notes):

    if chromagram.shape[1] == 12:
        chromagram = chromagram.T

    #chromagram = chromagram.numpy()

    fig, axs = plt.subplots(chromagram.shape[0] + 1)

    for note_index, chromagram_values in enumerate(chromagram):
        axs[note_index].plot(chromagram_values)

    plt.tight_layout()
    plt.show()