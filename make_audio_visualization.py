import latents
import audio_processing
import generator
import math
import plots
import mood_latents
import torch
import helper
import os
from tqdm.auto import tqdm

AUDIO_FILE = "audio.wav"
NETWORK_PKL = "model.pkl"
STEPS = 500
FAST_MODE = False # only generates 2 latents and interpolates to create {NOTES} latents
OUTPUT_PATH = "out"
NOTES = 12
FPS = 30
FADE_TIME_BETWEEN_ANNOTATIONS = 2 # in seconds

ONSETS_LOW_FMIN = 0
ONSETS_LOW_FMAX = 200
ONSETS_HIGH_FMIN = 500
ONSETS_HIGH_FMAX = 20000

annotations = [
    #("semantic describing the desired mood or content", ending time of annotation in seconds)
    ("A painting of an iceberg", 15.25),
    ("A painting of hell", 30.33),
    ("A painting of a calm place", -1)
]
audio_processing.SMF = FPS / 30
if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)
if not os.path.exists("base_gens"):
    os.makedirs("base_gens")


def make_visualization():
    #######################################
    # Load StyleGAN Generator
    #######################################
    G = generator.get_generator(NETWORK_PKL)
    G.synthesis.eval()
    #######################################
    # Load audio file
    # Returns: audio tensor, audio sampling rate
    #######################################
    audio, sampling_rate = audio_processing.load_audio(AUDIO_FILE)
    AUDIO_LENGTH_SECONDS = len(audio) / sampling_rate
    TOTAL_FRAMES = math.floor(FPS * AUDIO_LENGTH_SECONDS)
    #audio = audio[:TOTAL_FRAMES+1]

    print(f"Loaded audio {AUDIO_FILE} with sampling rate of {sampling_rate}Hz")
    print(f"Length: {AUDIO_LENGTH_SECONDS}s\n")

    #######################################
    # Find N latents according to entered annotation
    # N = number of notes in the chromagram (typically 7 - 12)
    #######################################
    annotation_latents = {}
    for annotation, time_until in annotations:
        if annotation in annotation_latents.keys():
            continue
        ### find latents
        ws = None
        #ws = latents.generate_random_wlatents(G, NOTES)
        
        if FAST_MODE:
            print(f"Finding 2 dlatents with {STEPS} steps each for annotation: {annotation}...")
            w1 = mood_latents.find_latent(G, annotation, STEPS)
            w2 = mood_latents.find_latent(G, annotation, STEPS)
            ws = latents.interpolate_dlatents(w1, w2, NOTES)
        else:
            print(f"Finding {NOTES} (num notes) dlatents with {STEPS} steps each for annotation: {annotation}...")
            ws = None
            for i in tqdm(range(NOTES)):
                w = mood_latents.find_latent(G, annotation, STEPS)
                if ws is None:
                    ws = w
                else:
                    ws = torch.cat((ws, w), 0)
        
        print(f"Finding dlatents for annotation: {annotation} done.")
        annotation_latents[annotation] = ws

        for i_w, w in enumerate(ws):
            img = generator.generate_from_dlatent(G, w, noise_mode="const")
            annotation_f = annotation.replace(" ", "_")
            generator.save_img(img, f"baseimage_{annotation_f}_{i_w}", "base_gens")

    #######################################
    # Create onsets of audio (representation drums and kicks of audio)
    # for percussive part of audio, harmonic later
    #######################################
    onsets_low = audio_processing.onsets(audio, sampling_rate, TOTAL_FRAMES, fmin=ONSETS_LOW_FMIN, fmax=ONSETS_LOW_FMAX, smooth=5, power=2)
    onsets_high = audio_processing.onsets(audio, sampling_rate, TOTAL_FRAMES, fmin=ONSETS_HIGH_FMIN, fmax=ONSETS_HIGH_FMAX, smooth=5, power=2)

    #plots.plot_simple_signal(onsets_low, "onsets_low", "test", "test")
    #plots.plot_simple_signal(onsets_high, "onsets_high", "test", "test")

    #######################################
    # Create chromagram of audio (representation of the pitch levels of audio)
    # for harmonic part of audio
    #######################################
    chromagram = audio_processing.chromagram_harmonic(audio, sampling_rate, TOTAL_FRAMES, notes=NOTES)
    #raw = audio_processing.chroma_raw(audio, sampling_rate, type="cens", nearest_neighbor=True)
    #plots.plot_chromagram(raw, NOTES)

    #######################################
    # Build base sequence of latents of annotations
    # and weight/modulate sequence by chromagram and onsets
    #######################################
    frame_annotations = helper.get_frame_annotation_dict(FPS, annotations, FADE_TIME_BETWEEN_ANNOTATIONS, TOTAL_FRAMES)
    final_ws = None
    # build w sequence for audio
    for annotation_entry in frame_annotations:
        annotation = annotation_entry["annotation"]
        from_frame = annotation_entry["from_frame"]
        to_frame = annotation_entry["to_frame"]

        ws_for_annotation = annotation_latents[annotation].detach().clone().unsqueeze(0)
        annotation_length = to_frame - from_frame + 1
        ws_for_annotation = ws_for_annotation.repeat(annotation_length, 1, 1, 1)

        if final_ws is None:
            final_ws = ws_for_annotation
        else:
            final_ws = torch.cat((final_ws, ws_for_annotation), 0)

    # fade w sequence
    fade_frame_count = helper.get_frame_index_from_seconds(FPS, FADE_TIME_BETWEEN_ANNOTATIONS)
    for i, annotation_entry in enumerate(frame_annotations):
        if i == 0:
            continue
        center_frame = annotation_entry["from_frame"]
        w_start = final_ws[center_frame-fade_frame_count//2].detach().clone().unsqueeze(0)
        w_end = final_ws[center_frame+fade_frame_count//2].detach().clone().unsqueeze(0)
        faded_ws = latents.interpolate_dlatents(w_start, w_end, fade_frame_count)
        final_ws[center_frame-fade_frame_count//2:center_frame+fade_frame_count//2] = faded_ws

    chroma_weighted_ws = latents.chromagram_weight_latents(chromagram, final_ws)
    chroma_weighted_ws = audio_processing.gaussian_filter(chroma_weighted_ws, 4)

    for frame_index, w in enumerate(chroma_weighted_ws):
        chroma_weighted_ws[frame_index] = onsets_high[frame_index] * final_ws[frame_index][-4] + (1 - onsets_high[frame_index]) * chroma_weighted_ws[frame_index]
        chroma_weighted_ws[frame_index] = onsets_low[frame_index] * final_ws[frame_index][-7] + (1 - onsets_low[frame_index]) * chroma_weighted_ws[frame_index]
    chroma_weighted_ws = audio_processing.gaussian_filter(chroma_weighted_ws, 2, 0.2)
    
    #######################################
    # Image generation loop
    #######################################
    print(f"Saving {TOTAL_FRAMES} frames...")
    for i_frame, dlatent in enumerate(tqdm(chroma_weighted_ws)):
        img = generator.generate_from_dlatent(G, dlatent, noise_mode="const")
        generator.save_img(img, i_frame, OUTPUT_PATH)
    print(f"Frame generation done into directory: {OUTPUT_PATH}")
    