import latents
import audio_processing
import generator
import math
import plots
import mood_latents
import torch
import helper

AUDIO_FILE = "audio.wav"
NETWORK_PKL = "model.pkl"
STEPS = 500
FAST_MODE = True # only generates 2 latents and interpolates to create {NOTES} latents
OUTPUT_PATH = "out"
NOTES = 12
FPS = 30
FADE_TIME_BETWEEN_ANNOTATIONS = 2 # in seconds
annotations = [
    #("semantic describing the desired mood or content", ending time of annotation in seconds)
    ("A painting of an iceberg", 15.25),
    ("A painting of hell", 30.33),
    ("A painting of a calm place", -1)
]
def make_visualization():
    #######################################
    # Load StyleGAN Generator
    #######################################
    G = generator.get_generator(NETWORK_PKL)

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
    # Generate N random latents
    # N = number of notes in the chromagram (typically 7 - 12)
    #######################################
    #ws = latents.generate_random_wlatents(G, NOTES)

    #w1 = mood_latents.find_latent("A painting of a calm place")
    #w2 = mood_latents.find_latent("A painting of a calm place")
    annotation_latents = {}
    for annotation, time_until in annotations:
        if annotation in annotation_latents.keys():
            continue
        print(f"Finding dlatents for annotation: {annotation}...")
        ### find latents
        ws = None
        #ws = latents.generate_random_wlatents(G, NOTES)
        
        if FAST_MODE:
            w1 = mood_latents.find_latent(annotation, STEPS)
            w2 = mood_latents.find_latent(annotation, STEPS)
            ws = latents.interpolate_dlatents(w1, w2, NOTES)
        else:
            ws = mood_latents.find_latent(annotation, STEPS)
            for i in range(NOTES-1):
                w = mood_latents.find_latent(annotation, STEPS)
                ws = torch.cat((ws, w), 0)
        
        print(f"Finding dlatents for annotation: {annotation} done.")
        annotation_latents[annotation] = ws

        for i_w, w in enumerate(ws):
            img = generator.generate_from_dlatent(G, w, noise_mode="const")
            annotation_f = annotation.replace(" ", "_")
            generator.save_img(img, f"baseimage_{annotation_f}_{i_w}", "base_gens")

    #ws = mood_latents.find_latent("A painting of a calm place")
    #for i in range(NOTES-1):
    #    w = mood_latents.find_latent("A painting of a calm place")
    #    ws = torch.cat((ws, w), 0)
    #    print(ws.size())
    #ws = latents.interpolate_dlatents(w1, w2, NOTES)


    #######################################
    # Create onsets of audio (representation drums and kicks of audio)
    # for percussive part of audio, percussive later
    #######################################
    onsets_low = audio_processing.onsets(audio, sampling_rate, TOTAL_FRAMES, fmax=200, smooth=5, power=2)
    onsets_high = audio_processing.onsets(audio, sampling_rate, TOTAL_FRAMES, fmin=500, smooth=5, power=2)
    # expand onsets to latent shape
    #onsets_low = onsets_low[:, None, None]
    #onsets_high = onsets_high[:, None, None]

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
    # Weight the generated latents by the chromgram of the audio (extends number of latents to TOTAL_FRAMES)
    # gaussian filter applied to smooth transitions
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
        #chromagram_weighted_ws = latents.chromagram_weight_latents(chromagram[from_frame:to_frame+1], ws_for_annotation)
        #chromagram_weighted_ws = audio_processing.gaussian_filter(chromagram_weighted_ws, 4)

        #chromagram_weighted_ws = onsets_high[from_frame:to_frame+1] * ws_for_annotation[[-4]] + (1 - onsets_high[from_frame:to_frame+1]) * chromagram_weighted_ws
        #chromagram_weighted_ws = onsets_low[from_frame:to_frame+1] * ws_for_annotation[[-7]] + (1 - onsets_low[from_frame:to_frame+1]) * chromagram_weighted_ws
        #chromagram_weighted_ws = audio_processing.gaussian_filter(chromagram_weighted_ws, 2, causal=0.2)

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
    print(chroma_weighted_ws.size())

    #chromagram_weighted_ws = latents.chromagram_weight_latents(chromagram, ws)
    #chromagram_weighted_ws = audio_processing.gaussian_filter(chromagram_weighted_ws, 4)

    #######################################
    # Modulate latents by onsets
    #######################################
    #chromagram_weighted_ws = onsets_high * ws[[-4]] + (1 - onsets_high) * chromagram_weighted_ws
    #chromagram_weighted_ws = onsets_low * ws[[-7]] + (1 - onsets_low) * chromagram_weighted_ws

    #chromagram_weighted_ws = audio_processing.gaussian_filter(chromagram_weighted_ws, 2, causal=0.2)

    #######################################
    # Image generation loop
    #######################################
    for i_frame, dlatent in enumerate(chroma_weighted_ws):
        img = generator.generate_from_dlatent(G, dlatent, noise_mode="const")
        generator.save_img(img, i_frame, OUTPUT_PATH)
    