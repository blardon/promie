import plots
import audio_processing
import time

audio, sr = audio_processing.load_audio("mixed_moods-calm-happy-tension.wav")

sekunden = len(audio) / sr
total_frames = round(sekunden*30)

chromagram = audio_processing.chromagram_harmonic(audio, sr, total_frames)
plots.plot_chromagram_multiple_lines(chromagram, 12)

time.sleep(5000)
