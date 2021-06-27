import cv2
import numpy as np
import glob
from moviepy.editor import *
import os

FILENAME = "mixed_moods-calm-happy-tension"
FPS = 30

images = sorted(glob.glob('out/*.png'), key=os.path.getmtime)

# can also be list of numpy arrays
video = ImageSequenceClip(images, fps=FPS)

audio = AudioFileClip(f"{FILENAME}.wav")
video.audio = audio
video.write_videofile(f"{FILENAME}.mp4")
