import cv2
import numpy as np
import glob
from moviepy.editor import *
import os

def create_video(FPS, video_file_name):
    print("Saving frames into video file...")
    images = sorted(glob.glob('out/*.png'), key=os.path.getmtime)

    # can also be list of numpy arrays
    video = ImageSequenceClip(images, fps=FPS)

    audio = AudioFileClip(f"audio.wav")
    video.audio = audio
    video.write_videofile(f"{video_file_name}.mp4")
    print(f"Done: Final video file: {video_file_name}.mp4")
