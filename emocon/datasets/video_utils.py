from emocon.constants import LABEL2EMOTION, IMG_MEAN, IMG_STD
import imageio
import numpy as np
from IPython.display import Image


def unnormalize_img(img):
    """Un-normalizes the image pixels."""
    img = (img * IMG_STD) + IMG_MEAN
    img = (img * 255).astype("uint8")
    return img.clip(0, 255)


def create_gif(video_tensor, filename="sample.gif"):
    """Prepares a GIF from a video tensor.
    
    The video tensor is expected to have the following shape:
    (num_frames, num_channels, height, width).
    """
    frames = []
    for video_frame in video_tensor:
        frame_unnormalized = unnormalize_img(video_frame.permute(1, 2, 0).numpy())
        frames.append(frame_unnormalized)
    kargs = {"duration": 0.25}
    imageio.mimsave(filename, frames, "GIF", **kargs)
    return filename


def display_gif(video_tensor, gif_name="sample.gif"):
    """Prepares and displays a GIF from a video tensor."""
    video_tensor = video_tensor.permute(1, 0, 2, 3)
    gif_filename = create_gif(video_tensor, gif_name)
    return Image(filename=gif_filename)


def investigate_video(sample_video):
    """Utility to investigate the keys present in a single video sample."""
    for key in sample_video.keys():
        if key == "video":
            print(key, sample_video[key].shape)
        elif key == "text":
            continue
        else:
            print(key, sample_video[key])

    print(f"Video label: {LABEL2EMOTION[sample_video['label']]}")