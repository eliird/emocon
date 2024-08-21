import pytorchvideo.data
import os

from pytorchvideo.data.labeled_video_paths import LabeledVideoPaths
from pytorchvideo.data.labeled_video_dataset import LabeledVideoDataset
import torch
import pandas as pd

from pytorchvideo.transforms import (
    ApplyTransformToKey,
    Normalize,
    RandomShortSideScale,
    RemoveKey,
    ShortSideScale,
    UniformTemporalSubsample,
)

from torchvision.transforms import (
    Compose,
    Lambda,
    RandomCrop,
    RandomHorizontalFlip,
    Resize,
)
from emocon.constants import (
    LABEL2EMOTION, 
    EMOTION2LABEL,
    NUM_FRAMES,
    IMG_STD, IMG_MEAN,
    RESIZE_TO,
    DATA_PATH_VIDEO,
    CLIP_DURATION
)

# Training dataset transformations.
TRAIN_TRANSFORM = Compose(
    [
        ApplyTransformToKey(
            key="video",
            transform=Compose(
                [
                    UniformTemporalSubsample(NUM_FRAMES),
                    Lambda(lambda x: x / 255.0),
                    Normalize(IMG_MEAN, IMG_STD),
                    RandomShortSideScale(min_size=256, max_size=320),
                    RandomCrop(RESIZE_TO),
                    RandomHorizontalFlip(p=0.5),
                ]
            ),
        ),
    ]
)

# Validation and evaluation datasets' transformations.
VAL_TRANSFORM = Compose(
    [
        ApplyTransformToKey(
            key="video",
            transform=Compose(
                [
                    UniformTemporalSubsample(NUM_FRAMES),
                    Lambda(lambda x: x / 255.0),
                    Normalize(IMG_MEAN, IMG_STD),
                    Resize(RESIZE_TO),
                ]
            ),
        ),
    ]
)


    

class MELDV(LabeledVideoDataset):  
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def __next__(self, *args, **kwargs):
        return super().__next__(*args, **kwargs)        

    
def load_video_dataset(
    data_path: str,
    clip_sampler,
    video_sampler,
    transform,
    video_path_prefix: str = "",
    decode_audio: bool = True,
    decoder: str = "pyav",
) -> MELDV:

    labeled_video_paths = LabeledVideoPaths.from_path(data_path)
    labeled_video_paths.path_prefix = video_path_prefix
    dataset = MELDV(
        labeled_video_paths,
        clip_sampler,
        video_sampler,
        transform,
        decode_audio=decode_audio,
        decoder=decoder,
    )
    return dataset

def filename(row, filemap, tokenizer):
    d_id = str(row['Dialogue_ID'])
    u_id = str(row['Utterance_ID'])
    filename = f'dia{d_id}_utt{u_id}.mp4'
    filemap[filename] = tokenizer(row['Utterance'], padding='max_length', truncation=True, return_tensors='pt')
    return filename    

def load_map(path, tokenizer):
    filemap = {}
    df = pd.read_csv(path)
    df['filename'] = df.apply(lambda row: filename(row, filemap, tokenizer), axis=1)
    df['Emotion'] = df['Emotion'].apply(lambda x: EMOTION2LABEL[x])
    
    df = df[['Utterance', 'Emotion', 'filename']]
    df = df.rename(
        columns={
            'Utterance': 'text',
            'Emotion': 'label'
        })
    return filemap


def load_video_dataloaders():
    train_dataset = load_video_dataset(
        data_path=os.path.join(DATA_PATH_VIDEO, 'train'),
        video_sampler=torch.utils.data.RandomSampler,
        clip_sampler=pytorchvideo.data.make_clip_sampler("uniform", CLIP_DURATION),
        decode_audio=False,
        transform=TRAIN_TRANSFORM
    )
    val_dataset = load_video_dataset(
        data_path=os.path.join(DATA_PATH_VIDEO, "dev"),
        video_sampler=torch.utils.data.RandomSampler,
        clip_sampler=pytorchvideo.data.make_clip_sampler("uniform", CLIP_DURATION),
        decode_audio=False,
        transform=VAL_TRANSFORM,
    )
    
    test_dataset = load_video_dataset(
        data_path=os.path.join(DATA_PATH_VIDEO, "test"),
        video_sampler=torch.utils.data.RandomSampler,
        clip_sampler=pytorchvideo.data.make_clip_sampler("uniform", CLIP_DURATION),
        decode_audio=False,
        transform=VAL_TRANSFORM,
    )
    
    return (train_dataset, test_dataset, val_dataset)
    
    