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
from emocon.constants import LABEL2EMOTION, EMOTION2LABEL, DATA_PATH



class MELD(LabeledVideoDataset):  
    def __init__(self, filemap, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.filemap = filemap
        
    def __next__(self, *args, **kwargs):
        iter = super().__next__(*args, **kwargs)        
        iter['text'] = self.filemap[iter['video_name']]
        return iter

    
def load_video_dataset(
    filemap,
    data_path: str,
    clip_sampler,
    video_sampler,
    transform,
    video_path_prefix: str = "",
    decode_audio: bool = True,
    decoder: str = "pyav",
) -> MELD:

    labeled_video_paths = LabeledVideoPaths.from_path(data_path)
    print("Something: ", labeled_video_paths._path_prefix)
    labeled_video_paths.path_prefix = video_path_prefix
    dataset = MELD(
        filemap,
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
    df['Emotion'] = df['Emotion'].apply(lambda x: label2id[x])
    df = df[['Utterance', 'Emotion', 'filename']]
    df = df.rename(
        columns={
            'Utterance': 'text',
            'Emotion': 'label'
        })
    return filemap


    
mean = image_processor.image_mean
std = image_processor.image_std
if "shortest_edge" in image_processor.size:
    height = width = image_processor.size["shortest_edge"]
else:
    height = image_processor.size["height"]
    width = image_processor.size["width"]
resize_to = (height, width)

num_frames_to_sample = video_model.mae.config.num_frames
sample_rate = 4
fps = 30
clip_duration = num_frames_to_sample * sample_rate / fps


# Training dataset transformations.
train_transform = Compose(
    [
        ApplyTransformToKey(
            key="video",
            transform=Compose(
                [
                    UniformTemporalSubsample(num_frames_to_sample),
                    Lambda(lambda x: x / 255.0),
                    Normalize(mean, std),
                    RandomShortSideScale(min_size=256, max_size=320),
                    RandomCrop(resize_to),
                    RandomHorizontalFlip(p=0.5),
                ]
            ),
        ),
    ]
)

# Training dataset.
train_dataset = load_video_dataset(
    filemap = load_map(os.path.join(csv_root_path, "train.csv"), tokenizer),
    data_path=os.path.join(dataset_root_path, "train"),
    video_sampler=torch.utils.data.RandomSampler,
    clip_sampler=pytorchvideo.data.make_clip_sampler("uniform", clip_duration),
    decode_audio=False,
    transform=train_transform,
)

# Validation and evaluation datasets' transformations.
val_transform = Compose(
    [
        ApplyTransformToKey(
            key="video",
            transform=Compose(
                [
                    UniformTemporalSubsample(num_frames_to_sample),
                    Lambda(lambda x: x / 255.0),
                    Normalize(mean, std),
                    Resize(resize_to),
                ]
            ),
        ),
    ]
)

 
val_dataset = load_video_dataset(
    filemap = load_map(os.path.join(csv_root_path, "dev.csv"), tokenizer),
    data_path=os.path.join(dataset_root_path, "dev"),
    video_sampler=torch.utils.data.RandomSampler,
    clip_sampler=pytorchvideo.data.make_clip_sampler("uniform", clip_duration),
    decode_audio=False,
    transform=val_transform,
)
 
test_dataset = load_video_dataset(
    filemap = load_map(os.path.join(csv_root_path, "test.csv"), tokenizer),
    data_path=os.path.join(dataset_root_path, "test"),
    video_sampler=torch.utils.data.RandomSampler,
    clip_sampler=pytorchvideo.data.make_clip_sampler("uniform", clip_duration),
    decode_audio=False,
    transform=val_transform,
)