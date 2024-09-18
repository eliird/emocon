from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification
from emocon.constants import LABEL2EMOTION, EMOTION2LABEL
from emocon.models.video_mae import VMAESimCLR

def load_mae_model(model_name):
    return VideoMAEForVideoClassification.from_pretrained(
        model_name,
        label2id=EMOTION2LABEL,
        id2label=LABEL2EMOTION,
        ignore_mismatched_sizes=True,  # provide this in case you're planning to fine-tune an already fine-tuned checkpoint
    )


def load_mae_processor(model_name):
    return VideoMAEImageProcessor.from_pretrained(model_name)


def load_mae_simCLR_model(model_name, num_labels=7):
    return VMAESimCLR(
        model_name=model_name, out_dim=num_labels
    )
     

