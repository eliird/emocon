DATA_PATH_TEXT = "/media/cv/Extreme Pro1/MELD.Raw/MELD.Raw"
DATA_PATH_VIDEO = "/media/cv/Extreme Pro1/MELD.Raw/reorganized_meld_data"

IMG_MEAN = [0.485, 0.456, 0.406]
IMG_STD = [0.229, 0.224, 0.225]
NUM_FRAMES = 16
RESIZE_TO = (224, 224)
SAMPLE_RATE = 4
FPS = 30
CLIP_DURATION = NUM_FRAMES * SAMPLE_RATE / FPS


# MODEL PARAMS
MODEL_NAME = "EMOCON"
NUM_EPOCHS = 11
LEARNING_RATE = 5e-5
BATCH_SIZE = 4


EMOTION2LABEL = {
    'neutral': 0,
    'joy': 1,
    'sadness': 2,
    'surprise': 3,
    'anger': 4,
    'disgust': 5,
    'fear': 6
}

LABEL2EMOTION = {
    0: 'neutral',
    1: 'joy',
    2: 'sadness',
    3: 'surprise',
    4: 'anger',
    5: 'disgust',
    6: 'fear'
}