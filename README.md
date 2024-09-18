# Distilation and Cross Distillation

This repository contains the notebooks and models for 
 - distillation information from BERT to a Video Model.
 - contrastive learning model for Emotion Recognition


# Distillation from LLM to Video Encoder 
Language models work better for emotion recognition than video models so the idea is to use the BERT model finetuned on the MELD dataset to distil a video model for the same dataset.

## Fine tuning language model
* Check the `finetune_bert.ipynb`
* The model used is `BERT` but any other model can be used

## Fine tuning the video model before distillation

* Check the `finetune_video_mae.ipynb` for the implementation
* The `Video MAE` model was used because it shows the best performance for encoding the video information.

## Distillation

# Contrastive Learning for Emotion Recognition

The idea is to use videos that depict similar emo