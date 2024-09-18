# Distilation and Cross Distillation

This repository contains the notebooks and models for 
 - distillation information from BERT to a Video Model.
 - contrastive learning model for Emotion Recognition

# Installation and Demo
use `pip install -e .` to install the repo and then check the notebooks on the how to use the models. More description on the models below.


# Distillation from LLM to Video Encoder 
Language models work better for emotion recognition than video models so the idea is to use the BERT model finetuned on the MELD dataset to distil a video model for the same dataset.

## Fine tuning language model
* Check the `finetune_bert.ipynb`
* The model used is `BERT` but any other model can be used

## Fine tuning the video model before distillation

* Check the `finetune_video_mae.ipynb` for the implementation
* The `Video MAE` model was used because it shows the best performance for encoding the video information.

## Distillation
* A custom wrapper is written around the hugging face trainer class and `KL Divergance loss` between the embeddings of teacher model (BERT) and the student model (Video MAE) is used to train the model.
* Check the `distillation.ipynb` notebook for details.

# Contrastive Learning for Emotion Recognition

The idea is to use videos that depict similar emotions to have a similar to that of [SimCLR](https://github.com/google-research/simclr). So same emotions embeddings are closer to one another compared to different others in the embedding space.

* Check the `contrastive.ipynb` for the details of implementation.