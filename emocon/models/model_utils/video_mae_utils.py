from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification



def load_mae_odel(model_name):
    return VideoMAEForVideoClassification.from_pretrained(
        model_name,
        label2id=label2id,
        id2label=id2label,
        ignore_mismatched_sizes=True,  # provide this in case you're planning to fine-tune an already fine-tuned checkpoint
    )

def load_mae_processor(model_name):
    return VideoMAEImageProcessor.from_pretrained(model_name)

def load_mae_model_and_processor(model_name):
    return (load_model(model_name), 
            load_processor(model_name))
     
    
