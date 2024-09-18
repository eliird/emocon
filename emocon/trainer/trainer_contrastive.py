from transformers import TrainingArguments, Trainer


class ContrastiveTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)        
        self.labels = None
     

    def compute_loss(self, model, inputs, return_outputs=False):
        self.labels = inputs['labels']         
        logits, _, loss = model(**inputs)

        return (loss, logits) if return_outputs else loss