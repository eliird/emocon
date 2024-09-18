import torch
import torch.nn as nn
from collections import namedtuple
import torch.nn.functional as F
from emocon.constants import EMOTION2LABEL, LABEL2EMOTION
from transformers import VideoMAEForVideoClassification


class VMAESimCLR(nn.Module):
    
    def __init__(self, model_name, out_dim, *args, **kwargs):
        super().__init__(*args, **kwargs)
        model = VideoMAEForVideoClassification.from_pretrained(
            model_name,
            label2id=EMOTION2LABEL,
            id2label=LABEL2EMOTION,
            ignore_mismatched_sizes=True,  # provide this in case you're planning to fine-tune an already fine-tuned checkpoint
        )
        self.features = model.videomae
        
        num_features = model.classifier.in_features
        # projection MLP
        self.l1 = nn.Linear(num_features, num_features)
        self.l2 = nn.Linear(num_features, out_dim)
        
        # TODO: Decide to remove or keep it
        self.criterion = nn.CrossEntropyLoss()
        self.output = namedtuple("output", ["loss","embeds", "logits"])
        
    def forward(self, **kwargs):
        labels = kwargs['labels']
        video = kwargs["pixel_values"]
        
        # video = video.to(self.mae.device)
        embeds = self.features(pixel_values=video).last_hidden_state[:, 0, :]
        embeds = embeds.squeeze()
        
        logits = self.l1(embeds)
        logits = F.relu(logits)
        logits = self.l2(logits)
        
        loss = self.get_loss(logits, labels, embeds)
        
        return  (logits, embeds, loss)# self.output(loss=loss, embeds=embeds, logits=logits)
    
    def get_loss(self, logits, labels, embeddings, temperature=0.5, embeddings_weight=1, logits_weight=1):
        try:
            num_embeddings = embeddings.size(0)
            loss = 0.0
            count = 0
            # Convert embeddings to probabilities
            softmax_embeddings = F.softmax(embeddings / temperature, dim=1)
            
            for i in range(num_embeddings):
                for j in range(i + 1, num_embeddings):
                    prob_i = softmax_embeddings[i]
                    prob_j = softmax_embeddings[j]
                    
                    # Compute KL divergence
                    kl_div_ij = F.kl_div(prob_i.log(), prob_j, reduction='batchmean')
                    kl_div_ji = F.kl_div(prob_j.log(), prob_i, reduction='batchmean')
                    

                    # Compute loss
                    if labels[i] == labels[j]:
                        # Same label: encourage KL divergence to be close to zero
                        loss += kl_div_ij + kl_div_ji
                    else:
                        # Different labels: encourage KL divergence to be large
                        loss -= kl_div_ij + kl_div_ji
                    count += 1
            loss =  loss / count if count > 0 else torch.tensor(0.0)
            loss += logits_weight * self.criterion(logits, labels)
            return loss

        except:
            return torch.tensor(0.0)