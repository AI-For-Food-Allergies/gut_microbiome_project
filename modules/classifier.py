# Classifier module integrating foundation model and classification head
import torch
import torch.nn as nn
from typing import Dict

from model import MicrobiomeTransformer

class MicrobiomeClassifier(nn.Module):
    def __init__(self, model: MicrobiomeTransformer, classification_head_type: str):
        super().__init__()
        self.model = model
        self.classification_head_type = classification_head_type
        
        self.classification_head = self.init_classification_head()
        
    def init_classification_head(self):
        ...

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        ...