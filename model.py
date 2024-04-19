import torch.nn as nn
import torch
from transformers import RobertaTokenizer, RobertaModel
import numpy as np
from scipy.spatial.distance import cdist
from typing import Optional, List, Dict, Union

class Model(nn.Module):
    def __init__(self, encoder: RobertaModel):
        super(Model, self).__init__()
        self.encoder = encoder

    def forward(self, code_inputs: Optional[torch.Tensor] = None, nl_inputs: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.encoder.device:  # Ensuring the encoder has a device attribute
            device = self.encoder.device
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if code_inputs is not None:
            code_inputs = code_inputs.to(device)
            outputs = self.encoder(code_inputs, attention_mask=code_inputs.ne(1))[0]
            outputs = (outputs * code_inputs.ne(1)[:, :, None]).sum(1) / code_inputs.ne(1).sum(-1)[:, None]
            return torch.nn.functional.normalize(outputs, p=2, dim=1)
        elif nl_inputs is not None:
            nl_inputs = nl_inputs.to(device)
            outputs = self.encoder(nl_inputs, attention_mask=nl_inputs.ne(1))[0]
            outputs = (outputs * nl_inputs.ne(1)[:, :, None]).sum(1) / nl_inputs.ne(1).sum(-1)[:, None]
            return torch.nn.functional.normalize(outputs, p=2, dim=1)
        else:
            raise ValueError("Either code_inputs or nl_inputs must be provided.")
