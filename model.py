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
        if code_inputs is not None:
            outputs = self.encoder(code_inputs, attention_mask=code_inputs.ne(1))[0]
            outputs = (outputs * code_inputs.ne(1)[:, :, None]).sum(1) / code_inputs.ne(1).sum(-1)[:, None]
            return torch.nn.functional.normalize(outputs, p=2, dim=1)
        else:
            outputs = self.encoder(nl_inputs, attention_mask=nl_inputs.ne(1))[0]
            outputs = (outputs * nl_inputs.ne(1)[:, :, None]).sum(1) / nl_inputs.ne(1).sum(-1)[:, None]
            return torch.nn.functional.normalize(outputs, p=2, dim=1)