import torch.nn as nn
import torch
from transformers import RobertaTokenizer, RobertaModel
import numpy as np
from scipy.spatial.distance import cdist
from typing import Optional, List, Dict, Union
import torch
import torch.nn as nn
from transformers import RobertaModel

class Model(nn.Module):
    def __init__(self, encoder: RobertaModel):
        super(Model, self).__init__()
        self.device =  torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.encoder = encoder.to(self.device)

    def forward(self, code_inputs: Optional[torch.Tensor] = None, nl_inputs: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Choose the correct inputs and ensure they are on the right device
        if code_inputs is not None:
            if not isinstance(code_inputs, torch.Tensor):
                code_inputs = torch.tensor(code_inputs, device=self.device)
            else:
                code_inputs = code_inputs.to(self.device)

            outputs = self.encoder(code_inputs, attention_mask=code_inputs.ne(1))[0]
            outputs = (outputs * code_inputs.ne(1)[:, :, None]).sum(1) / code_inputs.ne(1).sum(-1)[:, None]
            return torch.nn.functional.normalize(outputs, p=2, dim=1).detach().numpy()

        else:
            if not isinstance(nl_inputs, torch.Tensor):
                nl_inputs = torch.tensor(nl_inputs, device=self.device)
            else:
                nl_inputs = nl_inputs.to(self.device)

            outputs = self.encoder(nl_inputs, attention_mask=nl_inputs.ne(1))[0]
            outputs = (outputs * nl_inputs.ne(1)[:, :, None]).sum(1) / nl_inputs.ne(1).sum(-1)[:, None]
            return torch.nn.functional.normalize(outputs, p=2, dim=1)

