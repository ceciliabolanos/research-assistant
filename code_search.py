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
        
class CodeSearcher:
    def __init__(self, model_path: str, code_snippets: List[str], device: Optional[torch.device] = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else device
        self.tokenizer = RobertaTokenizer.from_pretrained('microsoft/unixcoder-base')
        self.model = self.load_model(model_path)
        self.code_snippets = code_snippets
        self.code_embeddings_cache = {}  # Cache for code embeddings
        self.query_embeddings_cache = {}  # Cache for query embeddings
        self.code_embeddings = []

    def load_model(self, model_path: str) -> Model:
        model = RobertaModel.from_pretrained('microsoft/unixcoder-base')
        model = Model(model)
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.to(self.device)
        model.eval()
        return model
    
    def generate_code_embeddings(self, code_snippet: List[str]) -> np.ndarray:
        embeddings = []
        for snippet in code_snippet:
            snippet_key = str(snippet)  # Convert the list to a string to use as a cache key
            if snippet_key in self.code_embeddings_cache:  # Check cache first
                self.code_embeddings.append(self.code_embeddings_cache[snippet_key])
                embeddings.append(self.code_embeddings_cache[snippet_key])
                continue
            inputs = self.tokenizer.encode_plus(snippet, add_special_tokens=True, max_length=256, truncation=True, padding='max_length', return_tensors='pt')
            with torch.no_grad():
                embedding = self.model(code_inputs=inputs['input_ids'].to(self.device)).cpu().numpy()
                self.code_embeddings_cache[snippet_key] = embedding  # Save to cache using the string key
            self.code_snippets.append(snippet)
            self.code_embeddings.append(embedding)
            embeddings.append(embedding)
        return np.vstack(embeddings)

    def get_query_embedding(self, query: str) -> np.ndarray:
        query_key = str(query)  # Use the query string itself as the key
        if query_key in self.query_embeddings_cache:  # Check cache first
            return self.query_embeddings_cache[query_key]
        inputs = self.tokenizer.encode_plus(query, add_special_tokens=True, max_length=128, truncation=True, padding='max_length', return_tensors='pt')
        with torch.no_grad():
            embedding = self.model(code_inputs=inputs['input_ids'].to(self.device)).cpu().numpy()
            self.query_embeddings_cache[query_key] = embedding  # Save to cache
        return embedding


    def get_similarity_search(self, query: str, k: int) -> Dict[str, Dict[str, Union[int, str, float]]]:
        query_embedding = self.get_query_embedding(query)
        similarities = 1 - cdist(query_embedding, np.vstack(self.code_embeddings), 'cosine').flatten()

        # Get top-k indices
        top_k_indices = np.argsort(similarities)[-k:]

        # Create a dictionary for the top k results
        results = {}
        for index in reversed(top_k_indices):
            snippet_info = {
                'index': index,
                'snippet': self.code_snippets[index],
                'similarity': similarities[index]
            }
            results[f'top_{k}'] = snippet_info
            k -= 1
    
        return results

