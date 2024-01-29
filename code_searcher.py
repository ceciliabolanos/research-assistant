import gdown
import os
import torch.nn as nn
import torch
from transformers import RobertaTokenizer, RobertaModel
import numpy as np
from scipy.spatial.distance import cdist

# current_path = os.getcwd()

# print(os.chdir(current_path))

class Model(nn.Module):
    def __init__(self, encoder):
        super(Model, self).__init__()
        self.encoder = encoder

    def forward(self, code_inputs=None, nl_inputs=None):
        if code_inputs is not None:
            outputs = self.encoder(code_inputs,attention_mask=code_inputs.ne(1))[0]
            outputs = (outputs*code_inputs.ne(1)[:,:,None]).sum(1)/code_inputs.ne(1).sum(-1)[:,None]
            return torch.nn.functional.normalize(outputs, p=2, dim=1)
        else:
            outputs = self.encoder(nl_inputs,attention_mask=nl_inputs.ne(1))[0]
            outputs = (outputs*nl_inputs.ne(1)[:,:,None]).sum(1)/nl_inputs.ne(1).sum(-1)[:,None]
            return torch.nn.functional.normalize(outputs, p=2, dim=1)


class CodeSearcher:
    def __init__(self, model_path, code_snippets, device=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else device
        self.tokenizer = RobertaTokenizer.from_pretrained('microsoft/unixcoder-base')
        self.model = self.load_model(model_path)
        self.code_snippets = code_snippets
        self.code_embeddings = self.generate_code_embeddings(code_snippets)

    def load_model(self, model_path):
        model = RobertaModel.from_pretrained('microsoft/unixcoder-base')
        model = Model(model)
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.to(self.device)
        model.eval()
        return model

    def generate_code_embeddings(self, code_snippets):
        embeddings = []
        for snippet in code_snippets:
            inputs = self.tokenizer.encode_plus(snippet, add_special_tokens=True, max_length=256, truncation=True, padding='max_length', return_tensors='pt')
            with torch.no_grad():
                embedding = self.model(code_inputs=inputs['input_ids'].to(self.device))
            embeddings.append(embedding.cpu().numpy())
        return np.vstack(embeddings)

    def get_query_embedding(self, query):
        inputs = self.tokenizer.encode_plus(query, add_special_tokens=True, max_length=128, truncation=True, padding='max_length', return_tensors='pt')
        with torch.no_grad():
            embedding = self.model(code_inputs=inputs['input_ids'].to(self.device))
        return embedding.cpu().numpy()

    def get_similarity_search(self, query, k):
        query_embedding = self.get_query_embedding(query)
        similarities = 1 - cdist(query_embedding, self.code_embeddings, 'cosine').flatten()

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





# def download_model(file_id, destination):
#     url = f'https://drive.google.com/uc?id={file_id}'
#     gdown.download(url, destination, quiet=False)

# # Replace YOUR_FILE_ID with the actual file ID from the shareable link
# file_id = 'YOUR_FILE_ID'
# destination = 'model.bin'

# download_model(file_id, destination)
    
current_directory = os.getcwd()
current_path = os.getcwd()

# Get the parent directory
parent_directory = os.path.dirname(current_directory)

# Print the parent directory
print("current directory:", current_directory)
print("current_patch:", current_path)
print("Parent directory:", parent_directory)