import torch.nn as nn
import torch
from transformers import RobertaTokenizer, RobertaModel
import numpy as np
from typing import Optional, List, Dict, Union, Any
from langchain_community.vectorstores import FAISS
from models.model import Model
import os
import subprocess

class CodeSearcher():
    def __init__(self, model_path: str, github_repo: str, 
                 device: Optional[torch.device] = None, faiss_path: Optional[str] = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else device
        self.tokenizer = RobertaTokenizer.from_pretrained('microsoft/unixcoder-base')
        self.items = []
        self.embeddings = []
        self.metadatas = []
        self.faiss_path = f'./databases/{github_repo}'
        if faiss_path:
            self.faiss_index = FAISS.load_local(faiss_path, self.model, allow_dangerous_deserialization=True)
            self.items = [doc.page_content for doc in self.faiss_index.docstore._dict.values()]
            self.embeddings = [self.faiss_index.index.reconstruct(int(i)) for i in range(self.faiss_index.index.ntotal)]
            self.metadatas = [doc.metadata for doc in self.faiss_index.docstore._dict.values()]
        else:
            self.faiss_index = None

    def load_model(self, model_path: Optional[str] = None) -> Model:
        model_name = "microsoft/unixcoder-base"
        
        if model_path is None:
            model = RobertaModel.from_pretrained(model_name)
        else:
            model = RobertaModel.from_pretrained(model_name)
            state_dict = torch.load(model_path, map_location=self.device)
            model.load_state_dict(state_dict)
        
        model = Model(model)
        model.to(self.device)
        model.eval()
        
        return model

    def generate_embeddings(self, code_snippets: List[str], metadatas: Optional[List[Dict[str, Any]]] = None) -> np.ndarray:
        embeddings = []
        for snippet in code_snippets:
            inputs = self.tokenizer.encode_plus(snippet, add_special_tokens=True, max_length=256, truncation=True, padding='max_length', return_tensors='pt')
            with torch.no_grad():
                embedding = self.model(code_inputs=inputs['input_ids'].to(self.device)).detach().numpy()
            self.items.append(snippet)
            self.embeddings.append(embedding)
            embeddings.append(embedding)
        if metadatas:
            self.metadatas.extend(metadatas)
        return 

    def similarity_search(self, query: str, k: int = 4, **kwargs: Any) -> List[Dict[str, Union[int, str, float]]]:
        if self.faiss_index is None:
            self.build_faiss_index()
        inputs = self.tokenizer.encode_plus(query, add_special_tokens=True, max_length=128, truncation=True, padding='max_length', return_tensors='pt')
        docs_and_scores = self.faiss_index.similarity_search_with_score(inputs['input_ids'], k=k)
        results = []
        for i, (doc, score) in enumerate(docs_and_scores, start=1):
            index = self.items.index(doc.page_content)
            metadata = self.metadatas[index]
            result = {
                f"top_{i}": {
                    "index": index,
                    "similarity": float(score),
                    "snippet": doc.page_content,
                    "metadata": metadata
                }
            }
            results.append(result)
        return results

    def display_tree(self, max_depth):
        try:
            output = subprocess.check_output(["tree", "-L", str(max_depth), self.github_repo], universal_newlines=True)
            return output
        except subprocess.CalledProcessError as e:
            print(f"Error: {e}")
        except FileNotFoundError:
            print("Error: 'tree' command not found. Please make sure it is installed.")

    def search_by_keywords(self, query, k, filters):
        filtered_indices = [i for i, snippet in enumerate(self.items) if all(f in snippet for f in filters)]
        filtered_embeddings = [self.embeddings[i] for i in filtered_indices]
        query_embedding = self.tokenizer.encode_plus(query, add_special_tokens=True, max_length=128, truncation=True, padding='max_length', return_tensors='pt')
        filtered_embeddings = np.array(filtered_embeddings)
        dot_product = np.dot(filtered_embeddings, query_embedding)
        norm_product = np.linalg.norm(filtered_embeddings, axis=1) * np.linalg.norm(query_embedding)
        similarities = dot_product / norm_product
        top_k_indices = np.argsort(similarities)[-k:][::-1]
        results = []
        for idx in top_k_indices:
            original_index = filtered_indices[idx]
            metadata = self.metadatas[original_index]
            results.append({
                'index': original_index,
                'similarity': similarities[idx],
                'snippet': self.items[original_index],
                'metadata': metadata
            })
        return results
    
    def save_to_disk(self):
        if self.faiss_index is None:
            self.build_faiss_index()
        self.faiss_index.save_local(folder_path = self.faiss_path)

    def build_faiss_index(self):
        self.faiss_index = FAISS.from_embeddings(list(zip(self.items, self.embeddings)), self.model, metadatas=self.metadatas)
        
    @classmethod
    def load_from_disk(cls, folder_path: str, model_path: str, device: Optional[torch.device] = None):
        code_searcher = cls(model_path, "", device)
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        code_searcher.device = device
        code_searcher.faiss_index = FAISS.load_local(folder_path, code_searcher.model)
        code_searcher.items = [doc.page_content for doc in code_searcher.faiss_index.docstore._dict.values()]
        code_searcher.embeddings = [code_searcher.faiss_index.index.reconstruct(int(i)) for i in range(code_searcher.faiss_index.index.ntotal)]
        code_searcher.metadatas = [doc.metadata for doc in code_searcher.faiss_index.docstore._dict.values()]
        return code_searcher