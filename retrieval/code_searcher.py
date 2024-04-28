import torch.nn as nn
import torch
from transformers import RobertaTokenizer, RobertaModel
import numpy as np
from typing import Optional, List, Dict, Union, Any
from langchain_community.vectorstores import FAISS
from models.model import Model
import os
import subprocess

class CodeSearcher:
    def __init__(self, model_path: str, github_repo: str, 
                 device: Optional[torch.device] = None, faiss_path: Optional[str] = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else device
        self.tokenizer = RobertaTokenizer.from_pretrained('microsoft/unixcoder-base')
        self.model = self.load_model(model_path)
        # self.code_embeddings_cache = self.load_cache(cache_path) if cache_path else {}  # Cache for code embeddings
        self.query_embeddings_cache = {}  # Cache for query embeddings
        self.code_embeddings = []
        self.embeddings_paths = []
        self.functions_names = []
        self.code = []
        self.code_snippets = []
        self.github_repo = github_repo
        self.faiss_path = f'./databases/{github_repo}'
        if faiss_path:
            self.faiss_index = FAISS.load_local(faiss_path, self.model, allow_dangerous_deserialization = True)
            self.code = [doc.page_content for doc in self.faiss_index.docstore._dict.values()]
            self.code_embeddings = [self.faiss_index.index.reconstruct(int(i)) for i in range(self.faiss_index.index.ntotal)]
            self.embeddings_paths = [doc.metadata['path'] for doc in self.faiss_index.docstore._dict.values()]
            self.functions_names = [doc.metadata['function_name'] for doc in self.faiss_index.docstore._dict.values()]
        else:
            self.faiss_index = None

    def load_model(self, model_path: str) -> Model:
        model = RobertaModel.from_pretrained('microsoft/unixcoder-base')
        model = Model(model)
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.to(self.device)
        model.eval()
        return model

    def save_path_and_function(self, path: str, function_name: str):
        self.functions_names.append(function_name)
        self.embeddings_paths.append(path)

    def generate_code_embeddings(self, code_snippet: List[str], code, path, function_name) -> np.ndarray:
        embeddings = []
        for snippet in code_snippet:
            inputs = self.tokenizer.encode_plus(snippet, add_special_tokens=True, max_length=256, truncation=True, padding='max_length', return_tensors='pt')
            with torch.no_grad():
                embedding = self.model(code_inputs=inputs['input_ids'].to(self.device)).detach().numpy()
            self.code.append(code)
            self.code_snippets.append(snippet)
            self.code_embeddings.append(embedding)
            self.functions_names.append(function_name)
            self.embeddings_paths.append(path)
            embeddings.append(embedding)
       
        return np.vstack(embeddings)

    def save_faiss_index(self):
        # Update the FAISS index
        if self.faiss_index is None:
            self.faiss_index = FAISS.from_embeddings(list(zip(self.code[-len(self.code_snippets):], self.code_embeddings)), self.model)
        else:
            self.faiss_index.add_embeddings(list(zip(self.code[-len(self.code_snippets):], self.code_embeddings)))
        
    def get_query_embedding(self, query: str) -> np.ndarray:
        query_key = str(query)
        if query_key in self.query_embeddings_cache:
            return self.query_embeddings_cache[query_key]
        inputs = self.tokenizer.encode_plus(query, add_special_tokens=True, max_length=128, truncation=True, padding='max_length', return_tensors='pt')
        with torch.no_grad():
            embedding = self.model(code_inputs=inputs['input_ids'].to(self.device))
            self.query_embeddings_cache[query_key] = embedding
        return embedding

    def build_faiss_index(self):
        metadata = [{'path': path, 'function_name': function_name} 
                for path, function_name in zip(self.embeddings_paths, self.functions_names)]
        self.faiss_index = FAISS.from_embeddings(list(zip(self.code, self.code_embeddings)), self.model, metadatas=metadata)

    def similarity_search(self, query: str, k: int = 4, **kwargs: Any) -> List[Dict[str, Union[int, str, float]]]:
        if self.faiss_index is None:
            self.build_faiss_index()
        inputs = self.tokenizer.encode_plus(query, add_special_tokens=True, max_length=128, truncation=True, padding='max_length', return_tensors='pt')
        docs_and_scores = self.faiss_index.similarity_search_with_score(inputs['input_ids'], k=k)
        results = []
        for i, (doc, score) in enumerate(docs_and_scores, start=1):
            index = self.code.index(doc.page_content)
            path, function_name, snippet = self.get_index_info(index)
            result = {
                f"top_{i}": {
                    "index": index,
                    "similarity": float(score),  # Convert numpy.float32 to float
                    "path": path,
                    "function_name": function_name,
                    "snippet": snippet
                }
            }
            results.append(result)
        return results

    def display_tree(self, path, max_depth): 
        try:
            # Run the 'tree' command with the specified path and maximum depth
            output = subprocess.check_output(["tree", "-L", str(max_depth), self.github_repo], universal_newlines=True)
            print(output)
        except subprocess.CalledProcessError as e:
            print(f"Error: {e}")
        except FileNotFoundError:
            print("Error: 'tree' command not found. Please make sure it is installed.")
            
    def search_by_keywords(self, query, k, filters):
        # Step 1: Pre-filter the snippets based on keywords
        filtered_indices = [i for i, snippet in enumerate(self.code) if all(f in snippet for f in filters)]
        
        # Step 2: Calculate embeddings for the filtered snippets
        filtered_embeddings = [self.code_embeddings[i] for i in filtered_indices]
        
        # Step 3: Compute the query embedding
        query_embedding = self.tokenizer.encode_plus(query, add_special_tokens=True, max_length=128, truncation=True, padding='max_length', return_tensors='pt')

        # Step 4: Compute similarities for the filtered embeddings
        filtered_embeddings = np.array(filtered_embeddings)
        dot_product = np.dot(filtered_embeddings, query_embedding)
        norm_product = np.linalg.norm(filtered_embeddings, axis=1) * np.linalg.norm(query_embedding)
        similarities = dot_product / norm_product
        
        # Step 5: Find the top-k similar snippets
        top_k_indices = np.argsort(similarities)[-k:][::-1]
        
        results = []
        for idx in top_k_indices:
            original_index = filtered_indices[idx]
            results.append({
                'index': original_index,
                'similarity': similarities[idx],
                'snippet': self.code[original_index],
                "function_name": self.functions_names[original_index],
                'path': self.embeddings_paths[original_index]
            })
            
        
        return results

    
    def get_index_info(self, index):
        return self.embeddings_paths[index], self.functions_names[index], self.code[index]

    def save_to_disk(self):
        if self.faiss_index is None:
            self.build_faiss_index()
        self.faiss_index.save_local(folder_path = self.faiss_path)

    @classmethod
    def load_from_disk(cls, folder_path: str, model_path: str, device: Optional[torch.device] = None):
        code_searcher = cls(model_path, [], device)
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        code_searcher.device = device
        code_searcher.faiss_index = FAISS.load_local(folder_path, code_searcher.model)
        code_searcher.code = [doc.page_content for doc in code_searcher.faiss_index.docstore._dict.values()]
        code_searcher.code_embeddings = [code_searcher.faiss_index.index.reconstruct(int(i)) for i in range(code_searcher.faiss_index.index.ntotal)]
        return code_searcher