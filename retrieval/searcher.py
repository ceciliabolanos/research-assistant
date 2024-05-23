import torch.nn as nn
import torch
from transformers import RobertaTokenizer, RobertaModel
import numpy as np
from typing import Optional, List, Dict, Union, Any
from langchain_community.vectorstores import FAISS
from models.model import Model
import os
import subprocess

class Searcher:
    def __init__(self, device: Optional[torch.device] = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else device
        self.query_embeddings_cache = {}
    
        self.embeddings = []
        self.items = []
        self.metadatas = []

    def load_model(self, model_path: str) -> Model:
        raise NotImplementedError("Subclasses must implement the load_model method.")

    def generate_embeddings(self, items: List[str], metadatas: Optional[List[Dict[str, Any]]] = None) -> np.ndarray:
        raise NotImplementedError("Subclasses must implement the generate_embeddings method.")

    def build_faiss_index(self):
        self.faiss_index = FAISS.from_embeddings(list(zip(self.items, self.embeddings)), self.model, metadatas=self.metadatas)

    def similarity_search(self, query: str, k: int = 4, **kwargs: Any) -> List[Dict[str, Union[int, str, float]]]:
        raise NotImplementedError("Subclasses must implement the similarity_search method.")

    def save_to_disk(self, folder_path: str):
        if self.faiss_index is None:
            self.build_faiss_index()
        self.faiss_index.save_local(folder_path=self.folder_path)

    @classmethod
    def load_from_disk(cls, folder_path: str, model_path: str, device: Optional[torch.device] = None):
        raise NotImplementedError("Subclasses must implement the load_from_disk method.")