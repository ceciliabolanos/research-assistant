import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Any, Optional, Union
from transformers import AutoTokenizer
from langchain.vectorstores import FAISS
from retrieval.searcher import Searcher

class PaperSearcher(Searcher):
    def __init__(self, device: Optional[torch.device] = None):
        super().__init__(device)
        model_path = 'Alibaba-NLP/gte-large-en-v1.5'  # Hardcoded model path
        self.model = self.load_model(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

    def load_model(self, model_path: str) -> AutoModel:
        return AutoModel.from_pretrained(model_path, trust_remote_code=True)

    def generate_embeddings(self, items: List[str], metadatas: Optional[List[Dict[str, Any]]] = None) -> np.ndarray:
        batch_dict = self.tokenizer(items, max_length=8192, padding=True, truncation=True, return_tensors='pt')
        outputs = self.model(**batch_dict)
        embeddings = outputs.last_hidden_state[:, 0]
        embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings.cpu().numpy()

    def similarity_search(self, query: str, k: int = 4, **kwargs: Any) -> List[Dict[str, Union[int, str, float]]]:
        if query in self.query_embeddings_cache:
            query_embedding = self.query_embeddings_cache[query]
        else:
            query_embedding = self.generate_embeddings([query])[0]
            self.query_embeddings_cache[query] = query_embedding

        if self.faiss_index is None:
            self.build_faiss_index()

        scores, indices = self.faiss_index.search(query_embedding, k)

        results = []
        for i, index in enumerate(indices):
            result = {
                "item": self.items[index],
                "score": scores[i],
                "metadata": self.metadatas[index] if self.metadatas else None
            }
            results.append(result)

        return results

    @classmethod
    def load_from_disk(cls, folder_path: str, device: Optional[torch.device] = None):
        searcher = cls(device)
        searcher.faiss_index = FAISS.load_local(folder_path=folder_path, embeddings_dimension=searcher.model.config.hidden_size)
        searcher.items = searcher.faiss_index.index_to_docstore_id.keys()
        searcher.metadatas = searcher.faiss_index.index_to_metadata.values() if searcher.faiss_index.index_to_metadata else None
        return searcher