import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Any, Optional, Union
from transformers import AutoTokenizer, AutoModel
from langchain_community.vectorstores import FAISS

class PaperSearcher():
    def __init__(self, paper_pdf: str, device: Optional[torch.device] = None, faiss_path: Optional[str] = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else device
        model_path = 'Alibaba-NLP/gte-large-en-v1.5'  # Hardcoded model path
        self.model = self.load_model(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.faiss_path = f'./databases/{paper_pdf}'
        self.items = []
        self.embeddings = []
        self.metadatas = []
        if faiss_path:
            self.faiss_index = FAISS.load_local(faiss_path, self.model, allow_dangerous_deserialization=True)
            self.items = [doc.page_content for doc in self.faiss_index.docstore._dict.values()]
            self.embeddings = [self.faiss_index.index.reconstruct(int(i)) for i in range(self.faiss_index.index.ntotal)]
            self.metadatas = [doc.metadata for doc in self.faiss_index.docstore._dict.values()]
        else:
            self.faiss_index = None

    def load_model(self, model_path: str) -> AutoModel:
        return AutoModel.from_pretrained(model_path, trust_remote_code=True)
    
    def generate_embeddings(self, paper_snippets: List[str], metadatas: Optional[List[Dict[str, Any]]] = None) -> np.ndarray:
        embeddings = []
        for snippet in paper_snippets:
            batch_dict = self.tokenizer(snippet, max_length=8192, padding=True, truncation=True, return_tensors='pt')
            outputs = self.model(**batch_dict)
            embedding = outputs.last_hidden_state[:, 0]
            embedding = F.normalize(embedding, p=2, dim=1).to(self.device).detach().numpy()
            self.items.append(snippet)
            self.embeddings.append(embedding[0])
            embeddings.append(embedding)
        if metadatas:
            self.metadatas.extend(metadatas)
        return embeddings
   

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
    
    def save_to_disk(self):
        if self.faiss_index is None:
            self.build_faiss_index()
        self.faiss_index.save_local(folder_path = self.faiss_path)
    
    def build_faiss_index(self):
        self.faiss_index = FAISS.from_embeddings(list(zip(self.items[-len(self.items):], self.embeddings)), self.model, metadatas=self.metadatas)

    @classmethod
    def load_from_disk(cls, folder_path: str, device: Optional[torch.device] = None):
        searcher = cls(device)
        searcher.faiss_index = FAISS.load_local(folder_path=folder_path, embeddings_dimension=searcher.model.config.hidden_size)
        searcher.items = searcher.faiss_index.index_to_docstore_id.keys()
        searcher.metadatas = searcher.faiss_index.index_to_metadata.values() if searcher.faiss_index.index_to_metadata else None
        return searcher