import abc
import faiss
import torch
import torch.nn.functional as nnf


class BaseSimilarity(abc.ABC):

    @abc.abstractmethod
    def __call__(self, query_embedding): ...


class FaissSimilarity(BaseSimilarity):

    def __init__(self, index_path):
        self.index = faiss.read_index(index_path)

    def __call__(self, query_embedding):
        query = query_embedding.copy()
        faiss.normalize_L2(query)
        all_similarities, _ = self.index.search(query, self.index.ntotal)

        return all_similarities


class TorchSimilarity(BaseSimilarity):

    def __init__(self, embedding_array):
        self.embedding_array = embedding_array

    def __call__(self, query_embedding):
        all_similarities = nnf.cosine_similarity(
            torch.from_numpy(query_embedding),
            torch.from_numpy(self.embedding_array),
        ).numpy()

        return all_similarities
