import abc
import logging

import faiss
import torch
import torch.nn.functional as nnf


class BaseSimilarity(abc.ABC):

    @abc.abstractmethod
    def __call__(self, query_embedding):
        """Compute the similarities and return the ordered result.

        Parameters
        ----------
        query_embedding : np.ndarry
            The embedding of the query text without a batch dimension.
            The shape of this array should be `(dim_embedding,)`.

        Returns
        -------
        all_indices : np.ndarray
            The sorted indices of the most similar embedding vectors. The
            result does not contain any batch dimension and the resulting
            array will have the shape `(size_database,)`.
        all_similarities : np.ndarray
            The sorted similarities (or distances) of the most
            similar embedding vectors. The result does not contain any batch
            dimension and the resulting array will have the shape
            `(size_database,)`.
        """


class FaissSimilarity(BaseSimilarity):

    def __init__(self, index_path=None, index=None):
        self.logger = logging.getLogger(__class__.__name__)

        if index_path is None and index is None:
            raise ValueError("One of the parameters `index_path` and `index` must be not None")
        elif index_path is not None and index is not None:
            raise ValueError("Exactly one of the parameters `index_path` and `index` must be not None")
        elif index_path is not None:
            self.index = faiss.read_index(index_path)
        else:  # index is not None
            self.index = index

        self.logger.info(f"Loaded index with {self.index.ntotal} elements")

    def __call__(self, query_embedding):
        self.logger.info(f"Got a query with {len(query_embedding)} elements.")
        query = query_embedding.copy()
        query = query[None, ...]  # FAISS needs shape = (1, dim_embedding)

        self.logger.info("Normalizing the query vectors")
        faiss.normalize_L2(query)

        self.logger.info("Computing FAISS similarities")
        all_similarities, all_indices = self.index.search(query, k=self.index.ntotal)

        self.logger.info("Done, returning similarities")

        # Depending on the FAISS Index class the similarities
        # could be ascending or descending!

        # The returned arrays have shape (1, index.ntotal)

        return all_indices[0], all_similarities[0]

    @classmethod
    def from_embeddings(cls, normalized_embedding_array):
        logger = logging.getLogger(cls.__name__ + ".from_embeddings")

        logger.info("Constructing FAISS computer from embeddings")
        n_samples, n_dim = normalized_embedding_array.shape

        # Instantiate the index
        logger.info("Instantiating a FAISS index")
        nlist = 10  # TODO: choose this hyper-parameter appropriately
        index_ip = faiss.IndexFlatIP(n_dim)
        index = faiss.IndexIVFFlat(index_ip, n_dim, nlist)

        # Train the index
        logger.info("Training the FAISS index")
        index.train(normalized_embedding_array)

        # Add vectors to the index
        logger.info("Adding embedding vectors to the FAISS index")
        index.add(normalized_embedding_array)

        return cls(index=index)


class TorchSimilarity(BaseSimilarity):

    def __init__(self, embedding_array):
        self.logger = logging.getLogger(__class__.__name__)
        self.embedding_array = torch.from_numpy(embedding_array)

        self.logger.info(f"Initialized with {len(self.embedding_array)} embedding vectors")

    def __call__(self, query_embedding):
        self.logger.info(f"Got a query with {len(query_embedding)} elements.")

        self.logger.info("Adding fake dimensions to query")
        query_embedding = query_embedding[None, ...]

        self.logger.info("Computing cosine similarities")
        all_similarities = nnf.cosine_similarity(
            torch.from_numpy(query_embedding),
            self.embedding_array,
        ).numpy()

        self.logger.info("Sorting the similarities")
        all_indices = torch.argsort(-all_similarities)

        self.logger.info("Re-shuffling the similarities")
        all_similarities = all_similarities[all_indices]

        self.logger.info("Done, returning similarities")
        return all_indices, all_similarities
