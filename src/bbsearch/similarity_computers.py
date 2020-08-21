import abc
import logging
import pathlib
import tempfile

import faiss
import torch
import torch.nn.functional as nnf


class BaseSimilarity(abc.ABC):

    @abc.abstractmethod
    def __call__(self, query_embedding): ...


class FaissSimilarity(BaseSimilarity):

    def __init__(self, index_path):
        self.logger = logging.getLogger(__class__.__name__)
        self.index = faiss.read_index(index_path)

        self.logger.info(f"Loaded index with {self.index.ntotal} elements")

    def __call__(self, query_embedding):
        self.logger.info(f"Got a query with {len(query_embedding)} elements.")
        query = query_embedding.copy()

        self.logger.info("Normalizing the query vectors")
        faiss.normalize_L2(query)

        self.logger.info("Computing FAISS similarities")
        all_similarities, _ = self.index.search(query, self.index.ntotal)

        self.logger.info("Done, returning similarities")
        return all_similarities

    @classmethod
    def from_embeddings(cls, normalized_embedding_array, target_index_file=None):
        logger = logging.getLogger(cls.__name__ + ".from_embeddings")

        logger.info("Constructing FAISS computer from embeddings")
        n_samples, n_dim = normalized_embedding_array.shape

        # Handle the target file name
        if target_index_file is None:
            logger.info("Creating a temporary directory for the index")
            temporary_directory = pathlib.Path(tempfile.mkdtemp())
            target_index_file = temporary_directory / "index.faiss"

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

        # Save index to file
        logger.info("Writing the FAISS index to a file")
        faiss.write_index(index, str(target_index_file))

        return cls(target_index_file)


class TorchSimilarity(BaseSimilarity):

    def __init__(self, embedding_array):
        self.logger = logging.getLogger(__class__.__name__)
        self.embedding_array = embedding_array

        self.logger.info(f"Initialized with {len(self.embedding_array)} embedding vectors")

    def __call__(self, query_embedding):
        self.logger.info(f"Got a query with {len(query_embedding)} elements.")
        all_similarities = nnf.cosine_similarity(
            torch.from_numpy(query_embedding),
            torch.from_numpy(self.embedding_array),
        ).numpy()

        self.logger.info("Done, returning similarities")
        return all_similarities
