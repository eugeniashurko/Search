"""Collection of functions focused on searching."""
import logging

import faiss
import numpy as np

from .similarity_computers import FaissSimilarity, TorchSimilarity
from .sql import SentenceFilter
from .utils import Timer

logger = logging.getLogger(__name__)


class LocalSearcher:
    """Search locally using assets on disk.

    This class requires for several deep-learning modules
    to be loaded and for pre-trained models, pre-computed
    embeddings, and the SQL database to be loaded in memory.

    This is more or less a wrapper around `run_search`
    from `bbsearch.search`.

    Parameters
    ----------
    embedding_models : dict
        The pre-trained models.
    precomputed_embeddings : dict
        The pre-computed embeddings.
    connection : SQLAlchemy connectable (engine/connection) or database str URI or DBAPI2 connection (fallback mode)
        The database connection.
    """

    def __init__(self, embedding_models, precomputed_embeddings, connection):
        self.embedding_models = embedding_models
        self.precomputed_embeddings = precomputed_embeddings
        self.connection = connection
        # self.similarity_computers = {
        #     model_name: TorchSimilarity(embeddings)
        #     for model_name, embeddings in precomputed_embeddings.items()
        # }

        self.similarity_computers = dict()
        for model_name, embeddings in precomputed_embeddings.items():
            faiss.normalize_L2(embeddings)
            self.similarity_computers[model_name] = FaissSimilarity.from_embeddings(embeddings)

    def query(self,
              which_model,
              k,
              query_text,
              has_journal=False,
              date_range=None,
              deprioritize_strength='None',
              exclusion_text="",
              deprioritize_text=None,
              verbose=True,
              ):
        """Do the search.

        Parameters
        ----------
        which_model : str
            The name of the model to use.
        k : int
            Number of top results to display.
        query_text : str
            Query.
        has_journal : bool
            If True, only consider papers that have a journal information.
        date_range : tuple
            Tuple of form (start_year, end_year) representing the considered
            time range.
        deprioritize_text : str
            Text query of text to be deprioritized.
        deprioritize_strength : str, {'None', 'Weak', 'Mild', 'Strong', 'Stronger'}
            How strong the deprioritization is.
        exclusion_text : str
            New line separated collection of strings that are automatically
            used to exclude a given sentence.
        verbose : bool
            If True, then printing statistics to standard output.

        Returns
        -------
        results : tuple
            All results returned by `run_search`.
        """
        results = run_search(
            self.embedding_models[which_model],
            self.similarity_computers[which_model],
            self.connection,
            k,
            query_text,
            has_journal,
            date_range,
            deprioritize_strength,
            exclusion_text,
            deprioritize_text,
            verbose)

        return results


def run_search(
        embedding_model,
        similarity_computer,
        connection,
        k,
        query_text,
        has_journal=False,
        date_range=None,
        deprioritize_strength='None',
        exclusion_text="",
        deprioritize_text=None,
        verbose=True
):
    """Generate search results.

    Parameters
    ----------
    embedding_model : bbsearch.embedding_models.EmbeddingModel
        Instance of EmbeddingModel of the model we want to use.

    similarity_computer : bbsearch.similarity_computers.BaseSimilarity
        2D array containing embeddings of the model corresponding of embedding_model. Rows are
        sentences and columns are different dimensions.

    connection : SQLAlchemy connectable (engine/connection) or database str URI or DBAPI2 connection (fallback mode)
        Connection to the database.

    k : int
        Number of top results to display.

    query_text : str
        Query.

    has_journal : bool
        If True, only consider papers that have a journal information.

    date_range : tuple
        Tuple of form (start_year, end_year) representing the considered time range.

    deprioritize_text : str
        Text query of text to be deprioritized.

    deprioritize_strength : str, {'None', 'Weak', 'Mild', 'Strong', 'Stronger'}
        How strong the deprioritization is.

    exclusion_text : str
        New line separated collection of strings that are automatically used to exclude a given sentence.

    verbose : bool
        If True, then printing statistics to standard output.

    Returns
    -------
    sentence_ids : np.array
        1D array representing the indices of the top `k` most relevant sentences.

    similarities : np.array
        1D array reresenting the similarities for each of the top `k` sentences. Note that this will
        include the deprioritization part.

    stats : dict
        Various statistics. There are following keys:
        - 'query_embed_time' - how much time it took to embed the `query_text` in seconds
        - 'deprioritize_embed_time' - how much time it took to embed the `deprioritize_text` in seconds
        -
    """
    logger.info("Starting run_search")

    # Replace empty `deprioritize_text` by None
    if deprioritize_text is not None and len(deprioritize_text.strip()) == 0:
        deprioritize_text = None

    timer = Timer(verbose=verbose)

    with timer('query_embed'):
        logger.info("Embedding the query text")
        preprocessed_query_text = embedding_model.preprocess(query_text)
        embedding_query = embedding_model.embed(preprocessed_query_text)

    if deprioritize_text is not None:
        with timer('deprioritize_embed'):
            logger.info("Embedding the deprioritization text")
            preprocessed_deprioritize_text = embedding_model.preprocess(deprioritize_text)
            embedding_deprioritize = embedding_model.embed(preprocessed_deprioritize_text)

    with timer('sentences_filtering'):
        logger.info("Applying sentence filtering")
        restricted_sentence_ids = (
            SentenceFilter(connection)
            .only_with_journal(has_journal)
            .date_range(date_range)
            .exclude_strings(exclusion_text.split())
            .run()
        )

        if len(restricted_sentence_ids) == 0:
            logger.info("No indices left after sentence filtering. Returning.")
            return np.array([]), np.array([]), timer.stats

    # Compute similarities
    deprioritizations = {
        'None': (1, 0),
        'Weak': (0.9, 0.1),
        'Mild': (0.8, 0.3),
        'Strong': (0.5, 0.5),
        'Stronger': (0.5, 0.7),
    }
    # now: maximize L = a1 * cos(x, query) - a2 * cos(x, exclusions)
    alpha_1, alpha_2 = deprioritizations[deprioritize_strength]
    logger.info("Combining query and deprioritizations")
    with timer('query_similarity'):
        logger.info("Computing cosine similarities")
        if deprioritize_text is None:
            effective_query_embedding = embedding_query
        else:
            effective_query_embedding = alpha_1 * embedding_query - alpha_2 * embedding_deprioritize

        top_indices, similarities = similarity_computer(effective_query_embedding)
        top_sentence_ids = top_indices + 1

    # [8311, 8279, 8317, 8355, 8285]
    mask = np.isin(top_sentence_ids, restricted_sentence_ids)
    top_sentence_ids_filtered = top_sentence_ids[mask]
    similarities_filtered = similarities[mask]

    logger.info("run_search finished")

    return top_sentence_ids_filtered[:k], similarities_filtered[:k], timer.stats
