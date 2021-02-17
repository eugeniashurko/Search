"""Entrypoint for launching an embedding server."""

# Blue Brain Search is a text mining toolbox focused on scientific use cases.
#
# Copyright (C) 2020  Blue Brain Project, EPFL.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

import logging
import pathlib
import sys

from ._helper import configure_logging, get_var, run_server


def get_embedding_app():
    """Construct the embedding flask app."""
    from ..embedding_models import BSV, SBioBERT, Sent2VecModel, SentTransformer
    from ..server.embedding_server import EmbeddingServer

    # Read configuration
    log_file = get_var("BBS_EMBEDDING_LOG_FILE", check_not_set=False)
    log_level = get_var("BBS_EMBEDDING_LOG_LEVEL", logging.INFO, var_type=int)

    bsv_checkpoint = get_var("BBS_EMBEDDING_BSV_CHECKPOINT_PATH")
    sent2vec_checkpoint = get_var("BBS_EMBEDDING_SENT2VEC_CHECKPOINT_PATH")

    # Configure logging
    configure_logging(log_file, log_level)
    logger = logging.getLogger(__name__)

    # Load embedding models
    logger.info("Loading embedding models")
    embedding_models = {
        "SBERT": SentTransformer("bert-base-nli-mean-tokens"),
        "BIOBERT NLI+STS": SentTransformer("clagator/biobert_v1.1_pubmed_nli_sts"),
        "Sent2Vec": Sent2VecModel(pathlib.Path(sent2vec_checkpoint)),
        "BSV": BSV(pathlib.Path(bsv_checkpoint)),
        "SBioBERT": SBioBERT(),
    }

    # Create Server app
    logger.info("Creating the server app")
    embedding_app = EmbeddingServer(embedding_models)

    return embedding_app


def run_embedding_server():
    """Run the embedding server."""
    run_server(get_embedding_app, "embedding")


if __name__ == "__main__":  # pragma: no cover
    sys.exit(run_embedding_server())