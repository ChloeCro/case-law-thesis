import os
import pandas as pd
import numpy as np

from typing import List
from sklearn.cluster import SpectralClustering
from utils import constants, logger_script
from sentence_transformers import SentenceTransformer

logger = logger_script.get_logger(constants.SEGMENTATION_LOGGER_NAME)


class TFSpectralClusterer:
    """
    A class to perform text segmentation using Spectral Clustering with sentence embeddings obtained from a
    Transformer model. The clustering process segments the input documents based on their semantic similarity.

    Attributes:
        cluster_model (SpectralClustering): The spectral clustering model used for segmenting the sentence embeddings
                                            into clusters.

    Methods:
        compute_embeddings(docs, batch_size=32, file_path=constants.EMBEDDINGS_SAVE_PATH, file_name="sent_embeddings",
                           force_compute=False):
            Computes or loads pre-computed sentence embeddings for the input documents using a Transformer model.

        process_sbert_spectral(input_df):
            Processes a DataFrame of input documents to compute sentence embeddings, perform spectral clustering, and
            assign cluster labels to the documents.
    """

    def __init__(self):
        """
        Initializes the TFSpectralClusterer with a Spectral Clustering model configured for clustering sentence
        embeddings into 8 clusters using nearest neighbors affinity.
        """
        self.cluster_model = SpectralClustering(n_clusters=8, affinity='nearest_neighbors', random_state=42)

    @staticmethod
    def compute_embeddings(docs: List[str],
                           batch_size: int = 32,
                           file_path: str = constants.EMBEDDINGS_SAVE_PATH,
                           file_name: str = "sent_embeddings",
                           force_compute: bool = False):
        """
        Computes or loads pre-computed sentence embeddings for a list of documents.

        This method uses a pre-trained SentenceTransformer model to encode the input documents into sentence embeddings.
        If embeddings have already been computed and saved to the specified file, they are loaded from that file.
        Otherwise, the embeddings are computed and saved for future use.

        :param docs: List of documents (strings) to compute embeddings for.
        :param batch_size: The number of documents to process in each batch during embedding computation. Default is 32.
        :param file_path: The directory path where embeddings should be saved or loaded from.
        :param file_name: The file name to use when saving/loading embeddings.
        :param force_compute: If True, forces recomputation of embeddings even if they already exist. Default is False.
        :return: A numpy array containing the computed sentence embeddings.
        """
        file_path = os.path.join(file_path, file_name + ".npz")

        if os.path.exists(file_path) and not force_compute:
            sent_embeddings = np.load(file_path)['embeddings']
            logger.debug("Loaded pre-computed embeddings from {}.".format(file_path))
        else:
            logger.debug("Computing sentence embeddings...")
            embedding_model = SentenceTransformer(constants.EMBEDDER_PATH)
            sent_embeddings = embedding_model.encode(sentences=docs, batch_size=batch_size, show_progress_bar=True,
                                                     normalize_embeddings=True)
            np.savez_compressed(file_path, embeddings=sent_embeddings)
            logger.debug("Sentence embeddings were computed and saved to {}.".format(file_path))

        return sent_embeddings

    def process_sbert_spectral(self, input_df: pd.DataFrame) -> pd.DataFrame:
        """
        Processes input documents for segmentation using sentence embeddings and spectral clustering.

        This method performs the following steps:
        1. Preprocesses the input data.
        2. Computes or loads pre-computed embeddings for the preprocessed data.
        3. Applies spectral clustering to the embeddings to assign cluster labels to each document.
        4. Performs post-processing (if needed) and returns the DataFrame with cluster labels.

        :param input_df: A pandas DataFrame containing the input documents to be segmented.
        :return: A pandas DataFrame with cluster labels assigned to each document.
        """

        # Do some input pre-processing here
        n_grams = ['']

        # Compute/load embeddings for clustering
        embeddings = self.compute_embeddings(n_grams)

        # Create clusters from embeddings and store labels
        labels = self.cluster_model.fit_predict(embeddings)

        # Do some post-processing with clusters here

        pass
