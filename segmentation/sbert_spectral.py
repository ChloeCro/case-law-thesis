import pandas as pd
import numpy as np

from typing import List
from sklearn.cluster import SpectralClustering
from utils import constants, logger_script, util_preprocessing
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

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
        self.embedding_model = SentenceTransformer(constants.DUTCH_BERT)
        self.cluster_model = SpectralClustering(n_clusters=8, affinity='nearest_neighbors', random_state=42)

    @staticmethod
    def aggregate_embeddings(embeddings: List[np.ndarray], method: str = 'mean') -> np.ndarray:
        """
        Aggregates a list of embeddings into a single embedding.
        :param embeddings: A list of numpy arrays, each representing a trigram embedding.
        :param method: The method used for aggregation. Default is 'mean'. Other options include 'max'.
        :return: A numpy array representing the aggregated embedding.
        """
        if method == 'mean':
            # Compute the mean of the embeddings if 'mean' method is chosen
            return np.mean(embeddings, axis=0)
        elif method == 'max':
            # Compute the maximum of the embeddings if 'max' method is chosen
            return np.max(embeddings, axis=0)
        else:
            raise ValueError(f"Unknown aggregation method: {method}")

    def process_sbert_spectral(self, input_df: pd.DataFrame) -> pd.DataFrame:
        """
        Processes input documents for segmentation using sentence embeddings and spectral clustering.

        This method performs the following steps:
        1. Creates a subset of the input data based on the proportions of values in the 'instantie' column.
        2. Preprocesses the input data.
        3. Computes embeddings for the preprocessed data.
        4. Applies spectral clustering to the embeddings to assign cluster labels to each document.
        5. Returns the DataFrame with cluster labels.

        :param input_df: A pandas DataFrame containing the input documents to be segmented.
        :return: A pandas DataFrame with cluster labels assigned to each document.
        """
        # Create a subset of the DataFrame based on the proportions of 'instantie'
        subset_df = util_preprocessing.create_subset_based_on_proportions(input_df)

        # Apply tokenization to the full texts and generate tri-grams
        subset_df = util_preprocessing.tokenize_sentences(subset_df, constants.FULLTEXT_COL)
        trigrams = subset_df[constants.TOKENIZED_COL].apply(util_preprocessing.generate_trigrams)

        aggregated_embeddings = []  # List to store the aggregated embeddings for each document

        # Compute embeddings for each trigram and aggregate them
        for trigrams in tqdm(trigrams, desc="Computing aggregated embeddings"):
            trigram_embeddings = [self.embedding_model.encode(' '.join(trigram), normalize_embeddings=True) for trigram
                                  in trigrams]
            # Aggregate embeddings for the trigrams using the specified method
            aggregated_embedding = self.aggregate_embeddings(trigram_embeddings, method='mean')
            aggregated_embeddings.append(aggregated_embedding)

        # Convert the list of aggregated embeddings to a numpy array
        embeddings = np.array(aggregated_embeddings)

        # Apply Spectral Clustering and create clusters from embeddings and store labels
        labels = self.cluster_model.fit_predict(embeddings)
        subset_df[constants.CLUSTER_COL] = labels

        return subset_df
