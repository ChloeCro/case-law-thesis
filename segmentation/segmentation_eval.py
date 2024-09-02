import numpy as np

from sklearn.metrics import silhouette_score, davies_bouldin_score
from typing import Tuple
from utils import constants, logger_script

logger = logger_script.get_logger(constants.SEGMENTATION_LOGGER_NAME)


class SegmentationEvaluator:

    def __init__(self):
        pass

    @staticmethod
    def evaluate_clusters(tfidf_matrix: np.array, cluster_labels: np.ndarray[int]) -> Tuple[float, float]:
        """
        Evaluates the quality of the clusters using Silhouette Score and Davies-Bouldin Index.
        :param tfidf_matrix: The TF-IDF matrix representing the text data.
        :param cluster_labels: An array of cluster labels assigned to each document.
        :return: None. Writes the scores to the logger.
        """
        silhouette_avg = silhouette_score(tfidf_matrix, cluster_labels)
        davies_bouldin = davies_bouldin_score(tfidf_matrix.toarray(), cluster_labels)
        return silhouette_avg, davies_bouldin
