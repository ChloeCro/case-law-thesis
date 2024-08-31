import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA
from typing import Tuple

from utils import constants, logger_script

logger = logger.get_logger(constants.SEGMENTATION_LOGGER_NAME)


class TfidfKMeansClusterer:
    """
    A class that combines TF-IDF vectorization with K-Means clustering to segment legal document headers.

    This class implements a semi-supervised approach to cluster headers from legal documents into predefined categories
    using TF-IDF features and K-Means clustering. The approach leverages seed embeddings based on legal topic keywords
    to guide the clustering process towards meaningful groupings.

    Attributes:
        vectorizer (TfidfVectorizer): A TF-IDF vectorizer used to transform text data into numerical vectors.
        stop_words (List[str]): A list of stopwords used by the vectorizer to ignore common words.
    """

    def __init__(self):
        """
        Initializes the TfidfKMeansClusterer with Dutch stopwords and additional custom stopwords.
        Sets up the TF-IDF vectorizer with these stopwords.
        """
        self.stop_words = stopwords.words('dutch') + constants.ADDITIONAL_STOPWORDS
        self.vectorizer = TfidfVectorizer(stop_words=self.stop_words)

    @staticmethod
    def extract_headers_for_tfidf(df: pd.DataFrame) -> list:
        """
        Extracts 'header' values from the 'sections' nested dictionaries within the DataFrame.
        :param df: Input DataFrame containing a 'sections' column with nested dictionaries.
        :return: A list of extracted headers in lowercase.
        """
        header_values = []
        for nested_dict in df['sections']:
            for key, value in nested_dict.items():
                if key.isdigit():  # Check if the key is a number
                    header_values.append(value['header'].lower())
        return header_values

    @staticmethod
    def apply_kmeans(init_value: np.ndarray[int], tfidf_matrix: np.ndarray[int]) -> np.ndarray[int]:
        """
        Applies K-Means clustering to the TF-IDF matrix using the provided seed matrix for initialization.
        :param init_value: The seed matrix used to initialize the K-Means centroids.
        :param tfidf_matrix: The TF-IDF matrix representing the text data.
        :return: An array of cluster labels assigned to each document in the TF-IDF matrix.
        """
        kmeans = KMeans(n_clusters=len(constants.SEED_WORDS_LIST), init=init_value, n_init=1, random_state=42)
        kmeans.fit(tfidf_matrix)
        cluster_labels = kmeans.predict(tfidf_matrix)
        return cluster_labels

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

    @staticmethod
    def generate_tfidf_kmeans_scatter_plot(tfidf_matrix: np.array, cluster_labels: list):
        """
        Generates a 2D scatter plot of the TF-IDF matrix data points, colored by their cluster labels.
        :param tfidf_matrix: The TF-IDF matrix representing the text data.
        :param cluster_labels: An array of cluster labels assigned to each document.
        :return: None. Shows the plot and saves it as PNG file.
        """
        # Plotting the new data points and centroids after clustering
        logger.info("Plotting data...")

        # Reduce dimensions to 2D using PCA for visualization
        pca = PCA(n_components=2)
        new_data_2d = pca.fit_transform(tfidf_matrix.toarray())

        # Configure the plot settings and show it
        plt.figure(figsize=(10, 6))
        plt.scatter(new_data_2d[:, 0], new_data_2d[:, 1], c=cluster_labels, cmap='viridis', marker='o',
                    label='New Data')
        plt.title("New Data Clustering Based on Refined TF-IDF Vectors")
        plt.legend()
        plt.show()  # TODO: add save statement

    def guided_kmeans_with_seed_words(self, input_df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies K-Means clustering to the headers in the input DataFrame using seed words to guide the clustering.
        :param input_df: The input DataFrame containing the 'sections' column with nested dictionaries.
        :return: A DataFrame with headers and their corresponding cluster labels.
        """
        # Extract 'header' values from the nested dictionaries
        header_values = self.extract_headers_for_tfidf(input_df)

        # Fit the vectorizer to the headers and transform the headers
        tfidf_matrix = self.vectorizer.fit_transform(header_values)

        # Vectorize the seed words
        seed_vectors = []
        for seed_words in constants.SEED_WORDS_LIST:
            seed_vector = self.vectorizer.transform(seed_words).mean(axis=0)
            seed_vectors.append(seed_vector)

        seed_matrix = np.array(seed_vectors)
        seed_matrix = np.squeeze(seed_matrix)

        # Apply K-Means clustering
        cluster_labels = self.apply_kmeans(seed_matrix, tfidf_matrix)

        # Create a DataFrame with headers and their corresponding cluster labels
        result_df = pd.DataFrame({
            'Header': header_values,
            'Cluster': cluster_labels
        })

        # Evaluate the quality of the clustering
        silhouette, db_index = self.evaluate_clusters(tfidf_matrix, cluster_labels)
        logger.info(f"Silhouette Score: {silhouette:.4f}")
        logger.info(f"Davies-Bouldin Index: {db_index:.4f}")

        return result_df

    def guided_kmeans_with_labeled(self, input_df: pd.DataFrame, labeled_df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies K-Means clustering to the headers in the input DataFrame using refined TF-IDF vectors
        derived from labeled data.
        :param input_df: The input DataFrame containing the 'sections' column with nested dictionaries.
        :param labeled_df: The labeled DataFrame containing 'Header' and 'Cluster' columns.
        :return: A DataFrame with headers and their predicted cluster labels.
        """
        # Extract 'header' and 'cluster' values from the labeled DataFrame
        labeled_headers = labeled_df['Header'].str.lower().tolist()
        labeled_clusters = labeled_df['Cluster'].tolist()

        # Fit the vectorizer on the entire labeled corpus
        self.vectorizer.fit(labeled_headers)

        # Initialize empty dictionary to store refined TF-IDF vectors for each cluster
        refined_tfidf_vectors = {}
        unique_clusters = sorted(set(labeled_clusters))

        # Process each cluster separately
        for cluster in unique_clusters:
            # Get headers for this cluster
            cluster_indices = [i for i, label in enumerate(labeled_clusters) if label == cluster]
            cluster_headers = [labeled_headers[i] for i in cluster_indices]

            # Get headers for the remaining data (i.e., all other clusters)
            remaining_indices = [i for i, label in enumerate(labeled_clusters) if label != cluster]
            remaining_headers = [labeled_headers[i] for i in remaining_indices]

            # Compute TF-IDF vectors for this cluster and remaining data using the unified vocabulary
            cluster_tfidf_matrix = self.vectorizer.transform(cluster_headers)
            remaining_tfidf_matrix = self.vectorizer.transform(remaining_headers)

            # Subtract the remaining TF-IDF vectors from the cluster-specific TF-IDF vectors
            refined_tfidf_matrix = cluster_tfidf_matrix.mean(axis=0) - remaining_tfidf_matrix.mean(axis=0)
            refined_tfidf_vectors[cluster] = refined_tfidf_matrix

        # Extract 'header' values from the nested dictionaries
        header_values = self.extract_headers_for_tfidf(input_df)

        # Compute the TF-IDF vectors for the extracted headers
        tfidf_matrix = self.vectorizer.transform(header_values)

        # Calculate the distance from each header to each cluster's refined vector
        distance_matrix = np.zeros((len(header_values), len(refined_tfidf_vectors)))
        for i, refined_vector in refined_tfidf_vectors.items():
            refined_vector_dense = np.squeeze(np.asarray(refined_vector))
            distance_matrix[:, i] = tfidf_matrix.dot(refined_vector_dense.T)

        # Assign each header to the closest cluster based on the distance matrix
        cluster_labels = np.argmax(distance_matrix, axis=1)

        # Create a DataFrame with headers and their predicted cluster labels
        result_df = pd.DataFrame({'Header': header_values, 'Cluster': cluster_labels})

        # Evaluate the quality of the clustering
        self.evaluate_clusters(tfidf_matrix, cluster_labels)

        return result_df
