import argparse
import pandas as pd

import tfidf_kmeans, se3, sbert_spectral, llm
from utils import constants, logger, util_functions

logger = logger.get_logger(constants.SEGMENTATION_LOGGER_NAME)


class SegmentationPipeline:
    """
    A class dedicated to extracting useful data from Rechtspraak case documents and saving them in a CSV file.

    This pipeline applies various data processing techniques to legal case documents, utilizing different clustering
    and segmentation methods to extract headers, full text, or specific sections based on the provided method.

    Attributes:
        tfidf_kmeans (TfidfKMeansClusterer): Extracts header information and full text content from documents using
                                                TF-IDF and K-Means clustering.
        se3_segmenter (Se3Clusterer): Extracts the full text content from documents using a self-segmentation approach.
        transformer_spectral (TFSpectralClusterer): Extracts specific sections of the documents using Transformer
                                                    embeddings combined with Spectral Clustering.
        llm_clusterer (LLMClusterer): A placeholder for data extraction and classification using a Large Language Model.

    Methods:
        segmentation_process_selector(method: int, input_path: str):
            Selects and applies the appropriate data processing method based on the given method number.
            - method 1: Header Extraction using TF-IDF and K-Means with seed words.
            - method 2: Full Text Extraction using TF-IDF and K-Means with labeled data.
            - method 3: Section Extraction using Se3 self-segmentation.
            - method 4: Section Extraction using S-BERT and Spectral Clustering.
            - method 5: Section Extraction using LLM-based classification.
    """

    def __init__(self):
        """
        Initializes the SegmentationPipeline with the specific components for clustering and segmentation methods.
        """
        self.tfidf_kmeans = tfidf_kmeans.TfidfKMeansClusterer()
        self.se3_segmenter = se3.Se3Clusterer()
        self.transformer_spectral = sbert_spectral.TFSpectralClusterer()
        self.llm_clusterer = llm.LLMClusterer()

    def segmentation_process_selector(self, method: int, input_path: str):
        """
        Selects and executes a data processing method based on the provided method identifier. Depending on the method
        chosen, the function extracts different types of data, saves the results to a CSV file, and logs the outcome.
        :param method: An integer representing the data processing method to execute. The options are:
                1: Extracts headers using TF-IDF and K-Means with seed words and saves them to a CSV file.
                2: Extracts full text using TF-IDF and K-Means with labeled data and saves it to a CSV file.
                3: Extracts sections using Se3 self-segmentation and saves them to a CSV file.
                4: Extracts sections using S-BERT and Spectral Clustering and saves them to a CSV file.
                5: Extracts sections using LLM-based classification and saves them to a CSV file.
        :param input_path: The path to the file to be processed.
        :return: The function performs data extraction and saves the results to a CSV file.
        """
        extracted_df = pd.DataFrame()
        method_name = ''

        # Load the input dataframe
        df_to_process = util_functions.load_csv_to_df(input_path)

        logger.info("Start segmentation process...")
        match method:
            case 1:
                method_name = 'Tf-idf and K-means clusters using seed words'
                extracted_df = self.tfidf_kmeans.guided_kmeans_with_seed_words(df_to_process)
            case 2:
                method_name = 'Tf-idf and K-means clusters using labeled data'
                labeled_df = util_functions.load_csv_to_df(constants.LABELED_HEADER_PATH)
                extracted_df = self.tfidf_kmeans.guided_kmeans_with_labeled(df_to_process, labeled_df)
            case 3:
                method_name = 'Se3 self-segmentation clusters'
                extracted_df = self.se3_segmenter.extract_sections(input_path)  # TODO: Implement
            case 4:
                method_name = 'S-BERT and Spectral Clustering clusters'
                extracted_df = self.transformer_spectral.extract_sections(input_path)  # TODO: Implement
                pass
            case 5:
                method_name = 'LLM classification'
                extracted_df = self.llm_clusterer.extract_sections(input_path)  # TODO: Implement
                pass

        if extracted_df is not None and not extracted_df.empty:
            extracted_df.to_csv(constants.EXTRACTED_DATA_SAVE_PATH, index=False)  # TODO: make this filename dynamic
            logger.info(f"CSV with {method_name} saved to {constants.EXTRACTED_DATA_SAVE_PATH}!")
        else:
            error_message = "Data extraction failed: the resulting DataFrame is empty or None."
            logger.error(error_message)
            raise ValueError(error_message)


if __name__ == '__main__':
    # create the argument parser, add the arguments
    parser = argparse.ArgumentParser(description='Rechtspraak Segmentation Pipeline',
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--method', type=int, choices=range(1, 6), default=1,  # TODO: update this help string
                        help=(
                            'Specify processing method (1-5):\n'
                            '1 = TF-IDF + K-MEANS with seed words: creates a dataframe with a column that holds a '
                            'dictionary with section header and section text,\n'
                            '2 = TF-IDF + K-MEANS with labels: ,\n'
                            '3 = Self-Segmentation (Se3): ,\n'
                            '4 = S-BERT + Spectral Clustering: ,\n'
                            '5 = LLM: .'
                        )
                        )
    parser.add_argument('--input', type=str, default=constants.RAW_DIR.format(year=2022),
                        help="The path to the input data CSV file")

    args = parser.parse_args()

    # Initialize the segmentation pipeline object and run the segmenting process
    logger.info("Start extraction pipeline...")
    segmentation_pipe = SegmentationPipeline()
    segmentation_pipe.segmentation_process_selector(args.method, args.input)

    logger.info("Extraction pipeline successfully finished!")
