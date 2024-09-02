import os
import argparse
import pandas as pd
from datetime import datetime

import extractive_summarization, abstractive_summarization
from utils import constants, logger_script, util_data_loader

logger = logger_script.get_logger(constants.SEGMENTATION_LOGGER_NAME)


class SummarizationPipeline:
    """
        A class dedicated to clustering legal document data from Rechtspraak case documents.

    This pipeline applies various data clustering techniques to legal case documents, utilizing different approaches
    such as TF-IDF with K-Means, self-segmentation, and spectral clustering to categorize headers, full texts,
    or specific sections based on the provided method.

    Attributes:
        tfidf_kmeans (TfidfKMeansClusterer): Clusters section headers and section content using TF-IDF and K-Means
                                            clustering. Includes methods for:
                                              - Clustering based on seed words.
                                              - Clustering based on labeled data.
        se3_segmenter (Se3Clusterer): Clusters full text content using a self-segmentation approach.
        transformer_spectral (TFSpectralClusterer): Clusters specific sections of documents using Transformer
                                                    embeddings combined with Spectral Clustering.
        llm_clusterer (LLMClusterer): Placeholder for clustering and classification using a Large Language Model.

    Methods:
        segmentation_process_selector(method: int, input_path: str):
            Selects and applies the appropriate clustering method based on the given method number.
            - method 1: Header Clustering using TF-IDF and K-Means with seed words.
            - method 2: Full Text Clustering using TF-IDF and K-Means with labeled data.
            - method 3: Section Clustering using Se3 self-segmentation. TODO
            - method 4: Section Clustering using S-BERT and Spectral Clustering. TODO
    """

    def __init__(self):
        """
        Initializes the SegmentationPipeline with the specific components for clustering and segmentation methods.
        """
        self.extractive_summarizer = extractive_summarization.ExtractiveSummarizer()
        self.abstractive_summarizer = abstractive_summarization.AbstractiveSummarizer()

    def summarization_process_selector(self, method: int, input_path: str, evaluate: bool):
        """
        Selects and executes a clustering method based on the provided method identifier. Depending on the method
        chosen, the function clusters different types of data, saves the results to a CSV file, and logs the outcome.
        :param method: An integer representing the clustering method to execute. The options are:
                1: Clusters headers using TF-IDF and K-Means with seed words and saves the results to a CSV file.
                2: Clusters full text using TF-IDF and K-Means with labeled data and saves the results to a CSV file.
                3: Clusters sections using Se3 self-segmentation and saves the results to a CSV file.
                4: Clusters sections using S-BERT and Spectral Clustering and saves the results to a CSV file.
                5: Clusters sections using LLM-based classification and saves the results to a CSV file.
        :param input_path: The path to the file to be processed.
        :param evaluate: A bool that indicates whether evaluation scores must be saved.
        :return: The function performs data clustering and saves the results to a CSV file.
        """
        extracted_df = pd.DataFrame()
        method_name = ''

        # Load the input dataframe
        logger.info("Loading input data...")
        df_to_process = util_data_loader.load_csv_to_df(input_path)

        # Determine the processing method and execute the corresponding segmentation technique.
        logger.info("Start segmentation process...")
        match method:
            case 1:
                method_name = 'TextRank'
                extracted_df = self.extractive_summarizer.apply_textrank(df_to_process, evaluate)
            case 2:
                method_name = 'BERT'
                extracted_df = self.extractive_summarizer.apply_bert(df_to_process, evaluate)
            case 3:
                method_name = 'BART'
                extracted_df = self.abstractive_summarizer.apply_bart(df_to_process, evaluate)
            case 4:
                method_name = 'Llama3_1-8B-instruct'
                extracted_df = self.abstractive_summarizer.apply_llama(df_to_process, evaluate)
                pass

        if extracted_df is not None and not extracted_df.empty:
            # Generate the current timestamp
            current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            # Create the filename using the method number and the current time
            filename = f"{method_name}_{current_time}.csv"
            # Combine the folder path and the filename
            file_path = os.path.join(constants.SUMMARIZATION_RESULTS_DIR, filename)
            # Save the DataFrame to the CSV file
            extracted_df.to_csv(file_path, index=False)
            logger.info(f"CSV with {method_name} saved to {file_path}!")
        else:
            error_message = "Summarization failed: the resulting DataFrame is empty or None."
            logger.error(error_message)
            raise ValueError(error_message)


if __name__ == '__main__':
    # create the argument parser, add the arguments
    parser = argparse.ArgumentParser(description='Rechtspraak Summarization Pipeline',
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--method', type=int, choices=range(1, 5), default=1,
                        help=(
                            'Specify clustering method (1-5):\n'
                            '1 = TextRank (extractive summarization),\n'
                            '2 = BERT (extractive summarization),\n'
                            '3 = BART (abstractive summarization),\n'
                            '4 = Llama3.1-8B:instruct (abstractive summarization).'
                        ))
    parser.add_argument('--input', type=str, default=constants.SECTIONS_PATH.format(year=2020),  # TODO: Make input dynamic
                        help="The path to the input data CSV file")
    parser.add_argument('--eval', action='store_true', help="If true, returns the evaluation scores in a CSV file.")

    args = parser.parse_args()

    # Initialize the segmentation pipeline object and run the segmenting process
    logger.info("Start segmentation pipeline...")
    summarization_pipe = SummarizationPipeline()
    summarization_pipe.summarization_process_selector(args.method, args.input, args.eval)

    logger.info("Segmentation pipeline successfully finished!")
