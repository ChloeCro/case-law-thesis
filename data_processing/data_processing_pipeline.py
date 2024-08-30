import argparse
import pandas as pd
import header_extraction, full_text_extraction, section_extraction

from utils import constants, logger

logger = logger.get_logger(constants.EXTRACTION_LOGGER_NAME)


class DataProcessing:
    """
        A class dedicated to extract useful data from Rechtspraak case documents and saves them in a CSV.

        Attributes:
            header_extractor (HeaderExtractor): Extracts header information from documents.
            full_text_extractor (FullTextExtractor): Extracts the full text content from documents.
            section_extractor (SectionExtractor): Extracts specific sections of the documents.

        Methods:
            data_process_selector(method: int):
                Selects and applies the appropriate data processing method based on the given method number.
                - method 1: Header Extraction
                - method 2: Full Text Extraction
                - method 3: Section Extraction
    """

    def __init__(self):
        """
        Initializes the data processing class with specific components for extracting data from the raw XML files.
        """
        self.header_extractor = header_extraction.HeaderExtractor()
        self.full_text_extractor = full_text_extraction.FullTextExtractor()
        self.section_extractor = section_extraction.SectionExtractor()

    def data_process_selector(self, method, input_path):
        """
        Selects and executes a data processing method based on the provided method identifier. Depending on the method
        chosen, the function extracts different types of data, saves the results to a CSV file, and logs the outcome.
        :param method: An integer representing the data processing method to execute. The options are:
            1: Extracts headers and saves them to a CSV file at the specified path.
            2: Extracts full text and saves it to a CSV file at the specified path.
            3: Extracts sections and saves them to a CSV file at the specified path.
        :param input_path: The path to the file to be processed.
        :return: The function performs data extraction and saves the results to a CSV file.
        """
        extracted_df = None

        match method:
            case 1:
                logger.info("Start header extraction process...")
                extracted_df = self.header_extractor.extract_headers(input_path)
            case 2:
                extracted_df = self.full_text_extractor.extract_fulltext(input_path)  # TODO: Implement
            case 3:
                extracted_df = self.section_extractor.extract_sections(input_path)  # TODO: Implement

        if extracted_df is not None and not extracted_df.empty:
            extracted_df.to_csv(constants.EXTRACTED_DATA_SAVE_PATH, index=False)  # TODO: make this filename dynamic
            logger.info(f"CSV with extracted headers saved to {constants.EXTRACTED_DATA_SAVE_PATH}!")
        else:
            error_message = "Data extraction failed: the resulting DataFrame is empty or None."
            logger.error(error_message)
            raise ValueError(error_message)


if __name__ == '__main__':
    # create the argument parser, add the arguments
    parser = argparse.ArgumentParser(description='Rechtspraak Data Processing')
    parser.add_argument('--method', type=int, choices=range(1, 4), default=1,
                        help=(
                            'Specify processing method (1-3): '
                            '1 = Header Extraction: creates a dataframe with a column that holds a dictionary with '
                            '   section header and section text, '
                            '2 = Full Text Extraction: creates a dataframe with a column that contains the document '
                            '   full text (composed from "procesverloop", "overwegingen", and "beslissing", '
                            '3 = Main Section Extraction: creates a dataframe with 3 columns that contains text from '
                            '   "procesverloop", "overwegingen", and "beslissing".'
                        )
                        )
    parser.add_argument('--input', type=str, default=constants.RAW_DIR.format(year=2022),
                        help="The path to the input data CSV file")
    parser.add_argument('--multi', action='store_true', help="Use multiprocessing?")

    args = parser.parse_args()

    # Initialize the data processing object and process data
    logger.info("Start extraction pipeline...")
    data_processor = DataProcessing()
    data_processor.data_process_selector(args.method, args.input)

    logger.info("Extraction pipeline successfully finished!")
