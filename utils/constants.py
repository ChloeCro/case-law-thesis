import os

REPO_PATH = "C:\\Users\\Chloe\\PycharmProjects\\case-law-thesis"

EXTRACTION_LOGGER_NAME = 'Extraction Pipeline'
SEGMENTATION_LOGGER_NAME = 'Segmentation Pipeline'
SUMMARIZATION_LOGGER_NAME = 'Summarization Pipeline'

#########################################
#        Data Processing Paths          #
#########################################

# FOLDER PATHS
DATA_DIR = os.path.join(REPO_PATH, 'data')
METADATA_DIR = os.path.join(DATA_DIR, 'metadata')
RAW_DIR = os.path.join(DATA_DIR, 'raw', '{year}')
DATA_PROCESSING_DIR = os.path.join(REPO_PATH, 'data_processing')

# FILE PATHS
METADATA_PATH = os.path.join(METADATA_DIR, '{year}_rs_data_t.csv')
COMBINED_PATH = os.path.join(METADATA_DIR, 'combined.csv')
SUBSET_PATH = os.path.join(METADATA_DIR, 'subset.csv')
COMBINED_SECTION_PATH = os.path.join(METADATA_DIR, 'combined_section.csv')
SUBSET_SECTION_PATH = os.path.join(METADATA_DIR, 'subset_section.csv')
MERGE_PATTERNS_PATH = os.path.join(DATA_PROCESSING_DIR, 'merge_patterns.txt')
SPLIT_PATTERNS_PATH = os.path.join(DATA_PROCESSING_DIR, 'split_patterns.txt')
SECTION_DATA_PATH = os.path.join(DATA_DIR, 'creation', 'sectioned_data.csv')

HEADER_DATA_INPUT_PATH = "C:\\Users\\Chloe\\Documents\\MaastrichtLaw&Tech\\Summarization\\unzip_data\\2021"
EXTRACTED_DATA_SAVE_PATH = "C:\\Users\\Chloe\\PycharmProjects\\case-law-thesis\\data\\judgements_data_2021.csv"
