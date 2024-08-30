import os
import sys
import re
import string
import argparse
import multiprocessing
import pandas as pd
import numpy as np

from bs4 import BeautifulSoup
from utils import constants


class SectionExtractor:

    def __init__(self):
        pass



    def process_section_extraction(self):
        # Set path to XML files
        years = [2020, 2021, 2022]
        # years = [2022]

        # year = args.year
        print("start")
        for year in years:
            folder_path = constants.RAW_DIR.format(year=year)
            # folder_path = f'Dataset/Raw/{year}'
            xml_files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith('.xml')]

            # Process the data
            result_lists = process_files_in_parallel(xml_files)
            filtered_results = [result for result in result_lists if result is not None]

            # Create df for metadata
            column_names = ['ecli', 'date', 'inhoudsindicatie', 'instantie',
                            'rechtsgebied', 'wetsverwijzing', 'procesverloop',
                            'overwegingen', 'beslissing']
            df = pd.DataFrame(filtered_results, columns=column_names)
            print(len(df))

            # Further processing or analysis with the DataFrame
            print(df.head())

            metadata_file_path = constants.METADATA_PATH.format(year=year)
            df.to_csv(metadata_file_path, index=False)
            # df.to_csv(f'Dataset/Metadata/{year}_rs_data.csv', index=False)


