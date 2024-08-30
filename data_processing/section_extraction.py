import os
import multiprocessing
import pandas as pd

from bs4 import BeautifulSoup
from utils import constants, logger

logger = logger.get_logger(constants.EXTRACTION_LOGGER_NAME)


class SectionExtractor:

    def is_valid_tag(self, tag):
        return tag.name != 'title'

    # Function to extract text without XML tags
    def extract_text_from_section(self, section):
        # Join all content within the section as a single string
        section_text = ''.join(str(child) for child in section.contents)
        # Create a new BeautifulSoup object to parse the content and strip tags
        clean_text = BeautifulSoup(section_text, 'html.parser').get_text()
        return clean_text

    def process_xml(self, xml_file):
        with open(xml_file, 'r', encoding='utf-8') as file:
            soup = BeautifulSoup(file, 'xml')

            # Initialize variables
            procesverloop_text, overwegingen_text, beslissing_text = '', '', ''
            ecli, date, inhoud, legal_body, rechtsgebied, wetsverwijzing = '', '', '', '', '', ''

            # Extract global information
            ecli_tag = soup.find("dcterms:identifier")
            date_tag = soup.find("dcterms:date", {"rdfs:label": "Uitspraakdatum"})
            inhoud_tag = soup.find("inhoudsindicatie")
            legal_body_tag = soup.find("dcterms:creator", {"rdfs:label": "Instantie"})
            rechtsgebied_tag = soup.find("dcterms:subject", {"rdfs:label": "Rechtsgebied"})
            wetsverwijzing_tag = soup.find("dcterms:references", {"rdfs:label": "Wetsverwijzing"})

            if ecli_tag: ecli = ecli_tag.text
            if date_tag: date = date_tag.text
            if inhoud_tag: inhoud = inhoud_tag.text
            if legal_body_tag: legal_body = legal_body_tag.text
            if rechtsgebied_tag: rechtsgebied = rechtsgebied_tag.text
            if wetsverwijzing_tag: wetsverwijzing = wetsverwijzing_tag.text

            # Process each section and convert to pure text
            sections = soup.find_all("section")
            for sec in sections:
                role = sec.get('role')

                # Use a common method to extract plain text
                section_text = self.extract_text_from_section(sec)

                # Append text based on the role of the section
                if role == 'procesverloop':
                    procesverloop_text += (' ' + section_text if procesverloop_text else section_text)
                elif role == 'beslissing':
                    beslissing_text += (' ' + section_text if beslissing_text else section_text)
                else:  # This includes 'overwegingen' or sections without a role
                    overwegingen_text += (' ' + section_text if overwegingen_text else section_text)

            # Check if critical sections are present
            if not procesverloop_text or not beslissing_text:
                return None  # Skip file if critical sections are missing

            # Compile all extracted information into a list
            judgement_list = [ecli, date, inhoud, legal_body, rechtsgebied, wetsverwijzing, procesverloop_text,
                              overwegingen_text, beslissing_text]
            return judgement_list

    def process_files_in_parallel(self, files):
        num_processes = multiprocessing.cpu_count()
        pool = multiprocessing.Pool(processes=num_processes)

        # Use the multiprocessing pool to process the XML files in parallel
        logger.info("Start multiprocessing...")
        result_lists = pool.map(self.process_xml, files)

        # Close the pool and wait for the worker processes to finish
        pool.close()
        pool.join()

        return result_lists

    def process_section_extraction(self):
        # Set path to XML files
        years = [2020, 2021, 2022]
        for year in years:
            folder_path = constants.RAW_DIR.format(year=year)
            # folder_path = f'Dataset/Raw/{year}'
            xml_files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith('.xml')]

            # Process the data
            result_lists = self.process_files_in_parallel(xml_files)
            filtered_results = [result for result in result_lists if result is not None]

            # Create df for metadata
            column_names = ['ecli', 'date', 'inhoudsindicatie', 'instantie',
                            'rechtsgebied', 'wetsverwijzing', 'procesverloop',
                            'overwegingen', 'beslissing']
            df = pd.DataFrame(filtered_results, columns=column_names)

            metadata_file_path = constants.METADATA_PATH.format(year=year)
            df.to_csv(metadata_file_path, index=False)
            # df.to_csv(f'Dataset/Metadata/{year}_rs_data.csv', index=False)


