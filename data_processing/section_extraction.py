import os
import multiprocessing
import pandas as pd

from typing import List
from bs4 import BeautifulSoup
from utils import constants, logger

logger = logger.get_logger(constants.EXTRACTION_LOGGER_NAME)


class SectionExtractor:
    """
    A class dedicated to extracting and processing specific sections of legal judgement documents.

    Methods:
        is_valid_tag(tag: bs4.element.Tag) -> bool:
            Checks whether a given tag is valid for processing (i.e., not a 'title' tag).

        extract_text_from_section(section: bs4.element.Tag) -> str:
            Extracts and concatenates all text content from a given HTML/XML section, removing all tags.

        process_xml(xml_file: str) -> list or None:
            Processes a single XML file to extract relevant legal judgement information, including global attributes
            and section-specific data, and returns it as a list.

        process_files_in_parallel(files: list) -> list:
            Processes multiple XML files in parallel, utilizing all available CPU cores.

        process_section_extraction() -> None:
            Processes XML files, extracts relevant sections, and saves the extracted data to CSV files.
    """
    @staticmethod
    def is_valid_tag(tag) -> bool:
        """
        Checks whether a given tag is valid for processing (i.e., not a 'title' tag).

        :param tag: A BeautifulSoup Tag object representing an element in the document.
        :return: True if the tag is not a 'title' tag, otherwise False.
        """
        return tag.name != 'title'

    @staticmethod
    def extract_text_from_section(section) -> str:
        """
        Extracts and concatenates all text content from a given HTML/XML section, removing all tags.

        This method converts the contents of the section into a string, then uses BeautifulSoup to parse and remove any
        remaining HTML/XML tags, leaving only the plain text.

        :param section: A BeautifulSoup Tag object representing a section of an HTML or XML document.
        :return: A single string containing the text content of the section, with all tags stripped.
        """
        # Join all content within the section as a single string
        section_text = ''.join(str(child) for child in section.contents)
        # Create a new BeautifulSoup object to parse the content and strip tags
        clean_text = BeautifulSoup(section_text, 'html.parser').get_text()
        return clean_text

    def process_xml(self, xml_file: str) -> list or None:
        """
        Processes a single XML file to extract relevant legal judgement information, including global attributes
        and section-specific data.

        This method parses the XML file to extract global information like ECLI, date, and legal references, as well
        as specific sections such as 'procesverloop', 'overwegingen', and 'beslissing'. The extracted data is compiled
        into a list for further processing or storage.

        :param xml_file: The path to the XML file to be processed.
        :return: A list containing the extracted data or None if critical sections are missing.
        """
        # Open the XML file and parse it using BeautifulSoup with the 'xml' parser
        with open(xml_file, 'r', encoding='utf-8') as file:
            soup = BeautifulSoup(file, 'xml')

            # Initialize variables for section text and global information
            procesverloop_text, overwegingen_text, beslissing_text = '', '', ''
            ecli, date, inhoud, legal_body, rechtsgebied, wetsverwijzing = '', '', '', '', '', ''

            # Extract global information from the XML tags
            ecli_tag = soup.find("dcterms:identifier")
            date_tag = soup.find("dcterms:date", {"rdfs:label": "Uitspraakdatum"})
            inhoud_tag = soup.find("inhoudsindicatie")
            legal_body_tag = soup.find("dcterms:creator", {"rdfs:label": "Instantie"})
            rechtsgebied_tag = soup.find("dcterms:subject", {"rdfs:label": "Rechtsgebied"})
            wetsverwijzing_tag = soup.find("dcterms:references", {"rdfs:label": "Wetsverwijzing"})

            # Assign text from the tags to the variables if the tags exist
            if ecli_tag:
                ecli = ecli_tag.text
            if date_tag:
                date = date_tag.text
            if inhoud_tag:
                inhoud = inhoud_tag.text
            if legal_body_tag:
                legal_body = legal_body_tag.text
            if rechtsgebied_tag:
                rechtsgebied = rechtsgebied_tag.text
            if wetsverwijzing_tag:
                wetsverwijzing = wetsverwijzing_tag.text

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

    def process_files_in_parallel(self, files: List[str]) -> List[list]:
        """
        Processes multiple XML files in parallel, utilizing all available CPU cores.

        This method leverages the multiprocessing module to speed up the processing of a large number of XML files
        by distributing the work across multiple CPU cores.

        :param files: A list of file paths to the XML files to be processed.
        :return: A list of lists, where each inner list contains the extracted data from one XML file.
        """
        # Determine the number of available CPU cores and create a multiprocessing pool
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
        """
        Processes XML files for several years, extracts relevant sections, and saves the extracted data to CSV files.

        This method iterates over specified years, processes the XML files for each year in parallel, and compiles
        the extracted data into a pandas DataFrame. The DataFrame is then saved to a CSV file for each year.

        :return: None
        """
        # TODO: Change this to single input year!!!
        # Set path to XML files
        years = [2020, 2021, 2022]
        for year in years:
            # Construct the folder path for the current year
            folder_path = constants.RAW_DIR.format(year=year)
            # folder_path = f'Dataset/Raw/{year}'
            # Get the list of all XML files in the folder
            xml_files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith('.xml')]

            # Process the data in parallel
            result_lists = self.process_files_in_parallel(xml_files)
            filtered_results = [result for result in result_lists if result is not None]

            # Create df for metadata
            column_names = ['ecli', 'date', 'inhoudsindicatie', 'instantie',
                            'rechtsgebied', 'wetsverwijzing', 'procesverloop',
                            'overwegingen', 'beslissing']
            df = pd.DataFrame(filtered_results, columns=column_names)

            return df
