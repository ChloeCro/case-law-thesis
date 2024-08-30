import os
import pandas as pd
from bs4 import BeautifulSoup

from utils import constants, logger

logger = logger.get_logger(constants.EXTRACTION_LOGGER_NAME)


class HeaderExtractor:

    def extract_text_from_section(self, section) -> str:
        """
        Extracts and concatenates all stripped strings from a given HTML/XML section.
        :param section: A BeautifulSoup tag object representing a section.
        :return: The concatenated string of all text elements within the section.
        """
        return ' '.join(section.stripped_strings)

    def extract_section_info(self, soup):
        """
        Extracts information from each section in the provided BeautifulSoup object, including section number, header
        text, and the full text of the section.
        :param soup: A BeautifulSoup object representing the parsed XML/HTML document.
        :return: A dictionary where each key is a section number and the value is another dictionary containing
              the 'header' (title of the section) and 'text' (full text of the section).
        """
        section_data = {}
        sections = soup.find_all("section")
        for sec in sections:
            title_tag = sec.find("title")
            if title_tag:
                nr_tag = title_tag.find("nr")
                if nr_tag and nr_tag.text:
                    section_number = nr_tag.text
                    header_text = ''.join(title_tag.stripped_strings).replace(nr_tag.text, '').strip()
                    section_text = self.extract_text_from_section(sec)
                    section_data[section_number] = {'header': header_text, 'text': section_text}
        return section_data

    def process_xml(self, xml_file):
        """
        Processes a single XML file to extract relevant legal judgement information, including global attributes
        and section-specific data.
        :param xml_file: The path to the XML file to be processed.
        :return: A dictionary containing extracted information such as 'ecli', 'date', 'inhoud', 'legal_body',
              'rechtsgebied', 'wetsverwijzing', and 'sections'. Returns None if no valid sections are found.
        """
        with open(xml_file, 'r', encoding='utf-8') as file:
            soup = BeautifulSoup(file, 'xml')

            # Initialize variables
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

            # Extract section information
            section_data = self.extract_section_info(soup)

            # Skip file if no section data is found
            if not section_data:
                logger.debug(f"Skipping file {xml_file}: No valid sections found")
                return None

            # Compile all extracted information into a dictionary
            judgement_data = {
                'ecli': ecli,
                'date': date,
                'inhoud': inhoud,
                'legal_body': legal_body,
                'rechtsgebied': rechtsgebied,
                'wetsverwijzing': wetsverwijzing,
                'sections': section_data
            }

            return judgement_data

    def process_xml_files_in_folder(self, folder_path):
        """
        Processes all XML files within a specified folder, extracting legal judgement information from each file.
        :param folder_path: The path to the folder containing XML files.
        :return: A list of dictionaries where each dictionary contains the extracted judgement data from an XML file.
        """
        all_judgements = []
        file_counter = 0

        # Loop over all files in the folder
        for filename in os.listdir(folder_path):
            if filename.endswith('.xml'):
                file_path = os.path.join(folder_path, filename)
                file_counter += 1
                logger.info(f"Processing file {file_counter}: {filename}")
                judgement_data = self.process_xml(file_path)
                if judgement_data:
                    all_judgements.append(judgement_data)
                else:
                    logger.debug(f"No valid data extracted from file {filename}")

        return all_judgements

    def extract_headers(self):
        """
        Extracts legal judgement headers from XML files in a specified folder and converts the extracted data into a
        DataFrame.
        :return: Saves the resulting DataFrame  to a CSV file specified in the constants.
        """
        # Process the XML files and extract the data
        all_judgements = self.process_xml_files_in_folder(constants.HEADER_DATA_INPUT_PATH)

        # Convert the extracted data to a DataFrame and save the DataFrame to a CSV file.
        if all_judgements:
            df = pd.DataFrame(all_judgements)
            df.to_csv(constants.HEADER_DATA_SAVE_PATH, index=False)
        else:
            logger.info("No valid judgements found in the XML files.")
