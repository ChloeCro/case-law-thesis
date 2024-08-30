import os
import pandas as pd
from bs4 import BeautifulSoup

from utils import constants

class HeaderExtractor:

    def extract_text_from_section(self, section) -> str:
        return ' '.join(section.stripped_strings)

    def extract_section_info(self, soup):
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
                print(f"Skipping file {xml_file}: No valid sections found")
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
        all_judgements = []
        file_counter = 0

        # Loop over all files in the folder
        for filename in os.listdir(folder_path):
            if filename.endswith('.xml'):
                file_path = os.path.join(folder_path, filename)
                file_counter += 1
                print(f"Processing file {file_counter}: {filename}")
                judgement_data = self.process_xml(file_path)
                if judgement_data:
                    all_judgements.append(judgement_data)
                else:
                    print(f"No valid data extracted from file {filename}")

        return all_judgements

    def extract_headers(self):
        # Process the XML files and extract the data
        all_judgements = self.process_xml_files_in_folder(constants.header_data_folder)

        # Convert the extracted data to a DataFrame
        if all_judgements:
            df = pd.DataFrame(all_judgements)
            # Display the DataFrame
            print(df)
            # Optionally save the DataFrame to a CSV file
            df.to_csv(constants.header_data_save_path, index=False)
        else:
            print("No valid judgements found in the XML files.")
