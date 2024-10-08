import pandas as pd
import difflib
import ollama

from utils import constants, logger_script
from typing import Tuple, List

logger = logger_script.get_logger(constants.SEGMENTATION_LOGGER_NAME)


class LLMClusterer:
    """
    A class for clustering legal document sections using a pre-trained Llama3-7B:instruct model. This class is designed
    to segment and classify text into predefined legal topics by utilizing LLM (Large Language Model) responses and
    Levenshtein distance for finding the closest matching topics.

    Attributes:
        model (Llama): The Llama model instance used for generating text completions.
        system_prompt (str): The system-level prompt provided to the LLM.
        prompt (str): The user-level prompt provided to the LLM.

    Methods:
        get_prompt() -> Tuple[str, str]:
            Loads the system and user prompts from file.

        get_response(input_text: str) -> str:
            Sends input text to the LLM and retrieves a response.

        find_closest_match(target: str, string_list: List[str]) -> str:
            Finds the closest matching topic using Levenshtein distance.

        process_llm_segmentation(input_df: pd.DataFrame) -> pd.DataFrame:
            Processes a DataFrame of legal document sections, segments them using the LLM, and classifies them
            into predefined legal topics. Adds the classification results to the DataFrame.
    """

    def __init__(self):
        """Initializes LLMClusterer with the locally stored Llama3.1-8B:instruct model and loads necessary prompts."""
        self.model = 'llama3:instruct'
        self.system_prompt, self.prompt = self.get_prompt()

    @staticmethod
    def get_prompt() -> Tuple[str, str]:
        """
        Loads the system and user prompts from file paths specified in constants.
        :return: A tuple containing the system prompt and the user prompt as strings.
        """
        # Load the system prompt
        with open(constants.LLM_SYS_PROMPT_PATH, 'r', encoding='utf-8') as file:
            sys_prompt = file.read()
        # Load the user prompt
        with open(constants.LLM_PROMPT_NL_PATH, 'r', encoding='utf-8') as file:
            prompt = file.read()
        return sys_prompt, prompt

    def get_response(self, input_text: str) -> str:
        """
        Sends input text to the LLM and retrieves a response, utilizing the pre-loaded system and user prompts.
        :param input_text: The text to send to the LLM for segmentation.
        :return: The response generated by the LLM.
        """
        # Create a chat completion request to the LLM
        full_response = ollama.chat(model=self.model, keep_alive=0, options={       # TODO: try generate
            'temperature': 0.0,
            'seed': 42,
            'top_p': 0.0},
            messages=[
                        {
                            'role': 'system',
                            'content': self.system_prompt
                        },
                        {
                            'role': 'user',
                            'content': self.prompt + input_text
                        },
        ])

        # Extract the actual response message from the model's output
        response = full_response['message']['content']
        return response

    @staticmethod
    def find_closest_match(target: str, string_list: List[str]) -> str:
        """
        Finds the closest matching topic from a list using Levenshtein distance (SequenceMatcher).
        :param target: The string to find a match for.
        :param string_list: The list of strings to match against.
        :return: The closest matching string from the list.
        """
        closest_match = None
        highest_ratio = 0.0

        for string in string_list:
            # Compute the similarity ratio
            similarity_ratio = difflib.SequenceMatcher(None, target, string).ratio()

            # Update the closest match if the current one is better
            if similarity_ratio > highest_ratio:
                highest_ratio = similarity_ratio
                closest_match = string

        return closest_match

    def process_llm_segmentation(self, input_df: pd.DataFrame) -> pd.DataFrame:
        """
        Processes a DataFrame of legal document sections, segments them using the LLM, and classifies them into
        predefined legal topics. Adds the classification results to the DataFrame.
        :param input_df: A Pandas DataFrame containing legal document sections to be segmented and classified.
        :return: A DataFrame with added classifications in a new column.
        """
        dict_list = []
        input_df = input_df[:2]

        # Iterate over each row in the input DataFrame
        logger.info("Start LLM section classification...")
        max_len = len(input_df)
        for i, row in enumerate(input_df.itertuples(index=True), start=1):
            # Log progress every 100 rows
            if i / max_len >= 0.1:
                logger.info(f"Processed {row.Index} documents with Llama3:instruct.")

            # Extract the nested dictionary from the current row
            dictionary = getattr(row, constants.SECTIONS_COL)
            nested_dict = {}

            # Process each key-value pair in the nested dictionary
            for key, value in dictionary.items():
                # Get the LLM response for the current text segment
                resp = self.get_response(value)

                # Classify the response based on its length
                if len(resp) >= 200:
                    closest_class = 'Faulty LLM answer'
                else:
                    closest_class = self.find_closest_match(resp, constants.LEGAL_TOPIC_LIST)

                # Add the classification result to the nested dictionary
                nested_dict[key] = {
                    'text': value,
                    'class': closest_class
                }
            # Append the processed dictionary to the list
            dict_list.append(nested_dict)

        # Add the classification results to the DataFrame
        input_df[constants.LLM_RESPONSE_COL] = dict_list
        result_df = pd.DataFrame(input_df)

        return result_df
