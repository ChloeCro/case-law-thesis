import pandas as pd
import difflib

from llama_cpp import Llama
from utils import constants, logger_script

logger = logger_script.get_logger(constants.SEGMENTATION_LOGGER_NAME)


class LLMClusterer:

    def __init__(self):
        # Creates a Llama3-7B:instruct instance from a locally stored model
        self.model = Llama(
            model_path=constants.LLAMA3_MODEL_PATH,
            chat_format='llama-3',
            n_ctx=20000
        )
        self.system_prompt, self.prompt = self.get_prompt()

    @staticmethod
    def get_prompt():
        # Load the system prompt
        with open(constants.LLM_SYS_PROMPT_PATH, 'r', encoding='utf-8') as file:
            sys_prompt = file.read()

        with open(constants.LLM_PROMPT_NL_PATH, 'r', encoding='utf-8') as file:
            prompt = file.read()
        return sys_prompt, prompt

    def get_response(self, input_text):
        full_response = self.model.create_chat_completion(messages=[
            {
                'role': 'system',
                'content': self.system_prompt
            },
            {
                'role': 'user',
                'content': self.prompt + input_text
            },
        ],
            temperature=0.0,
            max_tokens=10,
            top_p=0.0,
            frequency_penalty=0,
            presence_penalty=0)

        # Extract the actual response message
        response = full_response['choices'][0]['message']['content']
        return response

    @staticmethod
    def find_closest_match(target, string_list):
        """ Find closest matching topic using Levehstein. """
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

    def process_llm_segmentation(self, input_df):
        dict_list = []
        for i, row in enumerate(input_df.itertuples(index=True), start=1):
            if i % 10 == 0:
                print(f"Index: {row.Index}")
            dictionary = getattr(row, constants.SECTIONS_COL)
            nested_dict = {}
            for key, value in dictionary.items():
                resp = self.get_response(value)
                if len(resp) >= 200:
                    closest_class = 'Faulty LLM answer'
                else:
                    closest_class = self.find_closest_match(resp, constants.LEGAL_TOPIC_LIST)
                print(closest_class)
                nested_dict[key] = {
                    'text': value,
                    'class': closest_class
                }
            dict_list.append(nested_dict)

        input_df[constants.LLM_RESPONSE_COL] = dict_list

        result_df = pd.DataFrame(input_df)

        return result_df
