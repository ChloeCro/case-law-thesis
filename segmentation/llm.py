import pandas as pd

from utils import constants, logger_script

logger = logger_script.get_logger(constants.SEGMENTATION_LOGGER_NAME)


class LLMClusterer:

    def __init__(self, model):
        self.model = model
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
        response = ollama.chat(model=self.model, keep_alive=0, options={'temperature': 0.0, 'seed': 42, "top_p": 0.0}, messages=[
            {
                'role': 'system',
                'content': self.system_prompt
            },
            {
                'role': 'user',
                'content': self.prompt + input_text
            },
        ])
        return response

    def find_closest_match(self, target, string_list):
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
                resp = self.get_response(value)['message']['content']
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
        pass

