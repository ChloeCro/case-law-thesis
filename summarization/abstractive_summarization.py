from transformers import BartForConditionalGeneration, BartTokenizer
from utils import constants, logger_script

logger = logger_script.get_logger(constants.SUMMARIZATION_LOGGER_NAME)


class AbstractiveSummarizer:

    def __init__(self):
        model_name = "facebook/bart-large-cnn"
        self.bart_model = BartForConditionalGeneration.from_pretrained(model_name)
        self.bart_tokenizer = BartTokenizer.from_pretrained(model_name)
        pass

    def apply_bart(self, text):
        inputs = self.bart_tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=1024, truncation=True)
        summary_ids = self.bart_model.generate(inputs, max_length=500, min_length=300, length_penalty=1, num_beams=4,
                                     early_stopping=True)

        summary = self.bart_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        print("in bart module: ", summary)
        # formatted_summary = "\n".join(textwrap.wrap(summary, width=80))
        return summary

    def apply_llama(self):
        pass