import pandas as pd

from nltk.tokenize import sent_tokenize
from utils import constants
from typing import List
from tqdm import tqdm

# Enable the tqdm progress bar for pandas
tqdm.pandas()


def safe_tokenize(text):
    if isinstance(text, str):
        return sent_tokenize(text)
    else:
        return []


def tokenize_sentences(df: pd.DataFrame, column_name: str) -> pd.DataFrame:
    """
    Applies sentence tokenization to each text in the specified column of a DataFrame.
    :param df: The input dataframe with text that need to be tokenized.
    :param column_name: The name of the column to be tokenized.
    :return A dataframe with added column that contains tokenized text.
    """
    df[constants.TOKENIZED_COL] = df[column_name].progress_apply(safe_tokenize)
    return df


def generate_trigrams(sentences: List[str]) -> List[List[str]]:
    """
    Generates trigrams (lists of 3 consecutive sentences) from a list of sentences.

    If the list contains fewer than 3 sentences, returns the entire list as a single element list.
    If the list contains exactly 3 sentences, returns a list with one element containing those 3 sentences.
    If the list contains more than 3 sentences, returns a list of trigrams. If the number of sentences
    is not a multiple of 3, the last element will contain the remaining sentences.

    Args:
        sentences (List[str]): A list of sentences.

    Returns:
        List[List[str]]: A list of trigrams, where each trigram is a list of 3 sentences. The last element
        may contain fewer than 3 sentences if the total number of sentences is not divisible by 3.
    """
    n = len(sentences)

    # If there are less than 3 sentences, return list of all sentences
    if n < 3:
        return [sentences]

    trigrams = []

    # Exactly 3 sentences
    if n == 3:
        trigrams.append(sentences[:3])
    else:
        # More than 3 sentences
        i = 0
        while i < n:
            # Keep adding lists of 3 strings to the trigrams list
            if i + 3 <= n:
                trigrams.append(sentences[i:i+3])
                i += 3
            else:
                # When at the last 1 - 3 sentences, we append them together
                trigrams.append(sentences[i:])
                break

    return trigrams
