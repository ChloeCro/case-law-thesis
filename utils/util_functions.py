import os
import ast
import pandas as pd


def safe_literal_eval(val: str) -> any:
    """
    Safely evaluate a string literal to a Python object to infer the correct datatype. Returns original value if error
    is raised.
    :param val: The string to be evaluated.
    :return: The evaluated Python object, or the original string if evaluation fails.
    """
    try:
        return ast.literal_eval(val)
    except (ValueError, SyntaxError):
        return val


def load_file_to_df(file_path: str) -> pd.DataFrame:
    """
    Load a CSV or XLSX file into a Pandas DataFrame with safe literal evaluation to ensure correct datatypes.
    :param file_path: The path to the file to load.
    :return: A Pandas DataFrame with correct datatypes.
    """
    file_extension = os.path.splitext(file_path)[1].lower()

    if file_extension == '.csv':
        df = pd.read_csv(file_path)
    elif file_extension == '.xlsx':
        df = pd.read_excel(file_path)
    else:
        raise ValueError(f"Unsupported file extension: {file_extension}")

    df = df.applymap(safe_literal_eval)
    return df
