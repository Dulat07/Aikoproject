import pandas as pd

def load_csv(path: str) -> pd.DataFrame:
    """
    Load a keystroke CSV dataset.
    """
    df = pd.read_csv(path)
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean raw keystroke dataset:
    - replace invalid numbers
    - remove NaNs
    - remove infinite values
    """
    df = df.replace([float("inf"), float("-inf"), -999, -1e9], None)
    df = df.dropna().reset_index(drop=True)
    return df


def preprocess(path: str) -> pd.DataFrame:
    """
    Full preprocessing pipeline:
    load → clean → return dataframe
    """
    df = load_csv(path)
    df = clean(df)
    return df