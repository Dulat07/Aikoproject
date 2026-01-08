import pandas as pd

def extract_features(df: pd.DataFrame) -> pd.Series:
    """
    Extract statistical features from keystroke dataset.
    We use:
      - mean per column
      - std per column
    """
    means = df.mean()
    stds = df.std()

    # Naming convention
    means.index = [f"{col}_mean" for col in means.index]
    stds.index  = [f"{col}_std"  for col in stds.index]

    # Combine into single vector
    features = pd.concat([means, stds])
    return features