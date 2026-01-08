import numpy as np
import pandas as pd

class UserProfile:
    """
    Represents biometric profile of a user based on keystroke features.
    """

    def __init__(self, feature_vector: pd.Series):
        self.mean = feature_vector
        self.std = feature_vector.std()

    def distance(self, sample: pd.Series) -> float:
        """
        Compute normalized Manhattan distance.
        Lower = more similar.
        """
        return np.mean(np.abs(sample - self.mean) / (self.std + 1e-8))