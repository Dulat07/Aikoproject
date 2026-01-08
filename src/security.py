def is_suspicious(profile, new_sample, threshold=0.25) -> bool:
    """
    Return True if user's behavior is suspicious.
    """
    dist = profile.distance(new_sample)
    return dist > threshold