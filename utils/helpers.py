def flatten_dict(d, parent_key='', sep='.'):
    """
    Flattens a nested dictionary into a single dictionary with dot-separated keys.

    Parameters:
    - d: Dictionary to flatten.
    - parent_key: Key prefix for recursion (internal use).
    - sep: Separator for nested keys.

    Returns:
    - A flattened dictionary.
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)
