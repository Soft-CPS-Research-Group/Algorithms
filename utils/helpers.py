from typing import Dict, List
import numpy as np
from citylearn.citylearn import CityLearnEnv

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

def set_default_config(config, key_path, default_value):
    """
    Sets a default value in a nested dictionary if the value is None or empty.

    Parameters:
    - config: The dictionary to update.
    - key_path: A list of keys to navigate the nested structure.
    - default_value: The default value to set if the target is None or empty.
    """
    target = config
    for key in key_path[:-1]:
        target = target.setdefault(key, {})  # Navigate or create nested dicts
    if target.get(key_path[-1]) in (None, [], {}, ""):
        target[key_path[-1]] = default_value
