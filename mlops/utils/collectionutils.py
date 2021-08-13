import collections

import numpy as np


def convert_to_namedtuple(dictionary: dict):
    """convert dictionary to namedtuple

    Args:
        dictionary (dict): dictionary to be converted

    Returns:
        namedtuple: a immutable namedtuple
    """
    for key, value in dictionary.items():
        if isinstance(value, dict):
            dictionary[key] = convert_to_namedtuple(value)
    return collections.namedtuple("configuration", dictionary.keys())(**dictionary)


def update_dictionary(base, updated):
    """update dictionary base with the values from updated

    Args:
        base (dict): base dictionary
        updated (dict): update values

    Returns:
        dict: updated dictionary
    """
    for k, v in updated.items():
        if isinstance(v, collections.abs.Mapping):
            base[k] = update_dictionary(base.get(k, {}), v)
        else:
            base[k] = v
    return base


def convert_json_type(data: dict):
    def default(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray) or isinstance(obj, list):
            return [default(obj_item) for obj_item in obj]
        else:
            return obj

    for k, v in data.items():
        if isinstance(v, collections.abc.Mapping):
            data[k] = convert_json_type(v)
        else:
            data[k] = default(v)


def isnamedtupleinstance(x):
    _type = type(x)
    bases = _type.__bases__
    if len(bases) != 1 or bases[0] != tuple:
        return False
    fields = getattr(_type, "_fields", None)
    if not isinstance(fields, tuple):
        return False
    return all(type(i) == str for i in fields)


def unpack(obj):
    if isinstance(obj, dict):
        return {key: unpack(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [unpack(value) for value in obj]
    elif isnamedtupleinstance(obj):
        return {key: unpack(value) for key, value in obj._asdict().items()}
    elif isinstance(obj, tuple):
        return tuple(unpack(value) for value in obj)
    else:
        return obj
