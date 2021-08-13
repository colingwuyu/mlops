import json


def pretty_dict(dict_val):
    return json.dumps(dict_val, indent=2)


def pad_tab(orig_string: str, num_tabs: int = 1):
    paddings = "\t" * num_tabs
    return paddings + paddings.join(orig_string.splitlines(True))
