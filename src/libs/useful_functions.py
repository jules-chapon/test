"""Useful functions"""

from collections import defaultdict


def nested_defaultdict():
    """
    Create a defauldict (list).

    Returns:
        _type_: Defaultdict.
    """
    return defaultdict(list)


def double_nested_defaultdict():
    """
    Create a nested defaultdict (list).

    Returns:
        _type_: Nested defaultdict.
    """
    return defaultdict(nested_defaultdict)


def triple_nested_defaultdict():
    """
    Create a nested defaultdict (list).

    Returns:
        _type_: Nested defaultdict.
    """
    return defaultdict(double_nested_defaultdict)
