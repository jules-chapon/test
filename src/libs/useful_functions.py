"""Useful functions"""

from collections import defaultdict


def nested_defaultdict():
    return defaultdict(list)


def double_nested_defaultdict():
    return defaultdict(nested_defaultdict)


def triple_nested_defaultdict():
    return defaultdict(double_nested_defaultdict)
