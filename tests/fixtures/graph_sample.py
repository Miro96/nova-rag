"""Sample file with known call graph and class hierarchy for testing."""

import os
from pathlib import Path


class BaseProcessor:
    """Base class for processors."""

    def process(self, data):
        return data


class DataProcessor(BaseProcessor):
    """A processor that extends BaseProcessor."""

    def process(self, data):
        result = helper()
        validated = validate(data)
        self.log(result)
        return result, validated

    def log(self, message):
        print(message)


class AdvancedProcessor(DataProcessor):
    """Extends DataProcessor."""

    def process(self, data):
        result = super().process(data)
        return transform(result)


def helper():
    """A helper function called by others."""
    return Path.home()


def validate(data):
    """Validates input data."""
    path = os.path.join("/tmp", str(data))
    return os.path.exists(path)


def transform(data):
    """Transforms data."""
    return str(data).upper()


def unused_function():
    """This function is never called anywhere."""
    return 42


def another_unused():
    """Another dead code function."""
    return "dead"
