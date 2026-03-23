"""Sample file with known call graph for testing."""

import os
from pathlib import Path


def helper():
    """A helper function called by others."""
    return Path.home()


def process_data(data):
    """Processes data using helper."""
    result = helper()
    validated = validate(data)
    return result, validated


def validate(data):
    """Validates input data."""
    path = os.path.join("/tmp", str(data))
    return os.path.exists(path)


class DataProcessor:
    """A class that uses the functions above."""

    def run(self, data):
        result = process_data(data)
        self.log(result)
        return result

    def log(self, message):
        print(message)
