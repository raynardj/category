from .single import Category, C2I
from typing import List
import numpy as np

class MultiCategory:
    """
    Process multi-category situation
    """
    def __init__(self, category: Category, spliter: str = ","):
        self.category = category
        self.spliter = spliter

    @property
    def empty(self):
        return np.zeros(len(self.category))

    def __len__(self):
        return len(self.category)

    def string_to_list(self, text: str) -> List[str]:
        return list(c.strip()
            for c in text.split(self.spliter))

    def string_to_index(self, text: str) -> List[int]:
        """
        Split the string to a list of categories
        Then translate each category to index
        """
        return list(self.category.c2i[self.string_to_list(text)])

    def batch_strings_to_nhot(
        self, text_inputs: List[str]
        ) -> np.ndarray:
        """
        Convert a list of string to nhot array
        """
        empty = self.empty
        results = []
        for text_input in text_inputs:
            row = empty.copy()
            row[self.string_to_index(text_input)] = 1.
            results.append(row)
        return np.stack(results)

    def nhot_to_list(self, nhot: np.ndarray) -> List[str]:
        """
        Convert a nhot array to a list of string
        """
        results = []
        for row in nhot:
            results.append(list(self.category.i2c[row>.5]))
        return results

