from rust_category import Category as RustCategory
from rust_category import MultiCategory as RustMultiCategory
from typing import Iterable, Dict, List
import numpy as np
from pathlib import Path
import json


class C2I:
    def __init__(self,
                 arr: Iterable,
                 pad_mst: bool = False
                 ):
        self.pad_mst = pad_mst
        self.pad = ["[MST]", ] if self.pad_mst else []
        self.core = RustCategory(arr, pad_mst)

    def __getitem__(self, k: int):
        if type(k) in [np.ndarray, list]:
            return self.core.categories_to_indices(k)
        else:
            return self.core.categories_to_indices([k, ])[0]

    def __len__(self):
        return len(self.core)

    def __repr__(self):
        return f"C2I:{self.__len__()} categories"


class Category:
    """
    - Manage categorical translations
    - using the rust core
    c = Category(
            ["class 1", "class 2", ..., "class n"],
            pad_mst=True,)

    c.c2i[["class 3","class 6"]]
    c.i2c[[3, 2, 1]]
    """

    def __init__(
        self,
        arr: Iterable,
        pad_mst: bool = False
    ):
        self.c2i = C2I(arr, pad_mst=pad_mst)
        self.i2c = np.array(self.c2i.pad+list(arr))
        self.pad_mst = pad_mst
        self.core = self.c2i.core

    def __len__(self):
        return len(self.i2c)

    def __repr__(self):
        return f"Category:{self.__len__()} categories"

    def save(self, path: Path) -> None:
        """
        save category information to json file
        """
        with open(path, "w") as f:
            json.dump(self.i2c.tolist(), f)

    @classmethod
    def load(cls, path: Path):
        """
        load category information from a json file
        """
        with open(path, "r") as f:
            l = np.array(json.load(f))
        if l[0] == "[MST]":
            return cls(l[1:], pad_mst=True)
        else:
            return cls(l, pad_mst=False)


class MultiCategory:
    """
    Process multi-category situation
    """

    def __init__(
        self, category: Category, spliter: str = ",",
    ):
        self.category = category
        self.spliter = spliter
        self.pad_mst = category.pad_mst
        self.core = RustMultiCategory(
            category.i2c[1:] if self.pad_mst else category.i2c,
            self.pad_mst, spliter)

    @property
    def empty(self):
        return np.zeros(len(self.category))

    def __len__(self):
        return len(self.category)

    def string_to_index(self, text: str) -> List[str]:
        return self.core.categories_to_indices([text, ])[0]

    def batch_strings_to_nhot(
        self, text_inputs: List[str]
    ) -> np.ndarray:
        """
        Convert a list of string to nhot array
        """
        return np.array(
            self.core.categories_to_nhot(text_inputs))

    def nhot_to_list(self, nhot: np.ndarray) -> List[str]:
        """
        Convert a nhot array to list of string
        """
        return self.core.nhot_to_list(nhot.tolist())
