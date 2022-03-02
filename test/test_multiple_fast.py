from category.fast import (
    Category, MultiCategory
    )
import numpy as np

cates = list(f"category{i}" for i in range(1000))
cate_unpad = Category(cates, False)
cate_pad = Category(cates, True)

def test_multi_category_to_indices_pad():
    c = MultiCategory(cate_pad)
    assert (c.string_to_index("category1, category2, category999")==np.array([2,3,1000])).all()

def test_multi_category_to_indices_unpad():
    c = MultiCategory(cate_unpad)
    assert (c.string_to_index("category42, category108")==np.array([42,108])).all()

def test_multi_category_batch_strings_to_nhot():
    c = MultiCategory(cate_unpad)
    nhot = c.batch_strings_to_nhot(["category42, category108", "category999"])
    assert nhot.sum() == 3
    assert nhot[0].sum() == 2
    assert nhot[1].sum() == 1
    assert nhot[0][[42,108]].sum() == 2
    assert c.nhot_to_list(nhot)[0] == ["category42", "category108"]