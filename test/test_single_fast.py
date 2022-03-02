from category.fast import Category
import numpy as np

cates = list(f"category{i}"*2 for i in range(1000))

inputs = cates*1000

results_pad = list(i+1 for i in range(1000))*1000+[0,]
results_unpad = list(i for i in range(1000))*1000

def test_category_to_indices_pad():
    c = Category(cates, True)
    assert (c.c2i[inputs+["Unknown Category"]]==np.array(results_pad)).all()

def test_category_to_indices_unpad():
    c = Category(cates, False)
    assert (c.c2i[inputs]==np.array(results_unpad)).all()

def test_indices_to_categories_pad():
    c = Category(cates, True)
    assert (c.i2c[results_pad]==inputs+["[MST]"]).all()

def test_indices_to_categories_unpad():
    c = Category(cates, False)
    assert (c.i2c[results_unpad]==inputs).all()

