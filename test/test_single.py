from category import Category
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

