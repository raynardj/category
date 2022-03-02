# category
> Categorical transformation for data science

[![PyPI version](https://img.shields.io/pypi/v/category)](https://pypi.org/project/category)
![Python version](https://img.shields.io/pypi/pyversions/category)
![License](https://img.shields.io/github/license/raynardj/category)
![PyPI Downloads](https://img.shields.io/pypi/dm/category)

## Installation
pip install works for this library.

```shell
pip install category
```

## Single Category
```python
>>> from category import Category
>>> book = Category(['a', 'b', 'c', 'Category_d', 'e', 'f', 'g', 'h', 'i', 'j'], pad_mst = False)
>>> book.i2c[2]
'c'

>>> book.c2i[['Category_d','f']]
array([3, 5])
```

You can set ```pad_mst``` to ```True``` to handle the missing token
```python
>>> from category import Category
>>> book = Category(['a', 'b', 'c', 'Category_d', 'e', 'f', 'g', 'h', 'i', 'j'], pad_mst = True)
>>> book.i2c[2] # the 1st token is the missing token, not 'a' any more
'b'
>>> book.c2i[['Stranger','Category_d','Unknown','f']]
array([0, 4, 0, 6])
```

## Multi-Category
```python
>>> from category import (Category, MultiCategory)
>>> cates = list(f"category{i}" for i in range(1000))
>>> multi_cate = MultiCategory(Category(cates, pad_mst = True))
>>> multi_cate.string_to_index("category42, category108")
array([42, 108])
```

You can also try to convert a list of strings, containing multicategorical info (which the data input is frequently used in tabular data), to nhot encoded array, and back
```python
>>> nhot = multi_cate.batch_strings_to_nhot(["category42, category108","category999"])
>>> multi_cate.nhot_to_list(nhot)[0]
["category42", "category108"]
```

## Performance
The running speed of this library mostly depends on python dictionary and numpy operations. Though python is a 'slow' language, such application is pretty fast, and not easy to improve using other language.

Here we compare the this library with the [Rust implementation](https://github.com/raynardj/rust_category)

## References
* [GitHub](https://github.com/raynardj/category)
* [PyPI package](https://pypi.org/project/category/)
* [Rust implementation](https://github.com/raynardj/rust_category)
* Used in [Tai-Chi engine](https://github.com/tcengine/tai-chi), a verstile user-friendly deep learning library