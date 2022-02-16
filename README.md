# category
> Categorical transformation for data science

[![PyPI version](https://img.shields.io/pypi/v/tai-chi-engine)](https://pypi.org/project/category)
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

