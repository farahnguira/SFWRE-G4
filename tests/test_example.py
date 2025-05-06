def test_hello_world():
    assert True  # A very basic test that always passes

def test_simple_check():
    x = 10
    y = 20
    assert x < y

# You can also test functions.
# For example, if you had a function in a module like this:
# my_module.py
# def add(a, b):
#     return a + b
#
# You could test it like this (assuming my_module.py is in your PYTHONPATH or project structure):
# from ..my_module import add # Adjust import based on your structure
#
# def test_add_function():
#     assert add(2, 3) == 5
#     assert add(-1, 1) == 0