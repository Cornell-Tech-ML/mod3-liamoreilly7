"""Collection of the core mathematical operators used throughout the code base."""

import math

from typing import Callable, Iterable


def mul(x: float, y: float) -> float:
    """Multiplies two numbers."""
    return x * y


def id(x: float) -> float:
    """Returns the input unchanged."""
    return x


def add(x: float, y: float) -> float:
    """Returns the addition of two numbers."""
    return x + y


def neg(x: float) -> float:
    """Return the negated value of a number."""
    return -x


def lt(x: float, y: float) -> bool:
    """Compare two numbers x and y, return true if x < y and false otherwise."""
    return x < y


def eq(x: float, y: float) -> float:
    """Checks if two numbers are equal."""
    return x == y


def max(x: float, y: float) -> float:
    """Returns the larger of the two numbers."""
    return x if x > y else y


def is_close(x: float, y: float) -> bool:
    """Checks if two numbers are close in value."""
    return abs(x - y) < 1e-2


def sigmoid(x: float) -> float:
    """Calculate the sigmoid function of the value."""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    else:
        return math.exp(x) / (1.0 + math.exp(x))


def relu(x: float) -> float:
    """Applies the ReLu activation function to the value."""
    return max(0.0, x)


def log(x: float) -> float:
    """Calculates the natural logarithm of the number."""
    return math.log(x)


def exp(x: float) -> float:
    """Calculates the exponential function of the number."""
    return math.exp(x)


def log_back(x: float, y: float) -> float:
    """Calculates the derivative of log times a second arg."""
    return (1 / x) * y


def inv(x: float) -> float:
    """Calculates the reciprocal."""
    return 1 / x


def inv_back(x: float, y: float) -> float:
    """Computes the derivative of reciprocal times a second arg."""
    return -y / (x * x)


def relu_back(x: float, y: float) -> float:
    """Computes the derivative of ReLU times a second arg."""
    return y if x >= 0 else 0


def map(fn: Callable) -> Callable:
    """Returns a function that applies the given function `fn` to each element of an iterable.

    Args:
    ----
        fn (Callable): A function that takes one argument and returns an iterable.

    Returns:
    -------
        A function that takes an iterable and returns an iterable with `fn` applied to each element.

    """

    def new_fn(ls: Iterable) -> Iterable:
        return [fn(x) for x in ls]

    return new_fn


def zipWith(fn: Callable) -> Callable:
    """Higher order function that combines elements from two iterables using a given function.

    Args:
    ----
        fn (Callable): A function that takes two arguments and returns an iterable.

    Returns:
    -------
        A function that takes two arguments and applies the given function to corresponding elements from the two iterables.

    """

    def new_fn(ls1: Iterable, ls2: Iterable) -> Iterable:
        return [fn(x, y) for x, y in zip(ls1, ls2)]

    return new_fn


def reduce(fn: Callable) -> Callable:
    """Reduces an iterable to a single value using a function.

    Args:
    ----
        fn (Callable): Function that takes two arguments and returns a value.

    Returns:
    -------
        A function that reduces value after applying `fn` cumulatively to the elements of the iterable.

    """

    def new_fn(ls: Iterable, initializer: float) -> float:
        result = initializer
        for x in ls:
            result = fn(result, x)
        return result

    return new_fn


def negList(ls: Iterable) -> Iterable:
    """Negates all elements in a list by using map.

    Args:
    ----
        ls (Iterable): A list of numbers.

    Returns:
    -------
        Iterable: An iterable with each element negated.

    """
    f_neg = map(neg)
    return f_neg(ls)


def addLists(ls1: Iterable, ls2: Iterable) -> Iterable:
    """Adds corresponding elements from two lists by using zipWith.

    Args:
    ----
        ls1: list of numbers
        ls2: list of numbers.

    Returns:
    -------
        An iterable where each element is the sum of corresponding elements from `ls1` and `ls2`.

    """
    fn_zipWith = zipWith(add)
    return fn_zipWith(ls1, ls2)


def sum(ls: Iterable) -> float:
    """Sums all elements in a list by using reduce."""
    fn_sum = reduce(add)
    return fn_sum(ls, 0)


def prod(ls: Iterable) -> float:
    """Computes the product of all elements in a list by using reduce."""
    fn_prod = reduce(mul)
    return fn_prod(ls, 1)
