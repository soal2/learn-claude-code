"""Utility functions for the package."""


def _validate_numeric(value: float, name: str) -> None:
    """Validate that a value is numeric and not NaN or Infinity.

    Args:
        value: The value to validate.
        name: The name of the parameter (for error messages).

    Raises:
        TypeError: If the value is not a number.
        ValueError: If the value is NaN or Infinity.
    """
    if isinstance(value, bool):
        # bool is a subclass of int, but we should reject it explicitly
        raise TypeError(f"{name} must be a number, got bool")
    if not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number, got {type(value).__name__}")
    if isinstance(value, float):
        if value != value:  # NaN check (NaN != NaN)
            raise ValueError(f"{name} cannot be NaN")
        if value == float('inf') or value == float('-inf'):
            raise ValueError(f"{name} cannot be Infinity")


def add(a: float, b: float) -> float:
    """Return the sum of two numbers.

    Args:
        a: First number.
        b: Second number.

    Returns:
        The sum of a and b.

    Raises:
        TypeError: If either argument is not a number.
        ValueError: If either argument is NaN or Infinity.
    """
    _validate_numeric(a, "a")
    _validate_numeric(b, "b")
    return a + b


def subtract(a: float, b: float) -> float:
    """Return the difference of two numbers.

    Args:
        a: First number (minuend).
        b: Second number (subtrahend).

    Returns:
        The difference a - b.

    Raises:
        TypeError: If either argument is not a number.
        ValueError: If either argument is NaN or Infinity.
    """
    _validate_numeric(a, "a")
    _validate_numeric(b, "b")
    return a - b


def multiply(a: float, b: float) -> float:
    """Return the product of two numbers.

    Args:
        a: First number (multiplicand).
        b: Second number (multiplier).

    Returns:
        The product of a and b.

    Raises:
        TypeError: If either argument is not a number.
        ValueError: If either argument is NaN or Infinity.
    """
    _validate_numeric(a, "a")
    _validate_numeric(b, "b")
    return a * b


def divide(a: float, b: float) -> float:
    """Return the quotient of two numbers.

    Args:
        a: Dividend (numerator).
        b: Divisor (denominator).

    Returns:
        The quotient a / b.

    Raises:
        TypeError: If either argument is not a number.
        ValueError: If either argument is NaN or Infinity.
        ZeroDivisionError: If b is zero.
    """
    _validate_numeric(a, "a")
    _validate_numeric(b, "b")
    if b == 0:
        raise ZeroDivisionError("Cannot divide by zero")
    return a / b
