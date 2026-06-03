"""Tests for utils.py."""

import pytest

from agents.utils import add, subtract, multiply, divide


class TestAdd:
    def test_positive_numbers(self):
        assert add(2, 3) == 5

    def test_negative_numbers(self):
        assert add(-1, -1) == -2

    def test_zero(self):
        assert add(5, 0) == 5

    def test_floats(self):
        assert add(0.1, 0.2) == pytest.approx(0.3)

    def test_type_error_string(self):
        with pytest.raises(TypeError, match="a must be a number"):
            add("1", 2)

    def test_type_error_none(self):
        with pytest.raises(TypeError, match="b must be a number"):
            add(1, None)

    def test_type_error_bool(self):
        with pytest.raises(TypeError, match="a must be a number"):
            add(True, 1)

    def test_value_error_nan(self):
        with pytest.raises(ValueError, match="a cannot be NaN"):
            add(float('nan'), 1)

    def test_value_error_inf(self):
        with pytest.raises(ValueError, match="b cannot be Infinity"):
            add(1, float('inf'))


class TestSubtract:
    def test_positive_numbers(self):
        assert subtract(5, 3) == 2

    def test_negative_result(self):
        assert subtract(3, 5) == -2

    def test_zero(self):
        assert subtract(5, 0) == 5

    def test_type_error(self):
        with pytest.raises(TypeError, match="a must be a number"):
            subtract("5", 3)


class TestMultiply:
    def test_positive_numbers(self):
        assert multiply(3, 4) == 12

    def test_zero(self):
        assert multiply(5, 0) == 0

    def test_negative(self):
        assert multiply(-3, 4) == -12

    def test_floats(self):
        assert multiply(0.5, 0.5) == pytest.approx(0.25)

    def test_type_error(self):
        with pytest.raises(TypeError, match="b must be a number"):
            multiply(5, "4")


class TestDivide:
    def test_positive_numbers(self):
        assert divide(10, 2) == 5

    def test_float_result(self):
        assert divide(7, 2) == 3.5

    def test_zero_numerator(self):
        assert divide(0, 5) == 0

    def test_divide_by_zero(self):
        with pytest.raises(ZeroDivisionError, match="Cannot divide by zero"):
            divide(5, 0)

    def test_type_error(self):
        with pytest.raises(TypeError, match="a must be a number"):
            divide("10", 2)

    def test_value_error_nan(self):
        with pytest.raises(ValueError, match="a cannot be NaN"):
            divide(float('nan'), 1)
