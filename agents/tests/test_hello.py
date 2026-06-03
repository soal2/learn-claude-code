"""Tests for hello.py."""

from unittest.mock import patch

import pytest

from agents.hello import greet, main


class TestGreet:
    """Tests for the greet function."""

    def test_simple_name(self):
        """Test greeting with a simple name."""
        assert greet("Alice") == "Hello, Alice!"

    def test_name_with_spaces(self):
        """Test greeting with a name containing spaces."""
        assert greet("John Doe") == "Hello, John Doe!"

    def test_empty_string(self):
        """Test greeting with an empty string."""
        assert greet("") == "Hello, !"

    def test_single_character(self):
        """Test greeting with a single character name."""
        assert greet("A") == "Hello, A!"

    def test_numbers_as_name(self):
        """Test greeting with numbers as name."""
        assert greet("123") == "Hello, 123!"

    def test_special_characters(self):
        """Test greeting with special characters."""
        assert greet("@#$%") == "Hello, @#$%!"

    def test_unicode_name(self):
        """Test greeting with unicode characters."""
        assert greet("Ñoño") == "Hello, Ñoño!"
        assert greet("田中") == "Hello, 田中!"

    def test_whitespace_only_name(self):
        """Test greeting with whitespace-only name."""
        assert greet("   ") == "Hello,    !"

    def test_name_with_leading_trailing_spaces(self):
        """Test greeting preserves leading/trailing spaces."""
        assert greet(" Alice ") == "Hello,  Alice !"

    def test_long_name(self):
        """Test greeting with a very long name."""
        long_name = "A" * 1000
        result = greet(long_name)
        assert result == f"Hello, {long_name}!"
        assert len(result) == 1008  # "Hello, " + 1000 + "!"


class TestMain:
    """Tests for the main function."""

    @patch("builtins.print")
    def test_main_prints_greeting(self, mock_print):
        """Test that main function prints the correct greeting."""
        main()
        mock_print.assert_called_once_with("Hello, World!")

    @patch("builtins.print")
    def test_main_calls_print_once(self, mock_print):
        """Test that main function calls print exactly once."""
        main()
        assert mock_print.call_count == 1


class TestGreetReturnType:
    """Tests for return type and format of greet function."""

    def test_returns_string(self):
        """Test that greet returns a string type."""
        result = greet("Test")
        assert isinstance(result, str)

    def test_starts_with_hello(self):
        """Test that greeting starts with 'Hello'."""
        result = greet("Name")
        assert result.startswith("Hello")

    def test_ends_with_exclamation(self):
        """Test that greeting ends with exclamation mark."""
        result = greet("Name")
        assert result.endswith("!")

    def test_format_pattern(self):
        """Test that the greeting follows the expected format."""
        name = "TestUser"
        result = greet(name)
        # Should be exactly "Hello, <name>!"
        assert result == f"Hello, {name}!"
