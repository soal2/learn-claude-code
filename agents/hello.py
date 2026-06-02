"""A simple greeting module.

This module provides functions for generating greeting messages
and demonstrating best practices in Python code structure.
"""

from __future__ import annotations


def greet(name: str) -> str:
    """Return a greeting message for the given name.

    Args:
        name: The name of the person to greet.

    Returns:
        A greeting string in the form 'Hello, <name>!'.
    """
    return f"Hello, {name}!"


def main() -> None:
    """Run the greeting demo.

    Prints a friendly greeting to the console.
    """
    message: str = greet("World")
    print(message)


if __name__ == "__main__":
    main()
