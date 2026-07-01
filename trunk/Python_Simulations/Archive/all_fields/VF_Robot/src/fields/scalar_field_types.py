"""
Scalar field abstractions for scalar field environments.

All scalar field types provide a unified interface: get_value(x, y) -> z
"""
import numpy as np


class ScalarField:
    """Base class for scalar fields."""

    def get_value(self, x, y):
        """
        Get scalar field value at position (x, y).

        Args:
            x: x coordinate (scalar or array)
            y: y coordinate (scalar or array)

        Returns:
            z: Scalar field value at (x, y)
        """
        raise NotImplementedError("Subclasses must implement get_value()")


class AnalyticalScalarField(ScalarField):
    """
    Analytical scalar field using a mathematical function.
    """

    def __init__(self, field_function):
        """
        Args:
            field_function: Function with signature (x, y) -> z
        """
        self.field_function = field_function

    def get_value(self, x, y):
        """Get field value using analytical function."""
        return self.field_function(x, y)
