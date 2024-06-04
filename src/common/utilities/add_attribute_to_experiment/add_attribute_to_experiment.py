"""Functions to add an attribute to an experiment."""

from typing import TypeVar

T = TypeVar("T")


def add_attribute_to_experiment(
    experiment: T,
    attribute: str,
    attribute_attribute_value: T,
) -> None:
    """Add attribute to experiment.

    Some attributes from the experiment class are dropped after saving the class.
    Also, some attributes information is required in downstream functions.

    Function will enable user to add the missing/necessary attributes to the experiment.

    Args:
        experiment: An experiment created by PyCaret.
        attribute: Name of the attribute.
        attribute_attribute_value: Value of the attribute.
    """
    setattr(experiment, attribute, attribute_attribute_value)
