"""Functions to add an attribute to an experiment."""

from typing import TypeVar

T = TypeVar("T")


def add_attribute_to_experiment(experiment: T, attr_dict: dict[str, str]) -> None:
    """Add an attribute to an experiment.

    Args:
        experiment (T): The experiment object.
        attr_dict (dict[str, str]): The dictionary containing the attribute and its
            value.
    """
    for attribute, attribute_value in attr_dict.items():
        setattr(experiment, attribute, attribute_value)
