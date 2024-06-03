"""Utility function to extract the target variable name from a list of column names."""


def extract_target_variable_name(column_names: list[str]) -> str:
    """Extract the target variable name from a list of column names.

    Args:
    ----
        column_names (list[str]): A list of column names.

    Returns:
    -------
        str: The name of the target variable.

    Raises:
    ------
        ValueError: If multiple target variables are found without an underscore.
        ValueError: If no target variable is found without an underscore.

    """
    target_variable = None
    for name in column_names:
        if "_" not in name:
            if target_variable is not None:
                raise ValueError("Multiple target variables found without underscore.")
            target_variable = name
    if target_variable is None:
        raise ValueError("No target variable found without underscore.")
    return target_variable
