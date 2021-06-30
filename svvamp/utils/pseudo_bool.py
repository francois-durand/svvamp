import numpy as np


# noinspection PySimplifyBooleanCheck
def pseudo_bool(value):
    """Convert to pseudo-Boolean.

    Parameters
    ----------
    value : bool or nan
        True (or 1, 1., etc.), False (or 0, 0., etc.) or np.nan.

    Returns
    -------
    bool or nan
        True, False (as booleans) or np.nan.
    """
    if np.isnan(value):
        return np.nan
    elif value == True:
        return True
    elif value == False:
        return False
    else:
        raise ValueError("Expected Boolean or np.nan and got:" + format(value))


def neginf_to_nan(variable):
    """Convert -inf to NaN.

    If ``variable = -inf``, then ``variable`` is converted to ``NaN``. If ``variable`` is a numpy array, it is
    converted element-wise and IN PLACE (the original array is modified).

    Parameters
    ----------
    variable : number or array

    Returns
    -------
    number or array.
    """
    if isinstance(variable, np.ndarray):
        variable[np.isneginf(variable)] = np.nan
    elif np.isneginf(variable):
        variable = np.nan
    return variable


def neginf_to_zero(variable):
    """Convert -inf to 0 / False.

    If ``variable = -inf``, then ``variable`` is converted to 0. If ``variable`` is a numpy array, it is
    converted element-wise and IN PLACE (the original array is modified).

    Parameters
    ----------
    variable: number or array

    Returns
    -------
    number of array.
    """
    if isinstance(variable, np.ndarray):
        variable[np.isneginf(variable)] = 0
    elif np.isneginf(variable):
        variable = 0
    return variable
