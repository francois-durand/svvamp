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

    Examples
    --------
        >>> pseudo_bool(np.nan)
        nan
        >>> pseudo_bool(1.)
        True
        >>> pseudo_bool(0.)
        False
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

    Parameters
    ----------
    variable : number or array

    Returns
    -------
    number or array.

    Examples
    --------
    If ``variable = -inf``, then ``variable`` is converted to ``NaN`` :

        >>> neginf_to_nan(-np.inf)
        nan

    Otherwise, it is returned without modification:

        >>> neginf_to_nan(42)
        42

    If ``variable`` is a numpy array, it is converted element-wise and IN PLACE, i.e the original array is modified:

        >>> a = np.array([1, -np.inf, 2, -np.inf])
        >>> neginf_to_nan(a)
        array([ 1., nan,  2., nan])
        >>> a
        array([ 1., nan,  2., nan])

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

    Examples
    --------
    If ``variable = -inf``, then ``variable`` is converted to 0:

        >>> neginf_to_zero(-np.inf)
        0

    Otherwise, it is returned without modification:

        >>> neginf_to_zero(42)
        42

    If ``variable`` is a numpy array, it is converted element-wise and IN PLACE, i.e the original array is modified:

        >>> a = np.array([1, -np.inf, 2, -np.inf])
        >>> neginf_to_zero(a)
        array([1., 0., 2., 0.])
        >>> a
        array([1., 0., 2., 0.])
    """
    if isinstance(variable, np.ndarray):
        variable[np.isneginf(variable)] = 0
    elif np.isneginf(variable):
        variable = 0
    return variable
