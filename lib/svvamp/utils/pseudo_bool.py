import numpy as np


def equal_true(x):
    """
    Test whether `x` is equal to True.

    Parameters
    ----------
    x : object

    Returns
    -------
    bool
        True iff `x` is equal to True. This includes values such as 1., 1, etc.

    Examples
    --------
    By definition, `equal_true(x)` is equivalent to `x == True`.

    However, `equal_true(x)` is not the same as `x is True`:

        >>> my_x = 1.
        >>> equal_true(my_x)
        True
        >>> my_x is True
        False

    Similarly, `equal_true(x)` is not the same as `bool(x)`:

        >>> my_x = -np.inf
        >>> equal_true(my_x)
        False
        >>> bool(my_x)
        True
    """
    return x == True


def equal_false(x):
    """
    Test whether x is equal to False.

    Parameters
    ----------
    x : object

    Returns
    -------
    bool
        True iff `x` is equal to False. This includes values such as 0., 0, etc.

    Examples
    --------
    By definition, `equal_false(x)` is equivalent to `x == False.

    However, `equal_false(x)` is not the same as `x is False`:

        >>> my_x = 0.
        >>> equal_false(my_x)
        True
        >>> my_x is False
        False

    Similarly, `equal_false(x)` is not the same as `not bool(x)`:

        >>> my_x = None
        >>> equal_false(my_x)
        False
        >>> not bool(my_x)
        True
    """
    return x == False


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
    elif equal_true(value):
        return True
    elif equal_false(value):
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


def pseudo_bool_not(x):
    """
    Negation of `x`.

    Parameters
    ----------
    x: object

    Returns
    -------
    bool or float
        True, False, nan or -np.inf. Let us recall that in SVVAMP, `-np.inf` means "I have not started to compute
        this value", and `np.nan` means "I tried to compute this value, and I decided that I don't know".

    Examples
    --------
        >>> pseudo_bool_not(1.)
        False
        >>> pseudo_bool_not(0)
        True
        >>> pseudo_bool_not(np.nan)
        nan
        >>> pseudo_bool_not(-np.inf)
        -inf
    """
    if equal_true(x):
        return False
    elif equal_false(x):
        return True
    else:  # variable = nan or -np.inf
        return x
