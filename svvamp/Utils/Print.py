def printm(message="Variable =", variable=None):
    """Print a message and the value of a variable, typically a matrix.

    :param message: String.
    :param variable: the variable to be displayed.

    This method is well suited for a matrix because it skips to next line before printing the variable.

    >>> import numpy as np
    >>> printm("A nice matrix:", np.array([[2, 7, 6], [9, 5, 1], [4, 3, 8]]))
    A nice matrix:
    [[2 7 6]
     [9 5 1]
     [4 3 8]]
    """
    print(message)
    print(variable)


def print_title(s):
    """Print a title nicely.

    :param s: String.

    >>> print_title('A title')
    <BLANKLINE>
    ***************
    *   A title   *
    ***************
    """
    length = len(s)
    print('')
    print('*' * (length + 8))
    print('*   ' + s + '   *')
    print('*' * (length + 8))


def print_big_title(s):
    """Print a big title nicely.

    :param s: String.

    >>> print_big_title('A big title')
    <BLANKLINE>
    *******************
    *                 *
    *   A big title   *
    *                 *
    *******************
    """
    length = len(s)
    print('')
    print('*' * (length + 8))
    print('*' + ' ' * (length + 6) + '*')
    print('*   ' + s + '   *')
    print('*' + ' ' * (length + 6) + '*')
    print('*' * (length + 8))
