# -*- coding: utf-8 -*-
"""
Created on oct. 20, 2014, 11:03
Copyright François Durand 2014
fradurand@gmail.com

This file is part of SVVAMP.

    SVVAMP is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    SVVAMP is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with SVVAMP.  If not, see <http://www.gnu.org/licenses/>.
"""

import numbers


def is_bool(value):
    """Test whether it is a Boolean.

    Parameters
    ----------
    value : obj
        Value to be tested.

    Returns
    -------
    bool
        True iff ``value`` is a Boolean.

    Examples
    --------
        >>> is_bool(False)
        True
        >>> is_bool(1)
        False
    """
    return isinstance(value, bool)


def is_number(value):
    """Test whether it is a Boolean.

    Parameters
    ----------
    value : obj
        Value to be tested.

    Returns
    -------
    bool
        True iff ``value`` is a Boolean.

    Examples
    --------
        >>> is_number(3)
        True
        >>> is_number(3.)
        True
        >>> is_number('3')
        False
    """
    return isinstance(value, numbers.Number)
