# -*- coding: utf-8 -*-
"""
Created on oct. 19, 2014, 21:04 
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


class MyLog:
    """Object which can send simple log messages."""

    def __init__(self):
        self._log_depth = 0
        """Level of depth"""

        self._log_identity = 'MYLOG'
        """Name of the MyLog object. Will appear at the beginning of each
        log message.
        """

    def _mylog(self, message="I am alive", detail_level=1):
        """Print a log message

        Usage::

            >>> from svvamp.Utils.MyLog import MyLog
            >>> my_log_object = MyLog()
            >>> my_log_object._log_identity = "COMMENDATORE"
            >>> my_log_object._log_depth = 3
            >>> my_log_object._mylog("Don Giovanni!", 1)
            COMMENDATORE: Don Giovanni!

        :param message: A string, the message to display.
        :param detail_level: An integer, the level of detail of the message.
           The more it is, the less important is the message. Typically:
              * 1: Beginning of a method (except a simple get).
              * 2: Big steps of calculation.
              * 3: Computations inside loop (very verbose log).
           It is not recommended to use detail_level = 0 or lower.
        """
        if detail_level <= self._log_depth:
            print(self._log_identity + ": " + message)

    def _mylogv(self, message="Variable =", variable=None, detail_level=1):
        """Print a log message with the value of a variable

        Arguments:
        message -- String.
        variable -- Variable to be displayed.
        detail_level -- Integer. Typically:
            detail_level = 0: Should not be used
            detail_level = 1: Beginning of a method (except a simple get)
            detail_level = 2: Big steps of calculation
            detail_level = 3: Calculations inside loops (very verbose log)
        """
        if detail_level <= self._log_depth:
            print(self._log_identity + ": " + message, variable)

    def _mylogm(self, message="Variable =", variable=None, detail_level=1):
        """Print a log message with the value of a variable
        Adapted for a matrix: skip to next line before printing the variable.

        Arguments:
        message -- String.
        variable -- Variable to be displayed.
        detail_level -- Integer. Typically:
            detail_level = 0: Should not be used
            detail_level = 1: Beginning of a method (except a simple get)
            detail_level = 2: Big steps of calculation
            detail_level = 3: Calculations inside loops (very verbose log)
        """
        if detail_level <= self._log_depth:
            print(self._log_identity + ": " + message)
            print(variable)


def printm(message="Variable =", variable=None):
    """Print a message and the value of a variable
    Adapted for a matrix: skip to next line before printing the variable.

    Arguments:
    message -- String.
    variable -- Variable to be displayed.
    """
    print(message)
    print(variable)


def print_title(s):
    """Print a title

    Arguments:
    s -- String.
    """
    l = len(s)
    print('')
    print('*' * (l + 8))
    print('*   ' + s + '   *')
    print('*' * (l + 8))


def print_big_title(s):
    """Print a big title

    Arguments:
    s -- String.
    """
    l = len(s)
    print('')
    print('*' * (l + 8))
    print('*' + ' ' * (l + 6) + '*')
    print('*   ' + s + '   *')
    print('*' + ' ' * (l + 6) + '*')
    print('*' * (l + 8))


if __name__ == "__main__":
    import doctest
    import MyLog
    doctest.testmod(MyLog)
