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
    """Object that can send simple log messages."""

    def __init__(self, log_identity='MYLOG', log_depth=0):
        self.log_depth = log_depth
        """Level of depth"""

        self.log_identity = log_identity
        """Name of the MyLog object. Will appear at the beginning of each log message."""

    def mylog(self, message="I am alive", detail_level=1):
        """Print a log message.

        Parameters
        ----------
        message : str
            The message to display.
        detail_level : int
            The level of detail of the message. The more it is, the less important is the message. Typically:
            * 1: Beginning of a method (except a simple get).
            * 2: Big steps of calculation.
            * 3: Computations inside loop (very verbose log).
            It is not recommended to use detail_level = 0 or lower.

        Examples
        --------
            >>> from svvamp.utils.my_log import MyLog
            >>> my_log_object = MyLog(log_identity="COMMENDATORE", log_depth=3)
            >>> my_log_object.mylog("Don Giovanni!", 1)
            COMMENDATORE: Don Giovanni!
        """
        if detail_level <= self.log_depth:
            print(self.log_identity + ": " + message)

    def mylogv(self, message="Variable =", variable=None, detail_level=1):
        """Print a log message with the value of a variable.

        Parameters
        ----------
        message : str
        variable : object
            Variable to be displayed.
        detail_level : int
            Cf. :meth:`mylog`.

        Examples
        --------
            >>> from svvamp.utils.my_log import MyLog
            >>> my_log_object = MyLog(log_identity="HITCHHIKER", log_depth=3)
            >>> my_log_object.mylogv("The answer is", 42)
            HITCHHIKER: The answer is 42
        """
        if detail_level <= self.log_depth:
            print(self.log_identity + ": " + message, variable)

    def mylogm(self, message="Variable =", variable=None, detail_level=1):
        """Print a log message with the value of a variable, typically a matrix.

        This method is well suited for a matrix because it skips to next line before printing the variable.

        Parameters
        ----------
        message : str
        variable : object
            Variable to be displayed.
        detail_level : int
            Cf. :meth:`mylog`.

        Examples
        --------
            >>> from svvamp.utils.my_log import MyLog
            >>> import numpy as np
            >>> my_log_object = MyLog(log_identity="MAGIC_SQUARE", log_depth=3)
            >>> my_log_object.mylogm("A nice matrix:", np.array([[2, 7, 6], [9, 5, 1], [4, 3, 8]]))
            MAGIC_SQUARE: A nice matrix:
            [[2 7 6]
             [9 5 1]
             [4 3 8]]
        """
        if detail_level <= self.log_depth:
            print(self.log_identity + ": " + message)
            print(variable)
