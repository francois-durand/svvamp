# -*- coding: utf-8 -*-
"""
Created on Wed Oct  8 09:56:59 2014
Copyright François Durand 2014, 2015
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

from svvamp.VotingSystems.Election import Election
from svvamp.VotingSystems.CondorcetSumDefeatsResult import \
    CondorcetSumDefeatsResult
from svvamp.Preferences.Population import Population


class CondorcetSumDefeats(CondorcetSumDefeatsResult, Election):
    """Condorcet with sum of defeats.

    Inherits functions and optional parameters from superclasses
    :class:`~svvamp.ElectionResult` and :class:`~svvamp.Election`.

    :Example:

    >>> import svvamp
    >>> pop = svvamp.PopulationSpheroid(V=100, C=5)
    >>> election = svvamp.CondorcetSumDefeats(pop)

    An *elementary move* consists of reversing a voter's preference about a
    pair of candidate ``(c, d)`` (without demanding that her whole relation of
    preference stays transitive). The score for candidate ``c`` is minus the
    number of *elementary moves* needed so that ``c`` becomes a Condorcet
    winner.

    It is the same principle as Dodgson's method, but without looking
    for a transitive profile.

    In practice:

    .. math::

        \\texttt{scores}[c] = - \\sum_{c \\text{ does not beat } d}\\left(
        \\left\\lfloor\\frac{V}{2}\\right\\rfloor
        + 1 - \\texttt{matrix_duels_rk}[c, d]
        \\right)

    In particular, for :attr:`~svvamp.Population.V` odd:

    .. math::

        \\texttt{scores}[c] = - \\sum_{c \\text{ does not beat } d}\\left(
        \\left\\lceil\\frac{V}{2}\\right\\rceil
        - \\texttt{matrix_duels_rk}[c, d]
        \\right)

    :meth:`~svvamp.Election.CM`: Non-polynomial or non-exact algorithms
    from superclass :class:`~svvamp.Election`.

    :meth:`~svvamp.Election.ICM`: Algorithm from superclass
    :class:`~svvamp.Election`. It is polynomial and has a window of error of 1
    manipulator.

    :meth:`~svvamp.Election.IM`: Non-polynomial
    or non-exact algorithms from superclass :class:`~svvamp.Election`.

    :meth:`~svvamp.Election.not_IIA`: Non-polynomial
    or non-exact algorithms from superclass :class:`~svvamp.Election`.
    If :attr:`~svvamp.Election.IIA_subset_maximum_size` = 2, it runs in
    polynomial time and is exact up to ties (which can occur only if
    :attr:`~svvamp.Population.V` is even).

    :meth:`~svvamp.Election.TM`: Exact in polynomial time.

    :meth:`~svvamp.Election.UM`: Non-polynomial or non-exact algorithms from
    superclass :class:`~svvamp.Election`.
    """
    
    _layout_name = 'Condorcet Sum Defeats'
    _options_parameters = Election._options_parameters.copy()
    _options_parameters.update(CondorcetSumDefeatsResult._options_parameters)

    def __init__(self, population, **kwargs):
        super().__init__(population, **kwargs)
        self._log_identity = "CONDORCET_SUM_DEFEATS"
        self._class_result = CondorcetSumDefeatsResult
        self._with_two_candidates_reduces_to_plurality = True
        self._is_based_on_strict_rankings = True
        self._meets_Condorcet_c_vtb = True
        self._meets_InfMC_c_ctb = True
        self._precheck_ICM = False


if __name__ == '__main__':
    # A quick demo
    import numpy as np

    preferences_utilities = np.random.randint(-5, 5, (10, 5))
    pop = Population(preferences_utilities)
    election = CondorcetSumDefeats(pop)
    election.demo(log_depth=3)