# -*- coding: utf-8 -*-
"""
Created on Sat Oct  4 17:28:18 2014
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
from svvamp.VotingSystems.KemenyResult import KemenyResult
from svvamp.Preferences.Population import Population


class Kemeny(KemenyResult, Election):
    """Kemeny method.
    
    Inherits functions and optional parameters from superclasses
    :class:`~svvamp.ElectionResult` and :class:`~svvamp.Election`.

    :Example:

    >>> import svvamp
    >>> pop = svvamp.PopulationSpheroid(V=100, C=5)
    >>> election = svvamp.Kemeny(pop)

    We find the order on candidates whose total Kendall tau distance to the
    voters is minimal. The top element of this order is declared the winner.
    
    In case several orders are optimal, the first one by lexicographic
    order is given. This implies that if several winners are possible,
    the one with lowest index is declared the winner.

    For this voting system, even deciding the sincere winner is NP-hard.

    :meth:`~svvamp.Election.CM`: Non-polynomial or non-exact algorithms
    from superclass :class:`~svvamp.Election`.

    :meth:`~svvamp.Election.ICM`: Exact in polynomial time
    (once the sincere winner is computed).

    :meth:`~svvamp.Election.IM`: Non-polynomial
    or non-exact algorithms from superclass :class:`~svvamp.Election`.

    :meth:`~svvamp.Election.not_IIA`: Non-polynomial
    or non-exact algorithms from superclass :class:`~svvamp.Election`.

    :meth:`~svvamp.Election.TM`: Exact in the time needed to decide the winner
    of one election, multiplied by :attr:`~svvamp.Population.C`.

    :meth:`~svvamp.Election.UM`: Non-polynomial or non-exact algorithms from
    superclass :class:`~svvamp.Election`.

    References:

        'Mathematics without numbers', J. G. Kemeny, 1959.

        'A Consistent Extension of Condorcet’s Election Principle',
        H. P. Young and A. Levenglick, 1978.

        'On the approximability of Dodgson and Young elections',
        Ioannis Caragiannis et al., 2009.

        'Comparing and aggregating partial orders with Kendall tau distances',
        Franz J. Brandenburg, Andreas Gleißner and Andreas Hofmeier, 2013.
    """
    
    _layout_name = 'Kemeny'
    _options_parameters = Election._options_parameters.copy()
    _options_parameters.update(KemenyResult._options_parameters)
    _options_parameters['ICM_option'] = {'allowed': ['exact'],
                                           'default': 'exact'}

    def __init__(self, population, **kwargs):
        super().__init__(population, **kwargs)
        self._log_identity = "KEMENY"
        self._class_result = KemenyResult
        self._with_two_candidates_reduces_to_plurality = True
        self._is_based_on_strict_rankings = True
        self._meets_Condorcet_c_vtb_ctb = True
        self._precheck_ICM = False


if __name__ == '__main__':
    # A quick demo
    import numpy as np

    preferences_utilities = np.random.randint(-5, 5, (10, 5))
    pop = Population(preferences_utilities)
    election = Kemeny(pop)
    election.demo(log_depth=3)