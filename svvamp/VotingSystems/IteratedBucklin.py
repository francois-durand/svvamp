# -*- coding: utf-8 -*-
"""
Created on Thu Oct  2 14:10:45 2014
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
from svvamp.VotingSystems.IteratedBucklinResult import IteratedBucklinResult
from svvamp.Preferences.Population import Population


class IteratedBucklin(IteratedBucklinResult, Election):
    """Iterated Bucklin method.

    Inherits functions and optional parameters from superclasses
    :class:`~svvamp.ElectionResult` and :class:`~svvamp.Election`.

    :Example:

    >>> import svvamp
    >>> pop = svvamp.PopulationSpheroid(V=100, C=5)
    >>> election = svvamp.IteratedBucklin(pop)

    The candidate with least *adjusted median Borda score* (cf. below) is
    eliminated. Then the new Borda scores are computed. Etc. Ties are broken
    in favor of lower-index candidates: in case of a tie, the candidate with
    highest index is eliminated.

    Adjusted median Borda score:

        Let ``med_c`` be the median Borda score for candidate ``c``. Let
        ``x_c`` the number of voters who put a lower Borda score to ``c``.
        Then ``c``'s adjusted median is ``med_c - x_c / (V + 1)``.

        If ``med_c > med_d``, then it is also true for the adjusted median.
        If ``med_c = med_d``, then ``c`` has a better adjusted median iff
        ``x_c < x_d``, i.e. if more voters give to ``c`` the Borda score
        ``med_c`` or higher.

        So, the best candidate by adjusted median is the
        :class:`~svvamp.Bucklin` winner. Here, at each round, we eliminate the
        candidate with lowest adjusted median Borda score, which justifies the
        name of "Iterated Bucklin method".

    Unlike Baldwin method (= Iterated Borda), Iterated Bucklin does
    not meet the Condorcet criterion. Indeed, a Condorcet winner may have the
    (strictly) worst median ranking.

    :meth:`~svvamp.Election.CM`: Non-polynomial or non-exact algorithms
    from superclass :class:`~svvamp.Election`.

    :meth:`~svvamp.Election.ICM`: The algorithm from
    superclass :class:`~svvamp.Election` is polynomial and has a window of
    error of 1 manipulator.

    :meth:`~svvamp.Election.IM`: Non-polynomial
    or non-exact algorithms from superclass :class:`~svvamp.Election`.

    :meth:`~svvamp.Election.not_IIA`: Non-polynomial
    or non-exact algorithms from superclass :class:`~svvamp.Election`.

    :meth:`~svvamp.Election.TM`: Exact in polynomial time.

    :meth:`~svvamp.Election.UM`: Non-polynomial or non-exact algorithms from
    superclass :class:`~svvamp.Election`.
    """

    _layout_name = 'Iterated Bucklin'
    _options_parameters = Election._options_parameters.copy()
    _options_parameters.update(IteratedBucklinResult._options_parameters)

    def __init__(self, population, **kwargs):
        super().__init__(population, **kwargs)
        self._log_identity = "ITERATED_BUCKLIN"
        self._class_result = IteratedBucklinResult
        self._with_two_candidates_reduces_to_plurality = True
        self._is_based_on_rk = True
        self._meets_majority_favorite_c_rk = True


if __name__ == '__main__':
    # A quick demo
    import numpy as np

    preferences_utilities = np.random.randint(-5, 5, (10, 5))
    pop = Population(preferences_utilities)
    election = IteratedBucklin(pop)
    election.demo(log_depth=3)