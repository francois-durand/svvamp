# -*- coding: utf-8 -*-
"""
Created on june 29, 2015, 08:59
Copyright François Durand 2015
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
from svvamp.VotingSystems.IRVAverageResult import IRVAverageResult
from svvamp.Preferences.Population import Population


class IRVAverage(IRVAverageResult, Election):
    """Instant Runoff Voting based on Average Score (Nanson-like).

    Inherits functions and optional parameters from superclasses
    :class:`~svvamp.ElectionResult` and :class:`~svvamp.Election`.

    :Example:

    >>> import svvamp
    >>> pop = svvamp.PopulationSpheroid(V=100, C=5)
    >>> election = svvamp.IRVAverage(pop)

    At each round, all candidates with a Plurality score strictly lower than
    average are simultaneously eliminated. When all remaining candidates have
    the same Plurality score, the candidate with lowest index is
    declared the winner.

    This method meets MajFav criterion.

    :meth:`~svvamp.Election.CM`: Non-polynomial
    or non-exact algorithms from superclass :class:`~svvamp.Election`.

    :meth:`~svvamp.Election.ICM`: Exact in polynomial time.

    :meth:`~svvamp.Election.IM`: Non-polynomial
    or non-exact algorithms from superclass :class:`~svvamp.Election`.

    :meth:`~svvamp.Election.not_IIA`: Non-polynomial
    or non-exact algorithms from superclass :class:`~svvamp.Election`.

    :meth:`~svvamp.Election.TM`: Exact in polynomial time.

    :meth:`~svvamp.Election.UM`: Non-polynomial or non-exact algorithms from
    superclass :class:`~svvamp.Election`.
    """

    _layout_name = 'IRV Average'
    _options_parameters = Election._options_parameters.copy()
    _options_parameters.update(IRVAverageResult._options_parameters)

    def __init__(self, population, **kwargs):
        super().__init__(population, **kwargs)
        self._log_identity = "IRV_AVERAGE"
        self._class_result = IRVAverageResult
        self._with_two_candidates_reduces_to_plurality = True
        self._is_based_on_rk = True
        self._meets_majority_favorite_c_rk_ctb = True
        self._precheck_ICM = True


if __name__ == '__main__':
    # A quick demo
    import numpy as np

    preferences_utilities = np.random.randint(-5, 5, (8, 4))
    pop = Population(preferences_utilities)
    election = IRVAverage(pop)
    election.CM_option = 'exact'
    election.demo(log_depth=3)