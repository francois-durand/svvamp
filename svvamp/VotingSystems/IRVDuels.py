# -*- coding: utf-8 -*-
"""
Created on oct. 16, 2014, 11:35 
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
from svvamp.VotingSystems.IRVDuelsResult import IRVDuelsResult
from svvamp.Preferences.Population import Population


class IRVDuels(IRVDuelsResult, Election):
    """IRV with elimination duels.

    Inherits functions and optional parameters from superclasses
    :class:`~svvamp.ElectionResult` and :class:`~svvamp.Election`.

    :Example:

    >>> import svvamp
    >>> pop = svvamp.PopulationSpheroid(V=100, C=5)
    >>> election = svvamp.IRVDuels(pop)

    Principle: each round, perform a duel between the two least-favorite
    candidates and eliminate the loser of this duel.

    Even round ``r`` (including round 0): the two non-eliminated candidates
    who are ranked first (among the non-eliminated candidates) by least voters
    are selected for the elimination duels that is held in round ``r+1``.

    Odd round ``r``: voters vote for the selected candidate they like most in
    the duel. The candidate with least votes is eliminated.

    This method meets the Condorcet criterion.

    We thank Laurent Viennot for the idea of this voting system.

    :meth:`~svvamp.Election.CM`: Non-polynomial or non-exact algorithms
    from superclass :class:`~svvamp.Election`.

    :meth:`~svvamp.Election.ICM`: Exact in polynomial time.

    :meth:`~svvamp.Election.IM`: Non-polynomial
    or non-exact algorithms from superclass :class:`~svvamp.Election`.

    :meth:`~svvamp.Election.not_IIA`: Exact in polynomial time.

    :meth:`~svvamp.Election.TM`: Exact in polynomial time.

    :meth:`~svvamp.Election.UM`: Non-polynomial or non-exact algorithms from
    superclass :class:`~svvamp.Election`.

    .. seealso:: :class:`~svvamp.ExhaustiveBallot`,
                 :class:`~svvamp.IRV`,
                 :class:`~svvamp.ICRV`,
                 :class:`~svvamp.CondorcetAbsIRV`.
                 :class:`~svvamp.CondorcetVtbIRV`.
    """
    
    _layout_name = 'IRV Duels'
    _options_parameters = Election._options_parameters.copy()
    _options_parameters.update(IRVDuelsResult._options_parameters)
    _options_parameters['ICM_option'] = {'allowed': ['exact'],
                                           'default': 'exact'}

    def __init__(self, population, **kwargs):
        super().__init__(population, **kwargs)
        self._log_identity = "IRV_DUELS"
        self._class_result = IRVDuelsResult
        self._with_two_candidates_reduces_to_plurality = True
        self._is_based_on_rk = True
        self._meets_Condorcet_c_rk_ctb = True
        self._precheck_ICM = False


if __name__ == '__main__':
    # A quick demo
    import numpy as np

    preferences_utilities = np.random.randint(-5, 5, (10, 5))
    pop = Population(preferences_utilities)
    election = IRVDuels(pop)
    election.demo(log_depth=3)