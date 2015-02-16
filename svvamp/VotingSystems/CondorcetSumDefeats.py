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
    """'Condorcet with sum of defeats'.
    
    An 'elementary move' is reversing a voter's preference about a pair
    of candidate (c, d) (without demanding that her whole relation of
    preference stays transitive). The score for candidate c is minus the number
    of 'elementary moves' needed so that c becomes a Condorcet winner.

    It is the same principle as Dodgson's method, but without looking
    for a transitive profile.

    In practice:
    scores[c] = - sum_{c does not beat d}{floor(V/2) + 1
                                          - matrix_duels_vtb[c, d]}
    In particular, for V odd:
    scores[c] = - sum_{c does not beat d}{ceil(V/2)
                                          - matrix_duels_vtb[c, d]}
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