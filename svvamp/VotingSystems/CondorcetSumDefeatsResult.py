# -*- coding: utf-8 -*-
"""
Created on Tue Oct  7 18:14:43 2014
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

import numpy as np

from svvamp.VotingSystems.ElectionResult import ElectionResult
from svvamp.Preferences.Population import Population


class CondorcetSumDefeatsResult(ElectionResult):
    """Results of an election using 'Condorcet with sum of defeats'.

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

    _options_parameters = ElectionResult._options_parameters.copy()

    def __init__(self, population, **kwargs):
        super().__init__(population, **kwargs)
        self._log_identity = "CONDORCET_SUM_DEFEATS_RESULT"
        
    #%% Counting the ballots

    @property
    def scores(self):
        """1d array of integers.

        .. math::

            \\texttt{scores}[c] = - \\sum_{c \\text{ does not beat } d}\\left(
            \\left\\lfloor\\frac{V}{2}\\right\\rfloor
            + 1 - \\texttt{matrix_duels_vtb}[c, d]
            \\right)
        """
        if self._scores is None:
            self._mylog("Compute scores", 1)
            self._scores = np.zeros(self.pop.C)
            for c in range(self.pop.C):
                self._scores[c] = - np.sum(
                    np.floor(self.pop.V / 2) + 1
                    - self.pop.matrix_duels_vtb[
                        c, self.pop.matrix_victories_vtb[:, c] > 0]
                )
        return self._scores


if __name__ == '__main__':
    # A quick demo
    preferences_utilities = np.random.randint(-5, 5, (10, 5))
    pop = Population(preferences_utilities)
    election = CondorcetSumDefeatsResult(pop)
    election.demo(log_depth=3)