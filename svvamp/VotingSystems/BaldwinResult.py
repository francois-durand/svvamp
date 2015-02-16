# -*- coding: utf-8 -*-
"""
Created on Tue Sep 30 09:55:52 2014
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


class BaldwinResult(ElectionResult):
    """Baldwin method.

    The candidate with least Borda score is eliminated. Then the new
    Borda scores are computed. Etc.

    Ties are broken in favor of lower-index candidates: in case of a tie,
    the candidate with highest index is eliminated.

    This method meets the Condorcet criterion.
    """

    _options_parameters = ElectionResult._options_parameters.copy()

    def __init__(self, population, **kwargs):
        super().__init__(population, **kwargs)
        self._log_identity = "BALDWIN_RESULT"
    
    def _count_ballots(self):
        self._scores = np.zeros((self.pop.C - 1, self.pop.C))
        worst_to_best = []
        preferences_borda_temp = self.pop.preferences_borda_vtb.astype(float)
        for r in range(self.pop.C - 1):  # Round r
            self._scores[r, :] = np.sum(preferences_borda_temp, 0)
            loser = np.where(
                self._scores[r, :] == np.min(self._scores[r, :])
            )[0][-1]  # Tie-breaking: higher index
            worst_to_best.append(loser)
            if r == self.pop.C - 2:
                preferences_borda_temp[0, loser] = np.inf
                self._w = np.argmin(preferences_borda_temp[0, :])
                worst_to_best.append(self._w)
                break
            # Prepare for next round
            # If v was ranking c above loser, then c loses 1 point:
            preferences_borda_temp[
                preferences_borda_temp >
                preferences_borda_temp[:, loser][:, np.newaxis]
            ] -= 1
            preferences_borda_temp[:, loser] = np.inf
        self._candidates_by_scores_best_to_worst = worst_to_best[::-1]

    @property
    def scores(self):
        """2d array of integers. scores[r, c] is the Borda score of candidate
        c at elimination round r.
        
        By convention, if candidate c does not participate to round r, then
        scores[r, c] = Inf.
        """
        if self._scores is None:
            self._mylog("Count ballots", 1)
            self._count_ballots()
        return self._scores
        
    @property
    def w(self):
        """Integer (winning candidate).
        """
        if self._w is None:
            self._mylog("Count ballots", 1)
            self._count_ballots()
        return self._w
        
    @property
    def candidates_by_scores_best_to_worst(self):
        """1d array of integers. candidates_by_scores_best_to_worst[-r] is the
        candidate eliminated at elimination round r.
        
        By definition / convention, candidates_by_scores_best_to_worst[0] = w.
        """
        if self._candidates_by_scores_best_to_worst is None:
            self._mylog("Count ballots", 1)
            self._count_ballots()
        return self._candidates_by_scores_best_to_worst     


if __name__ == '__main__':
    # A quick demo
    preferences_utilities = np.random.randint(-5, 5, (10, 5))
    pop = Population(preferences_utilities)
    election = BaldwinResult(pop)
    election.demo(log_depth=3)