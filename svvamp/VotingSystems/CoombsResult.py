# -*- coding: utf-8 -*-
"""
Created on Fri Oct  3 11:41:14 2014
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


class CoombsResult(ElectionResult):
    """Results of an election using Coombs method.

    The candidate who is ranked last by most voters is eliminated. Then
    we iterate.

    Ties are broken in favor of lower-index candidates: in case of a tie,
    the candidate with highest index is eliminated.
    """
    
    _options_parameters = ElectionResult._options_parameters.copy()

    def __init__(self, population, **kwargs):
        super().__init__(population, **kwargs)
        self._log_identity = "COOMBS_RESULT"
        self._one_v_might_be_pivotal = None
        
    def _forget_results_subclass(self):
        self._one_v_might_be_pivotal = None

    def _count_ballots(self):
        self._scores = np.zeros((self.pop.C - 1, self.pop.C))
        is_eliminated = np.zeros(self.pop.C, dtype=np.bool)
        worst_to_best = []
        self._one_v_might_be_pivotal = False
        # preferences_borda_temp : we will put C+1 for eliminated candidates
        preferences_borda_temp = np.copy(self.pop.preferences_borda_vtb)
        for r in range(self.pop.C - 1):
            self._scores[r, :] = - np.bincount(np.argmin(
                preferences_borda_temp, 1), minlength=self.pop.C)
            self._scores[r, is_eliminated] = np.nan
            loser = np.where(
                self._scores[r, :] == np.nanmin(self._scores[r, :])
            )[0][-1]  # Tie-breaking: the last index
            is_eliminated[loser] = True
            worst_to_best.append(loser)
            preferences_borda_temp[:, loser] = self.pop.C + 1
            # Dealing with the possibility of pivotality
            # margins_r[c] = 0 : c is eliminated (i.e. c == 'loser').
            # margins_r[c] = x means that with x vetos more, c would be
            # eliminated instead of 'loser'.
            worst_score = self._scores[r, loser]
            margins_r = self._scores[r, :] - worst_score + (
                np.array(range(self.pop.C)) < loser)
            self._mylogv("margins_r =", margins_r, 3)
            self._mylogv("np.max(margins_r) =", np.max(margins_r), 3)
            if np.max(margins_r) <= 2:
                self._one_v_might_be_pivotal = True
        self._w = np.argmin(is_eliminated)
        worst_to_best.append(self._w)
        self._candidates_by_scores_best_to_worst = worst_to_best[::-1]

    @property
    def scores(self):
        """2d array of integers. scores[r, c] is minus the number of voters 
        who vote against candidate c at elimination round r.
        """
        if self._scores is None:
            self._count_ballots()
        return self._scores
        
    @property
    def w(self):
        """Integer (winning candidate).
        """
        if self._w is None:
            self._count_ballots()
        return self._w
        
    @property
    def candidates_by_scores_best_to_worst(self):
        """1d array of integers. Candidates are sorted according to their 
        order of elimination.
        
        By definition / convention, candidates_by_scores_best_to_worst[0] = w.
        """
        if self._candidates_by_scores_best_to_worst is None:
            self._count_ballots()
        return self._candidates_by_scores_best_to_worst     

    @property
    def _v_might_IM_for_c(self):
        if self._one_v_might_be_pivotal is None:
            self._count_ballots()
        return np.full((self.pop.V, self.pop.C), self._one_v_might_be_pivotal)


if __name__ == '__main__':
    # A quick demo
    preferences_utilities = np.random.randint(-5, 5, (10, 5))
    pop = Population(preferences_utilities)
    election = CoombsResult(pop)
    election.demo(log_depth=3)