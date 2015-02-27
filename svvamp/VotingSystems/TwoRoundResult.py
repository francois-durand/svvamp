# -*- coding: utf-8 -*-
"""
Created on Wed Oct  8 17:51:06 2014
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


class TwoRoundResult(ElectionResult):
    """Results of an election using the Two Round System.

    In case of a tie, candidates with lowest index are privileged.
    """
    
    _options_parameters = ElectionResult._options_parameters.copy()

    def __init__(self, population, **kwargs):
        super().__init__(population, **kwargs)
        self._log_identity = "TWO_ROUND_RESULT"
        self._selected_one = None
        self._selected_two = None

    def _forget_results_subclass(self):
        self._selected_one = None
        self._selected_two = None
        
    #%% Counting the ballots
        
    def _counts_ballots(self):
        self._mylog("Count ballots", 1)
        # First round
        scores_r = np.copy(self.pop.plurality_scores_rk)
        c = np.argmax(scores_r)
        self._selected_one = c
        scores_r[c] = 0
        d = np.argmax(scores_r)
        self._selected_two = d
        # Second round
        if self.pop.matrix_victories_rk_ctb[c, d]:
            self._w = c
        else:
            self._w = d

    @property
    def selected_one(self):
        """Integer. The candidate with highest score at round 1.
        """
        if self._selected_one is None:
            self._counts_ballots()
        return self._selected_one

    @property
    def selected_two(self):
        """Integer. The candidate with second highest score at round 1.
        """
        if self._selected_two is None:
            self._counts_ballots()
        return self._selected_two

    @property
    def w(self):
        """Integer (winning candidate). In case of a tie, the candidate with
        lowest index wins.
        """
        if self._w is None:
            self._counts_ballots()
        return self._w

    @property
    def ballots(self):
        """2d array of integers. ballots[v, r] is the candidate for which
        voter v votes at round r (= 0 or 1).
        """
        if self._ballots is None:
            self._ballots = np.zeros((self.pop.V, 2), dtype=np.int)
            self._ballots[:, 0] = self.pop.preferences_rk[:, 0]
            c = self.selected_one
            d = self.selected_two
            self._ballots[self.pop.preferences_borda_rk[:, c] >
                          self.pop.preferences_borda_rk[:, d], 1] = c
            self._ballots[self.pop.preferences_borda_rk[:, d] >
                          self.pop.preferences_borda_rk[:, c], 1] = d
        return self._ballots

    @property
    def scores(self):
        """2d array. scores[r, c] is the number of voters who vote for 
        candidate c at round r (= 0 or 1).
        """
        if self._scores is None:
            self._scores = np.zeros((2, self.pop.C))
            self._scores[0, :] = self.pop.plurality_scores_rk
            c = self.selected_one
            d = self.selected_two
            self._scores[1, c] = self.pop.matrix_duels_rk[c, d]
            self._scores[1, d] = self.pop.matrix_duels_rk[d, c]
        return self._scores
        
    @property
    def candidates_by_scores_best_to_worst(self):
        """1d array of integers. candidates_by_scores_best_to_worst[k] is the
        candidate with k-th best score. Finalist are sorted by their score 
        at second round. Other are sorted by their score at first round.
        
        By definition, candidates_by_scores_best_to_worst[0] = w.
        """
        if self._candidates_by_scores_best_to_worst is None:
            self._mylog("Count candidates_by_scores_best_to_worst", 1)
            self._candidates_by_scores_best_to_worst = np.argsort(
                - np.max(self.scores, 0))
        return self._candidates_by_scores_best_to_worst


if __name__ == '__main__':
    # A quick demo
    preferences_utilities = np.random.randint(-5, 5, (10, 5))
    pop = Population(preferences_utilities)
    election = TwoRoundResult(pop)
    election.demo(log_depth=3)