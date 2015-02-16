# -*- coding: utf-8 -*-
"""
Created on Sun Oct  5 22:01:40 2014
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


class NansonResult(ElectionResult):
    """Results of an election using Nanson method.

    At each round, all candidates with a Borda score strictly lower than
    average are simultaneously eliminated. When all remaining candidates have
    the same Borda score, it means that the matrix of duels (for this subset
    of candidates) has only ties. Then the candidate with lowest index is
    declared the winner.

    Since a Condorcet winner has always a Borda score higher than average,
    Nanson method meets the Condorcet criterion.
    """

    _options_parameters = ElectionResult._options_parameters.copy()

    def __init__(self, population, **kwargs):
        super().__init__(population, **kwargs)
        self._log_identity = "NANSON_RESULT"
        self._one_v_might_be_pivotal = None

    def _forget_results_subclass(self):
        self._one_v_might_be_pivotal = None

    def _count_ballots(self):
        self._scores = []
        candidates_worst_to_best = []
        self._one_v_might_be_pivotal = False
        # borda_temp[c] will be inf when c is eliminated
        borda_temp = np.copy(self.pop.borda_score_c_vtb).astype(float)
        borda_average = self.pop.V * (self.pop.C - 1) / 2
        while True:  # This is a round
            self._scores.append(np.copy(borda_temp))
            least_borda = np.min(borda_temp)
            self._mylogv("self._scores =", self._scores, 2)
#            self._mylogv("least_borda =", least_borda, 2)
            self._mylogv("borda_average =", borda_average, 2)
            if least_borda == borda_average:
                # Then all candidates have the average. This means that all
                # duels are ties. This also includes the case where only
                # one candidate is remaining.
                # We eliminate them by decreasing order of index (highest
                # indexes first).
                losing_candidates = np.where(np.isfinite(borda_temp))[0]
                candidates_worst_to_best.extend(losing_candidates[::-1]) 
                if len(losing_candidates) > 1:
                    self._one_v_might_be_pivotal = True
                    self._mylog("One voter might be pivotal", 2)
                self._mylogv("losing_candidates[::-1] =", 
                             losing_candidates[::-1], 2)
                break
            # Remove all candidates with borda score < average. Lowest Borda
            # score first, highest indexes first.
            nb_candidates_alive = np.sum(np.isfinite(borda_temp))
            # (alive at beginning of round)
            borda_for_next_round = np.copy(borda_temp).astype(float)
            borda_average_for_next_round = borda_average
            while least_borda < borda_average:
                if least_borda + (nb_candidates_alive-1) >= borda_average:
                    self._one_v_might_be_pivotal = True
                    self._mylog("One voter might be pivotal", 2)
                losing_candidates = np.where(borda_temp == least_borda)[0]
                candidates_worst_to_best.extend(losing_candidates[::-1])
                self._mylogv("losing_candidates[::-1] =", 
                             losing_candidates[::-1], 2)
#                self._mylogv("candidates_worst_to_best =", 
#                             candidates_worst_to_best, 2)
                borda_average_for_next_round -= (
                    len(losing_candidates) * (self.pop.V) / 2)
                borda_for_next_round[losing_candidates] = np.inf
                borda_for_next_round = borda_for_next_round - np.sum(
                    self.pop.matrix_duels_vtb[:, losing_candidates], 1)
                borda_temp[losing_candidates] = np.inf
                least_borda = np.min(borda_temp)
#                self._mylogv("borda_temp =", borda_temp, 2)
#                self._mylogv("least_borda =", least_borda, 2)
#                self._mylogv("borda_average =", borda_average, 2)
            if least_borda - (nb_candidates_alive-1) < borda_average:
                self._one_v_might_be_pivotal = True
                self._mylog("One voter might be pivotal", 2)
            borda_temp = borda_for_next_round
            borda_average = borda_average_for_next_round
        self._candidates_by_scores_best_to_worst = np.array(
            candidates_worst_to_best[::-1])
        self._w = self._candidates_by_scores_best_to_worst[0]
        self._scores = np.array(self._scores)

    @property
    def scores(self):
        """2d array of integers. scores[r, c] is c's Borda score at 
        elimination round r. When a candidate is eliminated, her score is 
        +inf.
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
        """1d array of integers. Candidates are sorted according to their 
        order of elimination. When several candidates are eliminated during
        the same round, they are sorted by Borda score at that round and,
        in case of a tie, by their index (highest indexes are eliminated 
        first).
        
        By definition / convention, candidates_by_scores_best_to_worst[0] = w.
        """
        if self._candidates_by_scores_best_to_worst is None:
            self._mylog("Count ballots", 1)
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
    election = NansonResult(pop)
    election.demo(log_depth=3)