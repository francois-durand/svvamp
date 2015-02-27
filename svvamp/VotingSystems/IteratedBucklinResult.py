# -*- coding: utf-8 -*-
"""
Created on Thu Oct  2 13:47:07 2014
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


class IteratedBucklinResult(ElectionResult):
    """Results of an election using Iterated Bucklin method.

    The candidate with least "adjusted median Borda score" (cf. below) is
    eliminated. Then the new Borda scores are computed. Etc.

    Ties are broken in favor of lower-index candidates: in case of a tie,
    the candidate with highest index is eliminated.

    Adjusted median Borda score:
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    Let med_c be the median Borda score for candidate c. Let x_c
    the number of voters who put a lower Borda score to c. Then c's
    adjusted median is med_c - x_c / (V + 1).

    If med_c > med_d, then it is also true for the adjusted median.
    If med_c = med_d, then c has a better adjusted median iff x_c < x_d,
    i.e. if more voters give to c the Borda score med_c or higher.

    So, the best candidate by adjusted median is the Bucklin winner. Here,
    at each round, we eliminate the candidate with lowest adjusted median
    Borda score, which justifies the name of "Iterated Bucklin method".
    """
    
    _options_parameters = ElectionResult._options_parameters.copy()

    def __init__(self, population, **kwargs):
        super().__init__(population, **kwargs)
        self._log_identity = "ITERATED_BUCKLIN_RESULT"
        self._one_v_might_be_pivotal = None

    def _forget_results_subclass(self):
        self._one_v_might_be_pivotal = None

    def _count_ballots(self):
        self._mylog("Count ballots", 1)
        self._scores = np.zeros((self.pop.C - 1, self.pop.C))
        worst_to_best = []
        is_eliminated = np.zeros(self.pop.C, dtype=np.bool)
        preferences_borda_temp = np.copy(self.pop.preferences_borda_rk)
        for r in range(self.pop.C - 1):
            for c in range(self.pop.C):
                if is_eliminated[c]:
                    self._scores[r, c] = np.inf
                    continue
                median = np.median(preferences_borda_temp[:, c])
                x = np.sum(np.less(preferences_borda_temp[:, c], median))
                self._scores[r, c] = median - x / (self.pop.V + 1)
            loser = np.where(
                self._scores[r, :] == np.min(self._scores[r, :])
            )[0][-1]  # Tie-breaking: last index
            is_eliminated[loser] = True
            worst_to_best.append(loser)
            preferences_borda_temp[np.less(
                preferences_borda_temp,
                preferences_borda_temp[:, loser][:, np.newaxis])
            ] += 1
        self._w = np.argmin(is_eliminated)
        worst_to_best.append(self._w)
        self._candidates_by_scores_best_to_worst = worst_to_best[::-1]

    @property
    def scores(self):
        """2d array of integers. ``scores[r, c]`` is the adjusted median Borda
        score of candidate ``c`` at elimination round ``r``.
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
        
        By definition / convention, ``candidates_by_scores_best_to_worst[0]`` =
        :attr:`~svvamp.IteratedBucklin.w`.
        """
        if self._candidates_by_scores_best_to_worst is None:
            self._count_ballots()
        return self._candidates_by_scores_best_to_worst     

    @property
    def _v_might_IM_for_c(self):
        if self._one_v_might_be_pivotal is None:
            scores_r = np.zeros(self.pop.C)
            is_eliminated = np.zeros(self.pop.C, dtype=np.bool)
            preferences_borda_temp = np.copy(self.pop.preferences_borda_rk)
            for r in range(self.pop.C - 1):
                self._mylogv("Pre-testing pivotality: round r =", r, 3)                
                loser_sincere = (
                    self.candidates_by_scores_best_to_worst[-1 - r])
                self._mylogv("Pre-testing pivotality: loser_sincere =", 
                             loser_sincere, 3)
                # At worst, one manipulator can put one c from top to bottom
                # or from bottom to top. Can this change the loser of this
                # round?
                for c in range(self.pop.C):
                    if is_eliminated[c]:
                        scores_r[c] = np.inf
                        continue
                    if c == loser_sincere:
                        median = np.median(np.concatenate((
                            preferences_borda_temp[:, c],
                            [self.pop.C, self.pop.C]
                        )))
                        x = sum(preferences_borda_temp[:, c] < median) - 1
                    else:
                        median = np.median(np.concatenate((
                            preferences_borda_temp[:, c],
                            [1, 1]
                        )))
                        x = sum(preferences_borda_temp[:, c] < median) + 1
                    scores_r[c] = median - x / (self.pop.V + 1)
                loser = np.where(scores_r == scores_r.min())[0][-1]
                self._mylogv("Pre-testing pivotality: loser =", loser, 3)                
                if loser != loser_sincere:
                    self._mylog("There might be a pivotal voter", 3)                
                    self._one_v_might_be_pivotal = True
                    break
                is_eliminated[loser] = True
                preferences_borda_temp[np.less(
                    preferences_borda_temp,
                    preferences_borda_temp[:, loser][:, np.newaxis])
                ] += 1
            else:
                self._mylog("There is no pivotal voter", 3)                
                self._one_v_might_be_pivotal = False
        return np.full((self.pop.V, self.pop.C), self._one_v_might_be_pivotal)


if __name__ == '__main__':
    # A quick demo
    preferences_utilities = np.random.randint(-5, 5, (10, 5))
    pop = Population(preferences_utilities)
    election = IteratedBucklinResult(pop)
    election.demo(log_depth=3)