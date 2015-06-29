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

import numpy as np

from svvamp.VotingSystems.ElectionResult import ElectionResult
from svvamp.Preferences.Population import Population


class KimRoushResult(ElectionResult):
    """Results of an election using Kim-Roush method.

    At each round, all candidates with a Veto score strictly lower than
    average are simultaneously eliminated. When all remaining candidates have
    the same Veto score, the candidate with lowest index is declared the
    winner.

    Kim-Roush method does not meets InfMC.
    """

    _options_parameters = ElectionResult._options_parameters.copy()

    def __init__(self, population, **kwargs):
        super().__init__(population, **kwargs)
        self._log_identity = "KIMROUSH_RESULT"
        self._one_v_might_be_pivotal = None

    def _forget_results_subclass(self):
        self._one_v_might_be_pivotal = None

    def _count_ballots(self):
        self._scores = []
        candidates_worst_to_best = []
        self._one_v_might_be_pivotal = False
        # score_temp[c] = - number of vetos
        # score_temp[c] will be inf when c is eliminated
        score_temp = - np.bincount(self.pop.preferences_rk[:, self.pop.C - 1],
                                   minlength=self.pop.C).astype(float)
        score_average = - self.pop.V / self.pop.C
        candidates = np.array(range(self.pop.C))
        is_alive = np.ones(self.pop.C, dtype=np.bool)
        while True:  # This is a round
            self._scores.append(np.copy(score_temp))
            least_score = np.min(score_temp)
            self._mylogv("self._scores =", self._scores, 2)
            # self._mylogv("least_score =", least_score, 2)
            self._mylogv("score_average =", score_average, 2)
            if least_score == score_average:
                # Then all candidates have the average. This also includes the
                # case where only one candidate is remaining.
                # We eliminate them by decreasing order of index (highest
                # indexes first).
                losing_candidates = np.where(np.isfinite(score_temp))[0]
                candidates_worst_to_best.extend(losing_candidates[::-1])
                if len(losing_candidates) > 1:
                    self._one_v_might_be_pivotal = True
                    self._mylog("One voter might be pivotal", 2)
                self._mylogv("losing_candidates[::-1] =",
                             losing_candidates[::-1], 2)
                break
            # Remove all candidates with score < average. Lowest
            # score first, highest indexes first.
            while least_score < score_average:
                if least_score + 1 >= score_average:
                    self._one_v_might_be_pivotal = True
                    self._mylog("One voter might be pivotal", 2)
                losing_candidates = np.where(score_temp == least_score)[0]
                candidates_worst_to_best.extend(losing_candidates[::-1])
                self._mylogv("losing_candidates[::-1] =",
                             losing_candidates[::-1], 2)
                is_alive[losing_candidates] = False
                score_temp[losing_candidates] = np.inf
                least_score = np.min(score_temp)
            if least_score - 1 < score_average:
                self._one_v_might_be_pivotal = True
                self._mylog("One voter might be pivotal", 2)
            candidates_alive = candidates[is_alive]
            score_temp = - np.bincount(
                candidates_alive[np.argmin(
                    self.pop.preferences_borda_rk[:, is_alive], 1)],
                minlength=self.pop.C).astype(float)
            score_temp[np.logical_not(is_alive)] = np.inf
            score_average = - self.pop.V / np.sum(is_alive)
        self._candidates_by_scores_best_to_worst = np.array(
            candidates_worst_to_best[::-1])
        self._w = self._candidates_by_scores_best_to_worst[0]
        self._scores = np.array(self._scores)

    @property
    def scores(self):
        """2d array of integers. ``scores[r, c]`` is minus the Veto score of
        candidate ``c`` at elimination round ``r``.

        By convention, if candidate ``c`` does not participate to round ``r``,
        then ``scores[r, c] = numpy.inf``.
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
        the same round, they are sorted by Veto score at that round (more
        vetos are eliminated first) and, in case of a tie, by their index (
        highest indexes are eliminated first).
        
        By definition / convention,
        ``candidates_by_scores_best_to_worst[0]`` = :attr:`~svvamp.KimRoush.w`.
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
    election = KimRoushResult(pop)
    election.demo(log_depth=3)