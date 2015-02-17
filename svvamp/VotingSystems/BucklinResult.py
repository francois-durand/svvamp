# -*- coding: utf-8 -*-
"""
Created on Thu Oct  2 06:38:52 2014
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


class BucklinResult(ElectionResult):
    """Results of an election using Bucklin method.

    At counting round r, all voters who rank candidate c in r-th position
    gives her an additional vote. As soon as at least one candidate has more
    than V/2 votes (accrued with previous rounds), the candidate with most
    votes is declared the winner.

    In case of a tie, the candidate with lowest index wins.
    """

    _options_parameters = ElectionResult._options_parameters.copy()

    def __init__(self, population, **kwargs):
        super().__init__(population, **kwargs)
        self._log_identity = "BUCKLIN_RESULT"
        self._v_might_IM_for_c_store = None

    def _forget_results_subclass(self):
        self._v_might_IM_for_c_store = None

    def _count_ballots(self):
        self._mylog("Count ballots", 1)
        self._scores = np.zeros((self.pop.C, self.pop.C))
        scores_r = np.zeros(self.pop.C, dtype=np.int)
        for r in range(self.pop.C):
            scores_r += np.bincount(self.pop.preferences_ranking[:, r],
                                    minlength=self.pop.C)
            self._scores[r, :] = np.copy(scores_r)
            if self._w is None:
                w_r = np.argmax(scores_r)
                if scores_r[w_r] > self.pop.V / 2:
                    self._w = w_r
                    self._candidates_by_scores_best_to_worst = np.argsort(
                        - scores_r, kind='mergesort')

    @property
    def scores(self):
        """2d array of integers. ``scores[r, c]`` is the accrued score of
        candidate ``c`` at elimination round ``r``. It is the number of voters
        who rank ``c`` between 0\ :sup:`th` and ``r``\ :sup:`th` rank on
        their ballot.

        For information, ballot are still counted after the round where the
        decision is made (it is used for manipulation algorithms).
        """
        if self._scores is None:
            self._count_ballots()
        return self._scores
        
    @property
    def w(self):
        """Integer (winning candidate). When at least one candidate has more 
        than :attr:`~svvamp.Population.V`/2 votes, the candidate with most
        votes gets elected. In case of a tie, the candidate with highest index
        wins.
        """
        if self._w is None:
            self._count_ballots()
        return self._w
        
    @property
    def candidates_by_scores_best_to_worst(self):
        """1d array of integers. Candidates are sorted according to their
        scores during the counting round during which at least one candidate
        reaches majority.
        
        By definition, ``candidates_by_scores_best_to_worst[0]`` =
        :attr:`~svvamp.Bucklin.w`.
        """
        if self._candidates_by_scores_best_to_worst is None:
            self._count_ballots()
        return self._candidates_by_scores_best_to_worst

    @property
    def _v_might_IM_for_c(self):
        if self._v_might_IM_for_c_store is None:
            # If a voter improves c from last to first on her ballot, then 
            # c's score gain 1 point each round (except last round C-1, but
            # the decision is always done before).
            # Conversely, if a voter bury w from first to last, then w loses
            # 1 point each round (except last round C-1).
            # Modifications by one elector are bounded by that.
            # With such modifications, can a challenger c do better than w ?
            # I.e., either reach the majority before, or have more votes
            # than w when the majority is reached?
            pseudo_scores = np.copy(self.scores)
            pseudo_scores[:, np.array(range(self.pop.C)) != self.w] += 1
            pseudo_scores[:, self.w] -= 1
            self._mylogm('Pseudo-scores =', pseudo_scores)
            r = np.argmax(pseudo_scores[self.w] > self.pop.V / 2)
            c_has_a_chance = np.zeros(self.pop.C)
            for c in range(self.pop.C):
                if c == self.w:
                    continue
                if r != 0:
                    if pseudo_scores[r - 1, c] > self.pop.V / 2:
                        c_has_a_chance[c] = True
                if pseudo_scores[r, c] > pseudo_scores[r, self.w]:
                    c_has_a_chance[c] = True
            self._v_might_IM_for_c_store = np.tile(
                np.logical_not(c_has_a_chance),
                (self.pop.V, 1))
        return self._v_might_IM_for_c_store


if __name__ == '__main__':
    # A quick demo
    preferences_utilities = np.random.randint(-5, 5, (10, 5))
    pop = Population(preferences_utilities)
    election = BucklinResult(pop)
    election.demo(log_depth=3)