# -*- coding: utf-8 -*-
"""
Created on Tue Oct 14 13:45:14 2014
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
from svvamp.VotingSystems.ExhaustiveBallotResult import ExhaustiveBallotResult
from svvamp.Preferences.Population import Population


class CondorcetVtbIRVResult(ElectionResult):
    """Results of an election using Vtb-Condorcet-IRV.

    Each voter must provide a strict total order.

    If there is a Condorcet winner (in the sense of matrix_victories_rk), then
    she is elected. Otherwise, IRV is used.

    If sincere preferences are strict total orders, then this voting system is
    equivalent to CondorcetAbsIRV for sincere voting, but manipulators have
    less possibilities (they are forced to provide strict total orders).

    See also 'ICRV' for another Condorcet variant of IRV.
    """

    _options_parameters = ElectionResult._options_parameters.copy()

    def __init__(self, population, **kwargs):
        super().__init__(population, **kwargs)
        self._log_identity = "CONDORCET_VTB_IRV_RESULT"

    def _forget_results_subclass(self):
        self._v_might_be_pivotal = None

    # %% Counting the ballots

    def _counts_ballots(self):
        self._mylog("Count ballots", 1)
        if not np.isnan(self.pop.condorcet_winner_rk):
            self._w = self.pop.condorcet_winner_rk
            self._scores = np.sum(self.pop.matrix_victories_rk, 1)
            self._candidates_by_scores_best_to_worst = (
                np.argsort(-self.scores, kind='mergesort'))
            self._v_might_be_pivotal = np.zeros(self.pop.V)
            for c in np.where(self.pop.matrix_duels_rk[self.w, :] <=
                              self.pop.V / 2 + 1)[0]:
                if c == self._w:
                    continue
                # Search voters who can prevent the victory for w against c.
                self._v_might_be_pivotal[
                    self.pop.preferences_borda_rk[:, self._w] >
                    self.pop.preferences_borda_rk[:, c]
                ] = True
        else:
            self.EB = ExhaustiveBallotResult(self.pop)
            self._w = self.EB.w
            self._scores = self.EB.scores
            self._candidates_by_scores_best_to_worst = (
                self.EB.candidates_by_scores_best_to_worst)
            # First way of being (maybe) pivotal: change the result of IRV.
            self._v_might_be_pivotal = np.copy(self.EB._v_might_be_pivotal)
            # Another way of being (maybe) pivotal: create a Condorcet winner.
            for c in range(self.pop.C):
                if np.any(self.pop.matrix_duels_rk[:, c] >=
                          self.pop.V / 2 + 1):
                    # c cannot become a Condorcet winner.
                    continue
                # close_candidates are the candidates against which c does
                # not have a victory.
                close_candidates = np.less_equal(
                    self.pop.matrix_duels_rk[c, :], self.pop.V / 2)
                # Voter v can make c become a Condorcet winner iff, among
                # close_candidates, she vtb-likes c the least (this way,
                # she can improve c's scores in all these duels, compared to
                # her sincere voting).
                self._v_might_be_pivotal[
                    np.all(np.less(
                        self.pop.preferences_borda_rk[:, c][:, np.newaxis],
                        self.pop.preferences_borda_rk[:, close_candidates]
                    ), 1)
                ] = True

    @property
    def w(self):
        """Integer (winning candidate).
        """
        if self._w is None:
            self._counts_ballots()
        return self._w

    @property
    def scores(self):
        """1d or 2d array. 
        
        If there is a Condorcet winner, then ``scores[c]`` is the number of
        victories for ``c`` in :attr:`~svvamp.Population.matrix_duels_rk`.
        
        Otherwise, ``scores[r, c]`` is defined like in
        :class:`~svvamp.IRV`: it is the number of
        voters who vote for candidate ``c`` at round ``r``. For eliminated
        candidates, ``scores[r, c] = numpy.nan``. At the opposite,
        ``scores[r, c] = 0`` means that ``c`` is present at round ``r`` but no
        voter votes for ``c``.
        """
        if self._scores is None:
            self._counts_ballots()
        return self._scores

    @property
    def candidates_by_scores_best_to_worst(self):
        """1d array of integers.

        If there is a Condorcet winner, candidates are sorted according to 
        their (scalar) score.
        
        Otherwise, ``candidates_by_scores_best_to_worst`` is the
        list of all candidates in the reverse order of their IRV elimination.
        
        By definition, ``candidates_by_scores_best_to_worst[0]`` =
        :attr:`~svvamp.CondorcetVtbIRV.w`.
        """
        if self._candidates_by_scores_best_to_worst is None:
            self._counts_ballots()
        return self._candidates_by_scores_best_to_worst

    @property
    def _v_might_IM_for_c(self):
        if self._v_might_be_pivotal is None:
            self._counts_ballots()
        return np.tile(
            self._v_might_be_pivotal[:, np.newaxis],
            (1, self.pop.C))


if __name__ == '__main__':
    # A quick demo
    preferences_utilities = np.random.randint(-5, 5, (10, 5))
    pop = Population(preferences_utilities)
    election = CondorcetVtbIRVResult(pop)
    election.demo(log_depth=3)