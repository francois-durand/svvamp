# -*- coding: utf-8 -*-
"""
Created on Wed Oct 15 16:53:00 2014
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


class ICRVResult(ElectionResult):
    """Results of an election using Instant-Condorcet Runoff Voting (ICRV).

    Even round r (including round 0): if a candidate w has only victories
    against all other non-eliminated candidates (i.e. is a Condorcet winner
    in this subset, in the sense of matrix_victories_vtb), then w is declared
    the winner.
    Odd round r: the candidate who is ranked first (among non-eliminated
    candidates) by least voters is eliminated, like in IRV.

    See also 'Condorcet-IRV' for another Condorcet variant of IRV.
    """

    _options_parameters = ElectionResult._options_parameters.copy()

    def __init__(self, population, **kwargs):
        super().__init__(population, **kwargs)
        self._log_identity = "ICRV_RESULT"

    # %% Counting the ballots

    def _counts_ballots(self):
        self._mylog("Count ballots", 1)
        self._scores = []
        is_candidate_alive = np.ones(self.pop.C, dtype=np.bool)
        worst_to_best = []
        r = 0
        while True:
            # Here, scores_r is the number of victories (when restricting
            # to alive candidates).
            scores_r = np.full(self.pop.C, np.nan)
            scores_r[is_candidate_alive] = np.sum(
                self.pop.matrix_victories_vtb[
                    is_candidate_alive, :][:, is_candidate_alive],
                1)
            self._scores.append(scores_r)
            best_c = np.nanargmax(scores_r)
            self._mylogv("best_c =", best_c, 3)
            if scores_r[best_c] == self.pop.C - 1 - r:
                self._w = best_c
                self._mylogm("scores (before) =", self._scores, 3)
                self._scores = np.array(self._scores)
                self._mylogm("scores (after) =", self._scores, 3)
                candidates_alive = np.array(range(self.pop.C))[
                    is_candidate_alive]
                self._candidates_by_scores_best_to_worst = np.concatenate((
                    candidates_alive[np.argsort(-scores_r[is_candidate_alive],
                                                kind='mergesort')],
                    worst_to_best[::-1]
                )).astype(dtype=np.int)
                return

            # Now, scores_r is the Plurality score (IRV-style)
            scores_r = np.full(self.pop.C, np.nan)
            scores_r[is_candidate_alive] = np.sum(
                self.pop.preferences_borda_vtb[:, is_candidate_alive] ==
                np.max(self.pop.preferences_borda_vtb[:, is_candidate_alive],
                       1)[:, np.newaxis],
                0)
            self._scores.append(scores_r)
            loser = np.where(scores_r == np.nanmin(scores_r))[0][-1]
            self._mylogv("loser =", loser, 3)
            is_candidate_alive[loser] = False
            worst_to_best.append(loser)

            r += 1

    @property
    def w(self):
        """Integer (winning candidate).
        """
        if self._w is None:
            self._counts_ballots()
        return self._w

    @property
    def scores(self):
        """2d array.

        For even rounds ``r`` (including round 0), ``scores[r, c]`` is the
        number of victories for ``c`` in
        :attr:`~svvamp.Population.matrix_victories_vtb` (restricted to
        non-eliminated candidates). Ties count for 0.5.

        For odd rounds ``r``, ``scores[r, c]`` is the number of voters who
        rank ``c`` first (among non-eliminated candidates).
        """
        if self._scores is None:
            self._counts_ballots()
        return self._scores

    @property
    def candidates_by_scores_best_to_worst(self):
        """1d array of integers.

        Candidates that are not eliminated at the moment a Condorcet winner
        is detected are sorted by their number of victories in
        :attr:`~svvamp.Population.matrix_victories_vtb` (restricted to
        candidates that are not eliminated at that time).

        Other candidates are sorted in the reverse order of their IRV
        elimination.

        By definition, ``candidates_by_scores_best_to_worst[0]`` =
        :attr:`~svvamp.ICRV.w`.
        """
        if self._candidates_by_scores_best_to_worst is None:
            self._counts_ballots()
        return self._candidates_by_scores_best_to_worst

    # TODO: self._v_might_IM_for_c


if __name__ == '__main__':
    # A quick demo
    preferences_utilities = np.random.randint(-5, 5, (10, 5))
    pop = Population(preferences_utilities)
    election = ICRVResult(pop)
    election.demo(log_depth=3)