# -*- coding: utf-8 -*-
"""
Created on Wed Oct  8 12:00:02 2014
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


class SchulzeResult(ElectionResult):
    """Results of an election using Schulze Method.

    scores[c, d] is equal to the width of the widest path from c to d (in the
    capacited graph defined by matrix_duels_vtb).
    We say that c is better than d if scores[c, d] > scores[d, c], or if there
    is an equality and c < d. The candidate who is "better" than all the
    others is declared the winner.
    """

    _options_parameters = ElectionResult._options_parameters.copy()

    def __init__(self, population, **kwargs):
        super().__init__(population, **kwargs)
        self._log_identity = "SCHULZE_RESULT"
        
    #%% Counting the ballots

    @staticmethod
    def _count_ballot_aux(matrix_duels):
        C = matrix_duels.shape[0]
        # Calculate widest path for candidate c to d
        widest_path = np.copy(matrix_duels)
        for i in range(C):
            for j in range(C):
                if i == j:
                    continue
                for k in range(C):
                    if k == i or k == j:
                        continue
                    widest_path[j, k] = max(
                        widest_path[j, k],
                        min(widest_path[j, i], widest_path[i, k]))
        # c is better than d if widest_path[c, d] > widest_path[d, c], or
        # if there is equality and c < d.
        # Potential winners: maximal better_or_tie[c].
        # Tie-break: index of the candidate.
        # N.B.: if we broke ties with better_c = sum(... > ...),
        # it would not meet condorcet_c_vtb_ctb.
        better_or_tie_c = np.sum(widest_path >= widest_path.T, 1)
        candidates_by_scores_best_to_worst = np.argsort(- better_or_tie_c,
                                                        kind='mergesort')
        w = candidates_by_scores_best_to_worst[0]
        return w, widest_path, candidates_by_scores_best_to_worst

    def _count_ballots(self):
        self._mylog("Count ballots", 1)
        self._w, self._scores, self._candidates_by_scores_best_to_worst = (
            SchulzeResult._count_ballot_aux(self.pop.matrix_duels_vtb))
            
    @property
    def w(self):
        """Integer (winning candidate).
        """
        if self._w is None:
            self._count_ballots()
        return self._w

    @property
    def candidates_by_scores_best_to_worst(self):
        """1d array of integers. ``candidates_by_scores_best_to_worst[k]`` is
        the ``k``\ :sup:`th` candidate by number of Schulze-victories, i.e.
        the number of candidates ``d`` such that ``c`` is *better* than ``d``.
        
        By definition, ``candidates_by_scores_best_to_worst[0]`` =
        :attr:`~svvamp.Schulze.w`.
        """
        if self._candidates_by_scores_best_to_worst is None:
            self._count_ballots()
        return self._candidates_by_scores_best_to_worst     

    @property
    def scores(self):
        """2d array of integers. ``scores[c, d]`` is equal to the width of the
        widest path from ``c`` to ``d``.

        .. note::

            Unlike for most other voting systems, ``scores`` matrix must be
            read in rows, in order to comply with our convention for the
            matrix of duels: ``c``'s score vector is ``scores[c, :]``.
        """
        if self._scores is None:
            self._count_ballots()
        return self._scores
        
    @property
    def score_w(self):
        """1d array. ``score_w`` is :attr:`~svvamp.RankedPairs.w`'s score
        vector: ``score_w`` =
        :attr:`~svvamp.RankedPairs.scores`\ ``[``:attr:`~svvamp.RankedPairs.w`\ ``, :]``.
        """
        if self._score_w is None:
            self._mylog("Compute winner's score", 1)
            self._score_w = self.scores[self.w, :]
        return self._score_w

    @property 
    def scores_best_to_worst(self):
        """2d array. ``scores_best_to_worst`` is the scores of the candidates,
        from the winner to the last candidate of the election.
        
        ``scores_best_to_worst[k, j]`` is the width of the widest path from the
        ``k``\ :sup:`th` best candidate of the election to the
        ``j``\ :sup:`th`.
        """
        if self._scores_best_to_worst is None:
            self._mylog("Compute scores_best_to_worst", 1)
            self._scores_best_to_worst = self.scores[ 
                self.candidates_by_scores_best_to_worst, :][
                    :, self.candidates_by_scores_best_to_worst]
        return self._scores_best_to_worst


if __name__ == '__main__':
    # A quick demo
    preferences_utilities = np.random.randint(-5, 5, (10, 5))
    pop = Population(preferences_utilities)
    election = SchulzeResult(pop)
    election.demo(log_depth=3)