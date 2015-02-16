# -*- coding: utf-8 -*-
"""
Created on oct. 16, 2014, 11:08 
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


class IRVDuelsResult(ElectionResult):
    """Results of an election using IRV with elimination duels.

    Even round r (including round 0): the two non-eliminated candidates
    who are ranked first (among the non-eliminated candidates) by least voters
    are selected for the elimination duels that is held in round r+1.
    Odd round r: voters vote for the selected candidate they like most in the
    duel. The candidate with least votes is eliminated.

    Obviously, this method meets the Condorcet criterion.

    Our thanks to Laurent Viennot for the idea of voting system.

    See also 'Condorcet-IRV' and 'ICRV' for other Condorcet variants of IRV.
    """

    _options_parameters = ElectionResult._options_parameters.copy()

    def __init__(self, population, **kwargs):
        super().__init__(population, **kwargs)
        self._log_identity = "IRV_DUELS_RESULT"

    # %% Counting the ballots

    def _counts_ballots(self):
        self._mylog("Count ballots", 1)
        self._scores = np.full((2 * self.pop.C - 2, self.pop.C), np.nan)
        is_candidate_alive = np.ones(self.pop.C, dtype=np.bool)
        worst_to_best = []
        for r in range(self.pop.C - 1):
            # Select the two worst scores
            self._scores[2*r, is_candidate_alive] = np.sum(
                self.pop.preferences_borda_vtb[:, is_candidate_alive] ==
                np.max(self.pop.preferences_borda_vtb[:, is_candidate_alive],
                       1)[:, np.newaxis],
                0)
            selected_one = np.where(
                self._scores[2 * r, :] == np.nanmin(self._scores[2 * r, :])
            )[0][-1]
            score_selected_one = self._scores[2*r, selected_one]
            self._scores[2*r, selected_one] = np.inf
            selected_two = np.where(
                self._scores[2 * r, :] == np.nanmin(self._scores[2 * r, :])
            )[0][-1]
            self._scores[2*r, selected_one] = score_selected_one
            self._mylogv("selected_one =", selected_one, 3)
            self._mylogv("selected_two =", selected_two, 3)
            # Do the duel
            self._scores[2*r + 1, selected_one] = self.pop.matrix_duels_vtb[
                selected_one, selected_two]
            self._scores[2*r + 1, selected_two] = self.pop.matrix_duels_vtb[
                selected_two, selected_one]
            if self.pop.matrix_victories_vtb_ctb[
                    selected_one, selected_two] == 1:
                loser = selected_two
            else:
                loser = selected_one
            is_candidate_alive[loser] = False
            worst_to_best.append(loser)
        self._w = np.argmax(is_candidate_alive)
        worst_to_best.append(self._w)
        self._candidates_by_scores_best_to_worst = np.array(
            worst_to_best[::-1])

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
        For even rounds r (including round 0), scores[r, c] is the number of
        voters who rank c first (among non-eliminated candidates).
        For odd rounds r, only the two candidates who are selected for the
        elimination duels get scores. scores[r, c] is the number of voters
        who vote for c in this elimination duel.
        """
        if self._scores is None:
            self._counts_ballots()
        return self._scores

    @property
    def candidates_by_scores_best_to_worst(self):
        """1d array of integers.
        Candidates are sorted in the reverse order of their elimination.

        By definition, candidates_by_scores_best_to_worst[0] = w.
        """
        if self._candidates_by_scores_best_to_worst is None:
            self._counts_ballots()
        return self._candidates_by_scores_best_to_worst

    # TODO: self.v_might_IM_for_c


if __name__ == '__main__':
    # A quick demo
    preferences_utilities = np.random.randint(-5, 5, (10, 5))
    pop = Population(preferences_utilities)
    election = IRVDuelsResult(pop)
    election.demo(log_depth=3)