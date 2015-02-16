# -*- coding: utf-8 -*-
"""
Created on Thu Oct  9 09:29:01 2014
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
from svvamp.Preferences.Population import _Storage


class ExhaustiveBallotResult(ElectionResult):
    """Results of an election using Exhaustive Ballot.

    At each round, voters vote for one non-eliminated candidate. The candidate
    with least votes is eliminated. Then the next round is held. Note that
    unlike IRV, voters actually vote at each round. This does not change
    anything for sincere voting, but offers a bit more possibilities for the
    manipulators.

    In case of a tie, the candidate with highest index is eliminated.
    """
    # Exceptionally, for this voting system, results are stored in the 
    # Population object, so that they can be used by IRV.
    
    _options_parameters = ElectionResult._options_parameters.copy()

    def __init__(self, population, **kwargs):
        super().__init__(population, **kwargs)
        self._log_identity = "EXHAUSTIVE_BALLOT_RESULT"

    def _forget_results(self):
        # This is called when the population is set. If this population
        # already stores the results for EB, we must not forget them.
        if self.pop._eb_result is None:
            self.pop._eb_result = _Storage()
            ElectionResult._forget_results(self)

    def _forget_results_subclass(self):
        self._v_might_be_pivotal = None
        self._margins = None
        self._elimination_path = None

    #%% Cache variables in the Population object, so that they can be used
    #   by IRV.
    @property
    def _v_might_be_pivotal(self):
        return self.pop._eb_result._v_might_be_pivotal

    @_v_might_be_pivotal.setter
    def _v_might_be_pivotal(self, value):
        self.pop._eb_result._v_might_be_pivotal = value

    @property
    def _margins(self):
        return self.pop._eb_result._margins

    @_margins.setter
    def _margins(self, value):
        self.pop._eb_result._margins = value

    @property
    def _elimination_path(self):
        return self.pop._eb_result._elimination_path

    @_elimination_path.setter
    def _elimination_path(self, value):
        self.pop._eb_result._elimination_path = value

    @property
    def _ballots(self):
        return self.pop._eb_result._ballots

    @_ballots.setter
    def _ballots(self, value):
        self.pop._eb_result._ballots = value

    @property
    def _w(self):
        return self.pop._eb_result._w

    @_w.setter
    def _w(self, value):
        self.pop._eb_result._w = value

    @property
    def _scores(self):
        return self.pop._eb_result._scores

    @_scores.setter
    def _scores(self, value):
        self.pop._eb_result._scores = value

    @property
    def _candidates_by_scores_best_to_worst(self):
        return self.pop._eb_result._candidates_by_scores_best_to_worst

    @_candidates_by_scores_best_to_worst.setter
    def _candidates_by_scores_best_to_worst(self, value):
        self.pop._eb_result._candidates_by_scores_best_to_worst = value

    @property
    def _score_w(self):
        return self.pop._eb_result._score_w

    @_score_w.setter
    def _score_w(self, value):
        self.pop._eb_result._score_w = value

    @property
    def _scores_best_to_worst(self):
        return self.pop._eb_result._scores_best_to_worst

    @_scores_best_to_worst.setter
    def _scores_best_to_worst(self, value):
        self.pop._eb_result._scores_best_to_worst = value

    #%% Counting the ballots
        
    def _counts_ballots(self):
        candidates = np.array(range(self.pop.C))
        self._ballots = np.zeros((self.pop.V, self.pop.C - 1), dtype=np.int)
        self._scores = np.zeros((self.pop.C - 1, self.pop.C))
        self._margins = np.zeros((self.pop.C - 1, self.pop.C))
        self._v_might_be_pivotal = np.zeros(self.pop.V)
        is_alive = np.ones(self.pop.C, dtype=np.bool)
        worst_to_best = []
        for r in range(self.pop.C - 1):
            # Compute the result of Plurality voting
            self._mylogv("is_alive =", is_alive, 3)
            self._mylogv("worst_to_best =", worst_to_best, 3)
            alive_candidates = candidates[is_alive]
            self._ballots[:, r] = alive_candidates[
                np.argmax(self.pop.preferences_borda_vtb[:, is_alive], 1)
            ]
            self._mylogv("ballots[:, r] =", self._ballots[:, r], 3)
            self._scores[r, :] = np.bincount(self._ballots[:, r],
                                             minlength=self.pop.C)
            self._scores[r, np.logical_not(is_alive)] = np.nan
            # self._scores[r, is_alive] = np.sum(
            #     np.equal(
            #         self.pop.preferences_borda_vtb[is_alive],
            #         np.max(self.pop.preferences_borda_vtb[:, is_alive], 1)[
            #             :, np.newaxis]),
            #     0)
            self._mylogv("scores[r, :] =", self._scores[r, :], 3)
            # Who gets eliminated?
            loser = np.where(
                self._scores[r, :] == np.nanmin(self._scores[r, :])
            )[0][-1]  # Tie-breaking: the last index
            self._mylogv("loser =", loser, 3)
            self._margins[r, :] = (
                self._scores[r, :] -
                self._scores[r, loser] +
                (np.array(range(self.pop.C)) < loser))
            min_margin_r = np.nanmin(
                self._margins[r, self._margins[r, :] != 0])
            self._mylogv("margins[r, :] =", self._margins[r, :], 3)
            # Are there pivot voters?
            if min_margin_r == 1:
                # Any voter who has not voted for 'loser' can save her:
                # she just needs to vote for her and a candidate who had
                # a margin of 1 (or 2) will be eliminated.
                self._v_might_be_pivotal = np.logical_or(
                    self._v_might_be_pivotal,
                    self._ballots[:, r] != loser)
            elif min_margin_r == 2:
                # To change the result of the round by one voter, it is 
                # necessary and sufficient one voter who voted for a candidate 
                # with margin 2 votes for 'loser' instead.
                self._v_might_be_pivotal = np.logical_or(
                    self._v_might_be_pivotal,
                    self._margins[r, self._ballots[:, r]] == 2)
            self._mylogv("v_might_be_pivotal =",
                         self._v_might_be_pivotal, 3)
            # Update tables
            is_alive[loser] = False
            worst_to_best.append(loser)
        self._w = np.argmax(is_alive)
        worst_to_best.append(self._w)
        self._elimination_path = np.array(worst_to_best)
        self._candidates_by_scores_best_to_worst = np.array(
            worst_to_best[::-1])
                        
    @property
    def ballots(self):
        """2d array of integers. ballots[v, r] is the candidate for which
        voter v votes at round r.
        """
        if self._ballots is None:
            self._counts_ballots()
        return self._ballots

    @property
    def w(self):
        """Integer (winning candidate).
        """
        if self._w is None:
            self._counts_ballots()
        return self._w
        
    @property
    def scores(self):
        """2d array. scores[r, c] is the number of voters who vote for 
        candidate c at round r.
        
        For eliminated candidates, scores[r, c] = NaN. At the opposite, 
        scores[r, c] = 0 means that c is present at round r but no voter votes
        for c.
        """
        if self._scores is None:
            self._counts_ballots()
        return self._scores
        
    @property
    def margins(self):
        """2d array. margins[r, c] is the number of votes that c must lose
        to be eliminated at round r (all other things being equal). The 
        candidate who is eliminated at round r is the only one for which 
        margins[r, c] = 0.
        
        For eliminated candidates, margins[r, c] = NaN.
        """
        if self._scores is None:
            self._counts_ballots()
        return self._scores
        
    @property
    def candidates_by_scores_best_to_worst(self):
        """1d array of integers. candidates_by_scores_best_to_worst is the 
        list of all candidates in the reverse order of their elimination.
        
        By definition, candidates_by_scores_best_to_worst[0] = w.
        """
        if self._candidates_by_scores_best_to_worst is None:
            self._counts_ballots()
        return self._candidates_by_scores_best_to_worst  
        
    @property
    def elimination_path(self):
        """1d array of integers. Same as candidates_by_scores_best_to_worst,
        but in the reverse order.
        """
        if self._elimination_path is None:
            self._counts_ballots()
        return self._elimination_path

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
    election = ExhaustiveBallotResult(pop)
    election.demo(log_depth=3)