# -*- coding: utf-8 -*-
"""
Created on Wed Oct  8 18:08:10 2014
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

from svvamp.VotingSystems.Election import Election
from svvamp.VotingSystems.TwoRoundResult import TwoRoundResult
from svvamp.Preferences.Population import Population


class TwoRound(TwoRoundResult, Election):
    """Two Round System.

    Inherits functions and optional parameters from superclasses
    :class:`~svvamp.ElectionResult` and :class:`~svvamp.Election`.

    :Example:

    >>> import svvamp
    >>> pop = svvamp.PopulationSpheroid(V=100, C=5)
    >>> election = svvamp.TwoRound(pop)

    Two rounds are actually held, which means that manipulators can change
    their vote between the first and second round. Hence for
    :attr:`~svvamp.Population.C` = 3, this voting system is equivalent to
    :class:`~svvamp.ExhaustiveBallot` (not :class:`~svvamp.IRV`).
    
    In case of a tie, candidates with lowest index are privileged.

    :meth:`~svvamp.Election.not_IIA`: Non-polynomial
    or non-exact algorithms from superclass :class:`~svvamp.Election`.

    :meth:`~svvamp.Election.CM`,
    :meth:`~svvamp.Election.ICM`,
    :meth:`~svvamp.Election.IM`,
    :meth:`~svvamp.Election.TM`,
    :meth:`~svvamp.Election.UM`: Exact in polynomial time.
    """
    
    _layout_name = 'Two Round'
    _options_parameters = Election._options_parameters.copy()
    _options_parameters.update(TwoRoundResult._options_parameters)
    _options_parameters['IM_option'] = {'allowed': ['exact'],
                                        'default': 'exact'}
    _options_parameters['TM_option'] = {'allowed': ['exact'],
                                        'default': 'exact'}
    _options_parameters['UM_option'] = {'allowed': ['exact'],
                                        'default': 'exact'}
    _options_parameters['ICM_option'] = {'allowed': ['exact'],
                                           'default': 'exact'}
    _options_parameters['CM_option'] = {'allowed': ['exact'],
                                        'default': 'exact'}

    def __init__(self, population, **kwargs):
        super().__init__(population, **kwargs)
        self._log_identity = "TWO_ROUND"
        self._class_result = TwoRoundResult
        self._with_two_candidates_reduces_to_plurality = True
        self._is_based_on_strict_rankings = True
        self._meets_majority_favorite_c_vtb_ctb = True
        self._precheck_UM = False
        self._precheck_TM = False
        self._precheck_ICM = False

    #%% Individual manipulation (IM)
            
    def _compute_IM(self, mode, c=None):
        """Compute IM: is_IM, candidates_IM, _voters_IM and v_IM_for_c.

        For Two Round, since calculation is not so expensive, we compute
        everything, even if complete_mode = False.
        """
        self._mylog("Compute IM", 1)
        self._IM_was_computed_with_candidates = True
        self._IM_was_computed_with_voters = True
        self._IM_was_computed_full = True
        self._v_IM_for_c = np.zeros((self.pop.V, self.pop.C))
        self._candidates_IM = np.zeros(self.pop.C)
        self._voters_IM = np.zeros(self.pop.V)
        self._is_IM = False
        # First, let us note y the opponent of w at second round (in sincere
        # voting). If the second round is held with w and y, then IM is
        # impossible. Indeed, if v prefers y to w, then she cannot do better
        # than her sincere voting.
        # At first round, when v removes her sincere vote, a candidate z,
        # either w or y, has the highest score (indeed, only one of them might
        # lose 1 point). Whatever v's strategic vote, z will be selected for
        # the second round (either as first selected or second selected). The
        # only thing v can do is choose z's opponent, another candidate
        # called c_test. For that, the best she can do is vote for this c_test.
        # If z = w, then choosing c_test = y is not interesting, because it
        # leads to a second round (w, y), not interesting as we noticed.
        # If z = y, then choosing c_test = w is not interesting for the same
        # reason.
        # So, the only votes that v should consider for first round are
        # candidates c_test that are not w or y.
        # In second round, several things can happen, like a duel between
        # y and c_test for example. Both candidates might be interesting for
        # the manipulator.
        #
        # Below, c_test is not the candidate v wants to get elected, but the
        # candidate that she wants to push to second round.
        for c_test in range(self.pop.C):
            if c_test == self.selected_one or c_test == self.selected_two:
                continue
            if (self.scores[0, self.selected_two] - 1 +
                    (self.selected_two < c_test) >
                    self.scores[0, c_test] + 1):
                # c_test cannot go to second round at all
                continue
            for v in range(self.pop.V):
                # First round
                scores_first_temp = np.copy(self.scores[0, :])
                scores_first_temp[self.ballots[v, 0]] -= 1
                scores_first_temp[c_test] += 1
                selected_one_m = np.argmax(scores_first_temp)
                scores_first_temp[selected_one_m] = -1
                selected_two_m = np.argmax(scores_first_temp)
                # Second round
                score_one_without_v = self.pop.matrix_duels_vtb[
                    selected_one_m, selected_two_m]
                score_two_without_v = self.pop.matrix_duels_vtb[
                    selected_two_m, selected_one_m]
                if (self.pop.preferences_borda_vtb[v, selected_one_m] >
                        self.pop.preferences_borda_vtb[v, selected_two_m]):
                    score_one_without_v -= 1
                else:
                    score_two_without_v -= 1
                # Conclusions
                if (self.v_wants_to_help_c[v, selected_one_m] and
                        score_one_without_v + 1 +
                        (selected_one_m < selected_two_m) >
                        score_two_without_v):
                    self._v_IM_for_c[v, selected_one_m] = True
                    self._candidates_IM[selected_one_m] = True
                    self._voters_IM[v] = True
                    self._is_IM = True
                    continue
                if (self.v_wants_to_help_c[v, selected_two_m] and
                        score_two_without_v + 1 +
                        (selected_two_m < selected_one_m) >
                        score_one_without_v):
                    self._v_IM_for_c[v, selected_two_m] = True
                    self._candidates_IM[selected_two_m] = True
                    self._voters_IM[v] = True
                    self._is_IM = True

    def _compute_IM_v(self, v, c_is_wanted, stop_if_true):
        self._compute_IM(mode='', c=None)

    #%% Trivial Manipulation (TM)

    # Use the general methods.

    #%% Unison manipulation (UM)

    def _UM_main_work_c(self, c):
        n_m = self.pop.matrix_duels.astype(int)[c, self.w]
        n_s = self.pop.V - n_m
        ballots_first_round_s = self.ballots[
            np.logical_not(self.v_wants_to_help_c[:, c]), 0]
        scores_first_round_s = np.bincount(ballots_first_round_s,
                                           minlength=self.pop.C)
        scores_temp = np.copy(scores_first_round_s)
        selected_one_s = np.argmax(scores_temp)
        scores_temp[selected_one_s] = -1
        selected_two_s = np.argmax(scores_temp)
        if c != selected_one_s:
            # Then c is either selected_two_s or another candidate.
            # After unison manipulation, selected_one_s will go to second
            # round anyway (either as first or second selected). So the only
            # interesting strategy at first round is to vote for c.
            # In that case, UM is equivalent to TM.
            # N.B.: the following 'if' is also correct when selected_two_s = c.
            if scores_first_round_s[c] + n_m < scores_first_round_s[
                    selected_two_s] + (selected_two_s < c):
                # c cannot go to second round
                self._candidates_UM[c] = False
                return
            # Second round
            d = selected_one_s
            d_vs_c = np.sum(
                self.pop.preferences_borda_vtb[
                    np.logical_not(self.v_wants_to_help_c[:, c]), d] >
                self.pop.preferences_borda_vtb[
                    np.logical_not(self.v_wants_to_help_c[:, c]), c])
            c_vs_d = n_s - d_vs_c
            if c_vs_d + n_m < d_vs_c + (d < c):
                self._candidates_UM[c] = False
                return
            else:
                self._candidates_UM[c] = True
                return
        # Now we know that c is selected_one_s. So, manipulators can try to
        # choose c's opponent for second round.
        # d will be any possible opponent of c during the second round.
        for d in range(self.pop.C):
            # First round
            # N.B.: the following 'if' is also correct when selected_two_s = d.
            if scores_first_round_s[d] + n_m < scores_first_round_s[
                    selected_two_s] + (selected_two_s < c):
                # d cannot go to second round
                continue
            # Second round
            d_vs_c = np.sum(
                self.pop.preferences_borda_vtb[
                    np.logical_not(self.v_wants_to_help_c[:, c]), d] >
                self.pop.preferences_borda_vtb[
                    np.logical_not(self.v_wants_to_help_c[:, c]), c])
            c_vs_d = n_s - d_vs_c
            if c_vs_d + n_m < d_vs_c + (d < c):
                self._candidates_UM[c] = False
                return
            else:
                self._candidates_UM[c] = True
                return

    #%% Ignorant-Coalition Manipulation (ICM)

    # Use the general methods.

    #%% Coalition Manipulation (CM)

    def _CM_main_work_c(self, c, optimize_bounds):
        n_s = self.pop.V - self.pop.matrix_duels.astype(int)[c, self.w]
        ballots_first_round_s = self.ballots[
            np.logical_not(self.v_wants_to_help_c[:, c]), 0]
        scores_first_round_s = np.bincount(ballots_first_round_s,
                                           minlength=self.pop.C)
        # d will be any possible opponent of c during the second round
        for d in range(self.pop.C):
            if d == c:
                continue
            # First round.
            if self.pop.C == 2:
                n_m_first = 0
            else:
                # Besides d and c, which candidate has the best score?
                scores_temp = np.copy(scores_first_round_s)
                scores_temp[d] = -1
                scores_temp[c] = -1
                e = np.argmax(scores_temp)
                n_m_first = (
                    max(0,
                        scores_first_round_s[e] - scores_first_round_s[c] + (
                        e < c)) +
                    max(0,
                        scores_first_round_s[e] - scores_first_round_s[d] + (
                        e < d))
                )
            # Second round.
            d_vs_c = np.sum(
                self.pop.preferences_borda_vtb[
                    np.logical_not(self.v_wants_to_help_c[:, c]), d] > 
                self.pop.preferences_borda_vtb[
                    np.logical_not(self.v_wants_to_help_c[:, c]), c])
            c_vs_d = n_s - d_vs_c
            n_m_second = max(0, d_vs_c - c_vs_d + (d < c))
            # Conclude: how many manipulators are needed?
            self._update_sufficient(
                self._sufficient_coalition_size_CM, c,
                max(n_m_first, n_m_second),
                'CM: Update sufficient_coalition_size_CM[c] ='
            )
            if self._sufficient_coalition_size_CM[c] == \
                    self._necessary_coalition_size_CM[c]:
                return
        self._necessary_coalition_size_CM[c] = (
            self._sufficient_coalition_size_CM[c])


if __name__ == '__main__':
    # A quick demo
    preferences_utilities = np.random.randint(-5, 5, (8, 4))
    pop = Population(preferences_utilities)
    election = TwoRound(pop)
    election.demo(log_depth=3)