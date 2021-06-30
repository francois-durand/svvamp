# -*- coding: utf-8 -*-
"""
Created on 11 dec. 2018, 13:39
Copyright François Durand 2014-2018
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
from svvamp.rules.rule import Rule
from svvamp.utils.util_cache import cached_property
from svvamp.preferences.profile import Profile


class RuleTwoRound(Rule):
    """Two Round System.

    Examples
    --------
        >>> import svvamp
        >>> profile = svvamp.Profile(preferences_rk=[[0, 2, 1], [0, 2, 1], [2, 0, 1], [2, 1, 0], [2, 1, 0]])
        >>> rule = svvamp.RuleTwoRound()(profile)
        >>> print(rule.scores_)
        [[2. 0. 3.]
         [2. 0. 3.]]
        >>> print(rule.candidates_by_scores_best_to_worst_)
        [2 0 1]
        >>> rule.w_
        2

    Notes
    -----
    Two rounds are actually held, which means that manipulators can change their vote between the first and second
    round. Hence for :attr:`n_c` = 3, this voting system is equivalent to :class:`RuleExhaustiveBallot` (not
    :class:`RuleIRV`). In case of a tie, candidates with lowest index are privileged.

    * :meth:`is_iia_`: Non-polynomial or non-exact algorithms from superclass :class:`Rule`.
    * :meth:`is_cm_`, :meth:`is_icm_`, :meth:`is_im_`, :meth:`is_tm_`, :meth:`is_um_`: Exact in polynomial time.
    """

    def __init__(self, **kwargs):
        super().__init__(
            options_parameters={
                'im_option': {'allowed': ['exact'], 'default': 'exact'},
                'tm_option': {'allowed': ['exact'], 'default': 'exact'},
                'um_option': {'allowed': ['exact'], 'default': 'exact'},
                'icm_option': {'allowed': ['exact'], 'default': 'exact'},
                'cm_option': {'allowed': ['exact'], 'default': 'exact'}
            },
            with_two_candidates_reduces_to_plurality=True, is_based_on_rk=True,
            precheck_um=False, precheck_icm=False, precheck_tm=False,
            log_identity="TWO_ROUND", **kwargs
        )

    # %% Counting the ballots

    @cached_property
    def _counts_ballots_(self):
        self.mylog("Count ballots", 1)
        # First round
        scores_r = np.copy(self.profile_.plurality_scores_rk)
        c = np.argmax(scores_r)
        selected_one = c
        scores_r[c] = 0
        d = np.argmax(scores_r)
        selected_two = d
        # Second round
        w = c if self.profile_.matrix_victories_rk_ctb[c, d] else d
        return {'w': w, 'selected_one': selected_one, 'selected_two': selected_two}

    @cached_property
    def selected_one(self):
        """Integer. The candidate with highest score at first round."""
        return self._counts_ballots_['selected_one']

    @cached_property
    def selected_two(self):
        """Integer. The candidate with second highest score at first round."""
        return self._counts_ballots_['selected_two']

    @cached_property
    def w_(self):
        return self._counts_ballots_['w']

    @cached_property
    def ballots_(self):
        """2d array of integers. ``ballots[v, r]`` is the candidate for which voter ``v`` votes at round ``r``,
        where ``r`` = 0 (first round) or ``r`` = 1 (second round).
        """
        ballots = np.zeros((self.profile_.n_v, 2), dtype=np.int)
        ballots[:, 0] = self.profile_.preferences_rk[:, 0]
        c, d = self.selected_one, self.selected_two
        ballots[self.profile_.preferences_borda_rk[:, c] > self.profile_.preferences_borda_rk[:, d], 1] = c
        ballots[self.profile_.preferences_borda_rk[:, d] > self.profile_.preferences_borda_rk[:, c], 1] = d
        return ballots

    @cached_property
    def scores_(self):
        """2d array. ``scores[r, c]`` is the number of voters who vote for candidate ``c`` at round ``r``,
        where ``r`` = 0 (first round) or ``r`` = 1 (second round).
        """
        scores = np.zeros((2, self.profile_.n_c))
        scores[0, :] = self.profile_.plurality_scores_rk
        c, d = self.selected_one, self.selected_two
        scores[1, c] = self.profile_.matrix_duels_rk[c, d]
        scores[1, d] = self.profile_.matrix_duels_rk[d, c]
        return scores

    @cached_property
    def candidates_by_scores_best_to_worst_(self):
        """1d array of integers. ``candidates_by_scores_best_to_worst[k]`` is the candidate with ``k``\ :sup:`th`
        best score. Finalists are sorted by their score at second round. Other candidates are sorted by their score
        at first round.
        """
        self.mylog("Count candidates_by_scores_best_to_worst", 1)
        return np.argsort(- np.max(self.scores_, 0))

    # %% Manipulation criteria of the voting system

    @cached_property
    def meets_majority_favorite_c_rk_ctb(self):
        return True

    # %% Individual manipulation (IM)

    def _compute_im_(self, mode, c=None):
        """Compute IM: is_im, candidates_im, _voters_im and v_im_for_c.

        For Two Round, since calculation is not so expensive, we compute everything, even if complete_mode = False.
        """
        self.mylog("Compute IM", 1)
        self._im_was_computed_with_candidates = True
        self._im_was_computed_with_voters = True
        self._im_was_computed_full = True
        self._v_im_for_c = np.zeros((self.profile_.n_v, self.profile_.n_c))
        self._candidates_im = np.zeros(self.profile_.n_c)
        self._voters_im = np.zeros(self.profile_.n_v)
        self._is_im = False
        # First, let us note ``y`` the opponent of ``w`` at second round (in sincere voting). If the second round is
        # held with ``w`` and ``y``, then IM is impossible. Indeed, if ``v`` prefers ``y`` to ``w``, then she cannot
        # do better than her sincere voting.
        #
        # At first round, when ``v`` removes her sincere vote, a candidate ``z``, either ``w`` or ``y``,
        # has the highest score (indeed, only one of them might lose 1 point). Whatever ``v``'s strategic vote,
        # ``z`` will be selected for the second round (either as first selected or second selected). The only thing
        # ``v`` can do is choose ``z``'s opponent, another candidate called ``c_test``. For that, the best she can do
        #  is vote for this ``c_test``.
        #
        #   * If ``z = w``, then choosing ``c_test = y`` is not interesting, because it leads to a second round
        #     ``(w, y)``, not interesting as we noticed.
        #   * If ``z = y``, then choosing ``c_test = w`` is not interesting for the same reason.
        #
        # So, the only votes that ``v`` should consider for first round are candidates ``c_test`` that are not ``w`` or
        # ``y``. In second round, several things can happen, like a duel between ``y`` and ``c_test`` for example. Both
        # candidates might be interesting for the manipulator.
        #
        # Below, ``c_test`` is not the candidate ``v`` wants to get elected, but the candidate that she wants to push
        # to second round.
        for c_test in range(self.profile_.n_c):
            if c_test == self.selected_one or c_test == self.selected_two:
                continue
            if self.scores_[0, self.selected_two] - 1 + (self.selected_two < c_test) > self.scores_[0, c_test] + 1:
                # c_test cannot go to second round at all
                continue
            for v in range(self.profile_.n_v):
                # First round
                scores_first_temp = np.copy(self.scores_[0, :])
                scores_first_temp[self.ballots_[v, 0]] -= 1
                scores_first_temp[c_test] += 1
                selected_one_m = np.argmax(scores_first_temp)
                scores_first_temp[selected_one_m] = -1
                selected_two_m = np.argmax(scores_first_temp)
                # Second round
                score_one_without_v = self.profile_.matrix_duels_rk[selected_one_m, selected_two_m]
                score_two_without_v = self.profile_.matrix_duels_rk[selected_two_m, selected_one_m]
                if (self.profile_.preferences_borda_rk[v, selected_one_m] >
                        self.profile_.preferences_borda_rk[v, selected_two_m]):
                    score_one_without_v -= 1
                else:
                    score_two_without_v -= 1
                # Conclusions
                if (self.v_wants_to_help_c_[v, selected_one_m]
                        and score_one_without_v + 1 + (selected_one_m < selected_two_m) > score_two_without_v):
                    self._v_im_for_c[v, selected_one_m] = True
                    self._candidates_im[selected_one_m] = True
                    self._voters_im[v] = True
                    self._is_im = True
                    continue
                if (self.v_wants_to_help_c_[v, selected_two_m]
                        and score_two_without_v + 1 + (selected_two_m < selected_one_m) > score_one_without_v):
                    self._v_im_for_c[v, selected_two_m] = True
                    self._candidates_im[selected_two_m] = True
                    self._voters_im[v] = True
                    self._is_im = True

    def _compute_im_v_(self, v, c_is_wanted, stop_if_true):
        self._compute_im_(mode='', c=None)

    # %% Trivial Manipulation (TM)

    # Use the general methods.

    # %% Unison manipulation (UM)

    def _um_main_work_c_(self, c):
        n_m = self.profile_.matrix_duels_ut.astype(int)[c, self.w_]
        n_s = self.profile_.n_v - n_m
        ballots_first_round_s = self.ballots_[np.logical_not(self.v_wants_to_help_c_[:, c]), 0]
        scores_first_round_s = np.bincount(ballots_first_round_s, minlength=self.profile_.n_c)
        scores_temp = np.copy(scores_first_round_s)
        selected_one_s = np.argmax(scores_temp)
        scores_temp[selected_one_s] = -1
        selected_two_s = np.argmax(scores_temp)
        if c != selected_one_s:
            # Then ``c`` is either ``selected_two_s`` or another candidate. After unison manipulation,
            # ``selected_one_s`` will go to second round anyway (either as first or second selected). So the only
            # interesting strategy at first round is to vote for ``c``. In that case, UM is equivalent to TM.
            # N.B.: the following 'if' is also correct when ``selected_two_s = c``.
            if scores_first_round_s[c] + n_m < scores_first_round_s[selected_two_s] + (selected_two_s < c):
                # c cannot go to second round
                self._candidates_um[c] = False
                return
            # Second round
            d = selected_one_s
            d_vs_c = np.sum(
                self.profile_.preferences_borda_rk[np.logical_not(self.v_wants_to_help_c_[:, c]), d]
                > self.profile_.preferences_borda_rk[np.logical_not(self.v_wants_to_help_c_[:, c]), c])
            c_vs_d = n_s - d_vs_c
            if c_vs_d + n_m < d_vs_c + (d < c):
                self._candidates_um[c] = False
                return
            else:
                self._candidates_um[c] = True
                return
        # Now we know that ``c`` is ``selected_one_s``. So, manipulators can try to choose ``c``'s opponent for
        # second round. ``d`` will be any possible opponent of ``c`` during the second round.
        for d in range(self.profile_.n_c):
            # First round
            # N.B.: the following 'if' is also correct when ``selected_two_s = d``.
            if scores_first_round_s[d] + n_m < scores_first_round_s[selected_two_s] + (selected_two_s < c):
                # d cannot go to second round
                continue
            # Second round
            d_vs_c = np.sum(
                self.profile_.preferences_borda_rk[np.logical_not(self.v_wants_to_help_c_[:, c]), d]
                > self.profile_.preferences_borda_rk[np.logical_not(self.v_wants_to_help_c_[:, c]), c])
            c_vs_d = n_s - d_vs_c
            if c_vs_d + n_m < d_vs_c + (d < c):
                self._candidates_um[c] = False
                return
            else:
                self._candidates_um[c] = True
                return

    # %% Ignorant-Coalition Manipulation (ICM)

    # Use the general methods.

    # %% Coalition Manipulation (CM)

    def _cm_main_work_c_(self, c, optimize_bounds):
        n_s = self.profile_.n_v - self.profile_.matrix_duels_ut.astype(int)[c, self.w_]
        ballots_first_round_s = self.ballots_[np.logical_not(self.v_wants_to_help_c_[:, c]), 0]
        scores_first_round_s = np.bincount(ballots_first_round_s, minlength=self.profile_.n_c)
        # ``d`` will be any possible opponent of ``c`` during the second round
        for d in range(self.profile_.n_c):
            if d == c:
                continue
            # First round.
            if self.profile_.n_c == 2:
                n_m_first = 0
            else:
                # Besides ``d`` and ``c``, which candidate has the best score?
                scores_temp = np.copy(scores_first_round_s)
                scores_temp[d] = -1
                scores_temp[c] = -1
                e = np.argmax(scores_temp)
                n_m_first = (max(0, np.add(scores_first_round_s[e] - scores_first_round_s[c], e < c))
                             + max(0, np.add(scores_first_round_s[e] - scores_first_round_s[d], e < d)))
            # Second round.
            d_vs_c = np.sum(
                self.profile_.preferences_borda_rk[np.logical_not(self.v_wants_to_help_c_[:, c]), d]
                > self.profile_.preferences_borda_rk[np.logical_not(self.v_wants_to_help_c_[:, c]), c])
            c_vs_d = n_s - d_vs_c
            n_m_second = max(0, d_vs_c - c_vs_d + (d < c))
            # Conclude: how many manipulators are needed?
            self._update_sufficient(self._sufficient_coalition_size_cm, c, max(n_m_first, n_m_second),
                                    'CM: Update sufficient_coalition_size_cm[c] =')
            if self._sufficient_coalition_size_cm[c] == self._necessary_coalition_size_cm[c]:
                return
        self._necessary_coalition_size_cm[c] = self._sufficient_coalition_size_cm[c]


if __name__ == '__main__':
    RuleTwoRound()(Profile(preferences_ut=np.random.randint(-5, 5, (5, 3)))).demo_()
