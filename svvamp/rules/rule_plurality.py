# -*- coding: utf-8 -*-
"""
Created on 10 dec. 2018, 15:46
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


class RulePlurality(Rule):
    """Plurality.

    Examples
    --------
        >>> import svvamp
        >>> profile = svvamp.Profile(preferences_rk=[[0, 2, 1], [0, 2, 1], [2, 0, 1], [2, 1, 0], [2, 1, 0]])
        >>> rule = svvamp.RulePlurality()(profile)
        >>> print(rule.scores_)
        [2 0 3]
        >>> print(rule.candidates_by_scores_best_to_worst_)
        [2 0 1]
        >>> rule.w_
        2

    Notes
    -----
    Each voter votes for one candidate. The candidate with most votes is declared the winner. In case of a tie,
    the tied candidate with lowest index wins.

    Sincere voters vote for their top-ranked candidate.

    * :meth:`is_iia`: Non-polynomial or non-exact algorithms from superclass :class:`Rule`.
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
            precheck_tm=False, precheck_um=False, precheck_icm=False,
            log_identity="PLURALITY", **kwargs
        )

    @cached_property
    def ballots_(self):
        """1d array of integers. ``ballots[v]`` is the candidate on voter ``v``'s ballot.
        """
        self.mylog("Compute ballots", 1)
        return self.profile_.preferences_rk[:, 0]

    @cached_property
    def scores_(self):
        """1d array of integers. ``scores[c]`` is the number of voters who vote for candidate ``c``.
        """
        self.mylog("Compute scores", 1)
        return self.profile_.plurality_scores_rk

    # %% Manipulation criteria of the voting system

    @cached_property
    def meets_majority_favorite_c_rk_ctb(self):
        return True

    # %% Independence of Irrelevant Alternatives (IIA)

    def _compute_winner_of_subset_(self, candidates_r):
        self.mylogv("IIA: Compute winner of subset ", candidates_r, 3)
        scores_r = np.bincount(np.argmax(self.profile_.preferences_borda_rk[:, candidates_r], 1),
                               minlength=candidates_r.shape[0])
        index_w_r_in_subset = np.argmax(scores_r)
        w_r = candidates_r[index_w_r_in_subset]
        self.mylogv("IIA: Winner =", w_r, 3)
        return w_r

    # %% Individual manipulation (IM)

    def _compute_im_(self, mode, c=None):
        """Compute IM: is_im, candidates_im.

        For Plurality, since calculation is quite cheap, we calculate everything directly, even if complete_mode is
        False.
        """
        self.mylog("Compute IM", 1)
        self._im_was_computed_with_candidates = True
        self._im_was_computed_with_voters = True
        self._im_was_computed_full = True
        close_races = np.zeros(self.profile_.n_c, dtype=bool)
        # Case 1 : ``w`` wins by only one ballot over ``c_test < w``.
        for c_test in range(0, self.w_):
            if self.scores_[c_test] == self.score_w_ - 1:
                close_races[c_test] = True
        # Case 2 : ``w`` is tied with ``c_test > w`` (and benefits for the tie-breaking rule).
        for c_test in range(self.w_ + 1, self.profile_.n_c):
            if self.scores_[c_test] == self.score_w_:
                close_races[c_test] = True
        # Voter ``v`` can manipulate for ``c_test`` iff:
        # * ``c_test`` is in a close race,
        # * And ``v`` wants to manipulate for ``c_test``,
        # * And ``v`` does not already vote for for ``c_test``,
        self._v_im_for_c = np.logical_and(
            close_races[np.newaxis, :],
            np.logical_and(self.v_wants_to_help_c_, self.profile_.preferences_borda_rk != self.profile_.n_c - 1))
        self._candidates_im = np.any(self._v_im_for_c, 0)
        self._voters_im = np.any(self._v_im_for_c, 1)
        self._is_im = np.any(self._candidates_im)

    def _compute_im_v_(self, v, c_is_wanted, stop_if_true):
        self._compute_im_(mode='', c=None)

    # %% Trivial Manipulation (TM)

    @cached_property
    def is_tm_(self):
        return self.is_cm_

    def is_tm_c_(self, c):
        return self.is_cm_c_(c)

    @cached_property
    def candidates_tm_(self):
        return self.candidates_cm_

    # %% Unison Manipulation (UM)

    @cached_property
    def is_um_(self):
        return self.is_cm_

    def is_um_c_(self, c):
        return self.is_cm_c_(c)

    @cached_property
    def candidates_um_(self):
        return self.candidates_cm_

    # %% Ignorant-Coalition Manipulation (ICM)

    # The voting system meets IgnMC_c_ctb: hence, general methods are exact.

    # %% Coalition Manipulation (CM)

    def _cm_main_work_c_exact_(self, c, optimize_bounds):
        scores_s = np.bincount(self.profile_.preferences_rk[np.logical_not(self.v_wants_to_help_c_[:, c]), 0],
                               minlength=self.profile_.n_c)
        # We need as many manipulators as ``scores_s[w] - scores_s[c]``, plus one if ``c > w`` (because in this case,
        #  ``c`` is disadvantaged by the tie-breaking rule). N.B.: this value cannot be negative, so it is not
        # necessary to use ``max(..., 0)``.
        self._sufficient_coalition_size_cm[c] = scores_s[self.w_] - scores_s[c] + (c > self.w_)
        self._necessary_coalition_size_cm[c] = self._sufficient_coalition_size_cm[c]


if __name__ == '__main__':
    RulePlurality()(Profile(preferences_ut=np.random.randint(-5, 5, (5, 3)))).demo_()
