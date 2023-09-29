# -*- coding: utf-8 -*-
"""
Created on 07 jul. 2022, 15:06
Copyright François Durand 2014-2022
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
from svvamp.utils import type_checker


class RuleKApproval(Rule):
    """k-Approval.

    Parameters
    ----------
    k : int
        Number of approved candidates per ballot.

    Options
    -------
        >>> RuleKApproval.print_options_parameters()
        cm_option: ['exact']. Default: 'exact'.
        icm_option: ['exact']. Default: 'exact'.
        iia_subset_maximum_size: is_number. Default: 2.
        im_option: ['exact']. Default: 'exact'.
        k: is_number. Default: 1.
        tm_option: ['exact']. Default: 'exact'.
        um_option: ['exact']. Default: 'exact'.

    Notes
    -----
    Each voter votes for `k` candidates. The candidate with most approvals is declared the winner. In case of
    a tie, the tied candidate with lowest index wins. Sincere voters vote for their `k` top candidates.

    * :meth:`is_iia`: Non-polynomial or non-exact algorithms from superclass :class:`Rule`.
    * :meth:`is_cm_`, :meth:`is_icm_`, :meth:`is_im_`, :meth:`is_tm_`, :meth:`is_um_`: Exact in polynomial time.

    Examples
    --------
        >>> profile = Profile(preferences_ut=[
        ...     [ 0. , -0.5, -1. ],
        ...     [ 1. , -1. ,  0.5],
        ...     [ 0.5,  0.5, -0.5],
        ...     [ 0.5,  0. ,  1. ],
        ...     [-1. , -1. ,  1. ],
        ... ], preferences_rk=[
        ...     [0, 1, 2],
        ...     [0, 2, 1],
        ...     [1, 0, 2],
        ...     [2, 0, 1],
        ...     [2, 1, 0],
        ... ])
        >>> rule = RuleKApproval(k=2)(profile)
        >>> rule.demo_results_(log_depth=0)  # doctest: +NORMALIZE_WHITESPACE
        <BLANKLINE>
        ************************
        *                      *
        *   Election Results   *
        *                      *
        ************************
        <BLANKLINE>
        ***************
        *   Results   *
        ***************
        profile_.preferences_ut (reminder) =
        [[ 0.  -0.5 -1. ]
         [ 1.  -1.   0.5]
         [ 0.5  0.5 -0.5]
         [ 0.5  0.   1. ]
         [-1.  -1.   1. ]]
        profile_.preferences_rk (reminder) =
        [[0 1 2]
         [0 2 1]
         [1 0 2]
         [2 0 1]
         [2 1 0]]
        ballots =
        [[ True  True False]
         [ True False  True]
         [ True  True False]
         [ True False  True]
         [False  True  True]]
        scores =
        [4 3 3]
        candidates_by_scores_best_to_worst
        [0 1 2]
        scores_best_to_worst
        [4 3 3]
        w = 0
        score_w = 4
        total_utility_w = 1.0
        <BLANKLINE>
        *********************************
        *   Condorcet efficiency (rk)   *
        *********************************
        w (reminder) = 0
        <BLANKLINE>
        condorcet_winner_rk_ctb = 0
        w_is_condorcet_winner_rk_ctb = True
        w_is_not_condorcet_winner_rk_ctb = False
        w_missed_condorcet_winner_rk_ctb = False
        <BLANKLINE>
        condorcet_winner_rk = 0
        w_is_condorcet_winner_rk = True
        w_is_not_condorcet_winner_rk = False
        w_missed_condorcet_winner_rk = False
        <BLANKLINE>
        ***************************************
        *   Condorcet efficiency (relative)   *
        ***************************************
        w (reminder) = 0
        <BLANKLINE>
        condorcet_winner_ut_rel_ctb = 0
        w_is_condorcet_winner_ut_rel_ctb = True
        w_is_not_condorcet_winner_ut_rel_ctb = False
        w_missed_condorcet_winner_ut_rel_ctb = False
        <BLANKLINE>
        condorcet_winner_ut_rel = 0
        w_is_condorcet_winner_ut_rel = True
        w_is_not_condorcet_winner_ut_rel = False
        w_missed_condorcet_winner_ut_rel = False
        <BLANKLINE>
        ***************************************
        *   Condorcet efficiency (absolute)   *
        ***************************************
        w (reminder) = 0
        <BLANKLINE>
        condorcet_admissible_candidates =
        [ True False False]
        w_is_condorcet_admissible = True
        w_is_not_condorcet_admissible = False
        w_missed_condorcet_admissible = False
        <BLANKLINE>
        weak_condorcet_winners =
        [ True False False]
        w_is_weak_condorcet_winner = True
        w_is_not_weak_condorcet_winner = False
        w_missed_weak_condorcet_winner = False
        <BLANKLINE>
        condorcet_winner_ut_abs_ctb = 0
        w_is_condorcet_winner_ut_abs_ctb = True
        w_is_not_condorcet_winner_ut_abs_ctb = False
        w_missed_condorcet_winner_ut_abs_ctb = False
        <BLANKLINE>
        condorcet_winner_ut_abs = 0
        w_is_condorcet_winner_ut_abs = True
        w_is_not_condorcet_winner_ut_abs = False
        w_missed_condorcet_winner_ut_abs = False
        <BLANKLINE>
        resistant_condorcet_winner = nan
        w_is_resistant_condorcet_winner = False
        w_is_not_resistant_condorcet_winner = True
        w_missed_resistant_condorcet_winner = False
        >>> rule.demo_manipulation_(log_depth=0)  # doctest: +NORMALIZE_WHITESPACE
        <BLANKLINE>
        *****************************
        *                           *
        *   Election Manipulation   *
        *                           *
        *****************************
        <BLANKLINE>
        *********************************************
        *   Basic properties of the voting system   *
        *********************************************
        with_two_candidates_reduces_to_plurality =  False
        is_based_on_rk =  True
        is_based_on_ut_minus1_1 =  False
        meets_iia =  False
        <BLANKLINE>
        ****************************************************
        *   Manipulation properties of the voting system   *
        ****************************************************
        Condorcet_c_ut_rel_ctb (False)     ==>     Condorcet_c_ut_rel (False)
         ||                                                               ||
         ||     Condorcet_c_rk_ctb (False) ==> Condorcet_c_rk (False)     ||
         ||           ||               ||       ||             ||         ||
         V            V                ||       ||             V          V
        Condorcet_c_ut_abs_ctb (False)     ==>     Condorcet_ut_abs_c (False)
         ||                            ||       ||                        ||
         ||                            V        V                         ||
         ||       maj_fav_c_rk_ctb (False) ==> maj_fav_c_rk (False)       ||
         ||           ||                                       ||         ||
         V            V                                        V          V
        majority_favorite_c_ut_ctb (False) ==> majority_favorite_c_ut (False)
         ||                                                               ||
         V                                                                V
        IgnMC_c_ctb (False)                ==>                IgnMC_c (False)
         ||                                                               ||
         V                                                                V
        InfMC_c_ctb (False)                ==>                InfMC_c (False)
        <BLANKLINE>
        *****************************************************
        *   Independence of Irrelevant Alternatives (IIA)   *
        *****************************************************
        w (reminder) = 0
        is_iia = True
        log_iia: iia_subset_maximum_size = 2.0
        example_winner_iia = nan
        example_subset_iia = nan
        <BLANKLINE>
        **********************
        *   c-Manipulators   *
        **********************
        w (reminder) = 0
        preferences_ut (reminder) =
        [[ 0.  -0.5 -1. ]
         [ 1.  -1.   0.5]
         [ 0.5  0.5 -0.5]
         [ 0.5  0.   1. ]
         [-1.  -1.   1. ]]
        v_wants_to_help_c =
        [[False False False]
         [False False False]
         [False False False]
         [False False  True]
         [False False  True]]
        <BLANKLINE>
        ************************************
        *   Individual Manipulation (IM)   *
        ************************************
        is_im = False
        log_im: im_option = exact
        candidates_im =
        [0. 0. 0.]
        <BLANKLINE>
        *********************************
        *   Trivial Manipulation (TM)   *
        *********************************
        is_tm = False
        log_tm: tm_option = exact
        candidates_tm =
        [0. 0. 0.]
        <BLANKLINE>
        ********************************
        *   Unison Manipulation (UM)   *
        ********************************
        is_um = False
        log_um: um_option = exact
        candidates_um =
        [0. 0. 0.]
        <BLANKLINE>
        *********************************************
        *   Ignorant-Coalition Manipulation (ICM)   *
        *********************************************
        is_icm = False
        log_icm: icm_option = exact
        candidates_icm =
        [0. 0. 0.]
        necessary_coalition_size_icm =
        [ 0. 11.  8.]
        sufficient_coalition_size_icm =
        [ 0. 11.  8.]
        <BLANKLINE>
        ***********************************
        *   Coalition Manipulation (CM)   *
        ***********************************
        is_cm = False
        log_cm: cm_option = exact
        candidates_cm =
        [0. 0. 0.]
        necessary_coalition_size_cm =
        [0. 2. 5.]
        sufficient_coalition_size_cm =
        [0. 2. 5.]
    """

    full_name = 'k-Approval'
    abbreviation = 'kAV'

    options_parameters = Rule.options_parameters.copy()
    options_parameters.update({
        'k': {'allowed': type_checker.is_number, 'default': 1},
        'im_option': {'allowed': ['exact'], 'default': 'exact'},
        'tm_option': {'allowed': ['exact'], 'default': 'exact'},
        'um_option': {'allowed': ['exact'], 'default': 'exact'},
        'icm_option': {'allowed': ['exact'], 'default': 'exact'},
        'cm_option': {'allowed': ['exact'], 'default': 'exact'}
    })

    def __init__(self, k=1, **kwargs):
        # noinspection PyTypeChecker
        self.k = None
        super().__init__(
            with_two_candidates_reduces_to_plurality=(k == 1), is_based_on_rk=True,
            precheck_um=False, precheck_icm=False, precheck_tm=False,
            log_identity="K-APPROVAL", k=k, **kwargs
        )

    @cached_property
    def ballots_(self):
        """2d array of values in {0, 1}. ``ballots_[v, c] = 1`` iff voter ``v`` votes for candidates ``c``.
        """
        self.mylog("Compute ballots", 1)
        return np.greater_equal(self.profile_.preferences_borda_rk, self.profile_.n_c - self.k)

    @cached_property
    def scores_(self):
        """1d array of integers. ``scores_[c]`` is the number of approvals for candidate ``c``.
        """
        self.mylog("Compute scores", 1)
        return np.sum(self.ballots_, axis=0)

    # %% Individual manipulation (IM)

    def _compute_im_(self, mode, c=None):
        """Compute IM: is_im, candidates_im.

        For k-Approval, since calculation is quite cheap, we calculate everything directly, even if complete_mode is
        False.

        Examples
        --------
            >>> profile = Profile(preferences_rk=[
            ...     [0, 1, 2],
            ...     [1, 0, 2],
            ...     [1, 2, 0],
            ...     [2, 0, 1],
            ...     [2, 0, 1],
            ... ])
            >>> rule = RuleKApproval(k=1)(profile)
            >>> rule.is_im_v_with_candidates_(0)
            (False, array([0., 0., 0.]))
        """
        self.mylog("Compute IM", 1)
        self._im_was_computed_with_candidates = True
        self._im_was_computed_with_voters = True
        self._im_was_computed_full = True
        n_c = self.profile_.n_c
        scores_without_v = self.scores_[np.newaxis, :] + np.array(range(n_c - 1, -1, -1)) / n_c - self.ballots_
        scores_without_v_ascending = np.sort(scores_without_v)  # sorted for each voter
        not_in_k_minus_one_weakest = (scores_without_v >= scores_without_v_ascending[:, self.k - 1][:, np.newaxis])
        scores_without_v_ascending[:, 0:self.k - 1] += 1  # Add 1 point to the `k-1` weakest candidates
        max_scores_approving_k_minus_one_weakest = np.max(scores_without_v_ascending, axis=1)
        scores_without_v_ascending[:, self.k - 1] += 1  # Add also 1 point to the `k`-th weakest candidate
        max_scores_approving_k_weakest = np.max(scores_without_v_ascending, axis=1)
        scores_without_v_plus_one = scores_without_v + 1
        self._v_im_for_c = self.v_wants_to_help_c_ & (
            (scores_without_v_plus_one >= max_scores_approving_k_weakest[:, np.newaxis])
            | (
                (scores_without_v_plus_one >= max_scores_approving_k_minus_one_weakest[:, np.newaxis])
                & not_in_k_minus_one_weakest
            )
        )
        self._candidates_im = np.any(self._v_im_for_c, 0)
        self._voters_im = np.any(self._v_im_for_c, 1)
        self._is_im = np.any(self._candidates_im)

    # %% Unison Manipulation (UM)

    def _um_main_work_c_exact_(self, c):
        n_m = self.profile_.matrix_duels_ut[c, self.w_]
        scores_from_sincere_voters = (
            (self.ballots_[np.logical_not(self.v_wants_to_help_c_[:, c]), :]).sum(axis=0)
            + np.array(range(self.profile_.n_c - 1, -1, -1)) / self.profile_.n_c
        )
        scores_other_candidates = np.sort(scores_from_sincere_voters[np.arange(self.profile_.n_c) != c])
        scores_other_candidates[0:self.k - 1] += n_m  # Give n_m points to the k-1 weakest other candidates
        self._candidates_um[c] = (
            scores_from_sincere_voters[c] + n_m > np.max(scores_other_candidates)
        )

    # %% Ignorant-Coalition Manipulation (ICM)

    def _icm_main_work_c_(self, c, optimize_bounds):
        """
            >>> profile = Profile(preferences_ut=[
            ...     [-1. ,  0. ,  0.5,  0.5,  0.5],
            ...     [ 0. , -0.5, -1. , -1. ,  0.5],
            ...     [ 0.5,  0.5, -1. ,  0. , -0.5],
            ...     [-1. , -0.5,  0.5,  1. ,  0.5],
            ...     [-1. ,  1. ,  0.5,  1. ,  0.5],
            ... ], preferences_rk=[
            ...     [2, 4, 3, 1, 0],
            ...     [4, 0, 1, 2, 3],
            ...     [1, 0, 3, 4, 2],
            ...     [3, 2, 4, 1, 0],
            ...     [1, 3, 2, 4, 0],
            ... ])
            >>> rule = RuleKApproval(k=1)(profile)
            >>> rule.candidates_icm_
            array([0., 0., 0., 0., 1.])
        """
        # Manipulators must distribute `(k - 1) n_m` approvals among the other `n_c - 1` candidates. In the best case,
        # these approvals are distributed equally (no remainder), so each other candidate has
        # `((k - 1) n_m) / (n_c - 1)` approvals from the manipulators. Then counter-manipulators will give `n_s`
        # approvals to the candidate with lowest index, and no approval to `c`. Hence to win (and considering
        # against the best case, where `c = 0`), we need:
        # `n_m >= ((k - 1) n_m) / (n_c - 1) + n_s`. After reduction, this amounts to
        # `n_m >= n_s (n_c - 1) / (n_c - k)`. But this is only a necessary condition.
        n_s = self.profile_.n_v - self.profile_.matrix_duels_ut[c, self.w_]
        n_m = n_s * (self.profile_.n_c - 1) / (self.profile_.n_c - self.k)
        while True:
            q = ((self.k - 1) * n_m) // (self.profile_.n_c - 1)
            # `q`: number of approvals given to all other candidates
            r = ((self.k - 1) * n_m) % (self.profile_.n_c - 1)
            # `r`: 1 more approval is given to the `r` weakest other candidates (= with highest index).
            if r == 0:
                if c == 0:
                    # The best other candidate is 1, with a score of `q + n_s`.
                    icm_succeeds = (n_m >= q + n_s)
                else:
                    # The best other candidate is 0, with a score of `q + n_s`.
                    icm_succeeds = (n_m > q + n_s)
            else:
                if c < self.profile_.n_c - r:
                    # The best other candidate is `n_c - r`, with a score of `q + 1 + n_s`.
                    icm_succeeds = (n_m >= q + 1 + n_s)
                else:
                    # The best other candidate has a lower index than `c` and a score of `q + 1 + n_s`.
                    icm_succeeds = (n_m > q + 1 + n_s)
            if icm_succeeds:
                self._sufficient_coalition_size_icm[c] = n_m
                self._necessary_coalition_size_icm[c] = self._sufficient_coalition_size_icm[c]
                return
            n_m += 1

    # %% Coalition Manipulation (CM)

    def _cm_main_work_c_(self, c, optimize_bounds):
        scores_from_sincere_voters = (
            (self.ballots_[np.logical_not(self.v_wants_to_help_c_[:, c]), :]).sum(axis=0)
            + np.array(range(self.profile_.n_c - 1, -1, -1)) / self.profile_.n_c
        )
        scores_other_candidates = scores_from_sincere_voters[np.arange(self.profile_.n_c) != c]
        gaps_other_candidates = np.maximum(0, np.ceil(scores_other_candidates - scores_from_sincere_voters[c]))
        p = self.profile_.n_c - self.k
        # For each manipulator, we remove 1 point from the gap of `p` other candidates.
        # In other words, we have an area looking like stairs of heights `gap[i]`, and for each manipulator, we fill in
        # `p` slots in different stairs. Necessary conditions:
        #     * n_m >= max_i(gap[i])  [Condition on the height],
        #     * n_m * p >= sum_i(gap[i])  [Condition on the area].
        # In fact, these conditions are sufficient. Indeed, consider a water-filling algorithm (we always choose the
        # highest stairs). Then after each manipulator, the condition on the area is clearly still verified. About
        # the one the height, two cases may happen (denoting `#max` the number of stairs with max height):
        #     * #max <= p: then we have filled the whole upper row, and the condition on the height still holds
        #       (one less manipulator, but one less in height).
        #     * #max > p: then we did not fill the whole upper row, i.e. the height stays as it is. However, in this
        #       case, we have n_m * p >= sum_i(gap[i]) >= #max * max_i(gap[i]) > p * max_i(gap[i]), which leads to
        #       n_m > max_i(gap[i]). With one manipulator less, the condition on the height still holds.
        self._sufficient_coalition_size_cm[c] = max(
            np.max(gaps_other_candidates),
            np.ceil(np.sum(gaps_other_candidates) / p)
        )
        self._necessary_coalition_size_cm[c] = self._sufficient_coalition_size_cm[c]
