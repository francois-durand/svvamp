# -*- coding: utf-8 -*-
"""
Created on 11 dec. 2018, 14:06
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


class RuleVeto(Rule):
    """Veto. Also called Antiplurality.

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
        >>> rule = RuleVeto()(profile)
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
        [2 1 2 1 0]
        scores =
        [-1 -2 -2]
        candidates_by_scores_best_to_worst
        [0 1 2]
        scores_best_to_worst
        [-1 -2 -2]
        w = 0
        score_w = -1
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
        with_two_candidates_reduces_to_plurality =  True
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

    Notes
    -----
    Each voter votes against one candidate (veto). The candidate with least vetos is declared the winner. In case of
    a tie, the tied candidate with lowest index wins. Sincere voters vote against their least-liked candidate.

    * :meth:`is_iia`: Non-polynomial or non-exact algorithms from superclass :class:`Rule`.
    * :meth:`is_cm_`, :meth:`is_icm_`, :meth:`is_im_`, :meth:`is_tm_`, :meth:`is_um_`: Exact in polynomial time.
    """

    full_name = 'Veto'
    abbreviation = 'Vet'

    options_parameters = Rule.options_parameters.copy()
    options_parameters.update({
        'im_option': {'allowed': ['exact'], 'default': 'exact'},
        'tm_option': {'allowed': ['exact'], 'default': 'exact'},
        'um_option': {'allowed': ['exact'], 'default': 'exact'},
        'icm_option': {'allowed': ['exact'], 'default': 'exact'},
        'cm_option': {'allowed': ['exact'], 'default': 'exact'}
    })

    def __init__(self, **kwargs):
        super().__init__(
            with_two_candidates_reduces_to_plurality=True, is_based_on_rk=True,
            precheck_um=False, precheck_icm=False, precheck_tm=False,
            log_identity="VETO", **kwargs
        )

    @cached_property
    def ballots_(self):
        """1d array of integers. ``ballots_[v]`` is the candidate on voter ``v``'s ballot.
        """
        self.mylog("Compute ballots", 1)
        return self.profile_.preferences_rk[:, -1]

    @cached_property
    def scores_(self):
        """1d array of integers. ``scores_[c]`` is minus one times the number of vetos against candidate ``c``.
        """
        self.mylog("Compute scores", 1)
        return - np.bincount(self.ballots_, minlength=self.profile_.n_c)

    # %% Independence of Irrelevant Alternatives (IIA)

    # noinspection PyProtectedMember
    def _compute_winner_of_subset_(self, candidates_r):
        """
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
            >>> rule = RuleVeto()(profile)
            >>> rule._compute_winner_of_subset_(candidates_r=np.array([0, 1]))
            0
        """
        self.mylogv("IIA: Compute winner of subset ", candidates_r, 3)
        scores_r = - np.bincount(np.argmin(self.profile_.preferences_borda_rk[:, candidates_r], 1),
                                 minlength=candidates_r.shape[0])
        index_w_r_in_subset = np.argmax(scores_r)
        w_r = candidates_r[index_w_r_in_subset]
        self.mylogv("IIA: Winner =", w_r, 3)
        return w_r

    # %% Individual manipulation (IM)

    def _im_main_work_v_(self, v, c_is_wanted, nb_wanted_undecided, stop_if_true):
        """
            >>> profile = Profile(preferences_ut=[
            ...     [ 0. ,  0.5, -1. ],
            ...     [-1. , -1. ,  0.5],
            ...     [ 0. , -0.5,  0.5],
            ...     [ 1. ,  0.5,  1. ],
            ...     [ 0. ,  1. ,  1. ],
            ... ], preferences_rk=[
            ...     [1, 0, 2],
            ...     [2, 0, 1],
            ...     [2, 0, 1],
            ...     [2, 0, 1],
            ...     [2, 1, 0],
            ... ])
            >>> rule = RuleVeto()(profile)
            >>> rule.v_im_for_c_
            array([[0., 0., 0.],
                   [0., 0., 1.],
                   [0., 0., 1.],
                   [0., 0., 0.],
                   [0., 0., 0.]])
        """
        # If voter ``v`` strictly prefers some ``c_test`` to ``w``, let us note that she cannot have voted against
        # ``c_test``. So, the only thing she can do better is to vote against ``w`` (if it is not already the case),
        # because otherwise ``w`` will still keep a better score than ``c_test``. This strategy does not depend on
        # ``c_test``!
        scores_with_v_manip = np.copy(self.scores_)
        # Remove ``v``'s sincere vote:
        scores_with_v_manip[self.ballots_[v]] += 1
        # Vote against ``w`` instead:
        scores_with_v_manip[self.w_] -= 1
        new_winner = np.argmax(scores_with_v_manip)
        self._v_im_for_c[v, :] = False
        if self.v_wants_to_help_c_[v, new_winner]:
            self._v_im_for_c[v, new_winner] = True
            self._candidates_im[new_winner] = True
            self._voters_im[v] = True
            self._is_im = True

    # %% Trivial Manipulation (TM)

    def _tm_main_work_c_(self, c):
        # Sincere voters:
        scores_test = np.array(- np.bincount(
            self.profile_.preferences_rk[np.logical_not(self.v_wants_to_help_c_[:, c]), -1],
            minlength=self.profile_.n_c))
        # Manipulators vote against ``w``.
        # Remark: for Veto, the trivial strategy is far from the best one!
        scores_test[self.w_] -= self.profile_.matrix_duels_ut[c, self.w_]
        w_test = np.argmax(scores_test)
        self._candidates_tm[c] = (w_test == c)

    # %% Unison Manipulation (UM)
    # In their sincere ballots, manipulators were not voting against ``c``, but against ``w`` or another ``d``. If
    # now they vote against some ``d``, then ``w``'s score might get better, while ``c``'s score will not change:
    # this strategy cannot succeed. So, their only hope is to vote against ``w``. This is precisely the trivial
    # strategy!

    @cached_property
    def is_um_(self):
        return self.is_tm_

    def is_um_c_(self, c):
        return self.is_tm_c_(c)

    @cached_property
    def candidates_um_(self):
        return self.candidates_tm_

    # %% Ignorant-Coalition Manipulation (ICM)

    def _icm_main_work_c_(self, c, optimize_bounds):
        # At worst, 'sincere' manipulators may respond by all voting against ``c``. We need to give as many vetos to
        # all other candidates, which makes ``(n_c - 1) * n_s`` (where ``n_s`` = sincere voters). For each candidate
        # of index lower than ``c``, we need to give an additional veto (because of the tie-breaking rule). This
        # makes ``c`` additional manipulators needed. Hence ``sufficient_...[c] = (n_c - 1) * n_s + c``.
        n_s = self.profile_.n_v - self.profile_.matrix_duels_ut[c, self.w_]
        self._sufficient_coalition_size_icm[c] = (self.profile_.n_c - 1) * n_s + c
        self._necessary_coalition_size_icm[c] = self._sufficient_coalition_size_icm[c]

    # %% Coalition Manipulation (CM)

    def _cm_main_work_c_(self, c, optimize_bounds):
        # Sincere voters:
        scores_test = np.array(- np.bincount(
            self.profile_.preferences_rk[np.logical_not(self.v_wants_to_help_c_[:, c]), -1],
            minlength=self.profile_.n_c))
        # Make each other candidate ``d`` have a lower score than ``c``:
        # ``manipulators against d = scores_test[d] - scores_test[c] + (d < c)``
        self._sufficient_coalition_size_cm[c] = np.sum(
            np.maximum(scores_test - scores_test[c] + (np.array(range(self.profile_.n_c)) < c), 0))
        self._necessary_coalition_size_cm[c] = self._sufficient_coalition_size_cm[c]
