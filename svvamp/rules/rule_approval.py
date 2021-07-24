# -*- coding: utf-8 -*-
"""
Created on 30 nov. 2018, 09:25
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
from svvamp.preferences.profile import Profile
from svvamp.rules.rule import Rule
from svvamp.utils import type_checker
from svvamp.utils.util_cache import cached_property


class RuleApproval(Rule):
    """Approval voting.

    Parameters
    ----------
    approval_comparator : str
        Can be ``'>'`` (default) or ``'>='``. When ``approval_comparator`` is
        ``'>'``, sincere voter ``v`` approves of candidates ``c`` iff :attr:`~svvamp.Profile.preferences_ut`\ ``[
        v, c]`` > :attr:`approval_threshold`. When ``approval_comparator`` is ``'>='``, previous relation is modified
        accordingly.
    approval_threshold : number
        Number (default 0). Utility above which a sincere voter approves of a candidate.

    Notes
    -----
    Each voter may vote for any number of candidates. The candidate with most votes is declared the winner. In case
    of a tie, the tied candidate with lowest index wins.

    :meth:`is_iia`: With our assumptions, Approval voting always meets IIA.

    :meth:`is_cm_`, :meth:`is_icm_`, :meth:`is_im_`, :meth:`is_tm_`, :meth:`is_um_`: Exact in polynomial time.

    References
    ----------
    'Approval voting', Steven Brams and Peter Fishburn. In: American Political Science Review 72 (3 1978), pp. 831–847.

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
        >>> rule = RuleApproval()(profile)
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
        [[False False False]
         [ True False  True]
         [ True  True False]
         [ True False  True]
         [False False  True]]
        scores =
        [3 1 3]
        candidates_by_scores_best_to_worst
        [0 2 1]
        scores_best_to_worst
        [3 3 1]
        w = 0
        score_w = 3
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
        is_based_on_rk =  False
        is_based_on_ut_minus1_1 =  True
        meets_iia =  True
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
        IgnMC_c_ctb (True)                 ==>                IgnMC_c (True)
         ||                                                               ||
         V                                                                V
        InfMC_c_ctb (True)                 ==>                InfMC_c (True)
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
        is_im = True
        log_im: im_option = exact
        candidates_im =
        [0. 0. 1.]
        <BLANKLINE>
        *********************************
        *   Trivial Manipulation (TM)   *
        *********************************
        is_tm = True
        log_tm: tm_option = exact
        candidates_tm =
        [0. 0. 1.]
        <BLANKLINE>
        ********************************
        *   Unison Manipulation (UM)   *
        ********************************
        is_um = True
        log_um: um_option = exact
        candidates_um =
        [0. 0. 1.]
        <BLANKLINE>
        *********************************************
        *   Ignorant-Coalition Manipulation (ICM)   *
        *********************************************
        is_icm = False
        log_icm: icm_option = exact
        candidates_icm =
        [0. 0. 0.]
        necessary_coalition_size_icm =
        [0. 6. 4.]
        sufficient_coalition_size_icm =
        [0. 6. 4.]
        <BLANKLINE>
        ***********************************
        *   Coalition Manipulation (CM)   *
        ***********************************
        is_cm = True
        log_cm: cm_option = exact
        candidates_cm =
        [0. 0. 1.]
        necessary_coalition_size_cm =
        [0. 3. 2.]
        sufficient_coalition_size_cm =
        [0. 3. 2.]
    """

    full_name = 'Approval Voting'
    abbreviation = 'AV'

    options_parameters = Rule.options_parameters.copy()
    options_parameters.update({
        'approval_threshold': {'allowed': type_checker.is_number, 'default': 0},
        'approval_comparator': {'allowed': ['>', '>='], 'default': '>'},
        'im_option': {'allowed': ['exact'], 'default': 'exact'},
        'tm_option': {'allowed': ['exact'], 'default': 'exact'},
        'um_option': {'allowed': ['exact'], 'default': 'exact'},
        'icm_option': {'allowed': ['exact'], 'default': 'exact'},
        'cm_option': {'allowed': ['exact'], 'default': 'exact'}
    })

    def __init__(self, approval_comparator='>', approval_threshold=0., **kwargs):
        self._approval_threshold = None
        self._approval_comparator = None
        super().__init__(
            with_two_candidates_reduces_to_plurality=False, is_based_on_rk=False,
            is_based_on_ut_minus1_1=True, meets_iia=True,
            precheck_um=False, precheck_tm=False, precheck_icm=False,
            approval_comparator=approval_comparator, approval_threshold=approval_threshold,
            log_identity="APPROVAL", **kwargs
        )

    # %% Setting the parameters

    @property
    def approval_threshold(self):
        return self._approval_threshold

    @approval_threshold.setter
    def approval_threshold(self, value):
        if self._approval_threshold == value:
            return
        if self.options_parameters['approval_threshold']['allowed'](value):
            self.mylogv("Setting approval_threshold =", value, 1)
            self._approval_threshold = value
            self._result_options['approval_threshold'] = value
            self.delete_cache()
        else:
            raise ValueError("Unknown value for approval_threshold: " + format(value))

    @property
    def approval_comparator(self):
        return self._approval_comparator

    @approval_comparator.setter
    def approval_comparator(self, value):
        if self._approval_comparator == value:
            return
        if value in self.options_parameters['approval_comparator']['allowed']:
            self.mylogv("Setting approval_comparator =", value, 1)
            self._approval_comparator = value
            self._result_options['approval_comparator'] = value
            self.delete_cache()
        else:
            raise ValueError("Unknown option for approval_comparator: " + format(value))

    # %% Manipulation criteria of the voting system

    @cached_property
    def meets_ignmc_c_ctb(self):
        return True

    # %% Counting the ballots

    @cached_property
    def ballots_(self):
        """2d array of values in {0, 1}. ``ballots_[v, c] = 1`` iff voter ``v`` votes for candidates ``c``.

        See Also
        --------
        :attr:`approval_comparator`, :attr:`approval_threshold`.
        """
        self.mylog("Compute ballots", 1)
        if self.approval_comparator == '>':
            return np.greater(self.profile_.preferences_ut, self.approval_threshold)
        else:
            return np.greater_equal(self.profile_.preferences_ut, self.approval_threshold)

    @cached_property
    def scores_(self):
        """1d array of integers. ``scores_[c]`` is the number of voters who vote for candidate ``c``.
        """
        self.mylog("Compute scores", 1)
        return np.sum(self.ballots_, 0)

    # %% Individual manipulation (IM)

    def _im_main_work_v_(self, v, c_is_wanted, nb_wanted_undecided, stop_if_true):
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
            >>> rule = RuleApproval()(profile)
            >>> rule.is_im_c_(2)
            True
            >>> rule.v_im_for_c_
            array([[0., 0., 0.],
                   [0., 0., 0.],
                   [0., 0., 0.],
                   [0., 0., 1.],
                   [0., 0., 0.]])

            >>> profile = Profile(preferences_ut=[
            ...     [ 1. ,  0.5,  1. ],
            ...     [ 0. ,  0.5, -1. ],
            ...     [ 0. ,  1. ,  1. ],
            ...     [-1. , -1. ,  0.5],
            ...     [ 0. , -0.5,  0.5],
            ... ], preferences_rk=[
            ...     [0, 2, 1],
            ...     [1, 0, 2],
            ...     [1, 2, 0],
            ...     [2, 0, 1],
            ...     [2, 0, 1],
            ... ])
            >>> rule = RuleApproval()(profile)
            >>> rule.v_im_for_c_
            array([[0., 0., 0.],
                   [0., 0., 0.],
                   [0., 0., 0.],
                   [0., 0., 0.],
                   [0., 0., 0.]])

            >>> profile = Profile(preferences_ut=[
            ...     [ 0. , -0.5, -1. ],
            ...     [ 1. ,  0. ,  1. ],
            ...     [ 0. ,  1. , -0.5],
            ...     [-1. ,  0. , -0.5],
            ...     [-1. ,  1. ,  0. ],
            ... ], preferences_rk=[
            ...     [0, 1, 2],
            ...     [0, 2, 1],
            ...     [1, 0, 2],
            ...     [1, 2, 0],
            ...     [1, 2, 0],
            ... ])
            >>> rule = RuleApproval()(profile)
            >>> rule.v_im_for_c_
            array([[1., 0., 0.],
                   [0., 0., 0.],
                   [0., 0., 0.],
                   [0., 0., 0.],
                   [0., 0., 0.]])

            >>> profile = Profile(preferences_ut=[
            ...     [ 0. , -0.5, -1. ],
            ...     [ 1. ,  0. ,  1. ],
            ...     [ 0. ,  1. , -0.5],
            ...     [-1. ,  0. , -0.5],
            ...     [-1. ,  1. ,  0. ],
            ... ], preferences_rk=[
            ...     [0, 1, 2],
            ...     [0, 2, 1],
            ...     [1, 0, 2],
            ...     [1, 2, 0],
            ...     [1, 2, 0],
            ... ])
            >>> rule = RuleApproval()(profile)
            >>> rule.is_im_
            True
        """
        scores_test = self.scores_ - self.ballots_[v, :]
        w_test = int(np.argmax(scores_test))
        # Best strategy: vote only for c
        for c in range(w_test + 1):  # c <= w_test
            if not c_is_wanted[c]:
                continue
            if not np.isneginf(self._v_im_for_c[v, c]):
                continue
            nb_wanted_undecided -= 1
            if scores_test[c] + 1 >= scores_test[w_test]:
                self._v_im_for_c[v, c] = True
                self._candidates_im[c] = True
                self._voters_im[v] = True
                self._is_im = True
                self.mylog("IM found", 3)
                if stop_if_true:
                    return
            else:
                self._v_im_for_c[v, c] = False
            if nb_wanted_undecided == 0:
                return
        for c in range(w_test + 1, self.profile_.n_c):  # c > w_test
            if not c_is_wanted[c]:
                continue
            if not np.isneginf(self._v_im_for_c[v, c]):
                continue
            nb_wanted_undecided -= 1
            if scores_test[c] + 1 > scores_test[w_test]:
                self._v_im_for_c[v, c] = True
                self._candidates_im[c] = True
                self._voters_im[v] = True
                self._is_im = True
                self.mylog("IM found", 3)
                if stop_if_true:
                    return
            else:
                self._v_im_for_c[v, c] = False
            if nb_wanted_undecided == 0:
                return

    # %% Coalition Manipulation (CM)

    # noinspection PyUnusedLocal
    def _cm_main_work_c_exact_(self, c, optimize_bounds):
        scores_test = np.sum(self.ballots_[np.logical_not(self.v_wants_to_help_c_[:, c]), :], 0)
        w_test = np.argmax(scores_test)
        self._sufficient_coalition_size_cm[c] = (scores_test[w_test] - scores_test[c] + (c > w_test))
        self._necessary_coalition_size_cm[c] = self._sufficient_coalition_size_cm[c]

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

    # Defined in the superclass.
