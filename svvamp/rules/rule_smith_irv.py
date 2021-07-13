# -*- coding: utf-8 -*-
"""
Created on 12 jul. 2021, 16:47
Copyright François Durand 2014-2021
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
from svvamp.rules.rule_irv import RuleIRV
from svvamp.preferences.profile import Profile
from svvamp.preferences.profile_subset_candidates import ProfileSubsetCandidates
from svvamp.utils.util_cache import cached_property


class RuleSmithIRV(Rule):
    """Smith-IRV Rule.

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
        >>> rule = RuleSmithIRV()(profile)
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
        [[0 1 2]
         [0 2 1]
         [1 0 2]
         [2 0 1]
         [2 1 0]]
        scores =
        [[1. 0. 0.]
         [0. 0. 0.]]
        candidates_by_scores_best_to_worst
        [0, 1, 2]
        scores_best_to_worst
        [[1. 0. 0.]
         [0. 0. 0.]]
        w = 0
        score_w = [1. 0.]
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
         ||     Condorcet_c_rk_ctb (False) ==> Condorcet_c_rk (True)      ||
         ||           ||               ||       ||             ||         ||
         V            V                ||       ||             V          V
        Condorcet_c_ut_abs_ctb (False)     ==>     Condorcet_ut_abs_c (True)
         ||                            ||       ||                        ||
         ||                            V        V                         ||
         ||       maj_fav_c_rk_ctb (True)  ==> maj_fav_c_rk (True)        ||
         ||           ||                                       ||         ||
         V            V                                        V          V
        majority_favorite_c_ut_ctb (True)  ==> majority_favorite_c_ut (True)
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
        is_im = nan
        log_im: im_option = lazy
        candidates_im =
        [ 0.  0. nan]
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
        is_um = nan
        log_um: um_option = lazy
        candidates_um =
        [ 0.  0. nan]
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
        is_cm = nan
        log_cm: cm_option = lazy, tm_option = exact
        candidates_cm =
        [ 0.  0. nan]
        necessary_coalition_size_cm =
        [0. 1. 2.]
        sufficient_coalition_size_cm =
        [0. 6. 4.]

    Smith-IRV does not :attr:`meets_condorcet_c_ut_abs_ctb`:

        >>> profile = Profile(preferences_ut=[
        ...     [ 1. ,  1. , -1. ],
        ...     [ 0. , -0.5,  1. ],
        ... ], preferences_rk=[
        ...     [1, 0, 2],
        ...     [2, 0, 1],
        ... ])
        >>> RuleSmithIRV()(profile).w_
        1
        >>> profile.condorcet_winner_ut_abs_ctb
        0

    Smith-IRV does not :attr:`meets_condorcet_c_ut_rel`:

        >>> profile = Profile(preferences_ut=[
        ...     [ 0.5,  1. ,  1. ],
        ...     [-0.5,  0. ,  0.5],
        ... ], preferences_rk=[
        ...     [1, 2, 0],
        ...     [2, 1, 0],
        ... ])
        >>> RuleSmithIRV()(profile).w_
        1
        >>> profile.condorcet_winner_ut_rel
        2

    Notes
    -----
    Each voter must provide a strict total order. Restrict the election to the Smith Set (in the sense of
    :attr:`smith_set_rk`), then run :class:`RuleIRV`.

    References
    ----------
    'Four Condorcet-Hare Hybrid Methods for Single-Winner Elections', James Green-Armytage, 2011.
    """

    def __init__(self, **kwargs):
        super().__init__(
            options_parameters={
                'cm_option': {'allowed': ['lazy', 'fast', 'exact'], 'default': 'lazy'},
                'tm_option': {'allowed': ['exact'], 'default': 'exact'},
                'icm_option': {'allowed': ['exact'], 'default': 'exact'}
            },
            with_two_candidates_reduces_to_plurality=True, is_based_on_rk=True,
            precheck_um=False, precheck_icm=False,
            log_identity="SMITH_IRV", **kwargs
        )

    # %% Counting the ballots

    @cached_property
    def _compute_winner_(self):
        self.mylog("Compute winner", 1)
        if self.profile_.exists_condorcet_winner_rk:
            w = self.profile_.condorcet_winner_rk
            irv = None
        else:
            smith_set = self.profile_.smith_set_rk
            profile_r = ProfileSubsetCandidates(parent_profile=self.profile_, candidates_subset=smith_set)
            irv = RuleIRV()(profile_r)
            w_r = irv.w_
            w = smith_set[w_r]
        return {'w': w, 'irv': irv}

    @cached_property
    def w_(self):
        return self._compute_winner_['w']

    @cached_property
    def scores_(self):
        self.mylog("Compute scores", 1)
        smith_set = self.profile_.smith_set_rk
        scores_smith = [(1 if c in smith_set else 0) for c in range(self.profile_.n_c)]
        if self.profile_.exists_condorcet_winner_rk:
            scores_irv = np.zeros(self.profile_.n_c)
        else:
            irv = self._compute_winner_['irv']
            n_c_r = irv.profile_.n_c
            scores_irv_r = sorted(range(n_c_r), key=irv.elimination_path_.__getitem__)
            scores_irv = np.zeros(self.profile_.n_c)
            scores_irv[smith_set] = scores_irv_r
        return np.array([scores_smith, scores_irv])

    @cached_property
    def candidates_by_scores_best_to_worst_(self):
        self.mylog("Compute candidates_by_scores_best_to_worst", 1)
        return sorted(
            range(self.profile_.n_c),
            key=lambda c: list(self.scores_[:, c]),
            reverse=True
        )

    # TO DO: implement v_might_im_for_c_

    # %% Manipulation criteria of the voting system

    @cached_property
    def meets_majority_favorite_c_rk_ctb(self):
        return True

    @cached_property
    def meets_condorcet_c_rk(self):
        return True

    # %% Individual manipulation (IM)

    # TODO: should be implemented.

    # %% Trivial Manipulation (TM)

    # Use the general methods from class Rule.

    # %% Unison manipulation (UM)

    # TODO: should be implemented .

    # %% Ignorant-Coalition Manipulation (ICM)

    # Use the general methods from class Rule.

    # %% Coalition Manipulation (CM)

    def _cm_main_work_c_fast_(self, c, optimize_bounds):
        irv = RuleIRV()(self.profile_)
        if irv.w_ != self.w_:
            return False
        # Now self.w_ == irv.w_
        irv = RuleIRV(cm_option='exact')(self.profile_)
        ballots_m = irv.example_ballots_cm_c_(c)
        if ballots_m is None:
            return False  # Not a quick escape (we did what we could)
        preferences_rk_s = self.profile_.preferences_rk[np.logical_not(self.v_wants_to_help_c_[:, c]), :]
        profile_test = Profile(
            preferences_rk=np.concatenate((preferences_rk_s, ballots_m))
        )
        winner_test = self.__class__()(profile_test).w_
        n_m = self.profile_.matrix_duels_ut[c, self.w_]
        if winner_test == c:
            self._update_sufficient(self._sufficient_coalition_size_cm, c, n_m,
                                    'CM: Manipulation found by Decondorcification/IRV heuristic =>\n'
                                    '    sufficient_coalition_size_cm = n_m =')
        return False
