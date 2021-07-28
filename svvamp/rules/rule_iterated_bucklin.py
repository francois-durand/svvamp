# -*- coding: utf-8 -*-
"""
Created on 7 dec. 2018, 16:35
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


class RuleIteratedBucklin(Rule):
    """Iterated Bucklin method.

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
        >>> rule = RuleIteratedBucklin()(profile)
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
        [[0.83333333 0.66666667 0.66666667]
         [1.66666667 1.                inf]]
        candidates_by_scores_best_to_worst
        [0, 1, 2]
        scores_best_to_worst
        [[0.83333333 0.66666667 0.66666667]
         [1.66666667 1.                inf]]
        w = 0
        score_w = [0.83333333 1.66666667]
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
         ||       maj_fav_c_rk_ctb (False) ==> maj_fav_c_rk (True)        ||
         ||           ||                                       ||         ||
         V            V                                        V          V
        majority_favorite_c_ut_ctb (False) ==> majority_favorite_c_ut (True)
         ||                                                               ||
         V                                                                V
        IgnMC_c_ctb (False)                ==>                IgnMC_c (True)
         ||                                                               ||
         V                                                                V
        InfMC_c_ctb (False)                ==>                InfMC_c (True)
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
        log_icm: icm_option = lazy
        candidates_icm =
        [0. 0. 0.]
        necessary_coalition_size_icm =
        [0. 5. 3.]
        sufficient_coalition_size_icm =
        [0. 6. 4.]
        <BLANKLINE>
        ***********************************
        *   Coalition Manipulation (CM)   *
        ***********************************
        is_cm = nan
        log_cm: cm_option = lazy, um_option = lazy, icm_option = lazy, tm_option = exact
        candidates_cm =
        [ 0.  0. nan]
        necessary_coalition_size_cm =
        [0. 0. 1.]
        sufficient_coalition_size_cm =
        [0. 2. 3.]

    Notes
    -----
    The candidate with least *adjusted median Borda score* (cf. below) is eliminated. Then the new Borda scores are
    computed. Etc. Ties are broken in favor of lower-index candidates: in case of a tie, the candidate with highest
    index is eliminated.

    Adjusted median Borda score:

        Let ``med_c`` be the median Borda score for candidate ``c``. Let ``x_c`` the number of voters who put a lower
        Borda score to ``c``. Then ``c``'s adjusted median is ``med_c - x_c / (n_v + 1)``.

        If ``med_c > med_d``, then it is also true for the adjusted median. If ``med_c = med_d``, then ``c`` has a
        better adjusted median iff ``x_c < x_d``, i.e. if more voters give to ``c`` the Borda score ``med_c`` or
        higher.

        So, the best candidate by adjusted median is the :class:`~Bucklin` winner. Here, at each round,
        we eliminate the candidate with lowest adjusted median Borda score, which justifies the name of "Iterated
        Bucklin method".

    Unlike Baldwin method (= Iterated Borda), Iterated Bucklin does not meet the Condorcet criterion. Indeed,
    a Condorcet winner may have the (strictly) worst median ranking.

    * :meth:`is_cm_`: Non-polynomial or non-exact algorithms from superclass :class:`Rule`.
    * :meth:`is_icm_`: The algorithm from superclass :class:`Rule` is polynomial and has a window of error of 1
      manipulator.
    * :meth:`is_im_`: Non-polynomial or non-exact algorithms from superclass :class:`Rule`.
    * :meth:`is_iia`: Non-polynomial or non-exact algorithms from superclass :class:`Rule`.
    * :meth:`is_tm_`: Exact in polynomial time.
    * :meth:`is_um_`: Non-polynomial or non-exact algorithms from superclass :class:`Rule`.
    """

    full_name = 'Iterated Bucklin'
    abbreviation = 'IB'

    options_parameters = Rule.options_parameters.copy()

    def __init__(self, **kwargs):
        super().__init__(
            with_two_candidates_reduces_to_plurality=True, is_based_on_rk=True,
            log_identity="ITERATED_BUCKLIN", **kwargs
        )

    @cached_property
    def _count_ballots_(self):
        self.mylog("Count ballots", 1)
        scores = np.zeros((self.profile_.n_c - 1, self.profile_.n_c))
        worst_to_best = []
        is_eliminated = np.zeros(self.profile_.n_c, dtype=np.bool)
        preferences_borda_temp = np.copy(self.profile_.preferences_borda_rk)
        for r in range(self.profile_.n_c - 1):
            for c in range(self.profile_.n_c):
                if is_eliminated[c]:
                    scores[r, c] = np.inf
                    continue
                median = np.median(preferences_borda_temp[:, c])
                x = np.sum(np.less(preferences_borda_temp[:, c], median))
                scores[r, c] = median - x / (self.profile_.n_v + 1)
            loser = np.where(scores[r, :] == np.min(scores[r, :]))[0][-1]  # Tie-breaking: last index
            is_eliminated[loser] = True
            worst_to_best.append(loser)
            preferences_borda_temp[
                np.less(preferences_borda_temp, preferences_borda_temp[:, loser][:, np.newaxis])
            ] += 1
        w = np.argmin(is_eliminated)
        worst_to_best.append(w)
        candidates_by_scores_best_to_worst = worst_to_best[::-1]
        return {'scores': scores, 'w': w, 'candidates_by_scores_best_to_worst': candidates_by_scores_best_to_worst}

    @cached_property
    def scores_(self):
        """2d array of integers. ``scores[r, c]`` is the adjusted median Borda score of candidate ``c`` at
        elimination round ``r``.
        """
        return self._count_ballots_['scores']

    @cached_property
    def w_(self):
        return self._count_ballots_['w']

    @cached_property
    def candidates_by_scores_best_to_worst_(self):
        """1d array of integers. Candidates are sorted according to their order of elimination.
        """
        return self._count_ballots_['candidates_by_scores_best_to_worst']

    # noinspection PyProtectedMember
    @cached_property
    def _one_v_might_be_pivotal_(self):
        """
            >>> profile = Profile(preferences_rk=[
            ...     [0, 1, 2],
            ...     [0, 2, 1],
            ...     [0, 2, 1],
            ...     [0, 2, 1],
            ...     [2, 0, 1],
            ... ])
            >>> rule = RuleIteratedBucklin()(profile)
            >>> rule._one_v_might_be_pivotal_
            False
        """
        scores_r = np.zeros(self.profile_.n_c)
        is_eliminated = np.zeros(self.profile_.n_c, dtype=np.bool)
        preferences_borda_temp = np.copy(self.profile_.preferences_borda_rk)
        for r in range(self.profile_.n_c - 1):
            self.mylogv("Pre-testing pivotality: round r =", r, 3)
            loser_sincere = (self.candidates_by_scores_best_to_worst_[-1 - r])
            self.mylogv("Pre-testing pivotality: loser_sincere =", loser_sincere, 3)
            # At worst, one manipulator can put one ``c`` from top to bottom or from bottom to top. Can this change the
            # loser of this round?
            for c in range(self.profile_.n_c):
                if is_eliminated[c]:
                    scores_r[c] = np.inf
                    continue
                if c == loser_sincere:
                    median = np.median(
                        np.concatenate((preferences_borda_temp[:, c], [self.profile_.n_c, self.profile_.n_c])))
                    x = sum(preferences_borda_temp[:, c] < median) - 1
                else:
                    median = np.median(np.concatenate((preferences_borda_temp[:, c], [1, 1])))
                    x = sum(preferences_borda_temp[:, c] < median) + 1
                scores_r[c] = median - x / (self.profile_.n_v + 1)
            loser = np.where(scores_r == scores_r.min())[0][-1]
            self.mylogv("Pre-testing pivotality: loser =", loser, 3)
            if loser != loser_sincere:
                self.mylog("There might be a pivotal voter", 3)
                return True
            is_eliminated[loser] = True
            preferences_borda_temp[
                np.less(preferences_borda_temp, preferences_borda_temp[:, loser][:, np.newaxis])
            ] += 1
        else:
            self.mylog("There is no pivotal voter", 3)
            return False

    @cached_property
    def v_might_im_for_c_(self):
        return np.full((self.profile_.n_v, self.profile_.n_c), self._one_v_might_be_pivotal_)

    @cached_property
    def meets_majority_favorite_c_rk(self):
        return True
