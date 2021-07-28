# -*- coding: utf-8 -*-
"""
Created on 7 dec. 2018, 16:54
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
from svvamp.utils.misc import compute_next_permutation, strong_connected_components
from svvamp.preferences.profile import Profile


class RuleKemeny(Rule):
    """Kemeny method.

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
        >>> rule = RuleKemeny()(profile)
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
        [2 0 1]
        candidates_by_scores_best_to_worst
        [0 2 1]
        scores_best_to_worst
        [2 1 0]
        w = 0
        score_w = 2
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
         ||     Condorcet_c_rk_ctb (True)  ==> Condorcet_c_rk (True)      ||
         ||           ||               ||       ||             ||         ||
         V            V                ||       ||             V          V
        Condorcet_c_ut_abs_ctb (True)      ==>     Condorcet_ut_abs_c (True)
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
        log_cm: cm_option = lazy, um_option = lazy, icm_option = exact, tm_option = exact
        candidates_cm =
        [ 0.  0. nan]
        necessary_coalition_size_cm =
        [0. 1. 2.]
        sufficient_coalition_size_cm =
        [0. 2. 3.]

    Notes
    -----
    We find the order on candidates whose total Kendall tau distance to the voters is minimal. The top element of
    this order is declared the winner. In case several orders are optimal, the first one by lexicographic order is
    given. This implies that if several winners are possible, the one with lowest index is declared the winner.

    For this voting system, even deciding the sincere winner is NP-hard.

    * :meth:`is_cm_`: Non-polynomial or non-exact algorithms from superclass :class:`Rule`.
    * :meth:`is_icm_`: Exact in polynomial time (once the sincere winner is computed).
    * :meth:`is_im_`: Non-polynomial or non-exact algorithms from superclass :class:`Rule`.
    * :meth:`is_iia_`: Exact in polynomial time (once the sincere winner is computed).
    * :meth:`is_tm_`: Exact in the time needed to decide the winner of one election, multiplied by :attr:`n_c`.
    * :meth:`is_um_`: Non-polynomial or non-exact algorithms from superclass :class:`Rule`.

    References
    ----------
    'Mathematics without numbers', J. G. Kemeny, 1959.

    'A Consistent Extension of Condorcet’s Election Principle', H. P. Young and A. Levenglick, 1978.

    'On the approximability of Dodgson and Young elections', Ioannis Caragiannis et al., 2009.

    'Comparing and aggregating partial orders with Kendall tau distances', Franz J. Brandenburg, Andreas Gleißner
    and Andreas Hofmeier, 2013.
    """

    full_name = 'Kemeny'
    abbreviation = 'Kem'

    options_parameters = Rule.options_parameters.copy()
    options_parameters['icm_option'] = {'allowed': ['exact'], 'default': 'exact'}

    def __init__(self, **kwargs):
        super().__init__(
            with_two_candidates_reduces_to_plurality=True, is_based_on_rk=True,
            precheck_icm=True,
            log_identity="KEMENY", **kwargs
        )

    @cached_property
    def _strong_connected_components_(self):
        return strong_connected_components(self.profile_.matrix_victories_rk > 0)

    @cached_property
    def score_w_(self):
        """Integer. With our convention, ``scores_w_`` = :attr:`n_c` - 1.
        """
        self.mylog("Compute winner's score", 1)
        return self.profile_.n_c - 1

    @cached_property
    def scores_best_to_worst_(self):
        """1d array of integers. With our convention, ``scores_best_to_worst`` is the vector [:attr:`n_c` - 1,
        :attr:`n_c` - 2, ..., 0].
        """
        self.mylog("Compute scores_best_to_worst", 1)
        return np.array(range(self.profile_.n_c - 1, -1, -1))

    @cached_property
    def _count_first_component_(self):
        """Compute the optimal order for the first strongly connected component of the victory matrix (ties count as
        a connection).
        """
        self.mylog("Count first strongly connected component", 1)
        candidates_by_scores_best_to_worst_first_component = []
        order = np.array(sorted(self._strong_connected_components_[0]))
        size_component = len(order)
        best_score = -1
        best_order = None
        while order is not None:
            self.mylogv("order =", order, 3)
            score = np.sum(self.profile_.matrix_duels_rk[:, order][order, :][np.triu_indices(size_component)])
            self.mylogv("score =", score, 3)
            if score > best_score:
                best_order, best_score = order, score
            order = compute_next_permutation(order, size_component)
        self.mylogv("best_order =", best_order, 3)
        w = best_order[0]
        candidates_by_scores_best_to_worst_first_component.extend(best_order)
        return {
            'candidates_by_scores_best_to_worst_first_component': candidates_by_scores_best_to_worst_first_component,
            'w': w}

    @cached_property
    def w_(self):
        return self._count_first_component_['w']

    @cached_property
    def _candidates_by_scores_best_to_worst_first_component_(self):
        return self._count_first_component_['candidates_by_scores_best_to_worst_first_component']

    @cached_property
    def _count_ballots_(self):
        """Compute the optimal order for all the candidates."""
        self.mylog("Count other strongly connected components", 1)
        candidates_by_scores_best_to_worst = self._candidates_by_scores_best_to_worst_first_component_.copy()
        for component_members in self._strong_connected_components_[1:]:
            order = np.array(sorted(component_members))
            size_component = len(order)
            best_score = -1
            best_order = None
            while order is not None:
                self.mylogv("order =", order, 3)
                score = np.sum(self.profile_.matrix_duels_rk[:, order][order, :][np.triu_indices(size_component)])
                self.mylogv("score =", score, 3)
                if score > best_score:
                    best_order, best_score = order, score
                order = compute_next_permutation(order, size_component)
            self.mylogv("best_order =", best_order, 3)
            candidates_by_scores_best_to_worst.extend(best_order)
        candidates_by_scores_best_to_worst = np.array(candidates_by_scores_best_to_worst)
        scores = self.profile_.n_c - 1 - np.argsort(candidates_by_scores_best_to_worst)
        return {'scores': scores, 'candidates_by_scores_best_to_worst': candidates_by_scores_best_to_worst}

    @cached_property
    def scores_(self):
        """1d array of integers. By convention, scores are integers from 1 to :attr:`n_c`, with :attr:`n_c` for the
        winner and 1 for the last candidate in Kemeny optimal order.
        """
        return self._count_ballots_['scores']

    @cached_property
    def candidates_by_scores_best_to_worst_(self):
        """1d array of integers. This is an optimal Kemeny order.

        In case several orders are optimal, the first one by lexicographic order is given. This implies that if
        several winners are possible, the one with lowest index is declared the winner.
        """
        return self._count_ballots_['candidates_by_scores_best_to_worst']

    @cached_property
    def meets_condorcet_c_rk_ctb(self):
        return True
