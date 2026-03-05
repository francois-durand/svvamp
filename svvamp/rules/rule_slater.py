# -*- coding: utf-8 -*-
"""
Created on 8 jul. 2022, 12:33
Copyright François Durand 2014-2012
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


class RuleSlater(Rule):
    """Slater method.

    Parameters
    ----------
    winner_option : str
        'exact' or 'lazy'. Default: 'exact'. If `winner_option` is 'exact', then the winner is computed as usual.
         If `winner_option` is 'lazy', then the winner is computed only in the obvious case where there is a Condorcet
         winner; note that if there is no Condorcet-admissible candidate, it is however possible to decide ``is_cm_``
         automatically to True.
    tie_break_rule : str
        'lexico' of 'random'. Default: 'lexico'. If `tie_break_rule` is 'lexico', then the candidate with the lowest
        index is selected in case of a tie (usual behavior of SVVAMP for the other voting rules). If `tie_break_rule`
        is `random`, then each time a profile is loaded, a tie-break order over the candidates is drawn at random.
        This tie-break is used for the result of the sincere election, but also in case of manipulation.

    Options
    -------
        >>> RuleSlater.print_options_parameters()
        cm_option: ['lazy', 'exact']. Default: 'lazy'.
        icm_option: ['exact']. Default: 'exact'.
        iia_subset_maximum_size: is_number. Default: 2.
        im_option: ['lazy', 'exact']. Default: 'lazy'.
        precheck_heuristic: is_bool. Default: True.
        tie_break_rule: ['lexico', 'random']. Default: 'lexico'.
        tm_option: ['lazy', 'exact']. Default: 'exact'.
        um_option: ['lazy', 'exact']. Default: 'lazy'.
        winner_option: ['exact', 'lazy']. Default: 'exact'.

    Notes
    -----
    The Slater method is defined similarly to the Kemeny method, but relying on the matrix of victories instead
    of the matrix of duels.

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
        >>> rule = RuleSlater()(profile)
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
        is_cm = False
        log_cm: cm_option = lazy, um_option = lazy, icm_option = exact, tm_option = exact
        candidates_cm =
        [0. 0. 0.]
        necessary_coalition_size_cm =
        [0. 1. 3.]
        sufficient_coalition_size_cm =
        [0. 2. 3.]
    """

    full_name = "Slater"
    abbreviation = "Sla"

    options_parameters = Rule.options_parameters.copy()
    options_parameters["icm_option"] = {"allowed": ["exact"], "default": "exact"}
    options_parameters["tie_break_rule"] = {"allowed": ["lexico", "random"], "default": "lexico"}
    options_parameters["winner_option"] = {"allowed": ["exact", "lazy"], "default": "exact"}

    def __init__(self, tie_break_rule="lexico", winner_option="exact", **kwargs):
        self._tie_break_rule = None
        self._winner_option = None
        super().__init__(
            with_two_candidates_reduces_to_plurality=True,
            is_based_on_rk=True,
            precheck_icm=True,
            tie_break_rule=tie_break_rule,
            winner_option=winner_option,
            log_identity="SLATER",
            **kwargs,
        )

    # %% Setting the parameters

    @property
    def winner_option(self):
        return self._winner_option

    @winner_option.setter
    def winner_option(self, value):
        if self._winner_option == value:
            return
        if value in self.options_parameters["winner_option"]["allowed"]:
            self.mylogv("Setting winner_option =", value, 1)
            self._winner_option = value
            self._result_options["winner_option"] = value
            self.delete_cache()
        else:
            raise ValueError("Unknown value for winner_option: " + format(value))

    @property
    def tie_break_rule(self):
        return self._tie_break_rule

    @tie_break_rule.setter
    def tie_break_rule(self, value):
        if self._tie_break_rule == value:
            return
        if value in self.options_parameters["tie_break_rule"]["allowed"]:
            self.mylogv("Setting tie_break_rule =", value, 1)
            self._tie_break_rule = value
            self._result_options["tie_break_rule"] = value
            self.delete_cache()
        else:
            raise ValueError("Unknown option for tie_break_rule: " + format(value))

    # %% Election results

    @cached_property
    def tie_break_weights_(self):
        """1d array of integers. Tie-break weights over the candidates.

        A larger number means a more favored candidate. For example, the lexico tie-break order is represented by
        [n_c - 1, ..., 0], i.e., candidate 0 has a `tie-break strength` of `n_c - 1`, and so on.
        """
        if self.tie_break_rule == "lexico":
            return np.arange(self.profile_.n_c - 1, -1, -1)
        else:
            return np.random.permutation(self.profile_.n_c)

    @cached_property
    def _strong_connected_components_(self):
        return strong_connected_components(self.profile_.matrix_victories_rk > 0)

    @cached_property
    def score_w_(self):
        """Integer. With our convention, ``scores_w_`` = :attr:`n_c` - 1."""
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
            score = np.sum(self.profile_.matrix_victories_rk[:, order][order, :][np.triu_indices(size_component)])
            self.mylogv("score =", score, 3)
            found_better_order = score > best_score or (
                score == best_score
                and self.tie_break_rule == "random"
                and tuple(self.tie_break_weights_[order]) > tuple(self.tie_break_weights_[best_order])
            )
            if found_better_order:
                best_order, best_score = order, score
            order = compute_next_permutation(order, size_component)
        self.mylogv("best_order =", best_order, 3)
        w = best_order[0]
        candidates_by_scores_best_to_worst_first_component.extend(best_order)
        return {
            "candidates_by_scores_best_to_worst_first_component": candidates_by_scores_best_to_worst_first_component,
            "w": w,
        }

    @cached_property
    def w_(self):
        if self.profile_.exists_condorcet_winner_rk_ctb:
            return self.profile_.condorcet_winner_rk_ctb
        if self.winner_option == "lazy":
            return np.nan
        return self._count_first_component_["w"]

    @cached_property
    def _candidates_by_scores_best_to_worst_first_component_(self):
        return self._count_first_component_["candidates_by_scores_best_to_worst_first_component"]

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
                score = np.sum(self.profile_.matrix_victories_rk[:, order][order, :][np.triu_indices(size_component)])
                self.mylogv("score =", score, 3)
                found_better_order = score > best_score or (
                    score == best_score
                    and self.tie_break_rule == "random"
                    and tuple(self.tie_break_weights_[order]) > tuple(self.tie_break_weights_[best_order])
                )
                if found_better_order:
                    best_order, best_score = order, score
                order = compute_next_permutation(order, size_component)
            self.mylogv("best_order =", best_order, 3)
            candidates_by_scores_best_to_worst.extend(best_order)
        candidates_by_scores_best_to_worst = np.array(candidates_by_scores_best_to_worst)
        scores = self.profile_.n_c - 1 - np.argsort(candidates_by_scores_best_to_worst)
        return {"scores": scores, "candidates_by_scores_best_to_worst": candidates_by_scores_best_to_worst}

    @cached_property
    def scores_(self):
        """1d array of integers. By convention, scores are integers from 1 to :attr:`n_c`, with :attr:`n_c` for the
        winner and 1 for the last candidate in Kemeny optimal order.
        """
        return self._count_ballots_["scores"]

    @cached_property
    def candidates_by_scores_best_to_worst_(self):
        """1d array of integers. This is an optimal Kemeny order.

        In case several orders are optimal, the first one by lexicographic order is given. This implies that if
        several winners are possible, the one with lowest index is declared the winner.
        """
        return self._count_ballots_["candidates_by_scores_best_to_worst"]

    @cached_property
    def meets_condorcet_c_rk_ctb(self):
        return True

    # %% Coalition Manipulation (CM)

    @cached_property
    def is_cm_(self):
        if not self.profile_.exists_condorcet_admissible:
            self.mylog("There no Condorcet-admissible candidate, so coalition manipulation is possible.", 2)
            return True
        return super().is_cm_

    # noinspection PyMethodMayBeStatic,PyUnusedLocal
    def _cm_main_work_c_fast_(self, c, optimize_bounds):
        """Do the main work in CM loop for candidate ``c``.

        * Try to improve bounds ``_sufficient_coalition_size_cm[c]`` and ``_necessary_coalition_size_cm[c]``.
        """
        # Necessary condition for CM: For certain (d_1, ..., d_k), the order `c > d_1 > ... > d_k > w` must
        # have a better score than `w > c > d_1 > ... > d_k`. Denoting `M` the matrix of victories (after manipulation)
        # and `A` the corresponding antisymmetric matrix, it means that `s = A[c, w] + \sum A[d_i, w] >= 0`, and it must
        # be a strict inequality if the tie-break favors `w` over `c`.
        pref_borda_rk_s = self.profile_.preferences_borda_rk[np.logical_not(self.v_wants_to_help_c_[:, c]), :]
        matrix_duels_s = Profile(preferences_ut=pref_borda_rk_s).matrix_duels_ut
        a_c_w_best_case = self.profile_.matrix_victories_rk[c, self.w_] - self.profile_.matrix_victories_rk[self.w_, c]
        n_defeats_w_best_case = np.sum(matrix_duels_s[self.w_, :] < self.profile_.n_v / 2) - 1
        s = a_c_w_best_case + n_defeats_w_best_case
        tie_breaks_favors_c_over_w = self.tie_break_weights_[c] > self.tie_break_weights_[self.w_]
        self.mylogm("CM: Fast algorithm: matrix_duels_s =", matrix_duels_s, 3)
        self.mylogv("CM: Fast algorithm: a_c_w_best_case =", a_c_w_best_case, 3)
        self.mylogv("CM: Fast algorithm: n_defeats_w_best_case =", n_defeats_w_best_case, 3)
        if not s + tie_breaks_favors_c_over_w > 0:
            n_m = self.profile_.matrix_duels_ut[c, self.w_]
            self._update_necessary(
                self._necessary_coalition_size_cm, c, n_m + 1, "CM: Fast algorithm: necessary_coalition_size_cm ="
            )
            # No need to continue, the second condition would not improve this bound.
            # But this is not a quick escape, since we wouldn't do better if we came in this method again.
            return False

        # Second necessary condition. The order `c > d_1 > ... > d_k > w` must have a better score than
        # `d_1 > ... > d_k > w > c`. With the same notations as above, it means that
        # `s = A[c, w] + \sum A[c, d_i] >= 0`, and it must be a strict inequality if the tie-break favors `d_1` over
        # `c`. We do not know d_1 (and we don't want to bother), but we know that the tie-break might help `c` only
        # if `c` is not last in the tie-break order.
        n_victories_c_best_case = np.sum(matrix_duels_s[:, c] < self.profile_.n_v / 2) - 1
        s = a_c_w_best_case + n_victories_c_best_case
        tie_break_might_help_c = self.tie_break_weights_[c] > 0
        self.mylogv("CM: Fast algorithm: n_victories_c_best_case =", n_victories_c_best_case, 3)
        if not s + tie_break_might_help_c > 0:
            n_m = self.profile_.matrix_duels_ut[c, self.w_]
            self._update_necessary(
                self._necessary_coalition_size_cm, c, n_m + 1, "CM: Fast algorithm: necessary_coalition_size_cm ="
            )
        return False

    def _cm_main_work_c_(self, c, optimize_bounds):
        """
        Examples
        --------
            >>> profile = Profile(preferences_rk=[
            ...     [0, 2, 1],
            ...     [0, 2, 1],
            ...     [2, 1, 0],
            ...     [2, 0, 1],
            ... ])
            >>> rule = RuleSlater(cm_option='exact')(profile)
            >>> float(rule.sufficient_coalition_size_cm_[1])
            3.0
        """
        is_quick_escape_fast = self._cm_main_work_c_fast_(c, optimize_bounds)
        if not self.cm_option == "exact":
            # With 'fast' option, we stop here anyway.
            return is_quick_escape_fast
        # From this point, we have necessarily the 'exact' option (which is, in fact, only an exhaustive exploration
        # with = ``n_m`` manipulators).
        is_quick_escape_exact = self._cm_main_work_c_exact_(c, optimize_bounds)
        return is_quick_escape_fast or is_quick_escape_exact

    @cached_property
    def theta_critical_(self):
        """
        This is only valid for at least 4 candidates.

        >>> profile = Profile(preferences_rk=[[0, 1, 2, 3]])
        >>> rule = RuleSlater()(profile)
        >>> rule.theta_critical_
        0.25
        """
        return 1 / 4
