# -*- coding: utf-8 -*-
"""
Created on 4 dec. 2018, 16:16
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


class RuleCoombs(Rule):
    """Coombs method.

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
        >>> rule = RuleCoombs()(profile)
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
        [[-1. -2. -2.]
         [-2. -3. nan]]
        candidates_by_scores_best_to_worst
        [0, 1, 2]
        scores_best_to_worst
        [[-1. -2. -2.]
         [-2. -3. nan]]
        w = 0
        score_w = [-1. -2.]
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
        log_im: im_option = fast
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
        log_um: um_option = fast
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
        log_cm: cm_option = fast, tm_option = exact
        candidates_cm =
        [ 0.  0. nan]
        necessary_coalition_size_cm =
        [0. 0. 0.]
        sufficient_coalition_size_cm =
        [0. 2. 3.]

    Notes
    -----
    The candidate who is ranked last by most voters is eliminated. Then we iterate. Ties are broken in favor of
    lower-index candidates: in case of a tie, the tied candidate with highest index is eliminated.

    * :meth:`is_cm_`:

        * :attr:`cm_option` = ``'fast'``: Polynomial heuristic. Can prove CM but unable to decide non-CM (except in
          rare obvious cases).
        * :attr:`cm_option` = ``'exact'``: Non-polynomial (:math:`n_c !`).

    * :meth:`is_icm_`: Exact in polynomial time.
    * :meth:`is_im_`:

        * :attr:`im_option` = ``'fast'``: Polynomial heuristic. Can prove IM but unable to decide non-IM (except in
          rare obvious cases).
        * :attr:`im_option` = ``'exact'``: Non-polynomial (:math:`n_c !`).

    * :meth:`is_iia`: Non-polynomial or non-exact algorithms from superclass :class:`Rule`.
    * :meth:`is_tm_`: Exact in polynomial time.
    * :meth:`is_um_`: For this voting system, UM and CM are equivalent. For this reason, :attr:`um_option` and
      :attr:`cm_option` are linked to each other: modifying one modifies the other accordingly.

    References
    ----------
    'On The Complexity of Manipulating Elections', Tom Coleman and Vanessa Teague, 2007.
    """

    full_name = 'Coombs'
    abbreviation = 'Coo'

    options_parameters = Rule.options_parameters.copy()
    options_parameters.update({
        'im_option': {'allowed': ['fast', 'exact'], 'default': 'fast'},
        'tm_option': {'allowed': ['exact'], 'default': 'exact'},
        'um_option': {'allowed': ['fast', 'exact'], 'default': 'fast'},
        'icm_option': {'allowed': ['exact'], 'default': 'exact'},
        'cm_option': {'allowed': ['fast', 'exact'], 'default': 'fast'},
    })

    def __init__(self, **kwargs):
        super().__init__(
            with_two_candidates_reduces_to_plurality=True, is_based_on_rk=True,
            precheck_um=False, precheck_icm=False,
            log_identity="COOMBS", **kwargs
        )

    @cached_property
    def _count_ballots_(self):
        self.mylog("Count ballots", 1)
        scores = np.zeros((self.profile_.n_c - 1, self.profile_.n_c))
        is_eliminated = np.zeros(self.profile_.n_c, dtype=np.bool)
        worst_to_best = []
        one_v_might_be_pivotal = False
        # preferences_borda_temp : we will put ``n_c + 1`` for eliminated candidates
        preferences_borda_temp = np.copy(self.profile_.preferences_borda_rk)
        for r in range(self.profile_.n_c - 1):
            scores[r, :] = - np.bincount(np.argmin(preferences_borda_temp, 1), minlength=self.profile_.n_c)
            scores[r, is_eliminated] = np.nan
            loser = np.where(scores[r, :] == np.nanmin(scores[r, :]))[0][-1]  # Tie-breaking: the last index
            is_eliminated[loser] = True
            worst_to_best.append(loser)
            preferences_borda_temp[:, loser] = self.profile_.n_c + 1
            # Dealing with the possibility of pivotality
            # ``margins_r[c] = 0`` : ``c`` is eliminated (i.e. ``c == loser``).
            # ``margins_r[c] = x`` means that with ``x`` vetos more, ``c`` would be eliminated instead of ``loser``.
            worst_score = scores[r, loser]
            margins_r = scores[r, :] - worst_score + (np.array(range(self.profile_.n_c)) < loser)
            self.mylogv("margins_r =", margins_r, 3)
            self.mylogv("np.max(margins_r) =", np.nanmax(margins_r), 3)
            if np.nanmax(margins_r) <= 2:
                one_v_might_be_pivotal = True
        w = np.argmin(is_eliminated)
        worst_to_best.append(w)
        candidates_by_scores_best_to_worst = worst_to_best[::-1]
        return {'scores': scores, 'one_v_might_be_pivotal': one_v_might_be_pivotal, 'w': w,
                'candidates_by_scores_best_to_worst': candidates_by_scores_best_to_worst}

    @cached_property
    def scores_(self):
        """2d array of integers. ``scores[r, c]`` is minus the number of voters who vote against candidate ``c`` at
        elimination round ``r``.
        """
        return self._count_ballots_['scores']

    @cached_property
    def w_(self):
        """Integer (winning candidate)."""
        return self._count_ballots_['w']

    @cached_property
    def candidates_by_scores_best_to_worst_(self):
        """1d array of integers. Candidates are sorted according to their order of elimination. By definition /
        convention, ``candidates_by_scores_best_to_worst_[0]`` = :attr:`w_`.
        """
        return self._count_ballots_['candidates_by_scores_best_to_worst']

    @cached_property
    def _one_v_might_be_pivotal_(self):
        return self._count_ballots_['one_v_might_be_pivotal']

    @cached_property
    def v_might_im_for_c_(self):
        return np.full((self.profile_.n_v, self.profile_.n_c), self._one_v_might_be_pivotal_)

    @cached_property
    def meets_ignmc_c_ctb(self):
        return True

    # %% Setting the options

    @property
    def um_option(self):
        return self._um_option

    @um_option.setter
    def um_option(self, value):
        if self._um_option == value:
            return
        if value in self.options_parameters['um_option']['allowed']:
            self.mylogv("Setting um_option =", value, 1)
            self._cm_option = value
            self._um_option = value
            self.delete_cache(contains='_um_', suffix='_')
            self.delete_cache(contains='_cm_', suffix='_')
        else:
            raise ValueError("Unknown value for um_option: " + format(value))

    @property
    def cm_option(self):
        return self._cm_option

    @cm_option.setter
    def cm_option(self, value):
        if self._cm_option == value:
            return
        if value in self.options_parameters['cm_option']['allowed']:
            self.mylogv("Setting cm_option =", value, 1)
            self._cm_option = value
            self._um_option = value
            self.delete_cache(contains='_um_', suffix='_')
            self.delete_cache(contains='_cm_', suffix='_')
        else:
            raise ValueError("Unknown value for cm_option: " + format(value))

    # %% Manipulation: general routine

    def _cm_aux_fast(self, c, n_max, preferences_borda_s):
        """Fast algorithm used for IM and CM (which is equivalent to UM)

        Parameters
        ----------
        c : int
            Candidate for which we want to manipulate.
        n_max : int or inf
            Maximum number of manipulators allowed.

            * IM --> put 1.
            * CM, complete and exact --> put the current value of ``sufficient_coalition_size[c] - 1`` (we want to find
                the best value for ``sufficient_coalition_size[c]``, even if it exceeds the number of manipulators).
            * CM, otherwise --> put the number of manipulators.

        preferences_borda_s : list of list
            Preferences of the sincere voters (in Borda format with vtb).

        Returns
        -------
        int or inf
            ``n_manip_fast``. If a manipulation is found with n_max manipulators or less, then a sufficient number
            of manipulators is returned. Otherwise, it is +inf.

        Examples
        --------
            >>> profile = Profile(preferences_rk=[
            ...     [1, 2, 0, 3],
            ...     [1, 3, 2, 0],
            ...     [2, 0, 3, 1],
            ...     [2, 1, 3, 0],
            ...     [3, 2, 0, 1],
            ... ])
            >>> rule = RuleCoombs()(profile)
            >>> rule.candidates_cm_
            array([0., 1., 0., 1.])
        """
        # Each step of the loop:
        # ^^^^^^^^^^^^^^^^^^^^^^
        # We start at the end of round ``r``, with ``candidates[is_candidate_alive_end_r]``. We find the ``d`` such
        # that, with ``d`` in addition at the beginning, we need the fewest manipulators to eliminate ``d`` (and reach
        # the final situation we wanted).
        # Initialization
        # ^^^^^^^^^^^^^^
        # End of last round: only ``c`` is alive.
        nb_manipulators_used = 0
        is_candidate_alive_end_r = np.zeros(self.profile_.n_c, dtype=np.bool)
        is_candidate_alive_end_r[c] = True
        for r in range(self.profile_.n_c - 2, -1, -1):
            self.mylogv("cm_aux_fast: Round r =", r, 3)
            least_nb_manipulators_for_r = np.inf
            # We look for a candidate with which round r was easy
            best_d = None
            for d in range(self.profile_.n_c):
                if is_candidate_alive_end_r[d]:
                    continue
                self.mylogv("cm_aux_fast: Try to add d =", d, 3)
                is_candidate_alive_begin_r = np.copy(is_candidate_alive_end_r)
                is_candidate_alive_begin_r[d] = True
                scores_s = np.zeros(self.profile_.n_c)
                scores_s[is_candidate_alive_begin_r] = - np.sum(np.equal(
                    preferences_borda_s[:, is_candidate_alive_begin_r],
                    np.min(preferences_borda_s[:, is_candidate_alive_begin_r], 1)[:, np.newaxis]
                ), 0)
                # self.mylogv("cm_AUX: scores_s =", scores_s, 3)
                normal_loser = np.where(scores_s == np.min(scores_s))[0][-1]  # Tie-breaking
                nb_manip_d = max(0, scores_s[d] - scores_s[normal_loser] + (d < normal_loser))
                self.mylogv("cm_aux_fast: nb_manip_d =", nb_manip_d, 3)
                if nb_manip_d < least_nb_manipulators_for_r:
                    least_nb_manipulators_for_r = nb_manip_d
                    best_d = d
            self.mylogv("cm_aux_fast: best_d =", best_d, 3)
            # Update variables for previous round
            nb_manipulators_used = max(nb_manipulators_used, least_nb_manipulators_for_r)
            if nb_manipulators_used > n_max:
                self.mylogv("cm_aux_fast: Conclusion: nb_manipulators_used =", np.inf, 3)
                return np.inf
            is_candidate_alive_end_r[best_d] = True

        self.mylogv("cm_aux_fast: Conclusion: nb_manipulators_used =", nb_manipulators_used, 3)
        return nb_manipulators_used

    def _cm_aux_exact(self, c, n_max, preferences_borda_s):
        """Exact algorithm used for IM and CM (which is equivalent to UM)

        Parameters
        ----------
        c : int
            Candidate for which we want to manipulate.
        n_max : int
            Maximum number of manipulators allowed.

            * IM --> put 1.
            * CM, complete and exact --> put the current value of ``sufficient_coalition_size[c] - 1`` (we want to find
                the best value for ``sufficient_coalition_size[c]``, even if it exceeds the number of manipulators).
            * CM, otherwise --> put the number of manipulators.

        preferences_borda_s : list of list
            Preferences of the sincere voters (in Borda format).

        Returns
        -------
        int or inf
            ``n_manip_exact``. If a manipulation is found with ``n_max`` manipulators or less, the minimal sufficient
            number of manipulators is returned. Otherwise, it is +inf.

        Examples
        --------
            >>> profile = Profile(preferences_rk=[
            ...     [0, 2, 1, 3],
            ...     [0, 2, 3, 1],
            ...     [0, 2, 3, 1],
            ...     [3, 0, 2, 1],
            ...     [3, 1, 2, 0],
            ... ])
            >>> rule = RuleCoombs(cm_option='exact')(profile)
            >>> rule.necessary_coalition_size_cm_
            array([0., 4., 4., 3.])
        """
        # Explore subsets that are reachable with less than the upper bound.
        # ``situations_end_r`` is a dictionary.
        # * keys: ``is_candidate_alive_end_r``, tuple of booleans.
        # * values: ``nb_manip_used_after_r``, number of manipulators used after round ``d`` to go from situation
        #   ``is_candidate_alive_end_r`` to the singleton ``c``.
        # Example: ``(1,1,0,1,0):3`` means that if we have candidates 0, 1 and 3, we need 3 manipulators to make ``c``
        # win at the end.
        situations_end_r = {tuple(np.array(range(self.profile_.n_c)) == c): 0}
        situations_begin_r = {}
        for r in range(self.profile_.n_c - 2, -1, -1):
            self.mylogv("cm_aux_exact: Round r =", r, 3)
            situations_begin_r = {}
            for is_candidate_alive_end_r, nb_manip_used_after_r in situations_end_r.items():
                self.mylogv("cm_aux_exact: is_candidate_alive_end_r =", is_candidate_alive_end_r, 3)
                self.mylogv("cm_aux_exact: nb_manip_used_after_r =", nb_manip_used_after_r, 3)
                for d in range(self.profile_.n_c):
                    if is_candidate_alive_end_r[d]:
                        continue
                    self.mylogv("cm_aux_exact: d =", d, 3)
                    is_candidate_alive_begin_r = np.copy(is_candidate_alive_end_r)
                    is_candidate_alive_begin_r[d] = True
                    scores_s = np.zeros(self.profile_.n_c)
                    scores_s[is_candidate_alive_begin_r] = - np.sum(np.equal(
                        preferences_borda_s[:, is_candidate_alive_begin_r],
                        np.min(preferences_borda_s[:, is_candidate_alive_begin_r], 1)[:, np.newaxis]
                    ), 0)
                    # self.mylogv("scores_s =", scores_s, 3)
                    normal_loser = np.where(scores_s == np.nanmin(scores_s))[0][-1]  # Tie-breaking
                    nb_manip_r = max(scores_s[d] - scores_s[normal_loser] + (d < normal_loser), 0)
                    self.mylogv("cm_aux_exact: nb_manip_r =", nb_manip_r, 3)
                    nb_manip_r_and_after = max(nb_manip_r, nb_manip_used_after_r)
                    if nb_manip_r_and_after > n_max:
                        continue
                    if tuple(is_candidate_alive_begin_r) in situations_begin_r:
                        situations_begin_r[tuple(is_candidate_alive_begin_r)] = min(
                            situations_begin_r[tuple(is_candidate_alive_begin_r)],
                            nb_manip_r_and_after)
                    else:
                        situations_begin_r[tuple(is_candidate_alive_begin_r)] = nb_manip_r_and_after
            self.mylogv("cm_aux_exact: situations_begin_r =", situations_begin_r, 3)
            if len(situations_begin_r) == 0:
                self.mylog("cm_aux_exact: Manipulation is impossible with n_max manipulators.", 3)
                return np.inf  # By convention
            situations_end_r = situations_begin_r
        else:
            self.mylogv("cm_aux_exact: situations_begin_r:", situations_begin_r, 3)
            is_candidate_alive_begin, nb_manip_used = situations_begin_r.popitem()
            self.mylogv("cm_aux_exact: is_candidate_alive_begin:", is_candidate_alive_begin, 3)
            self.mylogv("cm_aux_exact: Conclusion: nb_manip_used_new =", nb_manip_used, 3)
        return nb_manip_used

    # %% Individual manipulation (IM)

    def _im_main_work_v_(self, v, c_is_wanted, nb_wanted_undecided, stop_if_true):
        """
            >>> profile = Profile(preferences_rk=[
            ...     [0, 2, 3, 1],
            ...     [1, 0, 3, 2],
            ...     [1, 3, 2, 0],
            ...     [2, 1, 3, 0],
            ...     [3, 1, 0, 2],
            ... ])
            >>> rule = RuleCoombs()(profile)
            >>> rule.is_im_
            True

            >>> profile = Profile(preferences_rk=[
            ...     [0, 2, 1],
            ...     [0, 2, 1],
            ...     [1, 0, 2],
            ...     [2, 0, 1],
            ...     [2, 1, 0],
            ... ])
            >>> rule = RuleCoombs()(profile)
            >>> rule.is_im_c_with_voters_(2)
            (nan, array([ 0.,  0.,  0., nan, nan]))

            >>> profile = Profile(preferences_rk=[
            ...     [0, 2, 1],
            ...     [0, 2, 1],
            ...     [1, 0, 2],
            ...     [2, 0, 1],
            ...     [2, 1, 0],
            ... ])
            >>> rule = RuleCoombs(im_option='exact')(profile)
            >>> rule.is_im_
            False

            >>> profile = Profile(preferences_rk=[
            ...     [0, 1, 2],
            ...     [0, 1, 2],
            ...     [2, 0, 1],
            ...     [2, 1, 0],
            ...     [2, 1, 0],
            ... ])
            >>> rule = RuleCoombs(im_option='exact')(profile)
            >>> rule.is_im_
            True

            >>> profile = Profile(preferences_rk=[
            ...     [0, 1, 2],
            ...     [1, 0, 2],
            ...     [1, 2, 0],
            ...     [2, 0, 1],
            ...     [2, 0, 1],
            ... ])
            >>> rule = RuleCoombs()(profile)
            >>> rule.v_im_for_c_
            array([[ 0.,  0.,  0.],
                   [ 0., nan,  0.],
                   [ 0., nan,  1.],
                   [ 0.,  0., nan],
                   [ 0.,  0., nan]])
        """
        preferences_borda_s = self.profile_.preferences_borda_rk[np.array(range(self.profile_.n_v)) != v, :]
        for c in range(self.profile_.n_c):
            if not c_is_wanted[c]:
                continue
            if not np.isneginf(self._v_im_for_c[v, c]):
                continue
            self.mylogv('IM: Candidate c =', c, 3)
            n_manip_fast = self._cm_aux_fast(c, n_max=1, preferences_borda_s=preferences_borda_s)
            self.mylogv("IM: Fast: n_manip_fast =", n_manip_fast, 3)
            if n_manip_fast <= 1:
                self._v_im_for_c[v, c] = True
                self._candidates_im[c] = True
                self._voters_im[v] = True
                self._is_im = True
                if stop_if_true:
                    return
                continue  # pragma: no cover - Not really "executed" because of Python optimization
            if self.im_option != 'exact':
                self._v_im_for_c[v, c] = np.nan
                continue
            n_manip_exact = self._cm_aux_exact(c, n_max=1, preferences_borda_s=preferences_borda_s)
            self.mylogv("IM: Exact: n_manip_exact =", n_manip_exact, 3)
            if n_manip_exact <= 1:
                self._v_im_for_c[v, c] = True
                self._candidates_im[c] = True
                self._voters_im[v] = True
                self._is_im = True
                if stop_if_true:
                    return
            else:
                self._v_im_for_c[v, c] = False

    # %% Trivial Manipulation (TM)

    # Use the generic methods from class Rule.

    # %% Unison manipulation (UM)

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

    def _cm_main_work_c_(self, c, optimize_bounds):
        """
            >>> profile = Profile(preferences_rk=[
            ...     [0, 2, 1, 3],
            ...     [0, 2, 3, 1],
            ...     [0, 2, 3, 1],
            ...     [3, 0, 2, 1],
            ...     [3, 1, 2, 0],
            ... ])
            >>> rule = RuleCoombs(cm_option='exact')(profile)
            >>> rule.candidates_cm_
            array([0., 0., 0., 0.])

            >>> profile = Profile(preferences_rk=[
            ...     [0, 1, 2],
            ...     [0, 2, 1],
            ...     [0, 2, 1],
            ...     [0, 2, 1],
            ...     [2, 0, 1],
            ... ])
            >>> rule = RuleCoombs(cm_option='exact')(profile)
            >>> rule.necessary_coalition_size_cm_
            array([0., 5., 4.])

            >>> profile = Profile(preferences_rk=[
            ...     [1, 3, 0, 2],
            ...     [2, 0, 3, 1],
            ...     [2, 1, 3, 0],
            ...     [3, 1, 0, 2],
            ...     [3, 2, 1, 0],
            ... ])
            >>> rule = RuleCoombs(cm_option='exact')(profile)
            >>> rule.candidates_cm_
            array([0., 1., 1., 0.])

            >>> profile = Profile(preferences_rk=[
            ...     [1, 2, 0, 3],
            ...     [1, 3, 2, 0],
            ...     [2, 0, 3, 1],
            ...     [2, 1, 3, 0],
            ...     [3, 2, 0, 1],
            ... ])
            >>> rule = RuleCoombs(cm_option='exact')(profile)
            >>> rule.candidates_cm_
            array([0., 1., 0., 1.])

            >>> profile = Profile(preferences_ut=[
            ...     [ 0. ,  0. ],
            ...     [ 0.5, -0.5],
            ...     [-0.5, -1. ],
            ...     [-0.5,  0. ],
            ...     [ 0. ,  0.5],
            ...     [-0.5, -0.5],
            ... ], preferences_rk=[
            ...     [0, 1],
            ...     [0, 1],
            ...     [0, 1],
            ...     [1, 0],
            ...     [1, 0],
            ...     [1, 0],
            ... ])
            >>> rule = RuleCoombs(cm_option='exact')(profile)
            >>> rule.is_cm_c_(1)
            False
            >>> rule.sufficient_coalition_size_cm_
            array([0., 3.])
        """
        n_m = self.profile_.matrix_duels_ut[c, self.w_]
        exact = (self.cm_option == "exact")
        if optimize_bounds and exact:
            n_max = self._sufficient_coalition_size_cm[c] - 1
        else:
            n_max = n_m
        self.mylogv("CM: n_max =", n_max, 3)
        if not exact and self._necessary_coalition_size_cm[c] > n_max:  # pragma: no cover
            # TO DO: Investigate whether this case can actually happen.
            self._reached_uncovered_code()
            self.mylog("CM: Fast algorithm will not do better than what we already know", 3)
            return
        n_manip_fast = self._cm_aux_fast(
            c, n_max,
            preferences_borda_s=self.profile_.preferences_borda_rk[np.logical_not(self.v_wants_to_help_c_[:, c]), :])
        self.mylogv("CM: n_manip_fast =", n_manip_fast, 3)
        self._update_sufficient(self._sufficient_coalition_size_cm, c, n_manip_fast,
                                'CM: Update sufficient_coalition_size_cm =')
        if not exact:
            # With fast algo, we stop here anyway. It is not a "quick escape" (if we'd try again with
            # ``optimize_bound``, we would not try better).
            return

        # From this point, we have necessarily the 'exact' option.
        if self._sufficient_coalition_size_cm[c] == self._necessary_coalition_size_cm[c]:
            return
        if not optimize_bounds and n_m >= self._sufficient_coalition_size_cm[c]:
            # This is a quick escape: since we have the option 'exact', if we come back with ``optimize_bound``,
            # we will try to be more precise.
            return True
        # Either we're in complete mode (and might have succeeded), or we in incomplete mode (and we have failed).
        n_max_updated = min(n_manip_fast - 1, n_max)
        self.mylogv("CM: n_max_updated =", n_max_updated, 3)
        n_manip_exact = self._cm_aux_exact(
            c, n_max_updated,
            preferences_borda_s=self.profile_.preferences_borda_rk[np.logical_not(self.v_wants_to_help_c_[:, c]), :])
        self.mylogv("CM: n_manip_exact =", n_manip_exact)
        n_manip = min(n_manip_fast, n_manip_exact)
        self.mylogv("CM: n_manip =", n_manip, 3)
        self._update_sufficient(self._sufficient_coalition_size_cm, c, n_manip,
                                'CM: Update sufficient_coalition_size_cm =')
        # Update necessary coalition and return
        if optimize_bounds:
            self._necessary_coalition_size_cm[c] = self._sufficient_coalition_size_cm[c]
            return False
        else:
            if n_m >= n_manip:
                # We have optimized the size of the coalition.
                self._necessary_coalition_size_cm[c] = self._sufficient_coalition_size_cm[c]
                return False
            else:
                # We have explored everything with ``n_max = n_m`` but manipulation failed. However, we have not
                # optimized sufficient_size (which must be higher than ``n_m``), so it is a quick escape.
                self._update_necessary(self._necessary_coalition_size_cm, c, n_m + 1,
                                       'CM: Update necessary_coalition_size_cm = n_m + 1 =')
                return True
