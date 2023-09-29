# -*- coding: utf-8 -*-
"""
Created on 4 dec. 2018, 11:42
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


class RuleBaldwin(Rule):
    """Baldwin method.

    Options
    -------
        >>> RuleBaldwin.print_options_parameters()
        cm_option: ['fast', 'exact']. Default: 'fast'.
        icm_option: ['exact']. Default: 'exact'.
        iia_subset_maximum_size: is_number. Default: 2.
        im_option: ['lazy', 'exact']. Default: 'lazy'.
        tm_option: ['lazy', 'exact']. Default: 'exact'.
        um_option: ['lazy', 'exact']. Default: 'lazy'.

    Notes
    -----
    Each voter provides a strict order of preference. The candidate with lowest Borda score is eliminated. Then the
    new Borda scores are computed. Etc. Ties are broken in favor of lower-index candidates: in case of a tie,
    the candidate with highest index is eliminated.

    Since a Condorcet winner has always a Borda score higher than average, Baldwin method meets the Condorcet
    criterion.

    * :meth:`is_cm_`: Non-polynomial or non-exact algorithms from superclass :class:`Rule`.
    * :meth:`is_icm_`: Exact in polynomial time.
    * :meth:`is_im_`: Deciding IM is NP-complete. Non-polynomial or non-exact algorithms from superclass :class:`Rule`.
    * :meth:`is_iia_`: Exact in polynomial time.
    * :meth:`is_tm_`: Exact in polynomial time.
    * :meth:`is_um_`: Non-polynomial or non-exact algorithms from superclass :class:`Rule`.

    References
    ----------
    'Complexity of and algorithms for the manipulation of Borda, Nanson's and Baldwin's voting rules',
    Jessica Davies, George Katsirelos, Nina Narodytska, Toby Walsh and Lirong Xia, 2014.

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
        >>> rule = RuleBaldwin()(profile)
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
        [[ 6.  4.  5.]
         [ 3. inf  2.]]
        candidates_by_scores_best_to_worst
        [0 2 1]
        scores_best_to_worst
        [[ 6.  5.  4.]
         [ 3.  2. inf]]
        w = 0
        score_w = [6. 3.]
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
        log_cm: cm_option = fast, um_option = lazy, tm_option = exact
        candidates_cm =
        [0. 0. 0.]
        necessary_coalition_size_cm =
        [0. 1. 3.]
        sufficient_coalition_size_cm =
        [0. 2. 3.]
    """

    full_name = 'Baldwin'
    abbreviation = 'Bal'

    options_parameters = Rule.options_parameters.copy()
    options_parameters['icm_option'] = {'allowed': ['exact'], 'default': 'exact'}
    options_parameters['cm_option'] = {'allowed': ['fast', 'exact'], 'default': 'fast'}

    def __init__(self, **kwargs):
        super().__init__(
            with_two_candidates_reduces_to_plurality=True, is_based_on_rk=True,
            precheck_icm=False, log_identity="BALDWIN", **kwargs
        )

    def _count_ballots_aux_(self):
        """
        Returns
        -------
        dict
            A dictionary with ``scores``, ``w``, ``candidates_by_scores_best_to_worst``
        """
        self.mylog("Count ballots", 1)
        scores = np.zeros((self.profile_.n_c - 1, self.profile_.n_c))
        worst_to_best = []
        w = None
        matrix_duels_rk = self.profile_.matrix_duels_rk
        scores_borda_temp = np.sum(matrix_duels_rk, axis=1).astype(float)
        for r in range(self.profile_.n_c - 1):  # Round r
            scores[r, :] = scores_borda_temp
            loser = np.where(scores[r, :] == np.min(scores[r, :]))[0][-1]  # Tie-breaking: higher index
            worst_to_best.append(loser)
            if r == self.profile_.n_c - 2:
                scores_borda_temp[loser] = np.inf
                w = np.argmin(scores_borda_temp)
                worst_to_best.append(w)
                break
            # Prepare for next round
            scores_borda_temp -= self.profile_.matrix_duels_rk[:, loser]
            scores_borda_temp[loser] = np.inf
        candidates_by_scores_best_to_worst = np.array(worst_to_best[::-1])
        return {'scores': scores, 'w': w, 'candidates_by_scores_best_to_worst': candidates_by_scores_best_to_worst}

    @cached_property
    def _count_ballots_(self):
        return self._count_ballots_aux_()

    @cached_property
    def scores_(self):
        """2d array of integers. ``scores[r, c]`` is the Borda score of candidate ``c`` at elimination round ``r``.
        By convention, if candidate ``c`` does not participate to round ``r``, then ``scores[r, c] = numpy.inf``.
        """
        return self._count_ballots_['scores']

    @cached_property
    def w_(self):
        return self._count_ballots_['w']

    @cached_property
    def candidates_by_scores_best_to_worst_(self):
        """1d array of integers. ``candidates_by_scores_best_to_worst[-r]`` is the candidate eliminated at
        elimination round ``r``. By definition / convention, ``candidates_by_scores_best_to_worst[0]`` = :attr:`w_`.
        """
        return self._count_ballots_['candidates_by_scores_best_to_worst']

    @cached_property
    def meets_condorcet_c_rk_ctb(self):
        # Consider vtb case. If ``c`` is a Condorcet winner with vtb and ctb, then she has at worst ties in
        # ``matrix_duels_rk``, so she has at least the average Borda score. Can she be eliminated? For that, it is
        # necessary that all other candidates have exactly the average, i.e. ``matrix_duels_rk`` is a general tie.
        # Since she is Condorcet winner vtb/ctb, she must be candidate 0, so our tie-breaking rule eliminates another
        # candidate. Conclusion: this voting system meets Condorcet criterion vtb/ctb.
        #
        # However, let us consider the following example.
        # preferences_borda_ut:
        # [ 1   0 ]
        # [0.5 0.5]
        # [0.5 0.5]
        # ==> candidate 0 is a relative Condorcet winner.
        # preferences_borda_rk (with vtb):
        # [ 1   0 ]
        # [ 0   1 ]
        # [ 0   1 ]
        # ==> candidate 1 wins.
        # Conclusion: this voting system does not meet Condorcet criterion (ut/rel).
        return True

    # %% Coalition Manipulation (CM)

    def _cm_main_work_c_fast_(self, c, optimize_bounds):
        """Do the main work in CM loop for candidate ``c``.

        * Try to improve the bounds ``_sufficient_coalition_size_cm[c]`` and ``_necessary_coalition_size_cm[c]``.

            >>> profile = Profile(preferences_rk=[
            ...     [0, 2, 1],
            ...     [0, 2, 1],
            ...     [1, 0, 2],
            ...     [1, 0, 2],
            ...     [2, 0, 1],
            ... ])
            >>> rule = RuleBaldwin()(profile)
            >>> rule.candidates_cm_
            array([0., 0., 0.])

            >>> profile = Profile(preferences_rk=[
            ...     [0, 1, 2],
            ...     [2, 0, 1],
            ...     [2, 0, 1],
            ...     [1, 2, 0],
            ... ])
            >>> rule = RuleBaldwin()(profile)
            >>> rule.candidates_cm_
            array([0., 1., 0.])

            >>> profile = Profile(preferences_ut=[
            ...     [-0.5, -0.5,  0. , -0.5],
            ...     [ 1. , -0.5,  0.5,  0.5],
            ...     [ 0. ,  0. , -0.5, -0.5],
            ...     [-1. , -1. , -0.5,  0. ],
            ...     [ 1. , -1. ,  0.5, -1. ],
            ...     [-0.5,  0. ,  1. ,  1. ],
            ... ], preferences_rk=[
            ...     [2, 1, 3, 0],
            ...     [0, 3, 2, 1],
            ...     [1, 0, 3, 2],
            ...     [3, 2, 1, 0],
            ...     [0, 2, 3, 1],
            ...     [2, 3, 1, 0],
            ... ])
            >>> rule = RuleBaldwin()(profile)
            >>> rule.sufficient_coalition_size_cm_
            array([0., 1., 3., 2.])
        """
        # First part: is there a subset (including c) such that w has less than the average?
        n_v = self.profile_.n_v
        n_c = self.profile_.n_c
        n_m = self.profile_.matrix_duels_ut[c, self.w_]
        ballots_sincere = self.profile_.preferences_borda_rk[np.logical_not(self.v_wants_to_help_c_[:, c]), :]
        profile_sincere = Profile(preferences_borda_rk=ballots_sincere)
        matrix_duels_sincere = profile_sincere.matrix_duels_rk
        row_w = matrix_duels_sincere[self.w_, :]
        optimal_subset = list({c for c in range(n_c) if c != self.w_ and row_w[c] < n_v / 2} | {c})
        self.mylogv('CM: Further check: optimal_subset =', optimal_subset, 3)
        optimal_average_score = row_w[optimal_subset].sum() / len(optimal_subset)
        self.mylogv('CM: Further check: optimal_average_score =', optimal_average_score, 3)
        if optimal_average_score > n_v / 2 or (optimal_average_score == n_v / 2 and c > self.w_):
            necessary = n_m + 1
            self.mylogv('CM: Further check: necessary =', necessary)
            self._update_necessary(self._necessary_coalition_size_cm, c, necessary,
                                   'CM: Further check: necessary_coalition_size_cm =')
        # An opportunity to escape before real work
        if self._necessary_coalition_size_cm[c] == self._sufficient_coalition_size_cm[c]:
            return False
        if not optimize_bounds and n_m < self._necessary_coalition_size_cm[c]:
            # This is a quick escape: we have not optimized the bounds the best we could.
            return True

        # Second part: manipulation heuristic.
        optimal_subset = sorted(set(optimal_subset) | {self.w_})
        d_cand_i = {cand: i for i, cand in enumerate(optimal_subset)}
        i_w = d_cand_i[self.w_]
        i_c = d_cand_i[c]
        k = len(optimal_subset)
        scores_test = matrix_duels_sincere[:, optimal_subset][optimal_subset, :].sum(axis=1)
        # We add a tie-breaking term [(k-1)/k, (k-2)/k, ..., 0] to ease the computations.
        scores_test = scores_test + (np.array(range(k - 1, -1, -1)) / k)
        self.mylogv('CM: Fast algorithm: optimal_subset =', optimal_subset, 3)
        self.mylogv('CM: Fast algorithm: scores_test =', scores_test, 3)
        ballots_manipulators_subset = []
        for _ in range(n_m):
            # Balancing ballot: put candidates in the order of their current scores (least point to the most
            # dangerous).
            candidates_by_decreasing_score = np.argsort(- scores_test, kind='mergesort')
            ballot = np.argsort(candidates_by_decreasing_score)
            # Our priority is to kill w, so we put her at the bottom
            ballot += np.less(ballot, ballot[i_w])
            ballot[i_w] = 0
            # If w is already at the bottom, we can safely give an advantage to c for future rounds.
            if np.all(scores_test[i_w] <= np.array(scores_test)):
                ballot -= np.greater(ballot, ballot[i_c])
                ballot[i_c] = k - 1
            self.mylogv('CM: Fast algorithm: ballot =', ballot, 3)
            # New scores = old scores + ballot.
            scores_test += ballot
            self.mylogv('CM: Fast algorithm: scores_test =', scores_test, 3)
            ballots_manipulators_subset.append(ballot)
        ballots_manipulators = np.zeros((n_m, n_c), dtype=int)
        if n_m >= 1:  # Otherwise, `ballots_manipulators_subset` is empty hence numpy cannot cast the array.
            ballots_manipulators[:, optimal_subset] = ballots_manipulators_subset
        ballots_manipulators[:, optimal_subset] += n_c - k
        if k < n_c:
            other_candidates_by_decreasing_score = [
                cand for cand in np.argsort(- matrix_duels_sincere.sum(axis=1), kind='mergesort')
                if cand not in optimal_subset
            ]
            ballots_manipulators[:, other_candidates_by_decreasing_score] = np.arange(n_c - k)[np.newaxis, :]
        self.mylogv('CM: Fast algorithm: ballots_manipulators =', ballots_manipulators, 3)
        ballots = np.vstack((ballots_sincere, ballots_manipulators))
        w_test = RuleBaldwin()(profile=Profile(preferences_ut=ballots)).w_
        if w_test == c:
            sufficient = n_m
            self.mylogv('CM: Fast algorithm: sufficient =', sufficient)
            self._update_sufficient(self._sufficient_coalition_size_cm, c, sufficient,
                                    'CM: Fast algorithm: sufficient_coalition_size_cm =')
        is_quick_escape = False  # We will not do better if we come back in this method
        return is_quick_escape

    def _cm_main_work_c_(self, c, optimize_bounds):
        """
            >>> profile = Profile(preferences_rk=[
            ...     [0, 1, 2],
            ...     [0, 2, 1],
            ...     [0, 2, 1],
            ...     [0, 2, 1],
            ...     [2, 0, 1],
            ... ])
            >>> rule = RuleBaldwin(cm_option='exact')(profile)
            >>> rule.candidates_cm_
            array([0., 0., 0.])

            >>> profile = Profile(preferences_rk=[
            ...     [0, 1, 2],
            ...     [1, 2, 0],
            ...     [2, 1, 0],
            ...     [1, 2, 0],
            ... ])
            >>> rule = RuleBaldwin(cm_option='exact')(profile)
            >>> rule.sufficient_coalition_size_cm_
            array([3., 0., 3.])
        """
        is_quick_escape_fast = self._cm_main_work_c_fast_(c, optimize_bounds)
        if not self.cm_option == "exact":
            # With 'fast' option, we stop here anyway.
            return is_quick_escape_fast
        # From this point, we have necessarily the 'exact' option (which is, in fact, only an exhaustive exploration
        # with = ``n_m`` manipulators).
        is_quick_escape_exact = self._cm_main_work_c_exact_(c, optimize_bounds)
        return is_quick_escape_fast or is_quick_escape_exact
