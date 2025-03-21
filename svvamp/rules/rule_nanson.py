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


class RuleNanson(Rule):
    """Nanson method.

    Options
    -------
        >>> RuleNanson.print_options_parameters()
        cm_option: ['fast', 'exact']. Default: 'fast'.
        icm_option: ['exact']. Default: 'exact'.
        iia_subset_maximum_size: is_number. Default: 2.
        im_option: ['lazy', 'exact']. Default: 'lazy'.
        precheck_heuristic: is_bool. Default: True.
        tm_option: ['lazy', 'exact']. Default: 'exact'.
        um_option: ['lazy', 'exact']. Default: 'lazy'.

    Notes
    -----
    At each round, all candidates with a Borda score strictly lower than average are simultaneously eliminated. When
    all remaining candidates have the same Borda score, it means that the matrix of duels (for this subset of
    candidates) has only ties. Then the candidate with lowest index is declared the winner. Since a Condorcet winner
    has always a Borda score higher than average, Nanson method meets the Condorcet criterion.

    * :meth:`is_cm_`: Deciding CM is NP-complete. Non-polynomial or non-exact algorithms from superclass :class:`Rule`.
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
        >>> rule = RuleNanson()(profile)
        >>> rule.meets_condorcet_c_rk_ctb
        True
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
         [ 3. inf  2.]
         [ 0. inf inf]]
        candidates_by_scores_best_to_worst
        [0 2 1]
        scores_best_to_worst
        [[ 6.  5.  4.]
         [ 3.  2. inf]
         [ 0. inf inf]]
        w = 0
        score_w = [6. 3. 0.]
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

    full_name = 'Nanson'
    abbreviation = 'Nan'

    options_parameters = Rule.options_parameters.copy()
    options_parameters['icm_option'] = {'allowed': ['exact'], 'default': 'exact'}
    options_parameters['cm_option'] = {'allowed': ['fast', 'exact'], 'default': 'fast'}

    def __init__(self, **kwargs):
        super().__init__(
            with_two_candidates_reduces_to_plurality=True, is_based_on_rk=True,
            precheck_icm=False,
            log_identity="NANSON", **kwargs
        )

    @cached_property
    def _count_ballots_(self):
        self.mylog("Count ballots", 1)
        scores = []
        candidates_worst_to_best = []
        one_v_might_be_pivotal = False
        # borda_temp[c] will be inf when c is eliminated
        borda_temp = np.copy(self.profile_.borda_score_c_rk).astype(float)
        borda_average = self.profile_.n_v * (self.profile_.n_c - 1) / 2
        while True:  # This is a round
            scores.append(np.copy(borda_temp))
            least_borda = np.min(borda_temp)
            self.mylogv("scores =", scores, 2)
            # self.mylogv("least_borda =", least_borda, 2)
            self.mylogv("borda_average =", borda_average, 2)
            if least_borda == borda_average:
                # Then all candidates have the average. This means that all duels are ties. This also includes the
                # case where only one candidate is remaining. We eliminate them by decreasing order of index (highest
                # indexes first).
                losing_candidates = np.where(np.isfinite(borda_temp))[0]
                candidates_worst_to_best.extend(losing_candidates[::-1])
                if len(losing_candidates) > 1:
                    one_v_might_be_pivotal = True
                    self.mylog("One voter might be pivotal", 2)
                self.mylogv("losing_candidates[::-1] =", losing_candidates[::-1], 2)
                break
            # Remove all candidates with borda score < average. Lowest Borda score first, highest indexes first.
            nb_candidates_alive = np.sum(np.isfinite(borda_temp))
            # (alive at beginning of round)
            borda_for_next_round = np.copy(borda_temp).astype(float)
            borda_average_for_next_round = borda_average
            while least_borda < borda_average:
                if least_borda + (nb_candidates_alive-1) >= borda_average:
                    one_v_might_be_pivotal = True
                    self.mylog("One voter might be pivotal", 2)
                losing_candidates = np.where(borda_temp == least_borda)[0]
                candidates_worst_to_best.extend(losing_candidates[::-1])
                self.mylogv("losing_candidates[::-1] =", losing_candidates[::-1], 2)
                # self.mylogv("candidates_worst_to_best =", candidates_worst_to_best, 2)
                borda_average_for_next_round -= (len(losing_candidates) * self.profile_.n_v / 2)
                borda_for_next_round[losing_candidates] = np.inf
                borda_for_next_round = borda_for_next_round - np.sum(
                    self.profile_.matrix_duels_rk[:, losing_candidates], 1)
                borda_temp[losing_candidates] = np.inf
                least_borda = np.min(borda_temp)
                # self.mylogv("borda_temp =", borda_temp, 2)
                # self.mylogv("least_borda =", least_borda, 2)
                # self.mylogv("borda_average =", borda_average, 2)
            if least_borda - (nb_candidates_alive-1) < borda_average:
                one_v_might_be_pivotal = True
                self.mylog("One voter might be pivotal", 2)
            borda_temp = borda_for_next_round
            borda_average = borda_average_for_next_round
        candidates_by_scores_best_to_worst = np.array(candidates_worst_to_best[::-1])
        w = candidates_by_scores_best_to_worst[0]
        scores = np.array(scores)
        return {'scores': scores, 'w': w, 'candidates_by_scores_best_to_worst': candidates_by_scores_best_to_worst,
                'one_v_might_be_pivotal': one_v_might_be_pivotal}

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
        """1d array of integers. Candidates are sorted according to their order of elimination. When several
        candidates are eliminated during the same round, they are sorted by Borda score at that round and,
        in case of a tie, by their index (highest indexes are eliminated first).
        """
        return self._count_ballots_['candidates_by_scores_best_to_worst']

    @cached_property
    def v_might_im_for_c_(self):
        return np.full((self.profile_.n_v, self.profile_.n_c), self._count_ballots_['one_v_might_be_pivotal'])

    # %% Manipulation criteria of the voting system

    @cached_property
    def meets_condorcet_c_rk_ctb(self):
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
            >>> rule = RuleNanson()(profile)
            >>> rule.candidates_cm_
            array([0., 0., 0.])

            >>> profile = Profile(preferences_rk=[
            ...     [0, 1, 2],
            ...     [1, 0, 2],
            ...     [2, 1, 0],
            ...     [2, 0, 1],
            ... ])
            >>> rule = RuleNanson()(profile)
            >>> rule.necessary_coalition_size_cm_
            array([0., 1., 1.])
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
            # We want to save c, so we put her on top
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
        w_test = RuleNanson()(profile=Profile(preferences_ut=ballots)).w_
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
            >>> rule = RuleNanson(cm_option='exact')(profile)
            >>> rule.candidates_cm_
            array([0., 0., 0.])
        """
        is_quick_escape_fast = self._cm_main_work_c_fast_(c, optimize_bounds)
        if not self.cm_option == "exact":
            # With 'fast' option, we stop here anyway.
            return is_quick_escape_fast
        # From this point, we have necessarily the 'exact' option (which is, in fact, only an exhaustive exploration
        # with = ``n_m`` manipulators).
        is_quick_escape_exact = self._cm_main_work_c_exact_(c, optimize_bounds)
        return is_quick_escape_fast or is_quick_escape_exact
