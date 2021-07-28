# -*- coding: utf-8 -*-
"""
Created on 4 dec. 2018, 15:20
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
from svvamp.utils.pseudo_bool import equal_true
from svvamp.preferences.profile import Profile


class RuleBucklin(Rule):
    """Bucklin method.

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
        >>> rule = RuleBucklin()(profile)
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
        [[2. 1. 2.]
         [4. 3. 3.]
         [5. 5. 5.]]
        candidates_by_scores_best_to_worst
        [0 1 2]
        scores_best_to_worst
        [[2. 1. 2.]
         [4. 3. 3.]
         [5. 5. 5.]]
        w = 0
        score_w = [2. 4. 5.]
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
        [0. 6. 4.]
        sufficient_coalition_size_icm =
        [0. 6. 4.]
        <BLANKLINE>
        ***********************************
        *   Coalition Manipulation (CM)   *
        ***********************************
        is_cm = False
        log_cm: cm_option = exact
        candidates_cm =
        [0. 0. 0.]
        necessary_coalition_size_cm =
        [0. 1. 3.]
        sufficient_coalition_size_cm =
        [0. 2. 4.]

    Notes
    -----
    At counting round ``r``, all voters who rank candidate ``c`` in ``r``\ :sup:`th` position gives her an additional
    vote. As soon as at least one candidate has more than :attr:`n_v`/2 votes (accrued with previous rounds),
    the candidate with most votes is declared the winner. In case of a tie, the candidate with lowest index wins.

    * :meth:`is_cm_`: Exact in polynomial time.
    * :meth:`is_icm_`: Exact in polynomial time.
    * :meth:`is_im_`: Exact in polynomial time.
    * :meth:`is_iia_`: Non-polynomial or non-exact algorithms from superclass :class:`Rule`.
    * :meth:`is_tm_`: Exact in polynomial time.
    * :meth:`is_um_`: Exact in polynomial time.

    References
    ----------
    'The Majoritarian Compromise in large societies', Arkadii Slinko, 2002.

    'Complexity of Unweighted Coalitional Manipulation under Some Common Voting Rules', Lirong Xia,
    Michael Zuckerman, Ariel D. Procaccia, Vincent Conitzer and Jeffrey S. Rosenschein, 2009.
    """

    full_name = 'Bucklin'
    abbreviation = 'Buc'

    options_parameters = Rule.options_parameters.copy()
    options_parameters.update({
        'im_option': {'allowed': ['exact'], 'default': 'exact'},
        'tm_option': {'allowed': ['exact'], 'default': 'exact'},
        'um_option': {'allowed': ['exact'], 'default': 'exact'},
        'icm_option': {'allowed': ['exact'], 'default': 'exact'},
        'cm_option': {'allowed': ['exact'], 'default': 'exact'},
    })

    def __init__(self, **kwargs):
        super().__init__(
            with_two_candidates_reduces_to_plurality=True, is_based_on_rk=True,
            precheck_icm=False,  # Bucklin does not meet infmc_c_ctb, but precheck on ICM is not interesting anyway.
            log_identity="BUCKLIN", **kwargs
        )

    @cached_property
    def _count_ballots_(self):
        self.mylog("Count ballots", 1)
        scores = np.zeros((self.profile_.n_c, self.profile_.n_c))
        scores_r = np.zeros(self.profile_.n_c, dtype=np.int)
        w = None
        candidates_by_scores_best_to_worst = None
        for r in range(self.profile_.n_c):
            scores_r += np.bincount(self.profile_.preferences_rk[:, r], minlength=self.profile_.n_c)
            scores[r, :] = np.copy(scores_r)
            if w is None:
                w_r = np.argmax(scores_r)
                if scores_r[w_r] > self.profile_.n_v / 2:
                    w = w_r
                    candidates_by_scores_best_to_worst = np.argsort(- scores_r, kind='mergesort')
        return {'scores': scores, 'w': w, 'candidates_by_scores_best_to_worst': candidates_by_scores_best_to_worst}

    @property
    def scores_(self):
        """2d array of integers. ``scores[r, c]`` is the accrued score of candidate ``c`` at elimination round ``r``.
        It is the number of voters who rank ``c`` between 0\ :sup:`th` and ``r``\ :sup:`th` rank on their ballot.

        For information, ballot are still counted after the round where the decision is made (it is used for
        manipulation algorithms).
        """
        return self._count_ballots_['scores']

    @property
    def w_(self):
        """Integer (winning candidate). When at least one candidate has more than :attr:`n_v`/2 votes, the candidate
        with most votes gets elected. In case of a tie, the candidate with highest index wins.
        """
        return self._count_ballots_['w']

    @property
    def candidates_by_scores_best_to_worst_(self):
        """1d array of integers. Candidates are sorted according to their scores during the counting round during
        which at least one candidate reaches majority.

        By definition, ``candidates_by_scores_best_to_worst_[0]`` = :attr:`w_`.
        """
        return self._count_ballots_['candidates_by_scores_best_to_worst']

    @cached_property
    def v_might_im_for_c_(self):
        # If a voter improves ``c`` from last to first on her ballot, then ``c``'s score gain 1 point each round (except
        # last round ``n_c -  1``, but the decision is always done before). Conversely, if a voter bury ``w`` from first
        # to last, then ``w`` loses 1 point each round (except last round ``n_c - 1``). Modifications by one elector
        # are bounded by that. With such modifications, can a challenger ``c`` do better than ``w`` ? I.e., either
        # reach the majority before, or have more votes than ``w`` when the majority is reached?
        pseudo_scores = np.copy(self.scores_)
        pseudo_scores[:, np.array(range(self.profile_.n_c)) != self.w_] += 1
        pseudo_scores[:, self.w_] -= 1
        self.mylogm('Pseudo-scores =', pseudo_scores)
        r = np.argmax(pseudo_scores[self.w_] > self.profile_.n_v / 2)
        c_has_a_chance = np.zeros(self.profile_.n_c)
        for c in range(self.profile_.n_c):
            if c == self.w_:
                continue
            if r != 0:
                if pseudo_scores[r - 1, c] > self.profile_.n_v / 2:
                    c_has_a_chance[c] = True
            if pseudo_scores[r, c] > pseudo_scores[r, self.w_]:
                c_has_a_chance[c] = True
        return np.tile(np.logical_not(c_has_a_chance), (self.profile_.n_v, 1))

    @cached_property
    def meets_majority_favorite_c_rk(self):
        return True
        # N.B.: majority_favorite_c_ctb is not met.

    # %% Individual manipulation (IM)

    def _im_main_work_v_(self, v, c_is_wanted, nb_wanted_undecided, stop_if_true):
        """
            >>> profile = Profile(preferences_rk=[
            ...     [0, 1, 2],
            ...     [0, 2, 1],
            ...     [0, 2, 1],
            ...     [0, 2, 1],
            ...     [2, 0, 1],
            ... ])
            >>> rule = RuleBucklin()(profile)
            >>> rule.is_im_
            False

            >>> profile = Profile(preferences_rk=[
            ...     [1, 0, 2],
            ...     [2, 0, 1],
            ...     [2, 0, 1],
            ...     [2, 1, 0],
            ...     [2, 1, 0],
            ... ])
            >>> rule = RuleBucklin()(profile)
            >>> rule.is_im_
            False

            >>> profile = Profile(preferences_rk=[
            ...     [1, 2, 0],
            ...     [1, 2, 0],
            ...     [1, 2, 0],
            ...     [2, 0, 1],
            ...     [2, 0, 1],
            ... ])
            >>> rule = RuleBucklin()(profile)
            >>> rule.is_im_
            False

            >>> profile = Profile(preferences_rk=[
            ...     [0, 1, 2],
            ...     [0, 2, 1],
            ...     [0, 2, 1],
            ...     [0, 2, 1],
            ...     [2, 0, 1],
            ... ])
            >>> rule = RuleBucklin()(profile)
            >>> rule.is_im_c_(2)
            False

            >>> profile = Profile(preferences_rk=[
            ...     [0, 1, 2],
            ...     [1, 0, 2],
            ...     [1, 0, 2],
            ...     [1, 2, 0],
            ...     [2, 0, 1],
            ...     [2, 1, 0],
            ... ])
            >>> rule = RuleBucklin()(profile)
            >>> rule.is_im_
            True

            >>> profile = Profile(preferences_rk=[
            ...     [0, 1, 2],
            ...     [1, 0, 2],
            ...     [1, 0, 2],
            ...     [1, 2, 0],
            ...     [2, 0, 1],
            ...     [2, 1, 0],
            ... ])
            >>> rule = RuleBucklin()(profile)
            >>> rule.candidates_im_
            array([1., 0., 0.])

            >>> profile = Profile(preferences_ut=[
            ...     [ 0. ,  1. ,  0.5,  0. ,  0. ],
            ...     [ 0. ,  1. ,  1. , -1. ,  0. ],
            ...     [ 0. , -1. , -0.5,  1. ,  0. ],
            ...     [-0.5, -0.5, -0.5, -1. ,  1. ],
            ...     [ 0. ,  0. , -1. ,  0.5,  1. ],
            ... ], preferences_rk=[
            ...     [1, 2, 3, 0, 4],
            ...     [1, 2, 4, 0, 3],
            ...     [3, 0, 4, 2, 1],
            ...     [4, 0, 2, 1, 3],
            ...     [4, 3, 1, 0, 2],
            ... ])
            >>> rule = RuleBucklin()(profile)
            >>> rule.is_im_
            False
        """
        scores_without_v = np.copy(self.scores_)
        for k in range(self.profile_.n_c):
            scores_without_v[range(k, self.profile_.n_c), self.profile_.preferences_rk[v, k]] -= 1
        for c in range(self.profile_.n_c):
            if not c_is_wanted[c]:
                continue
            if not np.isneginf(self._v_im_for_c[v, c]):
                continue
            nb_wanted_undecided -= 1
            # r : round where c will have majority (with the manipulator).
            r = np.where(scores_without_v[:, c] + 1 > self.profile_.n_v / 2)[0][0]
            if r == 0:  # pragma: no cover
                # TO DO: Investigate whether this case can actually happen.
                self._reached_uncovered_code()
                self._v_im_for_c[v, c] = True
                self._candidates_im[c] = True
                self._voters_im[v] = True
                self._is_im = True
                self.mylog("IM found", 3)
                if stop_if_true:
                    return
                if nb_wanted_undecided == 0:
                    return
                continue
            scores_prev = np.copy(scores_without_v[r - 1, :])
            scores_prev[c] += 1
            scores_r = np.copy(scores_without_v[r, :])
            scores_r[c] += 1
            # Obvious problems
            if np.argmax(scores_r) != c:
                # One d has a better score than c!
                self._v_im_for_c[v, c] = False
                if nb_wanted_undecided == 0:
                    return
                continue  # pragma: no cover - Not really "executed" because of Python optimization
            if np.max(scores_prev) > self.profile_.n_v / 2:
                # One d reaches majority before c!
                self._v_im_for_c[v, c] = False
                if nb_wanted_undecided == 0:
                    return
                continue  # pragma: no cover - Not really "executed" because of Python optimization

            # Now, attribute other ranks in manipulator's ballots. For ``d`` to be added safely at rank ``r``
            # (corresponding to last round), we just need that ``d`` will not outperform ``c`` at rank ``r``.
            d_can_be_added = np.less(scores_r + 1, scores_r[c] + (c < np.array(range(self.profile_.n_c))))
            d_can_be_added[c] = False
            # For ``d`` to be added safely before rank ``r``, we need also that ``d`` will not have a majority before
            # round ``r``.
            d_can_be_added_before_last_round = np.logical_and(d_can_be_added, scores_prev + 1 <= self.profile_.n_v / 2)

            # We can conclude.
            self._v_im_for_c[v, c] = (np.sum(d_can_be_added) >= r - 1
                                      and np.sum(d_can_be_added_before_last_round) >= r - 2)
            if equal_true(self._v_im_for_c[v, c]):
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

    # %% Trivial Manipulation (TM)

    # Defined in the superclass Rule.

    # %% Unison manipulation (UM)

    def _um_main_work_c_(self, c):
        """
            >>> profile = Profile(preferences_rk=[
            ...     [1, 3, 0, 2],
            ...     [2, 0, 3, 1],
            ...     [2, 1, 3, 0],
            ...     [3, 1, 0, 2],
            ...     [3, 2, 1, 0],
            ... ])
            >>> rule = RuleBucklin()(profile)
            >>> rule.candidates_um_
            array([0., 0., 1., 1.])

            >>> profile = Profile(preferences_rk=[
            ...     [1, 2, 0, 3],
            ...     [1, 3, 2, 0],
            ...     [2, 0, 3, 1],
            ...     [2, 1, 3, 0],
            ...     [3, 2, 0, 1],
            ... ])
            >>> rule = RuleBucklin()(profile)
            >>> rule.candidates_um_
            array([0., 1., 0., 0.])
        """
        n_m = self.profile_.matrix_duels_ut[c, self.w_]
        scores_r = np.zeros(self.profile_.n_c)
        # Manipulators put ``c`` in first position anyway.
        scores_r[c] = n_m
        # Look for the round ``r`` where ``c`` has a majority. Compute how many sincere votes each candidate will
        # have at that time, + the ``n_m`` manipulating votes that we know for ``c``.
        r = None
        scores_prev = None
        for r in range(self.profile_.n_c):
            scores_prev = np.copy(scores_r)
            scores_r += np.bincount(self.profile_.preferences_rk[np.logical_not(self.v_wants_to_help_c_[:, c]), r],
                                    minlength=self.profile_.n_c)
            if scores_r[c] > self.profile_.n_v / 2:  # It is the last round
                if np.argmax(scores_r) != c:
                    # One ``d`` has a better score than ``c``!
                    self._candidates_um[c] = False
                    return
                break
            if np.max(scores_r) > self.profile_.n_v / 2:
                # One d reaches majority before c!
                self._candidates_um[c] = False
                return

        # Now, attribute other ranks in manipulator's ballots. For ``d`` to be added safely at rank ``r``
        # (corresponding to last round), we just need that ``d`` will not outperform ``c`` at rank ``r``.
        d_can_be_added = np.less(scores_r + n_m, scores_r[c] + (c < np.array(range(self.profile_.n_c))))
        d_can_be_added[c] = False
        # For ``d`` to be added safely before rank ``r``, we need also that ``d`` will not have a majority before
        # round ``r``.
        d_can_be_added_before_last_round = np.logical_and(d_can_be_added, scores_prev + n_m <= self.profile_.n_v / 2)

        # We can conclude.
        self._candidates_um[c] = (np.sum(d_can_be_added) >= r - 1
                                  and np.sum(d_can_be_added_before_last_round) >= r - 2)

    # %% Ignorant-Coalition Manipulation (ICM)

    def _icm_main_work_c_exact_(self, c, complete_mode=True):
        """
            >>> profile = Profile(preferences_ut=[
            ...     [ 0. ,  0.5],
            ...     [-0.5,  0.5],
            ... ], preferences_rk=[
            ...     [1, 0],
            ...     [1, 0],
            ... ])
            >>> rule = RuleBucklin()(profile)
            >>> rule.sufficient_coalition_size_icm_
            array([2., 0.])
        """
        # The only question is when we have exactly ``n_v / 2`` manipulators. If the counter-manipulators put ``c``
        # last, then ``c`` cannot be elected (except if there are 2 candidates and ``c == 0``). So exactly ``n_v / 2``
        # manipulators is not enough.
        n_s = self.profile_.n_v - self.profile_.matrix_duels_ut[c, self.w_]
        if self.profile_.n_c == 2 and c == 0:
            self._update_sufficient(self._sufficient_coalition_size_icm, c, n_s,
                                    'ICM: Tie-breaking: sufficient_coalition_size_icm = n_s =')
        else:
            self._update_necessary(self._necessary_coalition_size_icm, c, n_s + 1,
                                   'ICM: Tie-breaking: necessary_coalition_size_icm = n_s + 1 =')

    # %% Coalition Manipulation (CM)

    def _cm_main_work_c_exact_(self, c, optimize_bounds):
        """
            >>> profile = Profile(preferences_rk=[
            ...     [0, 1, 2],
            ...     [0, 1, 2],
            ...     [1, 0, 2],
            ...     [1, 2, 0],
            ...     [2, 1, 0],
            ... ])
            >>> rule = RuleBucklin()(profile)
            >>> rule.candidates_cm_
            array([1., 0., 0.])

            >>> profile = Profile(preferences_ut=[
            ...     [ 1. ,  0. ,  1. ,  0. ],
            ...     [ 1. ,  0. ,  1. ,  0. ],
            ...     [ 0. , -1. ,  1. ,  0. ],
            ...     [-0.5, -1. ,  0. ,  0.5],
            ... ], preferences_rk=[
            ...     [2, 0, 1, 3],
            ...     [2, 0, 3, 1],
            ...     [2, 3, 0, 1],
            ...     [3, 2, 0, 1],
            ... ])
            >>> rule = RuleBucklin()(profile)
            >>> rule.necessary_coalition_size_cm_
            array([2., 2., 0., 3.])

            >>> profile = Profile(preferences_ut=[
            ...     [ 1. ,  0.5, -1. , -1. ],
            ...     [ 0. ,  0.5, -0.5, -0.5],
            ...     [ 0.5, -1. ,  1. ,  0. ],
            ...     [ 0.5, -0.5,  0. ,  1. ],
            ...     [ 0. ,  0. ,  0. ,  1. ],
            ...     [-1. , -1. , -0.5,  1. ],
            ... ], preferences_rk=[
            ...     [0, 1, 3, 2],
            ...     [1, 0, 3, 2],
            ...     [2, 0, 3, 1],
            ...     [3, 0, 2, 1],
            ...     [3, 1, 2, 0],
            ...     [3, 2, 0, 1],
            ... ])
            >>> rule = RuleBucklin()(profile)
            >>> rule.sufficient_coalition_size_cm_
            array([0., 6., 5., 3.])
        """
        # We do not try to find optimal bounds. We just check whether it is possible to manipulate with the number of
        #  manipulators that we have.
        n_m = self.profile_.matrix_duels_ut[c, self.w_]
        if n_m < self._necessary_coalition_size_cm[c]:
            # This algorithm will not do better (so, this is not a quick escape).
            return
        if n_m >= self._sufficient_coalition_size_cm[c]:
            # Idem.
            return
        scores_r = np.zeros(self.profile_.n_c)
        # Manipulators put ``c`` in first position anyway.
        scores_r[c] = n_m
        # Look for the round ``r`` where ``c`` has a majority. Compute how many sincere votes each candidate will have
        # at that time, + the ``n_m`` manipulating votes that we know for ``c``.
        r = None
        scores_prev = None
        for r in range(self.profile_.n_c):
            scores_prev = np.copy(scores_r)
            scores_r += np.bincount(self.profile_.preferences_rk[np.logical_not(self.v_wants_to_help_c_[:, c]), r],
                                    minlength=self.profile_.n_c)
            if scores_r[c] > self.profile_.n_v / 2:
                break

        votes_can_be_added_before_last_r = np.zeros(self.profile_.n_c)
        votes_can_be_added = np.zeros(self.profile_.n_c)
        one_d_beats_c_anyway = False
        for d in range(self.profile_.n_c):
            if d == c:
                continue
            if scores_r[d] + (d < c) > scores_r[c] or scores_prev[d] > self.profile_.n_v / 2:
                one_d_beats_c_anyway = True
                break
            votes_can_be_added[d] = min(scores_r[c] - (d < c) - scores_r[d], n_m)
            votes_can_be_added_before_last_r[d] = min(np.floor(self.profile_.n_v / 2) - scores_prev[d],
                                                      votes_can_be_added[d])

        if (not one_d_beats_c_anyway and sum(votes_can_be_added) >= (r - 1) * n_m
                and sum(votes_can_be_added_before_last_r) >= (r - 2) * n_m):
            self._update_sufficient(self._sufficient_coalition_size_cm, c, n_m,
                                    'CM: Exact: Manipulation found for n_m manipulators =>\n'
                                    '    sufficient_coalition_size_cm = n_m =')
        else:
            self._update_necessary(self._necessary_coalition_size_cm, c, n_m + 1,
                                   'CM: Exact: Manipulation proven impossible for n_m manipulators =>\n'
                                   '    necessary_coalition_size_cm[c] = n_m + 1 =')
