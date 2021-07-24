# -*- coding: utf-8 -*-
"""
Created on 4 dec. 2018, 13:48
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


class RuleBorda(Rule):
    """Borda rule.

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
        >>> rule = RuleBorda()(profile)
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
        [[2 1 0]
         [2 0 1]
         [1 2 0]
         [1 0 2]
         [0 1 2]]
        scores =
        [6 4 5]
        candidates_by_scores_best_to_worst
        [0 2 1]
        scores_best_to_worst
        [6 5 4]
        w = 0
        score_w = 6
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
        log_icm: icm_option = fast
        candidates_icm =
        [0. 0. 0.]
        necessary_coalition_size_icm =
        [0. 7. 5.]
        sufficient_coalition_size_icm =
        [0. 7. 5.]
        <BLANKLINE>
        ***********************************
        *   Coalition Manipulation (CM)   *
        ***********************************
        is_cm = False
        log_cm: cm_option = fast, um_option = exact, tm_option = exact
        candidates_cm =
        [0. 0. 0.]
        necessary_coalition_size_cm =
        [0. 2. 3.]
        sufficient_coalition_size_cm =
        [0. 2. 3.]

    Notes
    -----
    Voter ``v`` gives (:attr:`n_c` - 1) points to her top-ranked candidate, (:attr:`n_c` - 2) to the second, ..., 0 to
    the last. Ties are broken by natural order on the candidates (lower index wins).

    * :meth:`is_cm_`: Deciding CM is NP-complete.

        * :attr:`cm_option` = ``'fast'``: Zuckerman et al. (2009). This approximation algorithm is polynomial and has
          a window of error of 1 manipulator.
        * :attr:`cm_option` = ``'exact'``: Non-polynomial algorithm from superclass :class:`Rule`.

    * :meth:`is_icm_`: Algorithm is polynomial and has a window of error of 1 manipulator.
    * :meth:`is_im_`: Exact in polynomial time.
    * :meth:`is_iia_`: Exact in polynomial time.
    * :meth:`is_tm_`: Exact in polynomial time.
    * :meth:`is_um_`: Exact in polynomial time.

    References
    ----------
    'Algorithms for the coalitional manipulation problem', M. Zuckerman, A. Procaccia and J. Rosenschein, 2009.

    'Unweighted Coalitional Manipulation Under the Borda Rule is NP-Hard', Nadja Betzler, Rolf Niedermeier and
    Gerhard Woeginger, 2011.

    'Complexity of and algorithms for the manipulation of Borda, Nanson's and Baldwin's voting rules',
    Jessica Davies, George Katsirelos, Nina Narodytska, Toby Walsh and Lirong Xia, 2014.
    """

    full_name = 'Borda'
    abbreviation = 'Bor'

    options_parameters = Rule.options_parameters.copy()
    options_parameters.update({
        'im_option': {'allowed': ['exact'], 'default': 'exact'},
        'tm_option': {'allowed': ['exact'], 'default': 'exact'},
        'um_option': {'allowed': ['exact'], 'default': 'exact'},
        'icm_option': {'allowed': ['fast'], 'default': 'fast'},
        'cm_option': {'allowed': ['fast', 'exact'], 'default': 'fast'},
    })

    def __init__(self, **kwargs):
        super().__init__(
            with_two_candidates_reduces_to_plurality=True, is_based_on_rk=True,
            precheck_icm=False, log_identity="BORDA", **kwargs
        )

    @cached_property
    def ballots_(self):
        """2d array of integers. ``ballots[v, c]`` is the Borda score attributed to candidate ``c`` by voter ``v``
        (between 0 and ``n_c - 1``).
        """
        self.mylog("Compute ballots", 1)
        return self.profile_.preferences_borda_rk

    @cached_property
    def scores_(self):
        """1d array of integers. ``scores[c]`` is the total Borda score for candidate ``c``.
        """
        self.mylog("Compute scores", 1)
        return self.profile_.borda_score_c_rk

    @cached_property
    def meets_infmc_c_ctb(self):
        return True

    # %% Independence of Irrelevant Alternatives (IIA)

    @cached_property
    def _compute_iia_(self):
        """
            >>> profile = Profile(preferences_rk=[
            ...     [0, 1, 2],
            ...     [1, 0, 2],
            ...     [1, 2, 0],
            ...     [2, 0, 1],
            ...     [2, 0, 1],
            ... ])
            >>> rule = RuleBorda()(profile)
            >>> rule.is_iia_
            False
        """
        self.mylog("Compute IIA", 1)
        # If we remove candidate ``d``, candidate ``c`` loses ``matrix_duels_rk[c, d]`` Borda points and ``w`` loses
        # ``matrix_duels_rk[w, d]``. So, candidate ``c`` gains the difference, when compared to ``w``.
        impact_removal_d_on_c = np.add(- self.profile_.matrix_duels_rk,
                                       self.profile_.matrix_duels_rk[self.w_, :][np.newaxis, :])
        impact_removal_d_on_c[np.diag_indices(self.profile_.n_c)] = 0
        impact_removal_d_on_c[:, self.w_] = 0  # Forbidden to remove w
        # For ``c``, any candidate ``d`` s.t. ``impact_removal_d_on_c[c, d] > 0`` is a good choice to remove.
        # ``pseudo_scores[c]`` is ``c``'s score minus ``w``'s score after these removals. After all these removals,
        # it may be that another candidate ``e`` will have a better score than ``c``, but we don't care: the only
        # important thing is to make ``w`` lose.
        pseudo_scores = (self.scores_ - self.score_w_ + np.sum(np.maximum(impact_removal_d_on_c, 0), 1))
        self.mylogv("pseudo_scores =", pseudo_scores, 2)
        # If ``pseudo_scores[c] > 0`` for some ``c``, then IIA is broken.
        best_c = np.argmax(pseudo_scores)
        self.mylogv("best_c =", best_c, 2)
        if pseudo_scores[best_c] + (best_c < self.w_) > 0:
            # Since we chose the best ``c``, we know that she is the winner in her optimal subset.
            return {'is_iia': False, 'example_winner_iia': best_c,
                    'example_subset_iia': np.logical_not(impact_removal_d_on_c[best_c, :] > 0)}
        else:
            return {'is_iia': True, 'example_winner_iia': np.nan, 'example_subset_iia': np.nan}

    # %% Individual manipulation (IM)

    def _im_preliminary_checks_general_subclass_(self):
        # If we change the ballot of only one ``c``-manipulator ``v``, then ``score[c] - score[w]`` increases by at
        # most ``n_c - 2``. Indeed, c gave at least 1 point of difference to ``c`` vs ``w`` (she strictly prefers ``c``
        # to ``w``), and now she can give  at most ``n_c - 1`` points of difference (with ``c`` on top and ``w`` at
        # the bottom).
        self.mylogv('Preliminary checks: scores =', self.scores_)
        self.mylogv('Preliminary checks: scores_w =', self.score_w_)
        temp = self.scores_ + self.profile_.n_c - 2 + (np.array(range(self.profile_.n_c)) < self.w_)
        self.mylogm('Preliminary checks: scores + n_c - 2 + (c < w) =', temp)
        # self.mylogm('Preliminary checks: possible candidates =', temp > self.scores[self.w])
        self._v_im_for_c[:, temp <= self.scores_[self.w_]] = False

    def _im_main_work_v_(self, v, c_is_wanted, nb_wanted_undecided, stop_if_true):
        """
            >>> profile = Profile(preferences_rk=[
            ...     [0, 2, 1],
            ...     [0, 2, 1],
            ...     [1, 0, 2],
            ...     [2, 0, 1],
            ...     [2, 1, 0],
            ... ])
            >>> rule = RuleBorda()(profile)
            >>> rule.v_im_for_c_
            array([[0., 0., 0.],
                   [0., 0., 0.],
                   [0., 0., 0.],
                   [0., 0., 1.],
                   [0., 0., 0.]])

            >>> profile = Profile(preferences_rk=[
            ...     [0, 2, 1],
            ...     [0, 2, 1],
            ...     [1, 2, 0],
            ...     [1, 2, 0],
            ...     [2, 0, 1],
            ... ])
            >>> rule = RuleBorda()(profile)
            >>> rule.is_im_c_(0)
            True

            >>> profile = Profile(preferences_rk=[
            ...     [0, 1, 2],
            ...     [0, 2, 1],
            ...     [2, 0, 1],
            ...     [2, 0, 1],
            ...     [2, 1, 0],
            ... ])
            >>> rule = RuleBorda()(profile)
            >>> rule.is_im_c_(0)
            True

            >>> profile = Profile(preferences_rk=[
            ...     [0, 2, 1],
            ...     [0, 2, 1],
            ...     [1, 0, 2],
            ...     [2, 0, 1],
            ...     [2, 1, 0],
            ... ])
            >>> rule = RuleBorda()(profile)
            >>> rule.is_im_c_(2)
            True
        """
        scores_without_v = self.scores_ - self.ballots_[v, :]
        self.mylogv('scores_without_v =', scores_without_v)
        candidates_by_decreasing_score = np.argsort(- scores_without_v, kind='mergesort')
        # ``ballot_balance`` is the ballot that ``v`` must give to balance the scores of all candidates the best she
        # can.
        ballot_balance = np.argsort(candidates_by_decreasing_score)
        self.mylogv('ballot_balance =', ballot_balance, 3)
        for c in range(self.profile_.n_c):
            if not c_is_wanted[c]:
                continue
            if not np.isneginf(self._v_im_for_c[v, c]):
                continue
            self.mylogv('Candidate c =', c, 3)
            # Compared to ``ballot_balance``, the candidates that ``v`` was putting better than ``c`` lost 1 point.
            ballot = ballot_balance - np.greater(ballot_balance, ballot_balance[c])
            # Voter v gives the maximal score to c.
            ballot[c] = self.profile_.n_c - 1
            self.mylogv('ballot =', ballot, 3)
            w_test = np.argmax(scores_without_v + ballot)
            self.mylogv('w_test =', w_test, 3)
            if w_test == c:
                self._v_im_for_c[v, c] = True
                self._candidates_im[c] = True
                self._voters_im[v] = True
                self._is_im = True
                self.mylog("IM found", 3)
                if stop_if_true:
                    return
            else:
                self._v_im_for_c[v, c] = False
            nb_wanted_undecided -= 1
            if nb_wanted_undecided == 0:
                return

    # %% Trivial Manipulation (TM)

    # Defined in the superclass.

    # %% Unison manipulation (UM)

    def _um_main_work_c_exact_(self, c):
        scores_test = np.sum(self.ballots_[np.logical_not(self.v_wants_to_help_c_[:, c]), :], 0)
        candidates_by_decreasing_score = np.argsort(-scores_test, kind='mergesort')
        # Balancing ballot: put candidates in the order of their current scores (least point to the most dangerous).
        ballot = np.argsort(candidates_by_decreasing_score)
        # Now put c on top. And modify other Borda scores accordingly on the ballot.
        ballot -= np.greater(ballot, ballot[c])
        ballot[c] = self.profile_.n_c - 1
        # New scores = old scores + n_manipulators * ballot.
        scores_test += np.multiply(self.profile_.matrix_duels_ut[c, self.w_], ballot)
        w_test = np.argmax(scores_test)
        self._candidates_um[c] = (w_test == c)

    # %% Ignorant-Coalition Manipulation (ICM)

    # noinspection PyUnusedLocal
    def _icm_main_work_c_fast_(self, c, optimize_bounds):
        # n_s: number of sincere voters (= counter-manipulators)
        n_s = self.profile_.n_v - self.profile_.matrix_duels_ut[c, self.w_]
        n_c = self.profile_.n_c
        # Necessary condition:
        # ^^^^^^^^^^^^^^^^^^^^
        # Denote ``n_m`` the number of manipulators for ``c``. Each manipulator gives ``n_c - 1`` points to ``c`` and an
        # average of ``(n_c - 2) / 2`` points to each other candidate. Let us note ``d`` the other candidate with
        # maximum ``score_m`` (from the manipulators).
        # ``score_m[c] = n_m * (n_c - 1)``.
        # ``score_m[d] >= n_m * (n_c - 2) / 2``.
        # At worst, each counter-manipulator will give ``n_c - 1`` points to ``d`` and 0 point to ``c``.
        # ``score_total[c] = n_m * (n_c - 1)``
        # ``score_total[d] >= n_m * (n_c - 2) / 2 + n_s * (n_c - 1)``
        # Since these scores may be half-integers, we will multiply them by 2 to apply the tie=breaking rule
        # correctly. We need:
        # ``2 * score_total[c] >= 2 * score_total[d] + (c > 0)``, which leads to:
        # ``n_m >= (2 * n_s * (n_c - 1) + (c > 0)) / n_c``.
        necessary = np.ceil((2 * n_s * (n_c - 1) + (c > 0)) / n_c)
        # Sufficient condition for n_m = 2 * p
        # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        # Manipulators can give ``n_c - 1`` points to ``c``. Other candidates have balanced scores (ballot [n_c - 2,
        # ..., 0] paired with [0, ..., n_c - 2]).
        # ``score_m[c] = 2p (n_c - 1)``
        # ``score_m[d] = 2p (n_c - 2) / 2``
        # After counter-manipulation:
        # ``score_total[c] = 2p (n_c - 1)``
        # ``score_total[d] = 2p (n_c - 2) / 2 + n_s (n_c - 1)``
        # For ICM, it is sufficient that ``score_total[c] >= score_total[d] + (c > 0)``, which translates as before to:
        # ``p >= (n_s (n_c - 1) + (c > 0)) / n_c``.
        # N.B.: From the general inequality ``2 ceil(x) <= ceil (2x)``, it can be deduced that
        # ``sufficient_even - necessary <= 1``.
        sufficient_even = 2 * np.ceil((n_s * (n_c - 1) + (c > 0)) / n_c)
        # Sufficient condition for n_m = 2 * p + 1
        # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        # Manipulators can give ``n_c - 1`` points to ``c``. With the first ``2p`` manipulators, other candidates
        # have balanced scores as before. In addition, the last manipulator gives ``n_c - 2`` points to the opponent
        # candidate with the higher index (so that she benefits less from t.b. rule).
        # ``score_m[c] = (2p + 1) (n_c - 1)``
        # ``score_m[d] = 2p (n_c - 2) / 2 + (n_c - 2) = (p + 1) (n_c - 2)``
        # After counter-manipulation:
        # ``score_total[c] = (2p + 1) (n_c - 1)``
        # ``score_total[d] = (p + 1) (n_c - 2) + n_s (n_c - 1)``
        # For ICM, it is sufficient that ``score_total[c] >= score_total[d] + (c == n_c - 1)`` (the tie-breaking rule
        # is against ``c`` only if she is the last candidate). This translates to:
        # ``p >= (n_s (n_c - 1) - 1 + (c == n_c - 1)) / n_c``.
        sufficient_odd = 1 + 2 * np.ceil((n_s * (n_c - 1) - 1 + (c == n_c-1)) / n_c)
        self._update_necessary(self._necessary_coalition_size_icm, c, necessary)
        self._update_sufficient(self._sufficient_coalition_size_icm, c, sufficient_even)
        self._update_sufficient(self._sufficient_coalition_size_icm, c, sufficient_odd)

    # %% Coalition Manipulation (CM)

    def _cm_main_work_c_fast_(self, c, optimize_bounds):
        """Do the main work in CM loop for candidate ``c``.

        * Try to improve bounds ``_sufficient_coalition_size_cm[c]`` and ``_necessary_coalition_size_cm[c]``.

        Algorithm: Zuckerman et al. (2009), Algorithms for the coalitional manipulation problem.

            >>> profile = Profile(preferences_rk=[
            ...     [0, 2, 1],
            ...     [0, 2, 1],
            ...     [1, 0, 2],
            ...     [1, 0, 2],
            ...     [2, 0, 1],
            ... ])
            >>> rule = RuleBorda()(profile)
            >>> rule.candidates_cm_
            array([0., 0., 0.])
        """
        scores_test = np.sum(self.ballots_[np.logical_not(self.v_wants_to_help_c_[:, c]), :], 0)
        # We add a tie-breaking term [(C-1)/C, (C-2)/C, ..., 0] to ease the computations.
        scores_test = scores_test + (np.array(range(self.profile_.n_c - 1, -1, -1)) / self.profile_.n_c)
        self.mylogv('CM: Further check: scores_test =', scores_test, 3)

        # A family of necessary conditions
        # For ``k`` in ``{1, ..., n_c - 1}``, let us note ``mean_k`` the mean of the ``k`` highest scores among other
        # candidates than ``c``. At best, we will have:
        # ``scores_tot[c] = scores_test[c] + n_m (n_c - 1)``
        # ``mean_k_tot = mean_k_test + n_m (k - 1) / 2``
        # We must have ``scores_tot[c] >= mean_k_tot`` so:
        # ``n_m >= 2 * (mean_k_test - scores_test[c]) / (2 n_c - k - 1)``
        increasing_scores = np.sort(scores_test[np.array(range(self.profile_.n_c)) != c])
        cum_sum = 0
        for k in range(1, self.profile_.n_c):
            cum_sum += increasing_scores[-k]
            necessary = np.ceil(2 * (cum_sum / k - scores_test[c]) / (2 * self.profile_.n_c - k - 1))
            self.mylogv('CM: Further check: k =', k)
            self.mylogv('CM: Further check: cum_sum =', cum_sum)
            self.mylogv('CM: Further check: necessary =', necessary)
            self._update_necessary(self._necessary_coalition_size_cm, c, necessary,
                                   'CM: Further check: necessary_coalition_size_cm =')
        # An opportunity to escape before real work
        if self._necessary_coalition_size_cm[c] == self._sufficient_coalition_size_cm[c]:
            return False
        if not optimize_bounds and self.profile_.matrix_duels_ut[c, self.w_] < self._necessary_coalition_size_cm[c]:
            # This is a quick escape: we have not optimized the bounds the best we could.
            return True

        # Now, the real work
        n_m = 0
        self.mylogv('CM: Fast algorithm: scores_test =', scores_test, 3)
        while True:
            n_m += 1
            # Balancing ballot: put candidates in the order of their current scores (least point to the most
            # dangerous).
            candidates_by_decreasing_score = np.argsort(- scores_test, kind='mergesort')
            ballot = np.argsort(candidates_by_decreasing_score)
            # Now put ``c`` on top. And modify other Borda scores accordingly on the ballot.
            ballot -= np.greater(ballot, ballot[c])
            ballot[c] = self.profile_.n_c - 1
            self.mylogv('CM: Fast algorithm: ballot =', ballot, 3)
            # New scores = old scores + ballot.
            scores_test += ballot
            self.mylogv('CM: Fast algorithm: scores_test =', scores_test, 3)
            w_test = np.argmax(scores_test)
            if w_test == c:
                sufficient = n_m
                self.mylogv('CM: Fast algorithm: sufficient =', sufficient)
                self._update_sufficient(self._sufficient_coalition_size_cm, c, sufficient,
                                        'CM: Fast algorithm: sufficient_coalition_size_cm =')
                # From Zuckerman et al.: the algorithm is optimal up to 1 voter. But since we do not have the same
                # tie-breaking rule, it makes 2 voters.
                self._update_necessary(self._necessary_coalition_size_cm, c, sufficient - 2,
                                       'CM: Fast algorithm: necessary_coalition_size_cm =')
                break

    def _cm_main_work_c_(self, c, optimize_bounds):
        """
            >>> profile = Profile(preferences_rk=[
            ...     [0, 1, 2],
            ...     [0, 2, 1],
            ...     [0, 2, 1],
            ...     [0, 2, 1],
            ...     [2, 0, 1],
            ... ])
            >>> rule = RuleBorda(cm_option='exact')(profile)
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
