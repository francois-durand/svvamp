# -*- coding: utf-8 -*-
"""
Created on 7 dec. 2018, 16:17
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
from svvamp.utils.misc import preferences_ut_to_matrix_duels_ut
from svvamp.preferences.profile import Profile


class RuleIRVDuels(Rule):
    """IRV with elimination duels. Also known as Viennot rule.

    Options
    -------
        >>> RuleIRVDuels.print_options_parameters()
        cm_option: ['lazy', 'exact']. Default: 'lazy'.
        icm_option: ['exact']. Default: 'exact'.
        iia_subset_maximum_size: is_number. Default: 2.
        im_option: ['lazy', 'exact']. Default: 'lazy'.
        tm_option: ['lazy', 'exact']. Default: 'exact'.
        um_option: ['lazy', 'exact']. Default: 'lazy'.

    Notes
    -----
    Principle: each round, perform a duel between the two least-favorite candidates and eliminate the loser of this
    duel.

    * Even round ``r`` (including round 0): the two non-eliminated candidates who are ranked first (among the
      non-eliminated candidates) by least voters are selected for the elimination duels that is held in round
      ``r + 1``.
    * Odd round ``r``: voters vote for the selected candidate they like most in the duel. The candidate with least
      votes is eliminated.

    This method meets the Condorcet criterion.

    We thank Laurent Viennot for the idea of this voting system.

    * :meth:`is_cm_`: Non-polynomial or non-exact algorithms from superclass :class:`Rule`.
    * :meth:`is_icm_`: Exact in polynomial time.
    * :meth:`is_im_`: Non-polynomial or non-exact algorithms from superclass :class:`Rule`.
    * :meth:`~svvamp.Election.not_iia`: Exact in polynomial time.
    * :meth:`is_tm_`: Exact in polynomial time.
    * :meth:`is_um_`: Non-polynomial or non-exact algorithms from superclass :class:`Rule`.

    See Also
    --------
    :class:`RuleExhaustiveBallot`, :class:`RuleIRV`, :class:`RuleICRV`, :class:`RuleCondorcetAbsIRV`,
    :class:`RuleCondorcetVtbIRV`.

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
        >>> rule = RuleIRVDuels()(profile)
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
        [[ 2.  1.  2.]
         [nan  2.  3.]
         [ 3. nan  2.]
         [ 3. nan  2.]]
        candidates_by_scores_best_to_worst
        [0 2 1]
        scores_best_to_worst
        [[ 2.  2.  1.]
         [nan  3.  2.]
         [ 3.  2. nan]
         [ 3.  2. nan]]
        w = 0
        score_w = [ 2. nan  3.  3.]
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
        log_cm: cm_option = lazy, um_option = lazy, tm_option = exact
        candidates_cm =
        [ 0.  0. nan]
        necessary_coalition_size_cm =
        [0. 1. 2.]
        sufficient_coalition_size_cm =
        [0. 2. 3.]
    """

    full_name = 'IRV-Duels'
    abbreviation = 'IRVD'

    options_parameters = Rule.options_parameters.copy()
    options_parameters['icm_option'] = {'allowed': ['exact'], 'default': 'exact'}
    options_parameters['cm_option'] = {'allowed': ['lazy', 'exact'], 'default': 'lazy'}

    def __init__(self, **kwargs):
        super().__init__(
            with_two_candidates_reduces_to_plurality=True, is_based_on_rk=True,
            precheck_icm=False,
            log_identity="IRV_DUELS", **kwargs
        )

    # %% Counting the ballots

    @cached_property
    def _count_ballots_(self):
        """
            >>> profile = Profile(preferences_rk=[
            ...     [0, 2, 1],
            ...     [0, 2, 1],
            ...     [1, 2, 0],
            ... ])
            >>> rule = RuleIRVDuels()(profile)
            >>> rule.scores_
            array([[ 2.,  1.,  0.],
                   [nan,  1.,  2.],
                   [ 2., nan,  1.],
                   [ 2., nan,  1.]])
        """
        self.mylog("Count ballots", 1)
        scores = np.full((2 * self.profile_.n_c - 2, self.profile_.n_c), np.nan)
        is_candidate_alive = np.ones(self.profile_.n_c, dtype=bool)
        worst_to_best = []
        for r in range(self.profile_.n_c - 1):
            # Select the two worst scores
            scores[2*r, is_candidate_alive] = np.sum(
                self.profile_.preferences_borda_rk[:, is_candidate_alive]
                == np.max(self.profile_.preferences_borda_rk[:, is_candidate_alive], 1)[:, np.newaxis],
                0)
            selected_one = np.where(scores[2 * r, :] == np.nanmin(scores[2 * r, :]))[0][-1]
            score_selected_one = scores[2*r, selected_one]
            scores[2*r, selected_one] = np.inf
            selected_two = np.where(scores[2 * r, :] == np.nanmin(scores[2 * r, :]))[0][-1]
            scores[2*r, selected_one] = score_selected_one
            self.mylogv("selected_one =", selected_one, 3)
            self.mylogv("selected_two =", selected_two, 3)
            # Do the duel
            scores[2*r + 1, selected_one] = self.profile_.matrix_duels_rk[selected_one, selected_two]
            scores[2*r + 1, selected_two] = self.profile_.matrix_duels_rk[selected_two, selected_one]
            if self.profile_.matrix_victories_rk_ctb[selected_one, selected_two] == 1:
                loser = selected_two
            else:
                loser = selected_one
            is_candidate_alive[loser] = False
            worst_to_best.append(loser)
        w = np.argmax(is_candidate_alive)
        worst_to_best.append(w)
        candidates_by_scores_best_to_worst = np.array(worst_to_best[::-1])
        return {'scores': scores, 'w': w, 'candidates_by_scores_best_to_worst': candidates_by_scores_best_to_worst}

    @cached_property
    def w_(self):
        self.mylog("Count ballots", 1)
        plurality_elimination_engine = self.profile_.plurality_elimination_engine()
        for r in range(self.profile_.n_c - 1):
            scores = plurality_elimination_engine.scores.copy()
            selected_one = np.where(scores == np.nanmin(scores))[0][-1]
            scores[selected_one] = np.nan
            selected_two = np.where(scores == np.nanmin(scores))[0][-1]
            if self.profile_.matrix_victories_rk_ctb[selected_one, selected_two] == 1:
                loser = selected_two
            else:
                loser = selected_one
            plurality_elimination_engine.eliminate_candidate_and_update_scores(loser)
        return int(plurality_elimination_engine.candidates_alive[0])

    @cached_property
    def scores_(self):
        """2d array.

        * For even rounds ``r`` (including round 0), ``scores[r, c]`` is the number of voters who rank ``c`` first
          (among non-eliminated candidates).
        * For odd rounds ``r``, only the two candidates who are selected for the elimination duels get scores.
          ``scores[r, c]`` is the number of voters who vote for ``c`` in this elimination duel.
        """
        return self._count_ballots_['scores']

    @cached_property
    def candidates_by_scores_best_to_worst_(self):
        """1d array of integers. Candidates are sorted in the reverse order of their elimination.
        """
        return self._count_ballots_['candidates_by_scores_best_to_worst']

    # TODO: self.v_might_im_for_c

    @cached_property
    def meets_condorcet_c_rk_ctb(self):
        return True

    # %% Coalition Manipulation (CM)

    def _cm_aux_exact(self, c, n_max, n_min, optimize_bounds, preferences_borda_s, matrix_duels_s):
        """Exact algorithm used for CM.

        Parameters
        ----------
        c : int
            Candidate for which we want to manipulate.
        n_max : int
            Maximum number of manipulators allowed.

            * CM, optimize_bounds and exact --> put the current value of ``sufficient_coalition_size[c] - 1`` (we want
              to find the best value for ``sufficient_coalition_size[c]``, even if it exceeds the number of
              manipulators)
            * CM, otherwise --> put the number of manipulators.

        n_min : int
            When we know that ``n_min`` manipulators are needed (necessary coalition size).
        optimize_bounds : bool
            True iff we need to continue, even after a manipulation is found.
        preferences_borda_s : ndarray
            Preferences of the sincere voters (in Borda format).
        matrix_duels_s : ndarray
            Matrix of duels of the sincere voters.

        Returns
        -------
        tuple
            ``(n_manip_final, example_path_losers, example_path_winners, quick_escape)``.

            * ``n_manip_final``: Integer or +inf. If manipulation is impossible with ``<= n_max`` manipulators, it is
              +inf. If manipulation is possible (with ``<= n_max``):

                * If ``optimize_bounds`` is True, it is the minimal number of manipulators.
                * Otherwise, it is a number of manipulators that allow this manipulation (not necessarily minimal).

            * ``example_path_losers``: An example of elimination path that realizes the manipulation with
              ``n_manip_final`` manipulators. ``example_path[k]`` is the ``k``-th candidate eliminated. If the
              manipulation is impossible, ``example_path_losers`` is NaN.
            * ``example_path_winners``: An example of elimination path that realizes the manipulation with
              ``n_manip_final`` manipulators. ``example_path[k]`` is the candidate selected for the ``k``-th duel
              and winning. If the manipulation is impossible, ``example_path_winners`` is NaN.
            * ``quick_escape``: Boolean. True if we get out without optimizing ``n_manip_final``.

        Examples
        --------
            >>> profile = Profile(preferences_rk=[
            ...     [1, 0, 2],
            ...     [2, 0, 1],
            ...     [2, 1, 0],
            ...     [1, 0, 2],
            ... ])
            >>> rule = RuleIRVDuels(cm_option='exact')(profile)
            >>> rule.sufficient_coalition_size_cm_
            array([2., 0., 3.])

            >>> profile = Profile(preferences_rk=[
            ...     [2, 0, 1],
            ...     [2, 0, 1],
            ...     [0, 1, 2],
            ...     [1, 2, 0],
            ... ])
            >>> rule = RuleIRVDuels(cm_option='exact')(profile)
            >>> rule.sufficient_coalition_size_cm_
            array([2., 2., 0.])

            >>> profile = Profile(preferences_rk=[
            ...     [1, 2, 0],
            ...     [0, 2, 1],
            ...     [2, 0, 1],
            ...     [2, 0, 1],
            ... ])
            >>> rule = RuleIRVDuels(cm_option='exact')(profile)
            >>> rule.is_cm_c_with_bounds_(1)
            (False, 2.0, 2.0)

            >>> profile = Profile(preferences_ut=[
            ...     [-1. ,  0. ,  0. ,  0. ],
            ...     [-1. ,  0.5, -1. , -0.5],
            ...     [ 1. ,  0.5, -1. , -1. ],
            ...     [ 0.5, -1. ,  0. ,  1. ],
            ...     [-1. ,  0.5,  0.5,  0. ],
            ...     [ 1. , -1. , -1. , -1. ],
            ... ], preferences_rk=[
            ...     [2, 3, 1, 0],
            ...     [1, 3, 0, 2],
            ...     [0, 1, 2, 3],
            ...     [3, 0, 2, 1],
            ...     [2, 1, 3, 0],
            ...     [0, 2, 1, 3],
            ... ])
            >>> rule = RuleIRVDuels(cm_option='exact')(profile)
            >>> rule.is_cm_c_(1)
            True
        """
        candidates = np.arange(self.profile_.n_c)
        n_s = preferences_borda_s.shape[0]
        n_max_updated = n_max  # Maximal number of manipulators allowed
        n_manip_final = np.inf  # Result: number of manipulators finally used
        example_path_losers = np.nan  # Result: example of elimination path (duel losers)
        example_path_winners = np.nan  # Result: example of elimination path (duel winners)
        r = 0
        is_candidate_alive_begin_r = np.zeros((self.profile_.n_c - 1, self.profile_.n_c), dtype=bool)
        is_candidate_alive_begin_r[0, :] = np.ones(self.profile_.n_c)
        n_manip_used_before_r = np.zeros(self.profile_.n_c - 1, dtype=int)
        scores_m_begin_r = np.zeros((self.profile_.n_c - 1, self.profile_.n_c))
        scores_tot_begin_r = np.zeros((self.profile_.n_c - 1, self.profile_.n_c))
        scores_tot_begin_r[0, :] = np.sum(np.equal(
            preferences_borda_s, np.max(preferences_borda_s, 1)[:, np.newaxis]
        ), 0)
        self.mylogv('cm_aux_exact: c =', c, 3)
        self.mylogv("cm_aux_exact: r =", r, 3)
        suggested_loser_r = {0: np.array([cand for cand in candidates if cand != c])}  # Possible losers of the duel
        suggested_winner_r = {0: np.array(candidates)}                                 # Possible winners of the duel
        index_loser_in_suggested_r = np.zeros(self.profile_.n_c - 1, dtype=int)
        # Index of the loser of the duel in ``suggested_loser_r``.
        index_winner_in_suggested_r = np.zeros(self.profile_.n_c - 1, dtype=int)
        # Index of the winner of the duel in ``suggested_winner_r``.
        while True:
            if r < 0:
                self.mylog("cm_aux_exact: End of exploration", 3)
                return n_manip_final, np.array(example_path_losers), np.array(example_path_winners), False
            if index_winner_in_suggested_r[r] >= suggested_winner_r[r].shape[0]:
                # We have tried all the possible winners for a given loser, change the loser.
                index_loser_in_suggested_r[r] += 1
                index_winner_in_suggested_r[r] = 0
            if (
                index_loser_in_suggested_r[r] >= suggested_loser_r[r].shape[0]
                or n_manip_used_before_r[r] > n_max_updated
            ):
                # First condition: we have tried all possible losers for this round, go back to previous round.
                # Second condition: may happen in ``optimize_bounds`` exact mode, if we have found a
                # solution and updated ``n_max_updated``.
                r -= 1
                self.mylogv("cm_aux_exact: Tried everything for round r, go back to r =", r, 3)
                self.mylogv("cm_aux_exact: r =", r, 3)
                if r >= 0:
                    index_winner_in_suggested_r[r] += 1
                continue
            loser = suggested_loser_r[r][index_loser_in_suggested_r[r]]
            winner = suggested_winner_r[r][index_winner_in_suggested_r[r]]
            self.mylogv("cm_aux_exact: suggested_loser_r[r] =", suggested_loser_r[r], 3)
            self.mylogv("cm_aux_exact: index_loser_in_suggested_r[r] =", index_loser_in_suggested_r[r], 3)
            self.mylogv("cm_aux_exact: loser =", loser, 3)
            self.mylogv("cm_aux_exact: suggested_winner_r[r] =", suggested_winner_r[r], 3)
            self.mylogv("cm_aux_exact: index_winner_in_suggested_r[r] =", index_winner_in_suggested_r[r], 3)
            self.mylogv("cm_aux_exact: winner =", winner, 3)

            if winner == loser:
                self.mylog("cm_aux_exact: winner == loser, skip this pair", 3)
                index_winner_in_suggested_r[r] += 1
                continue

            # How many manipulators are needed to make ``winner`` win the duel against ``loser``?
            n_m_needed_duel = matrix_duels_s[loser, winner] - matrix_duels_s[winner, loser] + (loser < winner)
            if n_m_needed_duel > n_max_updated:
                self.mylog("cm_aux_exact: `winner` cannot eliminate `loser`, try another pair.", 3)
                index_winner_in_suggested_r[r] += 1
                continue

            # What manipulators are needed to select `winner` and `loser` for the duel?
            self.mylogv("cm_aux_exact: scores_tot_begin_r[r, :] =", scores_tot_begin_r[r, :], 3)
            if scores_tot_begin_r[r, winner] >= scores_tot_begin_r[r, loser] + (loser < winner):
                score_to_beat = scores_tot_begin_r[r, winner]
                cand_to_beat = winner  # Candidate whose score we must beat
                other_selected = loser
            else:
                score_to_beat = scores_tot_begin_r[r, loser]
                cand_to_beat = loser  # Candidate whose score we must beat
                other_selected = winner
            scores_m_new_r = np.zeros(self.profile_.n_c)
            scores_m_new_r[is_candidate_alive_begin_r[r, :]] = np.maximum((
                score_to_beat - scores_tot_begin_r[r, is_candidate_alive_begin_r[r, :]]
                + (candidates[is_candidate_alive_begin_r[r, :]] > cand_to_beat)
            ), 0)
            scores_m_new_r[other_selected] = 0  # No need to give points to the other selected
            scores_m_end_r = scores_m_begin_r[r, :] + scores_m_new_r
            n_manip_r_and_before = max(n_manip_used_before_r[r], np.sum(scores_m_end_r))
            self.mylogv("cm_aux_exact: scores_m_new_r =", scores_m_new_r, 3)
            self.mylogv("cm_aux_exact: scores_m_end_r =", scores_m_end_r, 3)
            self.mylogv("cm_aux_exact: n_manip_r_and_before =", n_manip_r_and_before, 3)

            if n_manip_r_and_before > n_max_updated:
                self.mylog("cm_aux_exact: Cannot select `winner` and `loser` for the duel, try another pair.", 3)
                index_winner_in_suggested_r[r] += 1
                continue

            if r == self.profile_.n_c - 2:
                n_manip_final = n_manip_r_and_before
                example_path_losers = []
                example_path_winners = []
                for r in range(self.profile_.n_c - 1):
                    try:
                        example_path_losers.append(suggested_loser_r[r][index_loser_in_suggested_r[r]])
                        example_path_winners.append(suggested_winner_r[r][index_winner_in_suggested_r[r]])
                    except IndexError:  # pragma: no cover
                        # TO DO: Investigate whether this case can actually happen.
                        self._reached_uncovered_code()
                        print(f'{r=}')
                        print(f'{index_loser_in_suggested_r=}')
                        print(f'{suggested_loser_r=}')
                        print(f'{index_winner_in_suggested_r=}')
                        print(f'{suggested_winner_r=}')
                        raise IndexError
                self.mylog("cm_aux_exact: CM found", 3)
                self.mylogv("cm_aux_exact: n_manip_final =", n_manip_final, 3)
                self.mylogv("cm_aux_exact: example_path_losers =", example_path_losers, 3)
                self.mylogv("cm_aux_exact: example_path_winners =", example_path_winners, 3)
                if n_manip_final == n_min:
                    self.mylogv("cm_aux_exact: End of exploration: it is not possible to do better than n_min =",
                                n_min, 3)
                    return n_manip_final, np.array(example_path_losers), np.array(example_path_winners), False
                if not optimize_bounds:
                    return n_manip_final, np.array(example_path_losers), np.array(example_path_winners), True
                n_max_updated = n_manip_r_and_before - 1
                self.mylogv("cm_aux_exact: n_max_updated =", n_max_updated, 3)
                index_winner_in_suggested_r[r] += 1
                continue

            # Calculate scores for next round
            n_manip_used_before_r[r + 1] = n_manip_r_and_before
            is_candidate_alive_begin_r[r + 1, :] = is_candidate_alive_begin_r[r, :]
            is_candidate_alive_begin_r[r + 1, loser] = False
            self.mylogv("cm_aux_exact: is_candidate_alive_begin_r[r+1, :] =", is_candidate_alive_begin_r[r + 1, :], 3)
            scores_tot_begin_r[r + 1, :] = np.full(self.profile_.n_c, np.nan)
            scores_tot_begin_r[r + 1, is_candidate_alive_begin_r[r + 1, :]] = (
                np.sum(np.equal(
                    preferences_borda_s[:, is_candidate_alive_begin_r[r + 1, :]],
                    np.max(preferences_borda_s[:, is_candidate_alive_begin_r[r + 1, :]], 1)[:, np.newaxis]
                ), 0))
            self.mylogv("cm_aux_exact: scores_s_begin_r[r+1, :] =", scores_tot_begin_r[r + 1, :], 3)
            scores_m_begin_r[r + 1, :] = scores_m_end_r
            scores_m_begin_r[r + 1, loser] = 0
            self.mylogv("cm_aux_exact: scores_m_begin_r[r+1, :] =", scores_m_begin_r[r + 1, :], 3)
            scores_tot_begin_r[r + 1, :] += scores_m_begin_r[r + 1, :]
            self.mylogv("cm_aux_exact: scores_tot_begin_r[r+1, :] =", scores_tot_begin_r[r + 1, :], 3)

            # If an opponent has too many votes, then manipulation is not possible.
            max_score = np.nanmax(scores_tot_begin_r[r + 1, candidates != c])
            most_serious_opponent = np.where(scores_tot_begin_r[r + 1, :] == max_score)[0][0]
            if max_score + (most_serious_opponent < c) > n_s + n_max_updated - max_score:
                self.mylogv("cm_aux_exact: most_serious_opponent =", most_serious_opponent, 3)
                self.mylog(
                    "cm_aux_exact: Manipulation impossible by this path (an opponent will have too many votes)", 3)
                index_winner_in_suggested_r[r] += 1
                continue

            # Update other variables for next round
            # noinspection PyUnresolvedReferences
            suggested_loser_r[r + 1] = suggested_loser_r[r][suggested_loser_r[r][:] != loser]
            index_loser_in_suggested_r[r + 1] = 0
            # noinspection PyUnresolvedReferences
            suggested_winner_r[r + 1] = suggested_winner_r[r][suggested_winner_r[r][:] != loser]
            index_winner_in_suggested_r[r + 1] = 0
            r += 1
            self.mylogv("cm_aux_exact: r =", r, 3)

    def _cm_main_work_c_(self, c, optimize_bounds):
        """
            >>> profile = Profile(preferences_rk=[
            ...     [2, 1, 0],
            ...     [0, 2, 1],
            ...     [0, 2, 1],
            ...     [2, 0, 1],
            ... ])
            >>> rule = RuleIRVDuels(cm_option='exact')(profile)
            >>> rule.necessary_coalition_size_cm_
            array([0., 3., 3.])

            >>> profile = Profile(preferences_rk=[
            ...     [1, 0, 2],
            ...     [1, 2, 0],
            ...     [0, 2, 1],
            ...     [2, 1, 0],
            ... ])
            >>> rule = RuleIRVDuels(cm_option='exact')(profile)
            >>> rule.is_cm_c_(0)
            False
        """
        if not self.cm_option == "exact":
            # With 'lazy' option, we stop here anyway.
            return False
        # From this point, we have necessarily the 'exact' option.
        n_m = self.profile_.matrix_duels_ut[c, self.w_]
        n_max = self._sufficient_coalition_size_cm[c] - 1 if optimize_bounds else n_m
        n_min = self._necessary_coalition_size_cm[c]
        preferences_borda_s = self.profile_.preferences_borda_rk[np.logical_not(self.v_wants_to_help_c_[:, c]), :]
        matrix_duels_s = preferences_ut_to_matrix_duels_ut(preferences_borda_s)
        n_manip_exact, example_path_losers, example_path_winners, quick_escape = self._cm_aux_exact(
            c=c,
            n_max=n_max,
            n_min=n_min,
            optimize_bounds=optimize_bounds,
            preferences_borda_s=preferences_borda_s,
            matrix_duels_s=matrix_duels_s
        )
        self.mylogv('CM: Exact algorithm: n_manip_exact =', n_manip_exact)
        self._update_sufficient(self._sufficient_coalition_size_cm, c, n_manip_exact,
                                'CM: Fast algorithm: sufficient_coalition_size_cm =')
        # Update necessary coalition and return
        if optimize_bounds:
            self._update_necessary(self._necessary_coalition_size_cm, c, self._sufficient_coalition_size_cm[c],
                                   'CM: Update necessary_coalition_size_cm[c] = sufficient_coalition_size_cm[c] =')
        else:
            if n_m < self._sufficient_coalition_size_cm[c]:
                # We have explored everything with ``n_max = n_m`` but manipulation failed.
                self._update_necessary(
                    self._necessary_coalition_size_cm, c, n_m + 1,
                    'CM: Update necessary_coalition_size_cm[c] = n_m + 1 =')
        return quick_escape
