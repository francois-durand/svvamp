# -*- coding: utf-8 -*-
"""
Created on 7 dec. 2018, 15:01
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
from svvamp import GeneratorProfileLadder
from svvamp.utils.misc import initialize_random_seeds
from svvamp.rules.rule import Rule
from svvamp.utils.util_cache import cached_property
from svvamp.preferences.profile import Profile


class RuleICRV(Rule):
    """Instant-Condorcet Runoff Voting (ICRV).

    Examples
    --------
        >>> initialize_random_seeds()
        >>> profile = GeneratorProfileLadder(n_v=5, n_c=3, n_rungs=5)().profile_
        >>> rule = RuleICRV()(profile)
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
        [[2. 0. 1.]]
        candidates_by_scores_best_to_worst
        [0 2 1]
        scores_best_to_worst
        [[2. 1. 0.]]
        w = 0
        score_w = [2.]
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
        log_cm: cm_option = lazy, um_option = lazy, tm_option = exact
        candidates_cm =
        [ 0.  0. nan]
        necessary_coalition_size_cm =
        [0. 1. 2.]
        sufficient_coalition_size_cm =
        [0. 6. 4.]

    Notes
    -----
    Principle: eliminate candidates as in IRV; stop as soon as there is a Condorcet winner.

    * Even round ``r`` (including round 0): if a candidate ``w`` has only victories against all other non-eliminated
      candidates (i.e. is a Condorcet winner in this subset, in the sense of :attr:`matrix_victories_rk`), then ``w``
      is declared the winner.
    * Odd round ``r``: the candidate who is ranked first (among non-eliminated candidates) by least voters is
    eliminated, like in :class:`RuleIRV`.

    This method meets the Condorcet criterion.

    * :meth:`is_cm_`: Non-polynomial or non-exact algorithms from superclass :class:`Rule`.
    * :meth:`is_icm_`: Exact in polynomial time.
    * :meth:`is_im_`: Non-polynomial or non-exact algorithms from superclass :class:`Rule`.
    * :meth:`~svvamp.Election.not_iia`: Exact in polynomial time.
    * :meth:`is_tm_`: Exact in polynomial time.
    * :meth:`is_um_`: Non-polynomial or non-exact algorithms from superclass :class:`Rule`.

    References
    ----------
    'Four Condorcet-Hare Hybrid Methods for Single-Winner Elections', James Green-Armytage, 2011.

    See Also
    --------
    :class:`RuleExhaustiveBallot`, :class:`RuleIRV`, :class:`RuleIRVDuels`, :class:`RuleCondorcetAbsIRV`,
    :class:`RuleCondorcetVtbIRV`.
    """

    def __init__(self, **kwargs):
        super().__init__(
            options_parameters={
                'icm_option': {'allowed': ['exact'], 'default': 'exact'},
            },
            with_two_candidates_reduces_to_plurality=True, is_based_on_rk=True,
            precheck_icm=False,
            log_identity="ICRV", **kwargs
        )

    # %% Counting the ballots

    @cached_property
    def _counts_ballots_(self):
        self.mylog("Count ballots", 1)
        scores = []
        is_candidate_alive = np.ones(self.profile_.n_c, dtype=np.bool)
        worst_to_best = []
        r = 0
        while True:
            # Here, scores_r is the number of victories (when restricting to alive candidates).
            scores_r = np.full(self.profile_.n_c, np.nan)
            scores_r[is_candidate_alive] = np.sum(
                self.profile_.matrix_victories_rk[is_candidate_alive, :][:, is_candidate_alive], 1)
            scores.append(scores_r)
            best_c = np.nanargmax(scores_r)
            self.mylogv("best_c =", best_c, 3)
            if scores_r[best_c] == self.profile_.n_c - 1 - r:
                w = best_c
                self.mylogm("scores (before) =", scores, 3)
                scores = np.array(scores)
                self.mylogm("scores (after) =", scores, 3)
                candidates_alive = np.array(range(self.profile_.n_c))[is_candidate_alive]
                candidates_by_scores_best_to_worst = np.concatenate((
                    candidates_alive[np.argsort(-scores_r[is_candidate_alive], kind='mergesort')],
                    worst_to_best[::-1]
                )).astype(dtype=np.int)
                return {'scores': scores, 'w': w,
                        'candidates_by_scores_best_to_worst': candidates_by_scores_best_to_worst}

            # Now, scores_r is the Plurality score (IRV-style)
            scores_r = np.full(self.profile_.n_c, np.nan)
            scores_r[is_candidate_alive] = np.sum(
                self.profile_.preferences_borda_rk[:, is_candidate_alive]
                == np.max(self.profile_.preferences_borda_rk[:, is_candidate_alive], 1)[:, np.newaxis],
                0)
            scores.append(scores_r)
            loser = np.where(scores_r == np.nanmin(scores_r))[0][-1]
            self.mylogv("loser =", loser, 3)
            is_candidate_alive[loser] = False
            worst_to_best.append(loser)

            r += 1

    @cached_property
    def w_(self):
        return self._counts_ballots_['w']

    @cached_property
    def scores_(self):
        """2d array.

        For even rounds ``r`` (including round 0), ``scores[r, c]`` is the number of victories for ``c`` in
        :attr:`matrix_victories_rk` (restricted to non-eliminated candidates). Ties count for 0.5.

        For odd rounds ``r``, ``scores[r, c]`` is the number of voters who rank ``c`` first (among non-eliminated
        candidates).
        """
        return self._counts_ballots_['scores']

    @cached_property
    def candidates_by_scores_best_to_worst_(self):
        """1d array of integers.

        Candidates that are not eliminated at the moment a Condorcet winner is detected are sorted by their number of
        victories in :attr:`matrix_victories_rk` (restricted to candidates that are not eliminated at that time).

        Other candidates are sorted in the reverse order of their IRV elimination.
        """
        return self._counts_ballots_['candidates_by_scores_best_to_worst']

    # TODO: self._v_might_im_for_c

    @cached_property
    def meets_majority_favorite_c_rk_ctb(self):
        return True

    @cached_property
    def meets_condorcet_c_rk(self):
        return True


if __name__ == '__main__':
    RuleICRV()(Profile(preferences_ut=np.random.randint(-5, 5, (5, 3)))).demo_()
