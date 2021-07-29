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
from svvamp.rules.rule import Rule
from svvamp.rules.rule_irv import RuleIRV
from svvamp.utils.util_cache import cached_property
from svvamp.preferences.profile import Profile
from svvamp.utils.pseudo_bool import equal_true
from svvamp.utils.misc import preferences_ut_to_matrix_duels_ut


class RuleICRV(Rule):
    """Instant-Condorcet Runoff Voting (ICRV).

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
        log_cm: cm_option = fast, um_option = lazy, tm_option = exact
        candidates_cm =
        [ 0.  0. nan]
        necessary_coalition_size_cm =
        [0. 1. 2.]
        sufficient_coalition_size_cm =
        [0. 2. 4.]

    Notes
    -----
    Principle: eliminate candidates as in IRV; stop as soon as there is a Condorcet winner.

    * Even round ``r`` (including round 0): if a candidate ``w`` has only victories against all other non-eliminated
      candidates (i.e. is a Condorcet winner in this subset, in the sense of :attr:`matrix_victories_rk`), then ``w``
      is declared the winner.
    * Odd round ``r``: the candidate who is ranked first (among non-eliminated candidates) by least voters is
      eliminated, like in :class:`RuleIRV`.

    This method meets the Condorcet criterion.

    * :meth:`is_cm_`:

        * :attr:`cm_option` = ``'fast'``: Rely on :class:`RuleIRV`'s fast algorithm. Polynomial heuristic. Can prove
          CM but unable to decide non-CM (except in rare obvious cases).
        * :attr:`cm_option` = ``'slow'``: Rely on :class:`RuleExhaustiveBallot`'s exact algorithm. Non-polynomial
          heuristic (:math:`2^{n_c}`). Quite efficient to prove CM or non-CM.
        * :attr:`cm_option` = ``'very_slow'``: Rely on :class:`RuleIRV`'s exact algorithm. Non-polynomial
          heuristic (:math:`n_c!`). Very efficient to prove CM or non-CM.
        * :attr:`cm_option` = ``'exact'``: Non-polynomial algorithm from superclass :class:`Rule`.

        Each algorithm above exploits the faster ones. For example, if :attr:`cm_option` = ``'very_slow'``,
        SVVAMP tries the fast algorithm first, then the slow one, then the 'very slow' one. As soon as it reaches
        a decision, computation stops.

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

    full_name = 'Benham'
    abbreviation = 'Ben'

    options_parameters = Rule.options_parameters.copy()
    options_parameters.update({
        'cm_option': {'allowed': {'fast', 'slow', 'very_slow', 'exact'}, 'default': 'fast'},
        'icm_option': {'allowed': ['exact'], 'default': 'exact'},
    })

    def __init__(self, **kwargs):
        super().__init__(
            with_two_candidates_reduces_to_plurality=True, is_based_on_rk=True,
            precheck_icm=False,
            log_identity="ICRV", **kwargs
        )

    def __call__(self, profile):
        self.delete_cache(suffix='_')
        self.profile_ = profile
        # Grab the IRV ballot of the profile (or create it)
        irv_options = {}
        if self.cm_option == 'fast':
            irv_options['cm_option'] = 'fast'
        elif self.cm_option == 'slow':
            irv_options['cm_option'] = 'slow'
        else:  # self.cm_option in {'very_slow', 'exact'}:
            irv_options['cm_option'] = 'exact'
        self.irv_ = RuleIRV(**irv_options)(self.profile_)
        return self

    # %% Counting the ballots

    @cached_property
    def _count_ballots_(self):
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
        self.mylog("Compute w", 1)
        plurality_elimination_engine = self.profile_.plurality_elimination_engine()
        for r in range(self.profile_.n_c - 1):
            # Is there a Condorcet winner?
            is_alive = plurality_elimination_engine.is_alive
            nb_alive = plurality_elimination_engine.nb_candidates_alive
            is_condorcet_winner = (
                is_alive
                & (np.sum(self.profile_.matrix_victories_rk[:, is_alive], axis=1) == nb_alive - 1)
            )
            condorcet_winners = np.where(is_condorcet_winner)[0]
            if condorcet_winners.size > 0:
                return condorcet_winners[0]
            # Now, elimination by Plurality
            if r != 0:
                plurality_elimination_engine.update_scores()
            # Who gets eliminated?
            scores = plurality_elimination_engine.scores
            loser = np.where(scores == np.nanmin(scores))[0][-1]  # Tie-breaking: the last index
            plurality_elimination_engine.eliminate_candidate(loser)
            self.mylogv("loser =", loser, 3)
            # Save time if last round
            if plurality_elimination_engine.nb_candidates_alive == 1:
                return plurality_elimination_engine.candidates_alive[0]

    @cached_property
    def scores_(self):
        """2d array.

        For even rounds ``r`` (including round 0), ``scores[r, c]`` is the number of victories for ``c`` in
        :attr:`matrix_victories_rk` (restricted to non-eliminated candidates). Ties count for 0.5.

        For odd rounds ``r``, ``scores[r, c]`` is the number of voters who rank ``c`` first (among non-eliminated
        candidates).
        """
        return self._count_ballots_['scores']

    @cached_property
    def candidates_by_scores_best_to_worst_(self):
        """1d array of integers.

        Candidates that are not eliminated at the moment a Condorcet winner is detected are sorted by their number of
        victories in :attr:`matrix_victories_rk` (restricted to candidates that are not eliminated at that time).

        Other candidates are sorted in the reverse order of their IRV elimination.
        """
        return self._count_ballots_['candidates_by_scores_best_to_worst']

    # TODO: self._v_might_im_for_c

    @cached_property
    def meets_majority_favorite_c_rk_ctb(self):
        return True

    @cached_property
    def meets_condorcet_c_rk(self):
        return True

    # %% Unison manipulation (UM)

    def _um_preliminary_checks_c_(self, c):
        if self.um_option not in {'fast', 'lazy'} or self.cm_option not in {'fast', 'lazy'}:
            if (
                self.w_ == self.profile_.condorcet_winner_rk_ctb
                and not self.profile_.c_might_be_there_when_cw_is_eliminated_irv_style[c]
            ):
                # Impossible to manipulate with n_m manipulators
                self._candidates_um[c] = False

    # %% Coalition Manipulation (CM)

    @cached_property
    def losing_candidates_(self):
        """If ``irv_.w_ does not win, then we put her first. Other losers are sorted as usual. (scores in
        ``matrix_duels_ut``).
        """
        self.mylog("Compute ordered list of losing candidates", 1)
        if self.w_ == self.irv_.w_:
            # As usual
            losing_candidates = np.concatenate((
                np.array(range(0, self.w_), dtype=int), np.array(range(self.w_ + 1, self.profile_.n_c), dtype=int)))
            losing_candidates = losing_candidates[np.argsort(
                - self.profile_.matrix_duels_ut[losing_candidates, self.w_], kind='mergesort')]
        else:
            # Put irv_.w_ first.
            losing_candidates = np.array(
                [c for c in range(self.profile_.n_c) if c != self.w_ and c != self.irv_.w_]).astype(np.int)
            losing_candidates = losing_candidates[np.argsort(
                - self.profile_.matrix_duels_ut[losing_candidates, self.w_], kind='mergesort')]
            losing_candidates = np.concatenate(([self.irv_.w_], losing_candidates))
        return losing_candidates

    def _cm_preliminary_checks_c_subclass_(self, c, optimize_bounds):
        """CM: preliminary checks for challenger ``c``.

        Let us consider the case where `w` is the Condorcet winner (rk ctb). Then even after manipulation, `c` cannot
        be the Condorcet winner on a subset that contains `w`. Hence at some point, `w` must be eliminated, IRV-style,
        at a moment where `c` is still present in the election.
        """
        if self.cm_option not in {'fast', 'lazy'}:
            if self.w_ == self.profile_.condorcet_winner_rk_ctb:
                self._update_necessary(
                    self._necessary_coalition_size_cm, c,
                    self.profile_.necessary_coalition_size_to_break_irv_immunity[c],
                    'CM: Preliminary checks: IRV-Immunity => \n    necessary_coalition_size_cm[c] = '
                )

    def _cm_aux_(self, c, ballots_m, preferences_rk_s):
        profile_test = Profile(preferences_rk=np.concatenate((preferences_rk_s, ballots_m)), sort_voters=False)
        if profile_test.n_v != self.profile_.n_v:
            raise AssertionError('Uh-oh!')
        winner_test = self.__class__()(profile_test).w_
        return winner_test == c

    def _cm_main_work_c_(self, c, optimize_bounds):
        n_m = self.profile_.matrix_duels_ut[c, self.w_]
        n_s = self.profile_.n_v - n_m
        candidates = np.array(range(self.profile_.n_c))
        preferences_borda_s = self.profile_.preferences_borda_rk[np.logical_not(self.v_wants_to_help_c_[:, c]), :]
        preferences_rk_s = self.profile_.preferences_rk[np.logical_not(self.v_wants_to_help_c_[:, c]), :]
        matrix_duels_vtb_temp = (preferences_ut_to_matrix_duels_ut(preferences_borda_s))
        self.mylogm("CM: matrix_duels_vtb_temp =", matrix_duels_vtb_temp, 3)
        # More preliminary checks. It's more convenient to put them in that method, because we need
        # ``preferences_borda_s`` and ``matrix_duels_vtb_temp``.
        d_neq_c = (np.array(range(self.profile_.n_c)) != c)
        # Prevent another cond. Look at the weakest duel for ``d``, she has ``matrix_duels_vtb_temp[d, e]``. We simply
        # need that:
        # ``matrix_duels_vtb_temp[d, e] <= (n_s + n_m) / 2``
        # ``2 * max_d(min_e(matrix_duels_vtb_temp[d, e])) - n_s <= n_m``
        n_manip_prevent_cond = 0
        for d in candidates[d_neq_c]:
            e_neq_d = (np.array(range(self.profile_.n_c)) != d)
            n_prevent_d = np.maximum(2 * np.min(matrix_duels_vtb_temp[d, e_neq_d]) - n_s, 0)
            n_manip_prevent_cond = max(n_manip_prevent_cond, n_prevent_d)
        self.mylogv("CM: n_manip_prevent_cond =", n_manip_prevent_cond, 3)
        self._update_necessary(self._necessary_coalition_size_cm, c, n_manip_prevent_cond,
                               'CM: Update necessary_coalition_size_cm[c] = n_manip_prevent_cond =')
        if not optimize_bounds and self._necessary_coalition_size_cm[c] > n_m:
            return True

        # Let us work
        if self.w_ == self.irv_.w_:
            self.mylog('CM: c != self.irv_.w == self.w', 3)
            if self.cm_option == "fast":
                self.irv_.cm_option = "fast"
            elif self.cm_option == "slow":
                self.irv_.cm_option = "slow"
            else:
                self.irv_.cm_option = "exact"
            irv_is_cm_c = self.irv_.is_cm_c_(c)
            if equal_true(irv_is_cm_c):
                # Use IRV without bounds
                self.mylog('CM: Use IRV without bounds')
                suggested_path_one = self.irv_.example_path_cm_c_(c)
                self.mylogv("CM: suggested_path =", suggested_path_one, 3)
                ballots_m = self.irv_.example_ballots_cm_c_(c)
                manipulation_found = self._cm_aux_(c, ballots_m, preferences_rk_s)
                self.mylogv("CM: manipulation_found =", manipulation_found, 3)
                if manipulation_found:
                    self._update_sufficient(self._sufficient_coalition_size_cm, c, n_m,
                                            'CM: Update sufficient_coalition_size_cm[c] = n_m =')
                    # We will not do better with any algorithm (even the brute force algo).
                    return False
                # Use IRV with bounds
                self.irv_.is_cm_c_with_bounds_(c)
                self.mylog('CM: Use IRV with bounds')
                suggested_path_two = self.irv_.example_path_cm_c_(c)
                self.mylogv("CM: suggested_path =", suggested_path_two, 3)
                if np.array_equal(suggested_path_one, suggested_path_two):
                    self.mylog('CM: Same suggested path as before, skip computation')
                else:
                    ballots_m = self.irv_.example_ballots_cm_c_(c)
                    manipulation_found = self._cm_aux_(c, ballots_m, preferences_rk_s)
                    self.mylogv("CM: manipulation_found =", manipulation_found, 3)
                    if manipulation_found:
                        self._update_sufficient(self._sufficient_coalition_size_cm, c, n_m,
                                                'CM: Update sufficient_coalition_size_cm[c] = n_m =')
                        # We will not do better with any algorithm (even the brute force algo).
                        return False
        else:  # self.w_ != self.irv_.w_:
            if c == self.irv_.w_:
                self.mylog('CM: c == self.irv_.w != self._w', 3)
                suggested_path = self.irv_.elimination_path_
                self.mylogv("CM: suggested_path =", suggested_path, 3)
                ballots_m = self.irv_.example_ballots_cm_w_against_(w_other_rule=self.w_)
                manipulation_found = self._cm_aux_(c, ballots_m, preferences_rk_s)
                self.mylogv("CM: manipulation_found =", manipulation_found, 3)
                if manipulation_found:
                    self._update_sufficient(self._sufficient_coalition_size_cm, c, n_m,
                                            'CM: Update sufficient_coalition_size_cm[c] = n_m =')
                    # We will not do better with any algorithm (even the brute force algo).
                    return
            else:
                self.mylog('CM: c, self.irv_.w_ and self.w_ are all distinct', 3)
        if self.cm_option == 'exact':
            return self._cm_main_work_c_exact_(c, optimize_bounds)
