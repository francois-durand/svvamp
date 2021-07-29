# -*- coding: utf-8 -*-
"""
Created on 12 jul. 2021, 13:55
Copyright François Durand 2014-2021
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
from svvamp.preferences.profile import Profile
from svvamp.utils.util_cache import cached_property
from svvamp.utils.pseudo_bool import equal_true
from svvamp.utils.misc import preferences_ut_to_matrix_duels_ut


class RuleWoodall(Rule):
    """Woodall Rule.

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
        >>> rule = RuleWoodall()(profile)
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
        [[1 0 0]
         [2 0 1]]
        candidates_by_scores_best_to_worst
        [0, 2, 1]
        scores_best_to_worst
        [[1 0 0]
         [2 1 0]]
        w = 0
        score_w = [1 2]
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

    Woodall does not :attr:`meets_condorcet_c_ut_abs_ctb`:

        >>> profile = Profile(preferences_ut=[
        ...     [ 0. ,  0.5, -0.5],
        ...     [ 0.5, -0.5,  1. ],
        ... ], preferences_rk=[
        ...     [1, 0, 2],
        ...     [2, 0, 1],
        ... ])
        >>> RuleWoodall()(profile).w_
        1
        >>> profile.condorcet_winner_ut_abs_ctb
        0

    Woodall does not :attr:`meets_condorcet_c_ut_rel`:

        >>> profile = Profile(preferences_ut=[
        ...     [ 1. ,  1. ,  1. ],
        ...     [ 0.5, -0.5,  1. ],
        ... ], preferences_rk=[
        ...     [0, 1, 2],
        ...     [2, 0, 1],
        ... ])
        >>> RuleWoodall()(profile).w_
        0
        >>> profile.condorcet_winner_ut_rel
        2

    Notes
    -----
    Each voter must provide a strict total order. Among the candidates of the Smith set
    (in the sense of :attr:`smith_set_rk`), elect the one that is eliminated latest by :class:`RuleIRV`.

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

    References
    ----------
    'Four Condorcet-Hare Hybrid Methods for Single-Winner Elections', James Green-Armytage, 2011.
    """

    full_name = 'Woodall'
    abbreviation = 'Woo'

    options_parameters = Rule.options_parameters.copy()
    options_parameters.update({
        'cm_option': {'allowed': {'fast', 'slow', 'very_slow', 'exact'}, 'default': 'fast'},
        'tm_option': {'allowed': ['exact'], 'default': 'exact'},
        'icm_option': {'allowed': ['exact'], 'default': 'exact'}
    })

    def __init__(self, **kwargs):
        super().__init__(
            with_two_candidates_reduces_to_plurality=True, is_based_on_rk=True,
            precheck_icm=False,
            log_identity="WOODALL", **kwargs
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
    def w_(self):
        self.mylog("Count ballots", 1)
        if self.profile_.exists_condorcet_winner_rk:
            return self.profile_.condorcet_winner_rk
        elif self.irv_.w_ in self.profile_.smith_set_rk:
            # Included in the following case, but faster
            return self.irv_.w_
        else:
            return next(c for c in self.irv_.candidates_by_scores_best_to_worst_
                        if c in self.profile_.smith_set_rk)

    @cached_property
    def scores_(self):
        self.mylog("Compute scores", 1)
        smith_set = self.profile_.smith_set_rk
        scores_smith = [(1 if c in smith_set else 0) for c in range(self.profile_.n_c)]
        scores_irv = sorted(range(self.profile_.n_c), key=self.irv_.elimination_path_.__getitem__)
        return np.array([scores_smith, scores_irv])

    @cached_property
    def candidates_by_scores_best_to_worst_(self):
        self.mylog("Compute candidates_by_scores_best_to_worst", 1)
        return sorted(
            range(self.profile_.n_c),
            key=lambda c: list(self.scores_[:, c]),
            reverse=True
        )

    # TO DO: implement v_might_im_for_c_

    # %% Manipulation criteria of the voting system

    @cached_property
    def meets_majority_favorite_c_rk_ctb(self):
        return True

    @cached_property
    def meets_condorcet_c_rk(self):
        return True

    # %% Individual manipulation (IM)

    # TODO: should be implemented.

    # %% Trivial Manipulation (TM)

    # Use the general methods from class Rule.

    # %% Unison manipulation (UM)

    def _um_preliminary_checks_c_(self, c):
        if self.um_option not in {'fast', 'lazy'} or self.cm_option not in {'fast', 'lazy'}:
            if (
                self.w_ == self.profile_.condorcet_winner_rk_ctb
                and not self.profile_.c_might_be_there_when_cw_is_eliminated_irv_style[c]
            ):
                # Impossible to manipulate with n_m manipulators
                self._candidates_um[c] = False

    # %% Ignorant-Coalition Manipulation (ICM)

    # Use the general methods from class Rule.

    # %% Coalition Manipulation (CM)

    def _cm_preliminary_checks_c_subclass_(self, c, optimize_bounds):
        """CM: preliminary checks for challenger ``c``.

        Let us consider the case where `w` is the Condorcet winner (rk ctb) (otherwise it is easy to manipulate, at
        least if utility preferences are strict). We must have `c` in the (manipulated) Smith Set, hence we must have
        `w` too. Hence we must manipulate IRV so that `w` is eliminated before `c`. In other words, at some point,
        `w` must be eliminated, IRV-style, at a moment where `c` is still present in the election.
        """
        if self.cm_option not in {'fast', 'lazy'}:
            if self.w_ == self.profile_.condorcet_winner_rk_ctb:
                self._update_necessary(
                    self._necessary_coalition_size_cm, c,
                    self.profile_.necessary_coalition_size_to_break_irv_immunity[c],
                    'CM: Preliminary checks: IRV-Immunity => \n    necessary_coalition_size_cm[c] = '
                )

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
