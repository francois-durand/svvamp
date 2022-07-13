# -*- coding: utf-8 -*-
"""
Created on 16 jul. 2021, 13:32
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
from svvamp.utils.util_cache import cached_property
from svvamp.preferences.profile import Profile


class RuleCopeland(Rule):
    """
    Copeland rule

    Notes
    -----
    Copeland does not :attr:`meets_condorcet_c_ut_rel`:

        >>> profile = Profile(preferences_ut=[
        ...     [-0.5, -0.5, -1. ],
        ...     [ 0. ,  0.5, -1. ],
        ... ], preferences_rk=[
        ...     [0, 1, 2],
        ...     [1, 0, 2],
        ... ])
        >>> RuleCopeland()(profile).w_
        0
        >>> profile.condorcet_winner_ut_rel
        1

    Copeland does not :attr:`meets_majority_favorite_c_ut_ctb`:

        >>> profile = Profile(preferences_ut=[
        ...     [ 0. , -0.5, -1. ],
        ...     [-0.5,  1. , -0.5],
        ... ], preferences_rk=[
        ...     [0, 1, 2],
        ...     [1, 2, 0],
        ... ])
        >>> RuleCopeland()(profile).w_
        1
        >>> profile.majority_favorite_ut_ctb
        0
    """

    full_name = 'Copeland'
    abbreviation = 'Cop'

    options_parameters = Rule.options_parameters.copy()
    options_parameters['icm_option'] = {'allowed': ['exact'], 'default': 'exact'}
    options_parameters['cm_option'] = {'allowed': ['fast', 'exact'], 'default': 'fast'}

    def __init__(self, **kwargs):
        super().__init__(
            with_two_candidates_reduces_to_plurality=True, is_based_on_rk=True,
            precheck_icm=False,
            log_identity="COPELAND", **kwargs
        )

    @cached_property
    def scores_(self):
        return self.profile_.matrix_victories_rk.sum(axis=1)

    # %% Manipulation criteria of the voting system

    @cached_property
    def meets_condorcet_c_rk(self):
        return True

    @cached_property
    def meets_ignmc_c_ctb(self):
        return True

    # %% Coalition Manipulation (CM)

    # noinspection PyUnusedLocal
    def _cm_main_work_c_fast_(self, c, optimize_bounds):
        """Do the main work in CM loop for candidate ``c``.

        * Try to improve bounds ``_sufficient_coalition_size_cm[c]`` and ``_necessary_coalition_size_cm[c]``.
        """
        if self.profile_.n_c == 3:
            pref_borda_rk_s = self.profile_.preferences_borda_rk[np.logical_not(self.v_wants_to_help_c_[:, c]), :]
            matrix_duels_s = Profile(preferences_ut=pref_borda_rk_s).matrix_duels_ut
            matrix_duels_s_antisym = matrix_duels_s - matrix_duels_s.T
            d = np.argmax(matrix_duels_s_antisym[:, c])  # Opponent in the most difficult duel for `c`
            e = 3 - c - d  # Opponent in the easiest duel for `c`
            # 2 victories => `c` is CW => `c` wins
            sufficient_2_victories = matrix_duels_s_antisym[d, c] + 1
            # 1.5 victories (with a tie against `d`). Then the only dangerous opponent is `d`, who might also have
            # 1.5 victories.
            #   * If `c < d`, then `c` will win anyway (possibly by tie-breaking rule).
            #   * If `c > d`, then `d` must not win against `e`, i.e. `e` must at least tie with `d`.
            if c < d:
                sufficient_1_dot_5_victories = max(
                    matrix_duels_s_antisym[d, c],
                    matrix_duels_s_antisym[e, c] + 1
                )
            else:
                sufficient_1_dot_5_victories = max(
                    matrix_duels_s_antisym[d, c],
                    matrix_duels_s_antisym[e, c] + 1,
                    matrix_duels_s_antisym[d, e]
                )
            # 1 victory (i.e. against `e`).
            #   * If `c == 0`, then `e` must win against `d` and the tie-breaking rule makes `c` win.
            #   * If `c != 0`, then she cannot win with only 1 victory.
            if c == 0:
                sufficient_1_victory = max(
                    matrix_duels_s_antisym[e, c] + 1,
                    matrix_duels_s_antisym[d, e] + 1
                )
            else:
                sufficient_1_victory = np.inf
            # 2 ties.
            #   * If `c == 0`, then `e` must tie with `d` and the tie-breaking rule makes `c` win.
            #   * If `c != 0`, then she cannot win with only 2 ties.
            if c == 0:
                sufficient_2_ties = max(
                    matrix_duels_s_antisym[e, c],
                    matrix_duels_s_antisym[d, c],
                    np.abs(matrix_duels_s_antisym[d, e])
                )
            else:
                sufficient_2_ties = np.inf
            sufficient = min(
                sufficient_2_victories,
                sufficient_1_dot_5_victories,
                sufficient_1_victory,
                sufficient_2_ties
            )
            self.mylogv('CM: Fast algorithm: sufficient =', sufficient)
            self._update_sufficient(self._sufficient_coalition_size_cm, c, sufficient,
                                    'CM: Fast algorithm: sufficient_coalition_size_cm =')
            self._update_necessary(self._necessary_coalition_size_cm, c, sufficient,
                                   'CM: Fast algorithm: necessary_coalition_size_cm =')
        return False

    def _cm_main_work_c_(self, c, optimize_bounds):
        is_quick_escape_fast = self._cm_main_work_c_fast_(c, optimize_bounds)
        if not self.cm_option == "exact":
            # With 'fast' option, we stop here anyway.
            return is_quick_escape_fast
        # From this point, we have necessarily the 'exact' option (which is, in fact, only an exhaustive exploration
        # with = ``n_m`` manipulators).
        is_quick_escape_exact = self._cm_main_work_c_exact_(c, optimize_bounds)
        return is_quick_escape_fast or is_quick_escape_exact
