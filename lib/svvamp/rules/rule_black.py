# -*- coding: utf-8 -*-
"""
Created on 16 jul. 2021, 13:57
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


class RuleBlack(Rule):
    """
    Black rule

    Notes
    -----
    Black does not :attr:`meets_condorcet_c_ut_rel`:

        >>> profile = Profile(preferences_ut=[
        ...     [-1. , -0.5, -0.5],
        ...     [-1. , -0.5,  0. ],
        ... ], preferences_rk=[
        ...     [1, 2, 0],
        ...     [2, 1, 0],
        ... ])
        >>> RuleBlack()(profile).w_
        1
        >>> profile.condorcet_winner_ut_rel
        2

    Black does not :attr:`meets_majority_favorite_c_ut_ctb`:

        >>> profile = Profile(preferences_ut=[
        ...     [ 0.5, -1. ,  0. ],
        ...     [-1. ,  0. ,  0.5],
        ... ], preferences_rk=[
        ...     [0, 2, 1],
        ...     [2, 1, 0],
        ... ])
        >>> RuleBlack()(profile).w_
        2
        >>> profile.majority_favorite_ut_ctb
        0
    """

    full_name = 'Black'
    abbreviation = 'Bla'

    options_parameters = Rule.options_parameters.copy()
    options_parameters['icm_option'] = {'allowed': ['exact'], 'default': 'exact'}

    def __init__(self, **kwargs):
        super().__init__(
            with_two_candidates_reduces_to_plurality=True, is_based_on_rk=True,
            precheck_icm=False,
            log_identity="BLACK", **kwargs
        )

    # %% Count the ballots

    @cached_property
    def w_(self):
        self.mylog("Compute winner", 1)
        if self.profile_.exists_condorcet_winner_rk:
            return self.profile_.condorcet_winner_rk
        else:
            return int(np.argmax(self.profile_.borda_score_c_rk))

    @cached_property
    def scores_(self):
        self.mylog("Compute scores", 1)
        scores_condorcet = np.zeros(self.profile_.n_c)
        if self.profile_.exists_condorcet_winner_rk:
            scores_condorcet[self.profile_.condorcet_winner_rk] = 1
        scores_borda = self.profile_.borda_score_c_rk
        return np.array(scores_condorcet, scores_borda)

    @cached_property
    def candidates_by_scores_best_to_worst_(self):
        self.mylog("Compute candidates_by_scores_best_to_worst", 1)
        return sorted(
            range(self.profile_.n_c),
            key=lambda c: list(self.scores_[:, c]),
            reverse=True
        )

    # %% Manipulation criteria of the voting system

    @cached_property
    def meets_condorcet_c_rk(self):
        return True

    @cached_property
    def meets_ignmc_c_ctb(self):
        return True
