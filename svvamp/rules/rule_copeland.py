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
