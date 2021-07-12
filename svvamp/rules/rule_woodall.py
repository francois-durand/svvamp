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
        >>> rule.demo_manipulation_(log_depth=0)  # doctest: +NORMALIZE_WHITESPACE

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

    References
    ----------
    'Four Condorcet-Hare Hybrid Methods for Single-Winner Elections', James Green-Armytage, 2011.
    """

    def __init__(self, **kwargs):
        # TO DO: update this
        super().__init__(
            options_parameters={
                'tm_option': {'allowed': ['exact'], 'default': 'exact'},
                'icm_option': {'allowed': ['exact'], 'default': 'exact'}
            },
            with_two_candidates_reduces_to_plurality=True, is_based_on_rk=True,
            precheck_um=False, precheck_icm=False,
            log_identity="WOODALL", **kwargs
        )

    def __call__(self, profile):
        self.delete_cache(suffix='_')
        self.profile_ = profile
        # Grab the IRV ballot of the profile (or create it)
        self.irv_ = RuleIRV()(self.profile_)
        return self

    # %% Counting the ballots

    @cached_property
    def w_(self):
        if self.profile_.exists_condorcet_winner_rk:
            return self.profile_.condorcet_winner_rk
        else:
            return next(c for c in self.irv_.candidates_by_scores_best_to_worst_
                        if c in self.profile_.smith_set_rk)

    @cached_property
    def scores_(self):
        smith_set = self.profile_.smith_set_rk
        scores_smith = [(1 if c in smith_set else 0) for c in range(self.profile_.n_c)]
        scores_irv = sorted(range(self.profile_.n_c), key=self.irv_.elimination_path_.__getitem__)
        return np.array([scores_smith, scores_irv])

    @cached_property
    def candidates_by_scores_best_to_worst_(self):
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

    # TODO: should be implemented .

    # %% Ignorant-Coalition Manipulation (ICM)

    # Use the general methods from class Rule.

    # %% Coalition Manipulation (CM)

    # TODO: should be implemented.
