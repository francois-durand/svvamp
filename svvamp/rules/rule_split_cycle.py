# -*- coding: utf-8 -*-
"""
Created on 16 jul. 2021, 15:49
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


class RuleSplitCycle(Rule):
    """Split Cycle.

    Examples
    --------
    Split Cycle does not :attr:`meets_condorcet_c_ut_rel`:

        >>> profile = Profile(preferences_ut=[
        ...     [ 0. ,  0. ,  0. ],
        ...     [ 0.5,  1. , -0.5],
        ... ], preferences_rk=[
        ...     [0, 2, 1],
        ...     [1, 0, 2],
        ... ])
        >>> RuleSplitCycle()(profile).w_
        0
        >>> profile.condorcet_winner_ut_rel
        1
    """

    full_name = 'Split Cycle'
    abbreviation = 'SC'

    options_parameters = Rule.options_parameters.copy()
    options_parameters['icm_option'] = {'allowed': ['exact'], 'default': 'exact'}

    def __init__(self, **kwargs):
        super().__init__(
            with_two_candidates_reduces_to_plurality=True, is_based_on_rk=True,
            precheck_icm=False,
            log_identity="SPLIT_CYCLE", **kwargs
        )

    # %% Counting the ballots

    # noinspection PyProtectedMember
    @staticmethod
    def _count_ballot_aux(matrix_duels):
        """

        Parameters
        ----------
        matrix_duels

        Returns
        -------

        Examples
        --------
            >>> a, b, c, d, e, f = 1, 2, 3, 4, 5, 6
            >>> def cowinners(profile):
            ...     results = RuleSplitCycle._count_ballot_aux(profile.matrix_duels_rk)
            ...     better_or_tie_c = results['better_or_tie_c']
            ...     return list(np.where(better_or_tie_c == np.max(better_or_tie_c))[0])

            >>> profile = Profile(preferences_rk=
            ...     [[a, c, b]] * 5
            ...     + [[c, b, a]] * 4
            ...     + [[b, a, c]] * 4
            ...     + [[c, a, b]] * 2
            ...     + [[b, c, a]] * 2
            ... )
            >>> cowinners(profile)
            [2]

            >>> profile = Profile(preferences_rk=(
            ...     [[d, b, a, c]] * 3
            ...     + [[a, d, c, b]] * 1
            ...     + [[a, c, b, d]] * 2
            ...     + [[c, b, d, a]] * 1
            ...     + [[c, b, a, d]] * 2
            ... ))
            >>> cowinners(profile)
            [0, 1, 2]

            >>> profile = Profile(preferences_rk=(
            ...     [[b,a,c,e,d,f]] * 2
            ...     + [[e,d,f,b,a,c]] * 2
            ...     + [[a,c,b,d,f,e]] * 2
            ...     + [[d,f,e,a,c,b]] * 2
            ...     + [[c,b,a,f,e,d]] * 2
            ...     + [[f,e,d,c,b,a]] * 2
            ...     + [[a,f,b,c,d,e]] * 2
            ...     + [[e,d,c,b,a,f]] * 2
            ...     + [[d,a,b,c,e,f]] * 1
            ...     + [[f,e,c,b,d,a]] * 1
            ... ))
            >>> cowinners(profile)
            [0, 1, 2, 3, 4]

            >>> profile = Profile(preferences_rk=(
            ...     [[e, c, b, a, d]] * 4
            ...     + [[c, b, a, d, e]] * 3
            ...     + [[e, b, a, d, c]] * 7
            ...     + [[d, a, c, b, e]] * 7
            ...     + [[c, d, b, a, e]] * 2
            ...     + [[d, e, a, b, c]] * 1
            ... ))
            >>> cowinners(profile)
            [3]

            >>> profile = Profile(preferences_rk=(
            ...     [[d, b, c, a]] * 1
            ...     + [[c, a, d, b]] * 2
            ...     + [[c, d, b, a]] * 1
            ...     + [[b, a, d, c]] * 2
            ... ))
            >>> cowinners(profile)
            [1, 2, 3]

            >>> profile = Profile(preferences_rk=(
            ...     [[b, a, c, d]] * 9
            ...     + [[a, c, b, d]] * 3
            ...     + [[d, a, c, b]] * 2
            ...     + [[d, c, b, a]] * 5
            ...     + [[d, c, a, b]] * 4
            ... ))
            >>> cowinners(profile)
            [0, 1, 2]

            >>> profile = Profile(preferences_rk=(
            ...     [[b, a, c, d, e]] * 2
            ...     + [[b, a, c, e, d]] * 3
            ...     + [[a, e, c, b, d]] * 3
            ...     + [[d, e, a, c, b]] * 1
            ...     + [[d, a, e, c, b]] * 1
            ...     + [[d, e, c, b, a]] * 5
            ...     + [[b, e, a, c, d]] * 4
            ...     + [[d, c, a, b, e]] * 4
            ... ))
            >>> cowinners(profile)
            [0, 1, 3]

            >>> profile = Profile(preferences_rk=(
            ...     [[a, b, c, d]] * 5
            ...     + [[b, c, d, a]] * 3
            ...     + [[a, c, b, d]] * 1
            ...     + [[d, a, b, c]] * 7
            ...     + [[c, a, b, d]] * 4
            ...     + [[d, b, c, a]] * 4
            ...     + [[b, a, c, d]] * 2
            ...     + [[b, a, d, c]] * 11
            ...     + [[c, d, a, b]] * 16
            ... ))
            >>> cowinners(profile)
            [0]

            >>> profile = Profile(preferences_rk=(
            ...     [[a, b, c, d, e]] * 5
            ...     + [[b, c, e, d, a]] * 3
            ...     + [[c, e, d, a, b]] * 4
            ...     + [[e, a, c, b, d]] * 1
            ...     + [[d, a, b, c, e]] * 7
            ...     + [[c, a, e, b, d]] * 4
            ...     + [[e, d, b, c, a]] * 4
            ...     + [[e, b, a, c, d]] * 2
            ...     + [[e, b, a, d, c]] * 7
            ...     + [[b, e, a, d, c]] * 4
            ...     + [[c, d, a, e, b]] * 12
            ... ))
            >>> cowinners(profile)
            [0, 2]

            >>> profile = Profile(preferences_rk=(
            ...     [[c, b, a]] * 3
            ...     + [[b, a, c]] * 3
            ...     + [[a, c, b]] * 3
            ... ))
            >>> cowinners(profile)
            [0, 1, 2]

            >>> profile = Profile(preferences_rk=(
            ...     [[c, b, a, d]] * 3
            ...     + [[b, a, d, c]] * 2
            ...     + [[d, b, a, c]] * 1
            ...     + [[d, a, c, b]] * 3
            ... ))
            >>> cowinners(profile)
            [0, 1, 3]
        """
        n_c = matrix_duels.shape[0]
        # Margin graph
        margin_graph = np.array(matrix_duels - matrix_duels.T, dtype=float)
        margin_graph[margin_graph <= 0] = -np.inf
        # Weak Condorcet winners
        weak_condorcet_winners = np.all(margin_graph >= 0, axis=1)
        # Calculate widest path for candidate c to d
        strength = margin_graph.copy()
        for i in range(n_c):
            for j in range(n_c):
                if i == j:
                    continue
                if not weak_condorcet_winners[j]:
                    for k in range(n_c):
                        if k == i or k == j:
                            continue
                        strength[j][k] = max(strength[j][k], min(strength[j][i], strength[i][k]))
        # Winners
        better_or_tie_c = np.sum(strength >= margin_graph.T, 1)
        candidates_by_scores_best_to_worst = np.argsort(- better_or_tie_c, kind='mergesort')
        winners = np.where(better_or_tie_c == np.max(better_or_tie_c))[0]
        w = candidates_by_scores_best_to_worst[0]
        return {'w': w, 'winners': winners, 'margin_graph': margin_graph, 'strength': strength,
                'better_or_tie_c': better_or_tie_c}

    @cached_property
    def _count_ballots_(self):
        self.mylog("Count ballots", 1)
        return self._count_ballot_aux(self.profile_.matrix_duels_rk)

    @cached_property
    def w_(self):
        return self._count_ballots_['w']

    @cached_property
    def candidates_by_scores_best_to_worst_(self):
        return self._count_ballots_['candidates_by_scores_best_to_worst']

    @cached_property
    def scores_(self):
        return self._count_ballots_['better_or_tie_c']

    # %% Manipulation criteria of the voting system

    @cached_property
    def meets_condorcet_c_rk_ctb(self):
        # TO DO : to check
        return True
