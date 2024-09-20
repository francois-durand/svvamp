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

    Options
    -------
        >>> RuleSplitCycle.print_options_parameters()
        cm_option: ['lazy', 'exact']. Default: 'lazy'.
        icm_option: ['exact']. Default: 'exact'.
        iia_subset_maximum_size: is_number. Default: 2.
        im_option: ['lazy', 'exact']. Default: 'lazy'.
        tm_option: ['lazy', 'exact']. Default: 'exact'.
        um_option: ['lazy', 'exact']. Default: 'lazy'.

    Notes
    -----
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

    References
    ----------
    'Split cycle: A new Condorcet consistent voting method independent of clones and immune to spoilers.' Holliday
    & Pacuit, 2020.

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
        >>> rule = RuleSplitCycle()(profile)
        >>> rule.demo_results_(log_depth=0)  # doctest: +NORMALIZE_WHITESPACE
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
        [3 1 2]
        candidates_by_scores_best_to_worst
        [0 2 1]
        scores_best_to_worst
        [3 2 1]
        w = 0
        score_w = 3
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
            ...     my_better_or_tie_c = results['better_or_tie_c']
            ...     return [int(cand) for cand in np.where(my_better_or_tie_c == np.max(my_better_or_tie_c))[0]]

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
        w = int(candidates_by_scores_best_to_worst[0])
        return {'w': w, 'winners': winners, 'margin_graph': margin_graph, 'strength': strength,
                'better_or_tie_c': better_or_tie_c,
                'candidates_by_scores_best_to_worst': candidates_by_scores_best_to_worst}

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

    # %% Coalition Manipulation (CM)

    def _cm_preliminary_checks_c_subclass_(self, c, optimize_bounds):
        # For each candidate `k != c`, we must have `S(c, k) >= W(k, c)`, where `S` denotes the strength (width of
        # the widest path) and `W` denotes the result of the direct duel. Note that the widest path necessarily
        # goes through the edge `W(j, k)` for some `j != k`. So, there must exist `j` such that `W(j, k) >= W(k, c)`.
        # In other words, `max_j W(j, k) >= W(k, c)`. Since we may add manipulators, the condition is
        # `n_m + max_j W(j, k) >= W(k, c)`, and finally `n_m >= max_k ( W(k, c) - max_j W(j, k) )`.
        n_c = self.profile_.n_c
        profile_sincere = Profile(
            preferences_borda_rk=self.profile_.preferences_borda_rk[np.logical_not(self.v_wants_to_help_c_[:, c]), :]
        )
        matrix_duels_sincere = profile_sincere.matrix_duels_rk
        is_not_c = (np.arange(n_c) != c)
        worst_defeat_k = np.max(matrix_duels_sincere[:, is_not_c], axis=0)
        score_k_against_c = matrix_duels_sincere[is_not_c, c]
        n_m_necessary = np.max(score_k_against_c - worst_defeat_k)
        self._update_necessary(self._necessary_coalition_size_cm, c, n_m_necessary,
                               'CM: Preliminary check: necessary_coalition_size_cm =')
