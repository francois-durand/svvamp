# -*- coding: utf-8 -*-
"""
Created on 11 dec. 2018, 07:50
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
import networkx as nx
from svvamp.rules.rule import Rule
from svvamp.utils.util_cache import cached_property
from svvamp.preferences.profile import Profile


class RuleRankedPairs(Rule):
    """Tideman's Ranked Pairs.

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
        >>> rule = RuleRankedPairs()(profile)
        >>> rule.meets_condorcet_c_rk_ctb
        True
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
        [[0. 3. 3.]
         [0. 0. 0.]
         [0. 3. 0.]]
        candidates_by_scores_best_to_worst
        [0, 2, 1]
        scores_best_to_worst
        [[0. 3. 3.]
         [0. 0. 3.]
         [0. 0. 0.]]
        w = 0
        score_w = [[0. 3. 3.]]
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

    Notes
    -----
    In the matrix of duels :attr:`matrix_duels_rk`, victories (and ties) are sorted by decreasing amplitude. If two
    duels have the same score, we take first the one where the winner has the smallest index; if there is still a
    choice to make, we take first the duel where the loser has the highest index.

    Starting with the largest victory, we build a directed graph whose nodes are the candidates and edges are
    victories. But if a victory creates a cycle in the graph, it is not validated and the edge is not added.

    At the end, we have a transitive connected directed graph, whose adjacency relation is included in the relation
    of victories (with ties broken), :attr:`matrix_victories_rk_ctb`. The maximal node of this graph (by topological
    order) is declared the winner.

    This method meets the Condorcet criterion.

    * :meth:`is_cm_`: Deciding CM is NP-complete. Non-polynomial or non-exact algorithms from superclass :class:`Rule`.
    * :meth:`is_icm_`: Exact in polynomial time.
    * :meth:`is_im_`: Deciding IM is NP-complete. Non-polynomial or non-exact algorithms from superclass :class:`Rule`.
    * :meth:`is_iia_`: Exact in polynomial time.
    * :meth:`is_tm_`: Exact in polynomial time.
    * :meth:`is_um_`: Non-polynomial or non-exact algorithms from superclass :class:`Rule`.

    References
    ----------
    'Independence of clones as a criterion for voting rules', Nicolaus Tideman, 1987.

    'Complexity of Unweighted Coalitional Manipulation under Some Common Voting Rules', Lirong Xia et al., 2009.

    'Schulze and Ranked-Pairs Voting are Fixed-Parameter Tractable to Bribe, Manipulate, and Control',
    Lane A. Hemaspaandra, Rahman Lavaee and Curtis Menton, 2012.

    'A Complexity-of-Strategic-Behavior Comparison between Schulze’s Rule and Ranked Pairs', David Parkes and
    Lirong Xia, 2012.
    """

    full_name = 'Ranked Pairs'
    abbreviation = 'RP'

    options_parameters = Rule.options_parameters.copy()
    options_parameters['icm_option'] = {'allowed': ['exact'], 'default': 'exact'}

    def __init__(self, **kwargs):
        super().__init__(
            with_two_candidates_reduces_to_plurality=True, is_based_on_rk=True,
            precheck_icm=False,
            log_identity="RANKED_PAIRS", **kwargs
        )

    # %% Counting the ballots

    @cached_property
    def _count_ballots_(self):
        self.mylog("Count ballots", 1)
        matrix_duels_vtb_temp = np.copy(self.profile_.matrix_duels_rk)
        dig = nx.DiGraph()
        dig.add_nodes_from(range(self.profile_.n_c))
        while True:
            best_duel_result_c = np.max(matrix_duels_vtb_temp, 1)
            c = np.argmax(best_duel_result_c)
            best_duel_result = best_duel_result_c[c]
            d = np.where(matrix_duels_vtb_temp[c, :] == best_duel_result)[0][-1]
            if best_duel_result == 0:
                break
            matrix_duels_vtb_temp[c, d] = 0
            matrix_duels_vtb_temp[d, c] = 0
            if not nx.has_path(dig, d, c):
                dig.add_edge(c, d, weight=self.profile_.matrix_duels_rk[c, d])
        candidates_by_scores_best_to_worst = list(nx.topological_sort(dig))
        w = candidates_by_scores_best_to_worst[0]
        scores = nx.to_numpy_matrix(dig)
        return {'scores': scores, 'w': w, 'candidates_by_scores_best_to_worst': candidates_by_scores_best_to_worst}

    @cached_property
    def w_(self):
        return self._count_ballots_['w']

    @cached_property
    def candidates_by_scores_best_to_worst_(self):
        """1d array of integers. ``candidates_by_scores_best_to_worst[k]`` is the ``k``\ :sup:`th` candidate by
        topological order on the graph generated by Ranked Pairs.
        """
        return self._count_ballots_['candidates_by_scores_best_to_worst']

    @cached_property
    def scores_(self):
        """2d array of integers. ``scores[c, d]`` is equal to :attr:`matrix_duels_rk`\ ``[c, d]`` iff this duel was
        validated in Ranked Pairs, 0 otherwise.

        .. note::

            Unlike for most other voting systems, ``scores`` matrix must be read in rows, in order to comply with our
            convention for the matrix of duels: ``c``'s score vector is ``scores[c, :]``.
        """
        return self._count_ballots_['scores']

    @cached_property
    def score_w_(self):
        """1d array. ``score_w_`` is :attr:`w_`'s score vector: ``score_w_`` =
        :attr:`scores_`\ ``[``:attr:`w_`\ ``, :]``.
        """
        self.mylog("Compute winner's score", 1)
        return self.scores_[self.w_, :]

    @cached_property
    def scores_best_to_worst_(self):
        """2d array. ``scores_best_to_worst_`` is the scores of the candidates, from the winner to the last candidate
        of the election.

        ``scores_best_to_worst[k, j]`` is the score of the ``k``\ :sup:`th` best candidate of the election against
        the ``j``\ :sup:`th`. It is the result in :attr:`~svvamp.Population.matrix_duels_rk` if this duels was
        validated by Ranked Pairs, 0 otherwise.
        """
        self.mylog("Compute scores_best_to_worst", 1)
        return self.scores_[self.candidates_by_scores_best_to_worst_, :][:, self.candidates_by_scores_best_to_worst_]

    @cached_property
    def v_might_im_for_c_(self):
        # This is a quite lazy version, we could be more precise (possible future work). For a voter to be pivotal,
        # it is necessary that she makes a duel examined before another one in Ranked Pairs algorithm (this includes
        # the case of making a duel ``(c, d)`` examined before ``(d, c)``, i.e. changing a defeat into a victory)
        self.mylog("Compute v_might_im_for_c_", 1)
        scores_duels = np.sort(np.concatenate((
            self.profile_.matrix_duels_rk[np.triu_indices(self.profile_.n_c, 1)],
            self.profile_.matrix_duels_rk[np.tril_indices(self.profile_.n_c, -1)]
        )))
        one_v_might_be_pivotal = np.any(scores_duels[:-1] + 2 >= scores_duels[1:])
        return np.full((self.profile_.n_v, self.profile_.n_c), one_v_might_be_pivotal)

    # %% Manipulation criteria of the voting system

    @cached_property
    def meets_condorcet_c_rk_ctb(self):
        return True
