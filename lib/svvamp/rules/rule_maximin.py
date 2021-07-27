# -*- coding: utf-8 -*-
"""
Created on 10 dec. 2018, 15:11
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
from svvamp.utils.misc import preferences_ut_to_matrix_duels_ut
from svvamp.utils.util_cache import cached_property
from svvamp.utils.pseudo_bool import equal_true
from svvamp.preferences.profile import Profile


class RuleMaximin(Rule):
    """Maximin method.

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
        >>> rule = RuleMaximin()(profile)
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
        [3 2 2]
        candidates_by_scores_best_to_worst
        [0 1 2]
        scores_best_to_worst
        [3 2 2]
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
        is_im = False
        log_im: im_option = exact
        candidates_im =
        [0. 0. 0.]
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
        is_um = False
        log_um: um_option = exact
        candidates_um =
        [0. 0. 0.]
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
        is_cm = False
        log_cm: cm_option = fast, um_option = exact, tm_option = exact
        candidates_cm =
        [0. 0. 0.]
        necessary_coalition_size_cm =
        [0. 2. 3.]
        sufficient_coalition_size_cm =
        [0. 2. 3.]

    Notes
    -----
    Candidate ``c``'s score is the minimum of the row :attr:`matrix_duels_rk`\ ``[c, :]`` (except the diagonal term),
    i.e. the result of candidate ``c`` for her worst duel. The candidate with highest score is declared the winner.
    In case of a tie, the candidate with lowest index wins.

    This method meets the Condorcet criterion.

    * :meth:`is_cm_`: Deciding CM is NP-complete, even for 2 manipulators.

        * :attr:`cm_option` = ``'faster'``: Zuckerman et al. (2011) (cf. below). The difference with option ``fast`` is
          that, if CM is proven possible or impossible, we optimize the bounds only based on UM, and not on CM. Hence
          this option is as precise as ``fast`` to compute ``is_cm_``, but less precise for the bounds
          ``necessary_coalition_size_cm_`` and ``sufficient_coalition_size_cm_``.
        * :attr:`cm_option` = ``'fast'``: Zuckerman et al. (2011). This approximation algorithm is polynomial and has
          a multiplicative factor of error of 5/3 on the number of manipulators needed.
        * :attr:`cm_option` = ``'exact'``: Non-polynomial algorithm from superclass :class:`Rule`.

    * :meth:`is_icm_`: Exact in polynomial time.
    * :meth:`is_im_`: Exact in polynomial time.
    * :meth:`is_iia_`: Exact in polynomial time.
    * :meth:`is_tm_`: Exact in polynomial time.
    * :meth:`is_um_`: Exact in polynomial time.

    References
    ----------
    'Complexity of Unweighted Coalitional Manipulation under Some Common Voting Rules', Lirong Xia et al., 2009.

    'An algorithm for the coalitional manipulation problem under Maximin', Michael Zuckerman, Omer Lev and
    Jeffrey S. Rosenschein, 2011.
    """

    full_name = 'Maximin'
    abbreviation = 'Max'

    options_parameters = Rule.options_parameters.copy()
    options_parameters.update({
        'im_option': {'allowed': ['exact'], 'default': 'exact'},
        'tm_option': {'allowed': ['exact'], 'default': 'exact'},
        'um_option': {'allowed': ['exact'], 'default': 'exact'},
        'icm_option': {'allowed': ['exact'], 'default': 'exact'},
        'cm_option': {'allowed': ['faster', 'fast', 'exact'], 'default': 'fast'}
    })

    def __init__(self, **kwargs):
        super().__init__(
            with_two_candidates_reduces_to_plurality=True, is_based_on_rk=True,
            precheck_icm=False,
            log_identity="MAXIMIN", **kwargs
        )

    @cached_property
    def scores_(self):
        """1d array of integers. ``scores[c]`` is the minimum of the row :attr:`matrix_duels_rk`\ ``[c, :]`` (except
        the diagonal term), i.e. the result of candidate ``c`` for her worst duel.
        """
        self.mylog("Compute scores", 1)
        return np.add(self.profile_.n_v, - np.max(self.profile_.matrix_duels_rk, 0))

    # %% Manipulation criteria of the voting system

    @cached_property
    def meets_condorcet_c_rk_ctb(self):
        return True

    # %% Manipulation routine

    def _vote_strategically_(self, matrix_duels_r, scores_r, c, weight=1):
        """Strategic vote by one elector, in favor of candidate ``c``.

        Parameters
        ----------
        matrix_duels_r : list of list (or ndarray)
            2d array of integers. Diagonal coefficients must have an arbitrary value that is greater than all the
            other ones (``+ inf`` is a good idea).
        scores_r : ndarray
            1d array of integers. Scores corresponding to ``matrix_duels_r``.
        c : int
            Candidate.
        weight : int
            Weight of the voter (used for UM, 1 otherwise).

        Returns
        -------
        None
            This method modifies ``matrix_duels_r`` and ``scores_r`` IN PLACE.

        References
        ----------
        Algorithm from Zuckerman et al. : An Algorithm for the Coalitional Manipulation Problem under Maximin. For
        only one elector (or unison manipulation), this algorithm is optimal.
        """
        self.mylogv("AUX: c =", c, 3)
        self.mylogm("AUX: matrix_duels_r =", matrix_duels_r, 3)
        self.mylogv("AUX: scores_r =", scores_r, 3)
        # Manipulator's vote (denoted P_i in the paper) is represented in Borda format.
        strategic_ballot = np.zeros(self.profile_.n_c)
        strategic_ballot[c] = self.profile_.n_c
        next_borda_score_to_add = self.profile_.n_c - 1
        has_been_sent_to_stacks = np.zeros(self.profile_.n_c)
        candidates = np.array(range(self.profile_.n_c))
        stacks_q = StackFamily()

        # Building the directed graph (cf. paper).
        digraph = np.zeros((self.profile_.n_c, self.profile_.n_c))
        for x in range(self.profile_.n_c):
            if x == c:
                continue
            if matrix_duels_r[x, c] == scores_r[x]:
                continue
            for y in range(self.profile_.n_c):
                if y == c or y == x:
                    continue
                if matrix_duels_r[x, y] == scores_r[x]:
                    digraph[x, y] = True

        while next_borda_score_to_add > 0:
            self.mylogm("AUX: digraph =", digraph, 3)
            self.mylogv("AUX: strategic_ballot =", strategic_ballot, 3)
            # candidates_not_dangerous = set A in the paper
            candidates_not_dangerous = np.logical_and(np.logical_not(np.any(digraph, 1)),
                                                      np.logical_not(strategic_ballot))
            self.mylogv("AUX: candidates_not_dangerous =", candidates_not_dangerous, 3)
            if np.any(candidates_not_dangerous):
                # Transfer these candidates into the stacks
                for score, candidate in zip(scores_r[candidates_not_dangerous], candidates[candidates_not_dangerous]):
                    if not has_been_sent_to_stacks[candidate]:
                        stacks_q.push_front(score, candidate)
                        has_been_sent_to_stacks[candidate] = True
                self.mylogv("AUX: stacks_q =", stacks_q.data, 3)
                # Put a non-dangerous candidate in the ballot
                b = stacks_q.pop_front()
                self.mylogv("AUX: Found non-dangerous candidate b =", b, 3)
                self.mylogv("AUX: stacks_q =", stacks_q.data, 3)
                strategic_ballot[b] = next_borda_score_to_add
                next_borda_score_to_add -= 1
            else:
                s = np.min(scores_r[np.logical_not(strategic_ballot)])
                b_have_s = np.logical_and(np.logical_not(strategic_ballot), scores_r == s)
                self.mylogv("AUX: s =", s, 3)
                self.mylogv("AUX: b_have_s =", b_have_s, 3)
                # In the general case, we take any ``b`` with least score. We choose the one with highest index.
                b = np.where(b_have_s)[0][-1]
                self.mylogv("AUX: First guess b =", b, 3)
                # But there might be a particular case (special cycle)...
                if sum(b_have_s) > 1:
                    found_special_b = False
                    dig = nx.DiGraph(digraph)
                    self.mylogv("AUX: G.nodes = ", dig.nodes, 3)
                    self.mylogv("AUX: G.edges = ", dig.edges, 3)
                    # We look for ``b`` in reverse order of index. This way, if several candidates meet the condition,
                    # the one with highest index is chosen.
                    for b_test in candidates[b_have_s][::-1]:
                        possible_cs = np.where(np.logical_and(digraph[:, b_test], b_have_s))
                        for c_test in possible_cs[0]:
                            self.mylogv("AUX: b_test = ", b_test, 3)
                            self.mylogv("AUX: c_test = ", c_test, 3)
                            if nx.has_path(dig, b_test, c_test):
                                self.mylogv("AUX: Found special b =", b_test, 3)
                                b = b_test
                                found_special_b = True
                                break
                        if found_special_b:
                            break
                # We put ``b`` in the ballot
                strategic_ballot[b] = next_borda_score_to_add
                next_borda_score_to_add -= 1
            # Now update the digraph
            self.mylogm("AUX: digraph =", digraph, 3)
            for y in candidates[np.logical_not(strategic_ballot)]:
                if digraph[y, b] and not has_been_sent_to_stacks[y]:
                    self.mylogv("AUX: y =", y, 3)
                    self.mylogv("AUX: b =", b, 3)
                    self.mylogv("AUX: Transfer from digraph to stack: y =", y, 3)
                    digraph[y, :] = False
                    stacks_q.push_front(scores_r[y], y)
                    has_been_sent_to_stacks[y] = True
                    self.mylogv("AUX: stacks_q =", stacks_q.data, 3)
        # Finally, update ``matrix_duels_r`` and ``scores_r``.
        self.mylogm("AUX: matrix_duels_r =", matrix_duels_r, 3)
        self.mylogv("AUX: scores_r =", scores_r, 3)
        self.mylogv("AUX: strategic_ballot =", strategic_ballot, 3)
        for x in range(self.profile_.n_c):
            for y in range(x + 1, self.profile_.n_c):
                if strategic_ballot[x] > strategic_ballot[y]:
                    matrix_duels_r[x, y] += weight
                else:
                    matrix_duels_r[y, x] += weight
        # The ugly thing below (instead of ``scores_r = np.min(matrix_duels_r, 1)``) is made so that ``scores_r`` is
        # modified in place.
        scores_r_temp = np.min(matrix_duels_r, 1)
        for x in range(self.profile_.n_c):
            scores_r[x] = scores_r_temp[x]
        self.mylogm("AUX: matrix_duels_r =", matrix_duels_r, 3)
        self.mylogv("AUX: scores_r =", scores_r, 3)

    # %% Individual manipulation (IM)

    def _im_main_work_v_(self, v, c_is_wanted, nb_wanted_undecided, stop_if_true):
        """
            >>> profile = Profile(preferences_rk=[
            ...     [0, 2, 1],
            ...     [0, 2, 1],
            ...     [1, 0, 2],
            ...     [1, 0, 2],
            ...     [2, 0, 1],
            ... ])
            >>> rule = RuleMaximin()(profile)
            >>> rule.is_im_c_with_voters_(2)
            (False, array([0., 0., 0., 0., 0.]))

            >>> profile = Profile(preferences_rk=[
            ...     [0, 2, 1],
            ...     [0, 2, 1],
            ...     [1, 0, 2],
            ...     [1, 0, 2],
            ...     [2, 0, 1],
            ... ])
            >>> rule = RuleMaximin()(profile)
            >>> rule.is_im_
            False

            >>> profile = Profile(preferences_rk=[
            ...     [0, 1, 2],
            ...     [1, 0, 2],
            ...     [1, 2, 0],
            ...     [2, 0, 1],
            ...     [2, 0, 1],
            ... ])
            >>> rule = RuleMaximin()(profile)
            >>> rule.is_im_
            True

            >>> profile = Profile(preferences_rk=[
            ...     [0, 1, 2],
            ...     [0, 1, 2],
            ...     [1, 2, 0],
            ...     [2, 1, 0],
            ...     [2, 1, 0],
            ... ])
            >>> rule = RuleMaximin()(profile)
            >>> rule.is_im_
            False

            >>> profile = Profile(preferences_rk=[
            ...     [0, 2, 1, 3],
            ...     [1, 3, 0, 2],
            ...     [2, 1, 0, 3],
            ...     [2, 1, 3, 0],
            ...     [3, 0, 1, 2],
            ... ])
            >>> rule = RuleMaximin()(profile)
            >>> rule.is_im_
            True
        """
        for c in self.losing_candidates_:
            if not c_is_wanted[c]:
                continue
            if not np.isneginf(self._v_im_for_c[v, c]):
                continue
            self.mylogv("IM: c =", c, 3)
            nb_wanted_undecided -= 1

            matrix_duels_r = np.copy(self.profile_.matrix_duels_ut).astype(np.float)
            for x in range(self.profile_.n_c):
                matrix_duels_r[x, x] = np.inf
                for y in range(x + 1, self.profile_.n_c):
                    if self.profile_.preferences_borda_rk[v, x] > self.profile_.preferences_borda_rk[v, y]:
                        matrix_duels_r[x, y] -= 1
                    else:
                        matrix_duels_r[y, x] -= 1
            scores_r = np.min(matrix_duels_r, 1)

            w_temp = np.argmax(scores_r)
            if w_temp == c:
                self.mylogv("IM: scores_r =", scores_r, 3)
                self.mylog("IM: Manipulation easy (c wins without manipulator's vote)", 3)
                self._v_im_for_c[v, c] = True
                self._candidates_im[c] = True
                self._voters_im[v] = True
                self._is_im = True
                self.mylog("IM found", 3)
                if stop_if_true or nb_wanted_undecided == 0:
                    return
                continue  # pragma: no cover - Not really "executed" because of Python optimization
            # Best we can do: improve ``c`` by one, do not move the others
            if scores_r[w_temp] > scores_r[c] + 1 or (scores_r[w_temp] >= scores_r[c] + 1 and w_temp < c):
                self.mylogv("IM: scores_r =", scores_r, 3)
                self.mylog("IM: Manipulation impossible (score difference is too high)", 3)
                self._v_im_for_c[v, c] = False
                if nb_wanted_undecided == 0:
                    return
                continue  # pragma: no cover - Not really "executed" because of Python optimization

            self._vote_strategically_(matrix_duels_r, scores_r, c, 1)
            w_r = np.argmax(scores_r)
            self.mylogv("IM: w_r =", w_r, 3)
            if w_r == c:
                self.mylog("IM: Manipulation worked!", 3)
            else:
                self.mylog("IM: Manipulation failed...", 3)

            # We can conclude.
            self._v_im_for_c[v, c] = (w_r == c)
            if equal_true(self._v_im_for_c[v, c]):
                self._candidates_im[c] = True
                self._voters_im[v] = True
                self._is_im = True
                self.mylog("IM found", 3)
                if stop_if_true:
                    return
            if nb_wanted_undecided == 0:
                return

    # %% Trivial Manipulation (TM)

    # Use the general methods.

    # %% Ignorant-Coalition Manipulation (ICM)

    # Use the general methods.

    # %% Unison manipulation (UM)

    def _um_main_work_c_(self, c):
        """
            >>> profile = Profile(preferences_rk=[
            ...     [0, 1, 2],
            ...     [1, 0, 2],
            ...     [1, 2, 0],
            ...     [2, 0, 1],
            ...     [2, 0, 1],
            ... ])
            >>> rule = RuleMaximin()(profile)
            >>> rule.candidates_um_
            array([0., 1., 1.])
        """
        matrix_duels_temp = preferences_ut_to_matrix_duels_ut(
            self.profile_.preferences_borda_rk[np.logical_not(self.v_wants_to_help_c_[:, c]), :]
        ).astype(float)
        for x in range(self.profile_.n_c):
            matrix_duels_temp[x, x] = np.inf
        scores_temp = np.min(matrix_duels_temp, 1)
        n_m = self.profile_.matrix_duels_ut[c, self.w_]

        w_temp = np.argmax(scores_temp)
        if w_temp == c:  # pragma: no cover
            # TO DO: Investigate whether this case can really happen or not.
            self._reached_uncovered_code()
            self.mylogv("UM: scores_temp =", scores_temp, 3)
            self.mylog("UM: Manipulation easy (c wins without manipulators' votes)", 3)
            self._candidates_um[c] = True
            return
        # Best we can do: improve ``c`` by ``n_m``, do not move the others
        if scores_temp[w_temp] > scores_temp[c] + n_m or (scores_temp[w_temp] >= scores_temp[c] + n_m and w_temp < c):
            self.mylogv("UM: scores_temp =", scores_temp, 3)
            self.mylog("UM: Manipulation impossible (score difference is too high)", 3)
            self._candidates_um[c] = False
            return

        self._vote_strategically_(matrix_duels_temp, scores_temp, c, n_m)
        w_temp = np.argmax(scores_temp)
        self.mylogv("UM: w_temp =", w_temp, 3)
        self._candidates_um[c] = (w_temp == c)

    def sufficient_coalition_size_um_c_(self, c):
        # TODO: put this in cache
        n_s = self.profile_.n_v - self.profile_.matrix_duels_ut[c, self.w_]
        matrix_duels_s = preferences_ut_to_matrix_duels_ut(
            self.profile_.preferences_borda_rk[np.logical_not(self.v_wants_to_help_c_[:, c]), :]
        ).astype(float)
        for x in range(self.profile_.n_c):
            matrix_duels_s[x, x] = np.inf
        scores_s = np.min(matrix_duels_s, 1)
        w_s = np.argmax(scores_s)
        # If w_s == c, we win without any manipulator
        if w_s == c:
            self.mylogv("UM: scores_s =", scores_s, 3)
            self.mylog("UM: Manipulation easy (c wins without manipulators' votes)", 3)
            return 0
        # If we have n_s + 1 manipulators, obviously we can manipulate.
        n_m_sup = n_s + 1
        # Best we can do: improve ``c`` by ``n_m``, do not move the others => we need at least n_m_inf + 1 manipulators.
        n_m_inf = scores_s[w_s] - scores_s[c] - int(c < w_s)
        # Loop invariant: UM always possible n_m_sup manipulators, impossible for n_m_inf manipulators
        while n_m_sup - n_m_inf > 1:
            n_m = (n_m_inf + n_m_sup) // 2
            matrix_duels_temp = matrix_duels_s.copy()
            scores_temp = scores_s.copy()
            self._vote_strategically_(matrix_duels_temp, scores_temp, c, n_m)
            w_temp = np.argmax(scores_temp)
            self.mylogv("UM: w_temp =", w_temp, 3)
            if w_temp == c:
                n_m_sup = n_m
            else:
                n_m_inf = n_m
        return n_m_sup

    # %% Coalition Manipulation (CM)

    def _cm_main_work_c_fast(self, c, optimize_bounds):
        """
            >>> profile = Profile(preferences_rk=[
            ...     [0, 1, 2],
            ...     [0, 1, 2],
            ...     [1, 2, 0],
            ...     [2, 1, 0],
            ...     [2, 1, 0],
            ... ])
            >>> rule = RuleMaximin()(profile)
            >>> rule.candidates_cm_
            array([nan,  0.,  0.])
        """
        matrix_duels_temp = preferences_ut_to_matrix_duels_ut(
            self.profile_.preferences_borda_rk[np.logical_not(self.v_wants_to_help_c_[:, c]), :]
        ).astype(float)
        for x in range(self.profile_.n_c):
            matrix_duels_temp[x, x] = np.inf
        scores_temp = np.min(matrix_duels_temp, 1)
        w_temp = np.argmax(scores_temp)
        n_manipulators_used = 0

        # Easy lower bound: for each manipulator, ``c``'s score can at best increase by one, and ``w_temp``'s score
        # cannot decrease.
        self._update_necessary(
            self._necessary_coalition_size_cm, c, scores_temp[w_temp] - scores_temp[c] + (c > w_temp),
            'CM: Update necessary_coalition_size_cm = scores_s[w_s] - scores_s[c] + (c > w_s) =')
        if not optimize_bounds and (self._necessary_coalition_size_cm[c] > self.profile_.matrix_duels_ut[c, self.w_]):
            return True  # is_quick_escape

        # Relatively easy upper bound: if we can UM, we can CM.
        self._update_sufficient(
            self._sufficient_coalition_size_cm, c, self.sufficient_coalition_size_um_c_(c),
            'CM: Update sufficient_coalition_size_cm = sufficient_coalition_size_um =')
        if self._sufficient_coalition_size_cm[c] == self._necessary_coalition_size_cm[c]:
            return False  # Not a quick escape, we did all the job!
        if not optimize_bounds and (self.profile_.matrix_duels_ut[c, self.w_] >= self._sufficient_coalition_size_cm[c]):
            return True  # is_quick_escape

        # If the option is `faster` and if is_cm_c_ is decided, then we can exit here, because we do not want to
        # improve the bounds anymore.
        if self.cm_option == 'faster':
            if self.profile_.matrix_duels_ut[c, self.w_] >= self._sufficient_coalition_size_cm[c]:
                return False  # not a quick escape
            if self._necessary_coalition_size_cm[c] > self.profile_.matrix_duels_ut[c, self.w_]:
                return False  # not a quick escape

        while w_temp != c:
            self.mylogv("CM: w_temp =", w_temp, 3)
            self.mylogv("CM: c =", c, 3)
            n_manipulators_used += 1
            if n_manipulators_used >= self._sufficient_coalition_size_cm[c]:
                self.mylog("CM: I already know a strategy that works with n_manipulators_used (TM, UM, etc.).")
                self._update_necessary(self._necessary_coalition_size_cm, c, np.ceil(n_manipulators_used * 3 / 5),
                                       'CM: Update necessary_coalition_size_cm =')
                break
            self._vote_strategically_(matrix_duels_temp, scores_temp, c, 1)
            w_temp = np.argmax(scores_temp)
        else:
            self.mylogm("CM: matrix_duels_temp =", matrix_duels_temp, 3)
            self.mylogv("CM: scores_temp =", scores_temp, 3)
            self.mylogv("CM: w_temp =", w_temp, 3)
            self._update_sufficient(self._sufficient_coalition_size_cm, c, n_manipulators_used,
                                    'CM: Update sufficient_coalition_size_cm =')
            # For the 3/5 ratio, see Zuckerman et al.
            self._update_necessary(self._necessary_coalition_size_cm, c, np.ceil(n_manipulators_used * 3 / 5),
                                   'CM: Update necessary_coalition_size_cm =')
        return False

    def _cm_main_work_c_(self, c, optimize_bounds):
        """
            >>> profile = Profile(preferences_rk=[
            ...     [0, 2, 1],
            ...     [0, 2, 1],
            ...     [1, 0, 2],
            ...     [1, 0, 2],
            ...     [2, 0, 1],
            ... ])
            >>> rule = RuleMaximin(cm_option='exact')(profile)
            >>> rule.candidates_cm_
            array([0., 0., 0.])
        """
        is_quick_escape_fast = self._cm_main_work_c_fast(c, optimize_bounds)
        if not self.cm_option == "exact":
            # With 'fast' or 'faster' option, we stop here anyway.
            return is_quick_escape_fast

        # From this point, we have necessarily the 'exact' option (which is, in fact, only an exhaustive exploration
        # with = n_m manipulators).
        is_quick_escape_exact = self._cm_main_work_c_exact_(c, optimize_bounds)
        return is_quick_escape_fast or is_quick_escape_exact


class StackFamily:
    """Family of stacks. Used for the manipulation of Maximin.

    >>> sf = StackFamily()
    >>> print(sf.pop_front())
    None
    >>> sf.push_front(42, 4)
    >>> sf.push_front(51, 0)
    >>> sf.push_front(42, 2)
    >>> sf.push_front(42, 3)
    >>> sf
    {42: [4, 2, 3], 51: [0]}
    >>> sf.pop_front()
    3
    >>> sf
    {42: [4, 2], 51: [0]}

    A StackFamily contains a dictionary whose keys are numbers and values are lists.
    """

    def __init__(self):
        self.data = dict()

    def __repr__(self):
        return '{' + ', '.join(['%s: %s' % (k, self.data[k]) for k in sorted(self.data.keys())]) + '}'

    def push_front(self, score, candidate):
        """Push a value to the appropriate stack (given by 'score').
        """
        try:
            self.data[score].append(candidate)
        except KeyError:
            self.data[score] = [candidate]

    def pop_front(self):
        """Pop a value: choose the stack with minimum score, and pop the most recent item in this stack.

        When the object is empty, returns None.
        """
        if len(self.data) == 0:
            return None
        lowest_score = min(self.data.keys())
        c = self.data[lowest_score].pop()
        if len(self.data[lowest_score]) == 0:
            del self.data[lowest_score]
        return c
