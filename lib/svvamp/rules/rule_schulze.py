# -*- coding: utf-8 -*-
"""
Created on 11 dec. 2018, 08:45
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
from svvamp.utils.misc import preferences_ut_to_matrix_duels_ut
from svvamp.utils.pseudo_bool import equal_true
from svvamp.preferences.profile import Profile


class RuleSchulze(Rule):
    """Schulze method.

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
        >>> rule = RuleSchulze()(profile)
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
        [[0 3 3]
         [2 0 2]
         [2 3 0]]
        candidates_by_scores_best_to_worst
        [0 2 1]
        scores_best_to_worst
        [[0 3 3]
         [2 0 3]
         [2 2 0]]
        w = 0
        score_w = [0 3 3]
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
        log_im: im_option = fast
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
        log_um: um_option = fast
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
        log_cm: cm_option = fast, tm_option = exact
        candidates_cm =
        [ 0.  0. nan]
        necessary_coalition_size_cm =
        [0. 1. 2.]
        sufficient_coalition_size_cm =
        [0. 2. 3.]

    Notes
    -----
    :attr:`scores_`\ ``[c, d]`` is equal to the width of the widest path from candidate ``c`` to candidate ``d`` in
    the capacited graph defined by :attr:`matrix_duels_rk`. We say that ``c`` is *better* than ``d`` if ``scores_[c,
    d]`` > ``scores_[d, c]``. Candidate ``c`` is a *potential winner* if no candidate ``d`` is *better* than ``c``.

    Among the potential winners, the candidate with lowest index is declared the winner.

    .. note::

        In the original Schulze method, ties are broken at random. However, this feature is not supported by SVVAMP
        because it leads to difficulties for the *definition* of manipulation itself (and all the more for its
        implementation).

    * :meth:`is_cm_`:

        * :attr:`cm_option` = ``'fast'``: Gaspers et al. (2013). This algorithm is polynomial and has a window of
          error of 1 manipulator (due to the tie-breaking rule).
        * :attr:`cm_option` = ``'exact'``: Non-polynomial algorithm from superclass :class:`Rule`.

    * :meth:`is_icm_`: Exact in polynomial time.
    * :meth:`is_im_`:

        * :attr:`im_option` = ``'fast'``: Gaspers et al. (2013). This algorithm is polynomial and may not be able to
          decide IM (due to the tie-breaking rule).
        * :attr:`im_option` = ``'exact'``: Non-polynomial algorithm from superclass :class:`Rule`.

    * :meth:`~svvamp.Election.not_iia`: Exact in polynomial time.
    * :meth:`is_tm_`: Exact in polynomial time.
    * :meth:`is_um_`:

        * :attr:`um_option` = ``'fast'``: Gaspers et al. (2013). This algorithm is polynomial and has a window of
          error of 1 manipulator (due to the tie-breaking rule).
        * :attr:`um_option` = ``'exact'``: Non-polynomial algorithm from superclass :class:`Rule`.

    .. note:

        For this voting system, UM and CM are almost equivalent up to tie-breaking. For this reason,
        :attr:`um_option` and :attr:`cm_option` are linked to each other: modifying one modifies the other accordingly.

    References
    ----------
    'A new monotonic, clone-independent, reversal symmetric, and Condorcet-consistent single-winner election
    method ', Markus Schulze, 2011.

    'Schulze and Ranked-Pairs Voting are Fixed-Parameter Tractable to Bribe, Manipulate, and Control',
    Lane A. Hemaspaandra, Rahman Lavaee and Curtis Menton, 2012.

    'Manipulation and Control Complexity of Schulze Voting', Curtis Menton and Preetjot Singh, 2012.

    'A Complexity-of-Strategic-Behavior Comparison between Schulze’s Rule and Ranked Pairs', David Parkes and
    Lirong Xia, 2012.

    'Coalitional Manipulation for Schulze’s Rule', Serge Gaspers, Thomas Kalinowski, Nina Narodytska and Toby
    Walsh, 2013.
    """

    full_name = 'Schulze'
    abbreviation = 'Sch'

    options_parameters = Rule.options_parameters.copy()
    options_parameters.update({
        'im_option': {'allowed': ['fast', 'exact'], 'default': 'fast'},
        'tm_option': {'allowed': ['exact'], 'default': 'exact'},
        'um_option': {'allowed': ['fast', 'exact'], 'default': 'fast'},
        'icm_option': {'allowed': ['exact'], 'default': 'exact'},
        'cm_option': {'allowed': ['fast', 'exact'], 'default': 'fast'}
    })

    def __init__(self, **kwargs):
        super().__init__(
            with_two_candidates_reduces_to_plurality=True, is_based_on_rk=True,
            precheck_um=False, precheck_icm=False,
            log_identity="SCHULZE", **kwargs
        )

    # %% Counting the ballots

    @staticmethod
    def _count_ballot_aux(matrix_duels):
        n_c = matrix_duels.shape[0]
        # Calculate widest path for candidate c to d
        widest_path = np.copy(matrix_duels)
        for i in range(n_c):
            for j in range(n_c):
                if i == j:
                    continue
                for k in range(n_c):
                    if k == i or k == j:
                        continue
                    widest_path[j, k] = max(widest_path[j, k], min(widest_path[j, i], widest_path[i, k]))
        # ``c`` is better than ``d`` if ``widest_path[c, d] > widest_path[d, c]``, or if there is equality and
        # ``c < d``. Potential winners: maximal ``better_or_tie[c]``. Tie-break: index of the candidate. N.B.: if we
        # broke ties with ``better_c = sum(... > ...)``, it would not meet ``condorcet_c_vtb_ctb``.
        better_or_tie_c = np.sum(widest_path >= widest_path.T, 1)
        candidates_by_scores_best_to_worst = np.argsort(- better_or_tie_c, kind='mergesort')
        w = candidates_by_scores_best_to_worst[0]
        return {'w': w, 'scores': widest_path, 'candidates_by_scores_best_to_worst': candidates_by_scores_best_to_worst}

    @cached_property
    def _count_ballots_(self):
        self.mylog("Count ballots", 1)
        return self._count_ballot_aux(self.profile_.matrix_duels_rk)

    @cached_property
    def w_(self):
        return self._count_ballots_['w']

    @cached_property
    def candidates_by_scores_best_to_worst_(self):
        """1d array of integers. ``candidates_by_scores_best_to_worst_[k]`` is the ``k``\ :sup:`th` candidate by
        number of Schulze-victories, i.e. the number of candidates ``d`` such that ``c`` is *better* than ``d``.
        """
        return self._count_ballots_['candidates_by_scores_best_to_worst']

    @cached_property
    def scores_(self):
        """2d array of integers. ``scores[c, d]`` is equal to the width of the widest path from ``c`` to ``d``.

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
        """2d array. ``scores_best_to_worst`` is the scores of the candidates, from the winner to the last candidate
        of the election.

        ``scores_best_to_worst[k, j]`` is the width of the widest path from the ``k``\ :sup:`th` best candidate of
        the election to the ``j``\ :sup:`th`.
        """
        self.mylog("Compute scores_best_to_worst", 1)
        return self.scores_[self.candidates_by_scores_best_to_worst_, :][:, self.candidates_by_scores_best_to_worst_]

    # %% Manipulation criteria of the voting system

    @cached_property
    def meets_condorcet_c_rk_ctb(self):
        return True

    # %% Setting the options

    @property
    def um_option(self):
        return self._um_option

    @um_option.setter
    def um_option(self, value):
        if self._um_option == value:
            return
        if value in self.options_parameters['um_option']['allowed']:
            self.mylogv("Setting um_option =", value, 1)
            self._cm_option = value
            self._um_option = value
            self.delete_cache(contains='_um_', suffix='_')
            self.delete_cache(contains='_cm_', suffix='_')
        else:
            raise ValueError("Unknown value for um_option: " + format(value))

    @property
    def cm_option(self):
        return self._cm_option

    @cm_option.setter
    def cm_option(self, value):
        if self._cm_option == value:
            return
        if value in self.options_parameters['cm_option']['allowed']:
            self.mylogv("Setting cm_option =", value, 1)
            self._cm_option = value
            self._um_option = value
            self.delete_cache(contains='_um_', suffix='_')
            self.delete_cache(contains='_cm_', suffix='_')
        else:
            raise ValueError("Unknown value for cm_option: " + format(value))

    # %% Manipulation routine

    def _vote_strategically(self, matrix_duels_s, strength, c, n_m):
        """ Manipulation algorithm: "Coalitional Manipulation for Schulze’s Rule" (Gaspers, Kalinowski, Narodytska
        and Walsh 2013).

        Parameters
        ----------
        matrix_duels_s : ndarray
            2d array. Matrix of duels for sincere voters.
        strength : ndarray
            2d array. ``S[c, d]`` is the strength of the widest path from ``c`` to ``d`` (with sincere
            voters only). Be careful, it is based on the antisymmetric matrix of duels. I.e. while
            ``_count_ballot_aux`` provides ``widest_path_s``, we need to take ``S = 2 * (widest_path_s - n_s / 2)``,
            then set ``S``'s diagonal coefficients to 0.
        c : int
            Candidate. Our challenger.
        n_m : int
            Number of manipulators.
        """
        candidates_not_c = np.concatenate((np.array(range(c), dtype=int),
                                           np.array(range(c + 1, self.profile_.n_c), dtype=int)))
        w = matrix_duels_s - matrix_duels_s.T
        w_sup = w + n_m
        w_inf = w - n_m
        matrix_u = strength + n_m
        matrix_u_prev = None
        self.mylogv("candidates_not_c =", candidates_not_c, 3)
        self.mylogm("matrix_duels_s =", matrix_duels_s, 3)
        self.mylogm("S =", strength, 3)
        self.mylogm("w =", w, 3)
        self.mylogm("w_sup =", w_sup, 3)
        self.mylogm("w_inf =", w_inf, 3)
        self.mylogm("matrix_u =", matrix_u, 3)
        self.mylogm("matrix_u_prev =", matrix_u_prev, 3)
        # Algorithm: Pre-processing Bounds
        while not np.all(matrix_u_prev == matrix_u):
            matrix_u_prev = np.copy(matrix_u)
            # Rule 1
            matrix_u[:, c] = np.minimum(matrix_u[:, c], matrix_u[c, :])
            self.mylogm('Rule 1: U =', matrix_u, 3)
            # Rule 2
            for x in candidates_not_c:
                matrix_g_x = np.ones((self.profile_.n_c, self.profile_.n_c))
                # Remove all vertices y (candidates) s.t. U(y,c) < U(x,c)
                matrix_v_removed = np.less(matrix_u[:, c], matrix_u[x, c])
                matrix_v_removed[c] = False
                matrix_g_x[matrix_v_removed, :] = 0
                matrix_g_x[:, matrix_v_removed] = 0
                # Removed all edges (y,z) s.t. w_sup(y,z) < U(x,c)
                matrix_g_x[w_sup < matrix_u[x, c]] = 0
                # Does matrix_g_x contain a path from c to x ? If not, do stuff.
                if not nx.has_path(nx.DiGraph(matrix_g_x), c, x):
                    matrix_u[x, c] -= 2
                    self.mylogm('Rule 2: U =', matrix_u, 3)
            # Rule 3
            for x in candidates_not_c:
                for y in candidates_not_c:
                    if y == x:
                        continue
                    if matrix_u[x, c] < w_inf[x, y]:
                        matrix_u[y, c] = min(matrix_u[y, c], matrix_u[x, c])
                        self.mylogm('Rule 3: U =', matrix_u, 3)
            # Test possibility
            self.mylogm("U =", matrix_u, 3)
            self.mylogv("U[candidates_not_c, c] =", matrix_u[candidates_not_c, c], 3)
            self.mylogv("S[candidates_not_c, c] - n_m =", strength[candidates_not_c, c] - n_m, 3)
            if np.any(matrix_u[candidates_not_c, c] < strength[candidates_not_c, c] - n_m):
                self.mylog("Cowinner manipulation failed")
                return False, False
        # Algorithm : construction of ordering Lambda (when manipulation is possible)
        array_f = np.zeros(self.profile_.n_c, dtype=np.bool)
        array_f[c] = 1
        array_x = np.ones(self.profile_.n_c, dtype=np.bool)
        array_x[c] = 0
        order_lambda = [c]
        for i in range(self.profile_.n_c - 1):
            the_d = np.max(matrix_u[array_x, c])
            possible_y = np.logical_and(np.logical_and(array_x, matrix_u[:, c] == the_d),
                                        np.any(w_sup[array_f, :] >= the_d, 0))
            y = np.where(possible_y)[0][-1]
            array_f[y] = True
            array_x[y] = False
            order_lambda.append(y)
        order_lambda = np.array(order_lambda)
        self.mylogv("order_lambda =", order_lambda, 3)
        # Who wins with the tie-breaking rule?
        reciprocal_lambda = np.argsort(order_lambda)
        matrix_duels_m = n_m * np.triu(
            np.ones((self.profile_.n_c, self.profile_.n_c)), 1
        )[reciprocal_lambda, :][:, reciprocal_lambda]
        w_temp = self._count_ballot_aux(matrix_duels_s + matrix_duels_m)['w']
        self.mylogv("matrix_duels_m =", matrix_duels_m, 3)
        self.mylogv("w_temp =", w_temp, 3)
        if w_temp == c:
            self.mylog("Cowinner manipulation worked, with ctb also")
            return True, True
        else:
            self.mylog("Cowinner manipulation worked but not with ctb")
            return True, False

    # %% Individual manipulation (IM)

    def _im_main_work_v_fast(self, v, c_is_wanted, nb_wanted_undecided, stop_if_true):
        """
            >>> profile = Profile(preferences_rk=[
            ...     [0, 2, 1],
            ...     [0, 2, 1],
            ...     [1, 0, 2],
            ...     [1, 0, 2],
            ...     [2, 0, 1],
            ... ])
            >>> rule = RuleSchulze()(profile)
            >>> rule.is_im_
            False

            >>> profile = Profile(preferences_rk=[
            ...     [0, 1, 2],
            ...     [1, 0, 2],
            ...     [1, 2, 0],
            ...     [2, 0, 1],
            ...     [2, 0, 1],
            ... ])
            >>> rule = RuleSchulze()(profile)
            >>> rule.is_im_
            True

            >>> profile = Profile(preferences_rk=[
            ...     [0, 2, 1],
            ...     [0, 2, 1],
            ...     [1, 0, 2],
            ...     [1, 0, 2],
            ...     [2, 0, 1],
            ... ])
            >>> rule = RuleSchulze()(profile)
            >>> rule.is_im_c_(2)
            False

            >>> profile = Profile(preferences_rk=[
            ...     [1, 3, 2, 0, 4],
            ...     [2, 3, 0, 1, 4],
            ...     [3, 4, 2, 1, 0],
            ...     [4, 0, 1, 3, 2],
            ...     [4, 2, 0, 1, 3],
            ... ])
            >>> rule = RuleSchulze()(profile)
            >>> rule.is_im_c_(2)
            True

            >>> profile = Profile(preferences_rk=[
            ...     [0, 1, 3, 4, 2],
            ...     [2, 4, 0, 3, 1],
            ...     [2, 4, 1, 3, 0],
            ...     [3, 2, 0, 1, 4],
            ...     [4, 0, 3, 2, 1],
            ... ])
            >>> rule = RuleSchulze()(profile)
            >>> rule.v_im_for_c_
            array([[ 0.,  0.,  0.,  0.,  0.],
                   [ 0.,  0.,  1.,  0.,  1.],
                   [ 0.,  0., nan, nan,  1.],
                   [ 0.,  0.,  1., nan,  0.],
                   [ 0.,  0.,  0.,  0., nan]])
        """
        for c in self.losing_candidates_:
            if not c_is_wanted[c]:
                continue
            if not np.isneginf(self._v_im_for_c[v, c]):
                continue
            self.mylogv("IM: c =", c, 3)
            nb_wanted_undecided -= 1
            # Maybe we will not decide, but we will have done the 'fast' job for ``c`` anyway.

            matrix_duels_s = np.copy(self.profile_.matrix_duels_rk)
            for x in range(self.profile_.n_c):
                for y in range(x + 1, self.profile_.n_c):
                    if self.profile_.preferences_borda_rk[v, x] > self.profile_.preferences_borda_rk[v, y]:
                        matrix_duels_s[x, y] -= 1
                    else:
                        matrix_duels_s[y, x] -= 1

            count_ballot_s = self._count_ballot_aux(matrix_duels_s)
            w_s, widest_path = count_ballot_s['w'], count_ballot_s['scores']
            if w_s == c:
                self.mylog("IM: Manipulation easy (c wins without manipulator's vote)", 3)
                self._v_im_for_c[v, c] = True
                self._candidates_im[c] = True
                self._voters_im[v] = True
                self._is_im = True
                self.mylog("IM found", 3)
                if stop_if_true or nb_wanted_undecided == 0:
                    return
                continue  # pragma: no cover - Not really "executed" because of Python optimization

            the_s = 2 * (widest_path - (self.profile_.n_v - 1) / 2)
            np.fill_diagonal(the_s, 0)
            success_cowinner, success_tb = self._vote_strategically(matrix_duels_s, the_s, c, 1)
            if success_tb:
                self._v_im_for_c[v, c] = True
                self._candidates_im[c] = True
                self._voters_im[v] = True
                self._is_im = True
                self.mylog("IM found", 3)
                if stop_if_true:
                    return
            elif success_cowinner:
                self.mylog("IM not sure", 3)
                if self._im_option != 'exact':
                    self._v_im_for_c[v, c] = np.nan
            else:
                self._v_im_for_c[v, c] = False
            if nb_wanted_undecided == 0:
                return

    def _im_main_work_v_(self, v, c_is_wanted, nb_wanted_undecided, stop_if_true):
        """
            >>> profile = Profile(preferences_rk=[
            ...     [0, 1, 2],
            ...     [0, 2, 1],
            ...     [0, 2, 1],
            ...     [0, 2, 1],
            ...     [2, 0, 1],
            ... ])
            >>> rule = RuleSchulze(im_option='exact')(profile)
            >>> rule.is_im_
            False

            >>> profile = Profile(preferences_rk=[
            ...     [0, 2, 1],
            ...     [0, 2, 1],
            ...     [1, 0, 2],
            ...     [2, 0, 1],
            ...     [2, 1, 0],
            ... ])
            >>> rule = RuleSchulze(im_option='exact')(profile)
            >>> rule.is_im_
            False

            >>> profile = Profile(preferences_rk=[
            ...     [0, 1, 2],
            ...     [1, 0, 2],
            ...     [1, 2, 0],
            ...     [2, 0, 1],
            ...     [2, 0, 1],
            ... ])
            >>> rule = RuleSchulze(im_option='exact')(profile)
            >>> rule.is_im_
            True
        """
        self._im_main_work_v_fast(v, c_is_wanted, nb_wanted_undecided, stop_if_true)
        # Deal with 'exact' (brute force) option
        if self._im_option != 'exact':
            return
        if stop_if_true and np.any(np.equal(self._v_im_for_c[v, c_is_wanted], True)):
            return
        nb_wanted_undecided_updated = np.sum(np.isneginf(self._v_im_for_c[v, c_is_wanted]))
        if nb_wanted_undecided_updated == 0:
            self.mylog("IM: Job finished", 3)
        else:
            self.mylogv("IM: Still some work for v =", v, 3)
            self._im_main_work_v_exact_(v, c_is_wanted, nb_wanted_undecided_updated, stop_if_true)

    # %% Trivial Manipulation (TM)

    # Use the general methods.

    # %% Ignorant-Coalition Manipulation (ICM)

    # Use the general methods.

    # %% Coalition Manipulation (CM)

    def _cm_main_work_c_(self, c, optimize_bounds):
        """
            >>> profile = Profile(preferences_rk=[
            ...     [0, 3, 1, 2],
            ...     [1, 0, 2, 3],
            ...     [2, 1, 3, 0],
            ...     [2, 3, 0, 1],
            ...     [3, 2, 0, 1],
            ... ])
            >>> rule = RuleSchulze()(profile)
            >>> rule.candidates_cm_
            array([ 1.,  1.,  0., nan])

            >>> profile = Profile(preferences_rk=[
            ...     [0, 3, 2, 1],
            ...     [1, 0, 3, 2],
            ...     [2, 1, 0, 3],
            ...     [2, 3, 0, 1],
            ...     [3, 1, 0, 2],
            ... ])
            >>> rule = RuleSchulze()(profile)
            >>> rule.candidates_cm_
            array([ 0.,  1., nan,  1.])

            >>> profile = Profile(preferences_rk=[
            ...     [1, 0, 2, 3, 4],
            ...     [2, 1, 4, 0, 3],
            ...     [2, 3, 4, 0, 1],
            ...     [3, 4, 0, 1, 2],
            ...     [4, 0, 3, 1, 2],
            ...     [4, 0, 3, 1, 2],
            ... ])
            >>> rule = RuleSchulze()(profile)
            >>> rule.candidates_cm_
            array([0., 0., 1., 1., 0.])

            >>> profile = Profile(preferences_rk=[
            ...     [0, 2, 1],
            ...     [0, 2, 1],
            ...     [1, 0, 2],
            ...     [2, 0, 1],
            ...     [2, 1, 0],
            ... ])
            >>> rule = RuleSchulze(cm_option='exact')(profile)
            >>> rule.candidates_cm_
            array([0., 0., 0.])

            >>> profile = Profile(preferences_ut=[
            ...     [ 0. ,  1. ,  1. , -1. ,  0. ],
            ...     [-0.5,  1. ,  0. ,  0.5,  0.5],
            ...     [ 1. , -0.5,  0.5,  1. , -0.5],
            ...     [ 0. ,  0.5,  0.5,  1. , -0.5],
            ...     [-1. , -1. , -0.5,  0.5,  1. ],
            ...     [-0.5, -0.5,  0. ,  0.5,  1. ],
            ... ], preferences_rk=[
            ...     [1, 2, 4, 0, 3],
            ...     [1, 3, 4, 2, 0],
            ...     [3, 0, 2, 4, 1],
            ...     [3, 2, 1, 0, 4],
            ...     [4, 3, 2, 0, 1],
            ...     [4, 3, 2, 1, 0],
            ... ])
            >>> rule = RuleSchulze(cm_option='exact')(profile)
            >>> rule.is_um_c_(0)
            False
            >>> rule.necessary_coalition_size_cm_
            array([2., 2., 2., 0., 2.])
        """
        _ = self._um_variables_are_declared_
        matrix_duels_s = preferences_ut_to_matrix_duels_ut(
            self.profile_.preferences_borda_rk[np.logical_not(self.v_wants_to_help_c_[:, c]), :])
        n_m = self.profile_.matrix_duels_ut[c, self.w_]
        n_s = self.profile_.n_v - n_m
        count_ballot_s = self._count_ballot_aux(matrix_duels_s)
        w, widest_path = count_ballot_s['w'], count_ballot_s['scores']
        the_s = 2 * (widest_path - n_s / 2)
        np.fill_diagonal(the_s, 0)

        # We try the "co-winner" version with ``n_m - 1`` manipulators. If True, it implies that unique winner
        # version is OK for ``n_m`` manipulators.
        success_cowinner, success_tb = self._vote_strategically(matrix_duels_s, the_s, c, n_m - 1)
        if success_tb:
            self._update_sufficient(self._sufficient_coalition_size_cm, c, n_m - 1,
                                    'CM: Update sufficient_coalition_size_cm = n_m - 1 =')
            self._candidates_um[c] = True
            self._is_um = True
            return
        if success_cowinner:
            self._update_sufficient(self._sufficient_coalition_size_cm, c, n_m,
                                    'CM: Update sufficient_coalition_size_cm = n_m =')
            self._update_necessary(self._necessary_coalition_size_cm, c, n_m - 1,
                                   'CM: Update necessary_coalition_size_cm = n_m - 1 =')
            # We do not know for UM, but we do not really care
            return
        else:
            self._update_necessary(self._necessary_coalition_size_cm, c, n_m,
                                   'CM: Update necessary_coalition_size_cm = n_m =')

        # We try with n_m manipulators.
        success_cowinner, success_tb = self._vote_strategically(matrix_duels_s, the_s, c, n_m)
        self._um_fast_tested[c] = True
        if success_tb:
            self._update_sufficient(self._sufficient_coalition_size_cm, c, n_m,
                                    'CM: Update sufficient_coalition_size_cm = n_m =')
            self._candidates_um[c] = True
            self._is_um = True
            return
        if success_cowinner:
            self._update_sufficient(self._sufficient_coalition_size_cm, c, n_m + 1,
                                    'CM: Update sufficient_coalition_size_cm = n_m + 1 =')
            self._update_necessary(self._necessary_coalition_size_cm, c, n_m,
                                   'CM: Update necessary_coalition_size_cm = n_m =')
            if self.cm_option == 'exact':
                self._um_main_work_c_exact_rankings_(c)
                if equal_true(self._candidates_um[c]):
                    self._update_sufficient(self._sufficient_coalition_size_cm, c, n_m,
                                            'CM: Update sufficient_coalition_size_cm = n_m =')
                else:
                    self._cm_main_work_c_exact_(c, optimize_bounds)
        else:
            self._update_necessary(self._necessary_coalition_size_cm, c, n_m + 1,
                                   'CM: Update necessary_coalition_size_cm = n_m + 1 =')
            self._candidates_um[c] = False
            return

    # %% Unison manipulation (UM)

    @cached_property
    def _um_variables_are_declared_(self):
        self._um_was_computed_with_candidates = False
        # Exceptionally, we initialize the variables for UM here, because UM is in great part managed in CM.
        self._is_um = - np.inf
        self._candidates_um = np.full(self.profile_.n_c, -np.inf)
        self._candidates_um[self.w_] = False
        # ``_um_fast_tested[c]`` will be True iff if we have already launched ``_vote_strategically`` with ``n_m``
        # manipulators (for ``c``).
        self._um_fast_tested = np.zeros(self.profile_.n_c, dtype=np.bool)
        return True

    @cached_property
    def _um_is_initialized_general_(self):
        self.mylog("UM: Initialize", 2)
        _ = self._um_variables_are_declared_
        self._um_preliminary_checks_general_()
        return True

    def _um_preliminary_checks_general_subclass_(self):
        if not np.any(np.isneginf(self._candidates_um)):
            return
        _ = self.is_cm_
        self._candidates_um[np.equal(self._candidates_cm, False)] = False

    def _um_main_work_c_(self, c):
        """
            >>> profile = Profile(preferences_rk=[
            ...     [0, 1, 2],
            ...     [1, 0, 2],
            ...     [1, 2, 0],
            ...     [2, 0, 1],
            ...     [2, 0, 1],
            ... ])
            >>> rule = RuleSchulze()(profile)
            >>> rule.candidates_um_
            array([0., 1., 1.])

            >>> profile = Profile(preferences_ut=[
            ...     [ 1. ,  1. ,  0. ,  0.5, -0.5],
            ...     [ 0. , -0.5,  1. , -0.5,  0. ],
            ...     [ 0. ,  0. ,  1. ,  1. ,  1. ],
            ...     [ 0.5, -0.5, -0.5, -0.5,  0.5],
            ...     [ 0.5, -0.5, -1. ,  0.5,  1. ],
            ...     [-1. ,  0.5, -1. ,  0.5,  1. ],
            ... ], preferences_rk=[
            ...     [1, 0, 3, 2, 4],
            ...     [2, 4, 0, 1, 3],
            ...     [3, 2, 4, 1, 0],
            ...     [4, 0, 2, 1, 3],
            ...     [4, 0, 3, 1, 2],
            ...     [4, 1, 3, 0, 2],
            ... ])
            >>> rule = RuleSchulze()(profile)
            >>> rule.candidates_um_
            array([0., 0., 1., 0., 0.])

            >>> profile = Profile(preferences_rk=[
            ...     [0, 2, 1],
            ...     [0, 2, 1],
            ...     [1, 0, 2],
            ...     [1, 0, 2],
            ...     [2, 0, 1],
            ... ])
            >>> rule = RuleSchulze()(profile)
            >>> rule.candidates_cm_
            array([0., 0., 0.])
            >>> rule.candidates_um_
            array([0., 0., 0.])

            >>> profile = Profile(preferences_ut=[
            ...     [ 0. ,  0.5,  0. , -1. ],
            ...     [ 0. , -0.5,  0.5, -0.5],
            ... ], preferences_rk=[
            ...     [1, 2, 0, 3],
            ...     [2, 0, 3, 1],
            ... ])
            >>> rule = RuleSchulze(um_option='exact', cm_option='exact')(profile)
            >>> rule.candidates_um_
            array([1., 0., 0., 0.])
        """
        n_m = self.profile_.matrix_duels_ut[c, self.w_]
        if self._sufficient_coalition_size_cm[c] + 1 <= n_m:  # pragma: no cover
            # TO DO: Investigate whether this case can actually happen.
            self._reached_uncovered_code()
            self._candidates_um[c] = True
            return
        if self._necessary_coalition_size_cm[c] - 1 > n_m:  # pragma: no cover
            # TO DO: Investigate whether this case can actually happen.
            self._reached_uncovered_code()
            self._candidates_um[c] = False
            return
        if not self._um_fast_tested[c]:
            # We must try the fast algo first.
            matrix_duels_s = preferences_ut_to_matrix_duels_ut(
                self.profile_.preferences_borda_rk[np.logical_not(self.v_wants_to_help_c_[:, c]), :])
            n_s = self.profile_.n_v - n_m
            count_ballot_s = self._count_ballot_aux(matrix_duels_s)
            w, widest_path = count_ballot_s['w'], count_ballot_s['scores']
            the_s = 2 * (widest_path - n_s / 2)
            np.fill_diagonal(the_s, 0)
            success_cowinner, success_tb = self._vote_strategically(matrix_duels_s, the_s, c, n_m)
            self._um_fast_tested[c] = True
            if success_tb:
                self._candidates_um[c] = True
                return
            if not success_cowinner:
                self._candidates_um[c] = False
                return
        if self.um_option == 'exact':
            self._um_main_work_c_exact_rankings_(c)
        else:
            self._candidates_um[c] = np.nan
