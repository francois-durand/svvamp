# -*- coding: utf-8 -*-
"""
Created on 12 sep. 2026
Copyright François Durand 2014-2026
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

from svvamp.preferences.profile import Profile
from svvamp.rules.rule import Rule
from svvamp.utils.misc import preferences_ut_to_matrix_duels_ut
from svvamp.utils.prevent_condorcet_winner import prevent_condorcet_winner
from svvamp.utils.pseudo_bool import equal_false, equal_true
from svvamp.utils.util_cache import cached_property


class RuleCondorcetDuel(Rule):
    """Condorcet-duel rule.

    Options
    -------
        >>> RuleCondorcetDuel.print_options_parameters()
        cm_option: ['fast', 'exact']. Default: 'fast'.
        icm_option: ['exact']. Default: 'exact'.
        iia_subset_maximum_size: is_number. Default: 2.
        im_option: ['lazy', 'exact']. Default: 'lazy'.
        precheck_heuristic: is_bool. Default: True.
        tm_option: ['exact']. Default: 'exact'.
        um_option: ['exact']. Default: 'exact'.

    Notes
    -----
    Each voter must provide a strict total order. If there is a Condorcet winner (in the sense of
    :attr:`matrix_victories_rk`), then she is elected. Otherwise, the winner of the duel between candidates 0 and 1
    is elected (in the sense of :attr:`matrix_duels_rk`), with candidate 0 winning in case of a tie.

    This rule is a very simple Condorcet-consistent rule: it meets the Condorcet criterion but is not neutral.

    * :meth:`is_cm_`:

        * :attr:`cm_option` = ``'fast'``: Polynomial algorithm. It is exact, except when the candidates whom the
          manipulators must prevent from being a Condorcet winner cannot be handled one after the other (cf.
          :func:`~svvamp.utils.prevent_condorcet_winner.prevent_condorcet_winner`). In that case, it may return
          ``numpy.nan``.
        * :attr:`cm_option` = ``'exact'``: Exact algorithm. It is polynomial in the number of voters, but its cost
          may be exponential in the number of candidates (in practice, it is much faster than the exhaustive
          algorithm of the superclass :class:`Rule`).

    * :meth:`is_icm_`: Exact in polynomial time.
    * :meth:`is_im_`: Non-polynomial or non-exact algorithms from superclass :class:`Rule`.
    * :meth:`is_iia`: Non-polynomial or non-exact algorithms from superclass :class:`Rule`. If
      :attr:`iia_subset_maximum_size` = 2, it runs in polynomial time and is exact up to ties (which can occur only if
      :attr:`n_v` is even).
    * :meth:`is_tm_`: Exact in polynomial time.
    * :meth:`is_um_`: Exact in polynomial time.

    References
    ----------
    'Limit CM rate of classical voting rules', François Durand et al., 2026.

    See Also
    --------
    :class:`RuleCondorcetConstant`, :class:`RuleCondorcetDictatorship`, :class:`RuleCondorcetVtbIRV`.

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
        >>> rule = RuleCondorcetDuel()(profile)
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
        [[1. 0. 0.]
         [3. 2. 0.]]
        candidates_by_scores_best_to_worst
        [0 1 2]
        scores_best_to_worst
        [[1. 0. 0.]
         [3. 2. 0.]]
        w = 0
        score_w = [1. 3.]
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
         ||     Condorcet_c_rk_ctb (False) ==> Condorcet_c_rk (True)      ||
         ||           ||               ||       ||             ||         ||
         V            V                ||       ||             V          V
        Condorcet_c_ut_abs_ctb (False)     ==>     Condorcet_ut_abs_c (True)
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
        [0. 2. 4.]
        sufficient_coalition_size_cm =
        [0. 2. 4.]
    """

    full_name = "Condorcet-duel"
    abbreviation = "CDu"

    options_parameters = Rule.options_parameters.copy()
    options_parameters.update(
        {
            "cm_option": {"allowed": ["fast", "exact"], "default": "fast"},
            "tm_option": {"allowed": ["exact"], "default": "exact"},
            "icm_option": {"allowed": ["exact"], "default": "exact"},
            "um_option": {"allowed": ["exact"], "default": "exact"},
        }
    )

    def __init__(self, **kwargs):
        super().__init__(
            with_two_candidates_reduces_to_plurality=True,
            is_based_on_rk=True,
            precheck_icm=False,
            log_identity="CONDORCET_DUEL",
            **kwargs,
        )

    # %% Counting the ballots

    @cached_property
    def w_(self):
        self.mylog("Compute w", 1)
        if self.profile_.exists_condorcet_winner_rk:
            return self.profile_.condorcet_winner_rk
        elif self.profile_.matrix_duels_rk[0, 1] >= self.profile_.matrix_duels_rk[1, 0]:
            return 0
        else:
            return 1

    @cached_property
    def scores_(self):
        """2d array.

            * ``scores[0, c]`` is 1 if ``c`` is the Condorcet winner (in the sense of :attr:`matrix_victories_rk`),
              0 otherwise.
            * ``scores[1, c]`` is the score of ``c`` in the fallback rule: for candidate 0 (resp. 1), the number of
              voters who rank her before candidate 1 (resp. 0). For other candidates, 0.

        Examples
        --------
            >>> profile = Profile(preferences_rk=[[0, 1, 2], [1, 2, 0], [2, 0, 1]])
            >>> rule = RuleCondorcetDuel()(profile)
            >>> rule.scores_
            array([[0., 0., 0.],
                   [2., 1., 0.]])
        """
        self.mylog("Compute scores", 1)
        scores_condorcet = np.zeros(self.profile_.n_c)
        if self.profile_.exists_condorcet_winner_rk:
            scores_condorcet[self.profile_.condorcet_winner_rk] = 1
        scores_fallback = np.zeros(self.profile_.n_c)
        scores_fallback[0] = self.profile_.matrix_duels_rk[0, 1]
        scores_fallback[1] = self.profile_.matrix_duels_rk[1, 0]
        return np.array([scores_condorcet, scores_fallback])

    @cached_property
    def candidates_by_scores_best_to_worst_(self):
        """1d array of integers. Candidates are sorted lexicographically by :attr:`scores_`, i.e. the Condorcet winner
        first if she exists, then the winner of the duel between 0 and 1, then the loser of this duel, then the other
        candidates by increasing index.

        Examples
        --------
            >>> profile = Profile(preferences_rk=[[0, 1, 2], [1, 2, 0], [2, 0, 1]])
            >>> rule = RuleCondorcetDuel()(profile)
            >>> rule.candidates_by_scores_best_to_worst_
            array([0, 1, 2])

            >>> profile = Profile(preferences_rk=[[2, 1, 0], [1, 2, 0], [2, 0, 1]])
            >>> rule = RuleCondorcetDuel()(profile)
            >>> rule.candidates_by_scores_best_to_worst_
            array([2, 1, 0])
        """
        self.mylog("Compute candidates_by_scores_best_to_worst", 1)
        return np.array(sorted(range(self.profile_.n_c), key=lambda c: list(self.scores_[:, c]), reverse=True))

    # %% Manipulation criteria of the voting system

    @cached_property
    def meets_majority_favorite_c_rk_ctb(self):
        return True

    @cached_property
    def meets_condorcet_c_rk(self):
        return True

    # %% Individual manipulation (IM)

    # Use the general methods from class Rule.

    # %% Trivial Manipulation (TM)

    # Use the general methods from class Rule.

    # %% Unison manipulation (UM)

    def _um_main_work_c_exact_rankings_(self, c):
        """Do the main work in UM loop for candidate ``c``, with option 'exact'.

        Since all manipulators rank ``c`` first, ``c`` wins iff she becomes a Condorcet winner, or if there is no
        Condorcet winner and ``c`` wins the fallback rule. With identical ballots, the latter is decided exactly and
        in polynomial time by :func:`~svvamp.utils.prevent_condorcet_winner.prevent_condorcet_winner`.

        Examples
        --------
        Candidate 2 is the Condorcet winner. The two supporters of candidate 1 can make her win, even with the same
        ballot: by ranking 1 first, they make her a Condorcet winner. The two supporters of candidate 0 cannot make
        her win, because she loses the duel against candidate 1 anyway.

            >>> profile = Profile(preferences_rk=[
            ...     [0, 2, 1],
            ...     [0, 2, 1],
            ...     [1, 2, 0],
            ...     [1, 2, 0],
            ...     [2, 1, 0],
            ...     [2, 1, 0],
            ... ])
            >>> rule = RuleCondorcetDuel()(profile)
            >>> rule.w_
            2
            >>> rule.candidates_cm_
            array([0., 1., 0.])
            >>> rule.candidates_um_
            array([0., 1., 0.])
        """
        n_m = self.profile_.matrix_duels_ut[c, self.w_]
        n_s = self.profile_.n_v - n_m
        d_neq_c = np.array(range(self.profile_.n_c)) != c
        preferences_borda_s = self.profile_.preferences_borda_rk[np.logical_not(self.v_wants_to_help_c_[:, c]), :]
        matrix_duels_s = preferences_ut_to_matrix_duels_ut(preferences_borda_s)
        n_manip_becomes_cond = int(np.maximum(n_s + 1 - 2 * np.min(matrix_duels_s[c, d_neq_c]), 0))
        if n_m >= n_manip_becomes_cond:
            self.mylog("UM: c becomes a Condorcet winner", 3)
            self._candidates_um[c] = True
            return
        if not self._cm_fallback_elects_c_(c, n_m, matrix_duels_s):
            self.mylog("UM: c cannot win the fallback rule", 3)
            self._candidates_um[c] = False
            return
        result, ballots_m = prevent_condorcet_winner(matrix_duels_s, c, n_m, unison=True)
        self.mylogv("UM: prevent_condorcet_winner (unison) =", result, 3)
        if result and not self._cm_check_ballots_(c, ballots_m):  # pragma: no cover
            raise AssertionError("Uh-oh!")
        self._candidates_um[c] = result

    # %% Ignorant-Coalition Manipulation (ICM)

    # Use the general methods from class Rule. Since the rule meets IgnMC_c_ctb, they are exact.

    # %% Coalition Manipulation (CM)

    def _cm_fallback_can_elect_c_(self, c):
        """Whether ``c`` can win the fallback rule with some number of manipulators.

        Parameters
        ----------
        c : int
            Candidate for which we want to manipulate.

        Returns
        -------
        bool
            True iff ``c`` is elected by the fallback rule (for some number of manipulators). If not, then ``c`` can
            only win by becoming a Condorcet winner.
        """
        return c in {0, 1}

    def _cm_fallback_elects_c_(self, c, n_m, matrix_duels_s):
        """Whether ``c`` wins the fallback rule, assuming there is no Condorcet winner.

        Parameters
        ----------
        c : int
            Candidate for which we want to manipulate.
        n_m : int
            Number of manipulators (who all rank ``c`` first).
        matrix_duels_s : ndarray
            Matrix of duels of the sincere voters.

        Returns
        -------
        bool
            True iff ``c`` is elected by the fallback rule.
        """
        if c == 0:
            return matrix_duels_s[0, 1] + n_m >= matrix_duels_s[1, 0]
        elif c == 1:
            return matrix_duels_s[1, 0] + n_m > matrix_duels_s[0, 1]
        else:
            return False

    def _cm_decide_(self, c, n_m, matrix_duels_s, n_manip_becomes_cond):
        """Decide whether ``n_m`` manipulators can make ``c`` win.

        Parameters
        ----------
        c : int
            Candidate for which we want to manipulate.
        n_m : int
            Number of manipulators.
        matrix_duels_s : ndarray
            Matrix of duels of the sincere voters.
        n_manip_becomes_cond : int
            Number of manipulators needed to make ``c`` a Condorcet winner.

        Returns
        -------
        result : bool or nan
            True iff the manipulation is possible. ``nan`` if the algorithm cannot decide.
        ballots : ndarray or None
            If ``result`` is True, ``ballots[i, :]`` is the ranking cast by the ``i``-th manipulator.
        """
        if n_m >= n_manip_becomes_cond:
            self.mylog("CM: c becomes a Condorcet winner", 3)
            ballot = [c] + [d for d in range(self.profile_.n_c) if d != c]
            return True, np.tile(ballot, (n_m, 1))
        if not self._cm_fallback_elects_c_(c, n_m, matrix_duels_s):
            self.mylog("CM: c cannot win the fallback rule", 3)
            return False, None
        result, ballots = prevent_condorcet_winner(matrix_duels_s, c, n_m, exact=(self.cm_option == "exact"))
        self.mylogv("CM: prevent_condorcet_winner =", result, 3)
        return result, ballots

    def _cm_check_ballots_(self, c, ballots_m):
        """Check that the manipulation works.

        Parameters
        ----------
        c : int
            Candidate for which we want to manipulate.
        ballots_m : ndarray
            ``ballots_m[i, :]`` is the ranking cast by the ``i``-th manipulator.

        Returns
        -------
        bool
            True iff ``c`` wins when the manipulators cast these ballots (and other voters are sincere). The voters
            keep their positions in the profile.
        """
        preferences_rk_test = np.copy(self.profile_.preferences_rk)
        preferences_rk_test[self.v_wants_to_help_c_[:, c], :] = ballots_m
        winner_test = self._copy(profile=Profile(preferences_rk=preferences_rk_test, sort_voters=False)).w_
        return winner_test == c

    def _cm_main_work_c_(self, c, optimize_bounds):
        """
        >>> profile = Profile(preferences_rk=[
        ...     [0, 2, 1],
        ...     [1, 0, 2],
        ...     [2, 0, 1],
        ...     [2, 0, 1],
        ...     [0, 1, 2],
        ...     [1, 2, 0],
        ...     [2, 1, 0],
        ... ])
        >>> rule = RuleCondorcetDuel()(profile)
        >>> rule.w_
        2

        Candidates 0 and 1 can both win: their supporters can prevent 2 from being a Condorcet winner, and they
        can win the duel between 0 and 1.

        >>> rule.candidates_cm_
        array([1., 1., 0.])

        A case where the polynomial algorithm cannot decide, but the exact one can:

        >>> profile = Profile(preferences_rk=[
        ...     [3, 2, 1, 0],
        ...     [1, 3, 0, 2],
        ...     [2, 3, 1, 0],
        ... ])
        >>> rule = RuleCondorcetDuel(cm_option='fast')(profile)
        >>> rule.candidates_cm_
        array([ 0., nan,  0.,  0.])
        >>> rule = RuleCondorcetDuel(cm_option='exact')(profile)
        >>> rule.candidates_cm_
        array([0., 0., 0., 0.])

        A case where candidate 0 wins by the fallback rule, whereas the other candidates would need to become
        Condorcet winners (except candidate 1, who could also win the fallback rule with 3 manipulators):

        >>> profile = Profile(preferences_rk=[
        ...     [3, 2, 1, 0],
        ...     [0, 2, 1, 3],
        ...     [3, 2, 0, 1],
        ...     [0, 2, 1, 3],
        ...     [1, 2, 3, 0],
        ... ])
        >>> rule = RuleCondorcetDuel()(profile)
        >>> rule.w_
        2
        >>> rule.candidates_cm_
        array([1., 0., 0., 0.])
        >>> rule.necessary_coalition_size_cm_
        array([1., 3., 0., 4.])
        >>> rule.sufficient_coalition_size_cm_
        array([1., 3., 0., 4.])
        """
        n_m = self.profile_.matrix_duels_ut[c, self.w_]
        n_s = self.profile_.n_v - n_m
        candidates = np.array(range(self.profile_.n_c))
        preferences_borda_s = self.profile_.preferences_borda_rk[np.logical_not(self.v_wants_to_help_c_[:, c]), :]
        matrix_duels_s = preferences_ut_to_matrix_duels_ut(preferences_borda_s)
        self.mylogm("CM: matrix_duels_s =", matrix_duels_s, 3)
        d_neq_c = candidates != c
        # Sufficient condition: ``c`` becomes a Condorcet winner. Necessary and sufficient if ``c`` cannot win the
        # fallback rule.
        n_manip_becomes_cond = int(np.maximum(n_s + 1 - 2 * np.min(matrix_duels_s[c, d_neq_c]), 0))
        self.mylogv("CM: n_manip_becomes_cond =", n_manip_becomes_cond, 3)
        self._update_sufficient(
            self._sufficient_coalition_size_cm,
            c,
            n_manip_becomes_cond,
            "CM: Update sufficient_coalition_size_cm[c] = n_manip_becomes_cond =",
        )
        if not optimize_bounds and n_m >= self._sufficient_coalition_size_cm[c]:
            return True
        # Necessary condition: prevent each other candidate ``d`` from being a Condorcet winner. For that, we need
        # ``matrix_duels_s[d, e] <= (n_s + n_m) / 2`` for some ``e``, i.e. ``n_m >= 2 * min_e(matrix_duels_s[d, e])
        # - n_s``.
        n_manip_prevent_cond = 0
        for d in candidates[d_neq_c]:
            e_neq_d = candidates != d
            n_prevent_d = np.maximum(2 * np.min(matrix_duels_s[d, e_neq_d]) - n_s, 0)
            n_manip_prevent_cond = max(n_manip_prevent_cond, n_prevent_d)
        self.mylogv("CM: n_manip_prevent_cond =", n_manip_prevent_cond, 3)
        self._update_necessary(
            self._necessary_coalition_size_cm,
            c,
            min(n_manip_becomes_cond, n_manip_prevent_cond),
            "CM: Update necessary_coalition_size_cm[c] = min(n_manip_becomes_cond, n_manip_prevent_cond) =",
        )
        if not self._cm_fallback_can_elect_c_(c):
            # ``c`` can only win by becoming a Condorcet winner.
            self._update_necessary(
                self._necessary_coalition_size_cm,
                c,
                n_manip_becomes_cond,
                "CM: Update necessary_coalition_size_cm[c] = n_manip_becomes_cond =",
            )
        if not optimize_bounds and self._necessary_coalition_size_cm[c] > n_m:
            return True
        # Decide with the actual number of manipulators
        result, ballots_m = self._cm_decide_(c, n_m, matrix_duels_s, n_manip_becomes_cond)
        if equal_true(result):
            if not self._cm_check_ballots_(c, ballots_m):  # pragma: no cover
                raise AssertionError("Uh-oh!")
            self._update_sufficient(
                self._sufficient_coalition_size_cm, c, n_m, "CM: Update sufficient_coalition_size_cm[c] = n_m ="
            )
        elif equal_false(result):
            self._update_necessary(
                self._necessary_coalition_size_cm, c, n_m + 1, "CM: Update necessary_coalition_size_cm[c] = n_m + 1 ="
            )
        if not optimize_bounds:
            # Quick escape if the bounds are not tight: we could do better with ``optimize_bounds``.
            return self._necessary_coalition_size_cm[c] < self._sufficient_coalition_size_cm[c]
        # Optimize the bounds: scan the possible numbers of manipulators, from the necessary size upwards.
        n_m_test = int(self._necessary_coalition_size_cm[c])
        while n_m_test < self._sufficient_coalition_size_cm[c]:
            result, ballots_m = self._cm_decide_(c, n_m_test, matrix_duels_s, n_manip_becomes_cond)
            if equal_true(result):
                self._update_sufficient(
                    self._sufficient_coalition_size_cm,
                    c,
                    n_m_test,
                    "CM: Update sufficient_coalition_size_cm[c] =",
                )
                break
            elif equal_false(result):
                self._update_necessary(
                    self._necessary_coalition_size_cm,
                    c,
                    n_m_test + 1,
                    "CM: Update necessary_coalition_size_cm[c] =",
                )
            n_m_test += 1
        return False

    @cached_property
    def theta_critical_(self):
        """
        >>> profile = Profile(preferences_rk=[[0, 1, 2, 3]])
        >>> rule = RuleCondorcetDuel()(profile)
        >>> rule.theta_critical_
        0
        """
        return 0
