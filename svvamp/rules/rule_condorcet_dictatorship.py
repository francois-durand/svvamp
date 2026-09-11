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

import itertools

import numpy as np

from svvamp.preferences.profile import Profile
from svvamp.rules.rule import Rule
from svvamp.utils.misc import preferences_ut_to_matrix_duels_ut
from svvamp.utils.prevent_condorcet_winner import prevent_condorcet_winner
from svvamp.utils.pseudo_bool import equal_false, equal_true
from svvamp.utils.util_cache import cached_property


class RuleCondorcetDictatorship(Rule):
    """Condorcet-dictatorship rule.

    Options
    -------
        >>> RuleCondorcetDictatorship.print_options_parameters()
        cm_option: ['fast', 'exact']. Default: 'fast'.
        icm_option: ['exact']. Default: 'exact'.
        iia_subset_maximum_size: is_number. Default: 2.
        im_option: ['lazy', 'exact']. Default: 'lazy'.
        precheck_heuristic: is_bool. Default: True.
        tm_option: ['exact']. Default: 'exact'.
        um_option: ['lazy', 'exact']. Default: 'lazy'.

    Notes
    -----
    Each voter must provide a strict total order. If there is a Condorcet winner (in the sense of
    :attr:`matrix_victories_rk`), then she is elected. Otherwise, the candidate ranked first by voter 0 (the
    *dictator*) is elected.

    This rule is a minimal Condorcet-consistent rule: it meets the Condorcet criterion but is not anonymous. As a
    consequence, it does not meet the criteria "with candidate tie-breaking" (e.g. if half of the voters rank
    candidate 0 first, she does not necessarily win). Contrary to the other rules of SVVAMP, the position of each
    voter in the profile matters. In particular, for the manipulation problems where a coalition of a given size is
    considered (cf. :attr:`necessary_coalition_size_cm_`, :attr:`sufficient_coalition_size_cm_` and their
    counterparts for ICM), the following convention is used: if voter 0 does not prefer ``c`` to the sincere winner
    ``w`` (in the sense of utilities), she is a sincere voter and never joins the coalition; if she prefers ``c`` to
    ``w``, she is considered as the first manipulator, i.e. she belongs to the coalition as soon as it is not empty.

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
    * :meth:`is_um_`: Non-polynomial or non-exact algorithms from superclass :class:`Rule`.

    References
    ----------
    'Limit CM rate of classical voting rules', François Durand et al., 2026.

    See Also
    --------
    :class:`RuleCondorcetConstant`, :class:`RuleCondorcetDuel`, :class:`RuleCondorcetVtbIRV`.

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
        >>> rule = RuleCondorcetDictatorship()(profile)
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
         [2. 1. 0.]]
        candidates_by_scores_best_to_worst
        [0 1 2]
        scores_best_to_worst
        [[1. 0. 0.]
         [2. 1. 0.]]
        w = 0
        score_w = [1. 2.]
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
        with_two_candidates_reduces_to_plurality =  False
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
         ||       maj_fav_c_rk_ctb (False) ==> maj_fav_c_rk (True)        ||
         ||           ||                                       ||         ||
         V            V                                        V          V
        majority_favorite_c_ut_ctb (False) ==> majority_favorite_c_ut (True)
         ||                                                               ||
         V                                                                V
        IgnMC_c_ctb (False)                ==>                IgnMC_c (True)
         ||                                                               ||
         V                                                                V
        InfMC_c_ctb (False)                ==>                InfMC_c (True)
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
        is_cm = False
        log_cm: cm_option = fast, um_option = lazy, tm_option = exact
        candidates_cm =
        [0. 0. 0.]
        necessary_coalition_size_cm =
        [0. 2. 4.]
        sufficient_coalition_size_cm =
        [0. 2. 4.]
    """

    full_name = "Condorcet-dictatorship"
    abbreviation = "CDi"

    options_parameters = Rule.options_parameters.copy()
    options_parameters.update(
        {
            "cm_option": {"allowed": ["fast", "exact"], "default": "fast"},
            "tm_option": {"allowed": ["exact"], "default": "exact"},
            "icm_option": {"allowed": ["exact"], "default": "exact"},
        }
    )

    def __init__(self, **kwargs):
        super().__init__(
            with_two_candidates_reduces_to_plurality=False,
            is_based_on_rk=True,
            precheck_icm=False,
            log_identity="CONDORCET_DICTATORSHIP",
            **kwargs,
        )

    # %% Counting the ballots

    @cached_property
    def w_(self):
        self.mylog("Compute w", 1)
        if self.profile_.exists_condorcet_winner_rk:
            return self.profile_.condorcet_winner_rk
        else:
            return int(self.profile_.preferences_rk[0, 0])

    @cached_property
    def scores_(self):
        """2d array.

            * ``scores[0, c]`` is 1 if ``c`` is the Condorcet winner (in the sense of :attr:`matrix_victories_rk`),
              0 otherwise.
            * ``scores[1, c]`` is the score of ``c`` in the fallback rule, i.e. the Borda score of ``c`` in the
              ranking of voter 0 (cf. :attr:`preferences_borda_rk`).

        Examples
        --------
            >>> profile = Profile(preferences_rk=[[1, 0, 2], [0, 2, 1], [2, 1, 0]])
            >>> rule = RuleCondorcetDictatorship()(profile)
            >>> rule.scores_
            array([[0., 0., 0.],
                   [1., 2., 0.]])
        """
        self.mylog("Compute scores", 1)
        scores_condorcet = np.zeros(self.profile_.n_c)
        if self.profile_.exists_condorcet_winner_rk:
            scores_condorcet[self.profile_.condorcet_winner_rk] = 1
        scores_fallback = np.array(self.profile_.preferences_borda_rk[0, :], dtype=float)
        return np.array([scores_condorcet, scores_fallback])

    @cached_property
    def candidates_by_scores_best_to_worst_(self):
        """1d array of integers. Candidates are sorted lexicographically by :attr:`scores_`, i.e. the Condorcet winner
        first if she exists, then the other candidates in the order of voter 0's ranking.

        Examples
        --------
            >>> profile = Profile(preferences_rk=[[1, 0, 2], [0, 2, 1], [2, 1, 0]])
            >>> rule = RuleCondorcetDictatorship()(profile)
            >>> rule.candidates_by_scores_best_to_worst_
            array([1, 0, 2])

            >>> profile = Profile(preferences_rk=[[2, 1, 0], [1, 2, 0], [2, 0, 1]])
            >>> rule = RuleCondorcetDictatorship()(profile)
            >>> rule.candidates_by_scores_best_to_worst_
            array([2, 1, 0])
        """
        self.mylog("Compute candidates_by_scores_best_to_worst", 1)
        return np.array(sorted(range(self.profile_.n_c), key=lambda c: list(self.scores_[:, c]), reverse=True))

    # %% Manipulation criteria of the voting system

    @cached_property
    def meets_condorcet_c_rk(self):
        return True

    # %% Individual manipulation (IM)

    # Use the general methods from class Rule (the ballot of the manipulator is modified in place, so the positions of
    # the voters are preserved).

    # %% Trivial Manipulation (TM)

    # Use the general methods from class Rule (the ballots of the manipulators are modified in place, so the positions
    # of the voters are preserved).

    # %% Unison manipulation (UM)

    def _um_main_work_c_exact_rankings_(self, c):
        """Do the main work in UM loop for candidate ``c``, with option 'exact'.

        Contrary to the general method from class :class:`Rule`, the ballots of the manipulators are modified in
        place, so the positions of the voters are preserved.

        Examples
        --------
        Voter 0 prefers 1 to the sincere winner 0, so she is a manipulator. With her accomplice, they can prevent 0
        from being a Condorcet winner by ranking 2 before 0, and then voter 0 (the dictator) elects candidate 1:

            >>> profile = Profile(preferences_rk=[
            ...     [1, 0, 2],
            ...     [1, 0, 2],
            ...     [0, 1, 2],
            ...     [0, 2, 1],
            ...     [2, 0, 1],
            ... ])
            >>> rule = RuleCondorcetDictatorship(um_option='exact')(profile)
            >>> rule.w_
            0
            >>> rule.candidates_um_
            array([0., 1., 0.])

        The same profile, except that the dictator is now a sincere voter: candidate 1 cannot be elected.

            >>> profile = Profile(preferences_rk=[
            ...     [0, 1, 2],
            ...     [1, 0, 2],
            ...     [1, 0, 2],
            ...     [0, 2, 1],
            ...     [2, 0, 1],
            ... ])
            >>> rule = RuleCondorcetDictatorship(um_option='exact')(profile)
            >>> rule.candidates_um_
            array([0., 0., 0.])
        """
        is_manipulator = self.v_wants_to_help_c_[:, c]
        base_ballot = [c, *[i for i in range(self.profile_.n_c) if i != c]]  # Put c first for the first try...
        for ballot in itertools.permutations(base_ballot):
            self.mylogv("UM: Ballot =", ballot, 3)
            preferences_rk_test = np.copy(self.profile_.preferences_rk)
            preferences_rk_test[is_manipulator, :] = ballot
            w_test = self._copy(profile=Profile(preferences_rk=preferences_rk_test, sort_voters=False)).w_
            self.mylogv("UM: w_test =", w_test, 3)
            if w_test == c:
                self._candidates_um[c] = True
                return
        else:
            self._candidates_um[c] = False

    # %% Ignorant-Coalition Manipulation (ICM)

    def _icm_preliminary_checks_c_subclass_(self, c, optimize_bounds):
        """ICM: preliminary checks for challenger ``c``.

        If voter 0 prefers ``c`` to ``w``, she is the first manipulator (cf. the convention in the class docstring).
        Then a coalition of ``n_s`` manipulators is sufficient: no other candidate can beat ``c``, so either ``c`` is
        a Condorcet winner, or there is no Condorcet winner and the dictator elects ``c``. Otherwise, the dictator is
        a sincere voter, whose ballot is unknown to the (ignorant) manipulators, so they need a strict majority to
        make ``c`` a Condorcet winner.

        Examples
        --------
        Voter 0 prefers 1 to the sincere winner 0, so a coalition of half of the voters (3 out of 6, hence 3
        manipulators added to the 3 sincere voters) would be enough for candidate 1:

            >>> profile = Profile(preferences_rk=[
            ...     [1, 0, 2],
            ...     [1, 0, 2],
            ...     [0, 1, 2],
            ...     [0, 1, 2],
            ...     [0, 2, 1],
            ... ])
            >>> rule = RuleCondorcetDictatorship()(profile)
            >>> rule.w_
            0
            >>> rule.candidates_icm_
            array([0., 0., 0.])
            >>> rule.necessary_coalition_size_icm_
            array([0., 3., 6.])
            >>> rule.sufficient_coalition_size_icm_
            array([0., 3., 6.])

        The same profile, except that voters 0 and 1 are swapped: the dictator is now a sincere voter, so a strict
        majority would be needed for candidate 1.

            >>> profile = Profile(preferences_rk=[
            ...     [0, 1, 2],
            ...     [1, 0, 2],
            ...     [1, 0, 2],
            ...     [0, 1, 2],
            ...     [0, 2, 1],
            ... ])
            >>> rule = RuleCondorcetDictatorship()(profile)
            >>> rule.candidates_icm_
            array([0., 0., 0.])
            >>> rule.necessary_coalition_size_icm_
            array([0., 4., 6.])
            >>> rule.sufficient_coalition_size_icm_
            array([0., 4., 6.])
        """
        n_m = self.profile_.matrix_duels_ut[c, self.w_]
        n_s = self.profile_.n_v - n_m
        if self.v_wants_to_help_c_[0, c]:
            self._update_sufficient(
                self._sufficient_coalition_size_icm,
                c,
                n_s,
                "ICM: Voter 0 is a manipulator => sufficient_coalition_size_icm[c] = n_s =",
            )
        else:
            self._update_necessary(
                self._necessary_coalition_size_icm,
                c,
                n_s + 1,
                "ICM: Voter 0 is sincere => necessary_coalition_size_icm[c] = n_s + 1 =",
            )

    # %% Coalition Manipulation (CM)

    def _cm_preliminary_optimize_bound_heuristic_(self, c, optimize_bounds):
        """CM: Try to improve bounds with heuristic.

        The general heuristic from class :class:`Rule` does not preserve the positions of the voters, so it is not
        valid for this rule: this method does nothing.
        """
        pass

    def _cm_fallback_elects_c_(self, c, n_m):
        """Whether ``c`` wins the fallback rule, assuming there is no Condorcet winner.

        Parameters
        ----------
        c : int
            Candidate for which we want to manipulate.
        n_m : int
            Number of manipulators (who all rank ``c`` first).

        Returns
        -------
        bool
            True iff ``c`` is elected by the fallback rule, i.e. iff the dictator ranks ``c`` first. If she prefers
            ``c`` to ``w``, she is the first manipulator (cf. the convention in the class docstring), so this is
            true as soon as ``n_m >= 1``. Otherwise, she is sincere.
        """
        if self.v_wants_to_help_c_[0, c]:
            return n_m >= 1
        else:
            return self.profile_.preferences_rk[0, 0] == c

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
        if not self._cm_fallback_elects_c_(c, n_m):
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
        A case where the polynomial algorithm cannot decide, but the exact one can:

        >>> profile = Profile(preferences_rk=[
        ...     [1, 0, 2, 3],
        ...     [0, 3, 2, 1],
        ...     [3, 0, 1, 2],
        ...     [2, 1, 0, 3],
        ...     [3, 0, 2, 1],
        ... ])
        >>> rule = RuleCondorcetDictatorship(cm_option='fast')(profile)
        >>> rule.candidates_cm_
        array([ 0., nan,  0.,  0.])
        >>> rule = RuleCondorcetDictatorship(cm_option='exact')(profile)
        >>> rule.candidates_cm_
        array([0., 0., 0., 0.])

        A case where candidate 3, the favorite candidate of the dictator (voter 0), wins by the fallback rule,
        whereas the other candidates would need to become Condorcet winners:

        >>> profile = Profile(preferences_rk=[
        ...     [3, 2, 1, 0],
        ...     [0, 2, 1, 3],
        ...     [3, 2, 0, 1],
        ...     [0, 2, 1, 3],
        ...     [1, 2, 3, 0],
        ... ])
        >>> rule = RuleCondorcetDictatorship()(profile)
        >>> rule.w_
        2
        >>> rule.candidates_cm_
        array([0., 0., 0., 1.])
        >>> rule.necessary_coalition_size_cm_
        array([4., 5., 0., 1.])
        >>> rule.sufficient_coalition_size_cm_
        array([4., 5., 0., 1.])

        The same profile, except that voters 0 and 1 are swapped. The former dictator is now voter 1, who is a
        sincere voter for candidate 3: candidate 3 cannot win anymore. But the new dictator prefers candidate 0 to
        the sincere winner, so candidate 0 wins by the fallback rule now.

        >>> profile = Profile(preferences_rk=[
        ...     [0, 2, 1, 3],
        ...     [3, 2, 1, 0],
        ...     [3, 2, 0, 1],
        ...     [0, 2, 1, 3],
        ...     [1, 2, 3, 0],
        ... ])
        >>> rule = RuleCondorcetDictatorship()(profile)
        >>> rule.candidates_cm_
        array([1., 0., 0., 0.])
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
        >>> rule = RuleCondorcetDictatorship()(profile)
        >>> rule.theta_critical_
        0
        """
        return 0
