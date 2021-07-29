# -*- coding: utf-8 -*-
"""
Created on 13 dec. 2018, 09:04
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
from svvamp.rules.rule import Rule
from svvamp.utils.util_cache import cached_property
from svvamp.preferences.profile import Profile
from svvamp.utils.misc import preferences_ut_to_matrix_duels_ut
from svvamp.utils.pseudo_bool import equal_false
from svvamp.rules.rule_irv import RuleIRV


class RuleCondorcetAbsIRV(Rule):
    """Absolute-Condorcet Instant Runoff Voting.

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
        >>> rule = RuleCondorcetAbsIRV()(profile)
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
        [2. 0. 1.]
        candidates_by_scores_best_to_worst
        [0 2 1]
        scores_best_to_worst
        [2. 1. 0.]
        w = 0
        score_w = 2.0
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
         ||     Condorcet_c_rk_ctb (False) ==> Condorcet_c_rk (False)     ||
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
        log_cm: cm_option = fast, tm_option = exact
        candidates_cm =
        [ 0.  0. nan]
        necessary_coalition_size_cm =
        [0. 1. 2.]
        sufficient_coalition_size_cm =
        [0. 2. 4.]

    Notes
    -----
    .. note::

        When in doubt between ``CondorcetAbsIRV`` and :class:`CondorcetVtbIRV`, we suggest to use
        :class:`CondorcetVtbIRV`.

    Each voter must provide a weak order, and a strict total order that is coherent with this weak order (i.e.,
    is a tie-breaking of this weak order).

    If there is a Condorcet winner (computed with the weak orders, i.e. in the sense of
    :attr:`matrix_victories_ut_abs`), then she is elected. Otherwise, :class:`RuleIRV` is used (with the strict total
    orders).

    If sincere preferences are strict total orders, then this voting system is equivalent to :class:`CondorcetVtbIRV`
    for sincere voting, but manipulators have more possibilities (they can pretend to have ties in their
    preferences). In that case especially, it is a more 'natural' framework to use :class:`CondorcetVtbIRV`.

    * :meth:`is_cm_`:

        * :attr:`cm_option` = ``'fast'``: Rely on :class:`RuleIRV`'s fast algorithm. Polynomial heuristic. Can prove CM
          but unable to decide non-CM (except in rare obvious cases).
        * :attr:`cm_option` = ``'slow'``: Rely on :class:`RuleExhaustiveBallot`'s exact algorithm. Non-polynomial
          heuristic (:math:`2^{n_c}`). Quite efficient to prove CM or non-CM.
        * :attr:`cm_option` = ``'very_slow'``: Rely on :class:`RuleIRV`'s exact algorithm. Non-polynomial
          heuristic (:math:`n_c!`). Very efficient to prove CM or non-CM.
        * :attr:`cm_option` = ``'exact'``: Non-polynomial algorithm from superclass :class:`Rule`.

        Each algorithm above exploits the faster ones. For example, if :attr:`cm_option` = ``'very_slow'``, SVVAMP
        tries the fast algorithm first, then the slow one, then the 'very slow' one. As soon as it reaches a
        decision, computation stops.

    * :meth:`is_icm_`: Exact in polynomial time.
    * :meth:`is_im_`: Non-polynomial or non-exact algorithms from superclass :class:`Rule`.
    * :meth:`is_iia_`: Non-polynomial or non-exact algorithms from superclass :class:`Rule`.
    * :meth:`is_tm_`: Exact in polynomial time.
    * :meth:`is_um_`: Non-polynomial or non-exact algorithms from superclass :class:`Rule`.

    References
    ----------
    'Condorcet criterion, ordinality and reduction of coalitional manipulability', François Durand,
    Fabien Mathieu and Ludovic Noirie, 2014.
    """

    full_name = 'Absolute-Condorcet IRV'
    abbreviation = 'ACIRV'

    options_parameters = Rule.options_parameters.copy()
    options_parameters.update({
        'cm_option': {'allowed': {'fast', 'slow', 'very_slow', 'exact'}, 'default': 'fast'},
        'tm_option': {'allowed': ['exact'], 'default': 'exact'},
        'icm_option': {'allowed': {'exact'}, 'default': 'exact'}
    })

    def __init__(self, **kwargs):
        self.irv_ = None
        super().__init__(
            with_two_candidates_reduces_to_plurality=True, is_based_on_rk=True,
            precheck_um=False, precheck_icm=False,
            log_identity="CONDORCET_ABS_IRV", **kwargs
        )

    def __call__(self, profile):
        self.delete_cache(suffix='_')
        self.profile_ = profile
        # Grab the IRV ballot of the profile (or create it)
        irv_options = {}
        if self.cm_option == 'fast':
            irv_options['cm_option'] = 'fast'
        elif self.cm_option == 'slow':
            irv_options['cm_option'] = 'slow'
        else:  # self.cm_option in {'very_slow', 'exact'}
            irv_options['cm_option'] = 'exact'
        self.irv_ = RuleIRV(**irv_options)(self.profile_)
        return self

    # %% Counting the ballots

    @cached_property
    def w_(self):
        if not np.isnan(self.profile_.condorcet_winner_ut_abs):
            return self.profile_.condorcet_winner_ut_abs
        else:
            return self.irv_.w_

    @cached_property
    def scores_(self):
        """1d or 2d array.

            * If there is a Condorcet winner, then ``scores[c]`` is the number of victories for ``c`` in
              ``matrix_victories_ut_abs``.
            * Otherwise, ``scores[r, c]`` is defined like in :class:`RuleIRV`: it is the number of voters who vote for
              candidate ``c`` at round ``r``. For eliminated candidates, ``scores[r, c] = numpy.nan``. In contrast,
              ``scores[r, c] = 0`` means that ``c`` is present at round ``r`` but no voter votes for ``c``.
        """
        if not np.isnan(self.profile_.condorcet_winner_ut_abs):
            return np.sum(self.profile_.matrix_victories_ut_abs, 1)
        else:
            return self.irv_.scores_

    @cached_property
    def candidates_by_scores_best_to_worst_(self):
        """1d array of integers.

            * If there is a Condorcet winner, candidates are sorted according to their (scalar) score.
            * Otherwise, ``candidates_by_scores_best_to_worst`` is the list of all candidates in the reverse order of
              their IRV elimination.
        """
        if not np.isnan(self.profile_.condorcet_winner_ut_abs):
            return np.argsort(- self.scores_, kind='mergesort')
        else:
            return self.irv_.candidates_by_scores_best_to_worst_

    @cached_property
    def _v_might_be_pivotal_(self):
        self.mylog("Count ballots", 1)
        if not np.isnan(self.profile_.condorcet_winner_ut_abs):
            w = self.w_
            v_might_be_pivotal = np.zeros(self.profile_.n_v)
            for c in np.where(self.profile_.matrix_duels_ut[w, :] <= self.profile_.n_v / 2 + 1)[0]:
                if c == w:
                    continue
                # Search voters who can prevent the victory for ``w`` against ``c``.
                v_might_be_pivotal[self.profile_.preferences_ut[:, w] > self.profile_.preferences_ut[:, c]] = True
        else:
            eb = self.irv_.eb_
            # First way of being (maybe) pivotal: change the result of IRV.
            v_might_be_pivotal = np.copy(eb.v_might_be_pivotal_)
            # Another way of being (maybe) pivotal: create a Condorcet winner.
            for c in range(self.profile_.n_c):
                if np.any(self.profile_.matrix_duels_ut[c, np.not_equal(np.array(range(self.profile_.n_c)), c)]
                          <= self.profile_.n_v / 2 - 1):
                    # ``c`` cannot become a Condorcet winner.
                    continue
                # ``close_candidates`` are the candidates against which ``c`` does not have a victory.
                close_candidates = np.less_equal(self.profile_.matrix_duels_ut[c, :], self.profile_.n_v / 2)
                close_candidates[c] = False
                # Voter ``v`` can make ``c`` become a Condorcet winner iff, among ``close_candidates``, she likes ``c``
                # the least (this way, she can improve ``c``'s scores in all these duels, compared to her sincere
                # voting).
                v_might_be_pivotal[np.all(np.less(
                    self.profile_.preferences_ut[:, c][:, np.newaxis], self.profile_.preferences_ut[:, close_candidates]
                ), 1)] = True
        return v_might_be_pivotal

    @cached_property
    def v_might_im_for_c_(self):
        return np.tile(self._v_might_be_pivotal_[:, np.newaxis], (1, self.profile_.n_c))

    # %% Manipulation criteria of the voting system

    @cached_property
    def meets_majority_favorite_c_rk_ctb(self):
        return True

    @cached_property
    def meets_condorcet_c_ut_abs(self):
        return True

    # %% Individual manipulation (IM)

    # TODO: should be implemented.

    # %% Trivial Manipulation (TM)

    # Use the general methods from class Rule.

    # %% Unison manipulation (UM)

    # TODO: should be implemented .

    def _um_preliminary_checks_general_subclass_(self):
        if equal_false(self.irv_.is_cm_):
            self._is_um = False
            self._candidates_um[:] = False
            self._um_was_computed_with_candidates = True

    def _um_preliminary_checks_c_(self, c):
        if self.um_option not in {'fast', 'lazy'} or self.cm_option not in {'fast', 'lazy'}:
            if (
                self.w_ == self.profile_.condorcet_winner_rk_ctb
                and not self.profile_.c_might_be_there_when_cw_is_eliminated_irv_style[c]
            ):
                # Impossible to manipulate with n_m manipulators
                self._candidates_um[c] = False

    # %% Ignorant-Coalition Manipulation (ICM)

    # Use the general methods from class Rule.

    # %% Coalition Manipulation (CM)

    @cached_property
    def losing_candidates_(self):
        """If ``irv_.w_`` does not win, then we put her first. Other losers are sorted as usual. (scores in
        ``matrix_duels_ut``).

        Examples
        --------
            >>> profile = Profile(preferences_rk=[
            ...     [0, 1, 2],
            ...     [0, 1, 2],
            ...     [1, 2, 0],
            ...     [2, 1, 0],
            ...     [2, 1, 0],
            ... ])
            >>> rule = RuleCondorcetAbsIRV()(profile)
            >>> rule.candidates_cm_
            array([nan,  0.,  1.])
        """
        self.mylog("Compute ordered list of losing candidates", 1)
        if self.w_ == self.irv_.w_:
            # As usual
            losing_candidates = np.concatenate((
                np.array(range(0, self.w_), dtype=int), np.array(range(self.w_ + 1, self.profile_.n_c), dtype=int)))
            losing_candidates = losing_candidates[np.argsort(
                - self.profile_.matrix_duels_ut[losing_candidates, self.w_], kind='mergesort')]
        else:
            # Put irv_.w_ first.
            losing_candidates = np.array([
                c for c in range(self.profile_.n_c) if c != self.w_ and c != self.irv_.w_]).astype(np.int)
            losing_candidates = losing_candidates[np.argsort(
                - self.profile_.matrix_duels_ut[losing_candidates, self.w_], kind='mergesort')]
            losing_candidates = np.concatenate(([self.irv_.w_], losing_candidates))
        return losing_candidates

    def _cm_preliminary_checks_general_subclass_(self):
        # We check IRV first.
        if self.w_ != self.irv_.w_:
            # Then IRV is manipulable anyway (for ``w``, so a precheck on IRV would give us no information).
            return
        if self.cm_option != "fast":
            if self.cm_option == "slow":
                self.irv_.cm_option = "slow"
            else:
                self.irv_.cm_option = "exact"
            if equal_false(self.irv_.is_cm_):
                # Condorcification theorem apply.
                self.mylog("CM impossible (since it is impossible for IRV)", 2)
                self._is_cm = False
                self._candidates_cm[:] = False
                self._cm_was_computed_with_candidates = True

    def _cm_main_work_c_(self, c, optimize_bounds):
        """
            >>> profile = Profile(preferences_rk=[
            ...     [0, 2, 1],
            ...     [0, 2, 1],
            ...     [1, 0, 2],
            ...     [1, 0, 2],
            ...     [2, 0, 1],
            ... ])
            >>> rule = RuleCondorcetAbsIRV(cm_option='slow')(profile)
            >>> rule.candidates_cm_
            array([0., 0., 0.])

            >>> profile = Profile(preferences_rk=[
            ...     [0, 2, 1],
            ...     [0, 2, 1],
            ...     [1, 0, 2],
            ...     [1, 0, 2],
            ...     [2, 0, 1],
            ... ])
            >>> rule = RuleCondorcetAbsIRV(cm_option='very_slow')(profile)
            >>> rule.candidates_cm_
            array([0., 0., 0.])

            >>> profile = Profile(preferences_rk=[
            ...     [0, 2, 1],
            ...     [0, 2, 1],
            ...     [1, 0, 2],
            ...     [1, 0, 2],
            ...     [2, 0, 1],
            ... ])
            >>> rule = RuleCondorcetAbsIRV(cm_option='exact')(profile)
            >>> rule.candidates_cm_
            array([0., 0., 0.])

            >>> profile = Profile(preferences_rk=[
            ...     [0, 2, 1, 3],
            ...     [1, 0, 2, 3],
            ...     [1, 0, 3, 2],
            ...     [2, 3, 0, 1],
            ...     [3, 0, 1, 2],
            ... ])
            >>> rule = RuleCondorcetAbsIRV(cm_option='slow')(profile)
            >>> rule.is_cm_
            nan

            >>> profile = Profile(preferences_rk=[
            ...     [0, 2, 1, 3],
            ...     [1, 0, 2, 3],
            ...     [1, 0, 3, 2],
            ...     [2, 3, 0, 1],
            ...     [3, 0, 1, 2],
            ... ])
            >>> rule = RuleCondorcetAbsIRV(cm_option='very_slow')(profile)
            >>> rule.is_cm_
            False

            >>> profile = Profile(preferences_rk=[
            ...     [0, 2, 1, 3],
            ...     [1, 2, 0, 3],
            ...     [2, 1, 3, 0],
            ...     [3, 1, 0, 2],
            ...     [3, 2, 1, 0],
            ... ])
            >>> rule = RuleCondorcetAbsIRV(cm_option='exact')(profile)
            >>> rule.is_cm_
            True

            >>> profile = Profile(preferences_rk=[
            ...     [0, 2, 1],
            ...     [1, 0, 2],
            ...     [1, 0, 2],
            ...     [2, 1, 0],
            ...     [2, 1, 0],
            ... ])
            >>> rule = RuleCondorcetAbsIRV(cm_option='very_slow')(profile)
            >>> rule.is_cm_c_(1)
            False
        """
        # Decondorcification is independent from what we will do for IRV. It just gives a necessary coalition size.
        #
        # 1) If ``irv_.w_`` is Condorcet:
        #
        #    * Manipulate IRV --> use IRV, we know.
        #    * Prevent ``w`` from being a Condorcet winner --> we know.
        #    * If incomplete: we need ``irv_.cm_`` (incomplete). But if IRV works for ``c`` and Cond-IRV does not,
        #      we might need IRV.CM.complete (we will know after).
        #
        # 2) If ``irv_.w_ != CondIRV.w_`` (which is Condorcet):
        #
        #    a) If ``c = irv_.w_``:
        #
        #       * Make ``w`` win: easy, and we have ``suff = n_m`` (``nec`` can be less)
        #       * Prevent ``CondIRV.w`` from being Condorcet --> we know.
        #
        #   b) If ``c != irv_.w``:
        #
        #      * Make ``c`` win: complicated. We can use the elimination path we know, but it might not succeed.
        #      * Prevent ``CondIRV.w`` from being Condorcet --> we know.
        #      * If incomplete: we do not even need ``irv_.CM``. But we need to make sure that we try ``w`` first in
        #        ``losing_candidates_``. However, if it fails, we will need to try other ``c``'s. What do we do then?
        #        Use a fast IRV on the fly? Or simply try the elimination path suggested by IRV? Etc.  Maybe the
        #        first is better (cheap and likely to be more efficient).
        #
        # 3) If there is no Condorcet winner:
        #
        #    * Manipulate IRV --> use IRV, we know.
        #    * Prevent a Condorcet winner --> trivial.
        #    * If incomplete: cf. 1), in fact it is the same. When do we calculate IRV? It can be after we examine
        #      ``irv_.w_``; so, not at the very beginning
        n_m = self.profile_.matrix_duels_ut[c, self.w_]
        n_s = self.profile_.n_v - n_m
        candidates = np.array(range(self.profile_.n_c))
        preferences_utilities_s = self.profile_.preferences_ut[np.logical_not(self.v_wants_to_help_c_[:, c]), :]
        matrix_duels_temp = (preferences_ut_to_matrix_duels_ut(preferences_utilities_s))
        self.mylogm("CM: matrix_duels_temp =", matrix_duels_temp, 3)
        # More preliminary checks. It's more convenient to put them in that method, because we need
        # ``preferences_utilities_s`` and ``matrix_duels_temp``.
        # ``min_d(matrix_duels_temp[c, d]) + n_m > (n_s + n_m) / 2``, i.e.:
        # ``n_m >= n_s + 1 - 2 * min_d(matrix_duels_temp[c, d])``
        d_neq_c = (np.array(range(self.profile_.n_c)) != c)
        n_manip_becomes_cond = np.maximum(n_s + 1 - 2 * np.min(matrix_duels_temp[c, d_neq_c]), 0)
        self.mylogv("CM: n_manip_becomes_cond =", n_manip_becomes_cond, 3)
        self._update_sufficient(self._sufficient_coalition_size_cm, c, n_manip_becomes_cond,
                                'CM: Update sufficient_coalition_size_cm[c] = n_manip_becomes_cond =')
        if not optimize_bounds and n_m >= self._sufficient_coalition_size_cm[c]:  # pragma: no cover
            # TO DO: Investigate whether this case can actually happen.
            self._reached_uncovered_code()
            return True
        # Prevent another cond. Look at the weakest duel for ``d``, she has ``matrix_duels_temp[d, e]``. We simply need
        # that:
        # ``matrix_duels_temp[d, e] <= (n_s + n_m) / 2``
        # ``2 * max_d(min_e(matrix_duels_temp[d, e])) - n_s <= n_m``
        n_manip_prevent_cond = 0
        for d in candidates[d_neq_c]:
            e_neq_d = (np.array(range(self.profile_.n_c)) != d)
            n_prevent_d = np.maximum(2 * np.min(matrix_duels_temp[d, e_neq_d]) - n_s, 0)
            n_manip_prevent_cond = max(n_manip_prevent_cond, n_prevent_d)
        self.mylogv("CM: n_manip_prevent_cond =", n_manip_prevent_cond, 3)
        self._update_necessary(self._necessary_coalition_size_cm, c, n_manip_prevent_cond,
                               'CM: Update necessary_coalition_size_cm[c] = n_manip_prevent_cond =')
        if not optimize_bounds and self._necessary_coalition_size_cm[c] > n_m:  # pragma: no cover
            # TO DO: Investigate whether this case can actually happen.
            self._reached_uncovered_code()
            return True

        is_quick_escape_one = False
        if self.w_ == self.irv_.w_:
            if self.cm_option == "fast":
                self.irv_.cm_option = "fast"
            elif self.cm_option == "slow":
                self.irv_.cm_option = "slow"
            else:
                self.irv_.cm_option = "exact"
            if optimize_bounds:
                self.irv_.is_cm_c_with_bounds_(c)
            else:
                self.irv_.is_cm_c_(c)
                is_quick_escape_one = True
            self.mylog('CM: c != self.irv_.w_ == self.w_', 3)
            self.mylogv("CM: self.irv_.sufficient_coalition_size_cm_[c] =",
                        self.irv_.sufficient_coalition_size_cm_[c], 3)
            self._update_sufficient(self._sufficient_coalition_size_cm, c,
                                    max(self.irv_.sufficient_coalition_size_cm_[c], n_manip_prevent_cond),
                                    'CM: Update sufficient_coalition_size[c] =')
            self.mylogv("CM: self.irv_.necessary_coalition_size_cm_[c] =", self.irv_.necessary_coalition_size_cm_[c], 3)
            self._update_necessary(
                self._necessary_coalition_size_cm, c,
                min(n_manip_becomes_cond, max(self.irv_.necessary_coalition_size_cm_[c], n_manip_prevent_cond)),
                'CM: Update necessary_coalition_size[c] =')
        else:
            if c == self.irv_.w_:  # pragma: no cover
                # TO DO: Investigate whether this case can actually happen.
                # self._reached_uncovered_code()
                self.mylog('CM: c == self.irv_.w_ != self.w_', 3)
                self.mylogv("CM: sufficient size for IRV (sincere IRV) =", n_m, 3)
                self._update_sufficient(self._sufficient_coalition_size_cm, c, max(n_m, n_manip_prevent_cond),
                                        'CM: Update sufficient_coalition_size[c] =')
            else:
                self.mylog('CM: c, self.w_, self.irv_.w_ all distinct', 3)
                # We do not know how many manipulators can make ``c`` win in IRV (note that it would not be the same
                # set of manipulators in IRV and here)
        # self.mylogv("CM: Preliminary checks.2: necessary_coalition_size_cm[c] =",
        #             self._necessary_coalition_size_cm[c], 3)
        # self.mylogv("CM: Preliminary checks.2: sufficient_coalition_size_cm[c] =",
        #             self._sufficient_coalition_size_cm[c], 3)
        # self.mylogv("CM: Preliminary checks.2: n_m =", self.profile_.matrix_duels_ut[c, self.w], 3)
        # if self._necessary_coalition_size_cm[c] > n_m:
        #     self.mylogv("CM: Preliminary checks.2: CM is False for c =", c, 2)
        # elif self._sufficient_coalition_size_cm[c] <= n_m:
        #     self.mylogv("CM: Preliminary checks.2: CM is True for c =", c, 2)
        # else:
        #     self.mylogv("CM: Preliminary checks.2: CM is unknown for c =", c, 2)
        # Real work
        is_quick_escape_two = False
        if self.cm_option == 'exact':
            is_quick_escape_two = self._cm_main_work_c_exact_(c, optimize_bounds)
        return is_quick_escape_one or is_quick_escape_two
