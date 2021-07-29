# -*- coding: utf-8 -*-
"""
Created on 13 dec. 2018, 11:40
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
import itertools
import numpy as np
from svvamp.rules.rule import Rule
from svvamp.utils.util_cache import cached_property
from svvamp.utils.pseudo_bool import equal_true, equal_false
from svvamp.preferences.profile import Profile
from svvamp.utils.misc import preferences_ut_to_matrix_duels_ut
from svvamp.rules.rule_irv import RuleIRV


class RuleCondorcetVtbIRV(Rule):
    """Condorcet Instant Runoff Voting

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
        >>> rule = RuleCondorcetVtbIRV()(profile)
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
    Each voter must provide a strict total order. If there is a Condorcet winner (in the sense of
    :attr:`matrix_victories_rk`), then she is elected. Otherwise, :class:`RuleIRV` is used.

    If sincere preferences are strict total orders, then this voting system is equivalent to
    :class:`RuleCondorcetAbsIRV` for sincere voting, but manipulators have less possibilities (they are forced to
    provide strict total orders).

    * :meth:`is_cm_`:

        * :attr:`cm_option` = ``'fast'``: Rely on :class:`RuleIRV`'s fast algorithm. Polynomial heuristic. Can prove
          CM but unable to decide non-CM (except in rare obvious cases).
        * :attr:`cm_option` = ``'slow'``: Rely on :class:`RuleExhaustiveBallot`'s exact algorithm. Non-polynomial
          heuristic (:math:`2^{n_c}`). Quite efficient to prove CM or non-CM.
        * :attr:`cm_option` = ``'very_slow'``: Rely on :class:`RuleIRV`'s exact algorithm. Non-polynomial
          heuristic (:math:`n_c!`). Very efficient to prove CM or non-CM.
        * :attr:`cm_option` = ``'exact'``: Non-polynomial algorithm from superclass :class:`Rule`.

        Each algorithm above exploits the faster ones. For example, if :attr:`cm_option` = ``'very_slow'``,
        SVVAMP tries the fast algorithm first, then the slow one, then the 'very slow' one. As soon as it reaches
        a decision, computation stops.

    * :meth:`is_icm_`: Exact in polynomial time.
    * :meth:`is_im_`: Non-polynomial or non-exact algorithms from superclass :class:`Rule`.
    * :meth:`is_iia`: Non-polynomial or non-exact algorithms from superclass :class:`Rule`. If
      :attr:`iia_subset_maximum_size` = 2, it runs in polynomial time and is exact up to ties (which can occur only if
      :attr:`n_v` is even).
    * :meth:`is_tm_`: Exact in polynomial time.
    * :meth:`is_um_`: Non-polynomial or non-exact algorithms from superclass :class:`Rule`.

    References
    ----------
    'Condorcet criterion, ordinality and reduction of coalitional manipulability', François Durand,
    Fabien Mathieu and Ludovic Noirie, 2014.
    """

    full_name = 'Condorcet IRV'
    abbreviation = 'CIRV'

    options_parameters = Rule.options_parameters.copy()
    options_parameters.update({
        'cm_option': {'allowed': {'fast', 'slow', 'very_slow', 'exact'}, 'default': 'fast'},
        'tm_option': {'allowed': ['exact'], 'default': 'exact'},
        'icm_option': {'allowed': {'exact'}, 'default': 'exact'}
    })

    def __init__(self, **kwargs):
        # self.irv_ = None
        super().__init__(
            with_two_candidates_reduces_to_plurality=True, is_based_on_rk=True,
            precheck_um=False, precheck_icm=False,
            log_identity="CONDORCET_VTB_IRV", **kwargs
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
        else:  # self.cm_option in {'very_slow', 'exact'}:
            irv_options['cm_option'] = 'exact'
        self.irv_ = RuleIRV(**irv_options)(self.profile_)
        return self

    # %% Counting the ballots

    @cached_property
    def w_(self):
        if not np.isnan(self.profile_.condorcet_winner_rk):
            return self.profile_.condorcet_winner_rk
        else:
            return self.irv_.w_

    @cached_property
    def scores_(self):
        """1d or 2d array.

            * If there is a Condorcet winner, then ``scores[c]`` is the number of victories for ``c`` in
              :attr:`matrix_duels_rk`.
            * Otherwise, ``scores[r, c]`` is defined like in :class:`RuleIRV`: it is the number of voters who vote
              for candidate ``c`` at round ``r``. For eliminated candidates, ``scores[r, c] = numpy.nan``. In contrast,
              ``scores[r, c] = 0`` means that ``c`` is present at round ``r`` but no voter votes for ``c``.
        """
        if not np.isnan(self.profile_.condorcet_winner_rk):
            return np.sum(self.profile_.matrix_victories_rk, 1)
        else:
            return self.irv_.scores_

    @cached_property
    def candidates_by_scores_best_to_worst_(self):
        """1d array of integers.

            * If there is a Condorcet winner, candidates are sorted according to  their (scalar) score.
            * Otherwise, ``candidates_by_scores_best_to_worst`` is the list of all candidates in the reverse order of
              their IRV elimination.
        """
        if not np.isnan(self.profile_.condorcet_winner_rk):
            return np.argsort(- self.scores_, kind='mergesort')
        else:
            return self.irv_.candidates_by_scores_best_to_worst_

    @cached_property
    def _v_might_be_pivotal_(self):
        self.mylog("Compute _v_might_be_pivotal_", 1)
        if not np.isnan(self.profile_.condorcet_winner_rk):
            w = self.w_
            v_might_be_pivotal = np.zeros(self.profile_.n_v)
            for c in np.where(self.profile_.matrix_duels_rk[w, :] <= self.profile_.n_v / 2 + 1)[0]:
                if c == w:
                    continue
                # Search voters who can prevent the victory for ``w`` against ``c``.
                v_might_be_pivotal[
                    self.profile_.preferences_borda_rk[:, w] > self.profile_.preferences_borda_rk[:, c]
                ] = True
        else:
            eb = self.irv_.eb_
            # First way of being (maybe) pivotal: change the result of IRV.
            v_might_be_pivotal = np.copy(eb.v_might_be_pivotal_)
            # Another way of being (maybe) pivotal: create a Condorcet winner.
            for c in range(self.profile_.n_c):
                if np.any(self.profile_.matrix_duels_rk[:, c] >= self.profile_.n_v / 2 + 1):
                    # ``c`` cannot become a Condorcet winner.
                    continue
                # ``close_candidates`` are the candidates against which ``c`` does not have a victory.
                close_candidates = np.less_equal(self.profile_.matrix_duels_rk[c, :], self.profile_.n_v / 2)
                # Voter ``v`` can make ``c`` become a Condorcet winner iff, among ``close_candidates``, she vtb-likes
                # ``c`` the least (this way, she can improve ``c``'s scores in all these duels, compared to
                # her sincere voting).
                v_might_be_pivotal[np.all(np.less(
                        self.profile_.preferences_borda_rk[:, c][:, np.newaxis],
                        self.profile_.preferences_borda_rk[:, close_candidates]
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
    def meets_condorcet_c_rk(self):
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
        """If ``irv_.w_ does not win, then we put her first. Other losers are sorted as usual. (scores in
        ``matrix_duels_ut``).

            >>> profile = Profile(preferences_rk=[
            ...     [0, 1, 2],
            ...     [1, 0, 2],
            ...     [1, 0, 2],
            ...     [2, 0, 1],
            ...     [2, 0, 1],
            ... ])
            >>> rule = RuleCondorcetVtbIRV()(profile)
            >>> rule.candidates_cm_
            array([ 0.,  1., nan])
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
            losing_candidates = np.array(
                [c for c in range(self.profile_.n_c) if c != self.w_ and c != self.irv_.w_]).astype(np.int)
            losing_candidates = losing_candidates[np.argsort(
                - self.profile_.matrix_duels_ut[losing_candidates, self.w_], kind='mergesort')]
            losing_candidates = np.concatenate(([self.irv_.w_], losing_candidates))
        return losing_candidates

    def _cm_aux_(self, c, n_m, suggested_path, preferences_borda_s, matrix_duels_temp):
        """Algorithm used for CM (using suggested path).

        Parameters
        ----------
        c : int
            Candidate for which we want to manipulate.
        n_m : int
            The exact number of manipulators.
        suggested_path : List
            A suggested path of elimination (for the IRV part). It must work with ``n_m`` manipulators.
        preferences_borda_s : list of list (or ndarray)
            Preferences of the sincere voters (in Borda format).

        Returns
        -------
        bool
            ``manipulation_found``.

        Examples
        --------
            >>> profile = Profile(preferences_rk=[
            ...     [0, 1, 2, 3],
            ...     [0, 3, 1, 2],
            ...     [1, 3, 0, 2],
            ...     [2, 3, 1, 0],
            ...     [3, 2, 1, 0],
            ... ])
            >>> rule = RuleCondorcetVtbIRV(cm_option='very_slow')(profile)
            >>> rule.candidates_cm_
            array([1., 1., 1., 0.])
        """
        candidates = np.array(range(self.profile_.n_c))

        # Step 1: elimination path
        # And consequences on the majority matrix
        scores_m_begin_r = np.zeros(self.profile_.n_c)
        is_candidate_alive_begin_r = np.ones(self.profile_.n_c, dtype=np.bool)
        current_top_v = np.array(- np.ones(n_m))  # -1 means that v is available
        candidates_to_put_in_ballot = np.ones((n_m, self.profile_.n_c), dtype=np.bool)
        for r in range(self.profile_.n_c - 1):
            self.mylogv("cm_aux: r =", r, 3)
            scores_tot_begin_r = np.full(self.profile_.n_c, np.nan)
            scores_tot_begin_r[is_candidate_alive_begin_r] = np.sum(np.equal(
                preferences_borda_s[:, is_candidate_alive_begin_r],
                np.max(preferences_borda_s[:, is_candidate_alive_begin_r], 1)[:, np.newaxis]
            ), 0)
            self.mylogv("cm_aux: scores_s_begin_r =", scores_tot_begin_r, 3)
            self.mylogv("cm_aux: scores_m_begin_r =", scores_m_begin_r, 3)
            scores_tot_begin_r += scores_m_begin_r
            self.mylogv("cm_aux: scores_tot_begin_r =", scores_tot_begin_r, 3)
            d = suggested_path[r]
            self.mylogv("cm_aux: d =", d, 3)
            scores_m_new_r = np.zeros(self.profile_.n_c, dtype=np.int)
            scores_m_new_r[is_candidate_alive_begin_r] = np.maximum(
                0,
                scores_tot_begin_r[d] - scores_tot_begin_r[is_candidate_alive_begin_r]
                + (candidates[is_candidate_alive_begin_r] > d))
            self.mylogv("cm_aux: scores_m_new_r =", scores_m_new_r, 3)
            # Update variables for next round
            scores_m_begin_r = scores_m_begin_r + scores_m_new_r
            if np.sum(scores_m_begin_r) > n_m:
                raise NotImplementedError("Error: this should not happen.")
            scores_m_begin_r[d] = 0
            is_candidate_alive_begin_r[d] = False
            # We need to attribute manipulator's votes to specific manipulators. This is done arbitrarily,
            # and has consequences on the majority matrix. It is the main cause that makes the algorithm not exact in
            #  theory (the other one being step 3).
            free_manipulators = np.where(current_top_v == -1)[0]
            i_manipulator = 0
            for e in range(self.profile_.n_c):
                n_manip_new_e = scores_m_new_r[e]
                for k in range(n_manip_new_e):
                    manipulator = free_manipulators[i_manipulator]
                    current_top_v[manipulator] = e
                    candidates_to_put_in_ballot[manipulator, e] = False
                    matrix_duels_temp[e, candidates_to_put_in_ballot[manipulator, :]] += 1
                    i_manipulator += 1
            current_top_v[current_top_v == d] = -1
            self.mylogv("cm_aux: current_top_v =", current_top_v, 3)
            self.mylogm("cm_aux: matrix_duels_temp =", matrix_duels_temp, 3)

        # Step 2
        # Ensure that no candidate ``!= c`` is Condorcet winner. If ``c`` is not yet in all ballots, put it.
        for manipulator in np.where(candidates_to_put_in_ballot[:, c])[0]:
            candidates_to_put_in_ballot[manipulator, c] = False
            matrix_duels_temp[c, candidates_to_put_in_ballot[manipulator, :]] += 1
        self.mylogv("cm_aux: Adding to all ballots c =", c, 3)
        self.mylogm("cm_aux: matrix_duels_temp =", matrix_duels_temp, 3)
        # If some candidates already have some non-victories in the matrix of duels, they can safely be put in the
        # ballots. Maybe this will generate non-victories for other candidates, etc.
        candidates_ok = np.zeros(self.profile_.n_c, dtype=np.bool)
        candidates_ok[c] = True
        i_found_a_new_ok = True
        while i_found_a_new_ok:
            not_yet_ok = np.where(np.logical_not(candidates_ok))[0]
            if not_yet_ok.shape[0] == 0:
                self.mylog("cm_aux: Decondorcification succeeded", 3)
                return True
            i_found_a_new_ok = False
            for d in not_yet_ok:
                if np.any(matrix_duels_temp[:, d] >= self.profile_.n_v / 2):
                    candidates_ok[d] = True
                    i_found_a_new_ok = True
                    for manipulator in np.where(candidates_to_put_in_ballot[:, d])[0]:
                        candidates_to_put_in_ballot[manipulator, d] = False
                        matrix_duels_temp[d, candidates_to_put_in_ballot[manipulator, :]] += 1
                    self.mylogv("cm_aux: Found a non-Condorcet d =", d, 3)
                    self.mylogm("cm_aux: matrix_duels_temp =", matrix_duels_temp, 3)

        # Step 3
        # Some candidates have left, who do not have non-victories yet. We will put them in the ballots,
        # while favoring a Condorcet cycle 0 > 1 > ... > C-1 > 0. N.B.: In practice, this step seems never necessary.
        self.mylog("cm_aux: Step 3 needed", 1)
        for manipulator in range(n_m):
            candidate_start = manipulator % self.profile_.n_c
            for d in itertools.chain(range(candidate_start, self.profile_.n_c), range(candidate_start)):
                if candidates_to_put_in_ballot[manipulator, d]:
                    candidates_to_put_in_ballot[manipulator, d] = False
                    matrix_duels_temp[d, candidates_to_put_in_ballot[manipulator, :]] += 1
        for d in np.where(np.logical_not(candidates_ok))[0]:
            if np.all(matrix_duels_temp[:, d] < self.profile_.n_v / 2):
                self.mylog("cm_aux: Decondorcification failed", 1)
                return False
        # TO DO: Investigate whether this case can actually happen.
        self._reached_uncovered_code()  # pragma: no cover
        self.mylog("cm_aux: Decondorcification succeeded", 1)  # pragma: no cover
        return True  # pragma: no cover

    def _cm_main_work_c_(self, c, optimize_bounds):
        """
            >>> profile = Profile(preferences_rk=[
            ...     [0, 1, 2],
            ...     [0, 1, 2],
            ...     [1, 0, 2],
            ...     [1, 2, 0],
            ...     [2, 1, 0],
            ... ])
            >>> rule = RuleCondorcetVtbIRV(cm_option='very_slow')(profile)
            >>> rule.candidates_cm_
            array([0., 0., 0.])

            >>> profile = Profile(preferences_rk=[
            ...     [0, 2, 3, 1],
            ...     [1, 2, 0, 3],
            ...     [1, 3, 0, 2],
            ...     [2, 0, 3, 1],
            ...     [3, 2, 1, 0],
            ... ])
            >>> rule = RuleCondorcetVtbIRV()(profile)
            >>> rule.candidates_cm_
            array([ 1., nan,  0., nan])

            >>> profile = Profile(preferences_ut=[
            ...     [-1. , -1. , -1. ],
            ...     [ 1. ,  1. , -0.5],
            ...     [-0.5,  1. , -1. ],
            ...     [-1. ,  1. ,  0. ],
            ...     [ 0.5, -0.5,  0.5],
            ... ], preferences_rk=[
            ...     [0, 1, 2],
            ...     [0, 1, 2],
            ...     [1, 0, 2],
            ...     [1, 2, 0],
            ...     [2, 0, 1],
            ... ])
            >>> rule = RuleCondorcetVtbIRV()(profile)
            >>> rule.candidates_cm_
            array([ 0., nan,  0.])

            >>> profile = Profile(preferences_ut=[
            ...     [ 1. , -0.5, -1. ],
            ...     [ 0.5, -1. ,  0.5],
            ...     [-1. ,  0.5,  0. ],
            ...     [-1. ,  1. , -1. ],
            ...     [ 1. , -1. ,  1. ],
            ... ], preferences_rk=[
            ...     [0, 1, 2],
            ...     [0, 2, 1],
            ...     [1, 2, 0],
            ...     [1, 2, 0],
            ...     [2, 0, 1],
            ... ])
            >>> rule = RuleCondorcetVtbIRV()(profile)
            >>> rule.candidates_cm_
            array([ 0., nan,  1.])

            >>> profile = Profile(preferences_rk=[
            ...     [0, 1, 2],
            ...     [0, 1, 2],
            ...     [0, 2, 1],
            ...     [1, 0, 2],
            ...     [1, 2, 0],
            ...     [1, 2, 0],
            ...     [2, 0, 1],
            ...     [2, 1, 0],
            ...     [2, 1, 0],
            ...     [2, 1, 0],
            ... ])
            >>> rule = RuleCondorcetVtbIRV()(profile)
            >>> rule.necessary_coalition_size_cm_
            array([2., 3., 0.])

            >>> profile = Profile(preferences_rk=[
            ...     [0, 4, 3, 2, 1],
            ...     [1, 2, 3, 0, 4],
            ...     [2, 4, 3, 1, 0],
            ...     [3, 4, 1, 2, 0],
            ...     [4, 0, 1, 2, 3],
            ... ])
            >>> rule = RuleCondorcetVtbIRV()(profile)
            >>> rule.necessary_coalition_size_cm_
            array([1., 2., 1., 1., 0.])

            >>> profile = Profile(preferences_ut=[
            ...     [ 1. ,  0. , -0.5],
            ...     [ 1. ,  0.5, -1. ],
            ...     [ 0. , -1. , -1. ],
            ...     [ 0.5, -0.5,  0.5],
            ...     [-0.5,  0.5,  1. ],
            ... ], preferences_rk=[
            ...     [0, 1, 2],
            ...     [0, 1, 2],
            ...     [0, 2, 1],
            ...     [2, 0, 1],
            ...     [2, 1, 0],
            ... ])
            >>> rule = RuleCondorcetVtbIRV()(profile)
            >>> rule.necessary_coalition_size_cm_
            array([0., 3., 3.])

            >>> profile = Profile(preferences_rk=[
            ...     [0, 1, 2],
            ...     [0, 1, 2],
            ...     [0, 2, 1],
            ...     [2, 0, 1],
            ...     [2, 1, 0],
            ... ])
            >>> rule = RuleCondorcetVtbIRV(cm_option='slow')(profile)
            >>> rule.necessary_coalition_size_cm_
            array([0., 5., 4.])

            >>> profile = Profile(preferences_rk=[
            ...     [0, 2, 1],
            ...     [0, 2, 1],
            ...     [1, 0, 2],
            ...     [1, 0, 2],
            ...     [2, 1, 0],
            ... ])
            >>> rule = RuleCondorcetVtbIRV(cm_option='exact')(profile)
            >>> rule.necessary_coalition_size_cm_
            array([3., 0., 3.])

            >>> profile = Profile(preferences_ut=[
            ...     [ 0.5, -0.5,  0. , -0.5, -1. ],
            ...     [ 0.5,  1. ,  1. , -1. ,  0. ],
            ...     [-1. ,  1. ,  0. ,  0. ,  0. ],
            ...     [ 0. ,  0. ,  1. ,  0.5, -1. ],
            ...     [ 0. , -1. ,  0. ,  0.5, -1. ],
            ...     [ 0.5,  0.5,  0. ,  0.5,  0.5],
            ... ], preferences_rk=[
            ...     [0, 2, 3, 1, 4],
            ...     [1, 2, 0, 4, 3],
            ...     [1, 3, 2, 4, 0],
            ...     [2, 3, 1, 0, 4],
            ...     [3, 2, 0, 4, 1],
            ...     [4, 3, 1, 0, 2],
            ... ])
            >>> rule = RuleCondorcetVtbIRV(cm_option='very_slow')(profile)
            >>> rule.necessary_coalition_size_cm_
            array([3., 3., 2., 0., 4.])
        """
        self.mylogv("CM: Compute CM for c =", c, 1)
        n_m = self.profile_.matrix_duels_ut[c, self.w_]
        n_s = self.profile_.n_v - n_m
        candidates = np.array(range(self.profile_.n_c))
        preferences_borda_s = self.profile_.preferences_borda_rk[np.logical_not(self.v_wants_to_help_c_[:, c]), :]
        matrix_duels_vtb_temp = (preferences_ut_to_matrix_duels_ut(preferences_borda_s))
        self.mylogm("CM: matrix_duels_vtb_temp =", matrix_duels_vtb_temp, 3)
        # More preliminary checks. It's more convenient to put them in that method, because we need
        # ``preferences_borda_s`` and ``matrix_duels_vtb_temp``.
        d_neq_c = (np.array(range(self.profile_.n_c)) != c)
        n_manip_becomes_cond = np.maximum(n_s + 1 - 2 * np.min(matrix_duels_vtb_temp[c, d_neq_c]), 0)
        self.mylogv("CM: n_manip_becomes_cond =", n_manip_becomes_cond, 3)
        self._update_sufficient(self._sufficient_coalition_size_cm, c, n_manip_becomes_cond,
                                'CM: Update sufficient_coalition_size_cm[c] = n_manip_becomes_cond =')
        if not optimize_bounds and n_m >= self._sufficient_coalition_size_cm[c]:  # pragma: no cover
            # TO DO: Investigate whether this case can actually happen.
            self._reached_uncovered_code()
            return True
        # Prevent another cond. Look at the weakest duel for ``d``, she has ``matrix_duels_vtb_temp[d, e]``. We simply
        # need that:
        # ``matrix_duels_vtb_temp[d, e] <= (n_s + n_m) / 2``
        # ``2 * max_d(min_e(matrix_duels_vtb_temp[d, e])) - n_s <= n_m``
        n_manip_prevent_cond = 0
        for d in candidates[d_neq_c]:
            e_neq_d = (np.array(range(self.profile_.n_c)) != d)
            n_prevent_d = np.maximum(2 * np.min(matrix_duels_vtb_temp[d, e_neq_d]) - n_s, 0)
            n_manip_prevent_cond = max(n_manip_prevent_cond, n_prevent_d)
        self.mylogv("CM: n_manip_prevent_cond =", n_manip_prevent_cond, 3)
        self._update_necessary(self._necessary_coalition_size_cm, c, n_manip_prevent_cond,
                               'CM: Update necessary_coalition_size_cm[c] = n_manip_prevent_cond =')
        if not optimize_bounds and self._necessary_coalition_size_cm[c] > n_m:
            return True

        # Let us work
        if self.w_ == self.irv_.w_:
            self.mylog('CM: c != self.irv_.w == self.w', 3)
            if self.cm_option == "fast":
                self.irv_.cm_option = "fast"
            elif self.cm_option == "slow":
                self.irv_.cm_option = "slow"
            else:
                self.irv_.cm_option = "exact"
            # Use IRV without bounds
            self.mylog('CM: Use IRV without bounds')
            irv_is_cm_c = self.irv_.is_cm_c_(c)
            if equal_true(irv_is_cm_c):
                suggested_path_one = self.irv_.example_path_cm_c_(c)
                self.mylogv("CM: suggested_path =", suggested_path_one, 3)
                manipulation_found = self._cm_aux_(c, n_m, suggested_path_one, preferences_borda_s,
                                                   matrix_duels_vtb_temp)
                self.mylogv("CM: manipulation_found =", manipulation_found, 3)
                if manipulation_found:
                    self._update_sufficient(self._sufficient_coalition_size_cm, c, n_m,
                                            'CM: Update sufficient_coalition_size_cm[c] = n_m =')
                    if not optimize_bounds:
                        return True
            else:
                suggested_path_one = np.zeros(self.profile_.n_c)
            if equal_false(irv_is_cm_c):
                self.mylog('CM: irv_.is_cm_c_(c) = False', 3)
                self._update_necessary(self._necessary_coalition_size_cm, c,
                                       min(n_manip_becomes_cond, max(n_m + 1, n_manip_prevent_cond)),
                                       'CM: Update necessary_coalition_size[c] =')
                if not optimize_bounds:
                    return True
            if self._sufficient_coalition_size_cm[c] == self._necessary_coalition_size_cm[c]:
                return False
            # Use IRV with bounds. Either we have not decided manipulation for ``c``, or it is decided to False and
            # ``optimize_bounds = True`` (in that second case, we can only improve ``necessary_coalition_size[c]``).
            self.mylog('CM: Use IRV with bounds')
            self.irv_.is_cm_c_with_bounds_(c)
            self._update_necessary(
                self._necessary_coalition_size_cm, c,
                min(n_manip_becomes_cond, max(self.irv_.necessary_coalition_size_cm_[c], n_manip_prevent_cond)),
                'CM: Update necessary_coalition_size[c] =')
            if self.irv_.sufficient_coalition_size_cm_[c] <= n_m:
                suggested_path_two = self.irv_.example_path_cm_c_(c)
                self.mylogv("CM: suggested_path =", suggested_path_two, 3)
                if np.array_equal(suggested_path_one, suggested_path_two):
                    self.mylog('CM: Same suggested path as before, skip computation')
                else:
                    manipulation_found = self._cm_aux_(c, n_m, suggested_path_two, preferences_borda_s,
                                                       matrix_duels_vtb_temp)
                    self.mylogv("CM: manipulation_found =", manipulation_found, 3)
                    if manipulation_found:
                        self._update_sufficient(self._sufficient_coalition_size_cm, c, n_m,
                                                'CM: Update sufficient_coalition_size_cm[c] = n_m =')
                        # We will not do better with exact (brute force) algorithm.
                        return False
        else:
            if c == self.irv_.w_:
                self.mylog('CM: c == self.irv_.w != self._w', 3)
                suggested_path = self.irv_.elimination_path_
                self.mylogv("CM: suggested_path =", suggested_path, 3)
                manipulation_found = self._cm_aux_(c, n_m, suggested_path, preferences_borda_s,
                                                   matrix_duels_vtb_temp)
                self.mylogv("CM: manipulation_found =", manipulation_found, 3)
                if manipulation_found:
                    self._update_sufficient(self._sufficient_coalition_size_cm, c, n_m,
                                            'CM: Update sufficient_coalition_size_cm[c] = n_m =')
                    # We will not do better with exact (brute force) algorithm.
                    return
            else:
                self.mylog('CM: c, self.irv_.w_ and self.w_ are all distinct', 3)
        if self.cm_option == 'exact':
            return self._cm_main_work_c_exact_(c, optimize_bounds)
