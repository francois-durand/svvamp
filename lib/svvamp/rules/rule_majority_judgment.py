# -*- coding: utf-8 -*-
"""
Created on 10 dec. 2018, 13:54
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
from math import floor, ceil
from fractions import Fraction
from svvamp.rules.rule import Rule
from svvamp.utils import type_checker
from svvamp.utils.util_cache import cached_property
from svvamp.utils.pseudo_bool import equal_false
from svvamp.preferences.profile import Profile


class RuleMajorityJudgment(Rule):
    """Majority Judgment.

    Parameters
    ----------
    min_grade : number
        Minimal grade allowed.
    max_grade : number
        Maximal grade allowed.
    step_grade : number
        Interval between two consecutive allowed grades.

        * If ``step_grade = 0``, all grades in the interval [:attr:`min_grade`, :attr:`max_grade`] are allowed
          ('continuous' set of grades).
        * If ``step_grade > 0``, authorized grades are the multiples of :attr:`step_grade` lying in the interval
          [:attr:`min_grade`, :attr:`max_grade`]. In addition, the grades :attr:`min_grade` and :attr:`max_grade` are
          always authorized, even if they are not multiples of :attr:`step_grade`.

    rescale_grades : bool
        Whether sincere voters rescale their utilities to produce grades.

        * If ``rescale_grades`` = ``True``, then each sincere voter ``v`` applies an affine transformation to send
          her utilities into the interval [:attr:`min_grade`, :attr:`max_grade`].
        * If ``rescale_grades`` = ``False``, then each sincere voter ``v`` clips her utilities into the interval
          [:attr:`min_grade`, :attr:`max_grade`].

        See :attr:`ballots` for more details.

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
        >>> rule = RuleMajorityJudgment()(profile)
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
        [[1.   0.5  0.  ]
         [1.   0.   0.75]
         [1.   1.   0.  ]
         [0.5  0.   1.  ]
         [0.   0.   1.  ]]
        scores =
        [[ 1.    0.    0.75]
         [-2.    2.   -2.  ]]
        candidates_by_scores_best_to_worst
        [0 2 1]
        scores_best_to_worst
        [[ 1.    0.75  0.  ]
         [-2.   -2.    2.  ]]
        w = 0
        score_w = [ 1. -2.]
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
        is_based_on_rk =  False
        is_based_on_ut_minus1_1 =  True
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
        Condorcet_c_ut_abs_ctb (False)     ==>     Condorcet_ut_abs_c (False)
         ||                            ||       ||                        ||
         ||                            V        V                         ||
         ||       maj_fav_c_rk_ctb (False) ==> maj_fav_c_rk (False)       ||
         ||           ||                                       ||         ||
         V            V                                        V          V
        majority_favorite_c_ut_ctb (False) ==> majority_favorite_c_ut (False)
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
        log_cm: cm_option = exact
        candidates_cm =
        [0. 0. 0.]
        necessary_coalition_size_cm =
        [0. 3. 3.]
        sufficient_coalition_size_cm =
        [0. 3. 3.]

    Notes
    -----
    Each voter attributes a grade to each candidate. By default, authorized grades are all numbers in the interval
    [:attr:`min_grade`, :attr:`max_grade`]. To use a discrete set of notes, modify attribute :attr:`step_grade`.

    .. note::

        Majority Judgement, as promoted by its authors, uses a discrete set of non-numerical grades. For our purposes,
        using a discrete set of numerical grades (:attr:`step_grade` > 0) is isomorphic to this voting system. In
        contrast, using a continuous set of grades (:attr:`step_grade` = 0) is a variant of this voting system, which
        has the advantage of being canonical, in the sense that there is no need to choose the number of authorized
        grades more or less arbitrarily.

    The candidate with highest median grade wins. For the tie-breaking rule, see :attr:`scores`.

    Default behavior of sincere voters: voter ``v`` applies an affine transformation to her utilities
    :attr:`preferences_ut`\ ``[v, :]`` to get her grades, such that her least-liked candidate receives
    :attr:`min_grade` and her most-liked candidate receives :attr:`max_grade`. To modify this behavior, use attribute
    :attr:`rescale_grades`. For more details about the behavior of sincere voters, see :attr:`ballots`.

    * :meth:`not_iia`:

        * If :attr:`rescale_grades` = ``False``, then Majority Judgment always meets IIA.
        * If :attr:`rescale_grades` = ``True``, then non-polynomial or non-exact algorithms from superclass
          :class:`Rule` are used.

    * :meth:`is_cm_`, :meth:`is_icm_`, :meth:`is_im_`, :meth:`is_tm_`, :meth:`is_um_`: Exact in polynomial time.

    References
    ----------
    Majority Judgment : Measuring, Ranking, and Electing. Michel Balinski and Rida Laraki, 2010.
    """

    full_name = 'Majority Judgment'
    abbreviation = 'MJ'

    options_parameters = Rule.options_parameters.copy()
    options_parameters.update({
        'max_grade': {'allowed': np.isfinite, 'default': 1},
        'min_grade': {'allowed': np.isfinite, 'default': 0},
        'step_grade': {'allowed': np.isfinite, 'default': 0},
        'rescale_grades': {'allowed': type_checker.is_bool, 'default': True},
        'im_option': {'allowed': ['exact'], 'default': 'exact'},
        'tm_option': {'allowed': ['exact'], 'default': 'exact'},
        'um_option': {'allowed': ['exact'], 'default': 'exact'},
        'icm_option': {'allowed': ['exact'], 'default': 'exact'},
        'cm_option': {'allowed': ['exact'], 'default': 'exact'}
    })

    def __init__(self, **kwargs):
        self._min_grade = None
        self._max_grade = None
        self._step_grade = None
        self._rescale_grades = None
        super().__init__(
            with_two_candidates_reduces_to_plurality=False,
            # Even if ``rescale_grades = True``, a voter who has the same utility for ``c`` and ``d`` will not vote
            # the same in Majority Judgment and in Plurality.
            is_based_on_rk=False,
            precheck_um=False, precheck_tm=True, precheck_icm=False,
            log_identity="MAJORITY_JUDGMENT", **kwargs
        )

    # %% Setting the parameters

    @property
    def min_grade(self):
        return self._min_grade

    @min_grade.setter
    def min_grade(self, value):
        if self._min_grade == value:
            return
        if self.options_parameters['min_grade']['allowed'](value):
            self.mylogv("Setting min_grade =", value, 1)
            self._min_grade = value
            self._result_options['min_grade'] = value
            self.delete_cache()
        else:
            raise ValueError("Unknown value for min_grade: " + format(value))

    @property
    def max_grade(self):
        return self._max_grade

    @max_grade.setter
    def max_grade(self, value):
        if self._max_grade == value:
            return
        if self.options_parameters['max_grade']['allowed'](value):
            self.mylogv("Setting max_grade =", value, 1)
            self._max_grade = value
            self._result_options['max_grade'] = value
            self.delete_cache()
        else:
            raise ValueError("Unknown value for max_grade: " + format(value))

    @property
    def step_grade(self):
        return self._step_grade

    @step_grade.setter
    def step_grade(self, value):
        if self._step_grade == value:
            return
        if self.options_parameters['step_grade']['allowed'](value):
            self.mylogv("Setting step_grade =", value, 1)
            self._step_grade = value
            self._result_options['step_grade'] = value
            self.delete_cache()
        else:
            raise ValueError("Unknown value for step_grade: " + format(value))

    @property
    def rescale_grades(self):
        return self._rescale_grades

    @rescale_grades.setter
    def rescale_grades(self, value):
        if self._rescale_grades == value:
            return
        if self.options_parameters['rescale_grades']['allowed'](value):
            self.mylogv("Setting rescale_grades =", value, 1)
            self._rescale_grades = value
            self._result_options['rescale_grades'] = value
            self.delete_cache()
        else:
            raise ValueError("Unknown value for rescale_grades: " + format(value))

    @cached_property
    def allowed_grades(self):
        """List or None. If ``step_grade`` is positive, the list of authorized grades. If ``step_grade`` is zero
        (continuous set of grades), then None.
        """
        if self.step_grade == 0:
            return None
        else:
            i_lowest_rung = floor(self.min_grade / self.step_grade) + 1
            i_highest_rung = ceil(self.max_grade / self.step_grade) - 1
            return np.concatenate((
                [self.min_grade],
                np.array(range(i_lowest_rung, i_highest_rung + 1)) * self.step_grade,
                [self.max_grade]
            ))

    # %% Counting the ballots

    @cached_property
    def ballots_(self):
        """2d array of integers. ``ballots[v, c]`` is the grade attributed by voter ``v`` to candidate ``c`` (when
        voting sincerely). The following process is used.

        1. Convert utilities into grades in the interval [:attr:`min_grade`, :attr:`max_grade`].

            * If :attr:`rescale_grades` = ``True``, then each voter ``v`` applies an affine transformation to
              :attr:`preferences_ut`\ ``[v, :]`` such that her least-liked candidate receives :attr:`min_grade` and
              her most-liked candidate receives :attr:`max_grade`. Exception: if she is indifferent between all
              candidates, then she attributes (:attr:`min_grade` + :attr:`max_grade`) / 2 to all of them.

            * If :attr:`rescale_grades` = ``False``, then each voter ``v`` clips her utilities into the interval
              [:attr:`min_grade`, :attr:`max_grade`]: each utility greater than :attr:`max_grade` (resp. lower than
              :attr:`min_grade`) becomes :attr:`max_grade` (resp. :attr:`min_grade`).

        2. If :attr:`step_grades` > 0 (discrete set of grades), round each grade to the closest authorized grade.
        """
        self.mylog("Compute ballots", 1)
        # Rescale (or not)
        if self.rescale_grades:
            # Multiplicative renormalization
            max_util = np.max(self.profile_.preferences_ut, axis=1)
            min_util = np.min(self.profile_.preferences_ut, axis=1)
            delta_util = max_util - min_util
            ballots = np.divide(
                self.profile_.preferences_ut * (self.max_grade - self.min_grade),
                delta_util[:, np.newaxis],
                out=np.zeros(self.profile_.preferences_ut.shape),
                where=delta_util[:, np.newaxis] != 0
            )  # `out` and `where` ensure that when dividing by zero, the result will be zero.
            # Additive renormalization
            middle_grade = (np.max(ballots, axis=1) + np.min(ballots, axis=1)) / 2
            ballots += (self.max_grade - self.min_grade) / 2 - middle_grade[:, np.newaxis]
        else:
            ballots = np.clip(self.profile_.preferences_ut, self.min_grade, self.max_grade)
        # Round (or not)
        if self.step_grade != 0:
            if self.min_grade % 1 == 0 and self.max_grade % 1 == 0 and self.step_grade % 1 == 0:
                # There is a faster version for this (common) use case.
                ballots = np.array(np.rint(ballots), dtype=int)
            else:
                frontiers = (self.allowed_grades[0:-1] + self.allowed_grades[1:]) / 2
                for v in range(self.profile_.n_v):
                    ballots[v, :] = self.allowed_grades[np.digitize(ballots[v, :], frontiers)]
        return ballots

    @cached_property
    def scores_(self):
        """2d array of integers.

        ``scores[0, c]`` is the median grade of candidate ``c``.

        Let us note ``p`` (resp. ``q``) the number of voter who attribute to ``c`` a grade higher (resp. lower) than
        the median. If ``p`` > ``q``, then ``scores[1, c]`` = ``p``. Otherwise, ``scores[1, c]`` = ``-q``.
        """
        self.mylog("Compute scores", 1)
        scores = np.zeros((2, self.profile_.n_c))
        scores[0, :] = np.median(self.ballots_, 0)
        for c in range(self.profile_.n_c):
            p = np.sum(self.ballots_[:, c] > scores[0, c])
            q = np.sum(self.ballots_[:, c] < scores[0, c])
            scores[1, c] = - q if q >= p else p
        return scores

    @cached_property
    def candidates_by_scores_best_to_worst_(self):
        """1d array of integers. ``candidates_by_scores_best_to_worst[k] is the candidate ranked ``k``\ :sup:`th`.

        Candidates are sorted lexicographically by their median (:attr:`scores`\ ``[0, c]``) then their ``p`` or ``-q``
        (:attr:`scores`\ ``[1, c]``). If there is still a  tie, the tied candidate with lower index is favored.
        """
        self.mylog("Compute candidates_by_scores_best_to_worst", 1)
        # N.B. : ``np.array(range(self.profile_.C))`` below is the tie-breaking term by lowest index.
        return np.lexsort((np.array(range(self.profile_.n_c))[::-1], self.scores_[1, :], self.scores_[0, :]))[::-1]

    @cached_property
    def w_(self):
        """Integer (winning candidate).

        Candidates are sorted lexicographically by their median (:attr:`scores`\ ``[0, c]``) then their ``p`` or
        ``-q`` (:attr:`scores`\ ``[1, c]``). If there is still a tie, the tied candidate with lower index is declared
        the winner.
        """
        self.mylog("Compute winner", 1)
        return self.candidates_by_scores_best_to_worst_[0]

    # %% Manipulation criteria of the voting system

    @cached_property
    def meets_ignmc_c_ctb(self):
        return True

    # %% Independence of Irrelevant Alternatives (IIA)

    @property
    def meets_iia(self):
        return not self.rescale_grades

    @property
    def is_based_on_ut_minus1_1(self):
        return self._rescale_grades

    @meets_iia.setter
    def meets_iia(self, value):
        pass

    @is_based_on_ut_minus1_1.setter
    def is_based_on_ut_minus1_1(self, value):
        pass

    # %% Individual manipulation (IM)

    def _im_main_work_v_(self, v, c_is_wanted, nb_wanted_undecided, stop_if_true):
        """
            >>> profile = Profile([
            ...     [ 1. , -0.5,  0. ],
            ...     [ 0.5,  1. , -1. ],
            ...     [-0.5,  0.5, -1. ],
            ...     [ 1. ,  0. ,  1. ],
            ...     [-1. , -0.5,  1. ],
            ... ])
            >>> rule = RuleMajorityJudgment()(profile)
            >>> rule.is_im_c_(1)
            True
        """
        ballots_test = np.copy(self.ballots_)
        ballots_test[v, :] = self.min_grade
        scores_v_is_evil = np.zeros((2, self.profile_.n_c))
        scores_v_is_evil[0, :] = np.median(ballots_test, 0)
        for c in range(self.profile_.n_c):
            p = np.sum(ballots_test[:, c] > scores_v_is_evil[0, c])
            q = np.sum(ballots_test[:, c] < scores_v_is_evil[0, c])
            scores_v_is_evil[1, c] = - q if q >= p else p
        w_temp = max(range(self.profile_.n_c), key=lambda d: scores_v_is_evil[:, d].tolist())

        ballots_test[v, :] = self.max_grade
        score_v_is_nice_c = np.zeros(2)
        for c in range(self.profile_.n_c):
            if not c_is_wanted[c]:
                continue
            if not np.isneginf(self._v_im_for_c[v, c]):
                continue
            score_v_is_nice_c[0] = np.median(ballots_test[:, c])
            p = np.sum(ballots_test[:, c] > score_v_is_nice_c[0])
            q = np.sum(ballots_test[:, c] < score_v_is_nice_c[0])
            score_v_is_nice_c[1] = - q if q >= p else p
            if ([score_v_is_nice_c[0], score_v_is_nice_c[1], - c]
                    >= [scores_v_is_evil[0, w_temp], scores_v_is_evil[1, w_temp], - w_temp]):
                self._v_im_for_c[v, c] = True
                self._candidates_im[c] = True
                self._voters_im[v] = True
                self._is_im = True
                self.mylog("IM found", 3)
                if stop_if_true:
                    return
            else:
                self._v_im_for_c[v, c] = False
            nb_wanted_undecided -= 1
            if nb_wanted_undecided == 0:
                return

    # %% Coalition Manipulation (CM)

    def _cm_preliminary_checks_c_subclass_(self, c, optimize_bounds):
        if equal_false(self.is_tm_c_(c)):
            self._update_necessary(
                self._necessary_coalition_size_cm, c, self.profile_.matrix_duels_ut[c, self.w_] + 1,
                'CM: Preliminary checks: not TM => \n    _necessary_coalition_size_cm[c] = n_m + 1 =')

    def _cm_main_work_c_(self, c, optimize_bounds):
        """
            >>> profile = Profile([
            ...     [ 1. , -0.5,  0. ],
            ...     [ 0.5,  1. , -1. ],
            ...     [-0.5,  0.5, -1. ],
            ...     [ 1. ,  0. ,  1. ],
            ...     [-1. , -0.5,  1. ],
            ... ])
            >>> rule = RuleMajorityJudgment()(profile)
            >>> rule.candidates_cm_
            array([0., 1., 0.])
            >>> rule.necessary_coalition_size_cm_
            array([0., 3., 2.])
            >>> rule.sufficient_coalition_size_cm_
            array([0., 3., 2.])
        """
        preferences_ut_s = self.profile_.preferences_ut[np.logical_not(self.v_wants_to_help_c_[:, c]), :]
        n_s = self.profile_.n_v - self.profile_.matrix_duels_ut[c, self.w_]
        ballot_manip = np.ones(self.profile_.n_c) * self.min_grade
        ballot_manip[c] = self.max_grade
        n_m_inf = 0
        n_m_sup = n_s + 1
        # Loop invariant: CM always possible n_m_sup manipulators, impossible for n_m_inf manipulators
        while n_m_sup - n_m_inf > 1:
            n_m = (n_m_inf + n_m_sup) // 2
            preferences_ut_test = np.concatenate((
                preferences_ut_s,
                np.outer(np.ones(n_m), ballot_manip)
            ))
            profile_test = Profile(preferences_ut=preferences_ut_test, sort_voters=False)
            rule_test = self._copy(profile_test)
            winner_test = rule_test.w_
            if winner_test == c:
                n_m_sup = n_m
            else:
                n_m_inf = n_m
        self._sufficient_coalition_size_cm[c] = n_m_sup
        self._necessary_coalition_size_cm[c] = n_m_sup

    # def _cm_main_work_c_old_(self, c, optimize_bounds):
    #     # TODO: One day, I could combine the dichotomy of the new method with the approach of the old method.
    #     # In fact, in sorted_sincere, there will be sincere voters and one manipulator (so that median and other
    #     # stuff is always defined). Grades for ``c`` are sorted in ascending order along ``c``'s column. Grades for
    #     # other candidates are sorted in descending order. This way, when adding one manipulator, the median is
    #     # shifted down by a half-index.
    #     ballot_manip = np.ones(self.profile_.n_c) * self.min_grade
    #     ballot_manip[c] = self.max_grade
    #     sorted_sincere = np.sort(np.concatenate((
    #         self.ballots_[np.logical_not(self.v_wants_to_help_c_[:, c]), :], [ballot_manip]
    #     ), 0), 0)[::-1, :]
    #     sorted_sincere[:, c] = sorted_sincere[::-1, c]
    #     self.mylogm("CM: sorted_sincere + 1 manipulator =", sorted_sincere, 3)
    #     medians = np.median(sorted_sincere[:-1, :], 0)
    #     self.mylogm("CM: medians (sincere only) =", medians, 3)
    #     p = np.sum(sorted_sincere[:-1, :] > medians, 0)
    #     self.mylogm("CM: p (sincere only) =", p, 3)
    #     q = np.sum(sorted_sincere[:-1, :] < medians, 0)
    #     self.mylogm("CM: q (sincere only) =", q, 3)
    #
    #     n_s = self.profile_.n_v - self.profile_.matrix_duels_ut[c, self.w_]
    #     self.mylogv("CM: n_s =", n_s, 3)
    #     n_m = 0
    #     i_median = Fraction(n_s - 1, 2)
    #     cm_successful = False
    #     while not cm_successful:
    #         n_m += 1
    #         i_median += Fraction(1, 2)
    #         self.mylogv("CM: n_m =", n_m, 3)
    #
    #         # Computing c's scores
    #         if i_median % 1 == 0:
    #             median_c = sorted_sincere[int(i_median), c]
    #         else:
    #             median_c = (sorted_sincere[floor(i_median), c] + sorted_sincere[ceil(i_median), c]) / 2
    #         if median_c == medians[c]:
    #             if median_c != self.max_grade:
    #                 p[c] += 1
    #         else:
    #             medians[c] = median_c
    #             q[c] = ceil(i_median)
    #             p[c] = np.sum(sorted_sincere[:-1, c] > median_c) + n_m
    #         if q[c] >= p[c]:
    #             score_c = [medians[c], -q[c], -c]
    #         else:
    #             score_c = [medians[c], p[c], -c]
    #         self.mylogv("CM: score_c =", score_c, 3)
    #
    #         cm_successful = True
    #         for d in range(self.profile_.n_c):
    #             if d == c:
    #                 continue
    #
    #             # Computing d's scores
    #             if i_median % 1 == 0:
    #                 median_d = sorted_sincere[int(i_median), d]
    #             else:
    #                 median_d = (sorted_sincere[floor(i_median), d] + sorted_sincere[ceil(i_median), d]) / 2
    #             if median_d == medians[d]:
    #                 if median_d != self.min_grade:
    #                     q[d] += 1
    #             else:
    #                 medians[d] = median_d
    #                 p[d] = ceil(i_median)
    #                 q[d] = np.sum(sorted_sincere[:-1, d] < median_d) + n_m
    #             if q[d] >= p[d]:
    #                 score_d = [medians[d], -q[d], -d]
    #             else:
    #                 score_d = [medians[d], p[d], -d]
    #             self.mylogv("CM: score_d =", score_d, 3)
    #
    #             # Does manipulation work?
    #             if score_d > score_c:
    #                 cm_successful = False
    #
    #     self._sufficient_coalition_size_cm[c] = n_m
    #     self._necessary_coalition_size_cm[c] = n_m

    # %% Trivial Manipulation (TM)

    def _tm_main_work_c_(self, c):
        ballots_test = np.copy(self.ballots_)
        ballots_test[self.v_wants_to_help_c_[:, c], :] = self.min_grade
        ballots_test[self.v_wants_to_help_c_[:, c], c] = self.max_grade
        scores_test = np.zeros((2, self.profile_.n_c))
        scores_test[0, :] = np.median(ballots_test, 0)
        for d in range(self.profile_.n_c):
            p = np.sum(ballots_test[:, d] > scores_test[0, d])
            q = np.sum(ballots_test[:, d] < scores_test[0, d])
            scores_test[1, d] = - q if q >= p else p
        candidates_by_scores_best_to_worst_test = np.lexsort((
            np.array(range(self.profile_.n_c))[::-1], scores_test[1, :], scores_test[0, :]
        ))[::-1]
        w_test = candidates_by_scores_best_to_worst_test[0]
        self._candidates_tm[c] = (w_test == c)

    # %% Unison Manipulation (UM)

    @cached_property
    def is_um_(self):
        return self.is_cm_

    def is_um_c_(self, c):
        return self.is_cm_c_(c)

    @cached_property
    def candidates_um_(self):
        return self.candidates_cm_

    # %% Ignorant-Coalition Manipulation (ICM)

    # Since Majority Judgment meets IgnMC_c_tb, the general methods are exact.
