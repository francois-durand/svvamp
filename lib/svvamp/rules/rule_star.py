# -*- coding: utf-8 -*-
"""
Created on 18 jul. 2021, 15:41
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
from math import floor, ceil
from svvamp.rules.rule import Rule
from svvamp.utils import type_checker
from svvamp.utils.util_cache import cached_property
from svvamp.preferences.profile import Profile
from svvamp.utils.misc import preferences_ut_to_matrix_duels_ut


class RuleSTAR(Rule):
    """STAR (Scoring Then Automatic Runoff).

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
    STAR does not :attr:`meets_majority_favorite_c_ut`:

        >>> profile = Profile(preferences_ut=[
        ...     [ 0. ,  0. ,  0. , -1. , -0.5,  0. ],
        ...     [ 0.5,  0.5,  0.5,  1. ,  0. , -1. ],
        ...     [ 0. , -0.5, -1. ,  1. , -1. , -1. ],
        ... ], preferences_rk=[
        ...     [2, 0, 5, 1, 4, 3],
        ...     [3, 0, 1, 2, 4, 5],
        ...     [3, 0, 1, 4, 5, 2],
        ... ])
        >>> RuleSTAR()(profile).w_
        0
        >>> profile.majority_favorite_ut
        3

    STAR does not ::

    Notes
    -----
    Each voter attributes a grade to each candidate. By default, authorized grades are all numbers in the interval
    [:attr:`min_grade`, :attr:`max_grade`]. To use a discrete set of notes, modify attribute :attr:`step_grade`.

    The two candidates with highest average grades are selected for a virtual second round. During this second round,
    the two candidates are compared by majority voting: the score of one candidate is the number of voters who
    gave her a strictly larger grade than to the other candidate.

    Default behavior of sincere voters: voter ``v`` applies an affine transformation to her utilities
    :attr:`preferences_ut`\ ``[v, :]`` to get her grades, such that her least-liked candidate receives
    :attr:`min_grade` and her most-liked candidate receives :attr:`max_grade`. To modify this behavior, use attribute
    :attr:`rescale_grades`. For more details about the behavior of sincere voters, see :attr:`ballots`.

    * :meth:`is_cm_`, :meth:`is_icm_`, :meth:`is_im_`, :meth:`is_tm_`, :meth:`is_um_`: Exact in polynomial time.
    """

    full_name = 'Scoring Then Automatic Runoff'
    abbreviation = 'STAR'

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
            # # Even if ``rescale_grades = True``, a voter who has the same utility for ``c`` and ``d`` will not vote
            # # the same in STAR and in Plurality.
            is_based_on_rk=False,
            precheck_um=False, precheck_tm=False, precheck_icm=False,
            log_identity="STAR", **kwargs
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

    @cached_property
    def second_best_grade(self):
        """Number : second highest possible grade. Useful for manipulation algorithms.

        If step_grade = 0, then conventionally, this returns `max_grade` (whereas the actual second best possible
        grade is `max_grade` - epsilon, for any positive epsilon).
        """
        if self.step_grade == 0:
            return self.max_grade
        else:
            return self.allowed_grades[-2]

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
    def _count_ballots_(self):
        self.mylog("Count ballots", 1)
        # First round
        scores_first_round = np.sum(self.ballots_, 0)
        c = np.argmax(scores_first_round)
        d = np.argmax([-np.inf if i == c else s for i, s in enumerate(scores_first_round)])
        # Second round
        scores_second_round = np.zeros(self.profile_.n_c)
        scores_second_round[c] = np.sum(self.ballots_[:, c] > self.ballots_[:, d])
        scores_second_round[d] = np.sum(self.ballots_[:, d] > self.ballots_[:, c])
        w = c if scores_second_round[c] > scores_second_round[d] else d
        return {'w': w, 'selected_one': c, 'selected_two': d,
                'scores': np.array([scores_first_round, scores_second_round])}

    @cached_property
    def w_(self):
        return self._count_ballots_['w']

    @cached_property
    def scores_(self):
        """2d array of numbers. ``scores_[0, c]`` is the total grade of candidate ``c``. ``scores_[1, c]`` the number
        of voters who vote for ``c`` during the second round (and 0 if ``c`` is not selected for the second round).
        """
        return self._count_ballots_['scores']

    @cached_property
    def scores_average_(self):
        """2d array of floats. ``scores_[0, c]`` is the average grade of candidate ``c``. ``scores_[1, c]`` the number
        of voters who vote for ``c`` during the second round (and 0 if ``c`` is not selected for the second round).
        """
        return np.array([
            self.scores_[0, :] / self.profile_.n_v,
            self.scores_[1, :]
        ])

    @cached_property
    def candidates_by_scores_best_to_worst_(self):
        """1d array of integers. ``candidates_by_scores_best_to_worst[k]`` is the candidate with ``k``\ :sup:`th`
        best score. Finalists are sorted by their score at second round. Other candidates are sorted by their score
        at first round.
        """
        self.mylog("Count candidates_by_scores_best_to_worst", 1)
        return np.argsort(- np.max(self.scores_, 0))

    # %% Manipulation criteria of the voting system

    @cached_property
    def meets_ignmc_c_ctb(self):
        return True

    # %% Individual manipulation (IM)

    def _im_main_work_v_(self, v, c_is_wanted, nb_wanted_undecided, stop_if_true):
        other_voters = np.ones(self.profile_.n_v, dtype=bool)
        other_voters[v] = False
        scores_first_round_without_v = self.scores_[0, :] - self.ballots_[v, :]
        matrix_duels_without_v = preferences_ut_to_matrix_duels_ut(self.ballots_[other_voters, :])
        for c in range(self.profile_.n_c):
            if not c_is_wanted[c]:
                continue
            if not np.isneginf(self._v_im_for_c[v, c]):
                continue
            scores_first_round_c = np.around(scores_first_round_without_v[c] + self.max_grade, 12)
            scores_first_round_s2 = np.around(scores_first_round_without_v[c] + self.min_grade, 12)
            if scores_first_round_c < scores_first_round_s2:
                # Then c cannot go to second round
                self._v_im_for_c[v, c] = False
            else:
                # Case 1: c in {selected_one_without_v, selected_two_without_v} ==> c can go to second round.
                # Case 2: c not in {s1, s2} ==> since c can do better than s2, c can go to second round.
                # => Anyway, c might go to second round.
                n_m_needed_second_round_against_d = np.maximum(
                    matrix_duels_without_v[:, c] - matrix_duels_without_v[c, :]
                    + (np.array(range(self.profile_.n_c)) < c),
                    0
                )
                n_m_needed_second_round_against_d[c] = np.nan
                manipulation_found = False
                # First try
                d_loses_already_against_c = np.where(n_m_needed_second_round_against_d == 0)[0]
                d = max(d_loses_already_against_c, key=scores_first_round_without_v.__getitem__)
                scores_first_round = scores_first_round_without_v.copy()
                scores_first_round[c] += self.max_grade - self.min_grade
                scores_first_round[d] += self.max_grade - self.min_grade
                selected_one = np.argmax(scores_first_round)
                selected_two = np.argmax([-np.inf if i == selected_one else s
                                          for i, s in enumerate(scores_first_round)])
                if {selected_one, selected_two} == {c, d}:
                    manipulation_found = True
                else:
                    # Second try
                    d_can_lose_against_c_by_manipulation = np.where(n_m_needed_second_round_against_d == 1)[0]
                    d = max(d_can_lose_against_c_by_manipulation, key=scores_first_round_without_v.__getitem__)
                    scores_first_round = scores_first_round_without_v.copy()
                    scores_first_round[c] += self.max_grade - self.min_grade
                    scores_first_round[d] += self.second_best_grade - self.min_grade  # Minus epsilon if step_grade == 0
                    selected_one = np.argmax(scores_first_round)
                    selected_two = np.argmax([-np.inf if i == selected_one else s
                                              for i, s in enumerate(scores_first_round)])
                    if {selected_one, selected_two} == {c, d}:
                        manipulation_found = True
                    # ... but there is an exception:
                    if self.step_grade == 0 and scores_first_round[d] == scores_first_round[selected_two]:
                        selected_three = np.argmax([-np.inf if i in {selected_one, selected_two} else s
                                                    for i, s in enumerate(scores_first_round)])
                        if scores_first_round[selected_three] == scores_first_round[d]:
                            manipulation_found = False
                if manipulation_found:
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

    def _cm_main_work_c_(self, c, optimize_bounds):
        ballots_s = self.ballots_[np.logical_not(self.v_wants_to_help_c_[:, c]), :]
        scores_first_round_s = np.sum(ballots_s, 0)
        for d in range(self.profile_.n_c):
            # We want to put c and d at second round, and make c win.
            if d == c:
                continue
            # How many manipulator to ensure that c wins the second round?
            score_second_round_c_d = np.sum(ballots_s[:, c] > ballots_s[:, d])
            score_second_round_d_c = np.sum(ballots_s[:, d] > ballots_s[:, c])
            n_2 = max(score_second_round_d_c - score_second_round_c_d + int(d < c), 0)
            # Consequences on first round
            scores_first_round = scores_first_round_s.copy().astype(float)
            scores_first_round[c] += n_2 * self.max_grade
            scores_first_round[d] += n_2 * self.second_best_grade  # Exactly if step_grade > 0, minus epsilon if == 0
            if self.profile_.n_c == 2:
                self._update_sufficient(self._sufficient_coalition_size_cm, c, n_2,
                                        'CM: Update sufficient_coalition_size_cm[c] = n_m =')
                continue
            # How many additional manipulators to win the first round?
            score_c = scores_first_round[c]
            score_d = scores_first_round[d]
            scores_first_round[c] = -np.inf
            scores_first_round[d] = -np.inf
            best_challenger = np.argmax(scores_first_round)
            score_challenger = scores_first_round[best_challenger]
            n_c_not_rounded = np.around((score_challenger - score_c) / (self.max_grade - self.min_grade), 12)
            if n_c_not_rounded % 1 == 0 and c > best_challenger:
                n_c = n_c_not_rounded + 1
            else:
                n_c = ceil(n_c_not_rounded)
            n_d_not_rounded = np.around((score_challenger - score_d) / (self.max_grade - self.min_grade), 12)
            if n_d_not_rounded % 1 == 0 and ((self.step_grade == 0 and n_2 > 0) or d > best_challenger):
                n_d = n_d_not_rounded + 1
            else:
                n_d = ceil(n_d_not_rounded)
            n_1 = max(n_c, n_d, 0)
            # Total number of manipulators
            self._update_sufficient(self._sufficient_coalition_size_cm, c, n_1 + n_2,
                                    'CM: Update sufficient_coalition_size_cm[c] = n_m =')
        # Final conclusion: we know that this algorithm is exact.
        self._necessary_coalition_size_cm[c] = self._sufficient_coalition_size_cm[c]

    # %% Trivial Manipulation (TM)

    def _tm_main_work_c_(self, c):
        # Manipulators give min_grade to all candidates, except max_grade for c.
        preferences_test = np.copy(self.profile_.preferences_ut)
        preferences_test[self.v_wants_to_help_c_[:, c], :] = self.min_grade
        preferences_test[self.v_wants_to_help_c_[:, c], c] = self.max_grade
        w_test = self._copy(profile=Profile(preferences_ut=preferences_test, sort_voters=False)).w_
        self._candidates_tm[c] = (w_test == c)

    # %% Unison Manipulation (UM)

    def _um_main_work_c_(self, c):
        n_m = self.profile_.matrix_duels_ut[c, self.w_]
        ballots_s = self.ballots_[np.logical_not(self.v_wants_to_help_c_[:, c]), :]
        scores_first_round_s = np.sum(ballots_s, 0)
        self._candidates_um[c] = False  # until otherwise proven
        for d in range(self.profile_.n_c):
            # We want to put c and d at second round, and make c win.
            if d == c:
                continue
            # How many manipulator to ensure that c wins the second round?
            score_second_round_c_d = np.sum(ballots_s[:, c] > ballots_s[:, d])
            score_second_round_d_c = np.sum(ballots_s[:, d] > ballots_s[:, c])
            n_2 = max(score_second_round_d_c - score_second_round_c_d + int(d < c), 0)
            # Consequences on first round
            scores_first_round = scores_first_round_s.copy().astype(float)
            scores_first_round[c] += n_2 * self.max_grade
            scores_first_round[d] += n_2 * self.second_best_grade  # Exactly if step_grade > 0, minus epsilon if == 0
            if self.profile_.n_c == 2:
                if n_2 <= n_m:
                    self._candidates_um[c] = True
                    return
            # How many additional manipulators to win the first round?
            score_c = scores_first_round[c]
            score_d = scores_first_round[d]
            scores_first_round[c] = -np.inf
            scores_first_round[d] = -np.inf
            best_challenger = np.argmax(scores_first_round)
            score_challenger = scores_first_round[best_challenger]
            n_c_not_rounded = np.around((score_challenger - score_c) / (self.max_grade - self.min_grade), 12)
            if n_c_not_rounded % 1 == 0 and c > best_challenger:
                n_c = n_c_not_rounded + 1
            else:
                n_c = ceil(n_c_not_rounded)
            # N.B.: the following "if" clause is the main difference with CM algorithm.
            if n_2 > 0:
                # Due to UM, all manipulators need to put second-best grade to `d`.
                n_d_not_rounded = np.around((score_challenger - score_d) / (self.second_best_grade - self.min_grade),
                                            12)
            else:
                # No constraint on `d`'s grade, so we can use max_grade.
                n_d_not_rounded = np.around((score_challenger - score_d) / (self.max_grade - self.min_grade),
                                            12)
            if n_d_not_rounded % 1 == 0 and ((self.step_grade == 0 and n_2 > 0) or d > best_challenger):
                n_d = n_d_not_rounded + 1
            else:
                n_d = ceil(n_d_not_rounded)
            n_1 = max(n_c, n_d, 0)
            # Total number of manipulators
            if n_1 + n_2 <= n_m:
                self._candidates_um[c] = True
                return

    # %% Ignorant-Coalition Manipulation (ICM)

    # Since STAR meets IgnMC_c_ctb, the general methods are exact.
