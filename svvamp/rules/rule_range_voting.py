# -*- coding: utf-8 -*-
"""
Created on 10 dec. 2018, 16:53
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
from svvamp.rules.rule import Rule
from svvamp.utils import type_checker
from svvamp.utils.cache import cached_property
from svvamp.preferences.profile import Profile


class RuleRangeVoting(Rule):
    """Range Voting with Average.

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
        >>> import svvamp
        >>> profile = svvamp.Profile(preferences_ut=[[10, 4, -10], [10, -10, 0], [-2, 10, -10], [-10, 10, 0], [-10, 4, 10]])
        >>> rule = svvamp.RuleRangeVoting(min_grade=0, max_grade=1, step_grade=0, rescale_grades=True)(profile)
        >>> print(rule.scores_)
        [0.48 0.68 0.4 ]
        >>> print(rule.candidates_by_scores_best_to_worst_)
        [1 0 2]
        >>> rule.w_
        1

    Notes
    -----
    Each voter attributes a grade to each candidate. By default, authorized grades are all numbers in the interval
    [:attr:`min_grade`, :attr:`max_grade`]. To use a discrete set of notes, modify attribute :attr:`step_grade`.

    The candidate with highest average grade wins. In case of a tie, the tied candidate with lowest index is declared
    the winner.

    Default behavior of sincere voters: voter ``v`` applies an affine transformation to her utilities
    :attr:`preferences_ut`\ ``[v, :]`` to get her grades, such that her least-liked candidate receives
    :attr:`min_grade` and her most-liked candidate receives :attr:`max_grade`. To modify this behavior, use attribute
    :attr:`rescale_grades`. For more details about the behavior of sincere voters, see :attr:`ballots`.

    * :meth:`~svvamp.Election.not_iia`:

        * If :attr:`rescale_grades` = ``False``, then Range voting always meets IIA.
        * If :attr:`rescale_grades` = ``True``, then non-polynomial or non-exact algorithms from superclass
          :class:`Rule` are used.

    * :meth:`is_cm_`, :meth:`is_icm_`, :meth:`is_im_`, :meth:`is_tm_`, :meth:`is_um_`: Exact in polynomial time.
    """

    def __init__(self, **kwargs):
        self._min_grade = None
        self._max_grade = None
        self._step_grade = None
        self._rescale_grades = None
        super().__init__(
            options_parameters={
                'max_grade': {'allowed': np.isfinite, 'default': 1},
                'min_grade': {'allowed': np.isfinite, 'default': 0},
                'step_grade': {'allowed': np.isfinite, 'default': 0},
                'rescale_grades': {'allowed': type_checker.is_bool, 'default': True},
                'im_option': {'allowed': ['exact'], 'default': 'exact'},
                'tm_option': {'allowed': ['exact'], 'default': 'exact'},
                'um_option': {'allowed': ['exact'], 'default': 'exact'},
                'icm_option': {'allowed': ['exact'], 'default': 'exact'},
                'cm_option': {'allowed': ['exact'], 'default': 'exact'}
            },
            with_two_candidates_reduces_to_plurality=False,
            # # Even if ``rescale_grades = True``, a voter who has the same utility for ``c`` and ``d`` will not vote
            # # the same in Majority Judgment and in Plurality.
            is_based_on_rk=False,
            precheck_um=False, precheck_tm=False, precheck_icm=False,
            log_identity="RANGE_VOTING", **kwargs
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
            ballots = np.zeros((self.profile_.n_v, self.profile_.n_c))
            for v in range(self.profile_.n_v):
                max_util = np.max(self.profile_.preferences_ut[v, :])
                min_util = np.min(self.profile_.preferences_ut[v, :])
                if min_util == max_util:
                    # Same utilities for all candidates
                    ballots[v, :] = np.ones(self.profile_.n_c) * (self.max_grade + self.min_grade) / 2
                else:
                    # Generic case
                    ballots[v, :] = (
                        self.min_grade +
                        (self.max_grade - self.min_grade) * (self.profile_.preferences_ut[v, :] - min_util)
                        / (max_util - min_util)
                    )
        else:
            ballots = np.clip(self.profile_.preferences_ut, self.min_grade, self.max_grade)
        # Round (or not)
        if self.step_grade != 0:
            i_lowest_rung = ceil(self.min_grade / self.step_grade)
            i_highest_rung = floor(self.max_grade / self.step_grade)
            allowed_grades = np.concatenate((
                [self.min_grade],
                np.array(range(i_lowest_rung, i_highest_rung + 1)) * self.step_grade,
                [self.max_grade]
            ))
            frontiers = (allowed_grades[0:-1] + allowed_grades[1:]) / 2
            for v in range(self.profile_.n_v):
                ballots[v, :] = allowed_grades[np.digitize(ballots[v, :], frontiers)]
        return ballots

    @cached_property
    def scores_sum_(self):
        """1d array of floats. ``scores_sum_[c]`` is the total grade of candidate ``c``.
        """
        # Note for developers: this is used internally for manipulation computations. Compared to the average grade,
        # this limits problems of float approximation, especially for IM method.
        self.mylog("Compute scores_sum_", 1)
        return np.sum(self.ballots_, 0)

    @cached_property
    def scores_(self):
        """1d array of floats. ``scores_[c]`` is the average grade of candidate ``c``.
        """
        return np.around(self.scores_sum_ / self.profile_.n_v, 12)

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
        scores_without_v = self.scores_sum_ - self.ballots_[v, :]
        w_without_v = np.argmax(np.around(scores_without_v, 12))
        new_score_w = np.around(scores_without_v[w_without_v] + self.min_grade, 12)
        for c in range(self.profile_.n_c):
            if not c_is_wanted[c]:
                continue
            if not np.isneginf(self._v_im_for_c[v, c]):
                continue
            new_score_c = np.around(scores_without_v[c] + self.max_grade, 12)
            if (c < w_without_v and new_score_c >= new_score_w) or (c > w_without_v and new_score_c > new_score_w):
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
        scores_temp = np.sum(self.ballots_[np.logical_not(self.v_wants_to_help_c_[:, c]), :], 0)
        w_temp = np.argmax(np.around(scores_temp, 12))
        sufficient_not_rounded = np.around((scores_temp[w_temp] - scores_temp[c]) / (self.max_grade - self.min_grade),
                                           12)
        self.mylogv("sufficient_not_rounded =", sufficient_not_rounded, 2)
        if sufficient_not_rounded % 1 == 0 and c > w_temp:
            self._sufficient_coalition_size_cm[c] = sufficient_not_rounded + 1
        else:
            self._sufficient_coalition_size_cm[c] = ceil(sufficient_not_rounded)
        self._necessary_coalition_size_cm[c] = self._sufficient_coalition_size_cm[c]

    # %% Trivial Manipulation (TM)

    @cached_property
    def is_tm_(self):
        return self.is_cm_

    def is_tm_c_(self, c):
        return self.is_cm_c_(c)

    @cached_property
    def candidates_tm_(self):
        return self.candidates_cm_

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

    # Since Range Voting meets IgnMC_c_ctb, the general methods are exact.


if __name__ == '__main__':
    RuleRangeVoting()(Profile(preferences_ut=np.random.randint(-5, 5, (5, 3)))).demo_()
