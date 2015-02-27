# -*- coding: utf-8 -*-
"""
Created on Mon Oct  6 10:41:11 2014
Copyright François Durand 2014, 2015
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

from svvamp.Preferences.Population import Population
from svvamp.Utils import TypeChecker
from svvamp.VotingSystems.ElectionResult import ElectionResult


class RangeVotingAverageResult(ElectionResult):
    """Results of an election using Range Voting with Average.

    Each voter attributes a grade to each candidate, between min_grade and
    max_grade. The candidate with highest average grade wins. In case of a
    tie, the candidate with lowest index is declared the winner.

    See the documentation of options min_grade, max_grade, step_grade and
    rescale_grades.
    """

    _options_parameters = ElectionResult._options_parameters.copy()
    _options_parameters['max_grade'] = {'allowed': np.isfinite, 'default': 1}
    _options_parameters['min_grade'] = {'allowed': np.isfinite, 'default': 0}
    _options_parameters['step_grade'] = {'allowed': np.isfinite, 'default': 0}
    _options_parameters['rescale_grades'] = {'allowed': TypeChecker.is_bool,
                                             'default': True}

    def __init__(self, population, **kwargs):
        super().__init__(population, **kwargs)
        self._log_identity = "RANGE_VOTING_AVERAGE_RESULT"

    #%% Setting the parameters

    @property
    def max_grade(self):
        """Number. Maximal grade allowed.
        """
        return self._max_grade

    @max_grade.setter
    def max_grade(self, value):
        self._max_grade = value
        self._forget_all_computations()
        
    @property
    def min_grade(self):
        """Number. Minimal grade allowed.
        """
        return self._min_grade

    @min_grade.setter
    def min_grade(self, value):
        self._min_grade = value
        self._forget_all_computations()
        
    @property
    def step_grade(self):
        """Number. Interval between two consecutive allowed grades.

        If ``step_grade = 0``, all grades
        in the interval [:attr:`~svvamp.RangeVotingAverage.min_grade`,
        :attr:`~svvamp.RangeVotingAverage.max_grade`] are allowed
        ('continuous' set of grades).

        If ``step_grade > 0``, authorized grades are the multiples of
        :attr:`~svvamp.RangeVotingAverage.step_grade` lying in
        the interval [:attr:`~svvamp.RangeVotingAverage.min_grade`,
        :attr:`~svvamp.RangeVotingAverage.max_grade`]. In addition, the grades
        :attr:`~svvamp.RangeVotingAverage.min_grade` and
        :attr:`~svvamp.RangeVotingAverage.max_grade` are always authorized,
        even if they are not multiples of
        :attr:`~svvamp.RangeVotingAverage.step_grade`.
        """
        return self._step_grade

    @step_grade.setter
    def step_grade(self, value):
        self._step_grade = value
        self._forget_all_computations()
        
    @property
    def rescale_grades(self):
        """Boolean. Whether sincere voters rescale their utilities to produce
        grades.

        If ``rescale_grades`` = ``True``, then each sincere voter ``v``
        applies an affine transformation to send her utilities into the
        interval [:attr:`~svvamp.RangeVotingAverage.min_grade`,
        :attr:`~svvamp.RangeVotingAverage.max_grade`].

        If ``rescale_grades`` = ``False``, then each sincere voter ``v``
        clips her utilities into the interval
        [:attr:`~svvamp.RangeVotingAverage.min_grade`,
        :attr:`~svvamp.RangeVotingAverage.max_grade`].

        See :attr:`~svvamp.RangeVotingAverage.ballots` for more details.
        """
        return self._rescale_grades

    @rescale_grades.setter
    def rescale_grades(self, value):
        self._rescale_grades = value
        self._forget_all_computations()

    #%% Counting the ballots

    @property
    def ballots(self):
        """2d array of integers. ``ballots[v, c]`` is the grade attributed by
        voter ``v`` to candidate ``c`` (when voting sincerely). The
        following process is used.

        1.  Convert utilities into grades in interval
            [:attr:`~svvamp.RangeVotingAverage.min_grade`,
            :attr:`~svvamp.RangeVotingAverage.max_grade`].

            *   If :attr:`~svvamp.RangeVotingAverage.rescale_grades` =
                ``True``, then each voter ``v`` applies an affine
<<<<<<< HEAD
                transformation to her utilities
                :attr:`~svvamp.Population.preferences_ut`\ ``[v, :]``
=======
                transformation to
                :attr:`~svvamp.Population.preferences_utilities`\ ``[v, :]``
>>>>>>> origin/master
                such that her least-liked candidate receives
                :attr:`~svvamp.RangeVotingAverage.min_grade` and her
                most-liked candidate receives
                :attr:`~svvamp.RangeVotingAverage.max_grade`.

                Exception: if she is indifferent between all candidates,
                then she attributes
                (:attr:`~svvamp.RangeVotingAverage.min_grade` +
                :attr:`~svvamp.RangeVotingAverage.max_grade`) / 2 to all of
                them.

            *   If :attr:`~svvamp.RangeVotingAverage.rescale_grades` =
                ``False``, then each voter ``v`` clips her utilities into the
                interval [:attr:`~svvamp.RangeVotingAverage.min_grade`,
                :attr:`~svvamp.RangeVotingAverage.max_grade`]:
                each utility greater than
                :attr:`~svvamp.RangeVotingAverage.max_grade` (resp. lower than
                :attr:`~svvamp.RangeVotingAverage.min_grade`) becomes
                :attr:`~svvamp.RangeVotingAverage.max_grade` (resp.
                :attr:`~svvamp.RangeVotingAverage.min_grade`).

        2.  If :attr:`~svvamp.RangeVotingAverage.step_grades` > 0 (discrete
            set of grades), round each grade to the closest authorized grade.
        """
        if self._ballots is None:
            self._mylog("Compute ballots", 1)
            # Rescale (or not)
            if self.rescale_grades:
                self._ballots = np.zeros((self.pop.V, self.pop.C))
                for v in range(self.pop.V):
                    max_util = np.max(self.pop.preferences_ut[v, :])
                    min_util = np.min(self.pop.preferences_ut[v, :])
                    if min_util == max_util:
                        # Same utilities for all candidates
                        self._ballots[v, :] = (
                            np.ones(self.pop.C) *
                            (self.max_grade + self.min_grade) / 2)
                    else:
                        # Generic case
                        self._ballots[v, :] = (
                            self.min_grade +
                            (self.max_grade - self.min_grade) *
                            (self.pop.preferences_ut[v, :] -
                                min_util) /
                            (max_util - min_util)
                        )
            else:
                self._ballots = np.clip(
                    self.pop.preferences_ut,
                    self.min_grade, self.max_grade
                )
            # Round (or not)
            if self.step_grade != 0:
                i_lowest_rung = np.int(np.ceil(
                    self.min_grade / self.step_grade
                ))
                i_highest_rung = np.int(np.floor(
                    self.max_grade / self.step_grade
                ))
                allowed_grades = np.concatenate((
                    [self.min_grade],
                    np.array(range(
                        i_lowest_rung, i_highest_rung + 1
                    )) * self.step_grade,
                    [self.max_grade]
                ))
                frontiers = (allowed_grades[0:-1] + allowed_grades[1:]) / 2
                for v in range(self.pop.V):
                    self._ballots[v, :] = allowed_grades[
                        np.digitize(self._ballots[v, :], frontiers)
                    ]
        return self._ballots

    @property
    def scores(self):
        """1d array of integers. ``scores[c]`` is the average grade of
        candidate ``c``.
        """
        # Note for developers: internally, it is Range Voting with sum. This
        # limits problems of float approximation, especially for IM method.
        # Scores as averages are only used when displaying the results.
        if self._scores is None:
            self._mylog("Compute scores", 1)
            self._scores = np.sum(self.ballots, 0)
        return np.around(self._scores / self.pop.V, 12)


if __name__ == '__main__':
    # A quick demo
    preferences_utilities = np.random.randint(-5, 5, (10, 5))
    pop = Population(preferences_utilities)
    election = RangeVotingAverageResult(pop)
    election.demo(log_depth=3)