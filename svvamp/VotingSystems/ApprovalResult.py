# -*- coding: utf-8 -*-
"""
Created on Mon Sep 29 18:29:11 2014
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


class ApprovalResult(ElectionResult):
    """Results of an election using Approval voting.

    If approval_comparator is '>' (resp. '>='), then sincere voter v votes for
    candidates c iff preferences_utilities[v, c] > approval_threshold.

    Ties are broken by natural order on the candidates (lower index wins).
    """

    _options_parameters = ElectionResult._options_parameters.copy()
    _options_parameters['approval_threshold'] = {
        'allowed': TypeChecker.is_number, 'default': 0}
    _options_parameters['approval_comparator'] = {
        'allowed': ['>', '>='], 'default': '>'}

    def __init__(self, population, **kwargs):
        super().__init__(population, **kwargs)
        self._log_identity = "APPROVAL_RESULT"

    #%% Setting the parameters

    @property
    def approval_threshold(self):
        """Number -- Utility above which a sincere voter approves a candidate.
        See also approval_comparator.
        """
        return self._approval_threshold

    @approval_threshold.setter
    def approval_threshold(self, value):
        self._approval_threshold = value
        self._forget_all_computations()

    @property
    def approval_comparator(self):
        """String -- Can be '>' (default) or '>='.
        When approval_comparator is ''>' (resp. '>='), sincere voter v approves
        candidates c iff:
        pop.preferences_utilities[v, c] > (resp. >=) approval_threshold.
        """
        return self._approval_comparator

    @approval_comparator.setter
    def approval_comparator(self, value):
        if value in self.options_parameters['approval_comparator']['allowed']:
            self._mylogv("Setting approval_comparator =", value, 1)
            self._approval_comparator = value
            self._forget_all_computations()
        else:
            raise ValueError("Unknown option for approval_comparator: " +
                             format(value))

    #%% Counting the ballots

    @property
    def ballots(self):
        """2d array of {0, 1}. ballots[v, c] = 1 iff voter v votes for 
        candidates c.
        """
        if self._ballots is None:
            self._mylog("Compute ballots", 1)
            if self.approval_comparator == '>':
                self._ballots = np.greater(
                    self.pop.preferences_utilities, self.approval_threshold)
            else:
                self._ballots = np.greater_equal(
                    self.pop.preferences_utilities, self.approval_threshold)
        return self._ballots
        
    @property
    def scores(self):
        """1d array of integers. scores[c] is the number of voters who vote
        for candidate c.
        """
        if self._scores is None:
            self._mylog("Compute scores", 1)
            self._scores = np.sum(self.ballots, 0)
        return self._scores


if __name__ == '__main__':
    # A quick demo
    preferences_utilities = np.random.randint(-5, 5, (10, 5))
    pop = Population(preferences_utilities)
    election = ApprovalResult(pop)
    election.demo(log_depth=3)