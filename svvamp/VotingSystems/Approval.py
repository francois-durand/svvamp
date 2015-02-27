# -*- coding: utf-8 -*-
"""
Created on Mon Sep 29 18:32:45 2014
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

from svvamp.VotingSystems.Election import Election
from svvamp.VotingSystems.ApprovalResult import ApprovalResult
from svvamp.Preferences.Population import Population


class Approval(ApprovalResult, Election):
    """Approval voting.

    Inherits functions and optional parameters from superclasses
    :class:`~svvamp.ElectionResult` and :class:`~svvamp.Election`.

    :param approval_comparator: See attribute
        :attr:`~svvamp.Approval.approval_comparator`.
    :param approval_threshold: See attribute
        :attr:`~svvamp.Approval.approval_threshold`.

    :Example:

    >>> import svvamp
    >>> pop = svvamp.PopulationSpheroid(V=100, C=5)
    >>> election = svvamp.Approval(pop, approval_comparator='>', approval_threshold=0)

    Each voter may vote for any number of candidates. The candidate with most
    votes is declared the winner. In case of a tie, the tied candidate with
    lowest index wins.

    Default behavior of sincere voters: sincere voter ``v``
    approves candidate ``c`` iff
    :attr:`~svvamp.Population.preferences_ut`\ ``[v, c]`` > 0.
    To modify this behavior, use attributes
    :attr:`~svvamp.Approval.approval_comparator` and
    :attr:`~svvamp.Approval.approval_threshold`.

    :meth:`~svvamp.Election.not_IIA`: With our assumptions, Approval
    voting always meets IIA.

    :meth:`~svvamp.Election.CM`,
    :meth:`~svvamp.Election.ICM`,
    :meth:`~svvamp.Election.IM`,
    :meth:`~svvamp.Election.TM`,
    :meth:`~svvamp.Election.UM`: Exact in polynomial time.

    References:

        'Approval voting', Steven Brams and Peter Fishburn. In: American
        Political Science Review 72 (3 1978), pp. 831–847.
    """

    _layout_name = 'Approval'
    _options_parameters = Election._options_parameters.copy()
    _options_parameters.update(ApprovalResult._options_parameters)
    _options_parameters['IM_option'] = {'allowed': ['exact'],
                                        'default': 'exact'}
    _options_parameters['TM_option'] = {'allowed': ['exact'],
                                        'default': 'exact'}
    _options_parameters['UM_option'] = {'allowed': ['exact'],
                                        'default': 'exact'}
    _options_parameters['ICM_option'] = {'allowed': ['exact'],
                                           'default': 'exact'}
    _options_parameters['CM_option'] = {'allowed': ['exact'],
                                        'default': 'exact'}

    def __init__(self, population, **kwargs):
        super().__init__(population, **kwargs)
        self._log_identity = "APPROVAL"
        self._class_result = ApprovalResult
        self._log_depth = 0
        self._with_two_candidates_reduces_to_plurality = False
        self._is_based_on_rk = False
        self._is_based_on_ut_minus1_1 = True
        self._meets_IIA = True
        self._meets_IgnMC_c_ctb = True
        self._precheck_ICM = False
        self._precheck_TM = False
        self._precheck_UM = False

    def _create_result(self, pop_test):
        result_test = ApprovalResult(
            pop_test, approval_threshold=self.approval_threshold,
            approval_comparator=self.approval_comparator)
        return result_test

    #%% Independence of Irrelevant Alternatives (IIA)

    # Defined in the superclass.

    #%% Individual manipulation (IM)

    def _IM_main_work_v(self, v, c_is_wanted,
                        nb_wanted_undecided, stop_if_true):
        # self._mylogv('scores =', self.scores)
        # self._mylogv('ballots[v, :] =', self.ballots[v, :])
        scores_test = self.scores - self.ballots[v, :]
        w_test = np.argmax(scores_test)
        # Best strategy: vote only for c
        for c in range(w_test + 1):  # c <= w_test
            if not c_is_wanted[c]:
                continue
            if not np.isneginf(self._v_IM_for_c[v, c]):
                continue
            nb_wanted_undecided -= 1
            if scores_test[c] + 1 >= scores_test[w_test]:
                self._v_IM_for_c[v, c] = True
                self._candidates_IM[c] = True
                self._voters_IM[v] = True
                self._is_IM = True
                self._mylog("IM found", 3)
                if stop_if_true:
                    return
            else:
                self._v_IM_for_c[v, c] = False
            if nb_wanted_undecided == 0:
                return
        for c in range(w_test + 1, self.pop.C):  # c > w_test
            if not c_is_wanted[c]:
                continue
            if not np.isneginf(self._v_IM_for_c[v, c]):
                continue
            nb_wanted_undecided -= 1
            if scores_test[c] + 1 > scores_test[w_test]:
                self._v_IM_for_c[v, c] = True
                self._candidates_IM[c] = True
                self._voters_IM[v] = True
                self._is_IM = True
                self._mylog("IM found", 3)
                if stop_if_true:
                    return
            else:
                self._v_IM_for_c[v, c] = False
            if nb_wanted_undecided == 0:
                return

    #%% Coalition Manipulation (CM)

    def _CM_main_work_c_exact(self, c, optimize_bounds):
        scores_test = np.sum(
            self.ballots[np.logical_not(self.v_wants_to_help_c[:, c]), :],
            0)
        w_test = np.argmax(scores_test)
        self._sufficient_coalition_size_CM[c] = (
            scores_test[w_test] - scores_test[c] + (c > w_test))
        self._necessary_coalition_size_CM[c] = \
            self._sufficient_coalition_size_CM[c]

    #%% Trivial Manipulation (TM)

    def TM(self):
        return self.CM()

    def TM_c(self, c):
        return self.CM_c(c)

    def TM_with_candidates(self):
        return self.CM_with_candidates()

    #%% Unison Manipulation (UM)

    def UM(self):
        return self.CM()

    def UM_c(self, c):
        return self.CM_c(c)

    def UM_with_candidates(self):
        return self.CM_with_candidates()

    #%% Ignorant-Coalition Manipulation (ICM)

    # Defined in the superclass.


if __name__ == '__main__':
    # A quick demo
    preferences_utilities = np.random.randint(-5, 5, (10, 5))
    pop = Population(preferences_utilities)
    election = Approval(pop)
    election.demo(log_depth=3)
