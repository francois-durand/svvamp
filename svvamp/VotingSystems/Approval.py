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
    """Approval.
    
    If approval_comparator is '>' (resp. '>='), then sincere voter v votes for
    candidates c iff preferences_utilities[v, c] > approval_threshold.
    
    Ties are broken by natural order on the candidates (lower index wins).

    Selected references
    Brams, Steven and Peter Fishburn. « Approval voting ». In: American
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
        self._is_based_on_strict_rankings = False
        self._is_based_on_utilities_minus1_1 = True
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
        """Trivial manipulation, focus on one candidate.

        Arguments:
        c -- Integer. The candidate for whom we want to manipulate.

        Returns:
        is_TM_c -- Boolean (or NaN). True if TM for candidate c is possible,
            False otherwise. If the algorithm cannot decide, then NaN.
        log_TM -- String. Parameters used to compute TM.
        """
        return self.CM_c(c)

    def TM_with_candidates(self):
        """Trivial manipulation, complete mode.

        For ordinal voting systems, we call 'trivial manipulation' for
        candidate c against w the fact of putting c on top (compromising), w
        at bottom (burying), while keeping a sincere order on other candidates.

        For cardinal voting systems, we call 'trivial manipulation' for c
        (against w) the fact of putting the maximum grade for c and the
        minimum grade for other candidates.

        In both cases, the intuitive idea is the following: if I want to
        make c win and I only know that candidate w is 'dangerous' (but I know
        nothing else), then trivial manipulation is my 'best' strategy.

        We say that a situation is "trivially manipulable" for c (implicitly:
        by coalition) iff, when all voters preferring c to the sincere winner w
        use trivial manipulation, candidate c wins.

        Returns:
        is_TM -- Boolean (or NaN). True if TM is possible, False otherwise. If
            the algorithm cannot decide, then NaN.
        log_TM -- String. Parameters used to compute TM.
        candidates_TM -- 1d array of booleans (or NaN). candidates_TM[c]
            is True if a TM for candidate c is possible, False otherwise. If
            the algorithm cannot decide, then NaN. By convention,
            candidates_TM[w] = False.
        """
        return self.CM_with_candidates()

    #%% Unison Manipulation (UM)

    def UM(self):
        """Unison manipulation, incomplete mode.

        Returns:
        is_UM -- Boolean (or NaN). True if UM is possible, False otherwise. If
            the algorithm cannot decide, then NaN.
        log_UM -- String. Parameters used to compute UM.
        """
        return self.CM()

    def UM_c(self, c):
        """Unison manipulation, focus on one candidate.

        Arguments:
        c -- Integer. The candidate for whom we want to manipulate.

        Returns:
        is_UM_c -- Boolean (or NaN). True if UM for candidate c is possible,
            False otherwise. If the algorithm cannot decide, then NaN.
        log_UM -- String. Parameters used to compute UM.
        """
        return self.CM_c(c)

    def UM_with_candidates(self):
        """Unison manipulation, complete mode.

        We say that a situation is unison-manipulable for a candidate c != w
        iff all voters who prefer c to the sincere winner w can cast the SAME
        ballot so that c is elected (while other voters still vote sincerely).

        Returns:
        is_UM -- Boolean (or NaN). True if UM is possible, False otherwise. If
            the algorithm cannot decide, then NaN.
        log_UM -- String. Parameters used to compute UM.
        candidates_UM -- 1d array of booleans (or NaN). candidates_UM[c]
            is True if UM for candidate c is possible, False otherwise. If
            the algorithm cannot decide, then NaN. By convention,
            candidates_UM[w] = False.
        """
        return self.CM_with_candidates()

    #%% Ignorant-Coalition Manipulation (ICM)

    # Defined in the superclass.


if __name__ == '__main__':
    # A quick demo
    preferences_utilities = np.random.randint(-5, 5, (10, 5))
    pop = Population(preferences_utilities)
    election = Approval(pop)
    election.demo(log_depth=3)
