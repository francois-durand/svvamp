# -*- coding: utf-8 -*-
"""
Created on Mon Oct  6 10:50:13 2014
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
from svvamp.VotingSystems.RangeVotingAverageResult import \
    RangeVotingAverageResult
from svvamp.Preferences.Population import Population


class RangeVotingAverage(RangeVotingAverageResult, Election):
    """Range Voting with Average.
    
    Inherits functions and optional parameters from superclasses
    :class:`~svvamp.ElectionResult` and :class:`~svvamp.Election`.

    :param min_grade: See attribute
        :attr:`~svvamp.RangeVotingAverage.min_grade`.
    :param max_grade: See attribute
        :attr:`~svvamp.RangeVotingAverage.max_grade`.
    :param step_grade: See attribute
        :attr:`~svvamp.RangeVotingAverage.step_grade`.
    :param rescale_grades: See attribute
        :attr:`~svvamp.RangeVotingAverage.rescale_grades`.

    :Example:

    >>> import svvamp
    >>> pop = svvamp.PopulationSpheroid(V=100, C=5)
    >>> election = svvamp.RangeVotingAverage(pop, min_grade=0, max_grade=1, step_grade=0, rescale_grades=True)

    Each voter attributes a grade to each candidate. By default, authorized
    grades are all numbers in the interval
    [:attr:`~svvamp.RangeVotingAverage.min_grade`,
    :attr:`~svvamp.RangeVotingAverage.max_grade`]. To use a discrete set of
    notes, modify attribute :attr:`~svvamp.RangeVotingAverage.step_grade`.

    The candidate with highest average grade wins. In case of a tie, the tied
    candidate with lowest index is declared the winner.

    Default behavior of sincere voters: voter ``v`` applies an affine
    transformation to her utilities
    :attr:`~svvamp.Population.preferences_ut`\ ``[v, :]``
    to get her grades, such that her least-liked candidate receives
    :attr:`~svvamp.RangeVotingAverage.min_grade` and her most-liked candidate
    receives :attr:`~svvamp.RangeVotingAverage.max_grade`.
    For other possible behaviors, see
    :attr:`~svvamp.RangeVotingAverage.ballots`.

    :meth:`~svvamp.Election.not_IIA`:

        * If :attr:`~svvamp.RangeVotingAverage.rescale_grades` = ``False``,
          then Range voting always meets IIA.
        * If :attr:`~svvamp.RangeVotingAverage.rescale_grades` = ``True``, then
          then non-polynomial or non-exact algorithms from superclass
          :class:`~svvamp.Election` are used.

    :meth:`~svvamp.Election.CM`,
    :meth:`~svvamp.Election.ICM`,
    :meth:`~svvamp.Election.IM`,
    :meth:`~svvamp.Election.TM`,
    :meth:`~svvamp.Election.UM`: Exact in polynomial time.
    """

    _layout_name = 'Range voting with average'
    _options_parameters = Election._options_parameters.copy()
    _options_parameters.update(RangeVotingAverageResult._options_parameters)
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
        self._log_identity = "RANGE_VOTING_AVERAGE"
        self._class_result = RangeVotingAverageResult
        self._with_two_candidates_reduces_to_plurality = False
        # Even if rescale_grades = True, a voter who has the same utility
        # for c and d will not vote the same in Range Voting and in Plurality.
        self._is_based_on_strict_rankings = False
        self._meets_IgnMC_c_ctb = True

        self._precheck_UM = False
        self._precheck_TM = False
        self._precheck_ICM = False

    def _create_result(self, pop_test):
        result_test = RangeVotingAverageResult(
            pop_test, max_grade=self.max_grade, min_grade=self.min_grade,
            step_grade=self.step_grade, rescale_grades=self.rescale_grades)
        return result_test
        
    #%% Independence of Irrelevant Alternatives (IIA)
        
    @property
    def meets_IIA(self):
        return not self.rescale_grades

    @property
    def is_based_on_utilities_minus1_1(self):
        return self._rescale_grades
        
    #%% Individual manipulation (IM)
            
    def _IM_main_work_v(self, v, c_is_wanted,
                        nb_wanted_undecided, stop_if_true):
        _ = self.scores  # Ensure _scores are computed
        scores_without_v = self._scores - self.ballots[v, :]
        w_without_v = np.argmax(np.around(scores_without_v, 12))
        new_score_w = np.around(scores_without_v[w_without_v] + self.min_grade,
                                12)
        for c in range(self.pop.C):
            if not c_is_wanted[c]:
                continue
            if not np.isneginf(self._v_IM_for_c[v, c]):
                continue
            new_score_c = np.around(scores_without_v[c] + self.max_grade, 12)
            if ((c < w_without_v and new_score_c >= new_score_w) or
                    (c > w_without_v and new_score_c > new_score_w)):
                self._v_IM_for_c[v, c] = True
                self._candidates_IM[c] = True
                self._voters_IM[v] = True
                self._is_IM = True
                self._mylog("IM found", 3)
                if stop_if_true:
                    return
            else:
                self._v_IM_for_c[v, c] = False
            nb_wanted_undecided -= 1
            if nb_wanted_undecided == 0:
                return

    #%% Coalition Manipulation (CM)

    def _CM_main_work_c(self, c, optimize_bounds):
        scores_temp = np.sum(self.ballots[
            np.logical_not(self.v_wants_to_help_c[:, c]), :
        ], 0)
        w_temp = np.argmax(np.around(scores_temp, 12))
        sufficient_not_rounded = np.around(
            (scores_temp[w_temp] - scores_temp[c]) /
            (self.max_grade - self.min_grade),
            12)
        self._mylogv("sufficient_not_rounded =", sufficient_not_rounded, 2)
        if sufficient_not_rounded % 1 == 0 and c > w_temp:
            self._sufficient_coalition_size_CM[c] = sufficient_not_rounded + 1
        else:
            self._sufficient_coalition_size_CM[c] = np.ceil(
                sufficient_not_rounded)
        self._necessary_coalition_size_CM[c] = (
            self._sufficient_coalition_size_CM[c])

    #%% Trivial Manipulation (TM)

    def TM(self):
        """Trivial manipulation, incomplete mode.

        Returns:
        is_TM -- Boolean (or NaN). True if TM is possible, False otherwise. If
            the algorithm cannot decide, then NaN.
        log_TM -- String. Parameters used to compute TM.
        """
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

    # Since Range Voting meets IgnMC_c_ctb, the general methods are exact.


if __name__ == '__main__':
    # A quick demo
    preferences_utilities = np.random.randint(-5, 5, (10, 5))
    pop = Population(preferences_utilities)
    election = RangeVotingAverage(pop)
    election.demo(log_depth=3)