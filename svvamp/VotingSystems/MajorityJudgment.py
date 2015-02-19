# -*- coding: utf-8 -*-
"""
Created on Tue Oct  7 14:52:43 2014
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
from svvamp.VotingSystems.MajorityJudgmentResult import MajorityJudgmentResult
from svvamp.Preferences.Population import Population


class MajorityJudgment(MajorityJudgmentResult, Election):
    """Majority Judgment.
    
    Inherits functions and optional parameters from superclasses
    :class:`~svvamp.ElectionResult` and :class:`~svvamp.Election`.

    :param min_grade: See attribute
        :attr:`~svvamp.MajorityJudgment.min_grade`.
    :param max_grade: See attribute
        :attr:`~svvamp.MajorityJudgment.max_grade`.
    :param step_grade: See attribute
        :attr:`~svvamp.MajorityJudgment.step_grade`.
    :param rescale_grades: See attribute
        :attr:`~svvamp.MajorityJudgment.rescale_grades`.

    :Example:

    >>> import svvamp
    >>> pop = svvamp.PopulationSpheroid(V=100, C=5)
    >>> election = svvamp.MajorityJudgment(pop, min_grade=0, max_grade=1, step_grade=0, rescale_grades=True)

    Each voter attributes a grade to each candidate. By default, authorized
    grades are all numbers in the interval
    [:attr:`~svvamp.MajorityJudgment.min_grade`,
    :attr:`~svvamp.MajorityJudgment.max_grade`]. To use a discrete set of
    notes, modify attribute :attr:`~svvamp.MajorityJudgment.step_grade`.

    .. note::

        Majority Judgement, as promoted by its authors, uses a
        discrete set of non-numerical grades. For our purpose, using a
        discrete set of numerical grades is isomorphic to this voting system.
        In contrast, using a continuous set of grades is a variant of this
        voting system, which has the advantage of being canonical, in the sense
        that there is no need to choose the number of authorized grades more or
        less arbitrarily.

    The candidate with highest median grade wins. For the tie-breaking rule,
    see :attr:`~svvamp.MajorityJudgment.scores`.

    Default behavior of sincere voters: voter ``v`` applies an affine
    transformation to her utilities
    :attr:`~svvamp.Population.preferences_utilities`\ ``[v, :]``
    to get her grades, such that her least-liked candidate receives
    :attr:`~svvamp.MajorityJudgment.min_grade` and her most-liked candidate
    receives :attr:`~svvamp.MajorityJudgment.max_grade`.
    To modify this behavior, use attribute
    :attr:`~svvamp.MajorityJudgment.rescale_grades`.
    For more details about the behavior of sincere voters, see
    :attr:`~svvamp.MajorityJudgment.ballots`.

    :meth:`~svvamp.Election.not_IIA`:

        * If :attr:`~svvamp.MajorityJudgment.rescale_grades` = ``False``,
          then Majority Judgment always meets IIA.
        * If :attr:`~svvamp.MajorityJudgment.rescale_grades` = ``True``, then
          then non-polynomial or non-exact algorithms from superclass
          :class:`~svvamp.Election` are used.

    :meth:`~svvamp.Election.CM`,
    :meth:`~svvamp.Election.ICM`,
    :meth:`~svvamp.Election.IM`,
    :meth:`~svvamp.Election.TM`,
    :meth:`~svvamp.Election.UM`: Exact in polynomial time.

    References:
    
        Majority Judgment : Measuring, Ranking, and Electing. Michel
        Balinski and Rida Laraki, 2010.
    """
    
    _layout_name = 'Majority Judgment'
    _options_parameters = Election._options_parameters.copy()
    _options_parameters.update(MajorityJudgmentResult._options_parameters)
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
        self._log_identity = "MAJORITY_JUDGMENT"
        self._class_result = MajorityJudgmentResult
        self._with_two_candidates_reduces_to_plurality = False
        # Even if rescale_grades = True, a voter who has the same utility
        # for c and d will not vote the same in Majority Judgment and in
        # Plurality.
        self._is_based_on_strict_rankings = False
        self._meets_IgnMC_c_ctb = True

        self._precheck_UM = False
        self._precheck_TM = False
        self._precheck_ICM = False

    def _create_result(self, pop_test):
        resultTest = MajorityJudgmentResult(
            pop_test, max_grade=self.max_grade, min_grade=self.min_grade,
            step_grade=self.step_grade, rescale_grades=self.rescale_grades)
        return resultTest
        
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
        ballots_test = np.copy(self.ballots)
        ballots_test[v, :] = self.min_grade
        scores_v_is_evil = np.zeros((2, self.pop.C))
        scores_v_is_evil[0, :] = np.median(ballots_test, 0)
        for c in range(self.pop.C):
            p = np.sum(ballots_test[:, c] > scores_v_is_evil[0, c])
            q = np.sum(ballots_test[:, c] < scores_v_is_evil[0, c])
            if q >= p:
                scores_v_is_evil[1, c] = -q
            else:
                scores_v_is_evil[1, c] = p
        w_temp = max(range(self.pop.C), 
                     key=lambda d: scores_v_is_evil[:, d].tolist())
                
        ballots_test[v, :] = self.max_grade
        score_v_is_nice_c = np.zeros(2)
        for c in range(self.pop.C):        
            if not c_is_wanted[c]:
                continue
            if not np.isneginf(self._v_IM_for_c[v, c]):
                continue
            score_v_is_nice_c[0] = np.median(ballots_test[:, c])
            p = np.sum(ballots_test[:, c] > score_v_is_nice_c[0])
            q = np.sum(ballots_test[:, c] < score_v_is_nice_c[0])
            if q >= p:
                score_v_is_nice_c[1] = -q
            else:
                score_v_is_nice_c[1] = p
            if ([score_v_is_nice_c[0], score_v_is_nice_c[1], -c] >=
                    [scores_v_is_evil[0, w_temp], scores_v_is_evil[1, w_temp], 
                     -w_temp]):
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

    def _CM_main_work_c(self, c, optimize_bounds):
        # In fact, in sorted_sincere, there will be sincere voters and one
        # manipulator (so that median and other stuff is always defined).
        # Grades for c are sorted in ascending order along c's column.
        # Grades for other candidates are sorted in descending order.
        # This way, when adding one manipulator, the median is shifted down
        # by a half-index.
        ballot_manip = np.ones(self.pop.C) * self.min_grade
        ballot_manip[c] = self.max_grade
        sorted_sincere = np.sort(np.concatenate((
            self.ballots[np.logical_not(self.v_wants_to_help_c[:, c]), :],
            [ballot_manip]
        ), 0), 0)[::-1, :]
        sorted_sincere[:, c] = sorted_sincere[::-1, c]
        self._mylogm("CM: sorted_sincere + 1 manipulator =", sorted_sincere, 3)
        medians = np.median(sorted_sincere[:-1, :], 0)
        self._mylogm("CM: medians (sincere only) =", medians, 3)
        p = np.sum(sorted_sincere[:-1, :] > medians, 0)
        self._mylogm("CM: p (sincere only) =", p, 3)
        q = np.sum(sorted_sincere[:-1, :] < medians, 0)
        self._mylogm("CM: q (sincere only) =", q, 3)

        n_s = self.pop.V - self.pop.matrix_duels[c, self.w]
        self._mylogv("CM: n_s =", n_s, 3)
        n_m = 0
        i_median = (n_s - 1) / 2
        CM_successful = False
        while not CM_successful:
            n_m += 1
            i_median += 0.5
            self._mylogv("CM: n_m =", n_m, 3)
            
            # Computing c's scores
            if i_median % 1 == 0:
                median_c = sorted_sincere[i_median, c]
            else:
                median_c = (sorted_sincere[np.floor(i_median), c] + 
                            sorted_sincere[np.ceil(i_median), c]) / 2
            if median_c == medians[c]:
                if median_c != self.max_grade:
                    p[c] += 1
            else:
                medians[c] = median_c
                q[c] = np.ceil(i_median)
                p[c] = np.sum(sorted_sincere[:-1, c] > median_c) + n_m
            if q[c] >= p[c]:
                score_c = [medians[c], -q[c], -c]
            else:
                score_c = [medians[c], p[c], -c]
            self._mylogv("CM: score_c =", score_c, 3)
                
            CM_successful = True
            for d in range(self.pop.C):
                if d == c:
                    continue
                
                # Computing d's scores
                if i_median % 1 == 0:
                    median_d = sorted_sincere[i_median, d]
                else:
                    median_d = (sorted_sincere[np.floor(i_median), d] + 
                                sorted_sincere[np.ceil(i_median), d]) / 2
                if median_d == medians[d]:
                    if median_d != self.min_grade:
                        q[d] += 1
                else:
                    medians[d] = median_d
                    p[d] = np.ceil(i_median)
                    q[d] = np.sum(sorted_sincere[:-1, d] < median_d) + n_m
                if q[d] >= p[d]:
                    score_d = [medians[d], -q[d], -d]
                else:
                    score_d = [medians[d], p[d], -d]
                self._mylogv("CM: score_d =", score_d, 3)
                    
                # Does manipulation work?
                if score_d > score_c:
                    CM_successful = False

        self._sufficient_coalition_size_CM[c] = n_m
        self._necessary_coalition_size_CM[c] = n_m

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

    # Since Majority Judgment meets IgnMC_c_tb, the general methods are exact.


if __name__ == '__main__':
    # A quick demo
    preferences_utilities = np.random.randint(-5, 5, (10, 5))
    pop = Population(preferences_utilities)
    election = MajorityJudgment(pop)
    election.demo(log_depth=3)