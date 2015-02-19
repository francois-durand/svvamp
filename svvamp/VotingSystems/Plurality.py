# -*- coding: utf-8 -*-
"""
Created on Fri Sep 26 14:48:07 2014
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
from svvamp.VotingSystems.PluralityResult import PluralityResult
from svvamp.Preferences.Population import Population


class Plurality(PluralityResult, Election):
    """Plurality.

    Inherits functions and optional parameters from superclasses
    :class:`~svvamp.ElectionResult` and :class:`~svvamp.Election`.

    :Example:

    >>> import svvamp
    >>> pop = svvamp.PopulationSpheroid(V=100, C=5)
    >>> election = svvamp.Plurality(pop)

    Each voter votes for one candidate. The candidate with most
    votes is declared the winner. In case of a tie, the tied candidate with
    lowest index wins.

    Sincere voters vote for their top-ranked candidate.

    :meth:`~svvamp.Election.not_IIA`: Non-polynomial
    or non-exact algorithms from superclass :class:`~svvamp.Election`.

    :meth:`~svvamp.Election.CM`,
    :meth:`~svvamp.Election.ICM`,
    :meth:`~svvamp.Election.IM`,
    :meth:`~svvamp.Election.TM`,
    :meth:`~svvamp.Election.UM`: Exact in polynomial time.
    """

    _layout_name = 'Plurality'
    _options_parameters = Election._options_parameters.copy()
    _options_parameters.update(PluralityResult._options_parameters)
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
        self._log_identity = "PLURALITY"
        self._class_result = PluralityResult
        self._with_two_candidates_reduces_to_plurality = True
        self._is_based_on_strict_rankings = True
        self._meets_majority_favorite_c_vtb_ctb = True
        self._precheck_ICM = False
        self._precheck_TM = False
        self._precheck_UM = False

    #%% Independence of Irrelevant Alternatives (IIA)

    def _compute_winner_of_subset(self, candidates_r):
        self._mylogv("IIA: Compute winner of subset ", candidates_r, 3)
        scores_r = np.bincount(
            np.argmax(self.pop.preferences_borda_vtb[:, candidates_r], 1),
            minlength=candidates_r.shape[0])
        index_w_r_in_subset = np.argmax(scores_r)
        w_r = candidates_r[index_w_r_in_subset]
        self._mylogv("IIA: Winner =", w_r, 3)
        return w_r

    #%% Individual manipulation (IM)

    def _compute_IM(self, mode, c=None):
        """Compute IM: is_IM, candidates_IM.

        For Plurality, since calculation is quite cheap, we calculate
        everything directly, even if complete_mode is False.
        """
        self._mylog("Compute IM", 1)
        self._IM_was_computed_with_candidates = True
        self._IM_was_computed_with_voters = True
        self._IM_was_computed_full = True

        close_races = np.zeros(self.pop.C, dtype=bool)
         # Case 1 : w wins by only one ballot over c_test < w.
        for c_test in range(0, self.w):
            if self.scores[c_test] == self.score_w - 1:
                close_races[c_test] = True
        # Case 2 : w is tied with c_test > w (and benefits for the tie-breaking
        # rule).
        for c_test in range(self.w + 1, self.pop.C):
            if self.scores[c_test] == self.score_w:
                close_races[c_test] = True

        # Voter v can manipulate for c_test iif:
        # * c_test is in close race,
        # * And v wants to manipulate for c_test,
        # * And v does not already for for c_test,
        self._v_IM_for_c = np.logical_and(
            close_races[np.newaxis, :],
            np.logical_and(
                self.v_wants_to_help_c,
                self.pop.preferences_borda_vtb != self.pop.C - 1
            )
        )

        self._candidates_IM = np.any(self._v_IM_for_c, 0)
        self._is_IM = np.any(self._candidates_IM)

    def _compute_IM_v(self, v, c_is_wanted, stop_if_true):
        self._compute_IM(mode='', c=None)

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

    # The voting system meets IgnMC_c_ctb: hence, general methods are exact.

    #%% Coalition Manipulation (CM)

    def _CM_main_work_c_exact(self, c, optimize_bounds):
        scores_s = np.bincount(
            self.pop.preferences_ranking[np.logical_not(
                self.v_wants_to_help_c[:, c]), 0],
            minlength=self.pop.C)
        # We need as many manipulators as scores_s[w] - scores_s[c],
        # plus one if c > w (because in this case, c is disadvantaged by
        # the tie-breaking rule). N.B.: this value cannot be negative,
        # so it is not necessary to use max(..., 0).
        self._sufficient_coalition_size_CM[c] = (
            scores_s[self.w] - scores_s[c] + (c > self.w))
        self._necessary_coalition_size_CM[c] = (
            self._sufficient_coalition_size_CM[c])


if __name__ == '__main__':
    # A quick demo
    preferences_utilities = np.random.randint(-5, 5, (10, 5))
    pop = Population(preferences_utilities)
    election = Plurality(pop)
    election.demo(log_depth=3)