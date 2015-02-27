# -*- coding: utf-8 -*-
"""
Created on Mon Sep 29 14:39:07 2014
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
from svvamp.VotingSystems.VetoResult import VetoResult
from svvamp.Preferences.Population import Population


class Veto(VetoResult, Election):
    """Veto. Also called Antiplurality.
    
    Inherits functions and optional parameters from superclasses
    :class:`~svvamp.ElectionResult` and :class:`~svvamp.Election`.

    :Example:

    >>> import svvamp
    >>> pop = svvamp.PopulationSpheroid(V=100, C=5)
    >>> election = svvamp.Veto(pop)

    Each voter votes against one candidate (veto). The candidate with least
    vetos is declared the winner. In case of a tie, the tied candidate with
    lowest index wins.

    Sincere voters vote against their least-liked candidate.

    :meth:`~svvamp.Election.not_IIA`: Non-polynomial
    or non-exact algorithms from superclass :class:`~svvamp.Election`.

    :meth:`~svvamp.Election.CM`,
    :meth:`~svvamp.Election.ICM`,
    :meth:`~svvamp.Election.IM`,
    :meth:`~svvamp.Election.TM`,
    :meth:`~svvamp.Election.UM`: Exact in polynomial time.
    """

    _layout_name = 'Veto'
    _options_parameters = Election._options_parameters.copy()
    _options_parameters.update(VetoResult._options_parameters)
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
        self._log_identity = "VETO"
        self._class_result = Veto

        self._with_two_candidates_reduces_to_plurality = True
        self._is_based_on_strict_rankings = True

        # Just as a reminder...
        self._meets_InfMC_c = False

        self._precheck_ICM = False
        self._precheck_TM = False
        self._precheck_UM = False

    #%% Independence of Irrelevant Alternatives (IIA)

    def _compute_winner_of_subset(self, candidates_r):
        self._mylogv("IIA: Compute winner of subset ", candidates_r, 3)
        scores_r = - np.bincount(
            np.argmin(self.pop.preferences_borda_rk[:, candidates_r], 1),
            minlength=candidates_r.shape[0])
        index_w_r_in_subset = np.argmax(scores_r)
        w_r = candidates_r[index_w_r_in_subset]
        self._mylogv("IIA: Winner =", w_r, 3)
        return w_r
        
    #%% Individual manipulation (IM)            

    def _IM_main_work_v(self, v, c_is_wanted,
                        nb_wanted_undecided, stop_if_true):
        # If voter v strictly prefers some c_test to w, let us note that
        # she cannot have voted against c_test. So, the only thing she
        # can do better is to vote against w (if it is not already the
        # case), because otherwise w will still keep a better score than
        # c_test. This strategy does  not depend on c_test!
        scores_with_v_manip = np.copy(self.scores)
        # Remove v's sincere vote:
        scores_with_v_manip[self.ballots[v]] += 1
        # Vote against w instead:
        scores_with_v_manip[self.w] -= 1
        new_winner = np.argmax(scores_with_v_manip)
        self._v_IM_for_c[v, :] = False
        if self.v_wants_to_help_c[v, new_winner]:
            self._v_IM_for_c[v, new_winner] = True
            self._candidates_IM[new_winner] = True
            self._voters_IM[v] = True
            self._is_IM = True

    #%% Trivial Manipulation (TM)

    def _TM_main_work_c(self, c):
        # Sincere voters:
        scores_test = - np.bincount(
            self.pop.preferences_rk[
                np.logical_not(self.v_wants_to_help_c[:, c]), -1],
            minlength=self.pop.C
        )
        # Manipulators vote against w:
        # Remark: for Veto, the trivial strategy is far from the best one!
        scores_test[self.w] -= self.pop.matrix_duels_ut[c, self.w]
        # Conclude
        w_test = np.argmax(scores_test)
        self._candidates_TM[c] = (w_test == c)

    #%% Unison Manipulation (UM)
    # In their sincere ballots, manipulators were not voting against c,
    # but against w or another d. If now they vote against some d, then
    # w's score might get better, while c's score will not change: this
    # strategy cannot succeed. So, their only hope is to vote against w.
    # This is precisely the trivial strategy!

    def UM(self):
        """Unison manipulation, incomplete mode.

        Returns:
        is_UM -- Boolean (or NaN). True if UM is possible, False otherwise. If
            the algorithm cannot decide, then NaN.
        log_UM -- String. Parameters used to compute UM.
        """
        return self.TM()

    def UM_c(self, c):
        """Unison manipulation, focus on one candidate.

        Arguments:
        c -- Integer. The candidate for whom we want to manipulate.

        Returns:
        is_UM_c -- Boolean (or NaN). True if UM for candidate c is possible,
            False otherwise. If the algorithm cannot decide, then NaN.
        log_UM -- String. Parameters used to compute UM.
        """
        return self.TM_c(c)

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
        return self.TM_with_candidates()

    #%% Ignorant-Coalition Manipulation (ICM)

    def _ICM_main_work_c(self, c, optimize_bounds):
        # At worst, 'sincere' manipulators may respond by all voting
        # against c.
        # We need to give as many vetos to all other candidates, which
        # makes (C - 1) * n_s (where n_s = sincere voters).
        # For each candidate of index lower than c, we need to give an
        # additional veto (because of the tie-breaking rule). This makes
        # c additional manipulators needed.
        # Hence sufficient_...[c] = (C - 1) * n_s + c.
        n_s = self.pop.V - self.pop.matrix_duels_ut[c, self.w]
        self._sufficient_coalition_size_ICM[c] = (self.pop.C - 1) * n_s + c
        self._necessary_coalition_size_ICM[c] = (
            self._sufficient_coalition_size_ICM[c])

    #%% Coalition Manipulation (CM)

    def _CM_main_work_c(self, c, optimize_bounds):
        # Sincere voters:
        scores_test = - np.bincount(
            self.pop.preferences_rk[
                np.logical_not(self.v_wants_to_help_c[:, c]), -1],
            minlength=self.pop.C
        )
        # Make each other candidate d have a lower score than c:
        # manipulators against d = scores_test[d] - scores_test[c] + (d < c)
        self._sufficient_coalition_size_CM[c] = np.sum(
            np.maximum(
                scores_test - scores_test[c] + (
                    np.array(range(self.pop.C)) < c),
                0))
        self._necessary_coalition_size_CM[c] = (
            self._sufficient_coalition_size_CM[c])


if __name__ == '__main__':
    # A quick demo
    preferences_utilities = np.random.randint(-5, 5, (10, 5))
    pop = Population(preferences_utilities)
    election = Veto(pop)
    election.demo(log_depth=3)