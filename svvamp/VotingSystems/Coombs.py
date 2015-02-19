# -*- coding: utf-8 -*-
"""
Created on Fri Oct  3 15:17:58 2014
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
from svvamp.VotingSystems.CoombsResult import CoombsResult
from svvamp.Preferences.Population import Population


class Coombs(CoombsResult, Election):
    """Coombs method.

    Inherits functions and optional parameters from superclasses
    :class:`~svvamp.ElectionResult` and :class:`~svvamp.Election`.

    :Example:

    >>> import svvamp
    >>> pop = svvamp.PopulationSpheroid(V=100, C=5)
    >>> election = svvamp.Coombs(pop)

    The candidate who is ranked last by most voters is eliminated. Then
    we iterate. Ties are broken in favor of lower-index candidates: in case
    of a tie, the tied candidate with highest index is eliminated.

    :meth:`~svvamp.Election.CM`:

        * :attr:`~svvamp.Election.CM_option` = ``'fast'``:
          Polynomial heuristic. Can prove CM but unable to decide non-CM
          (except in rare obvious cases).
        * :attr:`~svvamp.Election.CM_option` = ``'exact'``:
          Non-polynomial (:math:`C!`).

    :meth:`~svvamp.Election.ICM`: Exact in polynomial time.

    :meth:`~svvamp.Election.IM`:

        * :attr:`~svvamp.Election.IM_option` = ``'fast'``:
          Polynomial heuristic. Can prove CM but unable to decide non-IM
          (except in rare obvious cases).
        * :attr:`~svvamp.Election.IM_option` = ``'exact'``:
          Non-polynomial (:math:`C!`).

    :meth:`~svvamp.Election.not_IIA`: Non-polynomial
    or non-exact algorithms from superclass :class:`~svvamp.Election`.

    :meth:`~svvamp.Election.TM`: Exact in polynomial time.

    :meth:`~svvamp.Election.UM`: For this voting system, UM and CM are
    equivalent. For this reason, :attr:`~svvamp.Election.UM_option` and
    :attr:`~svvamp.Election.CM_option` are linked to each other: modifying one
    modifies the other accordingly.

    References:

        'On The Complexity of Manipulating Elections', Tom Coleman
        and Vanessa Teague, 2007.
    """

    _layout_name = 'Coombs'
    _options_parameters = Election._options_parameters.copy()
    _options_parameters.update(CoombsResult._options_parameters)
    _options_parameters['IM_option'] = {'allowed': ['fast', 'exact'],
                                        'default': 'fast'}
    _options_parameters['TM_option'] = {'allowed': ['exact'],
                                        'default': 'exact'}
    _options_parameters['UM_option'] = {'allowed': ['fast', 'exact'],
                                        'default': 'fast'}
    _options_parameters['ICM_option'] = {'allowed': ['exact'],
                                           'default': 'exact'}
    _options_parameters['CM_option'] = {'allowed': ['fast', 'exact'],
                                        'default': 'fast'}

    def __init__(self, population, **kwargs):
        super().__init__(population, **kwargs)
        self._log_identity = "COOMBS"
        self._class_result = CoombsResult
        self._with_two_candidates_reduces_to_plurality = True
        self._is_based_on_strict_rankings = True
        self._meets_IgnMC_c_ctb = True
        self._precheck_UM = False  # No UM precheck before CM
        self._precheck_ICM = False

    @property
    def UM_option(self):
        return self._UM_option

    @UM_option.setter
    def UM_option(self, value):
        try:
            if self._UM_option == value:
                return
        except AttributeError:
            pass
        if value in self.options_parameters['UM_option']['allowed']:
            self._mylogv("Setting UM_option =", value, 1)
            self._CM_option = value
            self._UM_option = value
            self._forget_UM()
            self._forget_CM()
        else:
            raise ValueError("Unknown option for UM: " + format(value))

    @property
    def CM_option(self):
        return self._CM_option

    @CM_option.setter
    def CM_option(self, value):
        try:
            if self._CM_option == value:
                return
        except AttributeError:
            pass
        if value in self.options_parameters['CM_option']['allowed']:
            self._mylogv("Setting CM_option =", value, 1)
            self._CM_option = value
            self._UM_option = value
            self._forget_UM()
            self._forget_CM()
        else:
            raise ValueError("Unknown option for CM: " + format(value))

    #%% Manipulation: general routine
        
    def _CM_aux_fast(self, c, n_max, preferences_borda_s):
        """Fast algorithm used for IM and CM (which is equivalent to UM)
        
        Arguments:
        c -- Integer. Candidate for which we want to manipulate.
        n_max -- Integer or inf. Maximum number of manipulators allowed.
            IM --> put 1.
            CM, complete and exact --> put the current value of 
                sufficient_coalition_size[c] - 1 (we want to find the best 
                value for sufficient_coalition_size[c], even if it exceeds the 
                number of manipulators)
            CM, otherwise --> put the number of manipulators.
        preferences_borda_s -- 2d integer. Preferences of the sincere voters
            (in Borda format with vtb).

        Returns:
        n_manip_fast -- Integer or +inf. 
            If a manipulation is found with n_max manipulators or less,
            then a sufficient number of manipulators is returned.
            Otherwise, it is +inf.
        """
        # Each step of the loop:
        # ^^^^^^^^^^^^^^^^^^^^^^
        # We start at the end of round r, with 
        # candidates[is_candidate_alive_end_r]. We find the d such that, 
        # with d in addition at the beginning, we need the fewest manipulators
        # to eliminate d (and reach the final situation we wanted).
        # Initialization
        # ^^^^^^^^^^^^^^
        # End of last round: only c is alive.
        nb_manipulators_used = 0
        is_candidate_alive_end_r = np.zeros(self.pop.C, dtype=np.bool)
        is_candidate_alive_end_r[c] = True
        for r in range(self.pop.C - 2, -1, -1):
            self._mylogv("CM_aux_fast: Round r =", r, 3)
            least_nb_manipulators_for_r = np.inf
            # We look for a candidate with which round r was easy
            for d in range(self.pop.C):
                if is_candidate_alive_end_r[d]:
                    continue
                self._mylogv("CM_aux_fast: Try to add d =", d, 3)
                is_candidate_alive_begin_r = np.copy(is_candidate_alive_end_r)
                is_candidate_alive_begin_r[d] = True
                scores_s = np.zeros(self.pop.C)
                scores_s[is_candidate_alive_begin_r] = - np.sum(np.equal(
                    preferences_borda_s[:, is_candidate_alive_begin_r],
                    np.min(preferences_borda_s[:, is_candidate_alive_begin_r],
                           1)[:, np.newaxis]
                ), 0)
#                self._mylogv("CM_AUX: scores_s =", scores_s, 3)
                normal_loser = np.where(
                    scores_s == np.min(scores_s)
                )[0][-1]  # Tie-breaking
                nb_manip_d = max(0,
                                 scores_s[d] - scores_s[normal_loser] + 
                                 (d < normal_loser)
                                 )
                self._mylogv("CM_aux_fast: nb_manip_d =", nb_manip_d, 3)
                if nb_manip_d < least_nb_manipulators_for_r:
                    least_nb_manipulators_for_r = nb_manip_d
                    best_d = d
            self._mylogv("CM_aux_fast: best_d =", best_d, 3)
            # Update variables for previous round
            nb_manipulators_used = max(nb_manipulators_used, 
                                       least_nb_manipulators_for_r)
            if nb_manipulators_used > n_max:
                self._mylogv("CM_aux_fast: Conclusion: nb_manipulators_used =", 
                             np.inf, 3)
                return np.inf
            is_candidate_alive_end_r[best_d] = True
            
        self._mylogv("CM_aux_fast: Conclusion: nb_manipulators_used =", 
                     nb_manipulators_used, 3)
        return nb_manipulators_used

    def _CM_aux_exact(self, c, n_max, preferences_borda_s):
        """Exact algorithm used for IM and CM (which is equivalent to UM)
        
        Arguments:
        c -- Integer. Candidate for which we want to manipulate.
        n_max -- Integer. Maximum number of manipulators allowed.
            IM --> put 1.
            CM, complete and exact --> put the current value of 
                sufficient_coalition_size[c] - 1 (we want to find the best 
                value for sufficient_coalition_size[c], even if it exceeds the 
                number of manipulators)
            CM, otherwise --> put the number of manipulators.
        preferences_borda_s -- 2d integer. Preferences of the sincere voters
            (in Borda format).

        Returns:
        n_manip_exact -- Integer or +inf. 
            If a manipulation is found with n_max manipulators or less,
            the minimal sufficient number of manipulators is returned.
            Otherwise, it is +inf.
        """
        # Explore subsets that are reachable with less than the
        # upper bound.
        # situations_end_r is a dictionary.
        # keys: is_candidate_alive_end_r, tuple of booleans. 
        # values: nb_manip_used_after_r, number of manipulators used after 
        #   round d to go from situation is_candidate_alive_end_r to the
        #   singleton c.
        # Example: (1,1,0,1,0):3 means that if we have candidates 0, 1 and 3, 
        # we need 3 manipulators to make c win at the end.
        situations_end_r = {tuple(np.array(range(self.pop.C)) == c): 0}
        situations_begin_r = {}
        for r in range(self.pop.C - 2, -1, -1):
            self._mylogv("CM_aux_exact: Round r =", r, 3)
            situations_begin_r = {}
            for is_candidate_alive_end_r, nb_manip_used_after_r in \
                    situations_end_r.items():
                self._mylogv("CM_aux_exact: is_candidate_alive_end_r =", 
                             is_candidate_alive_end_r, 3)
                self._mylogv("CM_aux_exact: nb_manip_used_after_r =", 
                             nb_manip_used_after_r, 3)
                for d in range(self.pop.C):
                    if is_candidate_alive_end_r[d]:
                        continue
                    self._mylogv("CM_aux_exact: d =", d, 3)
                    is_candidate_alive_begin_r = np.copy(
                        is_candidate_alive_end_r)
                    is_candidate_alive_begin_r[d] = True
                    scores_s = np.zeros(self.pop.C)
                    scores_s[is_candidate_alive_begin_r] = - np.sum(np.equal(
                        preferences_borda_s[:, is_candidate_alive_begin_r],
                        np.min(
                            preferences_borda_s[:, is_candidate_alive_begin_r],
                            1)[:, np.newaxis]
                    ), 0)
#                    self._mylogv("scores_s =", scores_s, 3)
                    normal_loser = np.where(
                        scores_s == np.nanmin(scores_s)
                    )[0][-1]  # Tie-breaking
                    nb_manip_r = max(
                        scores_s[d] - scores_s[normal_loser] +
                        (d < normal_loser),
                        0
                    )
                    self._mylogv("CM_aux_exact: nb_manip_r =", nb_manip_r, 3)
                    nb_manip_r_and_after = max(nb_manip_r, 
                                               nb_manip_used_after_r)
                    if nb_manip_r_and_after > n_max:
                        continue
                    if tuple(is_candidate_alive_begin_r) in \
                            situations_begin_r:
                        situations_begin_r[
                            tuple(is_candidate_alive_begin_r)
                        ] = min(
                            situations_begin_r[
                                tuple(is_candidate_alive_begin_r)],
                            nb_manip_r_and_after
                        )
                    else:
                        situations_begin_r[tuple(
                            is_candidate_alive_begin_r)] = nb_manip_r_and_after
            self._mylogv("CM_aux_exact: situations_begin_r =",
                         situations_begin_r, 3)
            if len(situations_begin_r) == 0:
                self._mylog("CM_aux_exact: Manipulation is impossible with " + 
                            "n_max manipulators.", 3)
                return np.inf  # By convention
            situations_end_r = situations_begin_r
        else:
            self._mylogv("CM_aux_exact: situations_begin_r:",
                         situations_begin_r, 3)
            is_candidate_alive_begin, nb_manip_used = \
                situations_begin_r.popitem()
            self._mylogv("CM_aux_exact: is_candidate_alive_begin:", 
                         is_candidate_alive_begin, 3)
            self._mylogv("CM_aux_exact: Conclusion: "
                         "nb_manip_used_new =",
                         nb_manip_used, 3)
        return nb_manip_used
        
    #%% Individual manipulation (IM)

    def _IM_main_work_v(self, v, c_is_wanted,
                        nb_wanted_undecided, stop_if_true):
        preferences_borda_s = self.pop.preferences_borda_vtb[
            np.array(range(self.pop.V)) != v, :]
        for c in range(self.pop.C):
            if not c_is_wanted[c]:
                continue
            if not np.isneginf(self._v_IM_for_c[v, c]):
                continue
            self._mylogv('IM: Candidate c =', c, 3)
            n_manip_fast = self._CM_aux_fast(
                c, n_max=1,
                preferences_borda_s=preferences_borda_s)
            self._mylogv("IM: Fast: n_manip_fast =", n_manip_fast, 3)
            if n_manip_fast <= 1:
                self._v_IM_for_c[v, c] = True
                self._candidates_IM[c] = True
                self._voters_IM[v] = True
                self._is_IM = True
                if stop_if_true:
                    return
                continue
            if self.IM_option != 'exact':
                self._v_IM_for_c[v, c] = np.nan
                continue
            n_manip_exact = self._CM_aux_exact(
                c, n_max=1,
                preferences_borda_s=preferences_borda_s)
            self._mylogv("IM: Exact: n_manip_exact =", n_manip_exact, 3)
            if n_manip_exact <= 1:
                self._v_IM_for_c[v, c] = True
                self._candidates_IM[c] = True
                self._voters_IM[v] = True
                self._is_IM = True
                if stop_if_true:
                    return
            else:
                self._v_IM_for_c[v, c] = False

    #%% Trivial Manipulation (TM)

    # Use the generic methods from class Election.

    #%% Unison manipulation (UM)

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

    def _CM_main_work_c(self, c, optimize_bounds):
        n_m = self.pop.matrix_duels[c, self.w]
        exact = (self.CM_option == "exact")
        if optimize_bounds and exact:
            n_max = self._sufficient_coalition_size_CM[c] - 1
        else:
            n_max = n_m
        self._mylogv("CM: n_max =", n_max, 3)
        if not exact and self._necessary_coalition_size_CM[c] > n_max:
            self._mylog("CM: Fast algorithm will not do better than " +
                        "what we already know", 3)
            return
        n_manip_fast = self._CM_aux_fast(
            c, n_max,
            preferences_borda_s=self.pop.preferences_borda_vtb[
                np.logical_not(self.v_wants_to_help_c[:, c]), :]
        )
        self._mylogv("CM: n_manip_fast =", n_manip_fast, 3)
        self._update_sufficient(
            self._sufficient_coalition_size_CM, c, n_manip_fast,
            'CM: Update sufficient_coalition_size_CM =')
        if not exact:
            # With fast algo, we stop here anyway. It is not a "quick escape"
            # (if we'd try again with optimize_bound, we would not try better).
            return    

        # From this point, we have necessarily the 'exact' option
        if self._sufficient_coalition_size_CM[c] == (
                self._necessary_coalition_size_CM[c]):
            return
        if not optimize_bounds and (
                n_m >= self._sufficient_coalition_size_CM[c]):
            # This is a quick escape: since we have the option 'exact', if 
            # we come back with optimize_bound, we will try to be more precise.
            return True
        # Either we're in complete (and might have succeeded), or we in
        # incomplete mode (and we have failed)
        n_max_updated = min(n_manip_fast - 1, n_max)
        self._mylogv("CM: n_max_updated =", n_max_updated, 3)
        n_manip_exact = self._CM_aux_exact(
            c, n_max_updated,
            preferences_borda_s=self.pop.preferences_borda_vtb[
                np.logical_not(self.v_wants_to_help_c[:, c]), :])
        self._mylogv("CM: n_manip_exact =", n_manip_exact)
        n_manip = min(n_manip_fast, n_manip_exact)
        self._mylogv("CM: n_manip =", n_manip, 3)
        self._update_sufficient(
            self._sufficient_coalition_size_CM, c, n_manip,
            'CM: Update sufficient_coalition_size_CM =')
        # Update necessary coalition and return
        if optimize_bounds:
            self._necessary_coalition_size_CM[c] = (
                self._sufficient_coalition_size_CM[c])
            return False
        else:
            if n_m >= n_manip:
                # We have optimized the size of the coalition.
                self._necessary_coalition_size_CM[c] = (
                    self._sufficient_coalition_size_CM[c])
                return False
            else:
                # We have explored everything with n_max = n_m but 
                # manipulation failed. However, we have not optimized
                # sufficient_size (which must be higher than n_m), so it
                # is a quick escape.
                self._update_necessary(
                    self._necessary_coalition_size_CM, c, n_m + 1,
                    'CM: Update necessary_coalition_size_CM = n_m + 1 ='
                )
                return True


if __name__ == '__main__':
    # A quick demo
    preferences_utilities = np.random.randint(-5, 5, (10, 5))
    pop = Population(preferences_utilities)
    election = Coombs(pop)
    election.CM_option = 'exact'
    election.demo(log_depth=3)