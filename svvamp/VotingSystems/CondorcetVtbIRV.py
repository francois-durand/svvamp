# -*- coding: utf-8 -*-
"""
Created on Tue Oct 14 13:43:27 2014
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
from svvamp.Preferences.Population import preferences_utilities_to_matrix_duels
from svvamp.VotingSystems.Election import Election
from svvamp.VotingSystems.CondorcetVtbIRVResult import CondorcetVtbIRVResult
from svvamp.VotingSystems.IRV import IRV
from svvamp.VotingSystems.ExhaustiveBallot import ExhaustiveBallot


class CondorcetVtbIRV(CondorcetVtbIRVResult, Election):
    """Condorcet Instant Runoff Voting
    
    Inherits functions and optional parameters from superclasses
    :class:`~svvamp.ElectionResult` and :class:`~svvamp.Election`.

    :Example:

    >>> import svvamp
    >>> pop = svvamp.PopulationSpheroid(V=100, C=5)
    >>> election = svvamp.CondorcetVtbIRV(pop)

    Each voter must provide a strict total order.
    If there is a Condorcet winner (in the sense of
    :attr:`~svvamp.Population.matrix_victories_vtb`), then
    she is elected. Otherwise, :class:`~svvamp.IRV` is used.

    If sincere preferences are strict total orders, then this voting system is
    equivalent to :class:`~svvamp.CondorcetAbsIRV` for sincere voting,
    but manipulators have less possibilities (they are forced to provide
    strict total orders).

    :meth:`~svvamp.Election.CM`:

        * :attr:`~svvamp.Election.CM_option` = ``'fast'``:
          Rely on :class:`~svvamp.IRV`'s fast algorithm.
          Polynomial heuristic. Can prove CM but unable to decide non-CM
          (except in rare obvious cases).
        * :attr:`~svvamp.Election.CM_option` = ``'slow'``:
          Rely on :class:`~svvamp.ExhaustiveBallot`'s
          exact algorithm.
          Non-polynomial heuristic (:math:`2^C`). Quite efficient to prove CM
          or non-CM.
        * :attr:`~svvamp.Election.CM_option` = ``'almost_exact'``:
          Rely on :class:`~svvamp.IRV`'s exact algorithm.
          Non-polynomial heuristic (:math:`C!`). Very efficient to prove CM
          or non-CM.
        * :attr:`~svvamp.Election.CM_option` = ``'exact'``:
          Non-polynomial algorithm from superclass :class:`~svvamp.Election`.

        Each algorithm above exploits the faster ones. For example,
        if :attr:`~svvamp.Election.CM_option` = ``'almost_exact'``, SVVAMP
        tries the fast algorithm first, then the slow one, then the
        'almost exact' one. As soon as it reaches a decision, computation
        stops.

    :meth:`~svvamp.Election.ICM`: Exact in polynomial time.

    :meth:`~svvamp.Election.IM`: Non-polynomial
    or non-exact algorithms from superclass :class:`~svvamp.Election`.

    :meth:`~svvamp.Election.not_IIA`: Non-polynomial
    or non-exact algorithms from superclass :class:`~svvamp.Election`.

    :meth:`~svvamp.Election.TM`: Exact in polynomial time.

    :meth:`~svvamp.Election.UM`: Non-polynomial or non-exact algorithms from
    superclass :class:`~svvamp.Election`.

    References:

        'Condorcet criterion, ordinality and reduction of coalitional
        manipulability', François Durand, Fabien Mathieu and Ludovic Noirie,
        working paper, 2014.

    .. seealso:: :class:`~svvamp.ExhaustiveBallot`,
                 :class:`~svvamp.IRV`,
                 :class:`~svvamp.IRVDuels`,
                 :class:`~svvamp.ICRV`,
                 :class:`~svvamp.CondorcetAbsIRV`.
    """

    _layout_name = 'VTB-Condorcet IRV'
    _options_parameters = Election._options_parameters.copy()
    _options_parameters.update(CondorcetVtbIRVResult._options_parameters)
    _options_parameters['CM_option'] = {'allowed': {'fast', 'slow',
                                                    'almost_exact', 'exact'},
                                        'default': 'fast'}
    _options_parameters['TM_option'] = {'allowed': ['exact'],
                                        'default': 'exact'}
    _options_parameters['ICM_option'] = {'allowed': {'exact'},
                                           'default': 'exact'}

    def __init__(self, population, **kwargs):
        self.IRV = IRV(population, freeze_options=True)
        super().__init__(population, **kwargs)
        self._log_identity = "CONDORCET_VTB_IRV"
        self._class_result = CondorcetVtbIRVResult
        self._with_two_candidates_reduces_to_plurality = True
        self._is_based_on_strict_rankings = True
        self._meets_majority_favorite_c_vtb_ctb = True
        self._meets_Condorcet_c_vtb = True
        self._precheck_UM = False
        self._precheck_ICM = False

    #%% Individual manipulation (IM)

    # TODO: should be implemented.

    #%% Trivial Manipulation (TM)

    # Use the general methods from class Election.

    #%% Unison manipulation (UM)

    # TODO: should be implemented .

    #%% Ignorant-Coalition Manipulation (ICM)

    # Use the general methods from class Election.

    #%% Coalition Manipulation (CM)

    @property
    def losing_candidates(self):
        """If IRV.w does not win, then we put her first. Other losers are
        sorted as usual. (scores in matrix_duels).
        """
        if self._losing_candidates is None:
            self._mylog("Compute ordered list of losing candidates", 1)
            if self.w == self.IRV.w:
                # As usual
                self._losing_candidates = np.concatenate((
                    np.array(range(0, self.w), dtype=int),
                    np.array(range(self.w+1, self.pop.C), dtype=int)
                ))
                self._losing_candidates = self._losing_candidates[np.argsort(
                    -self.pop.matrix_duels[self._losing_candidates, self.w],
                    kind='mergesort')]
            else:
                # Put IRV.w first.
                self._losing_candidates = np.array(
                    [c for c in range(self.pop.C)
                     if c != self.w and c != self.IRV.w]
                ).astype(np.int)
                self._losing_candidates = self._losing_candidates[np.argsort(
                    -self.pop.matrix_duels[self._losing_candidates, self.w],
                    kind='mergesort')]
                self._losing_candidates = np.concatenate((
                    [self.IRV.w], self._losing_candidates))
        return self._losing_candidates

    def _CM_aux_almost_exact(self, c, n_m, suggested_path,
                             preferences_borda_s, matrix_duels_temp):
        """'Almost exact' algorithm used for CM.
        
        Arguments:
        c -- Integer. Candidate for which we want to manipulate.
        n_m -- Integer. The exact number of manipulators.
        suggested_path -- A suggested path of elimination (for the IRV part).
            It must work with n_m manipulators.
        preferences_borda_s -- 2d integer. Preferences of the sincere voters
            (in Borda format).
            
        Returns:
        manipulation_found -- Boolean.
        """ 
        candidates = np.array(range(self.pop.C))
        
        # Step 1: elimination path
        # And consequences on the majority matrix
        scores_m_begin_r = np.zeros(self.pop.C)
        is_candidate_alive_begin_r = np.ones(self.pop.C, dtype=np.bool)
        current_top_v = - np.ones(n_m)  # -1 means that v is available
        candidates_to_put_in_ballot = np.ones((n_m, self.pop.C), dtype=np.bool)
        for r in range(self.pop.C - 1):
            self._mylogv("CM_aux: r =", r, 3)
            scores_tot_begin_r = np.full(self.pop.C, np.nan)
            scores_tot_begin_r[is_candidate_alive_begin_r] = np.sum(np.equal(
                preferences_borda_s[:, is_candidate_alive_begin_r],
                np.max(preferences_borda_s[
                    :, is_candidate_alive_begin_r], 1)[:, np.newaxis]
            ), 0)
            self._mylogv("CM_aux: scores_s_begin_r =", scores_tot_begin_r, 3)
            self._mylogv("CM_aux: scores_m_begin_r =", scores_m_begin_r, 3)
            scores_tot_begin_r += scores_m_begin_r
            self._mylogv("CM_aux: scores_tot_begin_r =", scores_tot_begin_r, 3)
            d = suggested_path[r]
            self._mylogv("CM_aux: d =", d, 3)
            scores_m_new_r = np.zeros(self.pop.C, dtype=np.int)
            scores_m_new_r[is_candidate_alive_begin_r] = np.maximum(
                0,
                scores_tot_begin_r[d] -
                scores_tot_begin_r[is_candidate_alive_begin_r] +
                (candidates[is_candidate_alive_begin_r] > d))
            self._mylogv("CM_aux: scores_m_new_r =", scores_m_new_r, 3)
            # Update variables for next round
            scores_m_begin_r = scores_m_begin_r + scores_m_new_r
            if np.sum(scores_m_begin_r) > n_m:
                raise NotImplemented("Error: this should not happen.")
            scores_m_begin_r[d] = 0
            is_candidate_alive_begin_r[d] = False
            # We need to attribute manipulator's votes to specific 
            # manipulators. This is done arbitrarily, and has consequences
            # on the majority matrix. It is the main cause that makes the 
            # algorithm not exact in theory (the other one being step 3).
            free_manipulators = np.where(current_top_v == -1)[0]
            i_manipulator = 0
            for e in range(self.pop.C):
                n_manip_new_e = scores_m_new_r[e]
                for k in range(n_manip_new_e):
                    manipulator = free_manipulators[i_manipulator]
                    current_top_v[manipulator] = e
                    candidates_to_put_in_ballot[manipulator, e] = False
                    matrix_duels_temp[
                        e, candidates_to_put_in_ballot[manipulator, :]] += 1
                    i_manipulator += 1
            current_top_v[current_top_v == d] = -1
            self._mylogv("CM_aux: current_top_v =", current_top_v, 3)
            self._mylogm("CM_aux: matrix_duels_temp =", matrix_duels_temp, 3)
            
        # Step 2
        # Ensure that no candidate != c is Condorcet winner.
        # If c is not yet in all ballots, put it.
        for manipulator in np.where(candidates_to_put_in_ballot[:, c])[0]:
            candidates_to_put_in_ballot[manipulator, c] = False
            matrix_duels_temp[
                c, candidates_to_put_in_ballot[manipulator, :]] += 1
        self._mylogv("CM_aux: Adding to all ballots c =", c, 3)
        self._mylogm("CM_aux: matrix_duels_temp =", matrix_duels_temp, 3)
        # If some candidates already have some non-victories in the matrix
        # of duels, they can safely be put in the ballots. Maybe this will
        # generate non-victories for other candidates, etc.
        candidates_ok = np.zeros(self.pop.C, dtype=np.bool)
        candidates_ok[c] = True
        I_found_a_new_ok = True
        while I_found_a_new_ok:
            not_yet_ok = np.where(np.logical_not(candidates_ok))[0]
            if not_yet_ok.shape[0] == 0:
                self._mylog("CM_aux: Decondorcification succeeded", 3)
                return True
            I_found_a_new_ok = False
            for d in not_yet_ok:
                if np.any(matrix_duels_temp[:, d] >= self.pop.V / 2):
                    candidates_ok[d] = True
                    I_found_a_new_ok = True
                    for manipulator in np.where(
                            candidates_to_put_in_ballot[:, d])[0]:
                        candidates_to_put_in_ballot[manipulator, d] = False
                        matrix_duels_temp[
                            d, candidates_to_put_in_ballot[manipulator, :]
                        ] += 1
                    self._mylogv("CM_aux: Found a non-Condorcet d =", d, 3)
                    self._mylogm("CM_aux: matrix_duels_temp =",
                                 matrix_duels_temp, 3)
                                
        # Step 3
        # Some candidates have left, who do not have non-victories yet.
        # We will put them in the ballots, while favoring a Condorcet cycle
        # 0 > 1 > ... > C-1 > 0. 
        # N.B.: In practice, this step seems never necessary.
        self._mylog("CM_aux: Step 3 needed", 1)
        for manipulator in range(n_m):
            candidate_start = manipulator % self.pop.C
            for d in np.concatenate((range(candidate_start, self.pop.C), 
                                     range(candidate_start))):
                if candidates_to_put_in_ballot[manipulator, d]:
                    candidates_to_put_in_ballot[manipulator, d] = False
                    matrix_duels_temp[
                        d, candidates_to_put_in_ballot[manipulator, :]] += 1
        for d in np.where(np.logical_not(candidates_ok))[0]:
            if np.all(matrix_duels_temp[:, d] < self.pop.V / 2):
                self._mylog("CM_aux: Decondorcification failed", 1)
                return False
        self._mylog("CM_aux: Decondorcification succeeded", 1)
        return True
                
    def _CM_main_work_c(self, c, optimize_bounds):
        n_m = self.pop.matrix_duels[c, self.w]
        n_s = self.pop.V - n_m
        candidates = np.array(range(self.pop.C))
        preferences_borda_s = self.pop.preferences_borda_vtb[
            np.logical_not(self.v_wants_to_help_c[:, c]), :]
        matrix_duels_vtb_temp = (
            preferences_utilities_to_matrix_duels(preferences_borda_s))
        self._mylogm("CM: matrix_duels_vtb_temp =", matrix_duels_vtb_temp, 3)
        # More preliminary checks
        # It's more convenient to put them in that method, because we need
        # preferences_borda_s and matrix_duels_vtb_temp.
        d_neq_c = (np.array(range(self.pop.C)) != c)
        n_manip_becomes_cond = np.maximum(
            n_s + 1 - 2 * np.min(matrix_duels_vtb_temp[c, d_neq_c]),
            0)
        self._mylogv("CM: n_manip_becomes_cond =", n_manip_becomes_cond, 3)
        self._update_sufficient(
            self._sufficient_coalition_size_CM, c,
            n_manip_becomes_cond,
            'CM: Update sufficient_coalition_size_CM[c] = '
            'n_manip_becomes_cond =')
        if not optimize_bounds and (
                n_m >= self._sufficient_coalition_size_CM[c]):
            return True
        # Prevent another cond. Look at the weakest duel for d, she has
        # matrix_duels_vtb_temp[d, e]. We simply need that:
        # matrix_duels_vtb_temp[d, e] <= (n_s + n_m) / 2
        # 2 * max_d(min_e(matrix_duels_vtb_temp[d, e])) - n_s <= n_m
        n_manip_prevent_cond = 0
        for d in candidates[d_neq_c]:
            e_neq_d = (np.array(range(self.pop.C)) != d)
            n_prevent_d = np.maximum(
                2 * np.min(matrix_duels_vtb_temp[d, e_neq_d]) - n_s, 0)
            n_manip_prevent_cond = max(n_manip_prevent_cond, n_prevent_d)
        self._mylogv("CM: n_manip_prevent_cond =", n_manip_prevent_cond, 3)
        self._update_necessary(
            self._necessary_coalition_size_CM, c,
            n_manip_prevent_cond,
            'CM: Update necessary_coalition_size_CM[c] = '
            'n_manip_prevent_cond =')
        if not optimize_bounds and (
                self._necessary_coalition_size_CM[c] > n_m):
            return True

        # Let us work
        if self.w == self.IRV.w:
            self._mylog('CM: c != self.IRV.w == self.w', 3)
            if self.CM_option == "fast":
                self.IRV.CM_option = "fast"
            elif self.CM_option == "slow":
                self.IRV.CM_option = "slow"
            else:
                self.IRV.CM_option = "exact"
            # Use IRV without bounds
            self._mylog('CM: Use IRV without bounds')
            irv_is_CM_c = self.IRV.CM_c(c)[0]
            if irv_is_CM_c == True:
                suggested_path_one = self.IRV._example_path_CM[c]
                self._mylogv("CM: suggested_path =", suggested_path_one, 3)
                manipulation_found = self._CM_aux_almost_exact(
                    c, n_m, suggested_path_one,
                    preferences_borda_s, matrix_duels_vtb_temp)
                self._mylogv("CM: manipulation_found =", manipulation_found, 3)
                if manipulation_found:
                    self._update_sufficient(
                        self._sufficient_coalition_size_CM, c, n_m,
                        'CM: Update sufficient_coalition_size_CM[c] = n_m =')
                    if not optimize_bounds:
                        return True
            else:
                suggested_path_one = np.zeros(self.pop.C)
            if irv_is_CM_c == False:
                self._mylog('CM: IRV.CM_c[c] = False', 3)
                self._update_necessary(
                    self._necessary_coalition_size_CM, c,
                    min(n_manip_becomes_cond,
                        max(n_m + 1,
                            n_manip_prevent_cond)),
                    'CM: Update necessary_coalition_size[c] =')
                if not optimize_bounds:
                    return True
            if self._sufficient_coalition_size_CM[c] == \
                    self._necessary_coalition_size_CM[c]:
                return False
            # Use IRV with bounds
            # Either we have not decided manipulation for c, or it is
            # decided to False and optimize_bounds = True (in that second
            # case, we can only improve necessary_coalition_size[c]).
            self._mylog('CM: Use IRV with bounds')
            self.IRV.CM_c_with_bounds(c)
            self._update_necessary(
                self._necessary_coalition_size_CM, c,
                min(n_manip_becomes_cond,
                    max(self.IRV._necessary_coalition_size_CM[c],
                        n_manip_prevent_cond)),
                'CM: Update necessary_coalition_size[c] =')
            if self.IRV._sufficient_coalition_size_CM[c] <= n_m:
                suggested_path_two = self.IRV._example_path_CM[c]
                self._mylogv("CM: suggested_path =", suggested_path_two, 3)
                if np.equal(suggested_path_one, suggested_path_two):
                    self._mylog('CM: Same suggested path as before, '
                                'skip computation')
                else:
                    manipulation_found = self._CM_aux_almost_exact(
                        c, n_m, suggested_path_two,
                        preferences_borda_s, matrix_duels_vtb_temp)
                    self._mylogv("CM: manipulation_found =",
                                 manipulation_found, 3)
                    if manipulation_found:
                        self._update_sufficient(
                            self._sufficient_coalition_size_CM, c, n_m,
                            'CM: Update sufficient_coalition_size_CM[c] = '
                            'n_m =')
                        # We will not do better with exact (brute force)
                        # algorithm.
                        return False
        else:
            if c == self.IRV.w:
                self._mylog('CM: c == self.IRV.w != self.w', 3)
                suggested_path = self.IRV.elimination_path
                self._mylogv("CM: suggested_path =", suggested_path, 3)
                manipulation_found = self._CM_aux_almost_exact(
                    c, n_m, suggested_path,
                    preferences_borda_s, matrix_duels_vtb_temp)
                self._mylogv("CM: manipulation_found =", manipulation_found, 3)
                if manipulation_found:
                    self._update_sufficient(
                        self._sufficient_coalition_size_CM, c, n_m,
                        'CM: Update sufficient_coalition_size_CM[c] = '
                        'n_m =')
                    # We will not do better with exact (brute force)
                    # algorithm.
                    return
            else:
                self._mylog('CM: c, self.IRV.w and self.w are all distinct', 3)
        if self.CM_option == 'exact':
            return self._CM_main_work_c_exact(c, optimize_bounds)


if __name__ == '__main__':
    # A quick demo

    preferences_utilities = np.random.randint(-5, 5, (8, 4))
    pop = Population(preferences_utilities)
    eb = ExhaustiveBallot(pop)
    eb._log_depth = 3
    eb.CM_option = 'exact'
    print(eb.CM())
    irv = IRV(pop)
    irv._log_depth = 3
    irv.EB._log_depth = 3
    irv.CM_option = 'exact'
    print(irv.CM())
    election = CondorcetVtbIRV(pop)
    election._log_depth = 3
    election.IRV._log_depth = 3
    election.IRV.EB._log_depth = 3
    election.CM_option = 'slow'
    print(election.CM())
    # election.demo(log_depth=3)