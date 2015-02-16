# -*- coding: utf-8 -*-
"""
Created on Thu Oct  2 07:45:14 2014
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
from svvamp.VotingSystems.BucklinResult import BucklinResult
from svvamp.Preferences.Population import Population


class Bucklin(BucklinResult, Election):
    """Bucklin method.

    At counting round r, all voters who rank candidate c in r-th position
    gives her an additional vote. As soon as at least one candidate has more 
    than V/2 votes (accrued with previous rounds), the candidate with most
    votes is declared the winner.

    In case of a tie, the candidate with lowest index wins.    
    """
    
    _layout_name = 'Bucklin'
    _options_parameters = Election._options_parameters.copy()
    _options_parameters.update(BucklinResult._options_parameters)
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
        self._log_identity = "BUCKLIN"
        self._class_result = Bucklin
        self._with_two_candidates_reduces_to_plurality = True
        self._is_based_on_strict_rankings = True
        self._meets_majority_favorite_c_vtb = True
        # N.B.: majority_favorite_c_ctb is not met.
        self._precheck_ICM = False
        # Bucklin does not meet InfMC_c_ctb, but precheck on ICM is not
        # interesting anyway.

    #%% Individual manipulation (IM)

    def _IM_main_work_v(self, v, c_is_wanted,
                        nb_wanted_undecided, stop_if_true):
        scores_without_v = np.copy(self.scores)
        for k in range(self.pop.C):
            scores_without_v[range(k, self.pop.C), 
                             self.pop.preferences_ranking[v, k]] -= 1
        for c in range(self.pop.C):
            if not c_is_wanted[c]:
                continue
            if not np.isneginf(self._v_IM_for_c[v, c]):
                continue
            nb_wanted_undecided -= 1
            # r : round where c will have majority (with the manipulator).
            r = np.where(scores_without_v[:, c] + 1 > self.pop.V / 2)[0][0]
            if r == 0:
                self._v_IM_for_c[v, c] = True
                self._candidates_IM[c] = True
                self._voters_IM[v] = True
                self._is_IM = True
                self._mylog("IM found", 3)
                if stop_if_true:
                    return
                if nb_wanted_undecided == 0:
                    return
                continue
            scores_prev = np.copy(scores_without_v[r-1, :])
            scores_prev[c] += 1
            scores_r = np.copy(scores_without_v[r, :])
            scores_r[c] += 1
            # Obvious problems
            if np.argmax(scores_r) != c:
                # One d has a better score than c!
                self._v_IM_for_c[v, c] = False
                if nb_wanted_undecided == 0:
                    return
                continue
            if np.max(scores_prev) > self.pop.V / 2:
                # One d reaches majority before c!
                self._v_IM_for_c[v, c] = False
                if nb_wanted_undecided == 0:
                    return
                continue

            # Now, attribute other ranks in manipulator's ballots
            # For d to be added safely at rank r (corresponding to last round), 
            # we just need that d will not outperform c at rank r.
            d_can_be_added = np.less(
                scores_r + 1,
                scores_r[c] + (c < np.array(range(self.pop.C))))
            d_can_be_added[c] = False 
            # For d to be added safely before rank r, we need also that d will 
            # not have a majority before round r.
            d_can_be_added_before_last_round = np.logical_and(
                d_can_be_added,
                scores_prev + 1 <= self.pop.V / 2)
            
            # We can conclude.
            self._v_IM_for_c[v, c] = (
                np.sum(d_can_be_added) >= r - 1 and
                np.sum(d_can_be_added_before_last_round) >= r - 2)
            if self._v_IM_for_c[v, c] == True:
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

    #%% Trivial Manipulation (TM)

    # Defined in the superclass Election.

    #%% Unison manipulation (UM)

    def _UM_main_work_c(self, c):
        n_m = self.pop.matrix_duels[c, self.w]
        scores_r = np.zeros(self.pop.C)
        # Manipulators put c in first position anyway.
        scores_r[c] = n_m
        # Look for the round r where c has a majority. Compute how many 
        # sincere votes each candidate will have at that time, + the n_m
        # manipulating votes that we know for c.
        r = None
        scores_prev = None
        for r in range(self.pop.C):
            scores_prev = np.copy(scores_r)
            scores_r += np.bincount(
                self.pop.preferences_ranking[np.logical_not(
                    self.v_wants_to_help_c[:, c]), r],
                minlength=self.pop.C)
            if scores_r[c] > self.pop.V / 2:  # It is the last round
                if np.argmax(scores_r) != c:
                    # One d has a better score than c!
                    self._candidates_UM[c] = False
                    return
                break
            if np.max(scores_r) > self.pop.V / 2:
                # One d reaches majority before c!
                self._candidates_UM[c] = False
                return

        # Now, attribute other ranks in manipulator's ballots
        # For d to be added safely at rank r (corresponding to last round), we 
        # just need that d will not outperform c at rank r.
        d_can_be_added = np.less(
            scores_r + n_m,
            scores_r[c] + (c < np.array(range(self.pop.C))))
        d_can_be_added[c] = False 
        # For d to be added safely before rank r, we need also that d will 
        # not have a majority before round r.
        d_can_be_added_before_last_round = np.logical_and(
            d_can_be_added,
            scores_prev + n_m <= self.pop.V / 2)
        
        # We can conclude.
        self._candidates_UM[c] = (
            np.sum(d_can_be_added) >= r - 1 and
            np.sum(d_can_be_added_before_last_round) >= r - 2)

    #%% Ignorant-Coalition Manipulation (ICM)

    def _ICM_main_work_c_exact(self, c, complete_mode=True):
        # The only question is when we have exactly V/2 manipulators. If
        # the counter-manipulators put c last, then c cannot be elected
        # (except if there are 2 candidates and c == 0).
        # So exactly V/2 manipulators is not enough.
        n_s = self.pop.V - self.pop.matrix_duels[c, self.w]
        if self.pop.C == 2 and c == 0:
            self._update_sufficient(
                self._sufficient_coalition_size_ICM, c, n_s,
                'ICM: Tie-breaking: '
                'sufficient_coalition_size_ICM = n_s =')
        else:
            self._update_necessary(
                self._necessary_coalition_size_ICM, c, n_s + 1,
                'ICM: Tie-breaking: '
                'necessary_coalition_size_ICM = n_s + 1 =')

    #%% Coalition Manipulation (CM)

    def _CM_main_work_c_exact(self, c, optimize_bounds):
        # We do not try to find optimal bounds. We just check whether it
        # is possible to manipulate with the number of manipulators that
        # we have.
        n_m = self.pop.matrix_duels[c, self.w]
        if n_m < self._necessary_coalition_size_CM[c]:
            # This algorithm will not do better (so, this is not a
            # quick escape).
            return
        if n_m >= self._sufficient_coalition_size_CM[c]:
            # Idem.
            return
        scores_r = np.zeros(self.pop.C)
        # Manipulators put c in first position anyway.
        scores_r[c] = n_m
        # Look for the round r where c has a majority. Compute how many 
        # sincere votes each candidate will have at that time, + the n_m
        # manipulating votes that we know for c.
        r = None
        for r in range(self.pop.C):
            scores_prev = np.copy(scores_r)
            scores_r += np.bincount(
                self.pop.preferences_ranking[np.logical_not(
                    self.v_wants_to_help_c[:, c]), r],
                minlength=self.pop.C)
            if scores_r[c] > self.pop.V / 2:
                break

        votes_can_be_added_before_last_r = np.zeros(self.pop.C)
        votes_can_be_added = np.zeros(self.pop.C)
        one_d_beats_c_anyway = False
        for d in range(self.pop.C):
            if d == c:
                continue
            if (scores_r[d] + (d < c) > scores_r[c] or 
                    scores_prev[d] > self.pop.V/2):
                one_d_beats_c_anyway = True
                break
            votes_can_be_added[d] = min(
                scores_r[c] - (d < c) - scores_r[d],
                n_m)
            votes_can_be_added_before_last_r[d] = min(
                np.floor(self.pop.V/2) - scores_prev[d],
                votes_can_be_added[d])
                    
        if (not one_d_beats_c_anyway and
                sum(votes_can_be_added) >= (r - 1) * n_m and
                sum(votes_can_be_added_before_last_r) >= (r - 2) * n_m):
            self._update_sufficient(
                self._sufficient_coalition_size_CM, c, n_m,
                'CM: Exact: Manipulation found for n_m manipulators =>\n'
                '    sufficient_coalition_size_CM = n_m =')
        else:
            self._update_necessary(
                self._necessary_coalition_size_CM, c, n_m + 1,
                'CM: Exact: Manipulation proven impossible for n_m '
                'manipulators =>\n'
                '    necessary_coalition_size_CM[c] = n_m + 1 =')


if __name__ == '__main__':
    # A quick demo
    preferences_utilities = np.random.randint(-5, 5, (10, 5))
    pop = Population(preferences_utilities)
    election = Bucklin(pop)
    election._log_depth = 3
    print(election.CM_with_candidates())
    # election.demo(log_depth=3)