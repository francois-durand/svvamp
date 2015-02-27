# -*- coding: utf-8 -*-
"""
Created on Sat Oct  4 19:11:12 2014
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
import networkx as nx

from svvamp.VotingSystems.Election import Election
from svvamp.VotingSystems.MaximinResult import MaximinResult
from svvamp.Preferences.Population import Population
from svvamp.Preferences.Population import preferences_ut_to_matrix_duels_ut


class Maximin(MaximinResult, Election):
    """Maximin method.

    Inherits functions and optional parameters from superclasses
    :class:`~svvamp.ElectionResult` and :class:`~svvamp.Election`.

    :Example:

    >>> import svvamp
    >>> pop = svvamp.PopulationSpheroid(V=100, C=5)
    >>> election = svvamp.Maximin(pop)

    Candidate ``c``'s score is the minimum of the row
    :attr:`~svvamp.Population.matrix_duels_rk`\ ``[c, :]`` (except the
    diagonal term), i.e. the result of candidate ``c`` for her worst duel.
    The candidate with highest score is declared the winner. In case of a tie,
    the candidate with lowest index wins.

    This method meets the Condorcet criterion.

    :meth:`~svvamp.Election.CM`: Deciding CM is NP-complete, even for 2
    manipulators.

        * :attr:`~svvamp.Election.CM_option` = ``'fast'``:
          Zuckerman et al. (2011). This approximation algorithm is
          polynomial and has a multiplicative factor of error of 5/3 on the
          number of manipulators needed.
        * :attr:`~svvamp.Election.CM_option` = ``'exact'``:
          Non-polynomial algorithm from superclass :class:`~svvamp.Election`.

    :meth:`~svvamp.Election.ICM`: Exact in polynomial time.

    :meth:`~svvamp.Election.IM`: Exact in polynomial time.

    :meth:`~svvamp.Election.not_IIA`: Exact in polynomial time.

    :meth:`~svvamp.Election.TM`: Exact in polynomial time.

    :meth:`~svvamp.Election.UM`: Exact in polynomial time.

    References:

        'Complexity of Unweighted Coalitional Manipulation under Some Common
        Voting Rules', Lirong Xia et al., 2009.

        'An algorithm for the coalitional manipulation problem under Maximin',
        Michael Zuckerman, Omer Lev and Jeffrey S. Rosenschein, 2011.
    """

    _layout_name = 'Maximin'
    _options_parameters = Election._options_parameters.copy()
    _options_parameters.update(MaximinResult._options_parameters)
    _options_parameters['IM_option'] = {'allowed': ['exact'],
                                        'default': 'exact'}
    _options_parameters['TM_option'] = {'allowed': ['exact'],
                                        'default': 'exact'}
    _options_parameters['UM_option'] = {'allowed': ['exact'],
                                        'default': 'exact'}
    _options_parameters['ICM_option'] = {'allowed': ['exact'],
                                           'default': 'exact'}
    _options_parameters['CM_option'] = {'allowed': ['fast', 'exact'],
                                        'default': 'fast'}

    def __init__(self, population, **kwargs):
        super().__init__(population, **kwargs)
        self._log_identity = "MAXIMIN"
        self._class_result = Maximin
        self._with_two_candidates_reduces_to_plurality = True
        self._is_based_on_rk = True
        self._meets_Condorcet_c_rk_ctb = True
        self._precheck_ICM = False

    def _vote_strategically(self, matrix_duels_r, scores_r, c, weight=1):
        """Strategic vote by one elector, in favor of candidate c.
        
        Modifies matrix_duels_r and scores_r IN PLACE.
        
        Arguments:
        matrix_duels_r -- 2d array of integers. Diagonal coefficients must 
        have an arbitrary value that is greater than all the other ones (+inf
        is a good idea).
        scores_r -- 1d array of integers. Scores corresponding to 
        matrix_duels_r.
        c -- Integer (candidate).
        weight -- Integer. Weight of the voter (used for UM, 1 otherwise).
        
        Algorithm from Zuckerman et al. : An Algorithm for the Coalitional 
        Manipulation Problem under Maximin. For only one elector (or unison
        manipulation), this algorithm is optimal.
        """    
        self._mylogv("AUX: c =", c, 3)
        self._mylogm("AUX: matrix_duels_r =", matrix_duels_r, 3)
        self._mylogv("AUX: scores_r =", scores_r, 3)
        # Manipulator's vote (denoted P_i in the paper) is represented in 
        # Borda format.
        strategic_ballot = np.zeros(self.pop.C)
        strategic_ballot[c] = self.pop.C
        next_borda_score_to_add = self.pop.C - 1
        has_been_sent_to_stacks = np.zeros(self.pop.C)
        candidates = np.array(range(self.pop.C))
        stacks_Q = StackFamily()
        
        # Building the directed graph (cf. paper).
        digraph = np.zeros((self.pop.C, self.pop.C))
        for x in range(self.pop.C):
            if x == c:
                continue
            if matrix_duels_r[x, c] == scores_r[x]:
                continue
            for y in range(self.pop.C):
                if y == c or y == x:
                    continue
                if matrix_duels_r[x, y] == scores_r[x]:
                    digraph[x, y] = True

        while next_borda_score_to_add > 0:
            self._mylogm("AUX: digraph =", digraph, 3)
            self._mylogv("AUX: strategic_ballot =", strategic_ballot, 3)
            # candidates_not_dangerous = set A in the paper
            candidates_not_dangerous = np.logical_and(
                np.logical_not(np.any(digraph, 1)),
                np.logical_not(strategic_ballot))
            self._mylogv("AUX: candidates_not_dangerous =",
                         candidates_not_dangerous, 3)
            if np.any(candidates_not_dangerous):
                # Transfer these candidates into the stacks
                for score, candidate in zip(
                        scores_r[candidates_not_dangerous],
                        candidates[candidates_not_dangerous]):
                    if not has_been_sent_to_stacks[candidate]:
                        stacks_Q.push_front(score, candidate)
                        has_been_sent_to_stacks[candidate] = True
                self._mylogv("AUX: stacks_Q =", stacks_Q.data, 3)
                # Put a non-dangerous candidate in the ballot
                b = stacks_Q.pop_front()
                self._mylogv("AUX: Found non-dangerous candidate b =", b, 3)
                self._mylogv("AUX: stacks_Q =", stacks_Q.data, 3)
                strategic_ballot[b] = next_borda_score_to_add
                next_borda_score_to_add -= 1
            else:
                s = np.min(scores_r[np.logical_not(strategic_ballot)])
                b_have_s = np.logical_and(
                    np.logical_not(strategic_ballot),
                    scores_r == s)
                self._mylogv("AUX: s =", s, 3)
                self._mylogv("AUX: b_have_s =", b_have_s, 3)
                # In the general case, we take any b with least score.
                # We choose the one with highest index.
                b = np.where(b_have_s)[0][-1]
                self._mylogv("AUX: First guess b =", b, 3)
                # But there might be a particular case (special cycle)...
                if sum(b_have_s) > 1:
                    found_special_b = False
                    G = nx.DiGraph(digraph)
                    self._mylogv("AUX: G.node = ", G.node, 3)
                    self._mylogv("AUX: G.edge = ", G.edge, 3)
                    # We look for b in reverse order of index. This way,
                    # if several candidates meet the condition, the one with 
                    # highest index is chosen.
                    for b_test in candidates[b_have_s][::-1]:
                        possible_cs = np.where(np.logical_and(
                            digraph[:, b_test],
                            b_have_s))
                        for c_test in possible_cs[0]:
                            self._mylogv("AUX: b_test = ", b_test, 3)
                            self._mylogv("AUX: c_test = ", c_test, 3)
                            if nx.has_path(G, b_test, c_test):
                                self._mylogv(
                                    "AUX: Found special b =", b_test, 3)
                                b = b_test
                                found_special_b = True
                                break
                        if found_special_b:
                            break
                # We put b in the ballot
                strategic_ballot[b] = next_borda_score_to_add
                next_borda_score_to_add -= 1
            # Now update the digraph
            self._mylogm("AUX: digraph =", digraph, 3)
            for y in candidates[np.logical_not(strategic_ballot)]:
                if digraph[y, b] and not has_been_sent_to_stacks[y]:
                    self._mylogv("AUX: y =", y, 3)
                    self._mylogv("AUX: b =", b, 3)
                    self._mylogv(
                        "AUX: Transfer from digraph to stack: y =", y, 3)
                    digraph[y, :] = False
                    stacks_Q.push_front(scores_r[y], y)
                    has_been_sent_to_stacks[y] = True
                    self._mylogv("AUX: stacks_Q =", stacks_Q.data, 3)
        # Finally, update matrix_duels_r and scores_r
        self._mylogm("AUX: matrix_duels_r =", matrix_duels_r, 3)
        self._mylogv("AUX: scores_r =", scores_r, 3)
        self._mylogv("AUX: strategic_ballot =", strategic_ballot, 3)
        for x in range(self.pop.C):
            for y in range(x+1, self.pop.C):
                if strategic_ballot[x] > strategic_ballot[y]:
                    matrix_duels_r[x, y] += weight
                else:
                    matrix_duels_r[y, x] += weight
        # The ugly thing below (instead of scores_r = 
        # np.min(matrix_duels_r, 1)) is made so that scores_r is modified
        # in place.
        scores_r_temp = np.min(matrix_duels_r, 1)
        for x in range(self.pop.C):
            scores_r[x] = scores_r_temp[x]
        self._mylogm("AUX: matrix_duels_r =", matrix_duels_r, 3)
        self._mylogv("AUX: scores_r =", scores_r, 3)

    #%% Individual manipulation (IM)

    def _IM_main_work_v(self, v, c_is_wanted,
                        nb_wanted_undecided, stop_if_true):
        for c in self.losing_candidates:
            if not c_is_wanted[c]:
                continue
            if not np.isneginf(self._v_IM_for_c[v, c]):
                continue
            self._mylogv("IM: c =", c, 3)
            nb_wanted_undecided -= 1
            
            matrix_duels_r = np.copy(self.pop.matrix_duels_ut).astype(np.float)
            for x in range(self.pop.C):
                matrix_duels_r[x, x] = np.inf
                for y in range(x+1, self.pop.C):
                    if (self.pop.preferences_borda_rk[v, x] >
                            self.pop.preferences_borda_rk[v, y]):
                        matrix_duels_r[x, y] -= 1
                    else:
                        matrix_duels_r[y, x] -= 1
            scores_r = np.min(matrix_duels_r, 1)
            
            w_temp = np.argmax(scores_r)
            if w_temp == c:
                self._mylogv("IM: scores_r =", scores_r, 3)
                self._mylog("IM: Manipulation easy "
                            "(c wins without manipulator's vote)", 3)
                self._v_IM_for_c[v, c] = True
                self._candidates_IM[c] = True
                self._voters_IM[v] = True
                self._is_IM = True
                self._mylog("IM found", 3)
                if stop_if_true or nb_wanted_undecided == 0:
                    return
                continue
            # Best we can do: improve c by one, do not move the others
            if scores_r[w_temp] > scores_r[c] + 1 or (
                    scores_r[w_temp] >= scores_r[c] + 1 and w_temp < c):
                self._mylogv("IM: scores_r =", scores_r, 3)
                self._mylog(
                    "IM: Manipulation impossible (score difference is too "
                    "high)", 3)
                self._v_IM_for_c[v, c] = False
                if nb_wanted_undecided == 0:
                    return
                continue
            
            self._vote_strategically(matrix_duels_r, scores_r, c, 1)
            w_r = np.argmax(scores_r)
            self._mylogv("IM: w_r =", w_r, 3)
            if w_r == c:
                self._mylog("IM: Manipulation worked!", 3)
            else:
                self._mylog("IM: Manipulation failed...", 3)

            # We can conclude.
            self._v_IM_for_c[v, c] = (w_r == c)
            if self._v_IM_for_c[v, c] == True:
                self._candidates_IM[c] = True
                self._voters_IM[v] = True
                self._is_IM = True
                self._mylog("IM found", 3)
                if stop_if_true:
                    return
            if nb_wanted_undecided == 0:
                return

    #%% Trivial Manipulation (TM)

    # Use the general methods.

    #%% Ignorant-Coalition Manipulation (ICM)

    # Use the general methods.

    #%% Unison manipulation (UM)

    def _UM_main_work_c(self, c):
        matrix_duels_temp = preferences_ut_to_matrix_duels_ut(
            self.pop.preferences_borda_rk[
                np.logical_not(self.v_wants_to_help_c[:, c]), :]
        ).astype(float)
        for x in range(self.pop.C):
            matrix_duels_temp[x, x] = np.inf
        scores_temp = np.min(matrix_duels_temp, 1)
        n_m = self.pop.matrix_duels_ut[c, self.w]
                
        w_temp = np.argmax(scores_temp)
        if w_temp == c:
            self._mylogv("UM: scores_temp =", scores_temp, 3)
            self._mylog("UM: Manipulation easy "
                        "(c wins without manipulators' votes)", 3)
            self._candidates_UM[c] = True
            return
        # Best we can do: improve c by n_m, do not move the others
        if scores_temp[w_temp] > scores_temp[c] + n_m or (
                scores_temp[w_temp] >= scores_temp[c] + n_m and w_temp < c):
            self._mylogv("UM: scores_temp =", scores_temp, 3)
            self._mylog("UM: Manipulation impossible (score " +
                        "difference is too high)", 3)
            self._candidates_UM[c] = False
            return
        
        self._vote_strategically(matrix_duels_temp, scores_temp, c, n_m)
        w_temp = np.argmax(scores_temp)
        self._mylogv("UM: w_temp =", w_temp, 3)
        self._candidates_UM[c] = (w_temp == c)

    #%% Coalition Manipulation (CM)

    def _CM_main_work_c_fast(self, c, optimize_bounds):
        matrix_duels_temp = preferences_ut_to_matrix_duels_ut(
            self.pop.preferences_borda_rk[
                np.logical_not(self.v_wants_to_help_c[:, c]), :]
        ).astype(float)
        for x in range(self.pop.C):
            matrix_duels_temp[x, x] = np.inf
        scores_temp = np.min(matrix_duels_temp, 1)
        w_temp = np.argmax(scores_temp)
        n_manipulators_used = 0
        
        # Easy lower bound: for each manipulator, c's score can at best 
        # increase by one, and w_temp's score cannot decrease.
        self._update_necessary(
            self._necessary_coalition_size_CM, c,
            scores_temp[w_temp] - scores_temp[c] + (c > w_temp),
            'CM: Update necessary_coalition_size_CM = '
            'scores_s[w_s] - scores_s[c] + (c > w_s) =')
        if not optimize_bounds and (self._necessary_coalition_size_CM[c] >
                                    self.pop.matrix_duels_ut[c, self.w]):
            return True  # is_quick_escape

        while w_temp != c:
            self._mylogv("CM: w_temp =", w_temp, 3)
            self._mylogv("CM: c =", c, 3)
            n_manipulators_used += 1
            if n_manipulators_used >= self._sufficient_coalition_size_CM[c]:
                self._mylog("CM: I already know a strategy that works " +
                            "with n_manipulators_used (TM, UM, etc.).")
                self._update_necessary(
                    self._necessary_coalition_size_CM, c,
                    np.ceil(n_manipulators_used * 3 / 5),
                    'CM: Update necessary_coalition_size_CM =')
                break
            self._vote_strategically(matrix_duels_temp, scores_temp, c, 1)
            w_temp = np.argmax(scores_temp)
        else:
            self._mylogm("CM: matrix_duels_temp =", matrix_duels_temp, 3)
            self._mylogv("CM: scores_temp =", scores_temp, 3)
            self._mylogv("CM: w_temp =", w_temp, 3)
            self._update_sufficient(
                self._sufficient_coalition_size_CM, c, n_manipulators_used,
                'CM: Update sufficient_coalition_size_CM =')
            # For the 3/5 ratio, see Zuckerman et al.
            self._update_necessary(
                self._necessary_coalition_size_CM, c,
                np.ceil(n_manipulators_used * 3 / 5),
                'CM: Update necessary_coalition_size_CM =')
        return False

    def _CM_main_work_c(self, c, optimize_bounds):
        is_quick_escape_fast = self._CM_main_work_c_fast(c, optimize_bounds)
        if not self.CM_option == "exact":
            # With 'fast' option, we stop here anyway.
            return is_quick_escape_fast

        # From this point, we have necessarily the 'exact' option (which is,
        #  in fact, only an exhaustive exploration with = n_m manipulators).
        is_quick_escape_exact = self._CM_main_work_c_exact(c, optimize_bounds)
        return is_quick_escape_fast or is_quick_escape_exact


class StackFamily:
    """Family of stacks. Used for the manipulation of Maximin.

    A StackFamily looks like {42:[4, 2, 3], 51:[0]}, i.e. a dictionary
    whose keys are numbers and values are lists.
    """

    def __init__(self):
        self.data = {}

    def push_front(self, score, candidate):
        """Push a value to the appropriate stack (given by 'score').

        self.push_front(42, 7)
        :>>> Current state: {42:[4, 2, 3, 7], 51:[0]}

        self.push_front(46, 6)
        :>>> Current state: {42:[4, 2, 3, 7], 46:[6], 51:[0]}
        """
        if score in self.data.keys():
            self.data[score].append(candidate)
        else:
            self.data[score] = [candidate]

    def pop_front(self):
        """Pops a value: choose the stack with minimum score, and pop the most
        recent item in this stack.

        c = self.pop_front()
        :>>> c = 7
        :>>> Current state: {42:[4, 2, 3], 46:[6], 51:[0]}

        When the object is empty, returns None.
        """
        if len(self.data) == 0:
            return None
        lowest_score = min(self.data.keys())
        c = self.data[lowest_score].pop()
        if len(self.data[lowest_score]) == 0:
            del self.data[lowest_score]
        return c


if __name__ == '__main__':
    # A quick demo
    preferences_utilities = np.random.randint(-5, 5, (10, 5))
    pop = Population(preferences_utilities)
    election = Maximin(pop)
    election.demo(log_depth=3)