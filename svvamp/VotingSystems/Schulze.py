# -*- coding: utf-8 -*-
"""
Created on Wed Oct  8 13:06:52 2014
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

from svvamp.Preferences.Population import Population
from svvamp.Preferences.Population import preferences_utilities_to_matrix_duels
from svvamp.VotingSystems.Election import Election
from svvamp.VotingSystems.SchulzeResult import SchulzeResult


class Schulze(SchulzeResult, Election):
    """Schulze method.

    Inherits functions and optional parameters from superclasses
    :class:`~svvamp.ElectionResult` and :class:`~svvamp.Election`.

    :Example:

    >>> import svvamp
    >>> pop = svvamp.PopulationSpheroid(V=100, C=5)
    >>> election = svvamp.Schulze(pop)

    :attr:`~svvamp.Schulze.scores`\ ``[c, d]`` is equal to the width of the
    widest path from candidate ``c`` to candidate ``d`` in the capacited graph
    defined by :attr:`~svvamp.Population.matrix_duels_vtb`. We say that ``c``
    is *better* than ``d`` if ``scores[c, d]`` > ``scores[d, c]``. Candidate
    ``c`` is a *potential winner* if no candidate ``d`` is *better* than ``c``.

    Among the potential winners, the candidate with lowest index is declared
    the winner.

    .. note::

        In the original Schulze method, ties are broken at random. However,
        this feature is not supported by SVVAMP because it leads to
        difficulties for the *definition* of manipulation itself
        (and all the more for implementation).

    :meth:`~svvamp.Election.CM`:

        * :attr:`~svvamp.Election.CM_option` = ``'fast'``:
          Gaspers et al. (2013). This algorithm is polynomial and has a
          window of error of 1 manipulator (due to the tie-breaking rule).
        * :attr:`~svvamp.Election.CM_option` = ``'exact'``:
          Non-polynomial algorithm from superclass :class:`~svvamp.Election`.

    :meth:`~svvamp.Election.ICM`: Exact in polynomial time.

    :meth:`~svvamp.Election.IM`:

        * :attr:`~svvamp.Election.IM_option` = ``'fast'``:
          Gaspers et al. (2013). This algorithm is polynomial and may not be
          able to decide IM (due to the tie-breaking rule).
        * :attr:`~svvamp.Election.IM_option` = ``'exact'``:
          Non-polynomial algorithm from superclass :class:`~svvamp.Election`.

    :meth:`~svvamp.Election.not_IIA`: Non-polynomial
    or non-exact algorithms from superclass :class:`~svvamp.Election`.

    :meth:`~svvamp.Election.TM`: Exact in polynomial time.

    :meth:`~svvamp.Election.UM`:

        * :attr:`~svvamp.Election.UM_option` = ``'fast'``:
          Gaspers et al. (2013). This algorithm is polynomial and has a
          window of error of 1 manipulator (due to the tie-breaking rule).
        * :attr:`~svvamp.Election.UM_option` = ``'exact'``:
          Non-polynomial algorithm from superclass :class:`~svvamp.Election`.

    .. note:

        For this voting system, UM and CM are almost equivalent up to
        tie-breaking. For this reason, :attr:`~svvamp.Election.UM_option` and
        :attr:`~svvamp.Election.CM_option` are linked to each other: modifying
        one modifies the other accordingly.

    References:

        'A new monotonic, clone-independent, reversal symmetric, and
        Condorcet-consistent single-winner election method ', Markus Schulze,
        2011.

        'Schulze and Ranked-Pairs Voting are Fixed-Parameter Tractable to
        Bribe, Manipulate, and Control', Lane A. Hemaspaandra, Rahman Lavaee
        and Curtis Menton, 2012.

        'Manipulation and Control Complexity of Schulze Voting', Curtis Menton
        and Preetjot Singh, 2012.

        'A Complexity-of-Strategic-Behavior Comparison between Schulze’s Rule
        and Ranked Pairs', David Parkes and Lirong Xia, 2012.

        'Coalitional Manipulation for Schulze’s Rule', Serge Gaspers, Thomas
        Kalinowski, Nina Narodytska and Toby Walsh, 2013.
    """
    
    _layout_name = 'Schulze'
    _options_parameters = Election._options_parameters.copy()
    _options_parameters.update(SchulzeResult._options_parameters)
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
        self._log_identity = "SCHULZE"
        self._class_result = SchulzeResult
        self._with_two_candidates_reduces_to_plurality = True
        self._is_based_on_strict_rankings = True
        self._meets_Condorcet_c_vtb_ctb = True
        self._precheck_UM = False
        self._precheck_ICM = False

    def _forget_UM(self):
        self._UM_was_initialized = False
        self._UM_was_computed_with_candidates = False
        # Exceptionally, we initialize the variables for UM here, because UM
        # is managed in CM.
        self._is_UM = -np.inf
        self._candidates_UM = np.full(self.pop.C, -np.inf)
        self._candidates_UM[self.w] = False
        # _UM_fast_tested[c] will be True iff if we have already launched
        # _vote_strategically with n_m manipulators (for c).
        self._UM_fast_tested = np.zeros(self.pop.C, dtype=np.bool)

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

    def _vote_strategically(self, matrix_duels_s, S, c, n_m):
        """ Manipulation algorithm: "Coalitional Manipulation for Schulze’s 
        Rule" (Gaspers, Kalinowski, Narodytska and Walsh 2013).

        Arguments:
        matrix_duels_s -- 2d array. Matrix of duels for sincere voters.
        S -- 2d array. S[c, d] is the strength of the widest path from c to
            d (with sincere voters only). Be careful, it is based on the 
            antisymmetric matrix of duels. I.e. while 
            SchulzeResult._count_ballot_aux provides widest_path_s, we need 
            to take S = 2 * (widest_path_s - n_s / 2), then set S's diagonal 
            coefficients to 0.
        c -- Integer (candidate). Our challenger.
        n_m -- Integer. Number of manipulators.
        """

        candidates_not_c = np.concatenate((
            np.array(range(c), dtype=int),
            np.array(range(c+1, self.pop.C), dtype=int)))
        w = matrix_duels_s - matrix_duels_s.T
        w_sup = w + n_m
        w_inf = w - n_m
        U = S + n_m
        U_prev = None
        self._mylogv("candidates_not_c =", candidates_not_c, 3)
        self._mylogm("matrix_duels_s =", matrix_duels_s, 3)
        self._mylogm("S =", S, 3)
        self._mylogm("w =", w, 3)
        self._mylogm("w_sup =", w_sup, 3)
        self._mylogm("w_inf =", w_inf, 3)
        self._mylogm("U =", U, 3)
        self._mylogm("U_prev =", U_prev, 3)
        
        # Algorithm: Pre-processing Bounds
        while not np.all(U_prev == U):
            U_prev = np.copy(U)
            # Rule 1
            U[:, c] = np.minimum(U[:, c], U[c, :])
            self._mylogm('Rule 1: U =', U, 3)
            # Rule 2
            for x in candidates_not_c:
                G_x = np.ones((self.pop.C, self.pop.C))
                # Remove all vertices y (candidates) s.t. U(y,c) < U(x,c)
                V_removed = np.less(U[:, c], U[x, c])
                V_removed[c] = False
                G_x[V_removed, :] = 0
                G_x[:, V_removed] = 0
                # Removed all edges (y,z) s.t. w_sup(y,z) < U(x,c)
                G_x[w_sup < U[x, c]] = 0
                # Does G_x contain a path from c to x ? If not, do stuff.
                if not nx.has_path(nx.DiGraph(G_x), c, x):
                    U[x, c] -= 2
                    self._mylogm('Rule 2: U =', U, 3)
            # Rule 3
            for x in candidates_not_c:
                for y in candidates_not_c:
                    if y == x:
                        continue
                    if U[x, c] < w_inf[x, y]:
                        U[y, c] = min(U[y, c], U[x, c])
                        self._mylogm('Rule 3: U =', U, 3)
            # Test possibility
            self._mylogm("U =", U, 3)
            self._mylogv("U[candidates_not_c, c] =", U[candidates_not_c, c], 3)
            self._mylogv("S[candidates_not_c, c] - n_m =", 
                         S[candidates_not_c, c] - n_m, 3)
            if np.any(U[candidates_not_c, c] < S[candidates_not_c, c] - n_m):
                self._mylog("Cowinner manipulation failed")
                return False, False
        
        # Algorithm : construction of ordering Lambda (when manipulation is 
        # possible)
        F = np.zeros(self.pop.C, dtype=np.bool)
        F[c] = 1
        X = np.ones(self.pop.C, dtype=np.bool)
        X[c] = 0
        order_lambda = [c]
        for i in range(self.pop.C - 1):
            D = np.max(U[X, c])
            possible_y = np.logical_and(
                np.logical_and(X, U[:, c] == D),
                np.any(w_sup[F, :] >= D, 0))
            y = np.where(possible_y)[0][-1]
            F[y] = True
            X[y] = False
            order_lambda.append(y)
        order_lambda = np.array(order_lambda)
        self._mylogv("order_lambda =", order_lambda, 3)
            
        # Who wins with the tie-breaking rule?
        reciprocal_lambda = np.argsort(order_lambda)
        matrix_duels_m = n_m * np.triu(np.ones((self.pop.C, self.pop.C)), 1)[
            reciprocal_lambda, :][:, reciprocal_lambda]
        w_temp, foo, bar = SchulzeResult._count_ballot_aux(
            matrix_duels_s + matrix_duels_m)
        self._mylogv("matrix_duels_m =", matrix_duels_m, 3)
        self._mylogv("w_temp =", w_temp, 3)
        if w_temp == c:
            self._mylog("Cowinner manipulation worked, with ctb also")
            return True, True
        else:
            self._mylog("Cowinner manipulation worked but not with ctb")
            return True, False

    #%% Individual manipulation (IM)
            
    def _IM_main_work_v_fast(self, v, c_is_wanted,
                             nb_wanted_undecided, stop_if_true):
        for c in self.losing_candidates:
            if not c_is_wanted[c]:
                continue
            if not np.isneginf(self._v_IM_for_c[v, c]):
                continue
            self._mylogv("IM: c =", c, 3)
            nb_wanted_undecided -= 1
            # Maybe we will not decide, but we will have done the 'fast' job
            # for c anyway.

            matrix_duels_s = np.copy(self.pop.matrix_duels_vtb)
            for x in range(self.pop.C):
                for y in range(x+1, self.pop.C):
                    if (self.pop.preferences_borda_vtb[v, x] >
                            self.pop.preferences_borda_vtb[v, y]):
                        matrix_duels_s[x, y] -= 1
                    else:
                        matrix_duels_s[y, x] -= 1
                        
            w_s, widest_path, _ = SchulzeResult._count_ballot_aux(
                matrix_duels_s)
            if w_s == c:
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
                
            S = 2 * (widest_path - (self.pop.V - 1) / 2)
            np.fill_diagonal(S, 0)
            success_cowinner, success_tb = self._vote_strategically(
                matrix_duels_s, S, c, 1)
            if success_tb:
                self._v_IM_for_c[v, c] = True
                self._candidates_IM[c] = True
                self._voters_IM[v] = True
                self._is_IM = True
                self._mylog("IM found", 3)
                if stop_if_true:
                    return
            elif success_cowinner:
                self._mylog("IM not sure", 3)
                if self._IM_option != 'exact':
                    self._v_IM_for_c[v, c] = np.nan
            else:
                self._v_IM_for_c[v, c] = False
            if nb_wanted_undecided == 0:
                return

    def _IM_main_work_v(self, v, c_is_wanted,
                        nb_wanted_undecided, stop_if_true):
        self._IM_main_work_v_fast(v, c_is_wanted, nb_wanted_undecided,
                                  stop_if_true)
        # Deal with 'exact' (brute force) option
        if self._IM_option != 'exact':
            return
        if stop_if_true and np.any(np.equal(
                self._v_IM_for_c[v, c_is_wanted], True)):
            return
        nb_wanted_undecided_updated = np.sum(np.isneginf(
            self._v_IM_for_c[v, c_is_wanted]))
        if nb_wanted_undecided_updated == 0:
            self._mylog("IM: Job finished", 3)
        else:
            self._mylogv("IM: Still some work for v =",
                         v, 3)
            self._IM_main_work_v_exact(v, c_is_wanted,
                                       nb_wanted_undecided_updated,
                                       stop_if_true)

    #%% Trivial Manipulation (TM)

    # Use the general methods.

    #%% Ignorant-Coalition Manipulation (ICM)

    # Use the general methods.

    #%% Coalition Manipulation (CM)
        
    def _CM_main_work_c(self, c, optimize_bounds):
        matrix_duels_s = preferences_utilities_to_matrix_duels(
            self.pop.preferences_borda_vtb[
                np.logical_not(self.v_wants_to_help_c[:, c]), :])
        n_m = self.pop.matrix_duels[c, self.w]
        n_s = self.pop.V - n_m
        w, widest_path, _ = SchulzeResult._count_ballot_aux(matrix_duels_s)
        S = 2 * (widest_path - n_s / 2)
        np.fill_diagonal(S, 0)

        # We try the "co-winner" version with n_m - 1 manipulators. If True,
        # it implies that unique winner version is OK for n_m manipulators.
        success_cowinner, success_tb = self._vote_strategically(
            matrix_duels_s, S, c, n_m - 1)
        if success_tb:
            self._update_sufficient(
                self._sufficient_coalition_size_CM, c, n_m - 1,
                'CM: Update sufficient_coalition_size_CM = n_m - 1 =')
            self._candidates_UM[c] = True
            self._is_UM = True
            return
        if success_cowinner:
            self._update_sufficient(
                self._sufficient_coalition_size_CM, c, n_m,
                'CM: Update sufficient_coalition_size_CM = n_m =')
            self._update_necessary(
                self._necessary_coalition_size_CM, c, n_m - 1,
                'CM: Update necessary_coalition_size_CM = n_m - 1 =')
            # We do not know for UM, but we do not really care
            return
        else:
            self._update_necessary(
                self._necessary_coalition_size_CM, c, n_m,
                'CM: Update necessary_coalition_size_CM = n_m =')

        # We try with n_m manipulators. 
        success_cowinner, success_tb = self._vote_strategically(
            matrix_duels_s, S, c, n_m)
        self._UM_fast_tested[c] = True
        if success_tb:
            self._update_sufficient(
                self._sufficient_coalition_size_CM, c, n_m,
                'CM: Update sufficient_coalition_size_CM = n_m =')
            self._candidates_UM[c] = True
            self._is_UM = True
            return
        if success_cowinner:
            self._update_sufficient(
                self._sufficient_coalition_size_CM, c, n_m + 1,
                'CM: Update sufficient_coalition_size_CM = n_m + 1 =')
            self._update_necessary(
                self._necessary_coalition_size_CM, c, n_m,
                'CM: Update necessary_coalition_size_CM = n_m =')
            if self.CM_option == 'exact':
                self._UM_main_work_c_exact_rankings(c)
                if self._candidates_UM[c] == True:
                    self._update_sufficient(
                        self._sufficient_coalition_size_CM, c, n_m,
                        'CM: Update sufficient_coalition_size_CM = n_m =')
                else:
                    self._CM_main_work_c_exact(c, optimize_bounds)
        else:
            self._update_necessary(
                self._necessary_coalition_size_CM, c, n_m + 1,
                'CM: Update necessary_coalition_size_CM = n_m + 1 =')
            self._candidates_UM[c] = False
            return

    #%% Unison manipulation (UM)

    def _UM_initialize_general(self, with_candidates):
        self._mylog("UM: Initialize", 2)
        self._UM_was_initialized = True
        # Do not initialize the usual variables, it is done elsewhere.
        self._UM_preliminary_checks_general()

    def _UM_preliminary_checks_general_subclass(self):
        if not np.any(np.isneginf(self._candidates_UM)):
            return
        self.CM()
        self._candidates_UM[np.equal(self._candidates_CM, False)] = False

    def _UM_main_work_c(self, c):
        n_m = self.pop.matrix_duels[c, self.w]
        if self._sufficient_coalition_size_CM[c] + 1 <= n_m:
            self._candidates_UM[c] = True
            return
        if self._necessary_coalition_size_CM[c] - 1 > n_m:
            self._candidates_UM[c] = False
            return
        if not self._UM_fast_tested[c]:
            # We must try the fast algo first.
            matrix_duels_s = preferences_utilities_to_matrix_duels(
                self.pop.preferences_borda_vtb[
                    np.logical_not(self.v_wants_to_help_c[:, c]), :])
            n_s = self.pop.V - n_m
            w, widest_path, _ = SchulzeResult._count_ballot_aux(matrix_duels_s)
            S = 2 * (widest_path - n_s / 2)
            np.fill_diagonal(S, 0)
            success_cowinner, success_tb = self._vote_strategically(
                matrix_duels_s, S, c, n_m)
            self._UM_fast_tested[c] = True
            if success_tb:
                self._candidates_UM[c] = True
                return
            if not success_cowinner:
                self._candidates_UM[c] = False
                return
        if self.UM_option == 'exact':
            self._UM_main_work_c_exact_rankings(c)
        else:
            self._candidates_UM[c] = np.nan


if __name__ == '__main__':
    # A quick demo
    preferences_utilities = np.random.randint(-5, 5, (10, 5))
    pop = Population(preferences_utilities)
    election = Schulze(pop)
    election.demo(log_depth=3)