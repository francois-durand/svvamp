# -*- coding: utf-8 -*-
"""
Created on Tue Sep 30 20:43:45 2014
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
from svvamp.VotingSystems.BordaResult import BordaResult
from svvamp.Preferences.Population import Population


class Borda(BordaResult, Election):
    """Borda rule.

    Inherits functions and optional parameters from superclasses
    :class:`~svvamp.ElectionResult` and :class:`~svvamp.Election`.

    :Example:

    >>> import svvamp
    >>> pop = svvamp.PopulationSpheroid(V=100, C=5)
    >>> election = svvamp.Borda(pop)

    Voter ``v`` gives
    (:attr:`~svvamp.Population.C` - 1) points to her top-ranked candidate,
    (:attr:`~svvamp.Population.C` - 2) to the second, ...,
    0 to the last.
    Ties are broken by natural order on the candidates (lower index wins).

    :meth:`~svvamp.Election.CM`: Deciding CM is NP-complete.

        * :attr:`~svvamp.Election.CM_option` = ``'fast'``:
          Zuckerman et al. (2009). This approximation algorithm is
          polynomial and has a window of error of 1 manipulator.
        * :attr:`~svvamp.Election.CM_option` = ``'exact'``:
          Non-polynomial algorithm from superclass :class:`~svvamp.Election`.

    :meth:`~svvamp.Election.ICM`: Algorithm is polynomial and has a window of
    error of 1 manipulator.

    :meth:`~svvamp.Election.IM`: Exact in polynomial time.

    :meth:`~svvamp.Election.not_IIA`: Exact in polynomial time.

    :meth:`~svvamp.Election.TM`: Exact in polynomial time.

    :meth:`~svvamp.Election.UM`: Exact in polynomial time.

    References:

        'Algorithms for the coalitional manipulation problem',
        M. Zuckerman, A. Procaccia and J. Rosenschein, 2009.

        'Unweighted Coalitional Manipulation Under the Borda Rule is NP-Hard',
        Nadja Betzler, Rolf Niedermeier and Gerhard Woeginger, 2011.

        'Complexity of and algorithms for the manipulation of Borda,
        Nanson's and Baldwin's voting rules', Jessica Davies,
        George Katsirelos, Nina Narodytska, Toby Walsh and Lirong Xia, 2014.
    """

    _layout_name = 'Borda'
    _options_parameters = Election._options_parameters.copy()
    _options_parameters.update(BordaResult._options_parameters)
    _options_parameters['IM_option'] = {'allowed': ['exact'],
                                        'default': 'exact'}
    _options_parameters['TM_option'] = {'allowed': ['exact'],
                                        'default': 'exact'}
    _options_parameters['UM_option'] = {'allowed': ['exact'],
                                        'default': 'exact'}
    _options_parameters['ICM_option'] = {'allowed': ['fast'],
                                           'default': 'fast'}
    _options_parameters['CM_option'] = {'allowed': ['fast', 'exact'],
                                        'default': 'fast'}

    def __init__(self, population, **kwargs):
        super().__init__(population, **kwargs)
        self._log_identity = "BORDA"
        self._class_result = Borda
        self._with_two_candidates_reduces_to_plurality = True
        self._is_based_on_rk = True
        self._meets_InfMC_c_ctb = True
        self._precheck_ICM = False

    #%% Independence of Irrelevant Alternatives (IIA)

    def _compute_IIA(self):
        self._mylog("Compute IIA", 1)
        # If we remove candidate d, candidate c loses matrix_duels_rk[c, d]
        # Borda points and w loses matrix_duels_rk[w, d]. So, candidate c
        # gains the difference, when compared to w.
        impact_removal_d_on_c = np.add(
            - self.pop.matrix_duels_rk,
            + self.pop.matrix_duels_rk[self.w, :][np.newaxis, :])
        impact_removal_d_on_c[np.diag_indices(self.pop.C)] = 0
        impact_removal_d_on_c[:, self.w] = 0  # Forbidden to remove w

        # For c, any candidate d s.t. impact_removal_d_on_c[c, d] > 0 is
        # a good choice to remove. pseudo_scores[c] is c's score minus
        # w's score after these removals.
        # After all these removals, it may be that another candidate e will
        # have a better score than c, but we don't care: the only important
        # thing is to make w lose.
        pseudo_scores = (
            self.scores - self.score_w +
            np.sum(np.maximum(impact_removal_d_on_c, 0), 1))
        self._mylogv("pseudo_scores =", pseudo_scores, 2)

        # If pseudo_scores[c] > 0 for some c, then IIA is broken.
        best_c = np.argmax(pseudo_scores)
        self._mylogv("best_c =", best_c, 2)
        if pseudo_scores[best_c] + (best_c < self.w) > 0:
            # Since we chose the best c, we know that she is the winner in her
            # optimal subset.
            self._is_IIA = False
            self._example_winner_IIA = best_c
            # self._example_subset_IIA = np.ravel(np.where(
            #     np.logical_not(impact_removal_d_on_c[best_c, :] > 0)))
            self._example_subset_IIA = np.logical_not(
                impact_removal_d_on_c[best_c, :] > 0)
        else:
            self._is_IIA = True
            self._example_winner_IIA = np.nan
            self._example_subset_IIA = np.nan

    #%% Individual manipulation (IM)            

    def _IM_preliminary_checks_general_subclass(self):
        # If we change the ballot of only one c-manipulator v, then score[c] -
        # score[w] increases by at most C-2. Indeed, c gave at least 1 point
        # of difference to c vs w (she strictly prefers c to w), and now she
        # can give  at most C-1 points of difference (with c on top and w at
        # bottom).
        self._mylogv('Preliminary checks: scores =', self.scores)
        self._mylogv('Preliminary checks: scores_w =', self.score_w)
        temp = self.scores + self.pop.C - 2 + (
            np.array(range(self.pop.C)) < self.w)
        self._mylogm('Preliminary checks: scores + C - 2 + (c < w) =',
                     temp)
        # self._mylogm('Preliminary checks: possible candidates =',
        #              temp > self.scores[self.w])
        self._v_IM_for_c[:, temp <= self.scores[self.w]] = False

    def _IM_main_work_v(self, v, c_is_wanted,
                        nb_wanted_undecided, stop_if_true):
        scores_without_v = self.scores - self.ballots[v, :]
        self._mylogv('scores_without_v =', scores_without_v)
        candidates_by_decreasing_score = np.argsort(-scores_without_v,
                                                    kind='mergesort')
        # ballot_balance is the ballot that v must give to balance
        # the scores of all candidates the best she can.
        ballot_balance = np.argsort(candidates_by_decreasing_score)
        self._mylogv('ballot_balance =', ballot_balance, 3)
        for c in range(self.pop.C):
            if not c_is_wanted[c]:
                continue
            if not np.isneginf(self._v_IM_for_c[v, c]):
                continue
            self._mylogv('Candidate c =', c, 3)
            # Compared to ballot_balance, the candidates that v was
            # putting better than c lost 1 point.
            ballot = ballot_balance - np.greater(ballot_balance,
                                                 ballot_balance[c])
            # Voter v gives the maximal score to c.
            ballot[c] = self.pop.C - 1
            self._mylogv('ballot =', ballot, 3)
            w_test = np.argmax(scores_without_v + ballot)
            self._mylogv('w_test =', w_test, 3)
            if w_test == c:
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

    #%% Trivial Manipulation (TM)

    # Defined in the superclass.

    #%% Unison manipulation (UM)

    def _UM_main_work_c_exact(self, c):
        scores_test = np.sum(
            self.ballots[np.logical_not(self.v_wants_to_help_c[:, c]), :], 0)
        candidates_by_decreasing_score = np.argsort(-scores_test,
                                                    kind='mergesort')
        # Balancing ballot: put candidates in the order of their current
        # scores (least point to the most dangerous).
        ballot = np.argsort(candidates_by_decreasing_score)
        # Now put c on top. And modify other Borda scores accordingly on the
        # ballot.
        ballot -= np.greater(ballot, ballot[c])
        ballot[c] = self.pop.C - 1
        # New scores = old scores + n_manipulators * ballot.
        scores_test += np.multiply(self.pop.matrix_duels_ut[c, self.w], ballot)
        w_test = np.argmax(scores_test)
        self._candidates_UM[c] = (w_test == c)

    #%% Ignorant-Coalition Manipulation (ICM)

    def _ICM_main_work_c_fast(self, c, optimize_bounds):
        # n_s: number of sincere voters (= counter-manipulators)
        n_s = self.pop.V - self.pop.matrix_duels_ut[c, self.w]
        C = self.pop.C
        # Necessary condition:
        # ^^^^^^^^^^^^^^^^^^^^
        # Let us note n_m the number of manipulators for c. Each manipulator
        # gives C - 1 points to c and an average of (C - 2) / 2 points to each
        # other candidate. Let us note d the other candidate with maximum 
        # score_m (from the manipulators).
        # score_m[c] = n_m * (C - 1).
        # score_m[d] >= n_m * (C - 2) / 2.
        # At worst, each counter-manipulator will give C - 1 points to d and
        # 0 point to c.
        # score_total[c] = n_m * (C - 1)
        # score_total[d] >= n_m * (C - 2) / 2 + n_s * (C - 1)
        # Since these scores may be half-integers, we will multiply them by 2
        # to apply the tie=breaking rule correctly. We need:
        # 2 * score_total[c] >= 2 * score_total[d] + (c > 0), which leads to:
        # n_m >= (2 * n_s * (C - 1) + (c > 0)) / C.
        necessary = np.ceil((2 * n_s * (C - 1) + (c > 0)) / C)
        # Sufficient condition for n_m = 2*p
        # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        # Manipulators can give C - 1 points to c. Other candidates have
        # balanced scores (ballot [C-2, ..., 0] paired with [0, ..., C-2]).
        # score_m[c] = 2p (C - 1)
        # score_m[d] = 2p (C - 2) / 2
        # After counter-manipulation:
        # score_total[c] = 2p (C - 1)
        # score_total[d] = 2p (C - 2) / 2 + n_s (C - 1)
        # For ICM, it is sufficient that score_total[c] >= score_total[d] + 
        # (c > 0), which translates as before to:
        # p >= (n_s (C-1) + (c > 0)) / C.
        # N.B.: From the general inequality 2 ceil(x) <= ceil (2x), it can
        # be deduced that sufficient_even - necessary <= 1.
        sufficient_even = 2 * np.ceil((n_s * (C - 1) + (c > 0)) / C)
        # Sufficient condition for n_m = 2*p + 1
        # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        # Manipulators can give C - 1 points to c. With the first 2p
        # manipulators, other candidates have balanced scores as before. In
        # addition, the last manipulator gives C - 2 points to the opponent
        # candidate with the higher index (so that she benefits less from
        # t.b. rule).
        # score_m[c] = (2p + 1) (C - 1)
        # score_m[d] = 2p (C - 2) / 2 + (C - 2) = (p + 1) (C - 2)
        # After counter-manipulation:
        # score_total[c] = (2p + 1) (C - 1)
        # score_total[d] = (p + 1) (C - 2) + n_s (C - 1)
        # For ICM, it is sufficient that score_total[c] >= score_total[d] + 
        # (c == C-1) (the tie-breaking rule is against c only if she is the
        # last candidate). This translates to:
        # p >= (n_s (C - 1) - 1 + (c == C-1)) / C.
        sufficient_odd = 1 + 2 * np.ceil((n_s * (C - 1) - 1 + (c == C-1)) / C)
        self._update_necessary(
            self._necessary_coalition_size_ICM, c, necessary)
        self._update_sufficient(
            self._sufficient_coalition_size_ICM, c, sufficient_even)
        self._update_sufficient(
            self._sufficient_coalition_size_ICM, c, sufficient_odd)

    #%% Coalition Manipulation (CM)

    def _CM_main_work_c_fast(self, c, optimize_bounds):
        """Do the main work in CM loop for candidate c.
        * Try to improve bounds _sufficient_coalition_size_CM[c] and
        _necessary_coalition_size_CM[c].
        
        Algorithm: Zuckerman et al. (2009), Algorithms for the coalitional 
        manipulation problem.
        """
        scores_test = np.sum(
            self.ballots[np.logical_not(self.v_wants_to_help_c[:, c]), :], 0)
        # We add a tie-breaking term [(C-1)/C, (C-2)/C, ..., 0] to ease
        # the computations.
        scores_test = scores_test + (
            np.array(range(self.pop.C-1, -1, -1)) / self.pop.C)
        self._mylogv('CM: Further check: scores_test =', scores_test, 3)

        # A family of necessary conditions
        # For k in {1, ..., C-1}, let us note mean_k the mean of the k highest
        # scores among other candidates than c. At best, we will have:
        # scores_tot[c] = scores_test[c] + n_m (C - 1)
        # mean_k_tot = mean_k_test + n_m (k - 1) / 2
        # We must have scores_tot[c] >= mean_k_tot so:
        # n_m >= 2 * (mean_k_test - scores_test[c]) / (2 C - k - 1)
        increasing_scores = np.sort(
            scores_test[np.array(range(self.pop.C)) != c])
        cum_sum = 0
        for k in range(1, self.pop.C):
            cum_sum += increasing_scores[-k]
            necessary = np.ceil(2 * (cum_sum / k - scores_test[c]) /
                                (2 * self.pop.C - k - 1))
            self._mylogv('CM: Further check: k =', k)
            self._mylogv('CM: Further check: cum_sum =', cum_sum)
            self._mylogv('CM: Further check: necessary =', necessary)
            self._update_necessary(
                self._necessary_coalition_size_CM, c, necessary,
                'CM: Further check: necessary_coalition_size_CM =')
        # An opportunity to escape before real work
        if (self._necessary_coalition_size_CM[c] ==
                self._sufficient_coalition_size_CM[c]):
            return False
        if (not optimize_bounds and self.pop.matrix_duels_ut[c, self.w] <
                self._necessary_coalition_size_CM[c]):
            # This is a quick escape: we have not optimized the bounds the
            # best we could.
            return True
        
        # Now, the real work
        n_m = 0
        self._mylogv('CM: Fast algorithm: scores_test =', scores_test, 3)
        while True:
            n_m += 1
            # Balancing ballot: put candidates in the order of their current
            # scores (least point to the most dangerous).
            candidates_by_decreasing_score = np.argsort(-scores_test,
                                                        kind='mergesort')
            ballot = np.argsort(candidates_by_decreasing_score)
            # Now put c on top. And modify other Borda scores accordingly on
            # the ballot.
            ballot -= np.greater(ballot, ballot[c])
            ballot[c] = self.pop.C - 1
            self._mylogv('CM: Fast algorithm: ballot =', ballot, 3)
            # New scores = old scores + ballot.
            scores_test += ballot
            self._mylogv('CM: Fast algorithm: scores_test =', scores_test, 3)
            w_test = np.argmax(scores_test)
            if w_test == c:
                sufficient = n_m
                self._mylogv('CM: Fast algorithm: sufficient =', sufficient)
                self._update_sufficient(
                    self._sufficient_coalition_size_CM, c, sufficient,
                    'CM: Fast algorithm: sufficient_coalition_size_CM =')
                # From Zuckerman et al.: the algorithm is optimal up to 1
                # voter. But since we do not have the same tie-breaking
                # rule, it makes 2 voters.
                self._update_necessary(
                    self._necessary_coalition_size_CM, c, sufficient - 2,
                    'CM: Fast algorithm: necessary_coalition_size_CM =')
                break

    def _CM_main_work_c(self, c, optimize_bounds):
        is_quick_escape_fast = self._CM_main_work_c_fast(c, optimize_bounds)
        if not self.CM_option == "exact":
            # With 'fast' option, we stop here anyway.
            return is_quick_escape_fast

        # From this point, we have necessarily the 'exact' option (which is,
        #  in fact, only an exhaustive exploration with = n_m manipulators).
        is_quick_escape_exact = self._CM_main_work_c_exact(c, optimize_bounds)
        return is_quick_escape_fast or is_quick_escape_exact


if __name__ == '__main__':
    # A quick demo
    preferences_utilities = np.random.randint(-5, 5, (10, 5))
    pop = Population(preferences_utilities)
    election = Borda(pop)
    election.demo(log_depth=3)