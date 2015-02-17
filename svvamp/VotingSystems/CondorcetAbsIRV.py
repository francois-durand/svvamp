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
from svvamp.VotingSystems.CondorcetAbsIRVResult import CondorcetAbsIRVResult
from svvamp.VotingSystems.IRV import IRV


class CondorcetAbsIRV(CondorcetAbsIRVResult, Election):
    """Abs-Condorcet-IRV.

    Inherits functions and optional parameters from superclasses
    :class:`~svvamp.ElectionResult` and :class:`~svvamp.Election`.

    Each voter must provide a weak order, and a strict total order that is
    coherent with this weak order (i.e., is a tie-breaking of this weak order).

    If there is a Condorcet winner (computed with the weak orders, i.e. in
    the sense of :attr:`~svvamp.Population.matrix_victories_abs`),
    then she is elected. Otherwise, :class:`~svvamp.IRV` is used (with the
    strict total orders).

    If sincere preferences are strict total orders, then this voting system is
    equivalent to :class:`~svvamp.CondorcetTvbIRV` for sincere voting, but
    manipulators have more possibilities (they can pretend to have ties in
    their preferences). In that case, it is a more 'natural' framework to use
    :class:`~svvamp.CondorcetTvbIRV`.

    :meth:`~svvamp.Election.CM`:

        * :attr:`~svvamp.Election.CM_option` = ``'fast'``:
          Rely on the fast algorithm for :class:`~svvamp.IRV`.
          Polynomial heuristic. Can prove CM but unable to decide non-CM.
        * :attr:`~svvamp.Election.CM_option` = ``'slow'``:
          Same as fast but try also the exact algorithm for
          :class:`~svvamp.ExhaustiveBallot` to find an elimination path.
          Non-polynomial heuristic (:math:`2^C`). Can prove CM but unable to
          decide non-CM.
        * :attr:`~svvamp.Election.CM_option` = ``'almost_exact'``:
          Rely on the exact algorithm for :class:`~svvamp.IRV`.
          Non-polynomial heuristic (:math:`C!`). Very efficient to prove CM
          but unable to decide non-CM.
        * :attr:`~svvamp.Election.CM_option` = ``'exact'``:
          Non-polynomial algorithm from superclass :class:`~svvamp.Election`.

    :meth:`~svvamp.Election.ICM`: Exact in polynomial time.

    :meth:`~svvamp.Election.IM`: Non-polynomial
    or non-exact algorithms from superclass :class:`~svvamp.Election`.

    :meth:`~svvamp.Election.not_IIA`: Non-polynomial
    or non-exact algorithms from superclass :class:`~svvamp.Election`.

    :meth:`~svvamp.Election.TM`: Exact in polynomial time.

    :meth:`~svvamp.Election.UM`: Non-polynomial or non-exact algorithms from
    superclass :class:`~svvamp.Election`.

    .. seealso: :class:`svvamp.ExhaustiveBallot`,
                :class:`svvamp.IRV`,
                :class:`svvamp.ICRV`,
                :class:`svvamp.CondorcetTvbIRV`.
    """

    _layout_name = 'Absolute-Condorcet IRV'
    _options_parameters = Election._options_parameters.copy()
    _options_parameters.update(CondorcetAbsIRVResult._options_parameters)
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
        self._log_identity = "CONDORCET_ABS_IRV"
        self._class_result = CondorcetAbsIRVResult
        self._with_two_candidates_reduces_to_plurality = True
        self._is_based_on_strict_rankings = True
        self._meets_majority_favorite_c_vtb_ctb = True
        self._meets_Condorcet_c = True
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

    def _CM_preliminary_checks_general_subclass(self):
        # We check IRV first.
        if self.w != self.IRV.w:
            # Then IRV is manipulable anyway (for w, so a precheck on IRV
            # would give us no information).
            return
        if self.CM_option != "fast":
            if self.CM_option == "slow":
                self.IRV.CM_option = "slow"
            else:
                self.IRV.CM_option = "exact"
            if self.IRV.CM()[0] == False:
                # Condorcification theorem apply.
                self._mylog("CM impossible (since it is impossible for " +
                            "IRV)", 2)
                self._is_CM = False
                self._candidates_CM[:] = False
                self._CM_was_computed_with_candidates = True

    def _CM_main_work_c(self, c, optimize_bounds):
        # Decondorcification is independent from what we will do for IRV.
        # It just gives a necessary coalition size.
        # 1) If IRV.w is Condorcet:
        #   Manipulate IRV --> use IRV, we know.
        #   Prevent w from being a Condorcet winner --> we know.
        #   * If incomplete: we need IRV.CM (incomplete). But if IRV works
        # for c and Cond-IRV does not, we might need IRV.CM.complete (we
        # will know after).
        # 2) If IRV.w != CondIRV.w (which is Condorcet):
        #   a) If c = IRV.w:
        #       Make w win: easy, and we have suff = n_m (nec can be less)
        #       Prevent CondIRV.w from being Condorcet --> we know.
        #   b) If c != IRV.w:
        #       Make c win: complicated. We can use the elimination path we
        #       know, but it might not succeed.
        #       Prevent CondIRV.w from being Condorcet --> we know.
        #   * If incomplete: we do not even need IRV.CM. But we need to make
        # sure that we try w first in losing_candidates. However,
        # if it fails, we will need to try other c's. What do we do then?
        # Use a fast IRV on the fly? Or simply try the elimination path
        # suggested by IRV? Etc.  Maybe the first is better (cheap and
        # likely to be more efficient).
        # 3) If there is no Condorcet winner:
        #   Manipulate IRV --> use IRV, we know.
        #   Prevent a Condorcet winner --> trivial.
        #   * If incomplete: cf. 1), in fact it is the same.
        # When do we calculate IRV? It can be after we examine IRV.w; so,
        # not at the very beginning
        n_m = self.pop.matrix_duels[c, self.w]
        n_s = self.pop.V - n_m
        candidates = np.array(range(self.pop.C))
        preferences_utilities_s = self.pop.preferences_utilities[
            np.logical_not(self.v_wants_to_help_c[:, c]), :]
        matrix_duels_temp = (
            preferences_utilities_to_matrix_duels(preferences_utilities_s))
        self._mylogm("CM: matrix_duels_temp =", matrix_duels_temp, 3)
        # More preliminary checks
        # It's more convenient to put them in that method, because we need
        # preferences_utilities_s and matrix_duels_temp.
        # min_d(matrix_duels_temp[c, d]) + n_m > (n_s + n_m) / 2, i.e.:
        # n_m >= n_s + 1 - 2 * min_d(matrix_duels_temp[c, d])
        d_neq_c = (np.array(range(self.pop.C)) != c)
        n_manip_becomes_cond = np.maximum(
            n_s + 1 - 2 * np.min(matrix_duels_temp[c, d_neq_c]),
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
        # matrix_duels_temp[d, e]. We simply need that:
        # matrix_duels_temp[d, e] <= (n_s + n_m) / 2
        # 2 * max_d(min_e(matrix_duels_temp[d, e])) - n_s <= n_m
        n_manip_prevent_cond = 0
        for d in candidates[d_neq_c]:
            e_neq_d = (np.array(range(self.pop.C)) != d)
            n_prevent_d = np.maximum(
                2 * np.min(matrix_duels_temp[d, e_neq_d]) - n_s, 0)
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

        is_quick_escape_one = False
        if self.w == self.IRV.w:
            if self.CM_option == "fast":
                self.IRV.CM_option = "fast"
            elif self.CM_option == "slow":
                self.IRV.CM_option = "slow"
            else:
                self.IRV.CM_option = "exact"
            if optimize_bounds:
                self.IRV.CM_c_with_bounds(c)
            else:
                self.IRV.CM_c(c)
                is_quick_escape_one = True
            self._mylog('CM: c != self.IRV.w == self.w', 3)
            self._mylogv("CM: self.IRV._sufficient_coalition_size_CM[c] =",
                         self.IRV._sufficient_coalition_size_CM[c], 3)
            self._update_sufficient(
                self._sufficient_coalition_size_CM, c,
                max(self.IRV._sufficient_coalition_size_CM[c],
                    n_manip_prevent_cond),
                'CM: Update sufficient_coalition_size[c] =')
            self._mylogv("CM: self.IRV._necessary_coalition_size_CM[c] =",
                         self.IRV._necessary_coalition_size_CM[c], 3)
            self._update_necessary(
                self._necessary_coalition_size_CM, c,
                min(n_manip_becomes_cond,
                    max(self.IRV._necessary_coalition_size_CM[c],
                        n_manip_prevent_cond)),
                'CM: Update necessary_coalition_size[c] =')
        else:
            if c == self.IRV.w:
                self._mylog('CM: c == self.IRV.w != self.w', 3)
                self._mylogv("CM: sufficient size for IRV (sincere IRV) =",
                             n_m, 3)
                self._update_sufficient(
                    self._sufficient_coalition_size_CM, c,
                    max(n_m, n_manip_prevent_cond),
                    'CM: Update sufficient_coalition_size[c] =')
            else:
                self._mylog('CM: c, self.w, self.IRV.w all distinct', 3)
                # We do not know how many manipulators can make c win in IRV
                # (note that it would not be the same set of manipulators in
                # IRV and here)
        # self._mylogv("CM: Preliminary checks.2: " +
        #              "necessary_coalition_size_CM[c] =",
        #              self._necessary_coalition_size_CM[c], 3)
        # self._mylogv("CM: Preliminary checks.2: " +
        #              "sufficient_coalition_size_CM[c] =",
        #              self._sufficient_coalition_size_CM[c], 3)
        # self._mylogv("CM: Preliminary checks.2: " +
        #              "n_m =", self.pop.matrix_duels[c, self.w], 3)
        # if self._necessary_coalition_size_CM[c] > n_m:
        #     self._mylogv("CM: Preliminary checks.2: CM is False for c =",
        #                  c, 2)
        # elif self._sufficient_coalition_size_CM[c] <= n_m:
        #     self._mylogv("CM: Preliminary checks.2: CM is True for c =",
        #                  c, 2)
        # else:
        #     self._mylogv("CM: Preliminary checks.2: CM is unknown for c =",
        #                  c, 2)
        # Real work
        is_quick_escape_two = False
        if self.CM_option == 'exact':
            is_quick_escape_two = self._CM_main_work_c_exact(
                c, optimize_bounds)
        return is_quick_escape_one or is_quick_escape_two


if __name__ == '__main__':
    # A quick demo
    preferences_utilities = np.random.randint(-5, 5, (10, 5))
    pop = Population(preferences_utilities)
    election = CondorcetAbsIRV(pop)
    election.CM_option = 'slow'
    election.demo(log_depth=3)