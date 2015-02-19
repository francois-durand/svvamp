# -*- coding: utf-8 -*-
"""
Created on Thu Oct  9 13:31:30 2014
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
from svvamp.Preferences.Population import _Storage
from svvamp.VotingSystems.Election import Election
from svvamp.VotingSystems.Election import neginf_to_zero
from svvamp.VotingSystems.IRVResult import IRVResult
from svvamp.VotingSystems.ExhaustiveBallot import ExhaustiveBallot


class IRV(IRVResult, Election):
    """Instant-Runoff Voting (IRV). Also known as Single Transferable Voting,
    Alternative Vote, Hare method.
    
    Inherits functions and optional parameters from superclasses
    :class:`~svvamp.ElectionResult` and :class:`~svvamp.Election`.

    :Example:

        >>> import svvamp
        >>> pop = svvamp.PopulationSpheroid(V=100, C=5)
        >>> election = svvamp.IRV(pop)

    The candidate who is ranked first by least voters is eliminated. Then
    we iterate. Ties are broken in favor of lower-index candidates: in case
    of a tie, the tied candidate with highest index is eliminated.

    :meth:`~svvamp.Election.CM`: Deciding CM is NP-complete.

        * :attr:`~svvamp.Election.CM_option` = ``'fast'``:
          Polynomial heuristic. Can prove CM but unable to decide non-CM
          (except in rare obvious cases).
        * :attr:`~svvamp.Election.CM_option` = ``'slow'``:
          Rely on :class:`~svvamp.ExhaustiveBallot`'s
          exact algorithm.
          Non-polynomial heuristic (:math:`2^C`). Quite efficient to prove CM
          or non-CM.
        * :attr:`~svvamp.Election.CM_option` = ``'exact'``:
          Non-polynomial algorithm (:math:`C!`) adapted from Walsh, 2010.

    :meth:`~svvamp.Election.ICM`: Exact in polynomial time.

    :meth:`~svvamp.Election.IM`: Deciding IM is NP-complete.

        * :attr:`~svvamp.Election.IM_option` = ``'lazy'``:
          Lazy algorithm from superclass :class:`~svvamp.Election`.
        * :attr:`~svvamp.Election.IM_option` = ``'exact'``:
          Non-polynomial algorithm (:math:`C!`) adapted from Walsh, 2010.

    :meth:`~svvamp.Election.not_IIA`: Non-polynomial
    or non-exact algorithms from superclass :class:`~svvamp.Election`.

    :meth:`~svvamp.Election.TM`: Exact in polynomial time.

    :meth:`~svvamp.Election.UM`: Deciding UM is NP-complete.

        * :attr:`~svvamp.Election.UM_option` = ``'fast'``:
          Polynomial heuristic. Can prove UM but unable to decide non-UM
          (except in rare obvious cases).
        * :attr:`~svvamp.Election.UM_option` = ``'exact'``:
          Non-polynomial algorithm (:math:`C!`) adapted from Walsh, 2010.

    References:

        'Single transferable vote resists strategic voting', John J. Bartholdi
        and James B. Orlin, 1991.

        'On The Complexity of Manipulating Elections', Tom Coleman and Vanessa
        Teague, 2007.

        'Manipulability of Single Transferable Vote', Toby Walsh, 2010.

    .. seealso:: :class:`~svvamp.ExhaustiveBallot`,
                 :class:`~svvamp.IRVDuels`,
                 :class:`~svvamp.ICRV`,
                 :class:`~svvamp.CondorcetAbsIRV`.
                 :class:`~svvamp.CondorcetVtbIRV`.
    """
    # Exceptionally, for this voting system, results are stored in the 
    # Population object, so that they can be used by Condorcet-IRV.

    _layout_name = 'IRV'
    _options_parameters = Election._options_parameters.copy()
    _options_parameters.update(IRVResult._options_parameters)
    _options_parameters['UM_option'] = {'allowed': {'fast', 'exact'},
                                        'default': 'fast'}
    _options_parameters['CM_option'] = {'allowed': {'fast', 'slow', 'exact'},
                                        'default': 'fast'}
    _options_parameters['TM_option'] = {'allowed': ['exact'],
                                        'default': 'exact'}
    _options_parameters['ICM_option'] = {'allowed': {'exact'},
                                           'default': 'exact'}
    _options_parameters['fast_algo'] = {
        'allowed': {'c_minus_max', 'minus_max', 'hardest_first'},
        'default': 'c_minus_max'
    }

    def __init__(self, population, freeze_options=False, **kwargs):
        self.EB = ExhaustiveBallot(population, freeze_options=True)
        self.freeze_options = freeze_options
        super().__init__(population, **kwargs)
        self.freeze_options = False
        self._log_identity = "IRV"
        self._class_result = IRV
        self._log_depth = 0
        self._with_two_candidates_reduces_to_plurality = True
        self._is_based_on_strict_rankings = True
        self._meets_majority_favorite_c_vtb_ctb = True
        self._precheck_UM = False
        self._precheck_ICM = False

    #%% Dealing with the options

    @property
    def fast_algo(self):
        """String. Algorithm used for CM (resp. UM) when CM_option (resp.
        UM_option) is 'fast'. Mostly for developers.
        """
        return self._fast_algo

    @fast_algo.setter
    def fast_algo(self, value):
        try:
            if self._fast_algo == value:
                return
        except AttributeError:
            pass
        if value in self.options_parameters['fast_algo']['allowed']:
            self._mylogv("Setting fast_algo =", value, 1)
            self._fast_algo = value
            self._forget_UM()
            self._forget_CM()
        else:
            raise ValueError("Unknown fast algorithm: " + format(value))

    @property
    def log_UM(self):
        """String. Parameters used to compute UM."""
        if self.UM_option == 'exact':
            return "UM_option = exact"
        else:
            return "UM_option = " + self.UM_option + ", fast_algo = " + \
                   self.fast_algo

    @property
    def log_CM(self):
        """String. Parameters used to compute CM."""
        if self.CM_option == 'exact':
            return "CM_option = exact"
        else:
            return ("CM_option = " + self.CM_option +
                    ", fast_algo = " + self.fast_algo +
                    ", " + self.log_ICM +
                    ", " + self.log_TM)

    #%% Initialize the options

    def _initialize_options(self, **kwargs):
        if self.pop._irv_options is None:
            self.pop._irv_options = _Storage()
            Election._initialize_options(self, **kwargs)
        elif not self.freeze_options:
            Election._initialize_options(self, **kwargs)

    #%% Store / forget the examples of manipulation path

    def _forget_TM(self):
        self._TM_was_initialized = False
        self._TM_was_computed_with_candidates = False
        self._example_path_TM = {c: None for c in range(self.pop.C)}
        self._sufficient_coalition_size_TM = np.full(self.pop.C, np.inf)
        # This is a specificity for IRV and Exhaustive Ballot (cf. EB).

    def _forget_UM(self):
        self._UM_was_initialized = False
        self._UM_was_computed_with_candidates = False
        self._example_path_UM = {c: None for c in range(self.pop.C)}

    def _forget_CM(self):
        self._CM_was_initialized = False
        self._CM_was_computed_with_candidates = False
        self._CM_was_computed_full = False
        self._example_path_CM = {c: None for c in range(self.pop.C)}

    #%% Cache variables in the Population object, so that they can be used
    #   by Condorcet-IRV.

    def _forget_manipulations(self):
        # This is called only when a new IRV object is created.
        # If the population already stores the manipulation results for IRV,
        # we must not forget them.
        if self.pop._irv_manip is None:
            self.pop._irv_manip = _Storage()
            self.pop.ensure_voters_sorted_by_ordinal_preferences()
            self.EB = ExhaustiveBallot(self.pop, freeze_options=True)
            # All commented lines are managed by Exhaustive Ballot
            # self._v_wants_to_help_c = None
            # self._c_has_supporters = None
            # self._losing_candidates = None
            # self._forget_IIA()
            # self._forget_TM()
            # self._forget_ICM()
            self._forget_IM()
            self._forget_UM()
            self._forget_CM()
            self._forget_manipulations_subclass()

    # For these variables, we identify with Exhaustive Ballot

    @property
    def _v_wants_to_help_c(self):
        return self.pop._eb_manip._v_wants_to_help_c

    @_v_wants_to_help_c.setter
    def _v_wants_to_help_c(self, value):
        self.pop._eb_manip._v_wants_to_help_c = value

    @property
    def _c_has_supporters(self):
        return self.pop._eb_manip._c_has_supporters

    @_c_has_supporters.setter
    def _c_has_supporters(self, value):
        self.pop._eb_manip._c_has_supporters = value

    @property
    def _losing_candidates(self):
        return self.pop._eb_manip._losing_candidates

    @_losing_candidates.setter
    def _losing_candidates(self, value):
        self.pop._eb_manip._losing_candidates = value

    @property
    def _is_IIA(self):
        return self.pop._eb_manip._is_IIA

    @_is_IIA.setter
    def _is_IIA(self, value):
        self.pop._eb_manip._is_IIA = value

    @property
    def _example_winner_IIA(self):
        return self.pop._eb_manip._example_winner_IIA

    @_example_winner_IIA.setter
    def _example_winner_IIA(self, value):
        self.pop._eb_manip._example_winner_IIA = value

    @property
    def _example_subset_IIA(self):
        return self.pop._eb_manip._example_subset_IIA

    @_example_subset_IIA.setter
    def _example_subset_IIA(self, value):
        self.pop._eb_manip._example_subset_IIA = value

    @property
    def _TM_was_initialized(self):
        return self.pop._eb_manip._TM_was_initialized

    @_TM_was_initialized.setter
    def _TM_was_initialized(self, value):
        self.pop._eb_manip._TM_was_initialized = value

    @property
    def _TM_was_computed_with_candidates(self):
        return self.pop._eb_manip._TM_was_computed_with_candidates

    @_TM_was_computed_with_candidates.setter
    def _TM_was_computed_with_candidates(self, value):
        self.pop._eb_manip._TM_was_computed_with_candidates = value

    @property
    def _candidates_TM(self):
        return self.pop._eb_manip._candidates_TM

    @_candidates_TM.setter
    def _candidates_TM(self, value):
        self.pop._eb_manip._candidates_TM = value

    @property
    def _is_TM(self):
        return self.pop._eb_manip._is_TM

    @_is_TM.setter
    def _is_TM(self, value):
        self.pop._eb_manip._is_TM = value

    @property
    def _sufficient_coalition_size_TM(self):
        return self.pop._eb_manip._sufficient_coalition_size_TM

    @_sufficient_coalition_size_TM.setter
    def _sufficient_coalition_size_TM(self, value):
        self.pop._eb_manip._sufficient_coalition_size_TM = value

    @property
    def _example_path_TM(self):
        return self.pop._eb_manip._example_path_TM

    @_example_path_TM.setter
    def _example_path_TM(self, value):
        self.pop._eb_manip._example_path_TM = value

    @property
    def _ICM_was_initialized(self):
        return self.pop._eb_manip._ICM_was_initialized

    @_ICM_was_initialized.setter
    def _ICM_was_initialized(self, value):
        self.pop._eb_manip._ICM_was_initialized = value

    @property
    def _ICM_was_computed_with_candidates(self):
        return self.pop._eb_manip._ICM_was_computed_with_candidates

    @_ICM_was_computed_with_candidates.setter
    def _ICM_was_computed_with_candidates(self, value):
        self.pop._eb_manip._ICM_was_computed_with_candidates = value

    @property
    def _ICM_was_computed_full(self):
        return self.pop._eb_manip._ICM_was_computed_full

    @_ICM_was_computed_full.setter
    def _ICM_was_computed_full(self, value):
        self.pop._eb_manip._ICM_was_computed_full = value

    @property
    def _candidates_ICM(self):
        return self.pop._eb_manip._candidates_ICM

    @_candidates_ICM.setter
    def _candidates_ICM(self, value):
        self.pop._eb_manip._candidates_ICM = value

    @property
    def _sufficient_coalition_size_ICM(self):
        return self.pop._eb_manip._sufficient_coalition_size_ICM

    @_sufficient_coalition_size_ICM.setter
    def _sufficient_coalition_size_ICM(self, value):
        self.pop._eb_manip._sufficient_coalition_size_ICM = value

    @property
    def _necessary_coalition_size_ICM(self):
        return self.pop._eb_manip._necessary_coalition_size_ICM

    @_necessary_coalition_size_ICM.setter
    def _necessary_coalition_size_ICM(self, value):
        self.pop._eb_manip._necessary_coalition_size_ICM = value

    @property
    def _bounds_optimized_ICM(self):
        return self.pop._eb_manip._bounds_optimized_ICM

    @_bounds_optimized_ICM.setter
    def _bounds_optimized_ICM(self, value):
        self.pop._eb_manip._bounds_optimized_ICM = value

    @property
    def _is_ICM(self):
        return self.pop._eb_manip._is_ICM

    @_is_ICM.setter
    def _is_ICM(self, value):
        self.pop._eb_manip._is_ICM = value
#    @property
#    def _example_path_ICM(self):
#        return self.pop._eb_manip._example_path_ICM
#    @_example_path_ICM.setter
#    def _example_path_ICM(self, value):
#        self.pop._eb_manip._example_path_ICM = value
        
    @property
    def _IIA_subset_maximum_size(self):
        return self.pop._eb_options._IIA_subset_maximum_size

    @_IIA_subset_maximum_size.setter
    def _IIA_subset_maximum_size(self, value):
        self.pop._eb_options._IIA_subset_maximum_size = value

    @property
    def _TM_option(self):
        return self.pop._eb_options._TM_option

    @_TM_option.setter
    def _TM_option(self, value):
        self.pop._eb_options._TM_option = value

    @property
    def _ICM_option(self):
        return self.pop._eb_options._ICM_option

    @_ICM_option.setter
    def _ICM_option(self, value):
        self.pop._eb_options._ICM_option = value

   # For these variables, special storage for IRV

    @property
    def _IM_was_initialized(self):
        return self.pop._irv_manip._IM_was_initialized

    @_IM_was_initialized.setter
    def _IM_was_initialized(self, value):
        self.pop._irv_manip._IM_was_initialized = value

    @property
    def _IM_was_computed_with_candidates(self):
        return self.pop._irv_manip._IM_was_computed_with_candidates

    @_IM_was_computed_with_candidates.setter
    def _IM_was_computed_with_candidates(self, value):
        self.pop._irv_manip._IM_was_computed_with_candidates = value

    @property
    def _IM_was_computed_with_voters(self):
        return self.pop._irv_manip._IM_was_computed_with_voters

    @_IM_was_computed_with_voters.setter
    def _IM_was_computed_with_voters(self, value):
        self.pop._irv_manip._IM_was_computed_with_voters = value

    @property
    def _IM_was_computed_full(self):
        return self.pop._irv_manip._IM_was_computed_full

    @_IM_was_computed_full.setter
    def _IM_was_computed_full(self, value):
        self.pop._irv_manip._IM_was_computed_full = value

    @property
    def _v_IM_for_c(self):
        return self.pop._irv_manip._v_IM_for_c

    @_v_IM_for_c.setter
    def _v_IM_for_c(self, value):
        self.pop._irv_manip._v_IM_for_c = value

    @property
    def _candidates_IM(self):
        return self.pop._irv_manip._candidates_IM

    @_candidates_IM.setter
    def _candidates_IM(self, value):
        self.pop._irv_manip._candidates_IM = value

    @property
    def _is_IM(self):
        return self.pop._irv_manip._is_IM

    @_is_IM.setter
    def _is_IM(self, value):
        self.pop._irv_manip._is_IM = value
#    @property
#    def _example_path_IM(self):
#        return self.pop._irv_manip._example_path_IM
#    @_example_path_IM.setter
#    def _example_path_IM(self, value):
#        self.pop._irv_manip._example_path_IM = value
            
    @property
    def _UM_was_initialized(self):
        return self.pop._irv_manip._UM_was_initialized

    @_UM_was_initialized.setter
    def _UM_was_initialized(self, value):
        self.pop._irv_manip._UM_was_initialized = value

    @property
    def _UM_was_computed_with_candidates(self):
        return self.pop._irv_manip._UM_was_computed_with_candidates

    @_UM_was_computed_with_candidates.setter
    def _UM_was_computed_with_candidates(self, value):
        self.pop._irv_manip._UM_was_computed_with_candidates = value

    @property
    def _candidates_UM(self):
        return self.pop._irv_manip._candidates_UM

    @_candidates_UM.setter
    def _candidates_UM(self, value):
        self.pop._irv_manip._candidates_UM = value

    @property
    def _is_UM(self):
        return self.pop._irv_manip._is_UM

    @_is_UM.setter
    def _is_UM(self, value):
        self.pop._irv_manip._is_UM = value

    @property
    def _example_path_UM(self):
        return self.pop._irv_manip._example_path_UM

    @_example_path_UM.setter
    def _example_path_UM(self, value):
        self.pop._irv_manip._example_path_UM = value

    @property
    def _CM_was_initialized(self):
        return self.pop._irv_manip._CM_was_initialized

    @_CM_was_initialized.setter
    def _CM_was_initialized(self, value):
        self.pop._irv_manip._CM_was_initialized = value

    @property
    def _CM_was_computed_with_candidates(self):
        return self.pop._irv_manip._CM_was_computed_with_candidates

    @_CM_was_computed_with_candidates.setter
    def _CM_was_computed_with_candidates(self, value):
        self.pop._irv_manip._CM_was_computed_with_candidates = value

    @property
    def _CM_was_computed_full(self):
        return self.pop._eb_manip._CM_was_computed_full

    @_CM_was_computed_full.setter
    def _CM_was_computed_full(self, value):
        self.pop._eb_manip._CM_was_computed_full = value

    @property
    def _candidates_CM(self):
        return self.pop._irv_manip._candidates_CM

    @_candidates_CM.setter
    def _candidates_CM(self, value):
        self.pop._irv_manip._candidates_CM = value

    @property
    def _sufficient_coalition_size_CM(self):
        return self.pop._irv_manip._sufficient_coalition_size_CM

    @_sufficient_coalition_size_CM.setter
    def _sufficient_coalition_size_CM(self, value):
        self.pop._irv_manip._sufficient_coalition_size_CM = value

    @property
    def _necessary_coalition_size_CM(self):
        return self.pop._irv_manip._necessary_coalition_size_CM

    @_necessary_coalition_size_CM.setter
    def _necessary_coalition_size_CM(self, value):
        self.pop._irv_manip._necessary_coalition_size_CM = value

    @property
    def _bounds_optimized_CM(self):
        return self.pop._irv_manip._bounds_optimized_CM

    @_bounds_optimized_CM.setter
    def _bounds_optimized_CM(self, value):
        self.pop._irv_manip._bounds_optimized_CM = value

    @property
    def _is_CM(self):
        return self.pop._irv_manip._is_CM

    @_is_CM.setter
    def _is_CM(self, value):
        self.pop._irv_manip._is_CM = value

    @property
    def _example_path_CM(self):
        # Rule: each time self._sufficient_coalition_size_CM[c] is decreased,
        # the corresponding elimination path must be stored.
        return self.pop._irv_manip._example_path_CM

    @_example_path_CM.setter
    def _example_path_CM(self, value):
        self.pop._irv_manip._example_path_CM = value

    @property
    def _fast_algo(self):
        return self.pop._irv_options._fast_algo

    @_fast_algo.setter
    def _fast_algo(self, value):
        self.pop._irv_options._fast_algo = value

    @property
    def _IM_option(self):
        return self.pop._irv_options._IM_option

    @_IM_option.setter
    def _IM_option(self, value):
        self.pop._irv_options._IM_option = value

    @property
    def _UM_option(self):
        return self.pop._irv_options._UM_option

    @_UM_option.setter
    def _UM_option(self, value):
        self.pop._irv_options._UM_option = value

    @property
    def _CM_option(self):
        return self.pop._irv_options._CM_option

    @_CM_option.setter
    def _CM_option(self, value):
        self.pop._irv_options._CM_option = value

    #%% Individual manipulation (IM)            

    def _IM_main_work_v_exact(self, v, c_is_wanted,
                              nb_wanted_undecided, stop_if_true):
        self._mylogv("self._v_IM_for_c[v, :] =",
                     self._v_IM_for_c[v, :], 3)
        other_voters = (np.array(range(self.pop.V)) != v)
        n_s = self.pop.V - 1
        r = 0
        is_candidate_alive_begin_r = np.zeros((self.pop.C - 1, self.pop.C), 
                                              dtype=np.bool)
        is_candidate_alive_begin_r[0, :] = np.ones(self.pop.C)
        ballot_m_begin_r = -np.ones(self.pop.C - 1, dtype=np.int)
        scores_tot_begin_r = np.zeros((self.pop.C - 1, self.pop.C))
        scores_tot_begin_r[0, :] = np.sum(np.equal(
            self.pop.preferences_borda_vtb[other_voters, :],
            np.max(self.pop.preferences_borda_vtb[other_voters, :], 1)[
                :, np.newaxis]
        ), 0)
        self._mylogv("IM_aux_exact: r =", r, 3)
        self._mylogv("IM_aux_exact: scores_tot_begin_r[r] =", 
                     scores_tot_begin_r[0, :], 3)
        # strategy_r[r] is False (0) if we keep our ballot, True (1) if we
        # change it to vote for the natural loser. If strategy_r == 2, we have
        # tried everything.
        strategy_r = np.zeros(self.pop.C - 1, dtype=np.int)
        natural_loser_r = np.zeros(self.pop.C - 1, dtype=np.int)
        natural_loser_r[0] = np.where(
            scores_tot_begin_r[0, :] ==
            np.nanmin(scores_tot_begin_r[0, :]))[0][-1]
        eliminated_d_r = np.zeros(self.pop.C - 1)
        self._mylogv("IM_aux_exact: natural_loser_r[r] =", 
                     natural_loser_r[r], 3)
        # If w has too many votes, then manipulation is not possible.
        if (scores_tot_begin_r[0, self.w] + (self.w == 0) > 
                n_s + 1 - scores_tot_begin_r[0, self.w]):
            self._mylog("IM_aux_exact: Manipulation impossible by this " +
                        "path (w has too many votes)", 3)
            r = -1
        while True:
            if r < 0:
                self._mylog("IM_aux_exact: End of exploration", 3)
                neginf_to_zero(self._v_IM_for_c[v, :])
                return
            if strategy_r[r] > 1:
                r -= 1
                self._mylogv("IM_aux_exact: Tried everything for round r, " + 
                             "go back to r =", r, 3)
                self._mylogv("IM_aux_exact: r =", r, 3)
                if r >= 0:
                    strategy_r[r] += 1
                continue
            self._mylogv("IM_aux_exact: strategy_r[r] =", strategy_r[r], 3)
            if strategy_r[r] == 0:
                ballot_m_temp = ballot_m_begin_r[r]
                d = natural_loser_r[r]
            else:
                if ballot_m_begin_r[r] != -1:
                    # We cannot change our ballot.
                    self._mylog("IM_aux_exact: Cannot change our ballot.", 3)
                    strategy_r[r] += 1
                    continue
                else:
                    ballot_m_temp = natural_loser_r[r]
                    scores_tot_temp = np.copy(scores_tot_begin_r[r, :])
                    scores_tot_temp[ballot_m_temp] += 1
                    self._mylogv("IM_aux_exact: scores_tot_temp =", 
                                 scores_tot_temp, 3)
                    d = np.where(
                        scores_tot_temp ==
                        np.nanmin(scores_tot_temp))[0][-1]
                    if d == natural_loser_r[r]:
                        self._mylog("IM_aux_exact: Cannot save "
                                    "natural_loser_r.", 3)
                        strategy_r[r] += 1
                        continue
            self._mylogv("IM_aux_exact: d =", d, 3)
            eliminated_d_r[r] = d
            if r == self.pop.C - 2:
                is_candidate_alive_end = np.copy(
                    is_candidate_alive_begin_r[r, :])
                is_candidate_alive_end[d] = False
                c = np.argmax(is_candidate_alive_end)
                self._mylogv("IM_aux_exact: Winner =", c, 3)
                if np.isneginf(self._v_IM_for_c[v, c]):
                    self._v_IM_for_c[v, c] = True
                    self._candidates_IM[c] = True
                    self._voters_IM[v] = True
                    self._is_IM = True
                    self._mylogv("IM found for c =", c, 3)
                    if c_is_wanted[c]:
                        if stop_if_true:
                            return
                        nb_wanted_undecided -= 1
                    if nb_wanted_undecided == 0:
                        return  # We know everything we want for this voter
                strategy_r[r] += 1
                continue        
            # Calculate scores for next round
            is_candidate_alive_begin_r[r+1, :] = (
                is_candidate_alive_begin_r[r, :])
            is_candidate_alive_begin_r[r+1, d] = False
            self._mylogv("IM_aux_exact: is_candidate_alive_begin_r[r+1, :] =", 
                         is_candidate_alive_begin_r[r+1, :], 3)
            scores_tot_begin_r[r+1, :] = np.full(self.pop.C, np.nan) 
            scores_tot_begin_r[r+1, is_candidate_alive_begin_r[r+1, :]] = (
                np.sum(np.equal(
                    self.pop.preferences_borda_vtb[other_voters, :][
                        :, is_candidate_alive_begin_r[r+1, :]],
                    np.max(self.pop.preferences_borda_vtb[other_voters, :][
                        :, is_candidate_alive_begin_r[r+1, :]], 1
                    )[:, np.newaxis]
                ), 0))
            self._mylogv("IM_aux_exact: scores_s_begin_r[r+1, :] =", 
                         scores_tot_begin_r[r+1, :], 3)
            if ballot_m_temp == d:
                ballot_m_begin_r[r+1] = -1
            else:
                ballot_m_begin_r[r+1] = ballot_m_temp
            self._mylogv("IM_aux_exact: ballot_m_begin_r[r+1] =", 
                         ballot_m_begin_r[r+1], 3)
            if ballot_m_begin_r[r+1] != -1:
                scores_tot_begin_r[r+1, ballot_m_begin_r[r+1]] += 1
            self._mylogv("IM_aux_exact: scores_tot_begin_r[r+1, :] =", 
                         scores_tot_begin_r[r+1, :], 3)
        
            # If an opponent has too many votes, then manipulation is
            # not possible.
            if (scores_tot_begin_r[r+1, self.w] + (self.w == 0) > 
                    n_s + 1 - scores_tot_begin_r[r+1, self.w]):
                self._mylog("IM_aux_exact: Manipulation impossible by this " +
                            "path (w will have too many votes)", 3)
                strategy_r[r] += 1
                continue
        
            # Update other variables for next round
            strategy_r[r+1] = 0
            natural_loser_r[r+1] = np.where(
                scores_tot_begin_r[r+1, :] ==
                np.nanmin(scores_tot_begin_r[r+1, :]))[0][-1]
            r += 1
            self._mylogv("IM_aux_exact: r =", r, 3)
            self._mylogv("IM_aux_exact: natural_loser_r[r] =", 
                         natural_loser_r[r], 3)
        
    def _IM_preliminary_checks_general_subclass(self):
        if np.all(np.equal(self._v_IM_for_c, False)):
            return
        if self.IM_option == "exact":
            # In that case, we check Exhaustive Ballot first.
            self.EB.IM_option = "exact"
            if self.EB.IM()[0] == False:
                self._mylog("IM impossible (since it is impossible for " + 
                            "Exhaustive Ballot)", 2)
                self._v_IM_for_c[:] = False
                # Other variables will be updated in
                # _IM_preliminary_checks_general.

    def _IM_preliminary_checks_v_subclass(self, v):
        # Pretest based on Exhaustive Ballot
        if self.IM_option == "exact":
            if np.any(np.isneginf(self._v_IM_for_c[v, :])):
                # self.EB.IM_option = "exact"
                candidates_IM_v = self.EB.IM_v_with_candidates(v)[2]
                self._mylogv("IM: Preliminary checks: " +
                             "EB._v_IM_for_c[v, :] =",
                             candidates_IM_v, 3)
                self._v_IM_for_c[v, candidates_IM_v == False] = False

    #%% Trivial Manipulation (TM)

    def _TM_initialize_general(self, with_candidates):
        self.EB._TM_initialize_general(with_candidates)

    def _TM_preliminary_checks_general(self):
        self.EB._TM_preliminary_checks_general()

    def _TM_initialize_c(self, c):
        self.EB._TM_initialize_c(c)

    def _TM_main_work_c(self, c):
        self.EB._TM_main_work_c(c)

    #%% Unison manipulation (UM)

    # TODO: implement UM slow (use EB exact, but not IRV exact).
    
    def _UM_aux_fast(self, c, n_m, preferences_borda_s):
        """Fast algorithm used for UM.
        
        Arguments:
        c -- Integer. Candidate for which we want to manipulate.
        n_m -- Integer. Number of manipulators.
        preferences_borda_s -- 2d integer. Preferences of the sincere voters
            (in Borda format).
        
        Returns:
        manip_found_fast -- Boolean.
            Whether a manipulation was found or not.
        example_path_fast -- An example of elimination path that realizes the
            manipulation with 'n_m' manipulators. 
            example_path_fast[k] is the k-th candidate eliminated. If no 
            manipulation is found, example_path is NaN.
        """
        # At each round, we examine all candidates d that we can eliminate.
        # If several d are possible, we choose the one that maximize
        # a heuristic parameter denoted "situation_c".
        candidates = np.array(range(self.pop.C))
        n_s = preferences_borda_s.shape[0]
        example_path_fast = []
        is_candidate_alive = np.ones(self.pop.C, dtype=np.bool)
        # Sincere scores (eliminated candidates will have nan)
        scores_s = np.sum(np.equal(
            preferences_borda_s,
            np.max(preferences_borda_s, 1)[:, np.newaxis]
        ), 0)
        # Manipulators' ballot: if blocked, index of the candidate. Else, -1
        ballot_m = -1
        # Total scores (eliminated candidates will have nan)
        scores_tot = scores_s
        for r in range(self.pop.C-1):
            self._mylogv("UM_aux_fast: Round r =", r, 3)
            self._mylogv("UM_aux_fast: scores_s =", scores_s, 3)
            self._mylogv("UM_aux_fast: ballot_m =", ballot_m, 3)
            self._mylogv("UM_aux_fast: scores_tot =", scores_tot, 3)
            # If an opponent has too many votes, then manipulation is
            # not possible.
            max_score = np.nanmax(scores_tot[candidates != c])
            most_serious_opponent = np.where(scores_tot == max_score)[0][0]
            if (max_score + (most_serious_opponent < c) > 
                    n_s + n_m - max_score):
                self._mylogv("UM_aux_fast: most_serious_opponent =", 
                             most_serious_opponent, 3)
                self._mylog("UM_aux_fast: Manipulation impossible by this " +
                            "path (an opponent has too many votes)", 3)
                return False, np.nan
            # Initialize the loop on d
            best_d = -1      # d which is eliminated with the 'best' strategy
            best_situation_for_c = -np.inf
            ballot_free_r = False  # Will be updated
            scores_s_end_r = np.nan
            ballot_m_end_r = -1  # Will be updated
            scores_tot_end_r = np.nan
            natural_loser = np.where(
                scores_tot == np.nanmin(scores_tot))[0][-1]
            self._mylogv("UM_aux_fast: natural_loser =", natural_loser, 3)
            for vote_for_natural_loser in [False, True]:
                # Which candidate will lose?
                if not vote_for_natural_loser:
                    # In fact, it means that we do not change ballot_m. It
                    # includes the case where we vote already for 
                    # natural_loser.
                    self._mylog("UM_aux_fast: Strategy: keep our ballot", 3)
                    ballot_m_temp = ballot_m
                    d = natural_loser
                else:
                    if ballot_m != -1:
                        self._mylog("UM_aux_fast: No other strategy (cannot "
                                    "change our ballot).", 3)
                        continue
                    else:
                        self._mylog("UM_aux_fast: Strategy: vote for "
                                    "natural_loser", 3)
                        scores_tot_temp = np.copy(scores_tot)
                        ballot_m_temp = natural_loser
                        scores_tot_temp[natural_loser] += n_m
                        d = np.where(
                            scores_tot_temp == np.nanmin(scores_tot_temp)
                        )[0][-1]
                        self._mylogv("UM_aux_fast: ballot_m_temp =",
                                     ballot_m_temp, 3)
                        self._mylogv("UM_aux_fast: scores_tot_temp =",
                                     scores_tot_temp, 3)
                self._mylogv("UM_aux_fast: d =", d, 3)
                if d == c:
                    self._mylog("UM_aux_fast: This eliminates c", 3)
                    continue
                # Compute heuristic: "Situation" for c
                # Now we compute the tables at beginning of next round
                if r == self.pop.C - 2:
                    best_d = d
                    break
                is_candidate_alive_temp = np.copy(is_candidate_alive)
                is_candidate_alive_temp[d] = False
                scores_s_temp = np.full(self.pop.C, np.nan) 
                scores_s_temp[is_candidate_alive_temp] = np.sum(np.equal(
                    preferences_borda_s[:, is_candidate_alive_temp],
                    np.max(preferences_borda_s[:, is_candidate_alive_temp], 1)[
                        :, np.newaxis]
                ), 0)
                if ballot_m_temp == d:
                    ballot_m_temp = -1
                ballot_free_temp = (ballot_m_temp == -1)
                scores_tot_temp = np.copy(scores_s_temp)
                if ballot_m_temp != -1:
                    scores_tot_temp[ballot_m_temp] += n_m
                if self.fast_algo == "c_minus_max":
                    situation_for_c = (
                        scores_s_temp[c] -
                        np.nanmax(scores_tot_temp[candidates != c]))
                elif self.fast_algo == "minus_max":
                    situation_for_c = - np.nanmax(
                        scores_tot_temp[candidates != c])
                elif self.fast_algo == "hardest_first":
                    situation_for_c = (ballot_m_temp != -1)
                else:
                    raise ValueError("Unknown fast algorithm: " + 
                                     format(self.fast_algo))
                self._mylogv("UM_aux_fast: scores_s_temp =", 
                             scores_s_temp, 3)
                self._mylogv("UM_aux_fast: ballot_m_temp =", 
                             ballot_m_temp, 3)
                self._mylogv("UM_aux_fast: scores_tot_temp =", 
                             scores_tot_temp, 3)
                self._mylogv("UM_aux_fast: situation_for_c =", 
                             situation_for_c, 3)
                # Is the the best d so far?
                # Lexicographic comparison on three criteria (highest 
                # 'situation', lowest number of manipulators, lowest index). 
                if ([situation_for_c, ballot_free_temp, -d] >
                        [best_situation_for_c, ballot_free_r, -best_d]): 
                    best_d = d
                    best_situation_for_c = situation_for_c
                    ballot_free_r = ballot_free_temp
                    scores_s_end_r, scores_s_temp = scores_s_temp, None
                    ballot_m_end_r = ballot_m_temp
                    scores_tot_end_r, scores_tot_temp = scores_tot_temp, None
            self._mylogv("UM_aux_fast: best_d =", best_d, 3)
            if best_d == -1:
                return False, np.nan
            # Update variables for next round
            example_path_fast.append(best_d)
            is_candidate_alive[best_d] = False
            scores_s, scores_s_end_r = scores_s_end_r, None
            ballot_m = ballot_m_end_r
            scores_tot, scores_tot_end_r = scores_tot_end_r, None
        self._mylog("UM_aux_fast: Conclusion: manipulation found", 3)
        example_path_fast.append(c)
        self._mylogv("UM_aux_fast: example_path_fast =", example_path_fast, 3)
        return True, np.array(example_path_fast)

    def _UM_aux_exact(self, c, n_m, preferences_borda_s):
        """Exact algorithm used for UM.
        
        Arguments:
        c -- Integer. Candidate for which we want to manipulate.
        n_m -- Integer. Number of manipulators.
        preferences_borda_s -- 2d integer. Preferences of the sincere voters
            (in Borda format).
            
        Returns:
        manip_found_exact -- Boolean.
            Whether a manipulation was found or not.
        example_path -- An example of elimination path that realizes the
            manipulation with 'n_m' manipulators. 
            example_path[k] is the k-th candidate eliminated. If the 
            manipulation is impossible, example_path is NaN.
        """        
        candidates = np.array(range(self.pop.C))
        n_s = preferences_borda_s.shape[0]
        r = 0
        is_candidate_alive_begin_r = np.zeros((self.pop.C - 1, self.pop.C), 
                                              dtype=np.bool)
        is_candidate_alive_begin_r[0, :] = np.ones(self.pop.C)
        ballot_m_begin_r = -np.ones(self.pop.C - 1, dtype=np.int)
        scores_tot_begin_r = np.zeros((self.pop.C - 1, self.pop.C))
        scores_tot_begin_r[0, :] = np.sum(np.equal(
            preferences_borda_s,
            np.max(preferences_borda_s, 1)[:, np.newaxis]
        ), 0)
        self._mylogv("UM_aux_exact: r =", r, 3)
        self._mylogv("UM_aux_exact: scores_tot_begin_r[r] =", 
                     scores_tot_begin_r[0, :], 3)
        # If an opponent has too many votes, then manipulation is
        # not possible.
        max_score = np.nanmax(scores_tot_begin_r[0, candidates != c])
        most_serious_opponent = np.where(scores_tot_begin_r[0, :] == 
                                         max_score)[0][0]
        if (max_score + (most_serious_opponent < c) > 
                n_s + n_m - max_score):
            self._mylogv("UM_aux_exact: most_serious_opponent =", 
                         most_serious_opponent, 3)
            self._mylog("UM_aux_exact: Manipulation impossible by this " +
                        "path (an opponent has too many votes)", 3)
            r = -1
        # strategy_r[r] is False if we keep our ballot, True if we change it
        # to vote for the natural loser. If strategy_r == 2, we have 
        # tried everything.
        strategy_r = np.zeros(self.pop.C - 1, dtype=np.int)
        natural_loser_r = np.zeros(self.pop.C - 1, dtype=np.int)
        natural_loser_r[0] = np.where(
            scores_tot_begin_r[0, :] ==
            np.nanmin(scores_tot_begin_r[0, :])
        )[0][-1]
        eliminated_d_r = np.zeros(self.pop.C - 1)
        self._mylogv("UM_aux_exact: natural_loser_r[r] =", 
                     natural_loser_r[r], 3)
        while True:
            if r < 0:
                self._mylog("UM_aux_exact: End of exploration", 3)
                return False, np.nan
            if strategy_r[r] > 1:
                r -= 1
                self._mylogv("UM_aux_exact: Tried everything for round r, " + 
                             "go back to r =", r, 3)
                self._mylogv("UM_aux_exact: r =", r, 3)
                if r >= 0:
                    strategy_r[r] += 1
                continue
            self._mylogv("UM_aux_exact: strategy_r[r] =", strategy_r[r], 3)
            if strategy_r[r] == 0:
                ballot_m_temp = ballot_m_begin_r[r]
                d = natural_loser_r[r]
            else:
                if ballot_m_begin_r[r] != -1:
                    # We cannot change our ballot.
                    self._mylog("UM_aux_exact: Cannot change our ballot.", 3)
                    strategy_r[r] += 1
                    continue
                else:
                    ballot_m_temp = natural_loser_r[r]
                    scores_tot_temp = np.copy(scores_tot_begin_r[r, :])
                    scores_tot_temp[ballot_m_temp] += n_m
                    self._mylogv("UM_aux_exact: scores_tot_temp =", 
                                 scores_tot_temp, 3)
                    d = np.where(
                        scores_tot_temp ==
                        np.nanmin(scores_tot_temp)
                    )[0][-1]
                    if d == natural_loser_r[r]:
                        self._mylog("UM_aux_exact: Cannot save "
                                    "natural_loser_r.", 3)
                        strategy_r[r] += 1
                        continue
            self._mylogv("UM_aux_exact: d =", d, 3)
            if d == c:
                self._mylog("UM_aux_exact: This eliminates c.", 3)
                strategy_r[r] += 1
                continue
            eliminated_d_r[r] = d
            if r == self.pop.C - 2:
                example_path = np.concatenate((eliminated_d_r, np.array([c])))
                self._mylog("UM_aux_exact: UM found", 3)
                self._mylogv("UM_aux_exact: example_path =", example_path, 3)
                return True, example_path
        
            # Calculate scores for next round
            is_candidate_alive_begin_r[r+1, :] = (
                is_candidate_alive_begin_r[r, :])
            is_candidate_alive_begin_r[r+1, d] = False
            self._mylogv("UM_aux_exact: is_candidate_alive_begin_r[r+1, :] =", 
                         is_candidate_alive_begin_r[r+1, :], 3)
            scores_tot_begin_r[r+1, :] = np.full(self.pop.C, np.nan) 
            scores_tot_begin_r[r+1, is_candidate_alive_begin_r[r+1, :]] = (
                np.sum(np.equal(
                    preferences_borda_s[:, is_candidate_alive_begin_r[r+1, :]],
                    np.max(
                        preferences_borda_s[
                            :, is_candidate_alive_begin_r[r + 1, :]], 1
                    )[:, np.newaxis]
                ), 0))
            self._mylogv("UM_aux_exact: scores_s_begin_r[r+1, :] =", 
                         scores_tot_begin_r[r+1, :], 3)
            if ballot_m_temp == d:
                ballot_m_begin_r[r+1] = -1
            else:
                ballot_m_begin_r[r+1] = ballot_m_temp
            self._mylogv("UM_aux_exact: ballot_m_begin_r[r+1] =", 
                         ballot_m_begin_r[r+1], 3)
            if ballot_m_begin_r[r+1] != -1:
                scores_tot_begin_r[r+1, ballot_m_begin_r[r+1]] += n_m
            self._mylogv("UM_aux_exact: scores_tot_begin_r[r+1, :] =", 
                         scores_tot_begin_r[r+1, :], 3)
        
            # If an opponent has too many votes, then manipulation is
            # not possible.
            max_score = np.nanmax(scores_tot_begin_r[r+1, candidates != c])
            most_serious_opponent = np.where(scores_tot_begin_r[r+1, :] == 
                                             max_score)[0][0]
            if (max_score + (most_serious_opponent < c) > 
                    n_s + n_m - max_score):
                self._mylogv("UM_aux_exact: most_serious_opponent =", 
                             most_serious_opponent, 3)
                self._mylog("UM_aux_exact: Manipulation impossible by this " +
                            "path (an opponent will have too many votes)", 3)
                strategy_r[r] += 1
                continue
        
            # Update other variables for next round
            strategy_r[r+1] = 0
            natural_loser_r[r+1] = np.where(
                scores_tot_begin_r[r+1, :] ==
                np.nanmin(scores_tot_begin_r[r+1, :])
            )[0][-1]
            r += 1
            self._mylogv("UM_aux_exact: r =", r, 3)
            self._mylogv("UM_aux_exact: natural_loser_r[r] =", 
                         natural_loser_r[r], 3)

    def _UM_preliminary_checks_general_subclass(self):
        if np.all(np.equal(self._candidates_UM, False)):
            return
        if self.UM_option == "exact":
            # In that case, we check Exhaustive Ballot first.
            if not self.EB.UM_option == "exact":
                self.EB.UM_option = "exact"
            if self.EB.UM()[0] == False:
                self._mylog("UM impossible (since it is impossible for " + 
                            "Exhaustive Ballot)", 2)
                self._candidates_UM[:] = False
                # Other variables will be updated in
                # _UM_preliminary_checks_general.

    def _UM_main_work_c(self, c):
        exact = (self.UM_option == "exact")
        n_m = self.pop.matrix_duels[c, self.w]
        manip_found_fast, example_path_fast = self._UM_aux_fast(
            c, n_m,
            preferences_borda_s=self.pop.preferences_borda_vtb[
                np.logical_not(self.v_wants_to_help_c[:, c]), :])
        self._mylogv("UM: manip_found_fast =", manip_found_fast, 3)
        if manip_found_fast:
            self._candidates_UM[c] = True
            if self._example_path_UM[c] is None:
                self._example_path_UM[c] = example_path_fast
            return
        if not exact:
            self._candidates_UM[c] = np.nan
            return
            
        # From this point, we have necessarily the 'exact' option (and have
        # not found a manipulation for c yet).
        if np.isneginf(self.EB._candidates_UM[c]):
            self.EB.UM_with_candidates()
        if self.EB._candidates_UM[c] == False:
            self._mylog("UM impossible for c (since it is impossible for " +
                        "Exhaustive Ballot)", 2)
            self._candidates_UM[c] = False
            return
        manip_found_exact, example_path_exact = self._UM_aux_exact(
            c, n_m,
            preferences_borda_s=self.pop.preferences_borda_vtb[
                np.logical_not(self.v_wants_to_help_c[:, c]), :])
        self._mylogv("UM: manip_found_exact =", manip_found_exact)
        if manip_found_exact:
            self._candidates_UM[c] = True
            if self._example_path_UM[c] is None:
                self._example_path_UM[c] = example_path_exact
        else:
            self._candidates_UM[c] = False

    #%% Ignorant-Coalition Manipulation (ICM)

    # Defined in superclass Election

    #%% Coalition Manipulation (CM)

    def _CM_aux_fast(self, c, n_max, preferences_borda_s):
        """Fast algorithm used for CM.
        
        Arguments:
        c -- Integer. Candidate for which we want to manipulate.
        n_max -- Integer. Maximum number of manipulators allowed.
            CM, complete and exact --> put the current value of 
                sufficient_coalition_size[c] - 1 (we want to find the best 
                value for sufficient_coalition_size[c], even if it exceeds the 
                number of manipulators)
            CM, otherwise --> put the number of manipulators.
        preferences_borda_s -- 2d integer. Preferences of the sincere voters
            (in Borda format).
        
        Returns:
        n_manip_fast -- Integer or inf. 
            If a manipulation is found, a sufficient number of manipulators
            is returned. If no manipulation is found, it is +inf.
        example_path_fast -- An example of elimination path that realizes the
            manipulation with 'n_manip_fast' manipulators. 
            example_path_fast[k] is the k-th candidate eliminated. If no 
            manipulation is found, example_path is NaN.
        """
        # At each round, we examine all candidates d that we can eliminate.
        # If several d are possible, we choose the one that maximize
        # a heuristic parameter denoted "situation_c".
        candidates = np.array(range(self.pop.C))
        n_s = preferences_borda_s.shape[0]
        n_manip_fast = 0
        example_path_fast = []
        is_candidate_alive = np.ones(self.pop.C, dtype=np.bool)
        # Sincere scores (eliminated candidates will have nan)
        scores_s = np.sum(np.equal(
            preferences_borda_s,
            np.max(preferences_borda_s, 1)[:, np.newaxis]
        ), 0)
        # Manipulators' scores (eliminated candidates will have 0)
        scores_m = np.zeros(self.pop.C)
        # Total scores (eliminated candidates will have nan)
        scores_tot = scores_s
        for r in range(self.pop.C-1):
            self._mylogv("CM_aux_fast: Round r =", r, 3)
            self._mylogv("CM_aux_fast: scores_s =", scores_s, 3)
            self._mylogv("CM_aux_fast: scores_m =", scores_m, 3)
            self._mylogv("CM_aux_fast: scores_tot =", scores_tot, 3)
            # If an opponent has too many votes, then manipulation is
            # not possible.
            max_score = np.nanmax(scores_tot[candidates != c])
            most_serious_opponent = np.where(scores_tot == max_score)[0][0]
            if (max_score + (most_serious_opponent < c) > 
                    n_s + n_max - max_score):
                self._mylogv("CM_aux_fast: most_serious_opponent =", 
                             most_serious_opponent, 3)
                self._mylog("CM_aux_fast: Manipulation impossible by this " +
                            "path (an opponent has too many votes)", 3)
                n_manip_fast = np.inf  # by convention
                return n_manip_fast, np.nan
            # Initialize the loop on d
            best_d = -1
            best_situation_for_c = -np.inf
            n_manip_r = np.inf
            scores_s_end_r = np.nan
            scores_m_end_r = np.nan
            scores_tot_end_r = np.nan
            for d in candidates[is_candidate_alive]:
                if d == c:
                    continue
                self._mylogv("CM_aux_fast: d =", d, 3)
                # Is it possible to eliminate d now?
                scores_m_new = np.zeros(self.pop.C)
                scores_m_new[is_candidate_alive] = np.maximum(
                    0,
                    scores_tot[d] - scores_tot[is_candidate_alive] +
                    (candidates[is_candidate_alive] > d))
                self._mylogv("CM_aux_fast: scores_m_new =", scores_m_new, 3)
                scores_m_tot_d = scores_m + scores_m_new
                n_manip_d = np.sum(scores_m_tot_d)
                self._mylogv("CM_aux_fast: n_manip_d =", 
                             n_manip_d, 3)
                if n_manip_d > n_max:
                    self._mylogv("CM_aux_fast: n_manip_d > n_max =", 
                                 n_max, 3)
                    continue
                # Compute heuristic: "Situation" for c
                if r == self.pop.C - 2:
                    best_d = d
                    n_manip_r = n_manip_d
                    break
                is_candidate_alive_temp = np.copy(is_candidate_alive)
                is_candidate_alive_temp[d] = False
                scores_s_temp = np.full(self.pop.C, np.nan) 
                scores_s_temp[is_candidate_alive_temp] = np.sum(np.equal(
                    preferences_borda_s[:, is_candidate_alive_temp],
                    np.max(preferences_borda_s[
                        :, is_candidate_alive_temp], 1)[:, np.newaxis]
                ), 0)
                scores_m_temp = np.copy(scores_m_tot_d)
                scores_m_temp[d] = 0
                scores_tot_temp = scores_s_temp + scores_m_temp
                if self.fast_algo == "c_minus_max":
                    situation_for_c = (
                        scores_s_temp[c] -
                        np.nanmax(scores_tot_temp[candidates != c]))
                elif self.fast_algo == "minus_max":
                    situation_for_c = - np.nanmax(
                        scores_tot_temp[candidates != c])
                elif self.fast_algo == "hardest_first":
                    situation_for_c = n_manip_d
                else:
                    raise ValueError("Unknown fast algorithm: " + 
                                     format(self.fast_algo))
                self._mylogv("CM_aux_fast: scores_s_temp =", 
                             scores_s_temp, 3)
                self._mylogv("CM_aux_fast: scores_m_temp =", 
                             scores_m_temp, 3)
                self._mylogv("CM_aux_fast: scores_tot_temp =", 
                             scores_tot_temp, 3)
                self._mylogv("CM_aux_fast: situation_for_c =", 
                             situation_for_c, 3)
                # Is the the best d so far?
                # Lexicographic comparison on three criteria (highest 
                # 'situation', lowest number of manipulators, lowest index). 
                if ([situation_for_c, -n_manip_d, -d] >
                        [best_situation_for_c, -n_manip_r, -best_d]): 
                    best_d = d
                    best_situation_for_c = situation_for_c
                    n_manip_r = n_manip_d
                    scores_s_end_r, scores_s_temp = scores_s_temp, None
                    scores_m_end_r, scores_m_temp = scores_m_temp, None
                    scores_tot_end_r, scores_tot_temp = scores_tot_temp, None
            self._mylogv("CM_aux_fast: best_d =", best_d, 3)
            self._mylogv("CM_aux_fast: n_manip_r =", n_manip_r, 3)
            # Update variables for next round
            n_manip_fast = max(n_manip_fast, n_manip_r)
            if n_manip_fast > n_max:
                n_manip_fast = np.inf  # by convention
                break
            example_path_fast.append(best_d)
            is_candidate_alive[best_d] = False
            scores_s, scores_s_end_r = scores_s_end_r, None
            scores_m, scores_m_end_r = scores_m_end_r, None
            scores_tot, scores_tot_end_r = scores_tot_end_r, None
        self._mylogv("CM_aux_fast: Conclusion: n_manip_fast =", 
                     n_manip_fast, 3)
        if n_manip_fast <= n_max:
            example_path_fast.append(c)
            return n_manip_fast, np.array(example_path_fast)
        else:
            return n_manip_fast, np.nan

    def _CM_aux_slow(self, suggested_path, preferences_borda_s):
        """'Slow' algorithm used for CM.
        Checks only if suggested_path works.

        Arguments:
        c -- Integer. Candidate for which we want to manipulate.
        suggested_path -- A suggested path of elimination.
        preferences_borda_s -- 2d integer. Preferences of the sincere voters
            (in Borda format).

        Returns:
        n_manip_slow -- Number of manipulators needed to manipulate with
            suggested_path.
        """
        candidates = np.array(range(self.pop.C))
        is_candidate_alive = np.ones(self.pop.C, dtype=np.bool)
        # Manipulators' scores (eliminated candidates will have 0)
        scores_m = np.zeros(self.pop.C)
        n_manip_slow = 0
        for r in range(self.pop.C-1):
            # Sincere scores (eliminated candidates will have nan)
            scores_s = np.full(self.pop.C, np.nan)
            scores_s[is_candidate_alive] = np.sum(np.equal(
                preferences_borda_s[:, is_candidate_alive],
                np.max(preferences_borda_s[
                    :, is_candidate_alive], 1)[:, np.newaxis]
            ), 0)
            # Total scores (eliminated candidates will have nan)
            scores_tot = scores_s + scores_m
            self._mylogv("CM_aux_slow: Round r =", r, 3)
            self._mylogv("CM_aux_slow: scores_s =", scores_s, 3)
            self._mylogv("CM_aux_slow: scores_m =", scores_m, 3)
            self._mylogv("CM_aux_slow: scores_tot =", scores_tot, 3)
            # Let us manipulate
            d = suggested_path[r]
            scores_m_new = np.zeros(self.pop.C)
            scores_m_new[is_candidate_alive] = np.maximum(
                0,
                scores_tot[d] - scores_tot[is_candidate_alive] +
                (candidates[is_candidate_alive] > d))
            self._mylogv("CM_aux_slow: scores_m_new =", scores_m_new, 3)
            scores_m = scores_m + scores_m_new
            n_manip_r = np.sum(scores_m)
            self._mylogv("CM_aux_slow: n_manip_r =",
                         n_manip_r, 3)
            # Prepare for next round
            is_candidate_alive[d] = False
            scores_m[d] = 0
            n_manip_slow = max(n_manip_slow, n_manip_r)
        self._mylogv("CM_aux_slow: Conclusion: n_manip_slow =",
                     n_manip_slow, 3)
        return n_manip_slow

    def _CM_aux_exact(self, c, n_max, n_min, optimize_bounds, suggested_path,
                      preferences_borda_s):
        """Exact algorithm used for CM.
        
        Arguments:
        c -- Integer. Candidate for which we want to manipulate.
        n_max -- Integer. Maximum number of manipulators allowed.
            CM, optimize_bounds and exact --> put the current value of
                sufficient_coalition_size[c] - 1 (we want to find the best 
                value for sufficient_coalition_size[c], even if it exceeds the 
                number of manipulators)
            CM, otherwise --> put the number of manipulators.
        n_min -- Integer. When we know that n_min manipulators are needed
            (necessary coalition size).
        optimize_bounds -- Boolean. True iff we need to continue, even after
            a manipulation is found.
        suggested_path -- A suggested path of elimination.
        preferences_borda_s -- 2d integer. Preferences of the sincere voters
            (in Borda format).
            
        Returns:
        n_manip_final -- Integer or +inf. 
            If manipulation is impossible with <= n_max manipulators, it is 
            +inf. If manipulation is possible (with <= n_max):
            * If optimize_bounds is True, it is the minimal number of manipulators.
            * Otherwise, it is a number of manipulators that allow this 
            manipulation (not necessarily minimal). 
        example_path -- An example of elimination path that realizes the
            manipulation with 'n_manip_final' manipulators. 
            example_path[k] is the k-th candidate eliminated. If the 
            manipulation is impossible, example_path is NaN.
        quick_escape -- Boolean. True if we get out without optimizing
            n_manip_final.
        """        
        candidates = np.array(range(self.pop.C))
        n_s = preferences_borda_s.shape[0]
        n_max_updated = n_max   # Maximal number of manipulators allowed
        n_manip_final = np.inf  # Result: number of manipulators finally used
        example_path = np.nan   # Result: example of elimination path
        r = 0
        is_candidate_alive_begin_r = np.zeros((self.pop.C - 1, self.pop.C), 
                                              dtype=np.bool)
        is_candidate_alive_begin_r[0, :] = np.ones(self.pop.C)
        n_manip_used_before_r = np.zeros(self.pop.C - 1, dtype=np.int)
        scores_m_begin_r = np.zeros((self.pop.C - 1, self.pop.C))
        scores_tot_begin_r = np.zeros((self.pop.C - 1, self.pop.C))
        scores_tot_begin_r[0, :] = np.sum(np.equal(
            preferences_borda_s,
            np.max(preferences_borda_s, 1)[:, np.newaxis]
        ), 0)
        # suggested_path_r[r] is a list with all opponents (candidates != c) 
        # who are alive at the beginning of r, given in a suggested order of 
        # elimination.
        self._mylogv('CM_aux_exact: suggested_path =', suggested_path, 3)
        self._mylogv('CM_aux_exact: c =', c, 3)
        suggested_path_r = {0: suggested_path[suggested_path != c]}
        # index_in_path_r[r] is the index of the candidate we eliminate at 
        # round r in the list suggested_path_r[r].
        index_in_path_r = np.zeros(self.pop.C - 1, dtype=np.int)
        self._mylogv("CM_aux_exact: r =", r, 3)
        # If an opponent has too many votes, then manipulation is
        # not possible.
        max_score = np.nanmax(scores_tot_begin_r[0, candidates != c])
        most_serious_opponent = np.where(scores_tot_begin_r[0, :] == 
                                         max_score)[0][0]
        if (max_score + (most_serious_opponent < c) > 
                n_s + n_max_updated - max_score):
            self._mylogv("CM_aux_exact: scores_tot_begin_r =", 
                         scores_tot_begin_r[0, :], 3)
            self._mylogv("CM_aux_exact: most_serious_opponent =", 
                         most_serious_opponent, 3)
            self._mylog("CM_aux_exact: Manipulation impossible by this " +
                        "path (an opponent has too many votes)", 3)
            r = -1
        while True:
            if r < 0:
                self._mylog("CM_aux_exact: End of exploration", 3)
                return n_manip_final, np.array(example_path), False
            if (index_in_path_r[r] >= suggested_path_r[r].shape[0] or 
                    n_manip_used_before_r[r] > n_max_updated):
                # The second condition above may happen in optimize_bounds
                # exact mode, if we have found a solution and updated
                # n_max_updated.
                r -= 1
                self._mylogv("CM_aux_exact: Tried everything for round r, " + 
                             "go back to r =", r, 3)
                self._mylogv("CM_aux_exact: r =", r, 3)
                if r >= 0:
                    index_in_path_r[r] += 1
                continue
            d = suggested_path_r[r][index_in_path_r[r]]
            self._mylogv("CM_aux_exact: suggested_path_r[r] =", 
                         suggested_path_r[r], 3)
            self._mylogv("CM_aux_exact: index_in_path_r[r] =",
                         index_in_path_r[r], 3)
            self._mylogv("CM_aux_exact: d =", d, 3)
        
            # What manipulators are needed to make d lose?
            self._mylogv("CM_aux_exact: scores_tot_begin_r[r, :] =", 
                         scores_tot_begin_r[r, :], 3)
            scores_m_new_r = np.zeros(self.pop.C)
            scores_m_new_r[is_candidate_alive_begin_r[r, :]] = np.maximum(
                0,
                scores_tot_begin_r[r, d] -
                scores_tot_begin_r[r, is_candidate_alive_begin_r[r, :]] +
                (candidates[is_candidate_alive_begin_r[r, :]] > d))
            scores_m_end_r = scores_m_begin_r[r, :] + scores_m_new_r
            n_manip_r_and_before = max(n_manip_used_before_r[r],
                                       np.sum(scores_m_end_r))
            self._mylogv("CM_aux_exact: scores_m_new_r =", scores_m_new_r, 3)
            self._mylogv("CM_aux_exact: scores_m_end_r =", scores_m_end_r, 3)
            self._mylogv("CM_aux_exact: n_manip_r_and_before =",
                         n_manip_r_and_before, 3)
        
            if n_manip_r_and_before > n_max_updated:
                self._mylog("CM_aux_exact: Cannot eliminate d, try another "
                            "one.", 3)
                index_in_path_r[r] += 1
                continue
        
            if r == self.pop.C - 2:
                n_manip_final = n_manip_r_and_before
                example_path = []
                for r in range(self.pop.C - 1):
                    example_path.append(suggested_path_r[r][
                        index_in_path_r[r]])
                example_path.append(c)
                self._mylog("CM_aux_exact: CM found", 3)
                self._mylogv("CM_aux_exact: n_manip_final =", n_manip_final, 3)
                self._mylogv("CM_aux_exact: example_path =", example_path, 3)
                if n_manip_final == n_min:
                    self._mylogv("CM_aux_exact: End of exploration: it is "
                                 "not possible to do better than n_min =",
                                 n_min, 3)
                    return n_manip_final, np.array(example_path), False
                if not optimize_bounds:
                    return n_manip_final, np.array(example_path), True
                n_max_updated = n_manip_r_and_before - 1
                self._mylogv("CM_aux_exact: n_max_updated =", n_max_updated, 3)
                index_in_path_r[r] += 1
                continue
        
            # Calculate scores for next round
            n_manip_used_before_r[r+1] = n_manip_r_and_before
            is_candidate_alive_begin_r[r+1, :] = (
                is_candidate_alive_begin_r[r, :])
            is_candidate_alive_begin_r[r+1, d] = False
            self._mylogv("CM_aux_exact: is_candidate_alive_begin_r[r+1, :] =", 
                         is_candidate_alive_begin_r[r+1, :], 3)
            scores_tot_begin_r[r+1, :] = np.full(self.pop.C, np.nan) 
            scores_tot_begin_r[r+1, is_candidate_alive_begin_r[r+1, :]] = (
                np.sum(np.equal(
                    preferences_borda_s[:, is_candidate_alive_begin_r[r+1, :]],
                    np.max(preferences_borda_s[
                        :, is_candidate_alive_begin_r[r+1, :]
                    ], 1)[:, np.newaxis]
                ), 0))
            self._mylogv("CM_aux_exact: scores_s_begin_r[r+1, :] =", 
                         scores_tot_begin_r[r+1, :], 3)
            scores_m_begin_r[r+1, :] = scores_m_end_r
            scores_m_begin_r[r+1, d] = 0
            self._mylogv("CM_aux_exact: scores_m_begin_r[r+1, :] =", 
                         scores_m_begin_r[r+1, :], 3)
            scores_tot_begin_r[r+1, :] += scores_m_begin_r[r+1, :]
            self._mylogv("CM_aux_exact: scores_tot_begin_r[r+1, :] =", 
                         scores_tot_begin_r[r+1, :], 3)
        
            # If an opponent has too many votes, then manipulation is
            # not possible.
            max_score = np.nanmax(scores_tot_begin_r[r+1, candidates != c])
            most_serious_opponent = np.where(scores_tot_begin_r[r+1, :] == 
                                             max_score)[0][0]
            if (max_score + (most_serious_opponent < c) > 
                    n_s + n_max_updated - max_score):
                self._mylogv("CM_aux_exact: most_serious_opponent =", 
                             most_serious_opponent, 3)
                self._mylog("CM_aux_exact: Manipulation impossible by this " +
                            "path (an opponent will have too many votes)", 3)
                index_in_path_r[r] += 1
                continue
        
            # Update other variables for next round
            suggested_path_r[r+1] = suggested_path_r[r][
                suggested_path_r[r][:] != d]
            index_in_path_r[r+1] = 0
            r += 1
            self._mylogv("CM_aux_exact: r =", r, 3)
        
    def _CM_preliminary_checks_general_subclass(self):
        if self.CM_option == "slow" or self.CM_option == 'exact':
            # In that case, we check Exhaustive Ballot first
            if not self.EB.CM_option == "exact":
                self.EB.CM_option = "exact"
            if self.EB.CM()[0] == False:
                self._mylog("CM impossible (since it is impossible for " + 
                            "Exhaustive Ballot)", 2)
                self._is_CM = False
                self._candidates_CM[:] = False
                self._CM_was_computed_with_candidates = True

    def _CM_preliminary_checks_c(self, c, optimize_bounds):
        # A mandatory additional precheck, to ensure that
        # _example_path_CM[c] is updated if sufficient_coalition_size_CM[c]
        # has been updated.
        # We use the following syntax (instead of
        # _CM_preliminary_checks_c_subclass) because we want the test on TM
        # here to be done, even if another one succeeded.
        super()._CM_preliminary_checks_c(c, optimize_bounds)
        # As a test for sufficient size, this one is better (lower) than all
        # the other ones in _CM_preliminary_checks_c. So, as soon as one of
        # them updates sufficient size, this one will provide an example of
        # path.
        self.TM_c(c)
        if self._sufficient_coalition_size_TM[c] <= \
                self._sufficient_coalition_size_CM[c]:
            # The <= is not a typo.
            self._update_sufficient(
                self._sufficient_coalition_size_CM, c,
                self._sufficient_coalition_size_TM[c],
                'CM: Preliminary checks: TM improved => \n    '
                'sufficient_coalition_size_CM[c] = ')
            self._mylogv('CM: Preliminary checks: Update _example_path_CM[c] '
                         '= _example_path_TM[c] =',
                         self._example_path_TM[c], 3)
            self._example_path_CM[c] = self._example_path_TM[c]
        n_m = self.pop.matrix_duels[c, self.w]  # Number of manipulators
        if not optimize_bounds and (
                n_m >= self._sufficient_coalition_size_CM[c]):
            return
        if not optimize_bounds and (
                self._necessary_coalition_size_CM[c] > n_m):
            return
        # Another pretest, based on Exhaustive Ballot
        if self.CM_option == "slow" or self.CM_option == "exact":
            if not self.EB.CM_option == "exact":
                self.EB.CM_option = "exact"
            if optimize_bounds:
                self.EB.CM_c_with_bounds(c)
            else:
                # This will at least tell us when EB.CM is impossible for c,
                # and store an example path for EB when EB.CM is possible
                # for c.
                self.EB.CM_c(c)
            self._update_necessary(
                self._necessary_coalition_size_CM, c,
                self.EB._necessary_coalition_size_CM[c],
                'CM: Preliminary checks: Use EB =>\n    '
                'necessary_coalition_size_CM[c] = '
                'EB._necessary_coalition_size_CM[c] =')

    def _CM_main_work_c(self, c, optimize_bounds):
        exact = (self.CM_option == "exact")
        slow = (self.CM_option == "slow")
        fast = (self.CM_option == "fast")
        if optimize_bounds and exact:
            n_max = self._sufficient_coalition_size_CM[c] - 1
        else:
            n_max = self.pop.matrix_duels[c, self.w]
        self._mylogv("CM: n_max =", n_max, 3)
        if fast and self._necessary_coalition_size_CM[c] > n_max:
            self._mylog("CM: Fast algorithm will not do better than " +
                        "what we already know", 3)
            return False
        n_manip_fast, example_path_fast = self._CM_aux_fast(
            c, n_max,
            preferences_borda_s=self.pop.preferences_borda_vtb[
                np.logical_not(self.v_wants_to_help_c[:, c]), :])
        self._mylogv("CM: n_manip_fast =", n_manip_fast, 3)
        if n_manip_fast < self._sufficient_coalition_size_CM[c]:
            self._sufficient_coalition_size_CM[c] = n_manip_fast
            self._example_path_CM[c] = example_path_fast
            self._mylogv('CM: Update sufficient_coalition_size_CM[c] = '
                         'n_manip_fast =', n_manip_fast, 3)
        if fast:
            # With fast algo, we stop here anyway. It is not a "quick escape"
            # (if we'd try again with optimize_bounds, we would not try
            # better).
            return False
            
        # From this point, we have necessarily the 'slow' or 'exact' option
        if self._sufficient_coalition_size_CM[c] == (
                self._necessary_coalition_size_CM[c]):
            return False
        if not optimize_bounds and (self.pop.matrix_duels[c, self.w] >=
                                    self._sufficient_coalition_size_CM[c]):
            # This is a quick escape: since we have the option 'exact', if
            # we come back with optimize_bounds, we will try to be more
            # precise.
            return True

        # Either we're with optimize_bounds (and might have succeeded), or
        # without (and we have failed).
        n_max_updated = min(n_manip_fast - 1, n_max)
        self._mylogv("CM: n_max_updated =", n_max_updated, 3)

        # EB should always suggest an elimination path.
        # But just as precaution, we use the fact that we know that
        # self._example_path_CM[c] provides a path (thanks to 'improved' TM).
        if self.EB._example_path_CM[c] is None:
            suggested_path = self._example_path_CM[c]
            self._mylog(
                '***************************************************', 1)
            self._mylog('CM: WARNING: EB did not provide an elimination '
                        'path', 1)
            self._mylog(
                '***************************************************', 1)
            self._mylogv('CM: Use self._example_path_CM[c] =',
                         suggested_path, 3)
        else:
            suggested_path = self.EB._example_path_CM[c]
            self._mylogv('CM: Use EB._example_path_CM[c] =',
                         suggested_path, 3)
        if slow:
            n_manip_slow = self._CM_aux_slow(
                suggested_path,
                preferences_borda_s=self.pop.preferences_borda_vtb[
                    np.logical_not(self.v_wants_to_help_c[:, c]), :])
            self._mylogv("CM: n_manip_slow =", n_manip_slow, 3)
            if n_manip_slow < self._sufficient_coalition_size_CM[c]:
                self._sufficient_coalition_size_CM[c] = n_manip_slow
                self._example_path_CM[c] = suggested_path
                self._mylogv('CM: Update sufficient_coalition_size_CM[c] = '
                             'n_manip_slow =')
            return False
        else:
            n_manip_exact, example_path_exact, quick_escape = \
                self._CM_aux_exact(
                    c, n_max_updated, self._necessary_coalition_size_CM[c],
                    optimize_bounds, suggested_path,
                    preferences_borda_s=self.pop.preferences_borda_vtb[
                        np.logical_not(self.v_wants_to_help_c[:, c]), :])
            self._mylogv("CM: n_manip_exact =", n_manip_exact, 3)
            if n_manip_exact < self._sufficient_coalition_size_CM[c]:
                self._sufficient_coalition_size_CM[c] = n_manip_exact
                self._example_path_CM[c] = example_path_exact
                self._mylogv('CM: Update sufficient_coalition_size_CM[c] = '
                             'n_manip_exact =')
            # Update necessary coalition and return
            if optimize_bounds:
                self._update_necessary(
                    self._necessary_coalition_size_CM, c,
                    self._sufficient_coalition_size_CM[c],
                    'CM: Update necessary_coalition_size_CM[c] ='
                    'sufficient_coalition_size_CM[c] =')
                return False
            else:
                if self.pop.matrix_duels[c, self.w] >= \
                        self._sufficient_coalition_size_CM[c]:
                    # Manipulation worked. By design of _CM_aux_exact when
                    # running without optimize_bounds, we have not explored
                    # everything (quick escape).
                    return True
                else:
                    # We have explored everything with n_max = n_m but
                    # manipulation failed. However, we have not optimized
                    # sufficient_size (which must be higher than n_m), so it
                    # is a quick escape.
                    self._update_necessary(
                        self._necessary_coalition_size_CM, c,
                        self.pop.matrix_duels[c, self.w] + 1,
                        'CM: Update necessary_coalition_size_CM[c] = '
                        'n_m + 1 =')
                    return True


if __name__ == '__main__':
    # A quick demo
    preferences_utilities = np.random.randint(-5, 5, (10, 5))
    pop = Population(preferences_utilities)
    election = IRV(pop)
    election.demo(log_depth=3)