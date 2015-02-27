# -*- coding: utf-8 -*-
"""
Created on Thu Oct  9 10:15:32 2014
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
from svvamp.VotingSystems.ExhaustiveBallotResult import ExhaustiveBallotResult


class ExhaustiveBallot(ExhaustiveBallotResult, Election):
    """Exhaustive Ballot.
    
    Inherits functions and optional parameters from superclasses
    :class:`~svvamp.ElectionResult` and :class:`~svvamp.Election`.

    :Example:

    >>> import svvamp
    >>> pop = svvamp.PopulationSpheroid(V=100, C=5)
    >>> election = svvamp.ExhaustiveBallot(pop)

    At each round, voters vote for one non-eliminated candidate. The candidate
    with least votes is eliminated. Then the next round is held. Unlike
    :attr:`~svvamp.IRV`, voters actually vote at each round. This does not
    change anything for sincere voting, but offers a bit more possibilities
    for the manipulators. In case of a tie, the candidate with highest index
    is eliminated.

    :meth:`~svvamp.Election.CM`:

        * :attr:`~svvamp.Election.CM_option` = ``'fast'``:
          Polynomial heuristic. Can prove CM but unable to decide non-CM
          (except in rare obvious cases).
        * :attr:`~svvamp.Election.CM_option` = ``'exact'``:
          Non-polynomial algorithm (:math:`2^C`) adapted from Walsh, 2010.

    :meth:`~svvamp.Election.ICM`: Exact in polynomial time.

    :meth:`~svvamp.Election.IM`:

        * :attr:`~svvamp.Election.IM_option` = ``'lazy'``:
          Lazy algorithm from superclass :class:`~svvamp.Election`.
        * :attr:`~svvamp.Election.IM_option` = ``'exact'``:
          Non-polynomial algorithm (:math:`2^C`) adapted from Walsh, 2010.

    :meth:`~svvamp.Election.not_IIA`: Non-polynomial
    or non-exact algorithms from superclass :class:`~svvamp.Election`.

    :meth:`~svvamp.Election.TM`: Exact in polynomial time.

    :meth:`~svvamp.Election.UM`:

        * :attr:`~svvamp.Election.UM_option` = ``'fast'``:
          Polynomial heuristic. Can prove UM but unable to decide non-UM
          (except in rare obvious cases).
        * :attr:`~svvamp.Election.UM_option` = ``'exact'``:
          Non-polynomial algorithm (:math:`2^C`) adapted from Walsh, 2010.

    References:

        'Single transferable vote resists strategic voting', John J. Bartholdi
        and James B. Orlin, 1991.

        'On The Complexity of Manipulating Elections', Tom Coleman and Vanessa
        Teague, 2007.

        'Manipulability of Single Transferable Vote', Toby Walsh, 2010.

    .. seealso:: :class:`~svvamp.IRV`,
                 :class:`~svvamp.IRVDuels`,
                 :class:`~svvamp.ICRV`,
                 :class:`~svvamp.CondorcetAbsIRV`.
                 :class:`~svvamp.CondorcetVtbIRV`.
    """
    # Exceptionally, for this voting system, results are stored in the 
    # Population object, so that they can be used by IRV.

    _layout_name = 'Exhaustive Ballot'
    _options_parameters = Election._options_parameters.copy()
    _options_parameters.update(ExhaustiveBallotResult._options_parameters)
    _options_parameters['UM_option'] = {'allowed': {'fast', 'exact'},
                                        'default': 'fast'}
    _options_parameters['CM_option'] = {'allowed': {'fast', 'exact'},
                                        'default': 'fast'}
    _options_parameters['TM_option'] = {'allowed': {'exact'},
                                        'default': 'exact'}
    _options_parameters['ICM_option'] = {'allowed': {'exact'},
                                           'default': 'exact'}
    _options_parameters['fast_algo'] = {
        'allowed': {'c_minus_max', 'minus_max', 'hardest_first'},
        'default': 'c_minus_max'
    }

    def __init__(self, population, freeze_options=False, **kwargs):
        self.freeze_options = freeze_options
        super().__init__(population, **kwargs)
        self.freeze_options = False
        self._log_depth = 0
        self._log_identity = "EXHAUSTIVE_BALLOT"
        self._class_result = ExhaustiveBallotResult
        self._with_two_candidates_reduces_to_plurality = True
        self._is_based_on_rk = True
        self._meets_majority_favorite_c_rk_ctb = True
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

    def _initialize_options(self, **kwargs):
        if self.pop._eb_options is None:
            self.pop._eb_options = _Storage()
            Election._initialize_options(self, **kwargs)
        elif not self.freeze_options:
            Election._initialize_options(self, **kwargs)

    #%% Store / forget the examples of manipulation path

    # Not necessary for the moment: IM does not store an elimination path
    # def _forget_IM(self):
    #     self._IM_was_initialized = False
    #     self._IM_was_computed_with_candidates = False
    #     self._IM_was_computed_with_voters = False
    #     self._IM_was_computed_full = False
    #     self._example_path_IM = {c: None for c in range(self.pop.C)}

    def _forget_TM(self):
        self._TM_was_initialized = False
        self._TM_was_computed_with_candidates = False
        self._example_path_TM = {c: None for c in range(self.pop.C)}
        self._sufficient_coalition_size_TM = np.full(self.pop.C, np.inf)
        # This is a specificity for this voting system (cf. TM section).

    def _forget_UM(self):
        self._UM_was_initialized = False
        self._UM_was_computed_with_candidates = False
        self._example_path_UM = {c: None for c in range(self.pop.C)}

    # Not necessary for the moment: ICM does not store an elimination path
    # (it would be the same as TM)
    # def _forget_ICM(self):
    #     self._ICM_was_initialized = False
    #     self._ICM_was_computed_with_candidates = False
    #     self._ICM_was_computed_full = False
    #     self._example_path_ICM = {c: None for c in range(self.pop.C)}

    def _forget_CM(self):
        self._CM_was_initialized = False
        self._CM_was_computed_with_candidates = False
        self._CM_was_computed_full = False
        self._example_path_CM = {c: None for c in range(self.pop.C)}

    #%% Cache variables in the Population object, so that they can be used
    #   by IRV.

    def _forget_manipulations(self):
        # This is called when a new ExhaustiveBallot object is created or
        # when a new population is assigned to an existing 
        # ExhaustiveBallot object.
        # If the population already stores the manipulation results for EB,
        # we must not forget them.
        if self.pop._eb_manip is None:
            self.pop._eb_manip = _Storage()
            Election._forget_manipulations(self)

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
    def _IM_was_initialized(self):
        return self.pop._eb_manip._IM_was_initialized

    @_IM_was_initialized.setter
    def _IM_was_initialized(self, value):
        self.pop._eb_manip._IM_was_initialized = value

    @property
    def _IM_was_computed_with_candidates(self):
        return self.pop._eb_manip._IM_was_computed_with_candidates

    @_IM_was_computed_with_candidates.setter
    def _IM_was_computed_with_candidates(self, value):
        self.pop._eb_manip._IM_was_computed_with_candidates = value

    @property
    def _IM_was_computed_with_voters(self):
        return self.pop._eb_manip._IM_was_computed_with_voters

    @_IM_was_computed_with_voters.setter
    def _IM_was_computed_with_voters(self, value):
        self.pop._eb_manip._IM_was_computed_with_voters = value

    @property
    def _IM_was_computed_full(self):
        return self.pop._eb_manip._IM_was_computed_full

    @_IM_was_computed_full.setter
    def _IM_was_computed_full(self, value):
        self.pop._eb_manip._IM_was_computed_full = value

    @property
    def _v_IM_for_c(self):
        return self.pop._eb_manip._v_IM_for_c

    @_v_IM_for_c.setter
    def _v_IM_for_c(self, value):
        self.pop._eb_manip._v_IM_for_c = value

    @property
    def _candidates_IM(self):
        return self.pop._eb_manip._candidates_IM

    @_candidates_IM.setter
    def _candidates_IM(self, value):
        self.pop._eb_manip._candidates_IM = value

    @property
    def _is_IM(self):
        return self.pop._eb_manip._is_IM

    @_is_IM.setter
    def _is_IM(self, value):
        self.pop._eb_manip._is_IM = value
#    @property
#    def _example_path_IM(self):
#        return self.pop._eb_manip._example_path_IM
#    @_example_path_IM.setter
#    def _example_path_IM(self, value):
#        self.pop._eb_manip._example_path_IM = value
            
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
    def _UM_was_initialized(self):
        return self.pop._eb_manip._UM_was_initialized

    @_UM_was_initialized.setter
    def _UM_was_initialized(self, value):
        self.pop._eb_manip._UM_was_initialized = value

    @property
    def _UM_was_computed_with_candidates(self):
        return self.pop._eb_manip._UM_was_computed_with_candidates

    @_UM_was_computed_with_candidates.setter
    def _UM_was_computed_with_candidates(self, value):
        self.pop._eb_manip._UM_was_computed_with_candidates = value

    @property
    def _candidates_UM(self):
        return self.pop._eb_manip._candidates_UM

    @_candidates_UM.setter
    def _candidates_UM(self, value):
        self.pop._eb_manip._candidates_UM = value

    @property
    def _is_UM(self):
        return self.pop._eb_manip._is_UM

    @_is_UM.setter
    def _is_UM(self, value):
        self.pop._eb_manip._is_UM = value

    @property
    def _example_path_UM(self):
        return self.pop._eb_manip._example_path_UM

    @_example_path_UM.setter
    def _example_path_UM(self, value):
        self.pop._eb_manip._example_path_UM = value

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

    # @property
    # def _example_path_ICM(self):
    #    return self.pop._eb_manip._example_path_ICM
    #
    # @_example_path_ICM.setter
    # def _example_path_ICM(self, value):
    #    self.pop._eb_manip._example_path_ICM = value

    @property
    def _CM_was_initialized(self):
        return self.pop._eb_manip._CM_was_initialized

    @_CM_was_initialized.setter
    def _CM_was_initialized(self, value):
        self.pop._eb_manip._CM_was_initialized = value

    @property
    def _CM_was_computed_with_candidates(self):
        return self.pop._eb_manip._CM_was_computed_with_candidates

    @_CM_was_computed_with_candidates.setter
    def _CM_was_computed_with_candidates(self, value):
        self.pop._eb_manip._CM_was_computed_with_candidates = value

    @property
    def _CM_was_computed_full(self):
        return self.pop._eb_manip._CM_was_computed_full

    @_CM_was_computed_full.setter
    def _CM_was_computed_full(self, value):
        self.pop._eb_manip._CM_was_computed_full = value

    @property
    def _candidates_CM(self):
        return self.pop._eb_manip._candidates_CM

    @_candidates_CM.setter
    def _candidates_CM(self, value):
        self.pop._eb_manip._candidates_CM = value

    @property
    def _sufficient_coalition_size_CM(self):
        return self.pop._eb_manip._sufficient_coalition_size_CM

    @_sufficient_coalition_size_CM.setter
    def _sufficient_coalition_size_CM(self, value):
        self.pop._eb_manip._sufficient_coalition_size_CM = value

    @property
    def _necessary_coalition_size_CM(self):
        return self.pop._eb_manip._necessary_coalition_size_CM

    @_necessary_coalition_size_CM.setter
    def _necessary_coalition_size_CM(self, value):
        self.pop._eb_manip._necessary_coalition_size_CM = value

    @property
    def _bounds_optimized_CM(self):
        return self.pop._eb_manip._bounds_optimized_CM

    @_bounds_optimized_CM.setter
    def _bounds_optimized_CM(self, value):
        self.pop._eb_manip._bounds_optimized_CM = value

    @property
    def _is_CM(self):
        return self.pop._eb_manip._is_CM

    @_is_CM.setter
    def _is_CM(self, value):
        self.pop._eb_manip._is_CM = value

    @property
    def _example_path_CM(self):
        return self.pop._eb_manip._example_path_CM

    @_example_path_CM.setter
    def _example_path_CM(self, value):
        self.pop._eb_manip._example_path_CM = value

    @property
    def _IIA_subset_maximum_size(self):
        return self.pop._eb_options._IIA_subset_maximum_size

    @_IIA_subset_maximum_size.setter
    def _IIA_subset_maximum_size(self, value):
        self.pop._eb_options._IIA_subset_maximum_size = value

    @property
    def _IM_option(self):
        return self.pop._eb_options._IM_option

    @_IM_option.setter
    def _IM_option(self, value):
        self.pop._eb_options._IM_option = value

    @property
    def _TM_option(self):
        return self.pop._eb_options._TM_option

    @_TM_option.setter
    def _TM_option(self, value):
        self.pop._eb_options._TM_option = value

    @property
    def _UM_option(self):
        return self.pop._eb_options._UM_option

    @_UM_option.setter
    def _UM_option(self, value):
        self.pop._eb_options._UM_option = value

    @property
    def _ICM_option(self):
        return self.pop._eb_options._ICM_option

    @_ICM_option.setter
    def _ICM_option(self, value):
        self.pop._eb_options._ICM_option = value

    @property
    def _CM_option(self):
        return self.pop._eb_options._CM_option

    @_CM_option.setter
    def _CM_option(self, value):
        self.pop._eb_options._CM_option = value

    @property
    def _fast_algo(self):
        return self.pop._eb_options._fast_algo

    @_fast_algo.setter
    def _fast_algo(self, value):
        self.pop._eb_options._fast_algo = value

    #%% Independence of Irrelevant Alternatives (IIA)

    # TODO: A faster algorithm that the one implemented in superclass Election.
    # Simply check that for each subset of candidates including w, w cannot
    # be a Plurality loser.

    #%% Individual manipulation (IM)            

    def _IM_aux(self, anti_voter_allowed, preferences_borda_s):
        """
        Arguments:
        anti_voter_allowed -- Boolean. If True, we manipulate with one voter
            but also an anti-voter (who may decrease one candidate's score
            by 1 point at each round). If False, we manipulate only with one 
            voter.
        preferences_borda_s -- 2d integer. Preferences of the sincere voters
            (in Borda format with vtb).
            
        Returns:
        candidates_IM_aux -- 1d array of booleans. candidates_IM_aux[c] is 
            True if manipulation for c is possible (whether desired or not).
        """
        # Explore subsets that are reachable.
        # situations_begin_r is a dictionary.
        # keys: is_candidate_alive_begin_r, tuple of booleans. 
        # values: just a None for the moment. If we want to store an 
        # elimination path in a future version, it will be here.
        candidates = np.array(range(self.pop.C))
        n_s = preferences_borda_s.shape[0]
        situations_begin_r = {
            tuple(np.ones(self.pop.C, dtype=np.bool)): (0, [])}
        situations_end_r = {}
        for r in range(self.pop.C - 1):
            self._mylogv("IM_aux: Round r =", r, 3)
            situations_end_r = {}
            for is_candidate_alive_begin_r, _ in (situations_begin_r.items()):
                self._mylogv("IM_aux: is_candidate_alive_begin_r =", 
                             is_candidate_alive_begin_r, 3)
                # Sincere ballots
                is_candidate_alive_begin_r = np.array(
                    is_candidate_alive_begin_r)
                scores_s = np.full(self.pop.C, np.nan)
                scores_s[is_candidate_alive_begin_r] = np.sum(np.equal(
                    preferences_borda_s[:, is_candidate_alive_begin_r],
                    np.max(preferences_borda_s[:, is_candidate_alive_begin_r],
                           1)[:, np.newaxis]
                ), 0)
                self._mylogv("IM_aux: scores_s =", scores_s, 3)
                # If w has too many votes, then manipulation is
                # not possible.
                if (scores_s[self.w] + (self.w == 0) - anti_voter_allowed > 
                        n_s - scores_s[self.w] + 1):
                    self._mylog("IM_aux: Manipulation impossible by this " +
                                "path (w has too many votes)", 3)
                    continue
                # Try to limit the possible d's
                natural_loser = np.where(
                    scores_s == np.nanmin(scores_s))[0][-1]
                self._mylogv("IM_aux: natural_loser =", natural_loser, 3)
                if not anti_voter_allowed:
                    # v can change the result of the round by only one 
                    # strategy: vote for natural_loser. Does it lead to 
                    # another candidate losing?
                    scores_temp = np.copy(scores_s)
                    scores_temp[natural_loser] += 1
                    other_possible_loser = np.where(
                        scores_temp == np.nanmin(scores_temp))[0][-1]
                    self._mylogv("IM_aux: other_possible_loser =", 
                                 other_possible_loser, 3)
                    if other_possible_loser == natural_loser:
                        losers_to_test = [natural_loser]
                    else:
                        losers_to_test = [natural_loser, other_possible_loser]
                else:
                    # With an anti-voter, who can be eliminated?
                    # The candidate with margin 0 (natural loser).
                    # Candidates with margin 1: -1 vote is enough (definition
                    # of the margin).
                    # Candidates with margin 2: it is necessary to remove a
                    # vote to her, and add a vote to natural_loser. But it 
                    # might not be sufficient, so we need to check if it works.
                    margins_s = scores_s - scores_s[natural_loser] + (
                        candidates < natural_loser)
                    self._mylogv("IM_aux: margins_s =", margins_s, 3)
                    is_eliminable = np.equal(margins_s, 1)
                    is_eliminable[natural_loser] = True
                    for d in candidates[margins_s == 2]:
                        scores_temp = np.copy(scores_s)
                        scores_temp[d] -= 1
                        scores_temp[natural_loser] += 1
                        self._mylogv("IM_aux: scores_temp =", scores_temp, 3)
                        loser_temp = np.where(
                            scores_temp == np.nanmin(scores_temp))[0][-1]
                        self._mylogv("IM_aux: loser_temp =", loser_temp, 3)
                        if loser_temp == d:
                            is_eliminable[d] = True
                    losers_to_test = candidates[is_eliminable]
                self._mylogv("IM_aux: losers_to_test = ", losers_to_test, 3)
                # Loop on d
                for d in losers_to_test:
                    # At this point, we know that we can eliminate d.
                    # Feed the dictionary 'situations_end_r'
                    is_candidate_alive_end_r = np.copy(
                        is_candidate_alive_begin_r)
                    is_candidate_alive_end_r[d] = False
                    if tuple(is_candidate_alive_end_r) not in situations_end_r:
                        situations_end_r[
                            tuple(is_candidate_alive_end_r)] = None
            self._mylogv("IM_aux: situations_end_r =", situations_end_r, 3)
            situations_begin_r = situations_end_r
        candidates_IM_aux = np.zeros(self.pop.C)
        for is_candidate_alive_end, foobar in situations_end_r.items():
            candidates_IM_aux = np.logical_or(candidates_IM_aux, 
                                              is_candidate_alive_end)
        self._mylogv("IM_aux: candidates_IM_aux =", candidates_IM_aux, 3)
        return candidates_IM_aux

    def _IM_preliminary_checks_general_subclass(self):
        if self.IM_option != "exact":
            return
        if np.all(np.logical_not(np.isneginf(self._v_IM_for_c))):
            return
        self._mylog("IM: Test with a voter and an anti-voter...")
        # If it's impossible with a voter and an anti-voter on the
        # whole population, then we'll know that IM is impossible.
        candidates_IM_aux = self._IM_aux(
            anti_voter_allowed=True,
            preferences_borda_s=self.pop.preferences_borda_rk)
        candidates_IM_aux[self.w] = False
        for c in self.losing_candidates:
            if candidates_IM_aux[c] == False:
                self._mylogv("IM: Manipulation with voter and anti-voter " +
                             "failed for c =", c, 2)
                self._v_IM_for_c[:, c] = False

    def _IM_main_work_v(self, v, c_is_wanted,
                        nb_wanted_undecided, stop_if_true):
        if self.IM_option == "lazy":
            self._IM_main_work_v_lazy(v, c_is_wanted,
                                      nb_wanted_undecided, stop_if_true)
            return
        candidates_IM_aux = self._IM_aux(
            anti_voter_allowed=False,
            preferences_borda_s=self.pop.preferences_borda_rk[
                np.array(range(self.pop.V)) != v, :])
        for c in self.losing_candidates:
            if not np.isneginf(self._v_IM_for_c[v, c]):
                # v is not interested, or we already know for some reason
                continue
            if candidates_IM_aux[c]:
                self._v_IM_for_c[v, c] = True
                self._candidates_IM[c] = True
                self._voters_IM[v] = True
                self._is_IM = True
                self._mylogv("IM found for c =", c, 3)
            else:
                self._v_IM_for_c[v, c] = False

    #%% Collective manipulations (CM / UM): general routine

    def _CM_aux_fast(self, c, n_max, unison, preferences_borda_s):
        """Fast algorithm used for CM and UM.
        
        Arguments:
        c -- Integer. Candidate for which we want to manipulate.
        n_max -- Integer. Maximum number of manipulators allowed.
            UM --> put the number of manipulators.
            CM, with candidates and exact --> put the current value of
                sufficient_coalition_size[c] - 1 (we want to find the best 
                value for sufficient_coalition_size[c], even if it exceeds the 
                number of manipulators)
            CM, otherwise --> put the number of manipulators.
        unison -- Boolean. Must be True when computing UM, False for CM.
        preferences_borda_s -- 2d integer. Preferences of the sincere voters
            (in Borda format with vtb).
        
        Returns:
        n_manip_fast -- Integer or +inf. 
            If a manipulation is found, a sufficient number of manipulators
            is returned (if unison is True, this number will always be n_max).
            If no manipulation is found, it is +inf.
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
        n_manip_fast = 0  # Number of manipulators
        example_path_fast = []
        is_candidate_alive = np.ones(self.pop.C, dtype=np.bool)
        # Sincere scores (eliminated candidates will have nan)
        scores_s = np.sum(np.equal(
            preferences_borda_s,
            np.max(preferences_borda_s, 1)[:, np.newaxis]
        ), 0)
        for r in range(self.pop.C-1):
            self._mylogv("CM_aux_fast: Round r =", r, 3)
            self._mylogv("CM_aux_fast: scores_s =", scores_s, 3)
            # If an opponent has too many votes, then manipulation is
            # not possible.
            max_score = np.nanmax(scores_s[candidates != c])
            most_serious_opponent = np.where(scores_s == max_score)[0][0]
            if (max_score + (most_serious_opponent < c) > 
                    n_s + n_max - max_score):
                self._mylogv("CM_aux_fast: most_serious_opponent =", 
                             most_serious_opponent, 3)
                self._mylog("CM_aux_fast: Manipulation impossible by this " +
                            "path (an opponent has too many votes)", 3)
                n_manip_fast = np.inf  # by convention
                return n_manip_fast, np.nan
            # Try to limit the possible d's (for unison)
            if unison:
                natural_loser = np.where(
                    scores_s == np.nanmin(scores_s))[0][-1]
                self._mylogv("CM_aux_fast: natural_loser =", natural_loser, 3)
                # In unison, if we want to change the result of the round,
                # the only thing we can do is to vote for the natural loser.
                scores_temp = np.copy(scores_s)
                scores_temp[natural_loser] += n_max
                other_possible_loser = np.where(
                    scores_temp == np.nanmin(scores_temp))[0][-1]
                self._mylogv("CM_aux_fast: other_possible_loser =", 
                             other_possible_loser, 3)
                if other_possible_loser == natural_loser:
                    losers_to_test = [natural_loser]
                else:
                    losers_to_test = [natural_loser, other_possible_loser]
            else:
                losers_to_test = candidates[is_candidate_alive]
            self._mylogv("CM_aux_fast: losers_to_test =", losers_to_test, 3)
            self._mylogv("CM_aux_fast: but do not test ", c, 3)
            # Initialize the loop on d
            best_d = -1
            best_situation_for_c = -np.inf
            n_manip_r = np.inf
            scores_s_end_r = np.nan
            for d in losers_to_test:
                if d == c:
                    continue
                self._mylogv("CM_aux_fast: d =", d, 3)
                # Is it possible to eliminate d now?
                if unison:
                    # We already know that it is possible.
                    n_manip_d = n_max
                else:
                    scores_m = np.maximum(
                        0,
                        scores_s[d] - scores_s[is_candidate_alive] +
                        (candidates[is_candidate_alive] > d))
                    n_manip_d = np.sum(scores_m)
                    self._mylogv("CM_aux_fast: n_manip_d =", 
                                 n_manip_d, 3)
                    if n_manip_d > n_max:
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
                    np.max(preferences_borda_s[:, is_candidate_alive_temp], 1)[
                        :, np.newaxis]
                ), 0)
                if self.fast_algo == "c_minus_max":
                    situation_for_c = (scores_s_temp[c] - np.nanmax(
                        scores_s_temp[candidates != c]))
                elif self.fast_algo == "minus_max":
                    situation_for_c = - np.nanmax(
                        scores_s_temp[candidates != c])
                elif self.fast_algo == "hardest_first":
                    situation_for_c = n_manip_d
                else:
                    raise ValueError("Unknown fast algorithm: " + 
                                     format(self.fast_algo))
                self._mylogv("CM_aux_fast: scores_s_temp =", 
                             scores_s_temp, 3)
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
        self._mylogv("CM_aux_fast: : n_manip_fast =", 
                     n_manip_fast, 3)
        if n_manip_fast <= n_max:
            example_path_fast.append(c)
            return n_manip_fast, np.array(example_path_fast)
        else:
            return n_manip_fast, np.nan
                     
    def _CM_aux_exact(self, c, n_max, unison, preferences_borda_s):
        """Exact algorithm used for CM and UM.
        
        Arguments:
        c -- Integer. Candidate for which we want to manipulate.
        n_max -- Integer. Maximum number of manipulators allowed.
            UM --> put the number of manipulators.
            CM, with candidates and exact --> put the current value of
                sufficient_coalition_size[c] - 1 (we want to find the best 
                value for sufficient_coalition_size[c], even if it exceeds the 
                number of manipulators)
            CM, otherwise --> put the number of manipulators.
        unison -- Boolean. Must be True when computing UM, False for CM.
        preferences_borda_s -- 2d integer. Preferences of the sincere voters
            (in Borda format).
            
        Returns:
        n_manip_final -- Integer or +inf. 
            If manipulation is impossible with <= n_max manipulators, it is 
            +inf. If manipulation is possible (with <= n_max):
            * If unison is False, it is the minimal number of manipulators.
            * If unison is True, it is n_max.
        example_path -- An example of elimination path that realizes the
            manipulation with 'n_manip_final' manipulators. 
            example_path[k] is the k-th candidate eliminated. If the 
            manipulation is impossible, example_path is NaN.
        """        
        # Explore subsets that are reachable with less than the upper bound.
        # situations_begin_r is a dictionary.
        # keys: is_candidate_alive_begin_r, tuple of booleans. 
        # values: (n_manip_used_before_r, example_path_before_r)
        #   n_manip_used_before_r: number of manipulators used before round r
        #   to get to this subset of candidates. 
        #   example_path_before_r: as it sounds.
        candidates = np.array(range(self.pop.C))
        n_s = preferences_borda_s.shape[0]
        situations_begin_r = {
            tuple(np.ones(self.pop.C, dtype=np.bool)): (0, [])}
        situations_end_r = {}
        for r in range(self.pop.C - 1):
            self._mylogv("CM_aux_exact: Round r =", r, 3)
            situations_end_r = {}
            for is_candidate_alive_begin_r, (
                    n_manip_used_before_r, example_path_before_r) in (
                        situations_begin_r.items()):
                self._mylogv("CM_aux_exact: is_candidate_alive_begin_r =", 
                             is_candidate_alive_begin_r, 3)
                self._mylogv("CM_aux_exact: n_manip_used_before_r =", 
                             n_manip_used_before_r, 3)
                self._mylogv("CM_aux_exact: example_path_before_r =", 
                             example_path_before_r, 3)
                # Sincere ballots
                is_candidate_alive_begin_r = np.array(
                    is_candidate_alive_begin_r)
                scores_s = np.full(self.pop.C, np.nan) 
                scores_s[is_candidate_alive_begin_r] = np.sum(np.equal(
                    preferences_borda_s[:, is_candidate_alive_begin_r],
                    np.max(
                        preferences_borda_s[:, is_candidate_alive_begin_r],
                        1)[:, np.newaxis]
                ), 0)
                self._mylogv("CM_aux_exact: scores_s =", scores_s, 3)
                # If an opponent has too many votes, then manipulation is
                # not possible.
                max_score = np.nanmax(scores_s[candidates != c])
                most_serious_opponent = np.where(scores_s == max_score)[0][0]
                if (max_score + (most_serious_opponent < c) > 
                        n_s + n_max - max_score):
                    self._mylogv("CM_aux_exact: most_serious_opponent =", 
                                 most_serious_opponent, 3)
                    self._mylog("CM_aux_exact: Manipulation impossible by "
                                "this path (an opponent has too many votes)",
                                3)
                    continue
                # Try to limit the possible d's
                if unison:
                    natural_loser = np.where(
                        scores_s == np.nanmin(scores_s))[0][-1]
                    self._mylogv("CM_aux_exact: natural_loser =",
                                 natural_loser, 3)
                    scores_temp = np.copy(scores_s)
                    scores_temp[natural_loser] += n_max
                    other_possible_loser = np.where(
                        scores_temp == np.nanmin(scores_temp))[0][-1]
                    self._mylogv("CM_aux_exact: other_possible_loser =", 
                                 other_possible_loser, 3)
                    if other_possible_loser == natural_loser:
                        losers_to_test = [natural_loser]
                    else:
                        losers_to_test = [natural_loser, other_possible_loser]
                else:
                    losers_to_test = candidates[is_candidate_alive_begin_r]
                self._mylogv("CM_aux_exact: losers_to_test =",
                             losers_to_test, 3)
                self._mylogv("CM_aux_exact: but do not test ", c, 3)
                # Loop on d
                for d in losers_to_test:
                    if d == c:
                        continue
                    self._mylogv("CM_aux_exact: d =", d, 3)
                    # Is it possible to eliminate d now?
                    if unison:
                        # We already know that it is possible.
                        n_manip_r = n_max
                    else:
                        scores_m = np.maximum(
                            0,
                            scores_s[d] -
                            scores_s[is_candidate_alive_begin_r] +
                            (candidates[is_candidate_alive_begin_r] > d))
                        n_manip_r = np.sum(scores_m)
                    self._mylogv("CM_aux_exact: n_manip_r =", n_manip_r, 3)
                    if n_manip_r > n_max:
                        continue
                    # Feed the dictionary 'situations_end_r'
                    n_manip_r_and_before = max(n_manip_r, 
                                               n_manip_used_before_r)
                    is_candidate_alive_end_r = np.copy(
                        is_candidate_alive_begin_r)
                    is_candidate_alive_end_r[d] = False
                    example_path_end_r = example_path_before_r[:]
                    example_path_end_r.append(d)
                    if tuple(is_candidate_alive_end_r) in situations_end_r:
                        if n_manip_r_and_before < situations_end_r[
                                tuple(is_candidate_alive_end_r)][0]:
                            situations_end_r[tuple(is_candidate_alive_end_r)] \
                                = (n_manip_r_and_before, example_path_end_r)
                    else:
                        situations_end_r[tuple(is_candidate_alive_end_r)] \
                            = (n_manip_r_and_before, example_path_end_r)
            self._mylogv("CM_aux_exact: situations_end_r =",
                         situations_end_r, 3)
            if len(situations_end_r) == 0:
                self._mylog("CM_aux_exact: Manipulation is impossible with " + 
                            "n_max manipulators.", 3)
                return np.inf, np.nan
            situations_begin_r = situations_end_r
        # If we reach this point, we know that all rounds were successful.
        self._mylogv("CM_aux_exact: situations_end_r:", situations_end_r, 3)
        is_candidate_alive_end, (
            n_manip_exact, example_path_exact) = situations_end_r.popitem()
        example_path_exact.append(c)
        self._mylogv("CM_aux_exact: is_candidate_alive_end:", 
                     is_candidate_alive_end, 3)
        self._mylogv("CM_aux_exact: example_path_exact:", 
                     example_path_exact, 3)
        self._mylogv("CM_aux_exact: Conclusion phase 2: n_manip_exact =", 
                     n_manip_exact, 3)
        return n_manip_exact, np.array(example_path_exact)

    #%% Trivial Manipulation (TM)

    def _TM_initialize_general(self, with_candidates):
        super()._TM_initialize_general(with_candidates)
        self._sufficient_coalition_size_TM = np.full(self.pop.C, np.inf)
        self._sufficient_coalition_size_TM[self.w] = 0
        self._candidates_TM[self.w] = False

    def _TM_preliminary_checks_general(self):
        # We remove the general preliminary checks, because we always want
        # to run _TM_main_work_c to provide an example of path.
        pass

    def _TM_initialize_c(self, c):
        self._mylogv("TM: Candidate =", c, 2)
        # We remove the general preliminary checks on c for the same reason.

    def _TM_main_work_c(self, c):
        # For Exhaustive Ballot, if TM works, then manipulators always vote
        # for c, so the rest of their ballot has no impact on the result.
        # So we can define a minimum coalition size for TM: minimal number
        # of manipulators such that, when always voting for c, c gets elected.
        # It will help us for CM: indeed, it is a lower bound that is better
        #  than TM (= n_m, when it works) and also than ICM (= n_s or n_s
        # + 1).
        candidates_not_c = np.concatenate((
            range(c), range(c + 1, self.pop.C))).astype(int)
        example_path = []
        is_alive = np.ones(self.pop.C, dtype=np.bool)
        n_manip_used = 0
        for r in range(self.pop.C - 1):
            scores_tot = np.full(self.pop.C, np.nan)
            scores_tot[is_alive] = np.sum(np.equal(
                self.pop.preferences_borda_rk[
                    np.logical_not(self.v_wants_to_help_c[:, c]), :][
                        :, is_alive],
                np.max(self.pop.preferences_borda_rk[
                    np.logical_not(self.v_wants_to_help_c[:, c]), :][
                        :, is_alive], 1)[:, np.newaxis]
            ), 0)
            loser = candidates_not_c[np.where(
                scores_tot[candidates_not_c] ==
                np.nanmin(scores_tot[candidates_not_c])
            )[0][-1]]
            n_manip_used = max(
                n_manip_used,
                scores_tot[loser] - scores_tot[c] + (c > loser))
            is_alive[loser] = False
            example_path.append(loser)
        example_path.append(c)
        # Conclude for c
        self._sufficient_coalition_size_TM[c] = n_manip_used
        self._mylogv('TM: sufficient_coalition_size_TM[c]',
                     self._sufficient_coalition_size_TM[c], 2)
        self._example_path_TM[c] = np.array(example_path)
        self._mylogv('TM: example_path_TM[c] =',
                     self._example_path_TM[c], 2)
        if self.pop.matrix_duels_ut[c, self.w] >= \
                self._sufficient_coalition_size_TM[c]:
            self._candidates_TM[c] = True
            self._is_TM = True
        else:
            self._candidates_TM[c] = False

    #%% Unison manipulation (UM)
    # Note: if pretests conclude that UM is True, no elimination path is
    # computed. But in that cases, TM is True. So we can use the elimination
    # path of TM if we need to quickly provide one for CM.

    def _UM_main_work_c(self, c):
        exact = (self.UM_option == "exact")
        n_m = self.pop.matrix_duels_ut[c, self.w]
        self._mylogv("UM: n_m =", n_m, 3)
        n_manip_fast, example_path_fast = self._CM_aux_fast(
            c, n_max=n_m, unison=True,
            preferences_borda_s=self.pop.preferences_borda_rk[
                np.logical_not(self.v_wants_to_help_c[:, c]), :])
        self._mylogv("UM: n_manip_fast =", n_manip_fast, 3)
        if n_manip_fast <= n_m:
            self._candidates_UM[c] = True
            if self._example_path_UM[c] is None:
                self._example_path_UM[c] = example_path_fast
            return
        if not exact:
            self._candidates_UM[c] = np.nan
            return
            
        # From this point, we have necessarily the 'exact' option (and have
        # not found a manipulation for c yet).
        n_manip_exact, example_path_exact = self._CM_aux_exact(
            c, n_m, unison=True,
            preferences_borda_s=self.pop.preferences_borda_rk[
                np.logical_not(self.v_wants_to_help_c[:, c]), :])
        self._mylogv("UM: n_manip_exact =", n_manip_exact)
        if n_manip_exact <= n_m:
            self._candidates_UM[c] = True
            if self._example_path_UM[c] is None:
                self._example_path_UM[c] = example_path_exact
        else:
            self._candidates_UM[c] = False
            
    #%% Ignorant-Coalition Manipulation (ICM)

    # Use the methods from superclass.

    #%% Coalition Manipulation (CM)

    def _CM_preliminary_checks_c(self, c, optimize_bounds):
        # A mandatory additional precheck, to ensure that
        # _example_path_CM[c] is updated if sufficient_coalition_size_CM[c]
        # has been updated.
        # We use the following syntax (instead of
        # _CM_preliminary_checks_c_subclass) because we want the test here
        # to be done, even if another one succeeded.
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

    def _CM_main_work_c(self, c, optimize_bounds):
        exact = (self.CM_option == "exact")
        if optimize_bounds and exact:
            n_max = self._sufficient_coalition_size_CM[c] - 1
        else:
            n_max = self.pop.matrix_duels_ut[c, self.w]
        self._mylogv("CM: n_max =", n_max, 3)
        if not exact and self._necessary_coalition_size_CM[c] > n_max:
            self._mylog("CM: Fast algorithm will not do better than " +
                        "what we already know", 3)
            return
        n_manip_fast, example_path_fast = self._CM_aux_fast(
            c, n_max, unison=False,
            preferences_borda_s=self.pop.preferences_borda_rk[
                np.logical_not(self.v_wants_to_help_c[:, c]), :])
        self._mylogv("CM: n_manip_fast =", n_manip_fast, 3)
        if n_manip_fast < self._sufficient_coalition_size_CM[c]:
            self._sufficient_coalition_size_CM[c] = n_manip_fast
            self._example_path_CM[c] = example_path_fast
            self._mylogv('CM: Update sufficient_coalition_size_CM[c] = '
                         'n_manip_fast =', n_manip_fast, 3)
        if not exact:
            # With fast algo, we stop here anyway. It is not a "quick escape"
            # (if we'd try again with optimize_bounds, we would not try
            # better).
            return False
        
        # From this point, we have necessarily the 'exact' option
        if self._sufficient_coalition_size_CM[c] == (
                self._necessary_coalition_size_CM[c]):
            return False
        if not optimize_bounds and (self.pop.matrix_duels_ut[c, self.w] >=
                                    self._sufficient_coalition_size_CM[c]):
            # This is a quick escape: since we have the option 'exact', if 
            # we come back with optimize_bounds, we will try to be more
            # precise.
            return True

        # Either we're with optimize_bounds (and might have succeeded), or in
        # non-optimized mode (and we have failed)
        n_max_updated = min(n_manip_fast - 1, n_max)
        self._mylogv("CM: n_max_updated =", n_max_updated)
        n_manip_exact, example_path_exact = self._CM_aux_exact(
            c, n_max_updated, unison=False,
            preferences_borda_s=self.pop.preferences_borda_rk[
                np.logical_not(self.v_wants_to_help_c[:, c]), :])
        self._mylogv("CM: n_manip_exact =", n_manip_exact)
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
                'CM: Update necessary_coalition_size_CM[c] = '
                'sufficient_coalition_size_CM[c] =')
            return False
        else:
            if self.pop.matrix_duels_ut[c, self.w] >= \
                    self._sufficient_coalition_size_CM[c]:
                # We have optimized the size of the coalition.
                self._update_necessary(
                    self._necessary_coalition_size_CM, c,
                    self._sufficient_coalition_size_CM[c],
                    'CM: Update necessary_coalition_size_CM[c] = '
                    'sufficient_coalition_size_CM[c] =')
                return False
            else:
                # We have explored everything with n_max = n_m but 
                # manipulation failed. However, we have not optimized
                # sufficient_size (which must be higher than n_m), so it
                # is a quick escape.
                self._update_necessary(
                    self._necessary_coalition_size_CM, c,
                    self.pop.matrix_duels_ut[c, self.w] + 1,
                    'CM: Update necessary_coalition_size_CM[c] = n_m + 1 =')
                return True


if __name__ == '__main__':
    # A quick demo
    preferences_utilities = np.random.randint(-5, 5, (10, 5))
    pop = Population(preferences_utilities)
    election = ExhaustiveBallot(pop)
    election.demo(log_depth=3)