# -*- coding: utf-8 -*-
"""
Created on Mon Sep 22 15:24:52 2014
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
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from svvamp.Utils import MyLog


class _Storage:
    """This class is used to store things."""
    pass


class Population(MyLog.MyLog):

    #TODO: implement majority_favorite and co, since they are in doc

    def __init__(self, preferences_ut=None, preferences_rk=None,
                 log_creation=None, labels_candidates=None):
        """Create a population with preferences. 

        :param preferences_ut: 2d array of floats.
            ``preferences_ut[v, c]`` is the utility of candidate ``c``
            as seen by voter ``v``.
        :param preferences_rk: 2d array of integers.
            ``preferences_rk[v, k]`` is the candidate at rank ``k`` for
            voter ``v``.
        :param log_creation: Any type (string, list...). Some comments.
        :param labels_candidates: List of strings. Names of the candidates.

        You may enter ``preferences_ut``, ``preferences_rk`` or both
        to define the preferences of the population.

        If you provide ``preferences_rk`` only,
        then ``preferences_ut`` is set to the corresponding Borda
        scores (:attr:`~svvamp.Population.preferences_borda_rk`).

        If you provide ``preferences_ut`` only, then ``preferences_rk`` is
        naturally derived from utilities. If voter ``v`` has a greater utility
        for candidate ``c`` than for candidate ``d``, then she ranks ``c``
        before ``d``. If voter ``v`` attributes the same utility to several
        candidates, then the first time the attribute ``preferences_rk`` is
        called, a random ranking will be decided for these tied candidates
        (once and for all).

        If you provide both, then SVVAMP checks that they are coherent,
        in the sense that if ``v`` ranks ``c`` before ``d``, then she her
        utility for ``c`` must be at least equal to her utility for ``d``. If
        it is not the case, an error is raised.

        ``preferences_rk`` will be used for sincere voting when a voting
        system accepts only strict orders.

        In contrast, ``preferences_ut`` is always used for manipulation
        purposes, which means that indifference is taken
        into account as such to determine the interest in manipulation. If
        voter ``v`` attributes the same utility to candidates ``w`` and
        ``c``, and if ``w`` is the sincere winner in an election, then ``v``
        is not interested in a manipulation for ``c``.

        If all voters have a strict order of preference (in the sense of ut),
        then for functions below having variants with suffix ``_ut`` or
        ``_rk``, the two variants are equivalent.

        In some voting systems and in some of the attributes below,
        we use a process referred as **candidate tie-breaking** or **CTB** in
        SVVAMP. It means that lowest-index candidates are favored. When using
        CTB, if candidates ``c`` and ``d`` are tied (for example,
        in an election using Plurality), then ``c`` is favored over ``d``
        iff ``c < d``.

        Implications between majority favorite and Condorcet criteria (cf.
        corresponding attributes below).

        ::

            majority_favorite_ut          ==>         majority_favorite_ut_ctb
            ||              ||                                ||            ||
            V               ||                                ||            ||
            Resistant Cond. V                                 V             ||
            ||       majority_favorite_rk ==> majority_favorite_rk_ctb      ||
            ||                     ||                ||                     ||
            V                      ||                ||                     V
            Condorcet_ut_abs              ==>             Condorcet_ut_abs_ctb
            ||      ||             ||                ||            ||       ||
            ||      V              V                 V             V        ||
            V         Condorcet_rk        ==>        Condorcet_rk_ctb       V
            Condorcet_ut_rel              ==>             Condorcet_ut_rel_ctb
            ||
            V
            Weak Condorcet
            ||
            V
            Condorcet-admissible

        If all voters have strict orders of preference (in the sense of
        ut) and if there is an odd number of voters, then:

            *   ``majority_favorite_ut``, ``majority_favorite_rk``,
                ``majority_favorite_ut_ctb`` and ``majority_favorite_rk_ctb``
                are equivalent,
            *   ``Condorcet_ut_abs``, ``Condorcet_ut_abs_ctb``,
                ``Condorcet_rk``, ``Condorcet_rk_ctb``,
                ``Condorcet_ut_rel``, ``Condorcet_ut_rel_ctb``,
                ``Weak Condorcet`` and ``Condorcet-admissible`` are equivalent.
        """
        super().__init__()
        self._log_identity = "POPULATION"

        # Basic variables
        if preferences_ut is None:
            # Population defined by rankings only
            self._preferences_ut = None
            self._preferences_rk = np.array(preferences_rk)
            self._V, self._C = self._preferences_rk.shape
        elif preferences_rk is None:
            # Population defined by utilities only
            self._preferences_ut = np.array(preferences_ut)
            self._preferences_rk = None
            self._V, self._C = self._preferences_ut.shape
        else:
            # Population defined by both
            self._preferences_ut = np.array(preferences_ut)
            self._preferences_rk = np.array(preferences_rk)
            self._V, self._C = self._preferences_ut.shape
            V_temp, C_temp = self._preferences_rk.shape
            if self._V != V_temp or self._C != C_temp:
                raise ValueError('Dimensions of preferences_ut and '
                                 'preferences_rk do not match.')
            for v in range(self._V):
                if np.any(self._preferences_ut[0, self._preferences_rk[0, :-1]]
                          < self._preferences_ut[0, self._preferences_rk[0, 1:]]):
                    raise ValueError('preferences_ut and preferences_rk '
                                     'are not coherent.')

        self._labels_candidates = labels_candidates
        # Missing matrices will be computed this way (on demand):
        # If preferences_ut is provided:
        #       utilities -> rankings  -> borda_rk
        # If preferences_rk is provided:
        #       rankings  -> borda_rk -> utilities

        self.log_creation = log_creation
        if self.V < 2 or self.C < 2:
            raise ValueError("A population must have at least 2 voters and "
                             "2 candidates.")

        # Other variables
        self._preferences_borda_rk = None
        self._preferences_borda_ut = None
        self._voters_sorted_by_rk = False
        self._v_has_same_ordinal_preferences_as_previous_voter = None
        self._matrix_duels_ut = None
        self._matrix_duels_rk = None
        self._matrix_victories_rk = None
        self._matrix_victories_rk_ctb = None
        self._matrix_victories_ut_rel = None
        self._matrix_victories_ut_rel_ctb = None
        self._matrix_victories_ut_abs = None
        self._matrix_victories_ut_abs_ctb = None
        self._condorcet_admissible_candidates = None
        self._nb_condorcet_admissible_candidates = None
        self._weak_condorcet_winners = None
        self._nb_weak_condorcet_winners = None
        self._condorcet_winner_rk_ctb = None
        self._condorcet_winner_rk = None
        self._condorcet_winner_ut_rel_ctb = None
        self._condorcet_winner_ut_rel = None
        self._condorcet_winner_ut_abs_ctb = None
        self._condorcet_winner_ut_abs = None
        self._resistant_condorcet_winner = None
        self._threshold_c_prevents_w_Condorcet_ut_abs = None
        self._total_utility_c = None
        self._total_utility_min = None
        self._total_utility_mean = None
        self._total_utility_max = None
        self._total_utility_std = None
        self._mean_utility_c = None
        self._mean_utility_min = None
        self._mean_utility_mean = None
        self._mean_utility_max = None
        self._mean_utility_std = None
        self._plurality_scores_rk = None
        self._plurality_scores_ut = None
        self._borda_score_c_rk = None
        self._decreasing_borda_scores_rk = None
        self._candidates_by_decreasing_borda_score_rk = None
        self._borda_score_c_ut = None
        self._decreasing_borda_scores_ut = None
        self._candidates_by_decreasing_borda_score_ut = None
        self._eb_result = None  # Storage for Exhaustive Ballot
        self._eb_manip = None  # Storage for Exhaustive Ballot
        self._eb_options = None  # Storage for Exhaustive Ballot
        self._irv_manip = None  # Storage for IRV
        self._irv_options = None  # Storage for IRV

    # %% Basic variables

    @property
    def labels_candidates(self):
        """List of :attr:`~svvamp.Population.C` strings (names of the
        candidates).
        """
        if self._labels_candidates is None:
            self._labels_candidates = [str(x) for x in range(self._C)]
        return self._labels_candidates

    @labels_candidates.setter
    def labels_candidates(self, value):
        self._labels_candidates = value

    @property
    def C(self):
        """Integer (number of candidates)."""
        return self._C

    @property
    def V(self):
        """Integer (number of voters)."""
        return self._V

    @property
    def preferences_ut(self):
        """2d array of floats. ``preferences_ut[v, c]`` is the utility
        of  candidate ``c`` as seen by voter ``v``.
        """
        if self._preferences_ut is None:
            self._mylog("Compute preference utilities", 1)
            self._preferences_ut = self.preferences_borda_rk
        return self._preferences_ut

    @property
    def preferences_rk(self):
        """2d array of integers. ``preferences_rk[v, k]`` is the
        candidate at rank ``k`` for voter ``v``.

        For example, ``preferences_rk[v, 0]`` is ``v``'s preferred
        candidate.
        """
        if self._preferences_rk is None:
            self._mylog("Compute preference rankings", 1)
            self._preferences_rk = \
                preferences_ut_to_preferences_rk(
                    self._preferences_ut)
        return self._preferences_rk

    @property
    def preferences_borda_rk(self):
        """2d array of integers. ``preferences_borda_rk[v, c]`` gains 1 point
        for each candidate ``d`` such that voter ``v`` ranks ``c`` before
        ``d``.

        So, these Borda scores are between ``0`` and ``C - 1``.
        """
        if self._preferences_borda_rk is None:
            self._mylog("Compute preference rankings in Borda format", 1)
            self._preferences_borda_rk = \
                preferences_rk_to_preferences_borda_rk(
                    self.preferences_rk)
        return self._preferences_borda_rk

    @property
    def preferences_borda_ut(self):
        """2d array of integers. ``preferences_borda_ut[v, c]`` gains 1
        point for each ``d`` such that ``v`` strictly prefers ``c`` to ``d``
        (in the sense of utilities), and 0.5 point for each ``d`` such that
        ``v`` is indifferent between ``c`` and ``d``.

        So, these Borda scores are between ``0`` and ``C - 1``.
        """
        if self._preferences_borda_ut is None:
            self._mylog("Compute preference weak orders in Borda format", 1)
            self._preferences_borda_ut = \
                preferences_ut_to_preferences_borda_ut(
                    self.preferences_ut)
        return self._preferences_borda_ut

    #%% Sort voters

    def ensure_voters_sorted_by_rk(self):
        """Ensure that voters are sorted.
        
        This function sorts voters first by their strict order of preference
        (their row in :attr:`~svvamp.Population.preferences_rk`), then
        by their weak order of preference (their row in
        :attr:`~svvamp.Population.preferences_borda_ut`). Note
        that two voters having the same strict order may have different weak
        orders, and vice versa.

        This function will be called automatically when creating an election,
        because it allows to accelerate some algorithms (typically  Individual 
        Manipulation).

        This function actually performs the sort only when it is called on a
        Population object for the first time.
        """
        if self._voters_sorted_by_rk:
            return
        self._voters_sorted_by_rk = True
        self._v_has_same_ordinal_preferences_as_previous_voter = None
        self._mylog('Sort voters by ranking', 1)
        # Ensure that tables are computed beforehand
        _ = self.preferences_rk
        _ = self.preferences_ut
        _ = self.preferences_borda_rk
        _ = self.preferences_borda_ut
        # Sort by weak order
        list_borda_ut = self.preferences_borda_ut.tolist()
        indexes = sorted(range(len(list_borda_ut)),
                         key=list_borda_ut.__getitem__)
        self._preferences_rk = self.preferences_rk[indexes, ::]
        self._preferences_ut = self.preferences_ut[indexes, ::]
        self._preferences_borda_rk = self.preferences_borda_rk[indexes, ::]
        self._preferences_borda_ut = self.preferences_borda_ut[indexes,
                                                                     ::]
        # Sort by strict ranking
        list_rankings = self.preferences_rk.tolist()
        indexes = sorted(range(len(list_rankings)),
                         key=list_rankings.__getitem__)
        self._preferences_rk = self.preferences_rk[indexes, ::]
        self._preferences_ut = self.preferences_ut[indexes, ::]
        self._preferences_borda_rk = self.preferences_borda_rk[indexes, ::]
        self._preferences_borda_ut = self.preferences_borda_ut[indexes,
                                                                     ::]

    @property
    def v_has_same_ordinal_preferences_as_previous_voter(self):
        """1d array of booleans.
        ``v_has_same_ordinal_preferences_as_previous_voter[v]`` is
        ``True`` iff voter ``v`` has the same preference strict order (row in
        :attr:`~svvamp.Population.preferences_rk`) and the same
        preference weak order (row in
        :attr:`~svvamp.Population.preferences_borda_ut`) as voter ``v-1``.

        By convention, it is ``False`` for voter ``0``.
        """
        if self._v_has_same_ordinal_preferences_as_previous_voter is None:
            self._mylog(
                "Compute v_has_same_ordinal_preferences_as_previous_voter", 1)
            self._v_has_same_ordinal_preferences_as_previous_voter = \
                np.concatenate((
                    [False],
                    np.logical_and(
                        np.all(
                            self.preferences_rk[range(self.V - 1), :] ==
                            self.preferences_rk[range(1, self.V), :], 1),
                        np.all(
                            self._preferences_borda_ut[range(self.V - 1),
                                                          :] ==
                            self._preferences_borda_ut[range(1, self.V), :],
                            1)
                    )
                ))
        return self._v_has_same_ordinal_preferences_as_previous_voter

    #%% Plurality scores

    @property
    def plurality_scores_rk(self):
        """1d array of booleans. ``plurality_scores_rk[c]`` is the number of
        voters for whom ``c`` is the top-ranked candidate (with voter
        tie-breaking).
        """
        if self._plurality_scores_rk is None:
            self._mylog("Compute Plurality scores (rk)", 1)
            self._plurality_scores_rk = np.bincount(
                self.preferences_rk[:, 0],
                minlength=self.C
            )
        return self._plurality_scores_rk

    @property
    def plurality_scores_ut(self):
        """1d array of booleans. ``plurality_scores_ut[c]`` is the number of
        voters who strictly prefer ``c`` to all other candidates.

        If a voter has several candidates with maximal utility,
        then none of them receives any point.
        """
        if self._plurality_scores_ut is None:
            self._mylog("Compute Plurality scores (ut)", 1)
            self._plurality_scores_ut = np.zeros(self.C)
            for v in range(self.V):
                c = np.argmax(self.preferences_ut[v, :])
                if np.all(np.greater(
                    self.preferences_ut[v, c],
                    self.preferences_ut[v, np.array(range(self.C)) != c]
                )):
                    self._plurality_scores_ut[c] += 1
        return self._plurality_scores_ut

    #%% Matrix of duels

    @property
    def matrix_duels_ut(self):
        """2d array of integers. ``matrix_duels_ut[c, d]`` is the number of
        voters who have a strictly greater utility for ``c`` than for ``d``.
        By convention, diagonal coefficients are set to ``0``.
        """
        if self._matrix_duels_ut is None:
            self._mylog("Compute matrix of duels", 1)
            self._matrix_duels_ut = \
                preferences_ut_to_matrix_duels_ut(
                    self.preferences_borda_ut)
        return self._matrix_duels_ut

    @property
    def matrix_duels_rk(self):
        """2d array of integers. ``matrix_duels_rk[c, d]`` is the number of
        voters who rank candidate ``c`` before ``d`` (in the sense of
        :attr:`~svvamp.Population.preferences_rk`). By convention,
        diagonal coefficients are set to 0.
        """
        if self._matrix_duels_rk is None:
            self._mylog("Compute matrix of duels (with strict orders)", 1)
            self._matrix_duels_rk = \
                preferences_ut_to_matrix_duels_ut(
                    self.preferences_borda_rk)
        return self._matrix_duels_rk

    @property
    def matrix_victories_ut_abs(self):
        """2d array of values in {0, 0.5, 1}. Matrix of absolute victories
        based on :attr:`~svvamp.Population.matrix_duels_ut`.

        ``matrix_victories_ut_abs[c, d]`` is:

            * 1   iff ``matrix_duels_ut[c, d] > V / 2``.
            * 0.5 iff ``matrix_duels_ut[c, d] = V / 2``.
            * 0   iff ``matrix_duels_ut[c, d] < V / 2``.

        By convention, diagonal coefficients are set to 0.
        """
        if self._matrix_victories_ut_abs is None:
            self._mylog("Compute matrix_victories_ut_abs", 1)
            self._matrix_victories_ut_abs = (
                np.multiply(0.5, self.matrix_duels_ut >= self.V / 2) +
                np.multiply(0.5, self.matrix_duels_ut > self.V / 2)
            )
        return self._matrix_victories_ut_abs

    @property
    def matrix_victories_ut_abs_ctb(self):
        """2d array of values in {0, 1}. Matrix of absolute victories
        based on :attr:`~svvamp.Population.matrix_duels_ut`, with tie-breaks on
        candidates.

        ``matrix_victories_ut_abs_ctb[c, d]`` is:

            * 1   iff ``matrix_duels_ut[c, d] > V / 2``, or
              ``matrix_duels_ut[c, d] = V / 2`` and ``c < d``.
            * 0   iff ``matrix_duels_ut[c, d] < V / 2``, or
              ``matrix_duels_ut[c, d] = V / 2`` and ``d < c``.

        By convention, diagonal coefficients are set to 0.
        """
        if self._matrix_victories_ut_abs_ctb is None:
            self._mylog("Compute matrix_victories_ut_abs_ctb", 1)
            self._matrix_victories_ut_abs_ctb = np.zeros((self.C, self.C))
            for c in range(self.C):
                for d in range(self.C):
                    if c == d:
                        continue
                    if self.matrix_duels_ut[c, d] > self.V / 2 or (
                            self.matrix_duels_ut[c, d] == self.V / 2 and c < d):
                        self._matrix_victories_ut_abs_ctb[c, d] = 1
        return self._matrix_victories_ut_abs_ctb

    @property
    def matrix_victories_ut_rel(self):
        """2d array of values in {0, 0.5, 1}. Matrix of relative victories
        based on :attr:`~svvamp.Population.matrix_duels_ut`.

        ``matrix_victories_ut_rel[c, d]`` is:

            * 1   iff ``matrix_duels_ut[c, d] > matrix_duels_ut[d, c]``.
            * 0.5 iff ``matrix_duels_ut[c, d] = matrix_duels_ut[d, c]``.
            * 0   iff ``matrix_duels_ut[c, d] < matrix_duels_ut[d, c]``.

        By convention, diagonal coefficients are set to 0.
        """
        if self._matrix_victories_ut_rel is None:
            self._mylog("Compute matrix_victories_ut_rel", 1)
            self._matrix_victories_ut_rel = (
                np.multiply(0.5, self.matrix_duels_ut >= self.matrix_duels_ut.T) +
                np.multiply(0.5, self.matrix_duels_ut > self.matrix_duels_ut.T) -
                np.multiply(0.5, np.eye(self.C))
            )
        return self._matrix_victories_ut_rel

    @property
    def matrix_victories_ut_rel_ctb(self):
        """2d array of values in {0, 1}. Matrix of relative victories
        based on :attr:`~svvamp.Population.matrix_duels_ut`, with tie-breaks on
        candidates.

        ``matrix_victories_ut_rel_ctb[c, d]`` is:

            * 1   iff ``matrix_duels_ut[c, d] > matrix_duels_ut[d, c]``, or
              ``matrix_duels_ut[c, d] = matrix_duels_ut[d, c]`` and ``c < d``.
            * 0   iff ``matrix_duels_ut[c, d] < matrix_duels_ut[d, c]``, or
              ``matrix_duels_ut[c, d] = matrix_duels_ut[d, c]`` and ``d < c``.

        By convention, diagonal coefficients are set to 0.
        """
        if self._matrix_victories_ut_rel_ctb is None:
            self._mylog("Compute matrix_victories_ut_rel_ctb", 1)
            self._matrix_victories_ut_rel_ctb = np.zeros((self.C, self.C))
            for c in range(self.C):
                for d in range(c + 1, self.C):
                    if self.matrix_duels_ut[c, d] >= self.matrix_duels_ut[d, c]:
                        self._matrix_victories_ut_rel_ctb[c, d] = 1
                        self._matrix_victories_ut_rel_ctb[d, c] = 0
                    else:
                        self._matrix_victories_ut_rel_ctb[c, d] = 0
                        self._matrix_victories_ut_rel_ctb[d, c] = 1
        return self._matrix_victories_ut_rel_ctb

    @property
    def matrix_victories_rk(self):
        """2d array of values in {0, 0.5, 1}. Matrix of victories
        based on :attr:`~svvamp.Population.matrix_duels_rk`.

        ``matrix_victories_rk[c, d]`` is:

            * 1   iff ``matrix_duels_rk[c, d] > matrix_duels_rk[d, c]``,
              i.e. iff ``matrix_duels_rk[c, d] > V / 2``.
            * 0.5 iff ``matrix_duels_rk[c, d] = matrix_duels_rk[d, c]``,
              i.e. iff ``matrix_duels_rk[c, d] = V / 2``.
            * 0   iff ``matrix_duels_rk[c, d] < matrix_duels_rk[d, c]``,
              i.e. iff ``matrix_duels_rk[c, d] < V / 2``.

        By convention, diagonal coefficients are set to 0.
        """
        if self._matrix_victories_rk is None:
            self._mylog("Compute matrix_victories_rk", 1)
            self._matrix_victories_rk = (
                np.multiply(0.5, self.matrix_duels_rk >= self.V / 2) +
                np.multiply(0.5, self.matrix_duels_rk > self.V / 2)
            )
        return self._matrix_victories_rk

    @property
    def matrix_victories_rk_ctb(self):
        """2d array of values in {0, 1}. Matrix of victories based on
        :attr:`~svvamp.Population.matrix_duels_rk`,
        with tie-breaks on candidates.

        ``matrix_victories_rk_ctb[c, d]`` is:

            * 1   iff ``matrix_duels_rk[c, d] > matrix_duels_rk[d, c]``, or
              ``matrix_duels_rk[c, d] = matrix_duels_rk[d, c]`` and
              ``c < d``.
            * 0   iff ``matrix_duels_rk[c, d] < matrix_duels_rk[d, c]``, or
              ``matrix_duels_rk[c, d] = matrix_duels_rk[d, c]`` and
              ``d < c``.

        By convention, diagonal coefficients are set to 0.
        """
        if self._matrix_victories_rk_ctb is None:
            self._mylog("Compute matrix_victories_rk_ctb", 1)
            self._matrix_victories_rk_ctb = np.zeros((self.C, self.C))
            for c in range(self.C):
                for d in range(c + 1, self.C):
                    if self.matrix_duels_rk[c, d] >= self.V / 2:
                        self._matrix_victories_rk_ctb[c, d] = 1
                        self._matrix_victories_rk_ctb[d, c] = 0
                    else:
                        self._matrix_victories_rk_ctb[c, d] = 0
                        self._matrix_victories_rk_ctb[d, c] = 1
        return self._matrix_victories_rk_ctb

    #%% Condorcet winner and variants

    @property
    def condorcet_admissible_candidates(self):
        """1d array of booleans. ``condorcet_admissible_candidates[c]`` is
        ``True`` iff candidate ``c`` is Condorcet-admissible, i.e. iff no
        candidate ``d`` has an absolute victory against ``c`` (in
        the sense of :attr:`~svvamp.Population.matrix_victories_ut_abs`).

        .. seealso:: :attr:`~svvamp.Population.nb_condorcet_admissible`,
                     :attr:`~svvamp.Population.exists_condorcet_admissible`,
                     :attr:`~svvamp.Population.not_exists_condorcet_admissible`.
        """
        if self._condorcet_admissible_candidates is None:
            self._mylog("Compute Condorcet-admissible candidates", 1)
            self._condorcet_admissible_candidates = np.all(
                self.matrix_victories_ut_abs <= 0.5, 0)
        return self._condorcet_admissible_candidates

    @property
    def nb_condorcet_admissible(self):
        """Integer (number of Condorcet-admissible candidates,
        :attr:`~svvamp.Population.condorcet_admissible_candidates`).
        """
        if self._nb_condorcet_admissible_candidates is None:
            self._mylog("Compute number of Condorcet-admissible candidates", 1)
            self._nb_condorcet_admissible_candidates = np.sum(
                self.condorcet_admissible_candidates)
        return self._nb_condorcet_admissible_candidates

    @property
    def exists_condorcet_admissible(self):
        """Boolean (``True`` iff there is at least one Condorcet-admissible
        candidate, :attr:`~svvamp.Population.condorcet_admissible_candidates`).
        """
        return self.nb_condorcet_admissible > 0

    @property
    def not_exists_condorcet_admissible(self):
        """Boolean (``True`` iff there is no Condorcet-admissible candidate,
        :attr:`~svvamp.Population.condorcet_admissible_candidates`).
        """
        return self.nb_condorcet_admissible == 0

    @property
    def weak_condorcet_winners(self):
        """1d array of booleans. ``weak_condorcet_winners[c]`` is ``True`` iff
        candidate ``c`` is a weak Condorcet winner, i.e. iff no candidate
        ``d`` has a relative victory against ``c`` (in the sense of
        :attr:`~svvamp.Population.matrix_victories_ut_rel`).

        .. seealso:

            :attr:`~svvamp.Population.nb_weak_condorcet_winners`,
            :attr:`~svvamp.Population.exists_weak_condorcet_winner`,
            :attr:`~svvamp.Population.not_exists_weak_condorcet_winner`.
        """
        if self._weak_condorcet_winners is None:
            self._mylog("Compute weak Condorcet winners", 1)
            self._weak_condorcet_winners = np.all(
                self.matrix_victories_ut_rel <= 0.5, 0)
        return self._weak_condorcet_winners

    @property
    def nb_weak_condorcet_winners(self):
        """Integer (number of weak Condorcet winners,
        :attr:`~svvamp.Population.weak_condorcet_winners`).
        """
        if self._nb_weak_condorcet_winners is None:
            self._mylog("Compute number of weak Condorcet winners", 1)
            self._nb_weak_condorcet_winners = np.sum(
                self.weak_condorcet_winners)
        return self._nb_weak_condorcet_winners

    @property
    def exists_weak_condorcet_winner(self):
        """Boolean (``True`` iff there is at least one weak Condorcet winner,
        :attr:`~svvamp.Population.weak_condorcet_winners`).
        """
        return self.nb_weak_condorcet_winners > 0

    @property
    def not_exists_weak_condorcet_winner(self):
        """Boolean (``True`` iff there is no weak Condorcet winner,
        :attr:`~svvamp.Population.weak_condorcet_winners`).
        """
        return self.nb_weak_condorcet_winners == 0

    @property
    def condorcet_winner_rk_ctb(self):
        """Integer or ``NaN``. Candidate who has only victories in
        :attr:`~svvamp.Population.matrix_victories_rk_ctb`. If there is no
        such candidate, then ``NaN``.

        .. seealso::

            :attr:`~svvamp.Population.exists_condorcet_winner_rk_ctb`,
            :attr:`~svvamp.Population.not_exists_condorcet_winner_rk_ctb`.
        """
        if self._condorcet_winner_rk_ctb is None:
            self._mylog("Compute condorcet_winner_rk_ctb", 1)
            for c in range(self.C):
                # The whole COLUMN must be 0.
                if np.array_equiv(self.matrix_victories_rk_ctb[:, c], 0):
                    self._condorcet_winner_rk_ctb = c
                    break
            else:
                self._condorcet_winner_rk_ctb = np.nan
        return self._condorcet_winner_rk_ctb

    @property
    def exists_condorcet_winner_rk_ctb(self):
        """Boolean (``True`` iff there is a
        :attr:`~svvamp.Population.condorcet_winner_rk_ctb`).
        """
        return not np.isnan(self.condorcet_winner_rk_ctb)

    @property
    def not_exists_condorcet_winner_rk_ctb(self):
        """Boolean (``True`` iff there is no
        :attr:`~svvamp.Population.condorcet_winner_rk_ctb`).
        """
        return np.isnan(self.condorcet_winner_rk_ctb)

    @property
    def condorcet_winner_rk(self):
        """Integer or ``NaN``. Candidate who has only victories in
        :attr:`~svvamp.Population.matrix_victories_rk`. If there is no such
        candidate, then ``NaN``.

        .. seealso:: :attr:`~svvamp.Population.exists_condorcet_winner_rk`,
                     :attr:`~svvamp.Population.not_exists_condorcet_winner_rk`.
        """
        if self._condorcet_winner_rk is None:
            self._mylog("Compute condorcet_winner_rk", 1)
            for c in range(self.C):
                # The whole COLUMN must be 0.
                if np.array_equiv(self.matrix_victories_rk[:, c], 0):
                    self._condorcet_winner_rk = c
                    break
            else:
                self._condorcet_winner_rk = np.nan
        return self._condorcet_winner_rk

    @property
    def exists_condorcet_winner_rk(self):
        """Boolean (``True`` iff there is a
        :attr:`~svvamp.Population.condorcet_winner_rk`).
        """
        return not np.isnan(self.condorcet_winner_rk)

    @property
    def not_exists_condorcet_winner_rk(self):
        """Boolean (``True`` iff there is no
        :attr:`~svvamp.Population.condorcet_winner_rk`).
        """
        return np.isnan(self.condorcet_winner_rk)

    @property
    def condorcet_winner_ut_rel_ctb(self):
        """Integer or ``NaN``.
        Candidate who has only victories in
        :attr:`~svvamp.Population.matrix_victories_ut_rel_ctb`. If there is no
        such candidate, then ``NaN``.

        .. seealso::

            :attr:`~svvamp.Population.exists_condorcet_winner_ut_rel_ctb`,
            :attr:`~svvamp.Population.not_exists_condorcet_winner_ut_rel_ctb`.
        """
        if self._condorcet_winner_ut_rel_ctb is None:
            self._mylog("Compute condorcet_winner_ut_rel_ctb", 1)
            for c in range(self.C):
                # The whole COLUMN must be 0.
                if np.array_equiv(self.matrix_victories_ut_rel_ctb[:, c], 0):
                    self._condorcet_winner_ut_rel_ctb = c
                    break
            else:
                self._condorcet_winner_ut_rel_ctb = np.nan
        return self._condorcet_winner_ut_rel_ctb

    @property
    def exists_condorcet_winner_ut_rel_ctb(self):
        """Boolean (``True`` iff there is a
        :attr:`~svvamp.Population.condorcet_winner_ut_rel_ctb`).
        """
        return not np.isnan(self.condorcet_winner_ut_rel_ctb)

    @property
    def not_exists_condorcet_winner_ut_rel_ctb(self):
        """Boolean (``True`` iff there is no
        :attr:`~svvamp.Population.condorcet_winner_ut_rel_ctb`).
        """
        return np.isnan(self.condorcet_winner_ut_rel_ctb)

    @property
    def condorcet_winner_ut_rel(self):
        """Integer or ``NaN``. Candidate who has only victories in
        :attr:`~svvamp.Population.matrix_victories_ut_rel`. If there is no such
        candidate, then ``NaN``.

        .. seealso:: :attr:`~svvamp.Population.exists_condorcet_winner_ut_rel`,
                     :attr:`~svvamp.Population.not_exists_condorcet_winner_ut_rel`.
        """
        if self._condorcet_winner_ut_rel is None:
            self._mylog("Compute condorcet_winner_ut_rel", 1)
            for c in range(self.C):
                # The whole COLUMN must be 0.
                if np.array_equiv(self.matrix_victories_ut_rel[:, c], 0):
                    self._condorcet_winner_ut_rel = c
                    break
            else:
                self._condorcet_winner_ut_rel = np.nan
        return self._condorcet_winner_ut_rel

    @property
    def exists_condorcet_winner_ut_rel(self):
        """Boolean (``True`` iff there is a
        :attr:`~svvamp.Population.condorcet_winner_ut_rel`).
        """
        return not np.isnan(self.condorcet_winner_ut_rel)

    @property
    def not_exists_condorcet_winner_ut_rel(self):
        """Boolean (``True`` iff there is no
        :attr:`~svvamp.Population.condorcet_winner_ut_rel`).
        """
        return np.isnan(self.condorcet_winner_ut_rel)

    @property
    def condorcet_winner_ut_abs_ctb(self):
        """Integer or ``NaN``.
        Candidate who has only victories in
        :attr:`~svvamp.Population.matrix_victories_ut_abs_ctb`. If there is no
        such candidate, then ``NaN``.

        .. seealso:: :attr:`~svvamp.Population.exists_condorcet_winner_ut_abs_ctb`,
                     :attr:`~svvamp.Population.not_exists_condorcet_winner_ut_abs_ctb`
        """
        if self._condorcet_winner_ut_abs_ctb is None:
            self._mylog("Compute condorcet_winner_ut_abs_ctb", 1)
            for c in range(self.C):
                if np.array_equiv(self.matrix_victories_ut_abs_ctb[c,
                                  np.array(range(self.C)) != c],
                                  1):
                    self._condorcet_winner_ut_abs_ctb = c
                    break
            else:
                self._condorcet_winner_ut_abs_ctb = np.nan
        return self._condorcet_winner_ut_abs_ctb

    @property
    def exists_condorcet_winner_ut_abs_ctb(self):
        """Boolean (``True`` iff there is a
        :attr:`~svvamp.Population.condorcet_winner_ut_abs_ctb`).
        """
        return not np.isnan(self.condorcet_winner_ut_abs_ctb)

    @property
    def not_exists_condorcet_winner_ut_abs_ctb(self):
        """Boolean (``True`` iff there is no
        :attr:`~svvamp.Population.condorcet_winner_ut_abs_ctb`).
        """
        return np.isnan(self.condorcet_winner_ut_abs_ctb)

    @property
    def condorcet_winner_ut_abs(self):
        """Integer or ``NaN``. Candidate who has only victories in
        :attr:`~svvamp.Population.matrix_victories_ut_abs`. If there is no such
        candidate, then ``NaN``.

        .. seealso:: :attr:`~svvamp.Population.exists_condorcet_winner_ut_abs`,
                     :attr:`~svvamp.Population.not_exists_condorcet_winner_ut_abs`.
        """
        if self._condorcet_winner_ut_abs is None:
            self._mylog("Compute Condorcet winner", 1)
            for c in range(self.C):
                if np.array_equiv(self.matrix_victories_ut_abs[c,
                                  np.array(range(self.C)) != c],
                                  1):
                    self._condorcet_winner_ut_abs = c
                    break
            else:
                self._condorcet_winner_ut_abs = np.nan
        return self._condorcet_winner_ut_abs

    @property
    def exists_condorcet_winner_ut_abs(self):
        """Boolean (``True`` iff there is a
        :attr:`~svvamp.Population.condorcet_winner_ut_abs`).
        """
        return not np.isnan(self.condorcet_winner_ut_abs)

    @property
    def not_exists_condorcet_winner_ut_abs(self):
        """Boolean (``True`` iff there is no
        :attr:`~svvamp.Population.condorcet_winner_ut_abs`).
        """
        return np.isnan(self.condorcet_winner_ut_abs)

    @property
    def resistant_condorcet_winner(self):
        """Integer or ``NaN``. Resistant Condorcet Winner. If there is no such
        candidate, then ``NaN``.
        
        A Condorcet winner ``w`` (in the sense of
        :attr:`~svvamp.Population.condorcet_winner_ut_abs`) is resistant iff
        in any Condorcet voting system (in the same sense), the profile is not
        manipulable (cf.
        Durand et al., working paper 2014).
        This is equivalent to say that for any pair ``(c, d)`` of other
        distinct candidates, there is a strict majority of voters who
        simultaneously:

            1) Do not prefer ``c`` to ``w``,
            2) And prefer ``w`` to ``d``.

        .. seealso::

            :attr:`~svvamp.Population.exists_resistant_condorcet_winner`,
            :attr:`~svvamp.Population.not_exists_resistant_condorcet_winner`.
        """
        if self._resistant_condorcet_winner is None:
            self._mylog("Compute Resistant Condorcet winner", 1)
            if is_resistant_condorcet(self.condorcet_winner_ut_abs,
                                      self.preferences_ut):
                self._resistant_condorcet_winner = self.condorcet_winner_ut_abs
            else:
                self._resistant_condorcet_winner = np.nan
        return self._resistant_condorcet_winner

    @property
    def exists_resistant_condorcet_winner(self):
        """Boolean (``True`` iff there is a
        :attr:`~svvamp.Population.resistant_condorcet_winner`).
        """
        return not np.isnan(self.resistant_condorcet_winner)

    @property
    def not_exists_resistant_condorcet_winner(self):
        """Boolean (``True`` iff there is no
        :attr:`~svvamp.Population.resistant_condorcet_winner`).
        """
        return np.isnan(self.resistant_condorcet_winner)

    @property
    def threshold_c_prevents_w_Condorcet_ut_abs(self):
        """2d array of integers. Threshold for ``c``-manipulators to prevent
        ``w`` from being a Condorcet winner (in the sense of
        :attr:`~svvamp.Population.condorcet_winner_ut_abs`).

        Intuitively, the question is the following: in an election where ``w``
        is the winner, how many ``c``-manipulators are needed to prevent ``w``
        from  being a Condorcet winner?
        
        We start with the sub-population of :math:`n_s` 'sincere' voters,
        i.e. not preferring ``c`` to ``w``. The precise question is: how many
        ``c``-manipulators :math:`n_m` must we add in order to create a
        non-victory for
        ``w`` against some candidate ``d`` :math:`\\neq` ``w`` (possibly ``c``
        herself)?

        In the following, :math:`| c > d |` denotes the number of voters who
        strictly prefer candidate ``c`` to ``d``. We need:

        .. math::

            | \\text{sincere voters for whom } w > d |
            \\leq \\frac{n_s + n_m}{2}.

        I.e.:

        .. math::

            | \\text{non } c > w \\text{ and } w > d | \\leq
            \\frac{| \\text{non } c > w | + n_m}{2}.

        I.e.:

        .. math::

            n_m \\geq 2 \\cdot | \\text{non } c > w \\text{ and } w > d |
            - | \\text{non } c > w |.

        One candidate ``d`` is enough, so:

        ``threshold_c_prevents_w_Condorcet_ut_abs[c, w]`` :math:`=
        2 \\cdot \\min_{d \\neq w} |w \geq c \\text{ and } w > d|
        - |w \geq c|`.

        If this result is negative, it means that even without
        ``c``-manipulators, ``w`` is not a Condorcet winner. In that case,
        threshold is set to 0 instead.
        """
        if self._threshold_c_prevents_w_Condorcet_ut_abs is None:
            self._mylog("Compute threshold_c_prevents_w_Condorcet_ut_abs", 1)
            self._threshold_c_prevents_w_Condorcet_ut_abs = np.full((self.C, self.C),
                                                             np.inf)
            for w in range(self.C):
                for c in range(self.C):
                    if c == w:
                        self._threshold_c_prevents_w_Condorcet_ut_abs[c, w] = 0
                        continue
                    v_does_not_prefer_c_to_w = (
                        self.preferences_ut[:, w] >=
                        self.preferences_ut[:, c])
                    for d in range(self.C):
                        if d == w:
                            continue
                            # But d == c is allowed (useful if self.C == 2)
                        v_prefers_w_to_d = (
                            self.preferences_ut[:, w] >
                            self.preferences_ut[:, d])
                        threshold_c_makes_w_not_win_against_d = (
                            np.multiply(2, np.sum(np.logical_and(
                                v_does_not_prefer_c_to_w,
                                v_prefers_w_to_d
                            )))
                            - np.sum(v_does_not_prefer_c_to_w)
                        )
                        self._threshold_c_prevents_w_Condorcet_ut_abs[
                            c, w] = np.minimum(
                                self._threshold_c_prevents_w_Condorcet_ut_abs[c, w],
                                threshold_c_makes_w_not_win_against_d
                            )
                    self._threshold_c_prevents_w_Condorcet_ut_abs = np.maximum(
                        self._threshold_c_prevents_w_Condorcet_ut_abs, 0)
        return self._threshold_c_prevents_w_Condorcet_ut_abs

    #%% Total utilities        

    @property
    def total_utility_c(self):
        """1d array of floats. ``total_utility_c[c]`` is the total utility for
        candidate ``c`` (i.e. the sum of ``c``'s column in matrix
        :attr:`~svvamp.Population.preferences_ut`).
        """
        if self._total_utility_c is None:
            self._mylog("Compute total utility of candidates", 1)
            self._total_utility_c = np.sum(self.preferences_ut, 0)
        return self._total_utility_c

    @property
    def total_utility_min(self):
        """Float. ``total_utility_min`` is the minimum of
        :attr:`~svvamp.Population.total_utility_c`.
        """
        if self._total_utility_min is None:
            self._mylog("Compute total_utility_min", 1)
            self._total_utility_min = np.min(self.total_utility_c)
        return self._total_utility_min

    @property
    def total_utility_max(self):
        """Float. ``total_utility_max`` is the maximum of
        :attr:`~svvamp.Population.total_utility_c`.
        """
        if self._total_utility_max is None:
            self._mylog("Compute total_utility_max", 1)
            self._total_utility_max = np.max(self.total_utility_c)
        return self._total_utility_max

    @property
    def total_utility_mean(self):
        """Float. ``total_utility_mean`` is the mean of
        :attr:`~svvamp.Population.total_utility_c`.
        """
        if self._total_utility_mean is None:
            self._mylog("Compute total_utility_mean", 1)
            self._total_utility_mean = np.mean(self.total_utility_c)
        return self._total_utility_mean

    @property
    def total_utility_std(self):
        """Float. ``total_utility_std`` is the standard deviation of
        :attr:`~svvamp.Population.total_utility_c`.
        """
        if self._total_utility_std is None:
            self._mylog("Compute total_utility_std", 1)
            self._total_utility_std = np.std(self.total_utility_c, ddof=0)
        return self._total_utility_std

    #%% Mean utilities        

    @property
    def mean_utility_c(self):
        """1d array of floats. ``mean_utility_c[c]`` is the mean utility for
        candidate ``c`` (i.e. the mean of ``c``'s column in matrix
        :attr:`~svvamp.Population.preferences_ut`).
        """
        if self._mean_utility_c is None:
            self._mylog("Compute mean utility of candidates", 1)
            self._mean_utility_c = np.mean(self.preferences_ut, 0)
        return self._mean_utility_c

    @property
    def mean_utility_min(self):
        """Float. ``mean_utility_min`` is the minimum of
        :attr:`~svvamp.Population.mean_utility_c`.
        """
        if self._mean_utility_min is None:
            self._mylog("Compute mean_utility_min", 1)
            self._mean_utility_min = np.min(self.mean_utility_c)
        return self._mean_utility_min

    @property
    def mean_utility_max(self):
        """Float. ``mean_utility_max`` is the maximum of
        :attr:`~svvamp.Population.mean_utility_c`.
        """
        if self._mean_utility_max is None:
            self._mylog("Compute mean_utility_max", 1)
            self._mean_utility_max = np.max(self.mean_utility_c)
        return self._mean_utility_max

    @property
    def mean_utility_mean(self):
        """Float. ``mean_utility_mean`` is the mean of
        :attr:`~svvamp.Population.mean_utility_c`.
        """
        if self._mean_utility_mean is None:
            self._mylog("Compute mean_utility_mean", 1)
            self._mean_utility_mean = np.mean(self.mean_utility_c)
        return self._mean_utility_mean

    @property
    def mean_utility_std(self):
        """Float. ``mean_utility_std`` is the standard deviation of
        :attr:`~svvamp.Population.mean_utility_c`.
        """
        if self._mean_utility_std is None:
            self._mylog("Compute mean_utility_std", 1)
            self._mean_utility_std = np.std(self.mean_utility_c, ddof=0)
        return self._mean_utility_std

    #%% Borda scores

    @property
    def borda_score_c_rk(self):
        """1d array of integers. ``borda_score_c_rk[c]`` is the total Borda
        score of candidate ``c`` (using
        :attr:`~svvamp.Population.preferences_borda_rk`, i.e. strict
        preferences).
        """
        if self._borda_score_c_rk is None:
            self._mylog("Compute Borda scores of the candidates (rankings)", 1)
            self._borda_score_c_rk = np.sum(self.matrix_duels_rk, 1)
        return self._borda_score_c_rk

    @property
    def borda_score_c_ut(self):
        """1d array of integers. ``borda_score_c_ut[c]`` is the total Borda
        score of candidate ``c`` (using
        :attr:`~svvamp.Population.preferences_borda_ut`, i.e. weak
        preferences).
        """
        if self._borda_score_c_ut is None:
            self._mylog("Compute Borda scores of the candidates (weak "
                        "orders)", 1)
            self._borda_score_c_ut = np.sum(self.preferences_borda_ut, 0)
        return self._borda_score_c_ut

    @property
    def candidates_by_decreasing_borda_score_rk(self):
        """1d array of integers.
        ``candidates_by_decreasing_borda_score_rk[k]`` is
        the candidate ranked ``k``\ :sup:`th` by decreasing Borda score
        (using :attr:`~svvamp.Population.borda_score_c_rk`, i.e. strict
        preferences).

        For example, ``candidates_by_decreasing_borda_score_rk[0]`` is the
        candidate with highest Borda score (rk).
        """
        if self._candidates_by_decreasing_borda_score_rk is None:
            self._mylog("Compute candidates_by_decreasing_borda_score_rk", 1)
            self._candidates_by_decreasing_borda_score_rk = \
                np.argsort(-self.borda_score_c_rk, kind='mergesort')
            self._decreasing_borda_scores_rk = self.borda_score_c_rk[
                self._candidates_by_decreasing_borda_score_rk]
        return self._candidates_by_decreasing_borda_score_rk

    @property
    def decreasing_borda_scores_rk(self):
        """1d array of integers. ``decreasing_borda_scores_rk[k]`` is the
        ``k``\ :sup:`th` Borda score (using
        :attr:`~svvamp.Population.borda_score_c_rk`, i.e. strict
        preferences) by decreasing order.

        For example, ``decreasing_borda_scores_rk[0]`` is the highest
        Borda score for a candidate (rk).
        """
        if self._decreasing_borda_scores_rk is None:
            self._mylog("Compute decreasing_borda_scores_rk", 1)
            self._candidates_by_decreasing_borda_score_rk = \
                np.argsort(-self.borda_score_c_rk, kind='mergesort')
            self._decreasing_borda_scores_rk = self.borda_score_c_rk[
                self._candidates_by_decreasing_borda_score_rk]
        return self._decreasing_borda_scores_rk

    @property
    def candidates_by_decreasing_borda_score_ut(self):
        """1d array of integers.
        ``candidates_by_decreasing_borda_score_ut[k]``
        is the candidate ranked ``k``\ :sup:`th` by decreasing Borda score
        (using :attr:`~svvamp.Population.borda_score_c_ut`, i.e. weak
        preferences).

        For example, ``candidates_by_decreasing_borda_score_ut[0]`` is the
        candidate with highest Borda score (ut).
        """
        if self._candidates_by_decreasing_borda_score_ut is None:
            self._mylog("Compute candidates_by_decreasing_borda_score_ut",
                        1)
            self._candidates_by_decreasing_borda_score_ut = \
                np.argsort(-self.borda_score_c_ut, kind='mergesort')
            self._decreasing_borda_scores_ut = self.borda_score_c_ut[
                self._candidates_by_decreasing_borda_score_ut]
        return self._candidates_by_decreasing_borda_score_ut

    @property
    def decreasing_borda_scores_ut(self):
        """1d array of integers. ``decreasing_borda_scores_ut[k]`` is the
        ``k``\ :sup:`th` Borda score (using
        :attr:`~svvamp.Population.borda_score_c_ut`, i.e. weak
        preferences) by decreasing order.

        For example, ``decreasing_borda_scores_ut[0]`` is the highest
        Borda score for a candidate (rk).
        """
        if self._decreasing_borda_scores_ut is None:
            self._mylog("Compute decreasing_borda_scores_ut", 1)
            self._candidates_by_decreasing_borda_score_ut = \
                np.argsort(-self.borda_score_c_ut, kind='mergesort')
            self._decreasing_borda_scores_ut = self.borda_score_c_ut[
                self._candidates_by_decreasing_borda_score_ut]
        return self._decreasing_borda_scores_ut

    #%% Plot a population

    def plot3(self, indexes=None, normalize=True, use_labels=True):
        """Plot utilities for 3 candidates (with approval limit).

        :param indexes: List of 3 candidates. If None, defaults to [0, 1, 2].
        :param normalize: Boolean. Cf. below.
        :param use_labels: Boolean. If ``True``, then
            :attr:`~svvamp.Population.labels_candidates` is
            used to label the plot. Otherwise, candidates are simply
            represented by their index.

        Each red point of the plot represents a voter ``v``. Its position is
        :attr:`~svvamp.Population.preferences_ut`\ ``[v, indexes]``. If
        ``normalize`` is ``True``, then each position is normalized before
        plotting so that its Euclidean norm is equal to 1.

        The equator (in blue) is the set of points :math:`\\mathbf{u}` such
        that
        :math:`\\sum {u_i}^2 = 1` and
        :math:`\\sum u_i = 0`,
        i.e. the unit circle of the plan that is orthogonal to the main
        diagonal [1, 1, 1].

        Other blue circles are the frontiers between the 6 different strict
        total orders on the candidates.

        Cf. working paper by Durand et al., 'Geometry on the Utility
        Space'.
        """
        if indexes is None:
            _indexes = [0, 1, 2]
        else:
            _indexes = indexes
        # Define figure
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        # North-South axis (not plotted anymore)
        # ax.plot(xs=[-1, 1], ys=[-1, 1], zs=[-1, 1], c='k')
        # Equator
        theta = np.linspace(0, 2 * np.pi, 200)
        xs = np.cos(theta)/np.sqrt(2) - np.sin(theta)/np.sqrt(6)
        ys = -np.cos(theta)/np.sqrt(2) - np.sin(theta)/np.sqrt(6)
        zs = 2 * np.sin(theta)/np.sqrt(6)
        ax.scatter(xs, ys, zs, s=7, c='b')
        # Frontiers between strict orders
        for i in range(3):
            pole = np.full(3, 1/np.sqrt(3))
            ortho = np.ones(3)
            ortho[i] = -2
            ortho = ortho / np.sqrt(6)
            xs = np.cos(theta)*pole[0] + np.sin(theta)*ortho[0]
            ys = np.cos(theta)*pole[1] + np.sin(theta)*ortho[1]
            zs = np.cos(theta)*pole[2] + np.sin(theta)*ortho[2]
            ax.scatter(xs, ys, zs, s=7, c='b')
        # Voters
        mat_temp = np.copy(self.preferences_ut[:, _indexes]).astype(
            np.float)
        self._mylogm('mat_temp =', mat_temp, 3)
        if normalize:
            for v in range(mat_temp.shape[0]):
                norm = np.sqrt(np.sum(mat_temp[v, :]**2))
                if norm > 0:
                    mat_temp[v, :] /= norm
            self._mylogm('mat_temp =', mat_temp, 3)
        ax.scatter(mat_temp[:, 0], mat_temp[:, 1], mat_temp[:, 2],
                   s=40, c='r', marker='o')
        # Axes and labels
        if use_labels:
            ax.set_xlabel(self.labels_candidates[_indexes[0]] + ' (' +
                          str(_indexes[0]) + ')')
            ax.set_ylabel(self.labels_candidates[_indexes[1]] + ' (' +
                          str(_indexes[1]) + ')')
            ax.set_zlabel(self.labels_candidates[_indexes[2]] + ' (' +
                          str(_indexes[2]) + ')')
        else:
            ax.set_xlabel('Candidate ' + str(_indexes[0]))
            ax.set_ylabel('Candidate ' + str(_indexes[1]))
            ax.set_zlabel('Candidate ' + str(_indexes[2]))
        temp = 1 / np.sqrt(2)
        for best in range(3):
            for worst in range(3):
                if best == worst:
                    continue
                other = 3 - best - worst
                position = np.zeros(3)
                position[best] = temp
                position[worst] = - temp
                ax.text(position[0], position[1], position[2],
                        ' ' + str(_indexes[best]) + ' > ' +
                        str(_indexes[other]) + ' > ' +
                        str(_indexes[worst]))
        plt.show()

    def plot4(self, indexes=None, normalize=True, use_labels=True):
        """Plot utilities for 4 candidates (without approval limit).

        :param indexes: List of 4 candidates. If None, defaults to
            [0, 1, 2, 3].
        :param normalize: Boolean. Cf. below.
        :param use_labels: Boolean. If ``True``, then
            :attr:`~svvamp.Population.labels_candidates` is
            used to label the plot. Otherwise, candidates are simply
            represented by their index.

        Each red point of the plot represents a voter ``v``.

            * :attr:`~svvamp.Population.preferences_ut`\ ``[v, indexes]``
              is sent to the hyperplane that
              is orthogonal to [1, 1, 1, 1] (by orthogonal projection),
              which discards information related to approval limit and keeps
              only the relative preferences between candidates.
            * The plot is done in this 3d hyperplane. In practice, we use a
              mirror symmetry that exchanges [1, 1, 1, 1] and [0, 0, 0, 1].
              This way, the image vector is orthogonal to [0, 0, 0, 1] and
              can be plotted in the first 3 dimensions.
            * If ``normalize`` is True, then the image vector is normalized
              before plotting so that its Euclidean norm is equal to 1.

        Blue lines are the frontiers between the 24 different strict
        total orders on the candidates ('permutohedron').

        Cf. working paper by Durand et al., 'Geometry on the Utility Space'.
        """
        if indexes is None:
            _indexes = [0, 1, 2, 3]
        else:
            _indexes = indexes
        # Define figure
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Voters
        # We transform the population matrix to send it to R^3
        # 0. Keep only the specified candidates
        mat_temp = np.copy(self.preferences_ut[:, _indexes]).astype(
            np.float)
        # 1. Send each voter to the hyperplane orthogonal to (1, 1, 1, 1)
        mat_temp -= np.mean(mat_temp, 1)[:, np.newaxis]
        # 2. Use symmetry by a hyperplane to exchange (1, 1, 1, 1) and (0,
        # 0, 0, 1)
        # unitary_parallel = [1, 1, 1, 1] / sqrt(4) - [0, 0, 0, 1] then
        # normalize.
        unitary_parallel = np.array([0.5, 0.5, 0.5, -0.5])
        # unitary_parallel = unitary_parallel / np.sqrt(
        #     np.sum(unitary_parallel**2))
        temp_parallel = np.outer(
            np.sum(mat_temp * unitary_parallel[np.newaxis, :], 1),
            unitary_parallel)
        mat_temp -= 2 * temp_parallel
        # 3. Normalize if asked
        if normalize:
            for v in range(mat_temp.shape[0]):
                norm = np.sqrt(np.sum(mat_temp[v, :]**2))
                if norm > 0:
                    mat_temp[v, :] /= norm
        # 4. Now just do not care about last coordinate, it is 0.
        ax.scatter(mat_temp[:, 0], mat_temp[:, 1], mat_temp[:, 2],
                   s=40, c='r', marker='o')

        # Permutoedron
        xs = []
        ys = []
        zs = []
        def add_line(vertex1, vertex2):
            if normalize:
                theta_tot = np.arccos(np.sum(vertex1 * vertex2))
                ortho = vertex2 - np.sum(vertex2*vertex1) * vertex1
                ortho = ortho / np.sqrt(np.sum(ortho**2))
                theta = np.linspace(0, theta_tot, 100)
                xs.extend(np.cos(theta) * vertex1[0] + np.sin(theta) * ortho[0])
                ys.extend(np.cos(theta) * vertex1[1] + np.sin(theta) * ortho[1])
                zs.extend(np.cos(theta) * vertex1[2] + np.sin(theta) * ortho[2])
            else:
                xs.extend(np.linspace(vertex1[0], vertex2[0], 50))
                ys.extend(np.linspace(vertex1[1], vertex2[1], 50))
                zs.extend(np.linspace(vertex1[2], vertex2[2], 50))
        def image(vertex):
            result = np.multiply(vertex, 2) - 1
            result = result - np.mean(result)
            temp_parallel = np.sum(result*unitary_parallel) * unitary_parallel
            result -= 2 * temp_parallel
            if normalize:
                result = result / np.sqrt(np.sum(result**2))
            return result
        for i in range(4):
            for j in range(4):
                if j == i:
                    continue
                # Vertex 1 of type [1, 0, 0, 0]
                vertex1 = image(np.array(range(4)) == i)
                if use_labels:
                    pass
                    ax.text(vertex1[0], vertex1[1], vertex1[2],
                        '   Favorite = '
                        + self.labels_candidates[_indexes[i]])
                else:
                    ax.text(vertex1[0], vertex1[1], vertex1[2],
                        '   Favorite = ' + str(_indexes[i]))
                # Vertex 2 of type [1, 1, 1, 0]
                vertex2 = np.ones(4)
                vertex2[j] = 0
                vertex2 = image(vertex2)
                add_line(vertex1, vertex2)
                # Vertex 2 of type [1, 1, 0, 0]
                vertex2 = np.zeros(4)
                vertex2[i] = 1
                vertex2[j] = 1
                vertex2 = image(vertex2)
                add_line(vertex1, vertex2)
                # Vertex 1 of type [0, 1, 1, 1]
                vertex1 = image(np.array(range(4)) != i)
                # Vertex 2 of type [1, 1, 0, 0]
                vertex2 = np.ones(4)
                vertex2[i] = 0
                vertex2[j] = 0
                vertex2 = image(vertex2)
                add_line(vertex1, vertex2)
        ax.scatter(xs, ys, zs, s=7, c='b')

        # Conclude
        plt.axis('off')
        plt.show()

    #%% Demo

    def demo(self, log_depth=1):
        """Demonstrate the methods of :class:`~svvamp.Population` class.

        :param log_depth: Integer from 0 (basic info) to 3 (verbose).
        """
        old_log_depth = self._log_depth
        self._log_depth = log_depth
        self.ensure_voters_sorted_by_rk()
        def printm(variable_name, variable_value):
            print(variable_name)
            print(variable_value)
    
        MyLog.print_title("Basic properties")
        print("V =", self.V)
        print("C =", self.C)
        print("labels_candidates =", self.labels_candidates)
        MyLog.printm("preferences_ut =", self.preferences_ut)
        MyLog.printm("preferences_borda_ut =", self.preferences_borda_ut)
        MyLog.printm("preferences_borda_rk =", self.preferences_borda_rk)
        MyLog.printm("preferences_rk =", self.preferences_rk)
    
        MyLog.printm("v_has_same_ordinal_preferences_as_previous_voter =",
                 self.v_has_same_ordinal_preferences_as_previous_voter)
    
        MyLog.print_title("Plurality scores")
        MyLog.printm("preferences_rk (reminder) =",
                     self.preferences_rk)
        print("plurality_scores_rk =", self.plurality_scores_rk)
        print("")
        MyLog.printm("preferences_borda_ut (reminder) =",
               self.preferences_borda_ut)
        print("plurality_scores_ut =", self.plurality_scores_ut)
    
        MyLog.print_title("Borda scores")
        MyLog.printm("preferences_borda_rk (reminder) =",
               self.preferences_borda_rk)
        MyLog.printm("borda_score_c_rk =", self.borda_score_c_rk)
        print("Remark: Borda scores above are computed with the "
              "matrix of duels.")
        MyLog.printm("Check: np.sum(self.preferences_borda_rk, 0) =",
              np.sum(self.preferences_borda_rk, 0))
        MyLog.printm("decreasing_borda_scores_rk =",
               self.decreasing_borda_scores_rk)
        MyLog.printm("candidates_by_decreasing_borda_score_rk =",
                 self.candidates_by_decreasing_borda_score_rk)
        print("")
        MyLog.printm("preferences_borda_ut (reminder) =",
                 self.preferences_borda_ut)
        MyLog.printm("borda_score_c_ut =", self.borda_score_c_ut)
        MyLog.printm("decreasing_borda_scores_ut =",
                 self.decreasing_borda_scores_ut)
        MyLog.printm("candidates_by_decreasing_borda_score_ut =",
                 self.candidates_by_decreasing_borda_score_ut)

        MyLog.print_title("Utilities")
        MyLog.printm("preferences_ut (reminder) =",
               self.preferences_ut)
        MyLog.printm("total_utility_c = ", self.total_utility_c)
        print("total_utility_min =", self.total_utility_min)
        print("total_utility_max =", self.total_utility_max)
        print("total_utility_mean =", self.total_utility_mean)
        print("total_utility_std =", self.total_utility_std)

        MyLog.print_title("Condorcet notions based on rankings")
        MyLog.printm("preferences_rk (reminder) =", self.preferences_rk)
        MyLog.printm("matrix_duels_rk =", self.matrix_duels_rk)
    
        MyLog.printm("matrix_victories_rk =", self.matrix_victories_rk)
        print("condorcet_winner_rk =", self.condorcet_winner_rk)
    
        MyLog.printm("matrix_victories_rk_ctb =", self.matrix_victories_rk_ctb)
        print("condorcet_winner_rk_ctb =", self.condorcet_winner_rk_ctb)
    
        MyLog.print_title("Relative Condorcet notions (ut)")
        MyLog.printm("preferences_borda_ut (reminder) =",
                 self.preferences_borda_ut)
        MyLog.printm("matrix_duels_ut =", self.matrix_duels_ut)
    
        MyLog.printm("matrix_victories_ut_rel =", self.matrix_victories_ut_rel)
        print("condorcet_winner_ut_rel =", self.condorcet_winner_ut_rel)
    
        MyLog.printm("matrix_victories_ut_rel_ctb =", self.matrix_victories_ut_rel_ctb)
        print("condorcet_winner_ut_rel_ctb =", self.condorcet_winner_ut_rel_ctb)
    
        MyLog.print_title("Absolute Condorcet notions (ut)")
        MyLog.printm("matrix_duels_ut (reminder) =", self.matrix_duels_ut)
        MyLog.printm("matrix_victories_ut_abs =", self.matrix_victories_ut_abs)
        MyLog.printm("condorcet_admissible_candidates = ",
              self.condorcet_admissible_candidates)
        print("nb_condorcet_admissible =", self.nb_condorcet_admissible)
        MyLog.printm("weak_condorcet_winners =", self.weak_condorcet_winners)
        print("nb_weak_condorcet_winners =", self.nb_weak_condorcet_winners)
        print("condorcet_winner_ut_abs =", self.condorcet_winner_ut_abs)
        print("resistant_condorcet_winner =", self.resistant_condorcet_winner)
        MyLog.printm("threshold_c_prevents_w_Condorcet_ut_abs =",
                 self.threshold_c_prevents_w_Condorcet_ut_abs)

        MyLog.printm("matrix_victories_ut_abs_ctb =", self.matrix_victories_ut_abs_ctb)
        print("condorcet_winner_ut_abs_ctb =", self.condorcet_winner_ut_abs_ctb)

        MyLog.print_title("Implications between Condorcet notions")
        # Resistant Condorcet (False)
        #  ||
        #  V
        # Condorcet_ut_abs (False)       ==>      Condorcet_ut_abs_ctb (False)
        #  ||          ||                                     ||           ||
        #  ||          V                                      V            ||
        #  ||       Condorcet_rk (False) ==> Condorcet_rk_ctb (False)      ||
        #  V                                                               V
        # Condorcet_ut_rel (False)       ==>      Condorcet_ut_rel_ctb (False)
        #  ||
        #  V
        # Weak Condorcet (False)
        #  ||
        #  V
        # Condorcet-admissible (False)
        def display_bool(value):
            if value == True:
                return '(True) '
            else:
                return '(False)'
        print('Resistant Condorcet ' +
              display_bool(self.exists_resistant_condorcet_winner))
        print(' ||\n V')
        print('Condorcet_ut_abs ' + display_bool(
            self.exists_condorcet_winner_ut_abs) +
              '       ==>      Condorcet_ut_abs_ctb ' +
              display_bool(self.exists_condorcet_winner_ut_abs_ctb))
        print(' ||          ||                  '
              '                   ||           ||')
        print(' ||          V                   '
              '                   V            ||')
        print(' ||       Condorcet_rk ' +
              display_bool(self.exists_condorcet_winner_rk) +
              ' ==> Condorcet_rk_ctb ' +
              display_bool(self.exists_condorcet_winner_rk_ctb) +
              '      ||')
        print(' V                               '
              '                                V')
        print('Condorcet_ut_rel ' +
              display_bool(self.exists_condorcet_winner_ut_rel) +
              '       ==>      Condorcet_ut_rel_ctb ' +
              display_bool(self.exists_condorcet_winner_ut_rel_ctb))
        print(' ||')
        print(' V')
        print('Weak Condorcet ' +
              display_bool(self.exists_weak_condorcet_winner))
        print(' ||')
        print(' V')
        print('Condorcet-admissible ' +
              display_bool(self.exists_condorcet_admissible))

        self._log_depth = old_log_depth


def preferences_ut_to_preferences_rk(preferences_ut):
    """Convert utilities to rankings.

    Arguments:
    preferences_ut -- 2d array of floats.
        preferences_ut[v, c] is the utility of candidate c as
        seen by voter v.

    Returns:
    preferences_rk -- 2d array of integers. preferences_rk[v, k]
        is the candidate at rank k for voter v.

    If preferences_ut[v,c] == preferences_ut[v,d], then it is
    drawn at random whether c prefers c to d or d to c.
    """
    V, C = preferences_ut.shape
    tiebreaker = np.random.rand(V, C)
    return np.lexsort((tiebreaker, -preferences_ut), 1)


def preferences_rk_to_preferences_borda_rk(preferences_rk):
    """Convert rankings to Borda scores (with voter tie-breaking).

    Arguments:
    preferences_rk -- 2d array of integers. preferences_rk[v, k]
        is the candidate at rank k for voter v.

    Returns:
    preferences_borda_rk -- 2d array of integers.
        preferences_borda_rk[v, c] is the Borda score (between 0 and C - 1)
        of candidate c for voter v.
    """
    _, C = preferences_rk.shape
    return C - 1 - np.argsort(preferences_rk, 1)


def preferences_ut_to_preferences_borda_ut(preferences_ut):
    """Convert utilities to Borda scores, with equalities.

    Arguments:
    preferences_ut -- 2d array of floats.
        preferences_ut[v, c] is the utility of candidate c as
        seen by voter v.

    Returns:
    preferences_borda_ut -- 2d array of integers.
        preferences_borda_rk[v, c] gains 1 point for each d such that v
        prefers c to d, and 0.5 point for each d such that v is
        indifferent between c and d.
    """
    V, C = preferences_ut.shape
    preference_borda_ut = np.zeros((V, C))
    for c in range(C):
        preference_borda_ut[:, c] = np.sum(
            0.5 * (preferences_ut[:, c][:, np.newaxis] >=
                   preferences_ut) +
            0.5 * (preferences_ut[:, c][:, np.newaxis] >
                   preferences_ut),
            1) - 0.5
    return preference_borda_ut


def preferences_ut_to_matrix_duels_ut(preferences_ut):
    """Compute the matrix of duels.

    Arguments:
    preferences_ut -- 2d array of floats.
        preferences_ut[v, c] is the utility of candidate c as
        seen by voter v.

    Returns:
    matrix_duels_ut -- 2d array of integers.
        matrix_duels_ut[c, d] is the number of voters who strictly prefer
        candidate c to d. By convention, diagonal coefficients are set to
        0.
    """
    n, m = preferences_ut.shape
    matrix_duels = np.zeros((m, m), dtype=np.int)
    for c in range(m):
        for d in range(c + 1, m):
            matrix_duels[c, d] = np.sum(preferences_ut[:, c] >
                                        preferences_ut[:, d])
            matrix_duels[d, c] = np.sum(preferences_ut[:, d] >
                                        preferences_ut[:, c])
    return matrix_duels


def is_resistant_condorcet(w, preferences_ut):
    """Test for Resistant Condorcet winner.

    Arguments:
    w -- Integer (candidate). For compatibility reasons, NaN is allowed.
    preferences_ut -- 2d array of floats.
        preferences_ut[v, c] is the utility of candidate c as
        seen by voter v.

    Returns:
    is_resistant -- Boolean. If w is a Resistant Condorcet Winner, then
        is_resistant = True. Otherwise (or if w = NaN), then
        is_resistant = False.

    A Condorcet winner w is resistant iff in any Condorcet voting system,
    the profile is not manipulable (cf. Durand et al. 2014). This is
    equivalent to say that for any pair (c, d) of other
    candidates, there is a strict majority of voters who simultaneously:
    * Do not prefer c to w
    * And prefer w to d.
    """
    if np.isnan(w):
        return False
    V, C = preferences_ut.shape
    for c in range(C):
        if c == w:
            continue
        v_does_not_prefer_c_to_w = (preferences_ut[:, w] >=
                                    preferences_ut[:, c])
        for d in range(C):
            if d == w:
                continue
            v_prefers_w_to_d = (preferences_ut[:, w] >
                                preferences_ut[:, d])
            if np.sum(np.logical_and(v_does_not_prefer_c_to_w,
                                     v_prefers_w_to_d)) <= V / 2:
                return False
    return True


# def compute_condorcet_quick(preferences_ut):
#     """Compute Condorcet winner.
#
#     Arguments:
#     preferences_ut -- 2d array of floats.
#         preferences_ut[v, c] is the utility of candidate c as
#         seen by voter v.
#
#     Returns:
#     condorcet_winner_ut_abs_ctb -- Integer or NaN. 'tb' stands for 'ties
# broken'.
#         Candidate who has only victories in matrix_victories_ctb.
#         If there is no such candidate, then NaN.
#     condorcet_winner_ut_abs -- Integer or NaN. Candidate who has only victories
#         in matrix_victories. If there is no such candidate, then NaN.
#     """
#     n, m = preferences_ut.shape
#
#     # Step 1 : we move forward in the list of candidates, always keeping
#     # the winner
#     w_provisional = 0
#     has_used_tie_break = False
#     for d in range(1, m):
#         voters_d_vs_w = sum(preferences_ut[:, d] >
#                             preferences_ut[:, w_provisional])
#         if voters_d_vs_w > n / 2:
#             w_provisional = d
#             has_used_tie_break = False  # We do not know a tie for d
#         elif voters_d_vs_w == n / 2:
#             # Still ok for w, but used tie-break
#             has_used_tie_break = True
#
#
#     # Step 2 : We know that w_provisional wins versus all those after
#     # her. But does she win against those before her?
#     for c in range(0, w_provisional):
#         # Since c < w, c needs only some >= and not some > to win.
#         if sum(preferences_ut[:, c] >=
#                 preferences_ut[:, w_provisional]) >= n / 2:
#             return np.nan, np.nan
#     condorcet_winner_ut_abs_ctb = w_provisional
#     if has_used_tie_break:
#         condorcet_winner_ut_abs = np.nan
#     else:
#         condorcet_winner_ut_abs = condorcet_winner_ut_abs_ctb
#     return condorcet_winner_ut_abs_ctb, condorcet_winner_ut_abs


if __name__ == '__main__':
    # A quick demo
    preferences_ut = np.random.randint(-5, 5, (10, 5))
    pop = Population(preferences_ut=preferences_ut,
                     labels_candidates=['Adélaïde', 'Bartholomé', 'Cunégonde',
                                        'Dagobert', 'Eugénie'])
    pop.demo(log_depth=1)
    pop.plot3(indexes=[0,1,2], normalize=True)
    pop.plot4(indexes=[0,1,2,3], normalize=True)
