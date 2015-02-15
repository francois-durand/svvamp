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
    """Population of voters."""

    def __init__(self, preferences_utilities=None, preferences_ranking=None,
                 log_creation=None, labels_candidates=None):
        """Create a population with preferences. 

        :param preferences_utilities: 2d array of floats.
            ``preferences_utilities[v, c]`` is the utility of candidate ``c``
            as seen by voter ``v``.
        :param preferences_ranking: 2d array of integers.
            ``preferences_ranking[v, k]`` is the candidate at rank ``k`` for
            voter ``v``.
        :param log_creation: Any type (string, list...). Some comments.
        :param labels_candidates: List of strings. Names of the candidates.

        You may enter either ``preferences_utilities`` or
        ``preferences_ranking`` to define the preferences of the population.
        If you provide both, then only ``preferences_utilities`` is used.
        
        If voter ``v`` attributes the same utility to several candidates:

            * The first time the attribute ``preferences_ranking`` is
              called, a random ranking will be decided for these tied
              candidates (once and for all). This strict ranking will be used
              for sincere voting in all voting systems accepting only strict
              orders. This process is called **voter tie-breaking** or **VTB**.
            * However, for manipulation purposes, indifference is taken
              into account as such. If voter ``v`` attributes the same
              utility to candidates ``w`` and ``c``, and if ``w`` is the
              sincere winner of the election, then ``v`` is not interested in
              a manipulation in favor of ``c``.

        If you provide ``preferences_ranking`` only,
        then ``preferences_utilities`` is set to the corresponding Borda
        scores. Cf. :attr:`svvamp.Population.borda_score_c_vtb`.

        If all voters have a strict order of preference, either because you
        provided utilities without ties for each voter, or because you
        provided preference rankings only, then VTB does not matter. In
        that case, each function without VTB below is equivalent to its
        variant with VTB.

        In some voting systems and in some of the attributes below,
        we use a process referred as **candidate tie-breaking** or **CTB** in
        SVVAMP. It means that lowest-index candidates are favored. If
        candidates ``c`` and ``d`` are tied (for example, in an election
        using Plurality), then when using CTB, ``c`` is favored over ``d``
        iff ``c < d``.

        Implications between majority favorite and Condorcet criteria (cf.
        corresponding functions below).

        ::

            majority_favorite             ==>            majority_favorite_ctb
            ||              ||                                ||            ||
            V               ||                                ||            ||
            Resistant Cond. V                                 V             ||
            ||      majority_favorite_vtb ==> majority_favorite_vtb_ctb     ||
            ||                     ||                   ||                  ||
            V                      ||                   ||                  V
            Condorcet                     ==>                    Condorcet_ctb
            ||      ||             ||                   ||         ||       ||
            ||      V              V                    V          V        ||
            V         Condorcet_vtb       ==>         Condorcet_vtb_ctb     V
            Condorcet_rel                 ==>                Condorcet_rel_ctb
            ||
            V
            Weak Condorcet
            ||
            V
            Condorcet-admissible

        If all voters have strict orders of preference and if there is an
        odd number of voters, then:

            * ``majority_favorite``, ``majority_favorite_vtb``,
              ``majority_favorite_ctb`` and ``majority_favorite_vtb_ctb``
              are equivalent,
            * ``Condorcet``, ``Condorcet_ctb``, ``Condorcet_vtb``,
              ``Condorcet_vtb_ctb``, ``Condorcet_rel``, ``Condorcet_rel_ctb``,
              ``Weak Condorcet`` and ``Condorcet-admissible`` are equivalent.
        """
        super().__init__()
        self._log_identity = "POPULATION"

        # Basic variables
        if preferences_utilities is not None:
            self._preferences_utilities = np.array(preferences_utilities)
            self._preferences_ranking = None
            self._V, self._C = self._preferences_utilities.shape
        else:
            self._preferences_utilities = None
            self._preferences_ranking = np.array(preferences_ranking)
            self._V, self._C = self._preferences_ranking.shape
        self._labels_candidates = labels_candidates
        # Missing matrices will be computed this way (on demand):
        # If preferences_utilities is provided:
        #       utilities -> rankings  -> borda_vtb
        # If preferences_ranking is provided:
        #       rankings  -> borda_vtb -> utilities

        self.log_creation = log_creation
        if self.V < 2 or self.C < 2:
            raise ValueError("A population must have at least 2 voters and "
                             "2 candidates.")

        # Other variables
        self._preferences_borda_vtb = None
        self._preferences_borda_novtb = None
        self._voters_sorted_by_ranking = False
        self._v_has_same_ordinal_preferences_as_previous_voter = None
        self._matrix_duels = None
        self._matrix_duels_vtb = None
        self._matrix_victories_vtb = None
        self._matrix_victories_vtb_ctb = None
        self._matrix_victories_rel = None
        self._matrix_victories_rel_ctb = None
        self._matrix_victories_abs = None
        self._matrix_victories_abs_vtb = None
        self._condorcet_admissible_candidates = None
        self._nb_condorcet_admissible_candidates = None
        self._weak_condorcet_winners = None
        self._nb_weak_condorcet_winners = None
        self._condorcet_winner_vtb_ctb = None
        self._condorcet_winner_vtb = None
        self._condorcet_winner_rel_ctb = None
        self._condorcet_winner_rel = None
        self._condorcet_winner_ctb = None
        self._condorcet_winner = None
        self._resistant_condorcet_winner = None
        self._threshold_c_prevents_w_Condorcet = None
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
        self._plurality_scores_vtb = None
        self._plurality_scores_novtb = None
        self._borda_score_c_vtb = None
        self._decreasing_borda_scores_vtb = None
        self._candidates_by_decreasing_borda_score_vtb = None
        self._borda_score_c_novtb = None
        self._decreasing_borda_scores_novtb = None
        self._candidates_by_decreasing_borda_score_novtb = None
        self._eb_result = None  # Storage for Exhaustive Ballot
        self._eb_manip = None  # Storage for Exhaustive Ballot
        self._eb_options = None  # Storage for Exhaustive Ballot
        self._irv_manip = None  # Storage for IRV
        self._irv_options = None  # Storage for IRV

    # %% Basic variables

    @property
    def labels_candidates(self):
        """List of ``C`` strings (names of the candidates)."""
        if self._labels_candidates is None:
            self._labels_candidates = [str(x) for x in range(self._C)]
        return self._labels_candidates

    @property
    def C(self):
        """Integer (number of candidates)."""
        return self._C

    @property
    def V(self):
        """Integer (number of voters)."""
        return self._V

    @property
    def preferences_utilities(self):
        """2d array of floats. ``preferences_utilities[v, c]`` is the utility
        of  candidate ``c`` as seen by voter ``v``.
        """
        if self._preferences_utilities is None:
            self._mylog("Compute preference utilities", 1)
            self._preferences_utilities = self.preferences_borda_vtb
        return self._preferences_utilities

    @property
    def preferences_ranking(self):
        """2d array of integers. ``preferences_ranking[v, k]`` is the
        candidate at rank k for voter v. For example,
        ``preferences_ranking[v, 0]`` is ``v``'s preferred candidate.
        """
        if self._preferences_ranking is None:
            self._mylog("Compute preference rankings", 1)
            self._preferences_ranking = \
                preferences_utilities_to_preferences_ranking(
                    self._preferences_utilities)
        return self._preferences_ranking

    @property
    def preferences_borda_vtb(self):
        """2d array of integers. ``preferences_borda_vtb[v, c]`` is the Borda
        score (between ``0`` and ``C - 1``) of candidate ``c`` for voter ``v``,
        with voter tie-breaking.
        """
        if self._preferences_borda_vtb is None:
            self._mylog("Compute preferences in Borda format with vtb", 1)
            self._preferences_borda_vtb = \
                preferences_ranking_to_preferences_borda_vtb(
                    self.preferences_ranking)
        return self._preferences_borda_vtb

    @property
    def preferences_borda_novtb(self):
        """2d array of integers. ``preferences_borda_novtb[v, c]`` gains 1
        point for each ``d`` such that ``v`` prefers ``c`` to ``d``, and
        0.5 point for each ``d`` such that ``v`` is indifferent between ``c``
        and ``d``.
        """
        if self._preferences_borda_novtb is None:
            self._mylog("Compute preference weak orders in Borda format", 1)
            self._preferences_borda_novtb = \
                preferences_utilities_to_preferences_borda_novtb(
                    self.preferences_utilities)
        return self._preferences_borda_novtb

    #%% Sort voters

    def ensure_voters_sorted_by_ordinal_preferences(self):
        """Ensure that voters are sorted.
        
        If needed, sort voters first by their strict order of preference
        (row in ``preferences_ranking``), then by their weak order of
        preference (row in ``preferences_borda_novtb``). Note that two voters
        having the same strict order may have different weak orders,
        and vice versa.

        This method will be called automatically when creating an election, 
        because it allows to accelerate some algorithms (typically  Individual 
        Manipulation).
        """
        if self._voters_sorted_by_ranking:
            return
        self._voters_sorted_by_ranking = True
        self._v_has_same_ordinal_preferences_as_previous_voter = None
        self._mylog('Sort voters by ranking', 1)
        # Ensure that tables are computed beforehand
        _ = self.preferences_ranking
        _ = self.preferences_utilities
        _ = self.preferences_borda_vtb
        _ = self.preferences_borda_novtb
        # Sort by weak order
        list_borda_novtb = self.preferences_borda_novtb.tolist()
        indexes = sorted(range(len(list_borda_novtb)),
                         key=list_borda_novtb.__getitem__)
        self._preferences_ranking = self.preferences_ranking[indexes, ::]
        self._preferences_utilities = self.preferences_utilities[indexes, ::]
        self._preferences_borda_vtb = self.preferences_borda_vtb[indexes, ::]
        self._preferences_borda_novtb = self.preferences_borda_novtb[indexes,
                                                                     ::]
        # Sort by strict ranking
        list_rankings = self.preferences_ranking.tolist()
        indexes = sorted(range(len(list_rankings)),
                         key=list_rankings.__getitem__)
        self._preferences_ranking = self.preferences_ranking[indexes, ::]
        self._preferences_utilities = self.preferences_utilities[indexes, ::]
        self._preferences_borda_vtb = self.preferences_borda_vtb[indexes, ::]
        self._preferences_borda_novtb = self.preferences_borda_novtb[indexes,
                                                                     ::]

    @property
    def v_has_same_ordinal_preferences_as_previous_voter(self):
        """1d array of booleans.
        ``v_has_same_ordinal_preferences_as_previous_voter[v]`` is
        ``True`` iff voter ``v`` has the same preference strict order (row in
        ``preferences_ranking``) and the same preference weak order (row in
        ``preferences_borda_novtb``) as voter ``v-1``.
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
                            self.preferences_ranking[range(self.V - 1), :] ==
                            self.preferences_ranking[range(1, self.V), :], 1),
                        np.all(
                            self._preferences_borda_novtb[range(self.V - 1),
                                                          :] ==
                            self._preferences_borda_novtb[range(1, self.V), :],
                            1)
                    )
                ))
        return self._v_has_same_ordinal_preferences_as_previous_voter

    #%% Plurality scores

    @property
    def plurality_scores_vtb(self):
        """1d array of booleans. ``plurality_scores_vtb[c]`` is the number of
        voters for whom ``c`` is the top-ranked candidate (with voter
        tie-breaking).
        """
        if self._plurality_scores_vtb is None:
            self._mylog("Compute Plurality scores (with vtb)", 1)
            self._plurality_scores_vtb = np.bincount(
                self.preferences_ranking[:, 0],
                minlength=self.C
            )
        return self._plurality_scores_vtb

    @property
    def plurality_scores_novtb(self):
        """1d array of booleans. ``plurality_scores_novtb[c]`` is the number of
        voters who strictly prefer ``c`` to all other candidates (with no VTB,
        i.e. ex-aequo top-ranked candidates are not counted).
        """
        if self._plurality_scores_novtb is None:
            self._mylog("Compute Plurality scores (with no vtb)", 1)
            self._plurality_scores_novtb = np.zeros(self.C)
            for v in range(self.V):
                c = np.argmax(self.preferences_utilities[v, :])
                if np.all(np.greater(
                    self.preferences_utilities[v, c],
                    self.preferences_utilities[v, np.array(range(self.C)) != c]
                )):
                    self._plurality_scores_novtb[c] += 1
        return self._plurality_scores_novtb

    #%% Matrix of duels

    @property
    def matrix_duels(self):
        """2d array of integers. ``matrix_duels[c, d]`` is the number of voters
        who strictly prefer candidate ``c`` to ``d`` (using
        :attr:`svvamp.Population.preferences_borda_novtb`, i.e indifference
        is allowed). By convention, diagonal coefficients are set to ``0``.
        """
        if self._matrix_duels is None:
            self._mylog("Compute matrix of duels", 1)
            self._matrix_duels = \
                preferences_utilities_to_matrix_duels(
                    self.preferences_borda_novtb)
        return self._matrix_duels

    @property
    def matrix_duels_vtb(self):
        """2d array of integers. ``matrix_duels_vtb[c, d]`` is the number of
        voters who strictly prefer candidate ``c`` to ``d``, or who break a
        tie in favor of c over d (so we use
        :attr:`svvamp.Population.preferences_borda_vtb`). By convention,
        diagonal coefficients are set to 0.
        """
        if self._matrix_duels_vtb is None:
            self._mylog("Compute matrix of duels (with strict orders)", 1)
            self._matrix_duels_vtb = \
                preferences_utilities_to_matrix_duels(
                    self.preferences_borda_vtb)
        return self._matrix_duels_vtb

    @property
    def matrix_victories_abs(self):
        """2d array of values in {0, 0.5, 1}. Matrix of absolute victories
        based on :attr:`svvamp.Population.matrix_duels`.

        ``matrix_victories_abs[c, d]`` is:

            * 1   iff ``matrix_duels[c, d] > V / 2``.
            * 0.5 iff ``matrix_duels[c, d] = V / 2``.
            * 0   iff ``matrix_duels[c, d] < V / 2``.

        By convention, diagonal coefficients are set to 0.
        """
        if self._matrix_victories_abs is None:
            self._mylog("Compute matrix_victories_abs", 1)
            self._matrix_victories_abs = (
                np.multiply(0.5, self.matrix_duels >= self.V / 2) +
                np.multiply(0.5, self.matrix_duels > self.V / 2)
            )
        return self._matrix_victories_abs

    @property
    def matrix_victories_abs_ctb(self):
        """2d array of values in {0, 1}. Matrix of absolute victories
        based on :attr:`svvamp.Population.matrix_duels`, with tie-breaks on
        candidates.

        ``matrix_victories_abs_ctb[c, d]`` is:

            * 1   iff ``matrix_duels[c, d] > V / 2``, or
              ``matrix_duels[c, d] = V / 2`` and ``c < d``.
            * 0   iff ``matrix_duels[c, d] < V / 2``, or
              ``matrix_duels[c, d] = V / 2`` and ``d < c``.

        By convention, diagonal coefficients are set to 0.
        """
        if self._matrix_victories_abs_vtb is None:
            self._mylog("Compute matrix_victories_abs_ctb", 1)
            self._matrix_victories_abs_vtb = np.zeros((self.C, self.C))
            for c in range(self.C):
                for d in range(self.C):
                    if c == d:
                        continue
                    if self.matrix_duels[c, d] > self.V / 2 or (
                            self.matrix_duels[c, d] == self.V / 2 and c < d):
                        self._matrix_victories_abs_vtb[c, d] = 1
        return self._matrix_victories_abs_vtb

    @property
    def matrix_victories_rel(self):
        """2d array of values in {0, 0.5, 1}. Matrix of relative victories
        based on :attr:`svvamp.Population.matrix_duels`.

        ``matrix_victories_rel[c, d]`` is:

            * 1   iff ``matrix_duels[c, d] > matrix_duels[d, c]``.
            * 0.5 iff ``matrix_duels[c, d] = matrix_duels[d, c]``.
            * 0   iff ``matrix_duels[c, d] < matrix_duels[d, c]``.

        By convention, diagonal coefficients are set to 0.
        """
        if self._matrix_victories_rel is None:
            self._mylog("Compute matrix_victories_rel", 1)
            self._matrix_victories_rel = (
                np.multiply(0.5, self.matrix_duels >= self.matrix_duels.T) +
                np.multiply(0.5, self.matrix_duels > self.matrix_duels.T) -
                np.multiply(0.5, np.eye(self.C))
            )
        return self._matrix_victories_rel

    @property
    def matrix_victories_rel_ctb(self):
        """2d array of values in {0, 1}. Matrix of relative victories
        based on :attr:`svvamp.Population.matrix_duels`, with tie-breaks on
        candidates.

        ``matrix_victories_rel_ctb[c, d]`` is:

            * 1   iff ``matrix_duels[c, d] > matrix_duels[d, c]``, or
              ``matrix_duels[c, d] = matrix_duels[d, c]`` and ``c < d``.
            * 0   iff ``matrix_duels[c, d] < matrix_duels[d, c]``, or
              ``matrix_duels[c, d] = matrix_duels[d, c]`` and ``d < c``.

        By convention, diagonal coefficients are set to 0.
        """
        if self._matrix_victories_rel_ctb is None:
            self._mylog("Compute matrix_victories_rel_ctb", 1)
            self._matrix_victories_rel_ctb = np.zeros((self.C, self.C))
            for c in range(self.C):
                for d in range(c + 1, self.C):
                    if self.matrix_duels[c, d] >= self.matrix_duels[d, c]:
                        self._matrix_victories_rel_ctb[c, d] = 1
                        self._matrix_victories_rel_ctb[d, c] = 0
                    else:
                        self._matrix_victories_rel_ctb[c, d] = 0
                        self._matrix_victories_rel_ctb[d, c] = 1
        return self._matrix_victories_rel_ctb

    @property
    def matrix_victories_vtb(self):
        """2d array of values in {0, 0.5, 1}. Matrix of victories
        based on :attr:`svvamp.Population.matrix_duels_vtb` (i.e. each voter
        breaks her own ties).

        ``matrix_victories_vtb[c, d]`` is:

            * 1   iff ``matrix_duels_vtb[c, d] > matrix_duels_vtb[d, c]``,
              i.e. iff ``matrix_duels_vtb[c, d] > V / 2``.
            * 0.5 iff ``matrix_duels_vtb[c, d] = matrix_duels_vtb[d, c]``,
              i.e. iff ``matrix_duels_vtb[c, d] = V / 2``.
            * 0   iff ``matrix_duels_vtb[c, d] < matrix_duels_vtb[d, c]``,
              i.e. iff ``matrix_duels_vtb[c, d] < V / 2``.

        By convention, diagonal coefficients are set to 0.
        """
        if self._matrix_victories_vtb is None:
            self._mylog("Compute matrix_victories_vtb", 1)
            self._matrix_victories_vtb = (
                np.multiply(0.5, self.matrix_duels_vtb >= self.V / 2) +
                np.multiply(0.5, self.matrix_duels_vtb > self.V / 2)
            )
        return self._matrix_victories_vtb

    @property
    def matrix_victories_vtb_ctb(self):
        """2d array of values in {0, 1}. Matrix of victories based on
        :attr:`svvamp.Population.matrix_duels_vtb` (i.e. each voter breaks
        her own ties), with tie-breaks on candidates.

        ``matrix_victories_vtb_ctb[c, d]`` is:

            * 1   iff ``matrix_duels_vtb[c, d] > matrix_duels_vtb[d, c]``, or
              ``matrix_duels_vtb[c, d] = matrix_duels_vtb[d, c]`` and
              ``c < d``.
            * 0   iff ``matrix_duels_vtb[c, d] < matrix_duels_vtb[d, c]``, or
              ``matrix_duels_vtb[c, d] = matrix_duels_vtb[d, c]`` and
              ``d < c``.

        By convention, diagonal coefficients are set to 0.
        """
        if self._matrix_victories_vtb_ctb is None:
            self._mylog("Compute matrix_victories_vtb_ctb", 1)
            self._matrix_victories_vtb_ctb = np.zeros((self.C, self.C))
            for c in range(self.C):
                for d in range(c + 1, self.C):
                    if self.matrix_duels_vtb[c, d] >= self.V / 2:
                        self._matrix_victories_vtb_ctb[c, d] = 1
                        self._matrix_victories_vtb_ctb[d, c] = 0
                    else:
                        self._matrix_victories_vtb_ctb[c, d] = 0
                        self._matrix_victories_vtb_ctb[d, c] = 1
        return self._matrix_victories_vtb_ctb

    #%% Condorcet winner and variants

    @property
    def condorcet_admissible_candidates(self):
        """1d array of booleans. ``condorcet_admissible_candidates[c]`` is
        ``True`` iff candidate ``c`` is Condorcet-admissible, i.e. iff no
        candidate ``d`` has an absolute victory against ``c`` (in
        the sense of :attr:`svvamp.Population.matrix_victories_abs`).

        .. seealso:: :attr:`svvamp.Population.nb_condorcet_admissible`,
                     :attr:`svvamp.Population.exists_condorcet_admissible`,
                     :attr:`svvamp.Population.not_exists_condorcet_admissible`.
        """
        if self._condorcet_admissible_candidates is None:
            self._mylog("Compute Condorcet-admissible candidates", 1)
            self._condorcet_admissible_candidates = np.all(
                self.matrix_victories_abs <= 0.5, 0)
        return self._condorcet_admissible_candidates

    @property
    def nb_condorcet_admissible(self):
        """Integer (number of Condorcet-admissible candidates).

        .. seealso:: :attr:`svvamp.Population.condorcet_admissible_candidates`.
        """
        if self._nb_condorcet_admissible_candidates is None:
            self._mylog("Compute number of Condorcet-admissible candidates", 1)
            self._nb_condorcet_admissible_candidates = np.sum(
                self.condorcet_admissible_candidates)
        return self._nb_condorcet_admissible_candidates

    @property
    def exists_condorcet_admissible(self):
        """Boolean (``True`` iff there is at least one Condorcet-admissible
        candidate).

        .. seealso:: :attr:`svvamp.Population.condorcet_admissible_candidates`
        """
        return self.nb_condorcet_admissible > 0

    @property
    def not_exists_condorcet_admissible(self):
        """Boolean (``True`` iff there is no Condorcet-admissible candidate).

        .. seealso:: :attr:`svvamp.Population.condorcet_admissible_candidates`
        """
        return self.nb_condorcet_admissible == 0

    @property
    def weak_condorcet_winners(self):
        """1d array of booleans. ``weak_condorcet_winners[c]`` is ``True`` iff
        candidate ``c`` is a weak Condorcet winner, i.e. iff no candidate
        ``d`` has a relative victory against ``c`` (using
        :attr:`svvamp.Population.matrix_victories_rel`).
        """
        if self._weak_condorcet_winners is None:
            self._mylog("Compute weak Condorcet winners", 1)
            self._weak_condorcet_winners = np.all(
                self.matrix_victories_rel <= 0.5, 0)
        return self._weak_condorcet_winners

    @property
    def nb_weak_condorcet_winners(self):
        """Integer (number of weak Condorcet winners).

        .. seealso:: :attr:`svvamp.Population.weak_condorcet_winners`
        """
        if self._nb_weak_condorcet_winners is None:
            self._mylog("Compute number of weak Condorcet winners", 1)
            self._nb_weak_condorcet_winners = np.sum(
                self.weak_condorcet_winners)
        return self._nb_weak_condorcet_winners

    @property
    def exists_weak_condorcet_winner(self):
        """Boolean (``True`` iff there is at least one weak Condorcet winner).

        .. seealso:: :attr:`svvamp.Population.weak_condorcet_winners`
        """
        return self.nb_weak_condorcet_winners > 0

    @property
    def not_exists_weak_condorcet_winner(self):
        """Boolean (``True`` iff there is no weak Condorcet winner).

        .. seealso:: :attr:`svvamp.Population.weak_condorcet_winners`
        """
        return self.nb_weak_condorcet_winners == 0

    @property
    def condorcet_winner_vtb_ctb(self):
        """Integer or ``NaN``. Candidate who has only victories in the sense of
        :attr:`svvamp.Population.matrix_victories_vtb_ctb`. If there is no
        such candidate, then ``NaN``.

        .. seealso::

            :attr:`svvamp.Population.exists_condorcet_winner_vtb_ctb`,
            :attr:`svvamp.Population.not_exists_condorcet_winner_vtb_ctb`.
        """
        if self._condorcet_winner_vtb_ctb is None:
            self._mylog("Compute condorcet_winner_vtb_ctb", 1)
            for c in range(self.C):
                # The whole COLUMN must be 0.
                if np.array_equiv(self.matrix_victories_vtb_ctb[:, c], 0):
                    self._condorcet_winner_vtb_ctb = c
                    break
            else:
                self._condorcet_winner_vtb_ctb = np.nan
        return self._condorcet_winner_vtb_ctb

    @property
    def exists_condorcet_winner_vtb_ctb(self):
        """Boolean (``True`` iff there is a Condorcet winner with vtb and ctb).

        .. seealso:: :attr:`svvamp.Population.condorcet_winner_vtb_ctb`
        """
        return not np.isnan(self.condorcet_winner_vtb_ctb)

    @property
    def not_exists_condorcet_winner_vtb_ctb(self):
        """Boolean (``True`` iff there is no Condorcet winner with vtb and ctb).

        .. seealso:: :attr:`svvamp.Population.condorcet_winner_vtb_ctb`
        """
        return np.isnan(self.condorcet_winner_vtb_ctb)

    @property
    def condorcet_winner_vtb(self):
        """Integer or ``NaN``. Candidate who has only victories in the sense of
        :attr:`svvamp.Population.matrix_victories_vtb`. If there is no such
        candidate, then ``NaN``.

        .. seealso:: :attr:`svvamp.Population.exists_condorcet_winner_vtb`,
                     :attr:`svvamp.Population.not_exists_condorcet_winner_vtb`.
        """
        if self._condorcet_winner_vtb is None:
            self._mylog("Compute condorcet_winner_vtb", 1)
            for c in range(self.C):
                # The whole COLUMN must be 0.
                if np.array_equiv(self.matrix_victories_vtb[:, c], 0):
                    self._condorcet_winner_vtb = c
                    break
            else:
                self._condorcet_winner_vtb = np.nan
        return self._condorcet_winner_vtb

    @property
    def exists_condorcet_winner_vtb(self):
        """Boolean (``True`` iff there is a Condorcet winner with vtb).

        .. seealso:: :attr:`svvamp.Population.condorcet_winner_vtb`
        """
        return not np.isnan(self.condorcet_winner_vtb)

    @property
    def not_exists_condorcet_winner_vtb(self):
        """Boolean (``True`` iff there is no Condorcet winner with vtb).

        .. seealso:: :attr:`svvamp.Population.condorcet_winner_vtb`
        """
        return np.isnan(self.condorcet_winner_vtb)

    @property
    def condorcet_winner_rel_ctb(self):
        """Integer or ``NaN``.
        Candidate who has only victories in the sense of
        :attr:`svvamp.Population.matrix_victories_rel_ctb`. If there is no
        such candidate, then ``NaN``.

        .. seealso::

            :attr:`svvamp.Population.exists_condorcet_winner_rel_ctb`,
            :attr:`svvamp.Population.not_exists_condorcet_winner_rel_ctb`.
        """
        if self._condorcet_winner_rel_ctb is None:
            self._mylog("Compute condorcet_winner_rel_ctb", 1)
            for c in range(self.C):
                # The whole COLUMN must be 0.
                if np.array_equiv(self.matrix_victories_rel_ctb[:, c], 0):
                    self._condorcet_winner_rel_ctb = c
                    break
            else:
                self._condorcet_winner_rel_ctb = np.nan
        return self._condorcet_winner_rel_ctb

    @property
    def exists_condorcet_winner_rel_ctb(self):
        """Boolean (``True`` iff there is a relative Condorcet winner with ctb).

        .. seealso:: :attr:`svvamp.Population.condorcet_winner_rel_ctb`
        """
        return not np.isnan(self.condorcet_winner_rel_ctb)

    @property
    def not_exists_condorcet_winner_rel_ctb(self):
        """Boolean (``True`` iff there is no relative Condorcet winner with
        ctb).

        .. seealso:: :attr:`svvamp.Population.condorcet_winner_rel_ctb`
        """
        return np.isnan(self.condorcet_winner_rel_ctb)

    @property
    def condorcet_winner_rel(self):
        """Integer or ``NaN``. Candidate who has only victories in the sense of
        :attr:`svvamp.Population.matrix_victories_rel`. If there is no such
        candidate, then ``NaN``.

        .. seealso:: :attr:`svvamp.Population.exists_condorcet_winner_rel`,
                     :attr:`svvamp.Population.not_exists_condorcet_winner_rel`.
        """
        if self._condorcet_winner_rel is None:
            self._mylog("Compute condorcet_winner_rel", 1)
            for c in range(self.C):
                # The whole COLUMN must be 0.
                if np.array_equiv(self.matrix_victories_rel[:, c], 0):
                    self._condorcet_winner_rel = c
                    break
            else:
                self._condorcet_winner_rel = np.nan
        return self._condorcet_winner_rel

    @property
    def exists_condorcet_winner_rel(self):
        """Boolean (``True`` iff there is a relative Condorcet winner).

        .. seealso:: :attr:`svvamp.Population.condorcet_winner_rel`
        """
        return not np.isnan(self.condorcet_winner_rel)

    @property
    def not_exists_condorcet_winner_rel(self):
        """Boolean (``True`` iff there is no relative Condorcet winner).

        .. seealso:: :attr:`svvamp.Population.condorcet_winner_rel`
        """
        return np.isnan(self.condorcet_winner_rel)

    @property
    def condorcet_winner_ctb(self):
        """Integer or ``NaN``.
        Candidate who has only victories in the sense of
        :attr:`svvamp.Population.matrix_victories_abs_ctb`. If there is no
        such candidate, then ``NaN``.

        .. seealso:: :attr:`svvamp.Population.exists_condorcet_winner_ctb`,
                     :attr:`svvamp.Population.not_exists_condorcet_winner_ctb`
        """
        if self._condorcet_winner_ctb is None:
            self._mylog("Compute condorcet_winner_ctb", 1)
            for c in range(self.C):
                if np.array_equiv(self.matrix_victories_abs_ctb[c,
                                  np.array(range(self.C)) != c],
                                  1):
                    self._condorcet_winner_ctb = c
                    break
            else:
                self._condorcet_winner_ctb = np.nan
        return self._condorcet_winner_ctb

    @property
    def exists_condorcet_winner_ctb(self):
        """Boolean (``True`` iff there is a Condorcet winner with ctb).

        .. seealso:: :attr:`svvamp.Population.condorcet_winner_ctb`
        """
        return not np.isnan(self.condorcet_winner_ctb)

    @property
    def not_exists_condorcet_winner_ctb(self):
        """Boolean (``True`` iff there is no Condorcet winner with ctb).

        .. seealso:: :attr:`svvamp.Population.condorcet_winner_ctb`
        """
        return np.isnan(self.condorcet_winner_ctb)

    @property
    def condorcet_winner(self):
        """Integer or ``NaN``. Candidate who has only victories in the sense of
        :attr:`svvamp.Population.matrix_victories_abs`. If there is no such
        candidate, then ``NaN``.

        .. seealso:: :attr:`svvamp.Population.exists_condorcet_winner`,
                     :attr:`svvamp.Population.not_exists_condorcet_winner`.
        """
        if self._condorcet_winner is None:
            self._mylog("Compute Condorcet winner", 1)
            for c in range(self.C):
                if np.array_equiv(self.matrix_victories_abs[c,
                                  np.array(range(self.C)) != c],
                                  1):
                    self._condorcet_winner = c
                    break
            else:
                self._condorcet_winner = np.nan
        return self._condorcet_winner

    @property
    def exists_condorcet_winner(self):
        """Boolean (``True`` iff there is a Condorcet winner).

        .. seealso:: :attr:`svvamp.Population.condorcet_winner`
        """
        return not np.isnan(self.condorcet_winner)

    @property
    def not_exists_condorcet_winner(self):
        """Boolean (``True`` iff there is no Condorcet winner).

        .. seealso:: :attr:`svvamp.Population.condorcet_winner`
        """
        return np.isnan(self.condorcet_winner)

    @property
    def resistant_condorcet_winner(self):
        """Integer or ``NaN``. Resistant Condorcet Winner. If there is no such
        candidate, then ``NaN``.
        
        A Condorcet winner ``w`` is resistant iff in any Condorcet voting
        system, the profile is not manipulable (cf. Durand et al. 2014).
        This is equivalent to say that for any pair ``(c, d)`` of other
        distinct candidates, there is a strict majority of voters who
        simultaneously do not prefer ``c`` to ``w`` and prefer ``w`` to ``d``.
        """
        if self._resistant_condorcet_winner is None:
            self._mylog("Compute Resistant Condorcet winner", 1)
            if is_resistant_condorcet(self.condorcet_winner,
                                      self.preferences_utilities):
                self._resistant_condorcet_winner = self.condorcet_winner
            else:
                self._resistant_condorcet_winner = np.nan
        return self._resistant_condorcet_winner

    @property
    def exists_resistant_condorcet_winner(self):
        """Boolean (True iff there is a resistant Condorcet winner).

        .. seealso:: :attr:`svvamp.Population.resistant_condorcet_winner`
        """
        return not np.isnan(self.resistant_condorcet_winner)

    @property
    def not_exists_resistant_condorcet_winner(self):
        """Boolean (``True`` iff there is no resistant Condorcet winner).

        .. seealso:: :attr:`svvamp.Population.resistant_condorcet_winner`
        """
        return np.isnan(self.resistant_condorcet_winner)

    @property
    def threshold_c_prevents_w_Condorcet(self):
        """2d array of integers. Threshold for ``c``-manipulators to prevent
        ``w`` from being a Condorcet winner.

        Intuitively, the question is the following: in an election where ``w``
        is the winner, how many ``c``-manipulators are needed to prevent ``w``
        from  being a Condorcet winner?
        
        We start with the sub-population of :math:`n_s` 'sincere' voters,
        i.e. not preferring ``c`` to ``w``. The precise question is: how many
        ``c``-manipulators :math:`n_m` must we add in order to create a
        non-victory for
        ``w`` against some candidate ``d != w`` (possibly ``c`` herself)?

        We need:

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

        .. math::

            \\text{threshold_c_prevents_w_Condorcet[c, w]} =
            2 \\cdot \\min_{d \\neq w} |w \geq c \\text{ and } w > d|
            - |w \geq c|.

        If this result is negative, it means that even without
        ``c``-manipulators, ``w`` is not a Condorcet winner. In that case,
        threshold is set to 0 instead.
        """
        if self._threshold_c_prevents_w_Condorcet is None:
            self._mylog("Compute threshold_c_prevents_w_Condorcet", 1)
            self._threshold_c_prevents_w_Condorcet = np.full((self.C, self.C),
                                                             np.inf)
            for w in range(self.C):
                for c in range(self.C):
                    if c == w:
                        self._threshold_c_prevents_w_Condorcet[c, w] = 0
                        continue
                    v_does_not_prefer_c_to_w = (
                        self.preferences_utilities[:, w] >=
                        self.preferences_utilities[:, c])
                    for d in range(self.C):
                        if d == w:
                            continue
                            # But d == c is allowed (useful if self.C == 2)
                        v_prefers_w_to_d = (
                            self.preferences_utilities[:, w] >
                            self.preferences_utilities[:, d])
                        threshold_c_makes_w_not_win_against_d = (
                            np.multiply(2, np.sum(np.logical_and(
                                v_does_not_prefer_c_to_w,
                                v_prefers_w_to_d
                            )))
                            - np.sum(v_does_not_prefer_c_to_w)
                        )
                        self._threshold_c_prevents_w_Condorcet[
                            c, w] = np.minimum(
                                self._threshold_c_prevents_w_Condorcet[c, w],
                                threshold_c_makes_w_not_win_against_d
                            )
                    self._threshold_c_prevents_w_Condorcet = np.maximum(
                        self._threshold_c_prevents_w_Condorcet, 0)
        return self._threshold_c_prevents_w_Condorcet

    #%% Total utilities        

    @property
    def total_utility_c(self):
        """1d array of floats. ``total_utility_c[c]`` is the total utility for
        candidate ``c`` (i.e. the sum of ``c``'s column in matrix
        :attr:`svvamp.Population.preferences_utilities`).
        """
        if self._total_utility_c is None:
            self._mylog("Compute total utility of candidates", 1)
            self._total_utility_c = np.sum(self.preferences_utilities, 0)
        return self._total_utility_c

    @property
    def total_utility_min(self):
        """Float. ``total_utility_min`` is the minimum of
        :attr:`svvamp.Population.total_utility_c`.
        """
        if self._total_utility_min is None:
            self._mylog("Compute total_utility_min", 1)
            self._total_utility_min = np.min(self.total_utility_c)
        return self._total_utility_min

    @property
    def total_utility_max(self):
        """Float. ``total_utility_max`` is the maximum of
        :attr:`svvamp.Population.total_utility_c`.
        """
        if self._total_utility_max is None:
            self._mylog("Compute total_utility_max", 1)
            self._total_utility_max = np.max(self.total_utility_c)
        return self._total_utility_max

    @property
    def total_utility_mean(self):
        """Float. ``total_utility_mean`` is the mean of
        :attr:`svvamp.Population.total_utility_c`.
        """
        if self._total_utility_mean is None:
            self._mylog("Compute total_utility_mean", 1)
            self._total_utility_mean = np.mean(self.total_utility_c)
        return self._total_utility_mean

    @property
    def total_utility_std(self):
        """Float. ``total_utility_std`` is the standard deviation of
        :attr:`svvamp.Population.total_utility_c`.
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
        :attr:`svvamp.Population.preferences_utilities`).
        """
        if self._mean_utility_c is None:
            self._mylog("Compute mean utility of candidates", 1)
            self._mean_utility_c = np.mean(self.preferences_utilities, 0)
        return self._mean_utility_c

    @property
    def mean_utility_min(self):
        """Float. ``mean_utility_min`` is the minimum of
        :attr:`svvamp.Population.mean_utility_c`.
        """
        if self._mean_utility_min is None:
            self._mylog("Compute mean_utility_min", 1)
            self._mean_utility_min = np.min(self.mean_utility_c)
        return self._mean_utility_min

    @property
    def mean_utility_max(self):
        """Float. ``mean_utility_max`` is the maximum of
        :attr:`svvamp.Population.mean_utility_c`.
        """
        if self._mean_utility_max is None:
            self._mylog("Compute mean_utility_max", 1)
            self._mean_utility_max = np.max(self.mean_utility_c)
        return self._mean_utility_max

    @property
    def mean_utility_mean(self):
        """Float. ``mean_utility_mean`` is the mean of
        :attr:`svvamp.Population.mean_utility_c`.
        """
        if self._mean_utility_mean is None:
            self._mylog("Compute mean_utility_mean", 1)
            self._mean_utility_mean = np.mean(self.mean_utility_c)
        return self._mean_utility_mean

    @property
    def mean_utility_std(self):
        """Float. ``mean_utility_std`` is the standard deviation of
        :attr:`svvamp.Population.mean_utility_c`.
        """
        if self._mean_utility_std is None:
            self._mylog("Compute mean_utility_std", 1)
            self._mean_utility_std = np.std(self.mean_utility_c, ddof=0)
        return self._mean_utility_std

    #%% Borda scores

    @property
    def borda_score_c_vtb(self):
        """1d array of integers. ``borda_score_c_vtb[c]`` is the total Borda
        score of candidate ``c`` (using
        :attr:`svvamp.Population.preferences_borda_vtb`, i.e. strict orders).
        """
        if self._borda_score_c_vtb is None:
            self._mylog("Compute Borda scores of the candidates with vtb", 1)
            self._borda_score_c_vtb = np.sum(self.matrix_duels_vtb, 1)
        return self._borda_score_c_vtb

    @property
    def borda_score_c_novtb(self):
        """1d array of integers. ``borda_score_c_novtb[c]`` is the total Borda
        score of candidate c (using
        :attr:`svvamp.Population.preferences_borda_novtb`, i.e. weak orders).
        """
        if self._borda_score_c_novtb is None:
            self._mylog("Compute Borda scores of the candidates (weak "
                        "orders)", 1)
            self._borda_score_c_novtb = np.sum(self.preferences_borda_novtb, 0)
        return self._borda_score_c_novtb

    @property
    def candidates_by_decreasing_borda_score_vtb(self):
        """1d array of integers.
        ``candidates_by_decreasing_borda_score_vtb[k]`` is
        the candidate ranked ``k``\ :sup:`th` by decreasing Borda score
        (using strict orders with vtb,
        :attr:`svvamp.Population.borda_score_c_vtb`).

        For example, ``candidates_by_decreasing_borda_score_vtb[0]`` is the
        candidate with highest Borda score.
        """
        if self._candidates_by_decreasing_borda_score_vtb is None:
            self._mylog("Compute candidates_by_decreasing_borda_score_vtb", 1)
            self._candidates_by_decreasing_borda_score_vtb = \
                np.argsort(-self.borda_score_c_vtb, kind='mergesort')
            self._decreasing_borda_scores_vtb = self.borda_score_c_vtb[
                self._candidates_by_decreasing_borda_score_vtb]
        return self._candidates_by_decreasing_borda_score_vtb

    @property
    def decreasing_borda_scores_vtb(self):
        """1d array of integers. ``decreasing_borda_scores_vtb[k]`` is the
        ``k``\ :sup:`th` Borda score (using strict orders with vtb,
        :attr:`svvamp.Population.borda_score_c_vtb`) by decreasing
        order.

        For example, ``decreasing_borda_scores_vtb[0]`` is the highest
        Borda score for a candidate.
        """
        if self._decreasing_borda_scores_vtb is None:
            self._mylog("Compute decreasing_borda_scores_vtb", 1)
            self._candidates_by_decreasing_borda_score_vtb = \
                np.argsort(-self.borda_score_c_vtb, kind='mergesort')
            self._decreasing_borda_scores_vtb = self.borda_score_c_vtb[
                self._candidates_by_decreasing_borda_score_vtb]
        return self._decreasing_borda_scores_vtb

    @property
    def candidates_by_decreasing_borda_score_novtb(self):
        """1d array of integers.
        ``candidates_by_decreasing_borda_score_novtb[k]``
        is the candidate ranked ``k``\ :sup:`th` by decreasing Borda score
        (using weak orders, :attr:`svvamp.Population.borda_score_c_novtb`).

        For example, ``candidates_by_decreasing_borda_score_novtb[0]`` is the
        candidate with highest Borda score.
        """
        if self._candidates_by_decreasing_borda_score_novtb is None:
            self._mylog("Compute candidates_by_decreasing_borda_score_novtb",
                        1)
            self._candidates_by_decreasing_borda_score_novtb = \
                np.argsort(-self.borda_score_c_novtb, kind='mergesort')
            self._decreasing_borda_scores_novtb = self.borda_score_c_novtb[
                self._candidates_by_decreasing_borda_score_novtb]
        return self._candidates_by_decreasing_borda_score_novtb

    @property
    def decreasing_borda_scores_novtb(self):
        """1d array of integers. ``decreasing_borda_scores_novtb[k]`` is the
        ``k``\ :sup:`th` Borda score (using weak orders,
        :attr:`svvamp.Population.borda_score_c_novtb`) by decreasing order.

        For example, ``decreasing_borda_scores_novtb[0]`` is the highest
        Borda score for a candidate.
        """
        if self._decreasing_borda_scores_novtb is None:
            self._mylog("Compute decreasing_borda_scores_novtb", 1)
            self._candidates_by_decreasing_borda_score_novtb = \
                np.argsort(-self.borda_score_c_novtb, kind='mergesort')
            self._decreasing_borda_scores_novtb = self.borda_score_c_novtb[
                self._candidates_by_decreasing_borda_score_novtb]
        return self._decreasing_borda_scores_novtb

    #%% Plot a population

    def plot3(self, indexes=None, normalize=True, use_labels=True):
        """Plot utilities with approval limit for 3 candidates

        :param indexes: List of 3 candidates. If None, defaults to [0, 1, 1].
        :param normalize: Boolean. Cf. below.
        :param use_labels: Boolean. If True, then labels_candidates is
            used to label the plot. Otherwise, candidates are simply
            represented by their index.

        Each red point of the plot represents a voter v: its position is
        preferences_utilities[v, indexes]. If normalize is True,
        then each position is normalized before plotting such that
        its Euclidean norm is equal to 1.

        The equator (in blue) is the set of points such that sum(point[
        i]**2) = 1 and sum(point[i]) = 0, i.e. the unit circle of the plan
        that is orthogonal to the main diagonal [1, 1, 1].

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
        mat_temp = np.copy(self.preferences_utilities[:, _indexes]).astype(
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
        """Plot utilities for 4 candidates (without approval limit)

        :param indexes: List of 4 candidates. If None, defaults to
            [0, 1, 2, 3].
        :param normalize: Boolean. Cf. below.
        :param use_labels: Boolean. If True, then labels_candidates is
            used to label the plot. Otherwise, candidates are simply
            represented by their index.

        Each red point of the plot represents a voter v.
            * preferences_utilities[v, indexes] is sent to the hyperplane that
              is orthogonal to [1, 1, 1, 1], which discards information related
              to approval limit and keeps only the relative preferences between
              candidates.
            * The plot is done in this 3d hyperplane. In practice, we use a
              mirror symmetry that exchanges [1, 1, 1, 1] and [0, 0, 0, 1].
              This way, the new vector is orthogonal to [0, 0, 0, 1] and can be
              plotted in the first 3 dimensions.
            * If normalize is True, then the vector is normalized  before
              plotting such that its Euclidean norm is equal to 1.

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
        mat_temp = np.copy(self.preferences_utilities[:, _indexes]).astype(
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
        """Demonstrate the methods of Population class.

        :param log_depth: Integer from 0 (basic info) to 3 (verbose).
        """
        old_log_depth = self._log_depth
        self._log_depth = log_depth
        self.ensure_voters_sorted_by_ordinal_preferences()
        def printm(variable_name, variable_value):
            print(variable_name)
            print(variable_value)
    
        MyLog.print_title("Basic properties")
        print("V =", self.V)
        print("C =", self.C)
        print("labels_candidates =", self.labels_candidates)
        MyLog.printm("preferences_utilities =", self.preferences_utilities)
        MyLog.printm("preferences_borda_novtb =", self.preferences_borda_novtb)
        MyLog.printm("preferences_borda_vtb =", self.preferences_borda_vtb)
        MyLog.printm("preferences_ranking =", self.preferences_ranking)
    
        MyLog.printm("v_has_same_ordinal_preferences_as_previous_voter =",
                 self.v_has_same_ordinal_preferences_as_previous_voter)
    
        MyLog.print_title("Plurality scores")
        MyLog.printm("preferences_ranking (reminder) =",
                     self.preferences_ranking)
        print("plurality_scores_vtb =", self.plurality_scores_vtb)
        print("")
        MyLog.printm("preferences_borda_novtb (reminder) =",
               self.preferences_borda_novtb)
        print("plurality_scores_novtb =", self.plurality_scores_novtb)
    
        MyLog.print_title("Borda scores")
        MyLog.printm("preferences_borda_vtb (reminder) =",
               self.preferences_borda_vtb)
        MyLog.printm("borda_score_c_vtb =", self.borda_score_c_vtb)
        print("Remark: Borda scores above are computed with the "
              "matrix of duels.")
        MyLog.printm("Check: np.sum(self.preferences_borda_vtb, 0) =",
              np.sum(self.preferences_borda_vtb, 0))
        MyLog.printm("decreasing_borda_scores_vtb =",
               self.decreasing_borda_scores_vtb)
        MyLog.printm("candidates_by_decreasing_borda_score_vtb =",
                 self.candidates_by_decreasing_borda_score_vtb)
        print("")
        MyLog.printm("preferences_borda_novtb (reminder) =",
                 self.preferences_borda_novtb)
        MyLog.printm("borda_score_c_novtb =", self.borda_score_c_novtb)
        MyLog.printm("decreasing_borda_scores_novtb =",
                 self.decreasing_borda_scores_novtb)
        MyLog.printm("candidates_by_decreasing_borda_score_novtb =",
                 self.candidates_by_decreasing_borda_score_novtb)

        MyLog.print_title("Utilities")
        MyLog.printm("preferences_utilities (reminder) =",
               self.preferences_utilities)
        MyLog.printm("total_utility_c = ", self.total_utility_c)
        print("total_utility_min =", self.total_utility_min)
        print("total_utility_max =", self.total_utility_max)
        print("total_utility_mean =", self.total_utility_mean)
        print("total_utility_std =", self.total_utility_std)

        MyLog.print_title("Condorcet notions with vtb")
        MyLog.printm("preferences_ranking (reminder) =", self.preferences_ranking)
        MyLog.printm("matrix_duels_vtb =", self.matrix_duels_vtb)
    
        MyLog.printm("matrix_victories_vtb =", self.matrix_victories_vtb)
        print("condorcet_winner_vtb =", self.condorcet_winner_vtb)
    
        MyLog.printm("matrix_victories_vtb_ctb =", self.matrix_victories_vtb_ctb)
        print("condorcet_winner_vtb_ctb =", self.condorcet_winner_vtb_ctb)
    
        MyLog.print_title("Relative Condorcet notions")
        MyLog.printm("preferences_borda_novtb (reminder) =",
                 self.preferences_borda_novtb)
        MyLog.printm("matrix_duels =", self.matrix_duels)
    
        MyLog.printm("matrix_victories_rel =", self.matrix_victories_rel)
        print("condorcet_winner_rel =", self.condorcet_winner_rel)
    
        MyLog.printm("matrix_victories_rel_ctb =", self.matrix_victories_rel_ctb)
        print("condorcet_winner_rel_ctb =", self.condorcet_winner_rel_ctb)
    
        MyLog.print_title("Absolute Condorcet notions")
        MyLog.printm("matrix_duels (reminder) =", self.matrix_duels)
        MyLog.printm("matrix_victories_abs =", self.matrix_victories_abs)
        MyLog.printm("condorcet_admissible_candidates = ",
              self.condorcet_admissible_candidates)
        print("nb_condorcet_admissible =", self.nb_condorcet_admissible)
        MyLog.printm("weak_condorcet_winners =", self.weak_condorcet_winners)
        print("nb_weak_condorcet_winners =", self.nb_weak_condorcet_winners)
        print("condorcet_winner =", self.condorcet_winner)
        print("resistant_condorcet_winner =", self.resistant_condorcet_winner)
        MyLog.printm("threshold_c_prevents_w_Condorcet =",
                 self.threshold_c_prevents_w_Condorcet)

        MyLog.printm("matrix_victories_abs_ctb =", self.matrix_victories_abs_ctb)
        print("condorcet_winner_ctb =", self.condorcet_winner_ctb)

        MyLog.print_title("Implications between Condorcet notions")
        # Resistant Condorcet (False)
        #  ||
        #  V
        # Condorcet (False)              ==>             Condorcet_ctb (False)
        #  ||          ||                                     ||           ||
        #  ||          V                                      V            ||
        #  ||      Condorcet_vtb (False) ==> Condorcet_vtb_ctb (False)     ||
        #  V                                                               V
        # Condorcet_rel (False)          ==>         Condorcet_rel_ctb (False)
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
        print('Condorcet ' + display_bool(self.exists_condorcet_winner) +
              '              ==>             Condorcet_ctb ' +
              display_bool(self.exists_condorcet_winner_ctb))
        print(' ||          ||                  '
              '                   ||           ||')
        print(' ||          V                   '
              '                   V            ||')
        print(' ||      Condorcet_vtb ' +
              display_bool(self.exists_condorcet_winner_vtb) +
              ' ==> Condorcet_vtb_ctb ' +
              display_bool(self.exists_condorcet_winner_vtb_ctb) +
              '     ||')
        print(' V                               '
              '                                V')
        print('Condorcet_rel ' +
              display_bool(self.exists_condorcet_winner_rel) +
              '          ==>         Condorcet_rel_ctb ' +
              display_bool(self.exists_condorcet_winner_rel_ctb))
        print(' ||')
        print(' V')
        print('Weak Condorcet ' +
              display_bool(self.exists_weak_condorcet_winner))
        print(' ||')
        print(' V')
        print('Condorcet-admissible ' +
              display_bool(self.exists_condorcet_admissible))

        self._log_depth = old_log_depth


def preferences_utilities_to_preferences_ranking(preferences_utilities):
    """Convert utilities to rankings.

    Arguments:
    preferences_utilities -- 2d array of floats.
        preferences_utilities[v, c] is the utility of candidate c as
        seen by voter v.

    Returns:
    preferences_ranking -- 2d array of integers. preferences_ranking[v, k]
        is the candidate at rank k for voter v.

    If preferences_utilities[v,c] == preferences_utilities[v,d], then it is
    drawn at random whether c prefers c to d or d to c.
    """
    V, C = preferences_utilities.shape
    tiebreaker = np.random.rand(V, C)
    return np.lexsort((tiebreaker, -preferences_utilities), 1)


def preferences_ranking_to_preferences_borda_vtb(preferences_ranking):
    """Convert rankings to Borda scores (with voter tie-breaking).

    Arguments:
    preferences_ranking -- 2d array of integers. preferences_ranking[v, k]
        is the candidate at rank k for voter v.

    Returns:
    preferences_borda_vtb -- 2d array of integers.
        preferences_borda_vtb[v, c] is the Borda score (between 0 and C - 1)
        of candidate c for voter v.
    """
    _, C = preferences_ranking.shape
    return C - 1 - np.argsort(preferences_ranking, 1)


def preferences_utilities_to_preferences_borda_novtb(
        preferences_utilities):
    """Convert utilities to Borda scores, with equalities.

    Arguments:
    preferences_utilities -- 2d array of floats.
        preferences_utilities[v, c] is the utility of candidate c as
        seen by voter v.

    Returns:
    preferences_borda_novtb -- 2d array of integers.
        preferences_borda_vtb[v, c] gains 1 point for each d such that v
        prefers c to d, and 0.5 point for each d such that v is
        indifferent between c and d.
    """
    V, C = preferences_utilities.shape
    preference_borda_novtb = np.zeros((V, C))
    for c in range(C):
        preference_borda_novtb[:, c] = np.sum(
            0.5 * (preferences_utilities[:, c][:, np.newaxis] >=
                   preferences_utilities) +
            0.5 * (preferences_utilities[:, c][:, np.newaxis] >
                   preferences_utilities),
            1) - 0.5
    return preference_borda_novtb


def preferences_utilities_to_matrix_duels(preferences_utilities):
    """Compute the matrix of duels.

    Arguments:
    preferences_utilities -- 2d array of floats.
        preferences_utilities[v, c] is the utility of candidate c as
        seen by voter v.

    Returns:
    matrix_duels -- 2d array of integers.
        matrix_duels[c, d] is the number of voters who strictly prefer
        candidate c to d. By convention, diagonal coefficients are set to
        0.
    """
    n, m = preferences_utilities.shape
    matrix_duels = np.zeros((m, m), dtype=np.int)
    for c in range(m):
        for d in range(c + 1, m):
            matrix_duels[c, d] = np.sum(preferences_utilities[:, c] >
                                        preferences_utilities[:, d])
            matrix_duels[d, c] = np.sum(preferences_utilities[:, d] >
                                        preferences_utilities[:, c])
    return matrix_duels


def is_resistant_condorcet(w, preferences_utilities):
    """Test for Resistant Condorcet winner.

    Arguments:
    w -- Integer (candidate). For compatibility reasons, NaN is allowed.
    preferences_utilities -- 2d array of floats.
        preferences_utilities[v, c] is the utility of candidate c as
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
    V, C = preferences_utilities.shape
    for c in range(C):
        if c == w:
            continue
        v_does_not_prefer_c_to_w = (preferences_utilities[:, w] >=
                                    preferences_utilities[:, c])
        for d in range(C):
            if d == w:
                continue
            v_prefers_w_to_d = (preferences_utilities[:, w] >
                                preferences_utilities[:, d])
            if np.sum(np.logical_and(v_does_not_prefer_c_to_w,
                                     v_prefers_w_to_d)) <= V / 2:
                return False
    return True


# def compute_condorcet_quick(preferences_utilities):
#     """Compute Condorcet winner.
#
#     Arguments:
#     preferences_utilities -- 2d array of floats.
#         preferences_utilities[v, c] is the utility of candidate c as
#         seen by voter v.
#
#     Returns:
#     condorcet_winner_ctb -- Integer or NaN. 'tb' stands for 'ties
# broken'.
#         Candidate who has only victories in matrix_victories_ctb.
#         If there is no such candidate, then NaN.
#     condorcet_winner -- Integer or NaN. Candidate who has only victories
#         in matrix_victories. If there is no such candidate, then NaN.
#     """
#     n, m = preferences_utilities.shape
#
#     # Step 1 : we move forward in the list of candidates, always keeping
#     # the winner
#     w_provisional = 0
#     has_used_tie_break = False
#     for d in range(1, m):
#         voters_d_vs_w = sum(preferences_utilities[:, d] >
#                             preferences_utilities[:, w_provisional])
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
#         if sum(preferences_utilities[:, c] >=
#                 preferences_utilities[:, w_provisional]) >= n / 2:
#             return np.nan, np.nan
#     condorcet_winner_ctb = w_provisional
#     if has_used_tie_break:
#         condorcet_winner = np.nan
#     else:
#         condorcet_winner = condorcet_winner_ctb
#     return condorcet_winner_ctb, condorcet_winner


if __name__ == '__main__':
    # A quick demo
    preferences_utilities = np.random.randint(-5, 5, (10, 5))
    pop = Population(preferences_utilities=preferences_utilities,
                     labels_candidates=['Adélaïde', 'Bartholomé', 'Cunégonde',
                                        'Dagobert', 'Eugénie'])
    pop.demo(log_depth=1)
    pop.plot3(indexes=[0,1,2], normalize=True)
    pop.plot4(indexes=[0,1,2,3], normalize=True)
