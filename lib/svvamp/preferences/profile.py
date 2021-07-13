# -*- coding: utf-8 -*-
"""
Created on Mon Sep 22 15:24:52 2014
Copyright François Durand 2014-2018
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
# noinspection PyUnresolvedReferences
from mpl_toolkits.mplot3d import Axes3D

from svvamp.utils.printing import printm, print_title
from svvamp.utils import my_log
from svvamp.utils.util_cache import cached_property
from svvamp.utils.misc import initialize_random_seeds
from svvamp.utils.misc import preferences_ut_to_preferences_rk, preferences_rk_to_preferences_borda_rk, \
    preferences_ut_to_preferences_borda_ut, preferences_ut_to_matrix_duels_ut, is_resistant_condorcet, \
    matrix_victories_to_smith_set
from svvamp.utils.pseudo_bool import equal_true


class Profile(my_log.MyLog):

    def __init__(self, preferences_ut=None, preferences_rk=None, log_creation=None, labels_candidates=None,
                 sort_voters=True):
        """Create a profile of voters with preferences over some candidates.

        Parameters
        ----------
        preferences_ut : list of list (or 2d ndarray)
            2d array of floats. ``preferences_ut[v, c]`` is the utility of candidate ``c`` as seen by voter ``v``.
        preferences_rk : list of list (or 2d ndarray)
            2d array of integers. ``preferences_rk[v, k]`` is the candidate at rank ``k`` for voter ``v``.
        log_creation : object
            Any type (string, list...). Some comments.
        labels_candidates : list of str
            Names of the candidates.
        sort_voters : bool
            If True, then when the profile is created, voters are immediately sorted, first by their
            strict order of preference (their row in :attr:`~svvamp.Population.preferences_rk`), then by their weak
            order of preference (their row in :attr:`~svvamp.Population.preferences_borda_ut`). Note that two voters
            having the same strict order may have different weak orders, and vice versa. The objective of this
            sorting is to accelerate some algorithms (typically Individual Manipulation).

        Notes
        -----
        You may enter ``preferences_ut``, ``preferences_rk`` or both to define the preferences of the population.

        If you provide ``preferences_rk`` only, then ``preferences_ut`` is set to the corresponding Borda scores
        (:attr:`~svvamp.Population.preferences_borda_rk`).

        If you provide ``preferences_ut`` only, then ``preferences_rk`` is naturally derived from utilities. If voter
        ``v`` has a greater utility for candidate ``c`` than for candidate ``d``, then she ranks ``c`` before ``d``.
        If voter ``v`` attributes the same utility to several candidates, then the first time the attribute
        ``preferences_rk`` is called, a random ranking will be decided for these tied candidates (once and for all).

        If you provide both, then SVVAMP DOES NOT CHECK that they are coherent. You should ensure that they are,
        in the sense that if ``v`` ranks ``c`` before ``d``, then she her utility for ``c`` must be at least equal to
        her utility for ``d``.

        ``preferences_rk`` will be used for sincere voting when a voting system accepts only strict orders.

        In contrast, for manipulation purposes, ``preferences_ut`` is always used , which means that indifference is
        taken into account as such to determine the interest in manipulation. If voter ``v`` attributes the same
        utility to candidates ``w`` and ``c``, and if ``w`` is the sincere winner in an election, then ``v`` is not
        interested in a manipulation for ``c``.

        If all voters have a strict order of preference (in the sense of ut), then for functions below having
        variants with suffix ``_ut`` or ``_rk``, the two variants are equivalent.

        In some voting systems and in some of the attributes below, we use a process referred as **candidate
        tie-breaking** or **CTB** in SVVAMP. It means that lowest-index candidates are favored. When using CTB,
        if candidates ``c`` and ``d`` are tied (for example, in an election using Plurality), then ``c`` is favored
        over ``d`` iff ``c < d``.

        Implications between majority favorite and Condorcet criteria (cf. corresponding attributes below).

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

        If all voters have strict orders of preference (in the sense of ut) and if there is an odd number of voters,
        then:

            * ``majority_favorite_ut``, ``majority_favorite_rk``, ``majority_favorite_ut_ctb`` and
              ``majority_favorite_rk_ctb`` are equivalent,
            *  ``Condorcet_ut_abs``, ``Condorcet_ut_abs_ctb``, ``Condorcet_rk``, ``Condorcet_rk_ctb``,
               ``Condorcet_ut_rel``, ``Condorcet_ut_rel_ctb``, ``Weak Condorcet`` and ``Condorcet-admissible`` are
               equivalent.
        """
        super().__init__(log_identity="PROFILE")

        # Preference arrays
        if preferences_rk is None:
            preferences_rk = preferences_ut_to_preferences_rk(preferences_ut)
        self.preferences_rk = np.array(preferences_rk)
        """2d array of integers. ``preferences_rk[v, k]`` is the candidate at rank ``k`` for voter ``v``. For
        example, ``preferences_rk[v, 0]`` is ``v``'s preferred candidate.
        """
        self.preferences_borda_rk = preferences_rk_to_preferences_borda_rk(self.preferences_rk)
        """2d array of integers. ``preferences_borda_rk[v, c]`` gains 1 point for each candidate ``d`` such that
        voter ``v`` ranks ``c`` before ``d``. So, these Borda scores are between ``0`` and ``C - 1``.
        """
        if preferences_ut is None:
            preferences_ut = self.preferences_borda_rk
        self.preferences_ut = np.array(preferences_ut)
        """2d array of floats. ``preferences_ut[v, c]`` is the utility of  candidate ``c`` as seen by voter ``v``."""
        self.preferences_borda_ut = preferences_ut_to_preferences_borda_ut(self.preferences_ut)
        """2d array of integers. ``preferences_borda_ut[v, c]`` gains 1 point for each ``d`` such that ``v`` strictly
        prefers ``c`` to ``d`` (in the sense of utilities), and 0.5 point for each ``d`` such that ``v`` is
        indifferent between ``c`` and ``d``. So, these Borda scores are between ``0`` and ``C - 1``.
        """

        # Number of voters and candidates
        self.n_v = None
        """int : Number of voters."""
        self.n_c = None
        """int : Number of candidates."""
        self.n_v, self.n_c = self.preferences_ut.shape
        if self.n_v < 2 or self.n_c < 2:
            raise ValueError("A population must have at least 2 voters and 2 candidates.")

        if sort_voters:
            # Sort voters by weak order (deducted from utility)
            list_borda_ut = self.preferences_borda_ut.tolist()
            # noinspection PyTypeChecker,PyUnresolvedReferences
            indexes = sorted(range(len(list_borda_ut)), key=list_borda_ut.__getitem__)
            self.preferences_rk = self.preferences_rk[indexes, ::]
            self.preferences_ut = self.preferences_ut[indexes, ::]
            self.preferences_borda_rk = self.preferences_borda_rk[indexes, ::]
            self.preferences_borda_ut = self.preferences_borda_ut[indexes, ::]
            # Sort voters by strict ranking
            list_rankings = self.preferences_rk.tolist()
            indexes = sorted(range(len(list_rankings)), key=list_rankings.__getitem__)
            self.preferences_rk = self.preferences_rk[indexes, ::]
            self.preferences_ut = self.preferences_ut[indexes, ::]
            self.preferences_borda_rk = self.preferences_borda_rk[indexes, ::]
            self.preferences_borda_ut = self.preferences_borda_ut[indexes, ::]

        # Misc variables
        self._labels_candidates = labels_candidates
        self.log_creation = log_creation

    # %% Basic variables

    @property
    def labels_candidates(self):
        """List of :attr:`~svvamp.Population.n_c` strings (names of the candidates).
        """
        if self._labels_candidates is None:
            self._labels_candidates = [str(x) for x in range(self.n_c)]
        return self._labels_candidates

    @labels_candidates.setter
    def labels_candidates(self, value):
        self._labels_candidates = value

    # %% Identify voters with the same ordinal preferences.

    @cached_property
    def v_has_same_ordinal_preferences_as_previous_voter(self):
        """1d array of booleans. ``v_has_same_ordinal_preferences_as_previous_voter[v]`` is ``True`` iff voter ``v``
        has the same preference strict order (row in :attr:`~svvamp.Population.preferences_rk`) and the same
        preference weak order (row in :attr:`~svvamp.Population.preferences_borda_ut`) as voter ``v-1``. By convention,
        it is ``False`` for voter ``0``.

        >>> from svvamp import Profile
        >>> profile = Profile(preferences_rk=[[0, 1, 2], [0, 1, 2]])
        >>> profile.v_has_same_ordinal_preferences_as_previous_voter
        array([False,  True])
        """
        self.mylog("Compute v_has_same_ordinal_preferences_as_previous_voter", 1)
        return np.concatenate((
                [False],
                np.logical_and(
                    np.all(self.preferences_rk[range(self.n_v - 1), :]
                           == self.preferences_rk[range(1, self.n_v), :], 1),
                    np.all(self.preferences_borda_ut[range(self.n_v - 1), :]
                           == self.preferences_borda_ut[range(1, self.n_v), :], 1)
                )
        ))

    # %% Plurality scores

    @cached_property
    def plurality_scores_rk(self):
        """1d array of int. ``plurality_scores_rk[c]`` is the number of voters for whom ``c`` is the top-ranked
        candidate (with voter tie-breaking).

        >>> from svvamp import Profile
        >>> profile = Profile(preferences_rk=[[0, 1, 2], [0, 1, 2]])
        >>> profile.plurality_scores_rk
        array([2, 0, 0])
        """
        self.mylog("Compute Plurality scores (rk)", 1)
        return np.array(np.bincount(self.preferences_rk[:, 0], minlength=self.n_c), dtype=int)

    @cached_property
    def plurality_scores_ut(self):
        """1d array of int. ``plurality_scores_ut[c]`` is the number of voters who strictly prefer ``c`` to all
        other candidates. If a voter has several candidates with maximal utility, then none of them receives any point.

        >>> from svvamp import Profile
        >>> profile = Profile(preferences_ut=[[2, 1, 0], [0, 1, 1]])
        >>> profile.plurality_scores_ut
        array([1, 0, 0])
        """
        self.mylog("Compute Plurality scores (ut)", 1)
        result = np.zeros(self.n_c, dtype=int)
        for v in range(self.n_v):
            c = np.argmax(self.preferences_ut[v, :])
            if np.all(np.greater(self.preferences_ut[v, c], self.preferences_ut[v, np.array(range(self.n_c)) != c])):
                result[c] += 1
        return result

    # %% Matrix of duels

    @cached_property
    def matrix_duels_ut(self):
        """2d array of integers. ``matrix_duels_ut[c, d]`` is the number of voters who have a strictly greater
        utility for ``c`` than for ``d``. By convention, diagonal coefficients are set to ``0``.

        >>> from svvamp import Profile
        >>> profile = Profile(preferences_rk=[[0, 1, 2], [0, 1, 2]])
        >>> profile.matrix_duels_ut
        array([[0, 2, 2],
               [0, 0, 2],
               [0, 0, 0]])
        """
        self.mylog("Compute matrix of duels", 1)
        return preferences_ut_to_matrix_duels_ut(self.preferences_borda_ut)

    @cached_property
    def matrix_duels_rk(self):
        """2d array of integers. ``matrix_duels_rk[c, d]`` is the number of voters who rank candidate ``c`` before
        ``d`` (in the sense of :attr:`~svvamp.Population.preferences_rk`). By convention, diagonal coefficients are
        set to 0.

        >>> from svvamp import Profile
        >>> profile = Profile(preferences_rk=[[0, 1, 2], [0, 1, 2]])
        >>> profile.matrix_duels_rk
        array([[0, 2, 2],
               [0, 0, 2],
               [0, 0, 0]])
        """
        self.mylog("Compute matrix of duels (with strict orders)", 1)
        return preferences_ut_to_matrix_duels_ut(self.preferences_borda_rk)

    @cached_property
    def matrix_victories_ut_abs(self):
        """2d array of values in {0, 0.5, 1}. Matrix of absolute victories based on
        :attr:`~svvamp.Population.matrix_duels_ut`.

        ``matrix_victories_ut_abs[c, d]`` is:

            * 1   iff ``matrix_duels_ut[c, d] > V / 2``.
            * 0.5 iff ``matrix_duels_ut[c, d] = V / 2``.
            * 0   iff ``matrix_duels_ut[c, d] < V / 2``.

        By convention, diagonal coefficients are set to 0.

        Examples
        --------
        A victory:

            >>> profile = Profile(preferences_ut=[[1, 0], [1, 0]])
            >>> profile.matrix_victories_ut_abs
            array([[0., 1.],
                   [0., 0.]])

        A half-victory, but the other candidate has lost:

            >>> profile = Profile(preferences_ut=[[1, 0], [42, 42]])
            >>> profile.matrix_victories_ut_abs
            array([[0. , 0.5],
                   [0. , 0. ]])

        A half-victory for each candidate:

            >>> profile = Profile(preferences_ut=[[1, 0], [0, 1]])
            >>> profile.matrix_victories_ut_abs
            array([[0. , 0.5],
                   [0.5, 0. ]])

        A defeat for each candidate:

            >>> profile = Profile(preferences_ut=[[0, 0], [0, 0]])
            >>> profile.matrix_victories_ut_abs
            array([[0., 0.],
                   [0., 0.]])
        """
        self.mylog("Compute matrix_victories_ut_abs", 1)
        return (np.multiply(0.5, self.matrix_duels_ut >= self.n_v / 2)
                + np.multiply(0.5, self.matrix_duels_ut > self.n_v / 2))

    @cached_property
    def matrix_victories_ut_abs_ctb(self):
        """2d array of values in {0, 1}. Matrix of absolute victories based on
        :attr:`~svvamp.Population.matrix_duels_ut`, with tie-breaks on candidates.

        ``matrix_victories_ut_abs_ctb[c, d]`` is:

            * 1   iff ``matrix_duels_ut[c, d] > V / 2``, or ``matrix_duels_ut[c, d] = V / 2`` and ``c < d``.
            * 0   iff ``matrix_duels_ut[c, d] < V / 2``, or ``matrix_duels_ut[c, d] = V / 2`` and ``d < c``.

        By convention, diagonal coefficients are set to 0.

        Examples
        --------
        A victory (without tie-breaking):

            >>> profile = Profile(preferences_ut=[[1, 0], [1, 0]])
            >>> profile.matrix_victories_ut_abs_ctb
            array([[0., 1.],
                   [0., 0.]])

        A victory with tie-breaking:

            >>> profile = Profile(preferences_ut=[[1, 0], [0, 1]])
            >>> profile.matrix_victories_ut_abs_ctb
            array([[0., 1.],
                   [0., 0.]])

        Another case of victory with tie-breaking:

            >>> profile = Profile(preferences_ut=[[1, 0], [42, 42]])
            >>> profile.matrix_victories_ut_abs_ctb
            array([[0., 1.],
                   [0., 0.]])

        A defeat for each candidate:

            >>> profile = Profile(preferences_ut=[[0, 0], [0, 0]])
            >>> profile.matrix_victories_ut_abs_ctb
            array([[0., 0.],
                   [0., 0.]])
        """
        self.mylog("Compute matrix_victories_ut_abs_ctb", 1)
        result = np.zeros((self.n_c, self.n_c))
        for c in range(self.n_c):
            for d in range(self.n_c):
                if c == d:
                    continue
                if self.matrix_duels_ut[c, d] > self.n_v / 2 or (self.matrix_duels_ut[c, d] == self.n_v / 2 and c < d):
                    result[c, d] = 1
        return result

    @cached_property
    def matrix_victories_ut_rel(self):
        """2d array of values in {0, 0.5, 1}. Matrix of relative victories based on
        :attr:`~svvamp.Population.matrix_duels_ut`.

        ``matrix_victories_ut_rel[c, d]`` is:

            * 1   iff ``matrix_duels_ut[c, d] > matrix_duels_ut[d, c]``.
            * 0.5 iff ``matrix_duels_ut[c, d] = matrix_duels_ut[d, c]``.
            * 0   iff ``matrix_duels_ut[c, d] < matrix_duels_ut[d, c]``.

        By convention, diagonal coefficients are set to 0.

        Examples
        --------
        A relative victory (which is not an absolute victory):

            >>> profile = Profile(preferences_ut=[[1, 0], [0, 0], [0, 0]])
            >>> profile.matrix_victories_ut_rel
            array([[0., 1.],
                   [0., 0.]])

        A tie:

            >>> profile = Profile(preferences_ut=[[1, 0], [0, 1], [0, 0]])
            >>> profile.matrix_victories_ut_rel
            array([[0. , 0.5],
                   [0.5, 0. ]])
        """
        self.mylog("Compute matrix_victories_ut_rel", 1)
        return (
            np.multiply(0.5, self.matrix_duels_ut >= self.matrix_duels_ut.T)
            + np.multiply(0.5, self.matrix_duels_ut > self.matrix_duels_ut.T)
            - np.multiply(0.5, np.eye(self.n_c))
        )

    @cached_property
    def matrix_victories_ut_rel_ctb(self):
        """2d array of values in {0, 1}. Matrix of relative victories based on
        :attr:`~svvamp.Population.matrix_duels_ut`, with tie-breaks on candidates.

        ``matrix_victories_ut_rel_ctb[c, d]`` is:

            * 1   iff ``matrix_duels_ut[c, d] > matrix_duels_ut[d, c]``, or
              ``matrix_duels_ut[c, d] = matrix_duels_ut[d, c]`` and ``c < d``.
            * 0   iff ``matrix_duels_ut[c, d] < matrix_duels_ut[d, c]``, or
              ``matrix_duels_ut[c, d] = matrix_duels_ut[d, c]`` and ``d < c``.

        By convention, diagonal coefficients are set to 0.

        Examples
        --------
        A relative victory (without tie-breaking):

            >>> profile = Profile(preferences_ut=[[1, 0], [0, 0], [0, 0]])
            >>> profile.matrix_victories_ut_rel_ctb
            array([[0., 1.],
                   [0., 0.]])

        A relative victory with tie-breaking:

            >>> profile = Profile(preferences_ut=[[1, 0], [0, 1], [0, 0]])
            >>> profile.matrix_victories_ut_rel_ctb
            array([[0., 1.],
                   [0., 0.]])
        """
        self.mylog("Compute matrix_victories_ut_rel_ctb", 1)
        result = np.zeros((self.n_c, self.n_c))
        for c in range(self.n_c):
            for d in range(c + 1, self.n_c):
                if self.matrix_duels_ut[c, d] >= self.matrix_duels_ut[d, c]:
                    result[c, d] = 1
                    result[d, c] = 0
                else:
                    result[c, d] = 0
                    result[d, c] = 1
        return result

    @cached_property
    def matrix_victories_rk(self):
        """2d array of values in {0, 0.5, 1}. Matrix of victories based on :attr:`~svvamp.Population.matrix_duels_rk`.

        ``matrix_victories_rk[c, d]`` is:

            * 1   iff ``matrix_duels_rk[c, d] > matrix_duels_rk[d, c]``, i.e. iff ``matrix_duels_rk[c, d] > V / 2``.
            * 0.5 iff ``matrix_duels_rk[c, d] = matrix_duels_rk[d, c]``, i.e. iff ``matrix_duels_rk[c, d] = V / 2``.
            * 0   iff ``matrix_duels_rk[c, d] < matrix_duels_rk[d, c]``, i.e. iff ``matrix_duels_rk[c, d] < V / 2``.

        By convention, diagonal coefficients are set to 0.

        Examples
        --------
        A victory:

            >>> profile = Profile(preferences_rk=[[0, 1], [0, 1]])
            >>> profile.matrix_victories_rk
            array([[0., 1.],
                   [0., 0.]])

        A tie:

            >>> profile = Profile(preferences_rk=[[0, 1], [1, 0]])
            >>> profile.matrix_victories_rk
            array([[0. , 0.5],
                   [0.5, 0. ]])

        """
        self.mylog("Compute matrix_victories_rk", 1)
        return (np.multiply(0.5, self.matrix_duels_rk >= self.n_v / 2)
                + np.multiply(0.5, self.matrix_duels_rk > self.n_v / 2))

    @cached_property
    def matrix_victories_rk_ctb(self):
        """2d array of values in {0, 1}. Matrix of victories based on :attr:`~svvamp.Population.matrix_duels_rk`,
        with tie-breaks on candidates.

        ``matrix_victories_rk_ctb[c, d]`` is:

            * 1   iff ``matrix_duels_rk[c, d] > matrix_duels_rk[d, c]``, or
              ``matrix_duels_rk[c, d] = matrix_duels_rk[d, c]`` and ``c < d``.
            * 0   iff ``matrix_duels_rk[c, d] < matrix_duels_rk[d, c]``, or
              ``matrix_duels_rk[c, d] = matrix_duels_rk[d, c]`` and ``d < c``.

        By convention, diagonal coefficients are set to 0.

        Examples
        --------
        A victory (without tie-breaking):

            >>> profile = Profile(preferences_rk=[[0, 1], [0, 1]])
            >>> profile.matrix_victories_rk_ctb
            array([[0., 1.],
                   [0., 0.]])

        A victory with tie-breaking:

            >>> profile = Profile(preferences_rk=[[0, 1], [1, 0]])
            >>> profile.matrix_victories_rk_ctb
            array([[0., 1.],
                   [0., 0.]])
        """
        self.mylog("Compute matrix_victories_rk_ctb", 1)
        result = np.zeros((self.n_c, self.n_c))
        for c in range(self.n_c):
            for d in range(c + 1, self.n_c):
                if self.matrix_duels_rk[c, d] >= self.n_v / 2:
                    result[c, d] = 1
                    result[d, c] = 0
                else:
                    result[c, d] = 0
                    result[d, c] = 1
        return result

    # %% Existence of Condorcet order

    @cached_property
    def exists_condorcet_order_rk_ctb(self):
        """Boolean. True iff in matrix :attr:`~svvamp.Population.matrix_victories_rk_ctb`, there is a candidate with
        ``n_c - 1`` victories, a candidate with ``n_c - 2`` victories, etc.

        Examples
        --------
            >>> from svvamp import Profile
            >>> profile = Profile(preferences_rk=[[0, 1, 2], [0, 1, 2]])
            >>> profile.exists_condorcet_order_rk_ctb
            True

        See Also
        --------
        :attr:`~svvamp.Population.not_exists_condorcet_order_rk_ctb`.
        """
        self.mylog("Compute exists_condorcet_order_rk_ctb", 1)
        temp = np.copy(self.matrix_victories_rk_ctb.sum(1))
        temp.sort()
        return np.array_equal(range(self.n_c), temp)

    @cached_property
    def not_exists_condorcet_order_rk_ctb(self):
        """Boolean. Cf. :attr:`~svvamp.Population.exists_condorcet_order_rk_ctb`.

        Examples
        --------
            >>> from svvamp import Profile
            >>> profile = Profile(preferences_rk=[[0, 1, 2], [0, 1, 2]])
            >>> profile.not_exists_condorcet_order_rk_ctb
            False
        """
        return not self.exists_condorcet_order_rk_ctb

    @cached_property
    def exists_condorcet_order_rk(self):
        """Boolean. True iff in matrix :attr:`~svvamp.Population.matrix_victories_rk`, there is a candidate with
        ``n_c - 1`` victories, a candidate with ``n_c - 2`` victories, etc.

        Examples
        --------
            >>> from svvamp import Profile
            >>> profile = Profile(preferences_rk=[[0, 1, 2], [0, 1, 2]])
            >>> profile.exists_condorcet_order_rk
            True

        See Also
        --------
        :attr:`~svvamp.Population.not_exists_condorcet_order_rk`.
        """
        self.mylog("Compute exists_condorcet_order_rk", 1)
        temp = np.copy(self.matrix_victories_rk.sum(1))
        temp.sort()
        return np.array_equal(range(self.n_c), temp)

    @cached_property
    def not_exists_condorcet_order_rk(self):
        """Boolean. Cf. :attr:`~svvamp.Population.exists_condorcet_order_rk`.

        Examples
        --------
            >>> from svvamp import Profile
            >>> profile = Profile(preferences_rk=[[0, 1, 2], [0, 1, 2]])
            >>> profile.not_exists_condorcet_order_rk
            False
        """
        return not self.exists_condorcet_order_rk

    @cached_property
    def exists_condorcet_order_ut_abs(self):
        """Boolean. True iff in matrix :attr:`~svvamp.Population.matrix_victories_ut_abs`, there is a candidate with
        ``n_c - 1`` victories, a candidate with ``n_c - 2`` victories, etc.

        Examples
        --------
            >>> from svvamp import Profile
            >>> profile = Profile(preferences_rk=[[0, 1, 2], [0, 1, 2]])
            >>> profile.exists_condorcet_order_ut_abs
            True

        See Also
        --------
        :attr:`~svvamp.Population.not_exists_condorcet_order_ut_abs`.
        """
        self.mylog("Compute exists_condorcet_order_ut_abs", 1)
        temp = np.copy(self.matrix_victories_ut_abs.sum(1))
        temp.sort()
        return np.array_equal(range(self.n_c), temp)

    @cached_property
    def not_exists_condorcet_order_ut_abs(self):
        """Boolean. Cf. :attr:`~svvamp.Population.exists_condorcet_order_ut_abs`.

        Examples
        --------
            >>> from svvamp import Profile
            >>> profile = Profile(preferences_rk=[[0, 1, 2], [0, 1, 2]])
            >>> profile.not_exists_condorcet_order_ut_abs
            False
        """
        return not self.exists_condorcet_order_ut_abs

    @cached_property
    def exists_condorcet_order_ut_abs_ctb(self):
        """Boolean. True iff in matrix :attr:`~svvamp.Population.matrix_victories_ut_abs_ctb`, there is a candidate
        with ``n_c - 1`` victories, a candidate with ``n_c - 2`` victories, etc.

        Examples
        --------
            >>> from svvamp import Profile
            >>> profile = Profile(preferences_rk=[[0, 1, 2], [0, 1, 2]])
            >>> profile.exists_condorcet_order_ut_abs_ctb
            True

        See Also
        --------
        :attr:`~svvamp.Population.not_exists_condorcet_order_ut_abs_ctb`.
        """
        self.mylog("Compute exists_condorcet_order_ut_abs_ctb", 1)
        temp = np.copy(self.matrix_victories_ut_abs_ctb.sum(1))
        temp.sort()
        return np.array_equal(range(self.n_c), temp)

    @cached_property
    def not_exists_condorcet_order_ut_abs_ctb(self):
        """Boolean. Cf. :attr:`~svvamp.Population.exists_condorcet_order_ut_abs_ctb`.

        Examples
        --------
            >>> from svvamp import Profile
            >>> profile = Profile(preferences_rk=[[0, 1, 2], [0, 1, 2]])
            >>> profile.not_exists_condorcet_order_ut_abs_ctb
            False
        """
        return not self.exists_condorcet_order_ut_abs_ctb

    @cached_property
    def exists_condorcet_order_ut_rel(self):
        """Boolean. True iff in matrix :attr:`~svvamp.Population.matrix_victories_ut_rel`, there is a candidate with
        ``n_c - 1`` victories, a candidate with ``n_c - 2`` victories, etc.

        Examples
        --------
            >>> from svvamp import Profile
            >>> profile = Profile(preferences_rk=[[0, 1, 2], [0, 1, 2]])
            >>> profile.exists_condorcet_order_ut_rel
            True

        See Also
        --------
        :attr:`~svvamp.Population.not_exists_condorcet_order_ut_rel`.
        """
        self.mylog("Compute exists_condorcet_order_ut_rel", 1)
        temp = np.copy(self.matrix_victories_ut_rel.sum(1))
        temp.sort()
        return np.array_equal(range(self.n_c), temp)

    @cached_property
    def not_exists_condorcet_order_ut_rel(self):
        """Boolean. Cf. :attr:`~svvamp.Population.exists_condorcet_order_ut_rel`.

        Examples
        --------
            >>> from svvamp import Profile
            >>> profile = Profile(preferences_rk=[[0, 1, 2], [0, 1, 2]])
            >>> profile.not_exists_condorcet_order_ut_rel
            False
        """
        return not self.exists_condorcet_order_ut_rel

    @cached_property
    def exists_condorcet_order_ut_rel_ctb(self):
        """Boolean. True iff in matrix :attr:`~svvamp.Population.matrix_victories_ut_rel_ctb`, there is a candidate
        with ``n_c - 1`` victories, a candidate with ``n_c - 2`` victories, etc.

        Examples
        --------
            >>> from svvamp import Profile
            >>> profile = Profile(preferences_rk=[[0, 1, 2], [0, 1, 2]])
            >>> profile.exists_condorcet_order_ut_rel_ctb
            True

        See Also
        --------
        :attr:`~svvamp.Population.not_exists_condorcet_order_ut_rel_ctb`.
        """
        self.mylog("Compute exists_condorcet_order_ut_rel_ctb", 1)
        temp = np.copy(self.matrix_victories_ut_rel_ctb.sum(1))
        temp.sort()
        return np.array_equal(range(self.n_c), temp)

    @cached_property
    def not_exists_condorcet_order_ut_rel_ctb(self):
        """Boolean. Cf. :attr:`~svvamp.Population.exists_condorcet_order_ut_rel_ctb`.

        Examples
        --------
            >>> from svvamp import Profile
            >>> profile = Profile(preferences_rk=[[0, 1, 2], [0, 1, 2]])
            >>> profile.not_exists_condorcet_order_ut_rel_ctb
            False
        """
        return not self.exists_condorcet_order_ut_rel_ctb

    # %% Condorcet winner and variants

    @cached_property
    def condorcet_admissible_candidates(self):
        """1d array of booleans. ``condorcet_admissible_candidates[c]`` is ``True`` iff candidate ``c`` is
        Condorcet-admissible, i.e. iff no candidate ``d`` has an absolute victory against ``c`` (in the sense of
        :attr:`~svvamp.Population.matrix_victories_ut_abs`).

        Examples
        --------
            >>> from svvamp import Profile
            >>> profile = Profile(preferences_rk=[[0, 1, 2], [0, 1, 2]])
            >>> profile.condorcet_admissible_candidates
            array([ True, False, False])

        See Also
        --------
        :attr:`~svvamp.Population.nb_condorcet_admissible`,
        :attr:`~svvamp.Population.exists_condorcet_admissible`,
        :attr:`~svvamp.Population.not_exists_condorcet_admissible`.
        """
        self.mylog("Compute Condorcet-admissible candidates", 1)
        return np.all(self.matrix_victories_ut_abs <= 0.5, 0)

    @cached_property
    def nb_condorcet_admissible(self):
        """Integer (number of Condorcet-admissible candidates,
        :attr:`~svvamp.Population.condorcet_admissible_candidates`).

        Examples
        --------
            >>> from svvamp import Profile
            >>> profile = Profile(preferences_rk=[[0, 1, 2], [0, 1, 2]])
            >>> profile.nb_condorcet_admissible
            1
        """
        self.mylog("Compute number of Condorcet-admissible candidates", 1)
        return np.sum(self.condorcet_admissible_candidates)

    @cached_property
    def exists_condorcet_admissible(self):
        """Boolean (``True`` iff there is at least one Condorcet-admissible candidate,
        :attr:`~svvamp.Population.condorcet_admissible_candidates`).

        Examples
        --------
            >>> from svvamp import Profile
            >>> profile = Profile(preferences_rk=[[0, 1, 2], [0, 1, 2]])
            >>> profile.exists_condorcet_admissible
            True
        """
        return np.any(self.condorcet_admissible_candidates)

    @cached_property
    def not_exists_condorcet_admissible(self):
        """Boolean (``True`` iff there is no Condorcet-admissible candidate,
        :attr:`~svvamp.Population.condorcet_admissible_candidates`).

        Examples
        --------
            >>> from svvamp import Profile
            >>> profile = Profile(preferences_rk=[[0, 1, 2], [0, 1, 2]])
            >>> profile.not_exists_condorcet_admissible
            False
        """
        return not self.exists_condorcet_admissible

    @cached_property
    def weak_condorcet_winners(self):
        """1d array of booleans. ``weak_condorcet_winners[c]`` is ``True`` iff candidate ``c`` is a weak Condorcet
        winner, i.e. iff no candidate ``d`` has a relative victory against ``c`` (in the sense of
        :attr:`~svvamp.Population.matrix_victories_ut_rel`).

        Examples
        --------
            >>> from svvamp import Profile
            >>> profile = Profile(preferences_rk=[[0, 1, 2], [0, 1, 2]])
            >>> profile.weak_condorcet_winners
            array([ True, False, False])

        See Also
        --------
        :attr:`~svvamp.Population.nb_weak_condorcet_winners`,
        :attr:`~svvamp.Population.exists_weak_condorcet_winner`,
        :attr:`~svvamp.Population.not_exists_weak_condorcet_winner`.
        """
        self.mylog("Compute weak Condorcet winners", 1)
        return np.all(self.matrix_victories_ut_rel <= 0.5, 0)

    @cached_property
    def nb_weak_condorcet_winners(self):
        """Integer (number of weak Condorcet winners, :attr:`~svvamp.Population.weak_condorcet_winners`).

        Examples
        --------
            >>> from svvamp import Profile
            >>> profile = Profile(preferences_rk=[[0, 1, 2], [0, 1, 2]])
            >>> profile.nb_weak_condorcet_winners
            1
        """
        self.mylog("Compute number of weak Condorcet winners", 1)
        return np.sum(self.weak_condorcet_winners)

    @cached_property
    def exists_weak_condorcet_winner(self):
        """Boolean (``True`` iff there is at least one weak Condorcet winner,
        :attr:`~svvamp.Population.weak_condorcet_winners`).

        Examples
        --------
            >>> from svvamp import Profile
            >>> profile = Profile(preferences_rk=[[0, 1, 2], [0, 1, 2]])
            >>> profile.exists_weak_condorcet_winner
            True
        """
        return np.any(self.weak_condorcet_winners)

    @cached_property
    def not_exists_weak_condorcet_winner(self):
        """Boolean (``True`` iff there is no weak Condorcet winner, :attr:`~svvamp.Population.weak_condorcet_winners`).

        Examples
        --------
            >>> from svvamp import Profile
            >>> profile = Profile(preferences_rk=[[0, 1, 2], [0, 1, 2]])
            >>> profile.not_exists_weak_condorcet_winner
            False
        """
        return not self.exists_weak_condorcet_winner

    @cached_property
    def condorcet_winner_rk_ctb(self):
        """Integer or ``NaN``. Candidate who has only victories in
        :attr:`~svvamp.Population.matrix_victories_rk_ctb`. If there is no such candidate, then ``NaN``.

        Examples
        --------
            >>> profile = Profile(preferences_rk=[[0, 1, 2], [0, 1, 2]])
            >>> profile.condorcet_winner_rk_ctb
            0

        No Condorcet winner:

            >>> profile = Profile(preferences_rk=[[0, 1, 2], [1, 2, 0], [2, 0, 1]])
            >>> profile.condorcet_winner_rk_ctb
            nan

        See Also
        --------
        :attr:`~svvamp.Population.exists_condorcet_winner_rk_ctb`,
        :attr:`~svvamp.Population.not_exists_condorcet_winner_rk_ctb`.
        """
        self.mylog("Compute condorcet_winner_rk_ctb", 1)
        for c in range(self.n_c):
            # The whole COLUMN must be 0.
            if np.array_equiv(self.matrix_victories_rk_ctb[:, c], 0):
                return c
        return np.nan

    @cached_property
    def exists_condorcet_winner_rk_ctb(self):
        """Boolean (``True`` iff there is a :attr:`~svvamp.Population.condorcet_winner_rk_ctb`).

        Examples
        --------
            >>> from svvamp import Profile
            >>> profile = Profile(preferences_rk=[[0, 1, 2], [0, 1, 2]])
            >>> profile.exists_condorcet_winner_rk_ctb
            True
        """
        return not np.isnan(self.condorcet_winner_rk_ctb)

    @cached_property
    def not_exists_condorcet_winner_rk_ctb(self):
        """Boolean (``True`` iff there is no :attr:`~svvamp.Population.condorcet_winner_rk_ctb`).

        Examples
        --------
            >>> from svvamp import Profile
            >>> profile = Profile(preferences_rk=[[0, 1, 2], [0, 1, 2]])
            >>> profile.not_exists_condorcet_winner_rk_ctb
            False
        """
        return np.isnan(self.condorcet_winner_rk_ctb)

    @cached_property
    def condorcet_winner_rk(self):
        """Integer or ``NaN``. Candidate who has only victories in :attr:`~svvamp.Population.matrix_victories_rk`. If
        there is no such candidate, then ``NaN``.

        Examples
        --------
            >>> from svvamp import Profile
            >>> profile = Profile(preferences_rk=[[0, 1, 2], [0, 1, 2]])
            >>> profile.condorcet_winner_rk
            0

        No Condorcet winner:

            >>> profile = Profile(preferences_rk=[[0, 1, 2], [1, 2, 0], [2, 0, 1]])
            >>> profile.condorcet_winner_rk
            nan

        See Also
        --------
        :attr:`~svvamp.Population.exists_condorcet_winner_rk`,
        :attr:`~svvamp.Population.not_exists_condorcet_winner_rk`.
        """
        self.mylog("Compute condorcet_winner_rk", 1)
        for c in range(self.n_c):
            # The whole COLUMN must be 0.
            if np.array_equiv(self.matrix_victories_rk[:, c], 0):
                return c
        return np.nan

    @cached_property
    def exists_condorcet_winner_rk(self):
        """Boolean (``True`` iff there is a :attr:`~svvamp.Population.condorcet_winner_rk`).

        Examples
        --------
            >>> from svvamp import Profile
            >>> profile = Profile(preferences_rk=[[0, 1, 2], [0, 1, 2]])
            >>> profile.exists_condorcet_winner_rk
            True
        """
        return not np.isnan(self.condorcet_winner_rk)

    @cached_property
    def not_exists_condorcet_winner_rk(self):
        """Boolean (``True`` iff there is no :attr:`~svvamp.Population.condorcet_winner_rk`).

        Examples
        --------
            >>> from svvamp import Profile
            >>> profile = Profile(preferences_rk=[[0, 1, 2], [0, 1, 2]])
            >>> profile.not_exists_condorcet_winner_rk
            False
        """
        return np.isnan(self.condorcet_winner_rk)

    @cached_property
    def condorcet_winner_ut_rel_ctb(self):
        """Integer or ``NaN``. Candidate who has only victories in
        :attr:`~svvamp.Population.matrix_victories_ut_rel_ctb`. If there is no such candidate, then ``NaN``.

        Examples
        --------
            >>> from svvamp import Profile
            >>> profile = Profile(preferences_rk=[[0, 1, 2], [0, 1, 2]])
            >>> profile.condorcet_winner_ut_rel_ctb
            0

        No Condorcet winner:

            >>> profile = Profile(preferences_rk=[[0, 1, 2], [1, 2, 0], [2, 0, 1]])
            >>> profile.condorcet_winner_ut_rel_ctb
            nan

        See Also
        --------
        :attr:`~svvamp.Population.exists_condorcet_winner_ut_rel_ctb`,
        :attr:`~svvamp.Population.not_exists_condorcet_winner_ut_rel_ctb`.
        """
        self.mylog("Compute condorcet_winner_ut_rel_ctb", 1)
        for c in range(self.n_c):
            # The whole COLUMN must be 0.
            if np.array_equiv(self.matrix_victories_ut_rel_ctb[:, c], 0):
                return c
        return np.nan

    @cached_property
    def exists_condorcet_winner_ut_rel_ctb(self):
        """Boolean (``True`` iff there is a :attr:`~svvamp.Population.condorcet_winner_ut_rel_ctb`).

        Examples
        --------
            >>> from svvamp import Profile
            >>> profile = Profile(preferences_rk=[[0, 1, 2], [0, 1, 2]])
            >>> profile.exists_condorcet_winner_ut_rel_ctb
            True
        """
        return not np.isnan(self.condorcet_winner_ut_rel_ctb)

    @cached_property
    def not_exists_condorcet_winner_ut_rel_ctb(self):
        """Boolean (``True`` iff there is no :attr:`~svvamp.Population.condorcet_winner_ut_rel_ctb`).

        Examples
        --------
            >>> from svvamp import Profile
            >>> profile = Profile(preferences_rk=[[0, 1, 2], [0, 1, 2]])
            >>> profile.not_exists_condorcet_winner_ut_rel_ctb
            False
        """
        return np.isnan(self.condorcet_winner_ut_rel_ctb)

    @cached_property
    def condorcet_winner_ut_rel(self):
        """Integer or ``NaN``. Candidate who has only victories in
        :attr:`~svvamp.Population.matrix_victories_ut_rel`. If there is no such candidate, then ``NaN``.

        Examples
        --------
            >>> from svvamp import Profile
            >>> profile = Profile(preferences_rk=[[0, 1, 2], [0, 1, 2]])
            >>> profile.condorcet_winner_ut_rel
            0

        No Condorcet winner:

            >>> profile = Profile(preferences_rk=[[0, 1, 2], [1, 2, 0], [2, 0, 1]])
            >>> profile.condorcet_winner_ut_rel
            nan

        See Also
        --------
        :attr:`~svvamp.Population.exists_condorcet_winner_ut_rel`,
        :attr:`~svvamp.Population.not_exists_condorcet_winner_ut_rel`.
        """
        self.mylog("Compute condorcet_winner_ut_rel", 1)
        for c in range(self.n_c):
            # The whole COLUMN must be 0.
            if np.array_equiv(self.matrix_victories_ut_rel[:, c], 0):
                return c
        return np.nan

    @cached_property
    def exists_condorcet_winner_ut_rel(self):
        """Boolean (``True`` iff there is a :attr:`~svvamp.Population.condorcet_winner_ut_rel`).

        Examples
        --------
            >>> from svvamp import Profile
            >>> profile = Profile(preferences_rk=[[0, 1, 2], [0, 1, 2]])
            >>> profile.exists_condorcet_winner_ut_rel
            True
        """
        return not np.isnan(self.condorcet_winner_ut_rel)

    @cached_property
    def not_exists_condorcet_winner_ut_rel(self):
        """Boolean (``True`` iff there is no :attr:`~svvamp.Population.condorcet_winner_ut_rel`).

        Examples
        --------
            >>> from svvamp import Profile
            >>> profile = Profile(preferences_rk=[[0, 1, 2], [0, 1, 2]])
            >>> profile.not_exists_condorcet_winner_ut_rel
            False
        """
        return np.isnan(self.condorcet_winner_ut_rel)

    @cached_property
    def condorcet_winner_ut_abs_ctb(self):
        """Integer or ``NaN``. Candidate who has only victories in
        :attr:`~svvamp.Population.matrix_victories_ut_abs_ctb`. If there is no such candidate, then ``NaN``.

        Examples
        --------
            >>> from svvamp import Profile
            >>> profile = Profile(preferences_rk=[[0, 1, 2], [0, 1, 2]])
            >>> profile.condorcet_winner_ut_abs_ctb
            0

        No Condorcet winner:

            >>> profile = Profile(preferences_rk=[[0, 1, 2], [1, 2, 0], [2, 0, 1]])
            >>> profile.condorcet_winner_ut_abs_ctb
            nan

        See Also
        --------
        :attr:`~svvamp.Population.exists_condorcet_winner_ut_abs_ctb`,
        :attr:`~svvamp.Population.not_exists_condorcet_winner_ut_abs_ctb`
        """
        self.mylog("Compute condorcet_winner_ut_abs_ctb", 1)
        for c in range(self.n_c):
            if np.array_equiv(self.matrix_victories_ut_abs_ctb[c, np.array(range(self.n_c)) != c], 1):
                return c
        return np.nan

    @cached_property
    def exists_condorcet_winner_ut_abs_ctb(self):
        """Boolean (``True`` iff there is a :attr:`~svvamp.Population.condorcet_winner_ut_abs_ctb`).

        Examples
        --------
            >>> from svvamp import Profile
            >>> profile = Profile(preferences_rk=[[0, 1, 2], [0, 1, 2]])
            >>> profile.exists_condorcet_winner_ut_abs_ctb
            True
        """
        return not np.isnan(self.condorcet_winner_ut_abs_ctb)

    @cached_property
    def not_exists_condorcet_winner_ut_abs_ctb(self):
        """Boolean (``True`` iff there is no :attr:`~svvamp.Population.condorcet_winner_ut_abs_ctb`).

        Examples
        --------
            >>> from svvamp import Profile
            >>> profile = Profile(preferences_rk=[[0, 1, 2], [0, 1, 2]])
            >>> profile.not_exists_condorcet_winner_ut_abs_ctb
            False
        """
        return np.isnan(self.condorcet_winner_ut_abs_ctb)

    @cached_property
    def condorcet_winner_ut_abs(self):
        """Integer or ``NaN``. Candidate who has only victories in
        :attr:`~svvamp.Population.matrix_victories_ut_abs`. If there is no such candidate, then ``NaN``.

        Examples
        --------
            >>> from svvamp import Profile
            >>> profile = Profile(preferences_rk=[[0, 1, 2], [0, 1, 2]])
            >>> profile.condorcet_winner_ut_abs
            0

        No Condorcet winner:

            >>> profile = Profile(preferences_rk=[[0, 1, 2], [1, 2, 0], [2, 0, 1]])
            >>> profile.condorcet_winner_ut_abs
            nan

        See Also
        --------
        :attr:`~svvamp.Population.exists_condorcet_winner_ut_abs`,
        :attr:`~svvamp.Population.not_exists_condorcet_winner_ut_abs`.
        """
        self.mylog("Compute Condorcet winner", 1)
        for c in range(self.n_c):
            if np.array_equiv(self.matrix_victories_ut_abs[c, np.array(range(self.n_c)) != c], 1):
                return c
        return np.nan

    @cached_property
    def exists_condorcet_winner_ut_abs(self):
        """Boolean (``True`` iff there is a :attr:`~svvamp.Population.condorcet_winner_ut_abs`).

        Examples
        --------
            >>> from svvamp import Profile
            >>> profile = Profile(preferences_rk=[[0, 1, 2], [0, 1, 2]])
            >>> profile.exists_condorcet_winner_ut_abs
            True
        """
        return not np.isnan(self.condorcet_winner_ut_abs)

    @cached_property
    def not_exists_condorcet_winner_ut_abs(self):
        """Boolean (``True`` iff there is no :attr:`~svvamp.Population.condorcet_winner_ut_abs`).

        Examples
        --------
            >>> from svvamp import Profile
            >>> profile = Profile(preferences_rk=[[0, 1, 2], [0, 1, 2]])
            >>> profile.not_exists_condorcet_winner_ut_abs
            False
        """
        return np.isnan(self.condorcet_winner_ut_abs)

    # %% Resistant Condorcet winner

    @cached_property
    def resistant_condorcet_winner(self):
        """Integer or ``NaN``. Resistant Condorcet Winner. If there is no such candidate, then ``NaN``.

        A Condorcet winner ``w`` (in the sense of :attr:`~svvamp.Population.condorcet_winner_ut_abs`) is resistant
        iff in any Condorcet voting system (in the same sense), the profile is not manipulable (cf. Durand et al.).
        This is equivalent to say that for any pair ``(c, d)`` of other distinct candidates, there is a strict
        majority of voters who simultaneously:

            1) Do not prefer ``c`` to ``w``,
            2) And prefer ``w`` to ``d``.

        Examples
        --------
            >>> from svvamp import Profile
            >>> profile = Profile(preferences_rk=[[0, 1, 2], [0, 1, 2]])
            >>> profile.resistant_condorcet_winner
            0

        No resistant Condorcet winner:

            >>> profile = Profile(preferences_rk=[[0, 1, 2], [1, 2, 0], [2, 0, 1]])
            >>> profile.resistant_condorcet_winner
            nan

        See Also
        --------
        :attr:`~svvamp.Population.exists_resistant_condorcet_winner`,
        :attr:`~svvamp.Population.not_exists_resistant_condorcet_winner`.
        """
        self.mylog("Compute Resistant Condorcet winner", 1)
        if is_resistant_condorcet(self.condorcet_winner_ut_abs, self.preferences_ut):
            return self.condorcet_winner_ut_abs
        return np.nan

    @cached_property
    def exists_resistant_condorcet_winner(self):
        """Boolean (``True`` iff there is a :attr:`~svvamp.Population.resistant_condorcet_winner`).

        Examples
        --------
            >>> from svvamp import Profile
            >>> profile = Profile(preferences_rk=[[0, 1, 2], [0, 1, 2]])
            >>> profile.exists_resistant_condorcet_winner
            True
        """
        return not np.isnan(self.resistant_condorcet_winner)

    @cached_property
    def not_exists_resistant_condorcet_winner(self):
        """Boolean (``True`` iff there is no :attr:`~svvamp.Population.resistant_condorcet_winner`).

        Examples
        --------
            >>> from svvamp import Profile
            >>> profile = Profile(preferences_rk=[[0, 1, 2], [0, 1, 2]])
            >>> profile.not_exists_resistant_condorcet_winner
            False
        """
        return np.isnan(self.resistant_condorcet_winner)

    @cached_property
    def threshold_c_prevents_w_condorcet_ut_abs(self):
        """2d array of integers. Threshold for ``c``-manipulators to prevent ``w`` from being a Condorcet winner (in
        the sense of :attr:`~svvamp.Population.condorcet_winner_ut_abs`).

        Intuitively, the question is the following: in an election where ``w`` is the winner, how many
        ``c``-manipulators are needed to prevent ``w`` from  being a Condorcet winner?

        We start with the sub-population of :math:`n_s` 'sincere' voters, i.e. not preferring ``c`` to ``w``. The
        precise question is: how many ``c``-manipulators :math:`n_m` must we add in order to create a non-victory for
        ``w`` against some candidate ``d`` :math:`\\neq` ``w`` (possibly ``c`` herself)?

        In the following, :math:`| c > d |` denotes the number of voters who strictly prefer candidate ``c`` to
        ``d``. We need:

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

        ``threshold_c_prevents_w_Condorcet_ut_abs[c, w]`` :math:`= 2 \\cdot \\min_{d \\neq w} |w \geq c \\text{ and }
        w > d| - |w \geq c|`.

        If this result is negative, it means that even without ``c``-manipulators, ``w`` is not a Condorcet winner.
        In that case, threshold is set to 0 instead.

        Examples
        --------
            >>> from svvamp import Profile
            >>> profile = Profile(preferences_rk=[[0, 1, 2], [0, 1, 2]])
            >>> profile.threshold_c_prevents_w_condorcet_ut_abs
            array([[0, 0, 0],
                   [2, 0, 0],
                   [2, 0, 0]])
        """
        self.mylog("Compute threshold_c_prevents_w_Condorcet_ut_abs", 1)
        result = np.full((self.n_c, self.n_c), 42)
        for w in range(self.n_c):
            for c in range(self.n_c):
                if c == w:
                    result[c, w] = 0
                    continue
                v_does_not_prefer_c_to_w = (self.preferences_ut[:, w] >= self.preferences_ut[:, c])
                for d in range(self.n_c):
                    if d == w:
                        continue
                        # But d == c is allowed (useful if self.C == 2)
                    v_prefers_w_to_d = (self.preferences_ut[:, w] > self.preferences_ut[:, d])
                    threshold_c_makes_w_not_win_against_d = (
                        np.multiply(2, np.sum(np.logical_and(v_does_not_prefer_c_to_w, v_prefers_w_to_d)))
                        - np.sum(v_does_not_prefer_c_to_w)
                    )
                    result[c, w] = np.minimum(result[c, w], threshold_c_makes_w_not_win_against_d)
                result = np.maximum(result, 0)
        return result

    # %% Smith Set

    @cached_property
    def smith_set_rk_ctb(self):
        """List. Smith set computed from :attr:`~svvamp.Population.matrix_victories_rk_ctb`.

        Examples
        --------
            >>> from svvamp import Profile
            >>> profile = Profile(preferences_rk=[[0, 1, 2, 3], [1, 2, 0, 3], [2, 0, 1, 3]])
            >>> profile.smith_set_rk_ctb
            [0, 1, 2]
        """
        return matrix_victories_to_smith_set(self.matrix_victories_rk_ctb)

    @cached_property
    def smith_set_rk(self):
        """List. Smith set computed from :attr:`~svvamp.Population.matrix_victories_rk`.

        Examples
        --------
            >>> from svvamp import Profile
            >>> profile = Profile(preferences_rk=[[0, 1, 2, 3], [1, 2, 0, 3], [2, 0, 1, 3]])
            >>> profile.smith_set_rk
            [0, 1, 2]
        """
        return matrix_victories_to_smith_set(self.matrix_victories_rk)

    @cached_property
    def smith_set_ut_rel_ctb(self):
        """List. Smith set computed from :attr:`~svvamp.Population.matrix_victories_ut_rel_ctb`.

        Examples
        --------
            >>> from svvamp import Profile
            >>> profile = Profile(preferences_rk=[[0, 1, 2, 3], [1, 2, 0, 3], [2, 0, 1, 3]])
            >>> profile.smith_set_ut_rel_ctb
            [0, 1, 2]
        """
        return matrix_victories_to_smith_set(self.matrix_victories_ut_rel_ctb)

    @cached_property
    def smith_set_ut_rel(self):
        """List. Smith set computed from :attr:`~svvamp.Population.matrix_victories_ut_rel`.

        Examples
        --------
            >>> from svvamp import Profile
            >>> profile = Profile(preferences_rk=[[0, 1, 2, 3], [1, 2, 0, 3], [2, 0, 1, 3]])
            >>> profile.smith_set_ut_rel
            [0, 1, 2]
        """
        return matrix_victories_to_smith_set(self.matrix_victories_ut_rel)

    @cached_property
    def smith_set_ut_abs_ctb(self):
        """List. Smith set computed from :attr:`~svvamp.Population.matrix_victories_ut_abs_ctb`.

        Examples
        --------
            >>> from svvamp import Profile
            >>> profile = Profile(preferences_rk=[[0, 1, 2, 3], [1, 2, 0, 3], [2, 0, 1, 3]])
            >>> profile.smith_set_ut_abs_ctb
            [0, 1, 2]
        """
        return matrix_victories_to_smith_set(self.matrix_victories_ut_abs_ctb)

    @cached_property
    def smith_set_ut_abs(self):
        """List. Smith set computed from :attr:`~svvamp.Population.matrix_victories_ut_abs`.

        Examples
        --------
            >>> from svvamp import Profile
            >>> profile = Profile(preferences_rk=[[0, 1, 2, 3], [1, 2, 0, 3], [2, 0, 1, 3]])
            >>> profile.smith_set_ut_abs
            [0, 1, 2]
        """
        return matrix_victories_to_smith_set(self.matrix_victories_ut_abs)

    # %% Majority favorite

    @cached_property
    def majority_favorite_rk_ctb(self):
        """Integer or ``NaN``. Candidate who has strictly more than n_v/2 in
        :attr:`~svvamp.Population.plurality_scores_rk`. Can also be candidate 0 with exactly n_v/2. If there is no such
        candidate, then ``NaN``.

        Examples
        --------
            >>> from svvamp import Profile
            >>> profile = Profile(preferences_rk=[[0, 1, 2], [0, 1, 2]])
            >>> profile.majority_favorite_rk_ctb
            0

        No majority favorite:

            >>> profile = Profile(preferences_rk=[[0, 1, 2], [1, 2, 0], [2, 0, 1]])
            >>> profile.majority_favorite_rk_ctb
            nan

        See Also
        --------
        :attr:`~svvamp.Population.exists_majority_favorite_rk_ctb`,
        :attr:`~svvamp.Population.not_exists_majority_favorite_rk_ctb`.
        """
        self.mylog("Compute majority favorite (rk_ctb)", 1)
        c = np.argmax(self.plurality_scores_rk)
        score_c = self.plurality_scores_rk[c]
        if score_c > self.n_v / 2 or (c == 0 and score_c == self.n_v / 2):
            return c
        return np.nan

    @cached_property
    def exists_majority_favorite_rk_ctb(self):
        """Boolean (``True`` iff there is a :attr:`~svvamp.Population.majority_favorite_rk_ctb`).

        Examples
        --------
            >>> from svvamp import Profile
            >>> profile = Profile(preferences_rk=[[0, 1, 2], [0, 1, 2]])
            >>> profile.exists_majority_favorite_rk_ctb
            True
        """
        return not np.isnan(self.majority_favorite_rk_ctb)

    @cached_property
    def not_exists_majority_favorite_rk_ctb(self):
        """Boolean (``True`` iff there is no :attr:`~svvamp.Population.majority_favorite_rk_ctb`).

        Examples
        --------
            >>> from svvamp import Profile
            >>> profile = Profile(preferences_rk=[[0, 1, 2], [0, 1, 2]])
            >>> profile.not_exists_majority_favorite_rk_ctb
            False
        """
        return np.isnan(self.majority_favorite_rk_ctb)

    @cached_property
    def majority_favorite_ut_ctb(self):
        """Integer or ``NaN``. Candidate who has strictly more than n_v/2 in
        :attr:`~svvamp.Population.plurality_scores_ut`. Can also be candidate 0 with exactly n_v/2. If there is no
        such candidate, then ``NaN``.

        Examples
        --------
            >>> from svvamp import Profile
            >>> profile = Profile(preferences_rk=[[0, 1, 2], [0, 1, 2]])
            >>> profile.majority_favorite_ut_ctb
            0

        No majority favorite:

            >>> profile = Profile(preferences_rk=[[0, 1, 2], [1, 2, 0], [2, 0, 1]])
            >>> profile.majority_favorite_ut_ctb
            nan

        See Also
        --------
        :attr:`~svvamp.Population.exists_majority_favorite_ut_ctb`,
        :attr:`~svvamp.Population.not_exists_majority_favorite_ut_ctb`.
        """
        self.mylog("Compute majority favorite (ut_ctb)", 1)
        c = np.argmax(self.plurality_scores_ut)
        score_c = self.plurality_scores_ut[c]
        if score_c > self.n_v / 2 or (c == 0 and score_c == self.n_v / 2):
            return c
        return np.nan

    @cached_property
    def exists_majority_favorite_ut_ctb(self):
        """Boolean (``True`` iff there is a :attr:`~svvamp.Population.majority_favorite_ut_ctb`).

        Examples
        --------
            >>> from svvamp import Profile
            >>> profile = Profile(preferences_rk=[[0, 1, 2], [0, 1, 2]])
            >>> profile.exists_majority_favorite_ut_ctb
            True
        """
        return not np.isnan(self.majority_favorite_ut_ctb)

    @cached_property
    def not_exists_majority_favorite_ut_ctb(self):
        """Boolean (``True`` iff there is no :attr:`~svvamp.Population.majority_favorite_ut_ctb`).

        Examples
        --------
            >>> from svvamp import Profile
            >>> profile = Profile(preferences_rk=[[0, 1, 2], [0, 1, 2]])
            >>> profile.not_exists_majority_favorite_ut_ctb
            False
        """
        return np.isnan(self.majority_favorite_ut_ctb)

    @cached_property
    def majority_favorite_rk(self):
        """Integer or ``NaN``. Candidate who has strictly more than n_v/2 in
        :attr:`~svvamp.Population.plurality_scores_rk`. If there is no such candidate, then ``NaN``.

        Examples
        --------
            >>> from svvamp import Profile
            >>> profile = Profile(preferences_rk=[[0, 1, 2], [0, 1, 2]])
            >>> profile.majority_favorite_rk
            0

        No majority favorite:

            >>> profile = Profile(preferences_rk=[[0, 1, 2], [1, 2, 0], [2, 0, 1]])
            >>> profile.majority_favorite_rk
            nan

        See Also
        --------
        :attr:`~svvamp.Population.exists_majority_favorite_rk`,
        :attr:`~svvamp.Population.not_exists_majority_favorite_rk`.
        """
        self.mylog("Compute majority favorite (rk)", 1)
        c = np.argmax(self.plurality_scores_rk)
        score_c = self.plurality_scores_rk[c]
        if score_c > self.n_v / 2:
            return c
        return np.nan

    @cached_property
    def exists_majority_favorite_rk(self):
        """Boolean (``True`` iff there is a :attr:`~svvamp.Population.majority_favorite_rk`).

        Examples
        --------
            >>> from svvamp import Profile
            >>> profile = Profile(preferences_rk=[[0, 1, 2], [0, 1, 2]])
            >>> profile.exists_majority_favorite_rk
            True
        """
        return not np.isnan(self.majority_favorite_rk)

    @cached_property
    def not_exists_majority_favorite_rk(self):
        """Boolean (``True`` iff there is no :attr:`~svvamp.Population.majority_favorite_rk`).

        Examples
        --------
            >>> from svvamp import Profile
            >>> profile = Profile(preferences_rk=[[0, 1, 2], [0, 1, 2]])
            >>> profile.not_exists_majority_favorite_rk
            False
        """
        return np.isnan(self.majority_favorite_rk)

    @cached_property
    def majority_favorite_ut(self):
        """Integer or ``NaN``. Candidate who has strictly more than n_v/2 in
        :attr:`~svvamp.Population.plurality_scores_ut`. If there is no such candidate, then ``NaN``.

        Examples
        --------
            >>> from svvamp import Profile
            >>> profile = Profile(preferences_rk=[[0, 1, 2], [0, 1, 2]])
            >>> profile.majority_favorite_ut
            0

        No majority favorite:

            >>> profile = Profile(preferences_rk=[[0, 1, 2], [1, 2, 0], [2, 0, 1]])
            >>> profile.majority_favorite_ut
            nan

        See Also
        --------
        :attr:`~svvamp.Population.exists_majority_favorite_ut`,
        :attr:`~svvamp.Population.not_exists_majority_favorite_ut`.
        """
        self.mylog("Compute majority favorite (ut)", 1)
        c = np.argmax(self.plurality_scores_ut)
        score_c = self.plurality_scores_ut[c]
        if score_c > self.n_v / 2:
            return c
        return np.nan

    @cached_property
    def exists_majority_favorite_ut(self):
        """Boolean (``True`` iff there is a :attr:`~svvamp.Population.majority_favorite_ut`).

        Examples
        --------
            >>> from svvamp import Profile
            >>> profile = Profile(preferences_rk=[[0, 1, 2], [0, 1, 2]])
            >>> profile.exists_majority_favorite_ut
            True
        """
        return not np.isnan(self.majority_favorite_ut)

    @cached_property
    def not_exists_majority_favorite_ut(self):
        """Boolean (``True`` iff there is no :attr:`~svvamp.Population.majority_favorite_ut`).

        Examples
        --------
            >>> from svvamp import Profile
            >>> profile = Profile(preferences_rk=[[0, 1, 2], [0, 1, 2]])
            >>> profile.not_exists_majority_favorite_ut
            False
        """
        return np.isnan(self.majority_favorite_ut)

    # %% Total utilities

    @cached_property
    def total_utility_c(self):
        """1d array of numbers. ``total_utility_c[c]`` is the total utility for candidate ``c`` (i.e. the sum of
        ``c``'s column in matrix :attr:`~svvamp.Population.preferences_ut`).

        Examples
        --------
            >>> from svvamp import Profile
            >>> profile = Profile(preferences_ut=[[5, 1, 2], [4, 10, 1]])
            >>> profile.total_utility_c
            array([ 9, 11,  3])
        """
        self.mylog("Compute total utility of candidates", 1)
        return np.sum(self.preferences_ut, 0)

    @cached_property
    def total_utility_min(self):
        """Float. ``total_utility_min`` is the minimum of :attr:`~svvamp.Population.total_utility_c`.

        Examples
        --------
            >>> from svvamp import Profile
            >>> profile = Profile(preferences_ut=[[5, 1, 2], [4, 10, 1]])
            >>> profile.total_utility_min
            3
        """
        self.mylog("Compute total_utility_min", 1)
        return np.min(self.total_utility_c)

    @cached_property
    def total_utility_max(self):
        """Float. ``total_utility_max`` is the maximum of :attr:`~svvamp.Population.total_utility_c`.

        Examples
        --------
            >>> from svvamp import Profile
            >>> profile = Profile(preferences_ut=[[5, 1, 2], [4, 10, 1]])
            >>> profile.total_utility_max
            11
        """
        self.mylog("Compute total_utility_max", 1)
        return np.max(self.total_utility_c)

    @cached_property
    def total_utility_mean(self):
        """Float. ``total_utility_mean`` is the mean of :attr:`~svvamp.Population.total_utility_c`.

        Examples
        --------
            >>> from svvamp import Profile
            >>> profile = Profile(preferences_ut=[[5, 1, 2], [4, 10, 1]])
            >>> profile.total_utility_mean == (9 + 11 + 3) / 3
            True
        """
        self.mylog("Compute total_utility_mean", 1)
        return np.mean(self.total_utility_c)

    @cached_property
    def total_utility_std(self):
        """Float. ``total_utility_std`` is the standard deviation of :attr:`~svvamp.Population.total_utility_c`.

        Examples
        --------
            >>> from svvamp import Profile
            >>> import numpy as np
            >>> profile = Profile(preferences_ut=[[5, 1, 2], [4, 10, 1]])
            >>> profile.total_utility_std == np.std([9, 11, 3])
            True
        """
        self.mylog("Compute total_utility_std", 1)
        return np.std(self.total_utility_c, ddof=0)

    # %% Mean utilities

    @cached_property
    def mean_utility_c(self):
        """1d array of floats. ``mean_utility_c[c]`` is the mean utility for candidate ``c`` (i.e. the mean of
        ``c``'s column in matrix :attr:`~svvamp.Population.preferences_ut`).

        Examples
        --------
            >>> from svvamp import Profile
            >>> profile = Profile(preferences_ut=[[5, 1, 2], [4, 10, 1]])
            >>> profile.mean_utility_c
            array([4.5, 5.5, 1.5])
        """
        self.mylog("Compute mean utility of candidates", 1)
        return np.mean(self.preferences_ut, 0)

    @cached_property
    def mean_utility_min(self):
        """Float. ``mean_utility_min`` is the minimum of :attr:`~svvamp.Population.mean_utility_c`.

        Examples
        --------
            >>> from svvamp import Profile
            >>> profile = Profile(preferences_ut=[[5, 1, 2], [4, 10, 1]])
            >>> profile.mean_utility_min
            1.5
        """
        self.mylog("Compute mean_utility_min", 1)
        return np.min(self.mean_utility_c)

    @cached_property
    def mean_utility_max(self):
        """Float. ``mean_utility_max`` is the maximum of :attr:`~svvamp.Population.mean_utility_c`.

        Examples
        --------
            >>> from svvamp import Profile
            >>> profile = Profile(preferences_ut=[[5, 1, 2], [4, 10, 1]])
            >>> profile.mean_utility_max
            5.5
        """
        self.mylog("Compute mean_utility_max", 1)
        return np.max(self.mean_utility_c)

    @cached_property
    def mean_utility_mean(self):
        """Float. ``mean_utility_mean`` is the mean of :attr:`~svvamp.Population.mean_utility_c`.

        Examples
        --------
            >>> from svvamp import Profile
            >>> profile = Profile(preferences_ut=[[5, 1, 2], [4, 10, 1]])
            >>> profile.mean_utility_mean == (4.5 + 5.5 + 1.5) / 3
            True
        """
        self.mylog("Compute mean_utility_mean", 1)
        return np.mean(self.mean_utility_c)

    @cached_property
    def mean_utility_std(self):
        """Float. ``mean_utility_std`` is the standard deviation of :attr:`~svvamp.Population.mean_utility_c`.

        Examples
        --------
            >>> from svvamp import Profile
            >>> import numpy as np
            >>> profile = Profile(preferences_ut=[[5, 1, 2], [4, 10, 1]])
            >>> profile.mean_utility_std == np.std([4.5, 5.5, 1.5])
            True
        """
        self.mylog("Compute mean_utility_std", 1)
        return np.std(self.mean_utility_c, ddof=0)

    # %% Borda scores

    @cached_property
    def borda_score_c_rk(self):
        """1d array of integers. ``borda_score_c_rk[c]`` is the total Borda score of candidate ``c`` (using
        :attr:`~svvamp.Population.preferences_borda_rk`, i.e. strict preferences).

        Examples
        --------
            >>> from svvamp import Profile
            >>> profile = Profile(preferences_rk=[[0, 1, 2], [0, 1, 2]])
            >>> profile.borda_score_c_rk
            array([4, 2, 0])
        """
        self.mylog("Compute Borda scores of the candidates (rankings)", 1)
        return np.sum(self.matrix_duels_rk, 1)

    @cached_property
    def borda_score_c_ut(self):
        """1d array of floats. ``borda_score_c_ut[c]`` is the total Borda score of candidate ``c`` (using
        :attr:`~svvamp.Population.preferences_borda_ut`, i.e. weak preferences).

        Examples
        --------
            >>> from svvamp import Profile
            >>> profile = Profile(preferences_rk=[[0, 1, 2], [0, 1, 2]])
            >>> profile.borda_score_c_ut
            array([4., 2., 0.])
        """
        self.mylog("Compute Borda scores of the candidates (weak orders)", 1)
        return np.sum(self.preferences_borda_ut, 0)

    @cached_property
    def _tuple_decreasing_borda_score_rk(self):
        candidates_by_decreasing_borda_score_rk = np.array(np.argsort(- self.borda_score_c_rk, kind='mergesort'),
                                                           dtype=int)
        decreasing_borda_scores_rk = self.borda_score_c_rk[candidates_by_decreasing_borda_score_rk]
        return candidates_by_decreasing_borda_score_rk, decreasing_borda_scores_rk

    @cached_property
    def candidates_by_decreasing_borda_score_rk(self):
        """1d array of integers. ``candidates_by_decreasing_borda_score_rk[k]`` is the candidate ranked ``k``\
        :sup:`th` by decreasing Borda score (using :attr:`~svvamp.Population.borda_score_c_rk`, i.e. strict
        preferences).

        For example, ``candidates_by_decreasing_borda_score_rk[0]`` is the candidate with highest Borda score (rk).

        Examples
        --------
            >>> from svvamp import Profile
            >>> profile = Profile(preferences_rk=[[0, 1, 2], [0, 1, 2]])
            >>> profile.candidates_by_decreasing_borda_score_rk
            array([0, 1, 2])
        """
        self.mylog("Compute candidates_by_decreasing_borda_score_rk", 1)
        return self._tuple_decreasing_borda_score_rk[0]

    @cached_property
    def decreasing_borda_scores_rk(self):
        """1d array of integers. ``decreasing_borda_scores_rk[k]`` is the ``k``\ :sup:`th` Borda score (using
        :attr:`~svvamp.Population.borda_score_c_rk`, i.e. strict preferences) by decreasing order.

        For example, ``decreasing_borda_scores_rk[0]`` is the highest Borda score for a candidate (rk).

        Examples
        --------
            >>> from svvamp import Profile
            >>> profile = Profile(preferences_rk=[[0, 1, 2], [0, 1, 2]])
            >>> profile.decreasing_borda_scores_rk
            array([4, 2, 0])
        """
        self.mylog("Compute decreasing_borda_scores_rk", 1)
        return self._tuple_decreasing_borda_score_rk[1]

    @cached_property
    def _tuple_decreasing_borda_score_ut(self):
        candidates_by_decreasing_borda_score_ut = np.array(np.argsort(- self.borda_score_c_ut, kind='mergesort'),
                                                           dtype=int)
        decreasing_borda_scores_ut = self.borda_score_c_ut[candidates_by_decreasing_borda_score_ut]
        return candidates_by_decreasing_borda_score_ut, decreasing_borda_scores_ut

    @cached_property
    def candidates_by_decreasing_borda_score_ut(self):
        """1d array of integers. ``candidates_by_decreasing_borda_score_ut[k]`` is the candidate ranked ``k``\
        :sup:`th` by decreasing Borda score (using :attr:`~svvamp.Population.borda_score_c_ut`, i.e. weak preferences).

        For example, ``candidates_by_decreasing_borda_score_ut[0]`` is the candidate with highest Borda score (ut).

        Examples
        --------
            >>> from svvamp import Profile
            >>> profile = Profile(preferences_rk=[[0, 1, 2], [0, 1, 2]])
            >>> profile.candidates_by_decreasing_borda_score_ut
            array([0, 1, 2])
        """
        self.mylog("Compute candidates_by_decreasing_borda_score_ut", 1)
        return self._tuple_decreasing_borda_score_ut[0]

    @cached_property
    def decreasing_borda_scores_ut(self):
        """1d array of integers. ``decreasing_borda_scores_ut[k]`` is the ``k``\ :sup:`th` Borda score (using
        :attr:`~svvamp.Population.borda_score_c_ut`, i.e. weak preferences) by decreasing order.

        For example, ``decreasing_borda_scores_ut[0]`` is the highest Borda score for a candidate (rk).

        Examples
        --------
            >>> from svvamp import Profile
            >>> profile = Profile(preferences_rk=[[0, 1, 2], [0, 1, 2]])
            >>> profile.decreasing_borda_scores_ut
            array([4., 2., 0.])
        """
        self.mylog("Compute decreasing_borda_scores_ut", 1)
        return self._tuple_decreasing_borda_score_ut[1]

    # %% Plot a population

    def plot3(self, indexes=None, normalize=True, use_labels=True):
        """Plot utilities for 3 candidates (with approval limit).

        Parameters
        ----------
        indexes : list
            List of 3 candidates. If None, defaults to [0, 1, 2].
        normalize : bool
            Cf. below.
        use_labels : bool
            If ``True``, then :attr:`~svvamp.Population.labels_candidates` is used to label
            the plot. Otherwise, candidates are simply represented by their index.

        Notes
        -----
        Each red point of the plot represents a voter ``v``. Its position is
        :attr:`~svvamp.Population.preferences_ut`\ ``[v, indexes]``. If ``normalize`` is ``True``, then each position
        is normalized before plotting so that its Euclidean norm is equal to 1.

        The equator (in blue) is the set of points :math:`\\mathbf{u}` such that :math:`\\sum {u_i}^2 = 1` and
        :math:`\\sum u_i = 0`, i.e. the unit circle of the plan that is orthogonal to the main diagonal [1, 1, 1].

        Other blue circles are the frontiers between the 6 different strict total orders on the candidates.

        Examples
        --------
        Typical usage:

            >>> initialize_random_seeds()
            >>> preferences_ut_test = np.random.randint(-5, 5, (10, 5))
            >>> profile = Profile(preferences_ut=preferences_ut_test,
            ...                   labels_candidates=['Alice', 'Bob', 'Catherine', 'Dave', 'Ellen'])
            >>> profile.plot3(indexes=[0, 1, 2])

        If `indexes` is not given, it defaults to [0, 1, 2]:

            >>> profile.plot3()

        You can ignore the labels of the candidates and simply use their numbers:

            >>> profile.plot3(use_labels=False)

        References
        ----------
        Durand et al., 'Geometry on the Utility Space'.
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
        xs = np.cos(theta) / np.sqrt(2) - np.sin(theta) / np.sqrt(6)
        ys = - np.cos(theta) / np.sqrt(2) - np.sin(theta) / np.sqrt(6)
        zs = 2 * np.sin(theta) / np.sqrt(6)
        ax.scatter(xs, ys, zs, s=7, c='b')
        # Frontiers between strict orders
        for i in range(3):
            pole = np.full(3, 1 / np.sqrt(3))
            ortho = np.ones(3)
            ortho[i] = -2
            ortho = ortho / np.sqrt(6)
            xs = np.cos(theta) * pole[0] + np.sin(theta) * ortho[0]
            ys = np.cos(theta) * pole[1] + np.sin(theta) * ortho[1]
            zs = np.cos(theta) * pole[2] + np.sin(theta) * ortho[2]
            ax.scatter(xs, ys, zs, s=7, c='b')
        # Voters
        mat_temp = np.copy(self.preferences_ut[:, _indexes]).astype(np.float)
        self.mylogm('mat_temp =', mat_temp, 3)
        if normalize:
            for v in range(mat_temp.shape[0]):
                norm = np.sqrt(np.sum(mat_temp[v, :]**2))
                if norm > 0:
                    mat_temp[v, :] /= norm
            self.mylogm('mat_temp =', mat_temp, 3)
        ax.scatter(mat_temp[:, 0], mat_temp[:, 1], mat_temp[:, 2], s=40, c='r', marker='o')
        # Axes and labels
        if use_labels:
            ax.set_xlabel(self.labels_candidates[_indexes[0]] + ' (' + str(_indexes[0]) + ')')
            ax.set_ylabel(self.labels_candidates[_indexes[1]] + ' (' + str(_indexes[1]) + ')')
            ax.set_zlabel(self.labels_candidates[_indexes[2]] + ' (' + str(_indexes[2]) + ')')
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
                        ' ' + str(_indexes[best]) + ' > ' + str(_indexes[other]) + ' > ' + str(_indexes[worst]))
        # plt.show()

    def plot4(self, indexes=None, normalize=True, use_labels=True):
        """Plot utilities for 4 candidates (without approval limit).

        Parameters
        ----------
        indexes : list
            List of 4 candidates. If None, defaults to [0, 1, 2, 3].
        normalize : bool
            Cf. below.
        use_labels : bool
            If ``True``, then :attr:`~svvamp.Population.labels_candidates` is used to label
            the plot. Otherwise, candidates are simply represented by their index.

        Notes
        -----
        Each red point of the plot represents a voter ``v``.

            * :attr:`~svvamp.Population.preferences_ut`\ ``[v, indexes]`` is sent to the hyperplane that is
              orthogonal to [1, 1, 1, 1] (by orthogonal projection), which discards information related to approval
              limit and keeps only the relative preferences between candidates.
            * The plot is done in this 3d hyperplane. In practice, we use a mirror symmetry that exchanges [1, 1, 1,
              1] and [0, 0, 0, 1]. This way, the image vector is orthogonal to [0, 0, 0, 1] and can be plotted in the
              first 3 dimensions.
            * If ``normalize`` is True, then the image vector is normalized before plotting so that its Euclidean
              norm is equal to 1.

        Blue lines are the frontiers between the 24 different strict total orders on the candidates ('permutohedron').

        Examples
        --------
        Typical usage:

            >>> initialize_random_seeds()
            >>> preferences_ut_test = np.random.randint(-5, 5, (10, 5))
            >>> profile = Profile(preferences_ut=preferences_ut_test,
            ...                   labels_candidates=['Alice', 'Bob', 'Catherine', 'Dave', 'Ellen'])
            >>> profile.plot4(indexes=[0, 1, 2, 4])

        If `indexes` is not given, it defaults to [0, 1, 2, 3]:

            >>> profile.plot4()

        You can ignore the labels of the candidates and simply use their numbers:

            >>> profile.plot4(use_labels=False)

        It is possible to avoid normalization:

            >>> profile.plot4(normalize=False)

        References
        ----------
        Durand et al., 'Geometry on the Utility Space'.
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
        mat_temp = np.copy(self.preferences_ut[:, _indexes]).astype(np.float)
        # 1. Send each voter to the hyperplane orthogonal to (1, 1, 1, 1)
        mat_temp -= np.mean(mat_temp, 1)[:, np.newaxis]
        # 2. Use symmetry by a hyperplane to exchange (1, 1, 1, 1) and (0, 0, 0, 1)
        # unitary_parallel = [1, 1, 1, 1] / sqrt(4) - [0, 0, 0, 1] then normalize.
        unitary_parallel = np.array([0.5, 0.5, 0.5, -0.5])
        # unitary_parallel = unitary_parallel / np.sqrt(np.sum(unitary_parallel**2))
        temp_parallel = np.outer(np.sum(mat_temp * unitary_parallel[np.newaxis, :], 1), unitary_parallel)
        mat_temp -= 2 * temp_parallel
        # 3. Normalize if asked
        if normalize:
            for v in range(mat_temp.shape[0]):
                norm = np.sqrt(np.sum(mat_temp[v, :]**2))
                if norm > 0:
                    mat_temp[v, :] /= norm
        # 4. Now just do not care about last coordinate, it is 0.
        ax.scatter(mat_temp[:, 0], mat_temp[:, 1], mat_temp[:, 2], s=40, c='r', marker='o')

        # Permutohedron
        xs = []
        ys = []
        zs = []

        def add_line(v1, v2):
            """v1 and v2 are vertices."""
            if normalize:
                theta_tot = np.arccos(np.sum(v1 * v2))
                ortho = v2 - np.sum(v2 * v1) * v1
                ortho = ortho / np.sqrt(np.sum(ortho**2))
                theta = np.linspace(0, theta_tot, 100)
                xs.extend(np.cos(theta) * v1[0] + np.sin(theta) * ortho[0])
                ys.extend(np.cos(theta) * v1[1] + np.sin(theta) * ortho[1])
                zs.extend(np.cos(theta) * v1[2] + np.sin(theta) * ortho[2])
            else:
                xs.extend(np.linspace(v1[0], v2[0], 50))
                ys.extend(np.linspace(v1[1], v2[1], 50))
                zs.extend(np.linspace(v1[2], v2[2], 50))

        def image(vertex):
            result = np.multiply(vertex, 2) - 1
            result = result - np.mean(result)
            the_temp_parallel = np.sum(result*unitary_parallel) * unitary_parallel
            result -= 2 * the_temp_parallel
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
                    ax.text(vertex1[0], vertex1[1], vertex1[2], '   Favorite = ' + self.labels_candidates[_indexes[i]])
                else:
                    ax.text(vertex1[0], vertex1[1], vertex1[2], '   Favorite = ' + str(_indexes[i]))
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
        # plt.show()

    # %% Demo

    def demo(self, log_depth=1):
        """Demonstrate the methods of :class:`~svvamp.Population` class.

        Parameters
        ----------
        log_depth : int
            Integer from 0 (basic info) to 3 (verbose).

        Examples
        --------
            >>> initialize_random_seeds()
            >>> preferences_ut_test = np.random.randint(-5, 5, (10, 5))
            >>> profile = Profile(preferences_ut=preferences_ut_test,
            ...                   labels_candidates=['Alice', 'Bob', 'Catherine', 'Dave', 'Ellen'])
            >>> profile.demo(log_depth=0)  # doctest: +NORMALIZE_WHITESPACE
            <BLANKLINE>
            ************************
            *   Basic properties   *
            ************************
            n_v = 10
            n_c = 5
            labels_candidates = ['Alice', 'Bob', 'Catherine', 'Dave', 'Ellen']
            preferences_ut =
            [[ 4 -2  0 -3 -1]
             [ 4  4 -5 -1  2]
             [ 0  4  3  4 -1]
             [-2 -2  2 -5 -4]
             [-2 -3  2 -3 -5]
             [-3 -2  3 -4 -2]
             [ 2  1  3  3 -4]
             [-2 -5 -2  0 -5]
             [ 1  2  2  3 -4]
             [ 0 -5 -2 -2  2]]
            preferences_borda_ut =
            [[4.  1.  3.  0.  2. ]
             [3.5 3.5 0.  1.  2. ]
             [1.  3.5 2.  3.5 0. ]
             [2.5 2.5 4.  0.  1. ]
             [3.  1.5 4.  1.5 0. ]
             [1.  2.5 4.  0.  2.5]
             [2.  1.  3.5 3.5 0. ]
             [2.5 0.5 2.5 4.  0.5]
             [1.  2.5 2.5 4.  0. ]
             [3.  0.  1.5 1.5 4. ]]
            preferences_borda_rk =
            [[4 1 3 0 2]
             [3 4 0 1 2]
             [1 4 2 3 0]
             [3 2 4 0 1]
             [3 1 4 2 0]
             [1 3 4 0 2]
             [2 1 3 4 0]
             [2 0 3 4 1]
             [1 2 3 4 0]
             [3 0 2 1 4]]
            preferences_rk =
            [[0 2 4 1 3]
             [1 0 4 3 2]
             [1 3 2 0 4]
             [2 0 1 4 3]
             [2 0 3 1 4]
             [2 1 4 0 3]
             [3 2 0 1 4]
             [3 2 0 4 1]
             [3 2 1 0 4]
             [4 0 2 3 1]]
            v_has_same_ordinal_preferences_as_previous_voter =
            [False False False False False False False False False False]
            <BLANKLINE>
            ************************
            *   Plurality scores   *
            ************************
            preferences_rk (reminder) =
            [[0 2 4 1 3]
             [1 0 4 3 2]
             [1 3 2 0 4]
             [2 0 1 4 3]
             [2 0 3 1 4]
             [2 1 4 0 3]
             [3 2 0 1 4]
             [3 2 0 4 1]
             [3 2 1 0 4]
             [4 0 2 3 1]]
            plurality_scores_rk = [1 2 3 3 1]
            majority_favorite_rk = nan
            majority_favorite_rk_ctb = nan
            <BLANKLINE>
            preferences_borda_ut (reminder) =
            [[4.  1.  3.  0.  2. ]
             [3.5 3.5 0.  1.  2. ]
             [1.  3.5 2.  3.5 0. ]
             [2.5 2.5 4.  0.  1. ]
             [3.  1.5 4.  1.5 0. ]
             [1.  2.5 4.  0.  2.5]
             [2.  1.  3.5 3.5 0. ]
             [2.5 0.5 2.5 4.  0.5]
             [1.  2.5 2.5 4.  0. ]
             [3.  0.  1.5 1.5 4. ]]
            plurality_scores_ut = [1 0 3 2 1]
            majority_favorite_ut = nan
            majority_favorite_ut_ctb = nan
            <BLANKLINE>
            ********************
            *   Borda scores   *
            ********************
            preferences_borda_rk (reminder) =
            [[4 1 3 0 2]
             [3 4 0 1 2]
             [1 4 2 3 0]
             [3 2 4 0 1]
             [3 1 4 2 0]
             [1 3 4 0 2]
             [2 1 3 4 0]
             [2 0 3 4 1]
             [1 2 3 4 0]
             [3 0 2 1 4]]
            borda_score_c_rk =
            [23 18 28 19 12]
            Remark: Borda scores above are computed with the matrix of duels.
            Check: np.sum(self.preferences_borda_rk, 0) =
            [23 18 28 19 12]
            decreasing_borda_scores_rk =
            [28 23 19 18 12]
            candidates_by_decreasing_borda_score_rk =
            [2 0 3 1 4]
            <BLANKLINE>
            preferences_borda_ut (reminder) =
            [[4.  1.  3.  0.  2. ]
             [3.5 3.5 0.  1.  2. ]
             [1.  3.5 2.  3.5 0. ]
             [2.5 2.5 4.  0.  1. ]
             [3.  1.5 4.  1.5 0. ]
             [1.  2.5 4.  0.  2.5]
             [2.  1.  3.5 3.5 0. ]
             [2.5 0.5 2.5 4.  0.5]
             [1.  2.5 2.5 4.  0. ]
             [3.  0.  1.5 1.5 4. ]]
            borda_score_c_ut =
            [23.5 18.5 27.  19.  12. ]
            decreasing_borda_scores_ut =
            [27.  23.5 19.  18.5 12. ]
            candidates_by_decreasing_borda_score_ut =
            [2 0 3 1 4]
            <BLANKLINE>
            *****************
            *   Utilities   *
            *****************
            preferences_ut (reminder) =
            [[ 4 -2  0 -3 -1]
             [ 4  4 -5 -1  2]
             [ 0  4  3  4 -1]
             [-2 -2  2 -5 -4]
             [-2 -3  2 -3 -5]
             [-3 -2  3 -4 -2]
             [ 2  1  3  3 -4]
             [-2 -5 -2  0 -5]
             [ 1  2  2  3 -4]
             [ 0 -5 -2 -2  2]]
            total_utility_c =
            [  2  -8   6  -8 -22]
            total_utility_min = -22
            total_utility_max = 6
            total_utility_mean = -6.0
            total_utility_std = 9.715966241192895
            <BLANKLINE>
            *******************************************
            *   Condorcet notions based on rankings   *
            *******************************************
            preferences_rk (reminder) =
            [[0 2 4 1 3]
             [1 0 4 3 2]
             [1 3 2 0 4]
             [2 0 1 4 3]
             [2 0 3 1 4]
             [2 1 4 0 3]
             [3 2 0 1 4]
             [3 2 0 4 1]
             [3 2 1 0 4]
             [4 0 2 3 1]]
            matrix_duels_rk =
            [[0 6 3 6 8]
             [4 0 2 5 7]
             [7 8 0 5 8]
             [4 5 5 0 5]
             [2 3 2 5 0]]
            matrix_victories_rk =
            [[0.  1.  0.  1.  1. ]
             [0.  0.  0.  0.5 1. ]
             [1.  1.  0.  0.5 1. ]
             [0.  0.5 0.5 0.  0.5]
             [0.  0.  0.  0.5 0. ]]
            condorcet_winner_rk = nan
            exists_condorcet_order_rk = False
            matrix_victories_rk_ctb =
            [[0. 1. 0. 1. 1.]
             [0. 0. 0. 1. 1.]
             [1. 1. 0. 1. 1.]
             [0. 0. 0. 0. 1.]
             [0. 0. 0. 0. 0.]]
            condorcet_winner_rk_ctb = 2
            exists_condorcet_order_rk_ctb = True
            <BLANKLINE>
            ***************************************
            *   Relative Condorcet notions (ut)   *
            ***************************************
            preferences_borda_ut (reminder) =
            [[4.  1.  3.  0.  2. ]
             [3.5 3.5 0.  1.  2. ]
             [1.  3.5 2.  3.5 0. ]
             [2.5 2.5 4.  0.  1. ]
             [3.  1.5 4.  1.5 0. ]
             [1.  2.5 4.  0.  2.5]
             [2.  1.  3.5 3.5 0. ]
             [2.5 0.5 2.5 4.  0.5]
             [1.  2.5 2.5 4.  0. ]
             [3.  0.  1.5 1.5 4. ]]
            matrix_duels_ut =
            [[0 5 3 6 8]
             [3 0 2 4 6]
             [6 7 0 4 8]
             [4 4 4 0 5]
             [2 2 2 5 0]]
            matrix_victories_ut_rel =
            [[0.  1.  0.  1.  1. ]
             [0.  0.  0.  0.5 1. ]
             [1.  1.  0.  0.5 1. ]
             [0.  0.5 0.5 0.  0.5]
             [0.  0.  0.  0.5 0. ]]
            condorcet_winner_ut_rel = nan
            exists_condorcet_order_ut_rel = False
            matrix_victories_ut_rel_ctb =
            [[0. 1. 0. 1. 1.]
             [0. 0. 0. 1. 1.]
             [1. 1. 0. 1. 1.]
             [0. 0. 0. 0. 1.]
             [0. 0. 0. 0. 0.]]
            condorcet_winner_ut_rel_ctb = 2
            exists_condorcet_order_ut_rel_ctb = True
            <BLANKLINE>
            ***************************************
            *   Absolute Condorcet notions (ut)   *
            ***************************************
            matrix_duels_ut (reminder) =
            [[0 5 3 6 8]
             [3 0 2 4 6]
             [6 7 0 4 8]
             [4 4 4 0 5]
             [2 2 2 5 0]]
            matrix_victories_ut_abs =
            [[0.  0.5 0.  1.  1. ]
             [0.  0.  0.  0.  1. ]
             [1.  1.  0.  0.  1. ]
             [0.  0.  0.  0.  0.5]
             [0.  0.  0.  0.5 0. ]]
            condorcet_admissible_candidates =
            [False False  True False False]
            nb_condorcet_admissible = 1
            weak_condorcet_winners =
            [False False  True False False]
            nb_weak_condorcet_winners = 1
            condorcet_winner_ut_abs = nan
            exists_condorcet_order_ut_abs = False
            resistant_condorcet_winner = nan
            threshold_c_prevents_w_Condorcet_ut_abs =
            [[0 0 0 2 0]
             [0 0 0 0 0]
             [2 0 0 2 0]
             [0 0 2 0 0]
             [0 0 0 1 0]]
            matrix_victories_ut_abs_ctb =
            [[0. 1. 0. 1. 1.]
             [0. 0. 0. 0. 1.]
             [1. 1. 0. 0. 1.]
             [0. 0. 0. 0. 1.]
             [0. 0. 0. 0. 0.]]
            condorcet_winner_ut_abs_ctb = nan
            exists_condorcet_order_ut_abs_ctb = False
            <BLANKLINE>
            **********************************************
            *   Implications between Condorcet notions   *
            **********************************************
            maj_fav_ut (False)             ==>            maj_fav_ut_ctb (False)
             ||          ||                                     ||           ||
             ||          V                                      V            ||
             ||         maj_fav_rk (False) ==> maj_fav_rk_ctb (False)        ||
             V                         ||       ||                           ||
            Resistant Condorcet (False)                                      ||
             ||                        ||       ||                           ||
             V                         ||       ||                           V
            Condorcet_ut_abs (False)       ==>      Condorcet_ut_abs_ctb (False)
             ||          ||            ||       ||              ||           ||
             ||          V             V        V               V            ||
             ||       Condorcet_rk (False) ==> Condorcet_rk_ctb (True)       ||
             V                                                               V
            Condorcet_ut_rel (False)       ==>      Condorcet_ut_rel_ctb (True)
             ||
             V
            Weak Condorcet (True)
             ||
             V
            Condorcet-admissible (True)
        """
        old_log_depth = self.log_depth
        self.log_depth = log_depth

        print_title("Basic properties")
        print("n_v =", self.n_v)
        print("n_c =", self.n_c)
        print("labels_candidates =", self.labels_candidates)
        printm("preferences_ut =", self.preferences_ut)
        printm("preferences_borda_ut =", self.preferences_borda_ut)
        printm("preferences_borda_rk =", self.preferences_borda_rk)
        printm("preferences_rk =", self.preferences_rk)

        printm("v_has_same_ordinal_preferences_as_previous_voter =",
               self.v_has_same_ordinal_preferences_as_previous_voter)

        print_title("Plurality scores")
        printm("preferences_rk (reminder) =", self.preferences_rk)
        print("plurality_scores_rk =", self.plurality_scores_rk)
        print("majority_favorite_rk =", self.majority_favorite_rk)
        print("majority_favorite_rk_ctb =", self.majority_favorite_rk_ctb)
        print("")
        printm("preferences_borda_ut (reminder) =", self.preferences_borda_ut)
        print("plurality_scores_ut =", self.plurality_scores_ut)
        print("majority_favorite_ut =", self.majority_favorite_ut)
        print("majority_favorite_ut_ctb =", self.majority_favorite_ut_ctb)

        print_title("Borda scores")
        printm("preferences_borda_rk (reminder) =", self.preferences_borda_rk)
        printm("borda_score_c_rk =", self.borda_score_c_rk)
        print("Remark: Borda scores above are computed with the matrix of duels.")
        printm("Check: np.sum(self.preferences_borda_rk, 0) =", np.sum(self.preferences_borda_rk, 0))
        printm("decreasing_borda_scores_rk =", self.decreasing_borda_scores_rk)
        printm("candidates_by_decreasing_borda_score_rk =", self.candidates_by_decreasing_borda_score_rk)
        print("")
        printm("preferences_borda_ut (reminder) =", self.preferences_borda_ut)
        printm("borda_score_c_ut =", self.borda_score_c_ut)
        printm("decreasing_borda_scores_ut =", self.decreasing_borda_scores_ut)
        printm("candidates_by_decreasing_borda_score_ut =", self.candidates_by_decreasing_borda_score_ut)

        print_title("Utilities")
        printm("preferences_ut (reminder) =", self.preferences_ut)
        printm("total_utility_c = ", self.total_utility_c)
        print("total_utility_min =", self.total_utility_min)
        print("total_utility_max =", self.total_utility_max)
        print("total_utility_mean =", self.total_utility_mean)
        print("total_utility_std =", self.total_utility_std)

        print_title("Condorcet notions based on rankings")
        printm("preferences_rk (reminder) =", self.preferences_rk)
        printm("matrix_duels_rk =", self.matrix_duels_rk)

        printm("matrix_victories_rk =", self.matrix_victories_rk)
        print("condorcet_winner_rk =", self.condorcet_winner_rk)
        print("exists_condorcet_order_rk =", self.exists_condorcet_order_rk)

        printm("matrix_victories_rk_ctb =", self.matrix_victories_rk_ctb)
        print("condorcet_winner_rk_ctb =", self.condorcet_winner_rk_ctb)
        print("exists_condorcet_order_rk_ctb =", self.exists_condorcet_order_rk_ctb)

        print_title("Relative Condorcet notions (ut)")
        printm("preferences_borda_ut (reminder) =", self.preferences_borda_ut)
        printm("matrix_duels_ut =", self.matrix_duels_ut)

        printm("matrix_victories_ut_rel =", self.matrix_victories_ut_rel)
        print("condorcet_winner_ut_rel =", self.condorcet_winner_ut_rel)
        print("exists_condorcet_order_ut_rel =", self.exists_condorcet_order_ut_rel)

        printm("matrix_victories_ut_rel_ctb =", self.matrix_victories_ut_rel_ctb)
        print("condorcet_winner_ut_rel_ctb =", self.condorcet_winner_ut_rel_ctb)
        print("exists_condorcet_order_ut_rel_ctb =", self.exists_condorcet_order_ut_rel_ctb)

        print_title("Absolute Condorcet notions (ut)")
        printm("matrix_duels_ut (reminder) =", self.matrix_duels_ut)
        printm("matrix_victories_ut_abs =", self.matrix_victories_ut_abs)
        printm("condorcet_admissible_candidates = ", self.condorcet_admissible_candidates)
        print("nb_condorcet_admissible =", self.nb_condorcet_admissible)
        printm("weak_condorcet_winners =", self.weak_condorcet_winners)
        print("nb_weak_condorcet_winners =", self.nb_weak_condorcet_winners)
        print("condorcet_winner_ut_abs =", self.condorcet_winner_ut_abs)
        print("exists_condorcet_order_ut_abs =", self.exists_condorcet_order_ut_abs)
        print("resistant_condorcet_winner =", self.resistant_condorcet_winner)
        printm("threshold_c_prevents_w_Condorcet_ut_abs =", self.threshold_c_prevents_w_condorcet_ut_abs)

        printm("matrix_victories_ut_abs_ctb =", self.matrix_victories_ut_abs_ctb)
        print("condorcet_winner_ut_abs_ctb =", self.condorcet_winner_ut_abs_ctb)
        print("exists_condorcet_order_ut_abs_ctb =", self.exists_condorcet_order_ut_abs_ctb)

        print_title("Implications between Condorcet notions")
        # maj_fav_ut (False)             ==>            maj_fav_ut_ctb (False)
        #  ||          ||                                     ||           ||
        #  ||          V                                      V            ||
        #  ||         maj_fav_rk (False) ==> maj_fav_rk_ctb (False)        ||
        #  V                         ||       ||                           ||
        # Resistant Condorcet (False)                                      ||
        #  ||                        ||       ||                           ||
        #  V                         ||       ||                           V
        # Condorcet_ut_abs (False)       ==>      Condorcet_ut_abs_ctb (False)
        #  ||          ||            ||       ||              ||           ||
        #  ||          V             V        V               V            ||
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
            return '(True) ' if equal_true(value) else '(False)'
        print('maj_fav_ut ' + display_bool(self.majority_favorite_ut) + '             ==>            '
              + 'maj_fav_ut_ctb ' + display_bool(self.majority_favorite_ut_ctb))
        print(' ||          ||                                     ||           ||')
        print(' ||          V                                      V            ||')
        print(' ||         maj_fav_rk ' + display_bool(self.majority_favorite_rk) + ' ==> '
              + 'maj_fav_rk_ctb ' + display_bool(self.majority_favorite_rk_ctb) + '        ||')
        print(' V                         ||       ||                           ||')
        print('Resistant Condorcet ' + display_bool(self.resistant_condorcet_winner)
              + '                                      ||')
        print(' ||                        ||       ||                           ||')
        print(' V                         ||       ||                           V')
        print('Condorcet_ut_abs ' + display_bool(self.condorcet_winner_ut_abs) + '       ==>      '
              'Condorcet_ut_abs_ctb ' + display_bool(self.condorcet_winner_ut_abs_ctb) + '')
        print(' ||          ||            ||       ||              ||           ||')
        print(' ||          V             V        V               V            ||')
        print(' ||       Condorcet_rk ' + display_bool(self.exists_condorcet_winner_rk) +
              ' ==> Condorcet_rk_ctb ' + display_bool(self.exists_condorcet_winner_rk_ctb) + '      ||')
        print(' V                                                               V')
        print('Condorcet_ut_rel ' + display_bool(self.exists_condorcet_winner_ut_rel) +
              '       ==>      Condorcet_ut_rel_ctb ' + display_bool(self.exists_condorcet_winner_ut_rel_ctb))
        print(' ||')
        print(' V')
        print('Weak Condorcet ' + display_bool(self.exists_weak_condorcet_winner))
        print(' ||')
        print(' V')
        print('Condorcet-admissible ' + display_bool(self.exists_condorcet_admissible))

        self.log_depth = old_log_depth

    # %% For developers

    def to_doctest_string(self, ut=True, rk=True):
        """Convert to string, in the doctest format.

        Parameters
        ----------
        ut : bool
            Whether to print `preferences_ut`.
        rk : bool
            Whether to print `preferences_rk`.

        Returns
        -------
        str
            A string that can be copied-pasted to make a doctest.

        Examples
        --------
        >>> profile = Profile(preferences_rk=[[0, 1], [0, 1]])
        >>> print('.' + profile.to_doctest_string())  # doctest: +NORMALIZE_WHITESPACE
            . >>> profile = Profile(preferences_ut=[
            ...     [1, 0],
            ...     [1, 0],
            ... ], preferences_rk=[
            ...     [0, 1],
            ...     [0, 1],
            ... ])
        """
        s = '            >>> profile = Profile('
        arguments = []
        if ut:
            argument = ''
            argument += 'preferences_ut=[\n'
            argument += (
                repr(self.preferences_ut)
                    .replace('array([', '            ...     ')
                    .replace('\n       ', '\n            ...     ')
                    .replace('])', ',')
            )
            argument += '\n            ... ]'
            arguments.append(argument)
        if rk:
            argument = ''
            argument += 'preferences_rk=[\n'
            argument += (
                repr(self.preferences_rk)
                    .replace('array([', '            ...     ')
                    .replace('\n       ', '\n            ...     ')
                    .replace('])', ',')
            )
            argument += '\n            ... ]'
            arguments.append(argument)
        s += ', '.join(arguments)
        s += ')'
        return s
