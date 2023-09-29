# -*- coding: utf-8 -*-
"""
Created on oct. 21, 2014, 09:54
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
from svvamp.utils.util_cache import cached_property
from svvamp.preferences.profile import Profile
from svvamp.preferences.plurality_elimination_engine_profile_um import PluralityEliminationEngineProfileUM


class ProfileUM(Profile):
    """
    Profile used for Unison Manipulation.

    This class is used internally by SVVAMP. It is not intended for the end user.

    Parameters
    ----------
    profile_s: Profile
        Profile of all sincere voters (i.e. those who do not prefer `c` to `w`).
    n_m: int
        Number of manipulators.
    ballot_ut: List, optional
        Ballot, seen as utilities.
    ballot_rk: List, optional
        Ballot, seen as rankings.
    ballot_borda_rk = List, optional.
        Ballot, seen as Borda points.

    Examples
    --------
        >>> profile_s = Profile(preferences_ut=[[1., .5, 0.], [.5, 1., 0.]])

    Define with `ballot_ut`:

        >>> profile_s.preferences_ut
        array([[1. , 0.5, 0. ],
               [0.5, 1. , 0. ]])
        >>> profile_um = ProfileUM(profile_s=profile_s, n_m=2, ballot_ut=[0., 1., 0.])
        >>> profile_um.preferences_ut
        array([[1. , 0.5, 0. ],
               [0.5, 1. , 0. ],
               [0. , 1. , 0. ],
               [0. , 1. , 0. ]])

    Define with `ballot_rk`:

        >>> profile_s.preferences_rk
        array([[0, 1, 2],
               [1, 0, 2]])
        >>> profile_um = ProfileUM(profile_s=profile_s, n_m=2, ballot_rk=[1, 2, 0])
        >>> profile_um.preferences_rk
        array([[0, 1, 2],
               [1, 0, 2],
               [1, 2, 0],
               [1, 2, 0]])

    Define with `ballot_borda_rk`:

        >>> profile_s.preferences_borda_rk
        array([[2, 1, 0],
               [1, 2, 0]])
        >>> profile_um = ProfileUM(profile_s=profile_s, n_m=2, ballot_borda_rk=[0, 2, 1])
        >>> profile_um.preferences_borda_rk
        array([[2, 1, 0],
               [1, 2, 0],
               [0, 2, 1],
               [0, 2, 1]])

    But you need at least one of them:
        >>> profile_um = ProfileUM(profile_s=profile_s, n_m=2)
        Traceback (most recent call last):
        ValueError: Please provide at least ballot_ut, ballot_rk or ballot_borda_rk.
    """

    def __init__(self, profile_s, n_m, ballot_ut=None, ballot_rk=None, ballot_borda_rk=None):
        self.profile_s = profile_s
        self.n_m = n_m
        if ballot_ut is None:
            self._ballot_ut_input = None
        else:
            self._ballot_ut_input = np.array(ballot_ut)
        if ballot_rk is None:
            self._ballot_rk_input = None
        else:
            self._ballot_rk_input = np.array(ballot_rk)
        if ballot_borda_rk is None:
            self._ballot_borda_rk_input = None
        else:
            self._ballot_borda_rk_input = np.array(ballot_borda_rk)
        if ballot_ut is None and ballot_rk is None and ballot_borda_rk is None:
            raise ValueError('Please provide at least ballot_ut, ballot_rk or ballot_borda_rk.')
        super().__init__(preferences_ut=None, preferences_borda_rk=None, sort_voters=False)

    # %% Basic stuff

    @cached_property
    def n_c(self):
        return self.profile_s.n_c

    @cached_property
    def n_v(self):
        return self.profile_s.n_v + self.n_m

    @cached_property
    def labels_candidates(self):
        """List: labels of the candidates.

        Examples
        --------
            >>> profile_s = Profile(preferences_ut=[[1., .5, 0.], [.5, 1., 0.]])
            >>> profile_um = ProfileUM(profile_s=profile_s, n_m=2, ballot_ut=[0., 1., 0.])
            >>> profile_um.labels_candidates
            ['0', '1', '2']
        """
        return self.profile_s.labels_candidates

    # %% Ballot of the manipulators

    @cached_property
    def ballot_rk(self):
        """List: ballot of the manipulators, in 'rk' format.

        Examples
        --------
            >>> profile_s = Profile(preferences_ut=[[1., .5, 0.], [.5, 1., 0.]])

            >>> profile_um = ProfileUM(profile_s=profile_s, n_m=2, ballot_rk=[1, 2, 0])
            >>> profile_um.ballot_rk
            array([1, 2, 0])

            >>> profile_um = ProfileUM(profile_s=profile_s, n_m=2, ballot_borda_rk=[0, 2, 1])
            >>> profile_um.ballot_rk
            array([1, 2, 0])

            >>> profile_um = ProfileUM(profile_s=profile_s, n_m=2, ballot_ut=[0., 1., 0.])
            >>> profile_um.ballot_rk
            Traceback (most recent call last):
            ValueError: You should not rely on the implicit conversion from utility to ranking to compute UM.
        """
        if self._ballot_rk_input is not None:
            return self._ballot_rk_input
        if self._ballot_borda_rk_input is not None:
            return np.array(sorted(range(self.n_c), key=self._ballot_borda_rk_input.__getitem__, reverse=True))
        # if self._ballot_ut_input is not None:
        raise ValueError('You should not rely on the implicit conversion from utility to ranking '
                         'to compute UM.')

    @cached_property
    def ballot_borda_rk(self):
        if self._ballot_borda_rk_input is not None:
            return self._ballot_borda_rk_input
        else:
            return - np.array(np.argsort(self._ballot_rk_input), dtype=int) + self.n_c - 1

    @cached_property
    def ballot_ut(self):
        if self._ballot_ut_input is not None:
            return self._ballot_ut_input
        else:
            return self.ballot_borda_rk

    @cached_property
    def ballot_borda_ut(self):
        """List: ballot of the manipulators, in 'borda ut' format.

        Examples
        --------
            >>> profile_s = Profile(preferences_ut=[[1., .5, 0.], [.5, 1., 0.]])
            >>> profile_um = ProfileUM(profile_s=profile_s, n_m=2, ballot_ut=[0., 1., 0.])
            >>> profile_um.ballot_borda_ut
            array([0.5, 2. , 0.5])
        """
        return np.array([
            np.sum(0.5 * (self.ballot_ut[c] >= self.ballot_ut) + 0.5 * (self.ballot_ut[c] > self.ballot_ut)) - 0.5
            for c in range(self.n_c)
        ])

    # %% Preferences

    @cached_property
    def preferences_ut(self):
        return np.concatenate((
            self.profile_s.preferences_ut,
            np.outer(np.ones(self.n_m, dtype=int), self.ballot_ut)
        ))

    @cached_property
    def preferences_borda_rk(self):
        return np.concatenate((
            self.profile_s.preferences_borda_rk,
            np.outer(np.ones(self.n_m, dtype=int), self.ballot_borda_rk)
        ))

    @cached_property
    def preferences_rk(self):
        return np.concatenate((
            self.profile_s.preferences_rk,
            np.outer(np.ones(self.n_m, dtype=int), self.ballot_rk)
        ))

    @cached_property
    def preferences_borda_ut(self):
        """List: preferences, in 'borda ut' format.

        Examples
        --------
            >>> profile_s = Profile(preferences_ut=[[1., .5, 0.], [.5, 1., 0.]])
            >>> profile_um = ProfileUM(profile_s=profile_s, n_m=2, ballot_ut=[0., 1., 0.])
            >>> profile_um.preferences_borda_ut
            array([[2. , 1. , 0. ],
                   [1. , 2. , 0. ],
                   [0.5, 2. , 0.5],
                   [0.5, 2. , 0.5]])
        """
        return np.concatenate((
            self.profile_s.preferences_borda_ut,
            np.outer(np.ones(self.n_m, dtype=int), self.ballot_borda_ut)
        ))

    # %% Matrix of duels

    @cached_property
    def matrix_duels_ut(self):
        return (
            self.profile_s.matrix_duels_ut
            + self.n_m * (self.ballot_ut[:, np.newaxis] > self.ballot_ut[np.newaxis, :])
        )

    @cached_property
    def matrix_duels_rk(self):
        return (
            self.profile_s.matrix_duels_rk
            + self.n_m * (self.ballot_borda_rk[:, np.newaxis] > self.ballot_borda_rk[np.newaxis, :])
        )

    @cached_property
    def plurality_scores_rk(self):
        plurality_scores_rk = np.zeros(self.n_c)
        plurality_scores_rk[self.ballot_rk[0]] = self.n_m
        plurality_scores_rk += self.profile_s.plurality_scores_rk
        return plurality_scores_rk

    @cached_property
    def plurality_scores_ut(self):
        """List: plurality scores, relying on 'ut' preferences.

        Examples
        --------
            >>> profile_s = Profile(preferences_ut=[[1., .5, 0.], [.5, 1., 0.]])
            >>> profile_um = ProfileUM(profile_s=profile_s, n_m=2, ballot_ut=[0., 1., 0.])
            >>> profile_um.plurality_scores_ut
            array([1., 3., 0.])
        """
        plurality_scores_ut = np.zeros(self.n_c)
        preferred_candidates = np.where(self.ballot_ut == np.max(self.ballot_ut))[0]
        if preferred_candidates.size == 1:
            plurality_scores_ut[preferred_candidates[0]] = self.n_m
        plurality_scores_ut += self.profile_s.plurality_scores_ut
        return plurality_scores_ut

    def plurality_elimination_engine(self):
        return PluralityEliminationEngineProfileUM(self)
