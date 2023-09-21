# -*- coding: utf-8 -*-
"""
Created on jul. 07, 2022, 09:04
Copyright François Durand 2022
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
from svvamp.preferences.generator_profile import GeneratorProfile
from svvamp.preferences.profile import Profile


class GeneratorProfilePerturbedCulture(GeneratorProfile):
    """Profile generator using the 'Perturbed Culture' model.

    Parameters
    ----------
    n_v : int
        Number of voters.
    theta : float
        Weight of the unanimous part.
    n_c : int, optional
        Number of candidates.
    ranking : List, optional
        This will be the ranking of all the voters of the unanimous part. If not specified, it is drawn at
        random each time a profile is generated.
    sort_voters : bool
        This argument is passed to :class:`Profile` (but since voters are identical, there is no
        advantage in using True).

    Notes
    -----
    First, if `ranking` is not specified, it is drawn uniformly at random. Then, independently for each voter:

    * With probability `theta`, she has the ranking `ranking` ("unanimous part" of the profile).
    * With probability `1 - theta`, her ranking is drawn uniformly at random ("impartial culture" part of
      the profile).

    Examples
    --------
        >>> generator = GeneratorProfilePerturbedCulture(n_v=10, theta=.1, n_c=3)
        >>> profile = generator()
        >>> profile.preferences_rk.shape
        (10, 3)

        >>> generator = GeneratorProfilePerturbedCulture(n_v=10, theta=.1, ranking=[0, 1, 2])
        >>> profile = generator()
        >>> profile.preferences_rk.shape
        (10, 3)

        >>> generator = GeneratorProfilePerturbedCulture(n_v=10, theta=.1)
        Traceback (most recent call last):
        ValueError: GeneratorProfilePerturbedCulture: You should specify `n_c` or `ranking`.
    """

    def __init__(self, n_v, theta, n_c=None, ranking=None, sort_voters=False):
        if n_c is None:
            if ranking is None:
                raise ValueError("GeneratorProfilePerturbedCulture: You should specify `n_c` or `ranking`.")
            else:
                n_c = len(ranking)
        self.n_v = n_v
        self.n_c = n_c
        self.theta = theta
        self.ranking = ranking
        self.log_creation = ['Perturbed Culture', n_c, n_v, theta, ranking]
        super().__init__(sort_voters=sort_voters)

    def __call__(self):
        if self.ranking is None:
            ranking = np.random.permutation(self.n_c)
        else:
            ranking = self.ranking
        preferences_rk = np.array([
            ranking if np.random.rand() < self.theta else np.random.permutation(self.n_c)
            for _ in range(self.n_v)
        ])
        return Profile(
            preferences_rk=preferences_rk,
            log_creation=self.log_creation, sort_voters=self.sort_voters
        )
