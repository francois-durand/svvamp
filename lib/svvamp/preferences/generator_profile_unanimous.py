# -*- coding: utf-8 -*-
"""
Created on jul. 07, 2022, 08:57
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


class GeneratorProfileUnanimous(GeneratorProfile):
    """Profile generator with identical voters.

    Parameters
    ----------
    n_v : int
        Number of voters.
    n_c : int, optional
        Number of candidates.
    ranking : List, optional
        This will be the ranking of all the voters. If not specified, it is drawn at random each time
        a profile is generated.
    sort_voters : bool
        This argument is passed to :class:`Profile` (but since voters are identical, there is no
        advantage in using True).

    Notes
    -----
    All the voters have the given ranking.

    Examples
    --------
        >>> generator = GeneratorProfileUnanimous(n_v=10, n_c=3)
        >>> profile = generator()
        >>> profile.preferences_rk.shape
        (10, 3)

        >>> generator = GeneratorProfileUnanimous(n_v=3, ranking=[0, 1, 2])
        >>> profile = generator()
        >>> profile.preferences_rk
        array([[0, 1, 2],
               [0, 1, 2],
               [0, 1, 2]])

        >>> generator = GeneratorProfileUnanimous(n_v=3)
        Traceback (most recent call last):
        ValueError: GeneratorProfileUnanimous: You should specify `n_c` or `ranking`.
    """

    def __init__(self, n_v, n_c=None, ranking=None, sort_voters=False):
        if n_c is None:
            if ranking is None:
                raise ValueError("GeneratorProfileUnanimous: You should specify `n_c` or `ranking`.")
            else:
                n_c = len(ranking)
        self.n_v = n_v
        self.n_c = n_c
        self.ranking = ranking
        self.log_creation = ['Unanimous', n_c, n_v, ranking]
        super().__init__(sort_voters=sort_voters)

    def __call__(self):
        if self.ranking is None:
            ranking = np.random.permutation(self.n_c)
        else:
            ranking = self.ranking
        preferences_rk = np.array([ranking] * self.n_v)
        return Profile(
            preferences_rk=preferences_rk,
            log_creation=self.log_creation, sort_voters=self.sort_voters
        )
