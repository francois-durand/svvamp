# -*- coding: utf-8 -*-
"""
Created on jul. 07, 2022, 08:48
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


class GeneratorProfileIc(GeneratorProfile):
    """Profile generator using the 'Impartial Culture' model.

    Parameters
    ----------
    n_v : int
        Number of voters.
    n_c : int
        Number of candidates.
    sort_voters : bool
        This argument is passed to :class:`Profile`.

    Notes
    -----
    The ranking of each voter is drawn independently and uniformly.

    Examples
    --------
        >>> generator = GeneratorProfileIc(n_v=10, n_c=3)
        >>> profile = generator()
        >>> profile.preferences_rk.shape
        (10, 3)
    """

    def __init__(self, n_v, n_c, sort_voters=False):
        self.n_v = n_v
        self.n_c = n_c
        self.log_creation = ['Impartial Culture', n_c, n_v]
        super().__init__(sort_voters=sort_voters)

    def __call__(self):
        preferences_rk = np.array([
            np.random.permutation(self.n_c)
            for _ in range(self.n_v)
        ])
        return Profile(
            preferences_rk=preferences_rk,
            log_creation=self.log_creation, sort_voters=self.sort_voters
        )
