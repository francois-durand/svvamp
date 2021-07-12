# -*- coding: utf-8 -*-
"""
Created on nov. 28, 2018, 14:30
Copyright François Durand 2018
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


class GeneratorProfileCubicUniform(GeneratorProfile):
    """Profile generator using the 'Cubic uniform' model.

    Each coefficient ``preferences_ut[v, c]`` is drawn independently and uniformly in the interval [-1, 1]. The
    ordinal part of this distribution is the Impartial Culture.

    Parameters
    ----------
    n_v : int
        Number of voters.
    n_c : int
        Number of candidates.

    Examples
    --------
        >>> generator = GeneratorProfileCubicUniform(n_v=10, n_c=3)
        >>> profile = generator()
        >>> profile.preferences_rk.shape
        (10, 3)
    """

    def __init__(self, n_v, n_c):
        self.n_v = n_v
        self.n_c = n_c
        self.log_creation = ['Cubic uniform', n_c, n_v]
        super().__init__()

    def __call__(self):
        return Profile(preferences_ut=2 * np.random.rand(self.n_v, self.n_c) - 1, log_creation=self.log_creation)
