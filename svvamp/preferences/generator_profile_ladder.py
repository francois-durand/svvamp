# -*- coding: utf-8 -*-
"""
Created on nov. 28, 2018, 14:47
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


class GeneratorProfileLadder(GeneratorProfile):
    """Profile generator using the 'Ladder' model.

    Parameters
    ----------
    n_v : int
        Number of voters.
    n_c : int
        Number of candidates.
    n_rungs : int
        Number of rungs of the ladder.

    Notes
    -----
    Each utility ``preferences_ut[v, c]`` is drawn independently and uniformly between ``n_rungs`` values that
    divide the interval [-1, 1] equally. For example, if ``n_rungs = 21``, then values in {-1, -0.9, ..., 1} are used.

    The ordinal part of this distribution is **not** the Impartial Culture. Indeed, weak orders of preference come
    with non-zero probability. This model is interesting the study the effect of voters' ties.

    Examples
    --------
        >>> generator = GeneratorProfileLadder(n_v=10, n_c=3, n_rungs=5)
        >>> profile = generator()
        >>> profile.preferences_rk.shape
        (10, 3)
    """

    def __init__(self, n_v, n_c, n_rungs):
        self.n_v = n_v
        self.n_c = n_c
        self.n_rungs = n_rungs
        self.log_creation = ['Ladder', n_c, n_v, 'Number of rungs', n_rungs]
        super().__init__()

    def __call__(self):
        return Profile(
            preferences_ut=np.random.randint(self.n_rungs, size=(self.n_v, self.n_c)) * 2 / (self.n_rungs - 1) - 1,
            log_creation=self.log_creation)
