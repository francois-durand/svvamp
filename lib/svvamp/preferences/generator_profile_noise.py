# -*- coding: utf-8 -*-
"""
Created on nov. 29, 2018, 09:34
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


class GeneratorProfileNoise(GeneratorProfile):
    """Profile generator adding noise to a given profile

    Parameters
    ----------
    base_profile : Profile
        The initial profile.
    relative_noise : number
        The relative noise.
    absolute_noise : number
        The absolute noise

    Notes
    -----
    We compute ``total_noise = absolute_noise + relative_noise * amplitude``, where ``amplitude`` is the difference
    between the lowest and the highest utility. Then to each ``preferences_ut[v, c]``, a random noise is added which
    is drawn independently and uniformly in the interval ``[- total_noise, total_noise]``.

    Examples
    --------
        >>> generator = GeneratorProfileNoise(base_profile=Profile(preferences_ut=[[5, 1, 2], [4, 10, 1]]),
        ...                                   absolute_noise=.1)
        >>> profile = generator()
        >>> profile.preferences_rk.shape
        (2, 3)
    """

    def __init__(self, base_profile, relative_noise=0., absolute_noise=0.):
        self.base_profile = base_profile
        self.base_ut = self.base_profile.preferences_ut.astype(np.float)
        self.n_v, self.n_c = self.base_profile.n_v, self.base_profile.n_c
        self.relative_noise = relative_noise
        self.absolute_noise = absolute_noise
        # Total noise
        self.total_noise = self.absolute_noise
        if relative_noise != 0:
            amplitude = np.max(self.base_ut) - np.min(self.base_ut)
            self.total_noise += relative_noise * amplitude
        self.log_creation = ['Noise Adder', 'Base profile', str(base_profile),
                             'Relative noise', relative_noise, 'Absolute noise', absolute_noise]
        super().__init__()

    def __call__(self):
        preferences_ut = self.base_ut.copy()
        if self.total_noise != 0:
            preferences_ut += self.total_noise * 2 * (0.5 - np.random.rand(self.n_v, self.n_c))
        return Profile(preferences_ut=preferences_ut, log_creation=self.log_creation)
