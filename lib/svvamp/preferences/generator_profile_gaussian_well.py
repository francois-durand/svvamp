# -*- coding: utf-8 -*-
"""
Created on nov. 28, 2018, 14:39
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
from scipy.spatial import distance
from svvamp.preferences.generator_profile import GeneratorProfile
from svvamp.preferences.profile import Profile


class GeneratorProfileGaussianWell(GeneratorProfile):
    """Profile generator using the 'Gaussian well' model.

    Parameters
    ----------
    n_v : int
        Number of voters.
    n_c : int
        Number of candidates.
    sigma : list or ndarray
        1d array of numbers. The variance of the gaussian distribution along each dimension.
    shift : list or ndarray
        1d array of numbers, same dimension as ``sigma``. Shift for the mean position of the candidates.

    Notes
    -----
    Let us note ``n_dim`` the number of elements in ``sigma``. For voter ``v`` (resp. each candidate ``c``) and each
    axis ``i`` in ``range(n_dim)``, a position ``x_i[v]`` (resp. ``y_i[c]``) is independently drawn according to a
    normal distribution of mean 0 and variance ``sigma[i]``. If ``shift`` is used, the distribution of positions for
    candidates is displaced by this vector.

    Let ``d[v, c]`` denote the Euclidean distance between voter ``v``'s position ``x[v]`` and candidate ``c``'s
    position ``y[c]``. Then ``preferences_ut[v, c] = A - d[v, c]``, where ``A`` is such that the average utility is 0
    over the whole population.

    Remark: if ``n_dim = 1``, the population is single-peaked.

    Examples
    --------
        >>> generator = GeneratorProfileGaussianWell(n_v=10, n_c=3, sigma=[1], shift=[10])
        >>> profile = generator()
        >>> profile.preferences_rk.shape
        (10, 3)
    """

    def __init__(self, n_v, n_c, sigma, shift=None):
        self.n_v = n_v
        self.n_c = n_c
        self.sigma = np.array(sigma)
        self.n_dim = len(self.sigma)
        self.shift = np.zeros(self.n_dim) if shift is None else np.array(shift)
        self.log_creation = ['Gaussian well', self.n_c, self.n_v, 'Sigma', self.sigma,
                             'Shift', self.shift, 'Number of dimensions', self.n_dim]
        super().__init__()

    def __call__(self):
        voters_positions = np.random.randn(self.n_v, self.n_dim) * self.sigma
        candidates_positions = self.shift + np.random.randn(self.n_c, self.n_dim) * self.sigma
        preferences_utilities = - distance.cdist(voters_positions, candidates_positions, 'euclidean')
        preferences_utilities -= np.average(preferences_utilities)
        return Profile(preferences_ut=preferences_utilities, log_creation=self.log_creation)
