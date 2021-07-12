# -*- coding: utf-8 -*-
"""
Created on nov. 28, 2018, 13:46
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


class GeneratorProfileEuclideanBox(GeneratorProfile):
    """Profile generator using the 'Euclidean box' model.

    Parameters
    ----------
    n_v : int
        Number of voters.
    n_c : int
        Number of candidates.
    box_dimensions : list or ndarray
        1d array of numbers. The length of the Euclidean box along each axis.
    shift : list or ndarray
        1d array of numbers, same dimension as ``box_dimensions``. Shift for the mean position of the candidates.

    Notes
    -----
    Denote ``n_dim`` the number of elements in ``box_dimensions``. For each voter and each candidate, a position is
    independently and uniformly drawn in a rectangular box of dimensions ``box_dimensions[0]``,... ,
    ``box_dimensions[n_dim - 1]``. If ``shift`` is used, the distribution of positions for candidates is displaced by
    this vector.

    Let ``d[v, c]`` denote the Euclidean distance between voter ``v`` and candidate ``c``. Then
    ``preferences_ut[v, c] = A - d[v, c]``, where ``A`` is such that the average utility is 0 over the whole profile.

    Remark: if ``n_dim = 1``, then the profile is single-peaked.

    Examples
    --------
        >>> generator = GeneratorProfileEuclideanBox(n_v=10, n_c=3, box_dimensions=[1])
        >>> profile = generator()
        >>> profile.preferences_rk.shape
        (10, 3)
    """

    def __init__(self, n_v, n_c, box_dimensions, shift=None):
        self.n_v = n_v
        self.n_c = n_c
        self.box_dimensions = np.array(box_dimensions)
        self.n_dim = len(self.box_dimensions)
        self.shift = np.zeros(self.n_dim) if shift is None else np.array(shift)
        self.log_creation = ['Euclidean box', self.n_c, self.n_v, 'Box dimensions', self.box_dimensions,
                             'Shift', self.shift, 'Number of dimensions', self.n_dim]
        super().__init__()

    def __call__(self):
        voters_positions = np.random.rand(self.n_v, self.n_dim) * self.box_dimensions
        candidates_positions = self.shift + np.random.rand(self.n_c, self.n_dim) * self.box_dimensions
        preferences_utilities = - distance.cdist(voters_positions, candidates_positions, 'euclidean')
        preferences_utilities -= np.average(preferences_utilities)
        return Profile(preferences_ut=preferences_utilities, log_creation=self.log_creation)
