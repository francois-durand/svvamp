# -*- coding: utf-8 -*-
"""
Created on nov. 28, 2018, 14:55
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


class GeneratorProfileSpheroid(GeneratorProfile):
    """Profile generator using the 'Spheroid' model.

    Parameters
    ----------
    n_v : int
        Number of voters.
    n_c : int
        Number of candidates.
    stretching : number
        Number between 0 and ``numpy.inf`` (both included).

    Notes
    -----
    The utility vector of each voter is drawn independently and uniformly on a sphere in :math:`\\mathbb{R}^n_c`. Then,
    it is sent on the spheroid that is the image of the sphere by a dilatation of factor ``stretching`` in direction
    [1, ..., 1]. Cf. Durand et al., 'Geometry on the Utility Sphere'.

    The ordinal part of this distribution is the Impartial Culture.

    The parameter ``stretching`` has an influence only on voting systems based on utilities, especially Approval
    voting.

        * ``stretching = 0``: pure Von Neumann-Morgenstern utility, normalized to :math:`\\sum_c u_v(c) = 0` (
          spherical model with ``n_c-2`` dimensions).
       * ``stretching = 1``: spherical model with ``n_c-1`` dimensions.
        * ``stretching = inf``: axial/cylindrical model with only two possible values, all-approval [1, ...,
          1] and all-reject [-1, ..., -1].

    N.B.: This model gives is equivalent to a Von Mises-Fisher model with concentration = 0.

    Examples
    --------
    Typical usage:

        >>> generator = GeneratorProfileSpheroid(n_v=10, n_c=3, stretching=1)
        >>> profile = generator()
        >>> profile.preferences_rk.shape
        (10, 3)

    With some stretching:

        >>> generator = GeneratorProfileSpheroid(n_v=10, n_c=3, stretching=2)
        >>> profile = generator()
        >>> profile.preferences_rk.shape
        (10, 3)

        >>> generator = GeneratorProfileSpheroid(n_v=10, n_c=3, stretching=.5)
        >>> profile = generator()
        >>> profile.preferences_rk.shape
        (10, 3)
    """

    def __init__(self, n_v, n_c, stretching=1):
        self.n_v = n_v
        self.n_c = n_c
        self.stretching = stretching
        self.log_creation = ['Spheroid', n_c, n_v, 'Stretching', stretching]
        super().__init__()

    def __call__(self):
        # Step 1: spherical distribution
        preferences_ut = np.random.randn(self.n_v, self.n_c)
        preferences_ut = preferences_ut / np.sqrt(np.sum(preferences_ut**2, 1))[:, np.newaxis]
        # Step 2: apply stretching
        # These are not really different cases: same formula up to a factor 'stretching', which has no impact since we
        # work in a projective space. This distinction is only used to deal with stretching close or equal to 0 or
        # infinity. The matrix of the transformation is (up to a multiplicative factor):
        # (Id - 1/n_c J) + stretching * 1/n_c J,
        # where 1/n_c J is the orthogonal projection on diagonal (1, ..., 1) and (Id - 1/n_c J) is the projection on its
        # orthogonal hyperplane.
        if self.stretching < 1:
            preferences_ut = (preferences_ut
                              + np.sum(preferences_ut, 1)[:, np.newaxis] * (self.stretching - 1) / self.n_c)
        elif self.stretching > 1:
            preferences_ut = (preferences_ut / self.stretching
                              + np.sum(preferences_ut, 1)[:, np.newaxis] * (1 - 1 / self.stretching) / self.n_c)
        return Profile(preferences_ut=preferences_ut, log_creation=self.log_creation)
