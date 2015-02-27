# -*- coding: utf-8 -*-
"""
Created on oct. 30, 2014, 19:07 
Copyright François Durand 2014, 2015
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

import itertools

import numpy as np

from svvamp.Preferences.Population import Population


class PopulationSpheroid(Population):

    _layout_name = 'Spheroid'

    def __init__(self, V, C, stretching=1):
        """Population with 'Spheroid' model.

        :param V: Integer. Number of voters.
        :param C: Integer. Number of candidates.
        :param stretching: Number between 0 and ``numpy.inf`` (both included).

        :return: A :class:`~svvamp.Population` object.

        The utility vector of each voter is drawn independently and uniformly
        on a sphere in :math:`\\mathbb{R}^C`. Then, it is sent on the spheroid
        that is the image of the sphere by a dilatation of factor
        ``stretching`` in direction [1, ..., 1]. Cf. working paper Durand et
        al. 'Geometry on the Utility Sphere'.

        The ordinal part of this distribution is the Impartial Culture.

        The parameter ``stretching`` has only influence on voting systems
        based on utilities, especially Approval voting.

            * ``stretching = 0``: pure Von Neumann-Morgenstern utility,
              normalized to :math:`\\sum_c u_v(c) = 0` (spherical model
              with ``C-2`` dimensions).
            * ``stretching = 1``: spherical model with ``C-1`` dimensions.
            * ``stretching = inf``: axial/cylindrical model with only two
              possible values, all-approval [1, ..., 1] and all-reject
              [-1, ..., -1].

        N.B.: This model gives the same probability distribution as a Von
        Mises-Fisher with concentration = 0.
        """
        # Step 1: spherical distribution
        preferences_utilities = np.random.randn(V, C)
        preferences_utilities = preferences_utilities / np.sqrt(np.sum(
            preferences_utilities**2, 1))[:, np.newaxis]

        # Step 2: apply stretching
        # This are not really different cases: same formula up to a factor
        # 'stretching', which has no impact since we work in a projective
        # space. This distinction is only used to deal with stretching close
        # or equal to 0 or infinity.
        # Matrix of the transformation is (up to a multiplicative factor):
        # (Id - 1/C J) + stretching * 1/C J,
        # where 1/C J is the orthogonal projection on diagonal (1, ..., 1)
        # and (Id - 1/C J) is the projection on its orthogonal hyperplane.
        if stretching < 1:
            preferences_utilities = (
                preferences_utilities +
                np.sum(preferences_utilities, 1)[
                    :, np.newaxis] * (stretching - 1) / C)
        elif stretching > 1:
            preferences_utilities = (
                preferences_utilities / stretching +
                np.sum(preferences_utilities, 1)[
                    :, np.newaxis] * (1 - 1/stretching) / C)
        log_creation = ['Spheroid', C, V,
                        'Stretching', stretching]
        super().__init__(preferences_ut=preferences_utilities,
                         log_creation=log_creation)

    @staticmethod
    def iterator(C, V, culture_parameters, nb_populations):
        for i in range(nb_populations):
            yield PopulationSpheroid(V=V, C=C, **culture_parameters)

    @staticmethod
    def meta_iterator(C_list, V_list, culture_parameters_list, nb_populations):
        for C, V, culture_parameters in itertools.product(
                C_list, V_list, culture_parameters_list):
            log_csv = ['Spheroid',
                       'Stretching', culture_parameters['stretching']]
            log_print = ('Spheroid, V = ' + str(V) + ', C = ' + str(C) +
                         ', stretching = ' +
                         format(culture_parameters['stretching']))
            yield log_csv, log_print, PopulationSpheroid.iterator(
                C, V, culture_parameters, nb_populations)


if __name__ == '__main__':
    # A quick demo
    pop = PopulationSpheroid(V=100, C=4, stretching=1)
    pop._labels_candidates = ['1','2','3','4']
    pop.demo()
    # pop.plot3(normalize=False)
    # pop.plot3()
    pop.plot4()