# -*- coding: utf-8 -*-
"""
Created on oct. 16, 2014, 15:06 
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
from scipy.spatial import distance

from svvamp.Preferences.Population import Population


class PopulationEuclideanBox(Population):

    _layout_name = 'Euclidean box'

    def __init__(self, V, C, box_dimensions):
        """Population with 'Euclidean box' model.

        Arguments:
        V -- Integer. Number of voters.
        C -- Integer. Number of candidates.
        box_dimensions -- 1d array of numbers. The length of the
        Euclidean box along each axis.

        For each voter and each candidate, a position is independently and
        uniformly drawn in a rectangular box of dimensions box_dimensions[0],
        ..., box_dimensions[n_dim - 1], where n_dim is the number of
        elements in box_dimensions. Let d[v, c] denote the Euclidean distance
        between voter v and candidate c. Then
        preferences_utilities[v, c] = A - d[v, c],
        where A is such that the average utility is 0 over the whole
        population.
        """
        d = len(box_dimensions)
        voters_positions = np.random.rand(V, d) * box_dimensions
        candidates_positions = np.random.rand(C, d) * box_dimensions
        preferences_utilities = - distance.cdist(
            voters_positions, candidates_positions, 'euclidean')
        preferences_utilities -= np.average(preferences_utilities)
        log_creation = ['Euclidean box', C, V,
                        'Box dimensions', box_dimensions,
                        'Number of dimensions', d]
        super().__init__(preferences_utilities=preferences_utilities,
                         log_creation=log_creation)

    @staticmethod
    def iterator(C, V, culture_parameters, nb_populations):
        for i in range(nb_populations):
            yield PopulationEuclideanBox(V=V, C=C, **culture_parameters)

    @staticmethod
    def meta_iterator(C_list, V_list, culture_parameters_list, nb_populations):
        for C, V, culture_parameters in itertools.product(
                C_list, V_list, culture_parameters_list):
            log_csv = ['Euclidean box',
                       'Box dimensions', culture_parameters['box_dimensions'],
                       'Number of dimensions',
                       len(culture_parameters['box_dimensions'])]
            log_print = ('Euclidean box, V = ' + str(V) + ', C = ' + str(C) +
                         ', box dimensions = ' +
                         format(culture_parameters['box_dimensions']))
            yield log_csv, log_print, PopulationEuclideanBox.iterator(
                C, V, culture_parameters, nb_populations)


if __name__ == '__main__':
    # A quick demo
    pop = PopulationEuclideanBox(V=1000, C=4, box_dimensions=[1])
    pop.demo()
    pop.plot3()
    pop.plot4()