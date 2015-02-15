# -*- coding: utf-8 -*-
"""
Created on oct. 30, 2014, 18:25 
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
from scipy.spatial.distance import cdist

from svvamp.Preferences.Population import Population


class PopulationGaussianWell(Population):

    _layout_name = 'Gaussian well'

    def __init__(self, V, C, sigma, shift=None):
        """Population with 'Gaussian well' model.

        Arguments:
        V -- Integer. Number of voters.
        C -- Integer. Number of candidates.
        sigma -- 1d array of numbers. The variance of the gaussian law along
        each dimension.
        shift -- 1d array of numbers, same dimension as sigma. Shift for the
            mean position of the candidates.

        Let us note d the number of elements in sigma.
        For voter v (resp. each candidate c) and each axis i in range(d),
        a position x_i[v] (resp. y_i[c]) is independently drawn according
        to a normal law of mean 0 and variance sigma[i]. If shift is used,
        the distribution of positions for candidates is displaced by this
        vector.

        Let d[v, c] denote the Euclidean distance between voter v's position
        x[v] and candidate c's position y[c]. Then
        preferences_utilities[v, c] = A - d[v, c],
        where A is such that the average utility is 0 over the whole
        population.
        """
        d = len(sigma)
        voters_positions = np.random.randn(V, d) * sigma
        if shift is None:
            shift = np.zeros(len(sigma))
        else:
            shift = np.array(shift)
        candidates_positions = shift + np.random.randn(C, d) * sigma
        preferences_utilities = - cdist(
            voters_positions, candidates_positions, 'euclidean')
        preferences_utilities -= np.average(preferences_utilities)
        log_creation = ['Gaussian well', C, V,
                        'Sigma', sigma,
                        'Number of dimensions', d]
        super().__init__(preferences_utilities=preferences_utilities,
                         log_creation=log_creation)

    @staticmethod
    def iterator(C, V, culture_parameters, nb_populations):
        for i in range(nb_populations):
            yield PopulationGaussianWell(V=V, C=C, **culture_parameters)

    @staticmethod
    def meta_iterator(C_list, V_list, culture_parameters_list, nb_populations):
        for C, V, culture_parameters in itertools.product(
                C_list, V_list, culture_parameters_list):
            log_csv = ['Gaussian well',
                       'Sigma', culture_parameters['sigma'],
                       'Number of dimensions',
                       len(culture_parameters['sigma']),
                       'Shift', culture_parameters['shift']]
            log_print = ('Gaussian well, V = ' + str(V) + ', C = ' + str(C) +
                         ', sigma = ' +
                         format(culture_parameters['sigma']) +
                         ', shift = ' +
                         format(culture_parameters['shift']))
            yield log_csv, log_print, PopulationGaussianWell.iterator(
                C, V, culture_parameters, nb_populations)


if __name__ == '__main__':
    # A quick demo
    pop = PopulationGaussianWell(V=1000, C=4, sigma=[1], shift=[10])
    pop.demo()
    pop.plot3()
    pop.plot4()