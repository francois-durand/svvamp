# -*- coding: utf-8 -*-
"""
Created on oct. 31, 2014, 19:54 
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


class PopulationCubicUniform(Population):

    _layout_name = 'Cubic uniform'

    def __init__(self, V, C):
        """Population with 'Cubic uniform' model.

        :param V: Integer. Number of voters.
        :param C: Integer. Number of candidates.

        :return: A :class:`~svvamp.Population` object.

        Each coefficient ``preferences_ut[v, c]`` is drawn independently
        and uniformly in the interval [-1, 1].

        The ordinal part of this distribution is the Impartial Culture.
        """
        preferences_utilities = 2 * np.random.rand(V, C) - 1
        log_creation = ['Cubic uniform', C, V]
        super().__init__(preferences_ut=preferences_utilities,
                         log_creation=log_creation)

    @staticmethod
    def iterator(C, V, culture_parameters, nb_populations):
        for i in range(nb_populations):
            yield PopulationCubicUniform(V=V, C=C, **culture_parameters)

    @staticmethod
    def meta_iterator(C_list, V_list, culture_parameters_list, nb_populations):
        for C, V, culture_parameters in itertools.product(
                C_list, V_list, culture_parameters_list):
            log_csv = ['Cubic uniform']
            log_print = ('Cubic uniform, V = ' + str(V) + ', C = ' + str(C))
            yield log_csv, log_print, PopulationCubicUniform.iterator(
                C, V, culture_parameters, nb_populations)


if __name__ == '__main__':
    # A quick demo
    pop = PopulationCubicUniform(V=5000, C=3)
    pop.demo()
    pop.plot3(normalize=False)

    pop = PopulationCubicUniform(V=5000, C=4)
    pop.demo()
    pop.plot4(normalize=False)