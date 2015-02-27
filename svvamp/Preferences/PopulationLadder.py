# -*- coding: utf-8 -*-
"""
Created on oct. 30, 2014, 18:45 
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


class PopulationLadder(Population):

    _layout_name = 'Ladder'

    def __init__(self, V, C, n_rungs):
        """Population with 'Ladder' model.

        :param V: Integer. Number of voters.
        :param C: Integer. Number of candidates.
        :param n_rungs: Integer. Number of rungs of the ladder.

        :return: A :class:`~svvamp.Population` object.

        Each utility ``preferences_ut[v, c]`` is drawn independently and
        equiprobably between ``n_rungs`` values that divide the interval
        [-1, 1] equally. For example, if ``n_rungs = 21``, then values in
        {-1, -0.9, ..., 1} are used.

        The ordinal part of this distribution is **not** the Impartial Culture.
        Indeed, weak orders of preference come with non-zero probability.
        This model is interesting the study the effect of voters' ties.
        """
        preferences_utilities = np.random.randint(
            n_rungs, size=(V, C)
        ) * 2 / (n_rungs - 1) - 1
        log_creation = ['Ladder', C, V,
                        'Number of rungs', n_rungs]
        super().__init__(preferences_ut=preferences_utilities,
                         log_creation=log_creation)

    @staticmethod
    def iterator(C, V, culture_parameters, nb_populations):
        for i in range(nb_populations):
            yield PopulationLadder(V=V, C=C, **culture_parameters)

    @staticmethod
    def meta_iterator(C_list, V_list, culture_parameters_list, nb_populations):
        for C, V, culture_parameters in itertools.product(
                C_list, V_list, culture_parameters_list):
            log_csv = ['Ladder',
                       'Number of rungs', culture_parameters['n_rungs']]
            log_print = ('Ladder, V = ' + str(V) + ', C = ' + str(C) +
                         ', n_rungs = ' +
                         format(culture_parameters['n_rungs']))
            yield log_csv, log_print, PopulationLadder.iterator(
                C, V, culture_parameters, nb_populations)


if __name__ == '__main__':
    # A quick demo
    pop = PopulationLadder(V=1000, C=3, n_rungs=5)
    pop.demo()
    pop.plot3(normalize=False)

    pop = PopulationLadder(V=1000, C=4, n_rungs=5)
    pop.demo()
    pop.plot4(normalize=False)