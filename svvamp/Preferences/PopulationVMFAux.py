# -*- coding: utf-8 -*-
"""
Created on oct. 31, 2014, 10:15 
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

import numpy as np


def population_vmf_aux(C, V, concentration, pole=None):
    """Von Mises-Fisher distribution

    Arguments:
    C -- Integer. Number of candidates.
    V -- Integer. Number of voters.
    concentration -- Number >= 0. Concentration of the VMF distribution.
    pole -- 1d array of size C. Center of the VMF distribution.

    Returns:
    preferences_utilities -- 2d array of size (V, C). Preferences of the
        population.

    Each row (voter) of preferences_utilities is drawn according to a Von
    Mises-Fisher law, using Ulrich's method modified by Wood. We work on the
    'C-1'-sphere (in R^C).

    Up to a normalization constant, the density of probability for a unit
    vector x is exp(concentration pole.x), where pole.x is the dot product
    of pole and x.

    pole is normalized before being used. So, the only source of
    concentration is the argument 'concentration', not the norm of 'pole'. If
    pole is not given, it is drawn at random on the sphere (uniformly),
    then the algorithm is launched (all voters are drawn with the same pole).

    References:
    Ulrich (1984) - Computer Generation of Distributions on the m-Sphere
    Wood (1994) - Simulation of the von Mises Fisher distribution
    """
    if concentration == 0:
        # Uniform model: the normal algorithm would work, but we can go
        # faster.
        preferences_utilities = np.random.randn(V, C)
        preferences_utilities = preferences_utilities / np.sum(
            preferences_utilities**2, 1)[:, np.newaxis]
        return preferences_utilities
    if np.isposinf(concentration):
        return np.tile(pole, (V, 1))
    if pole is None:
        _pole = np.random.randn(C)
        _pole = _pole / np.sqrt(np.sum(_pole**2))
    else:
        _pole = np.array(pole) / np.sqrt(np.sum(np.array(pole)**2))

    w_store = np.zeros(V)

    # Step 0 of the algorithm (compute constants)
    b = (- 2 * concentration + np.sqrt(4 * concentration**2 + (C - 1)**2)) / (
        C - 1)
    x_0 = (1 - b) / (1 + b)
    c = concentration * x_0 + (C - 1) * np.log(1 - x_0**2)

    for v in range(V):
        while True:
            # Step 1
            z = np.random.beta((C - 1) / 2, (C - 1) / 2)
            u = np.random.random()
            w = (1 - (1 + b) * z) / (1 - (1 - b) * z)

            # Step 2: test if found
            if not (concentration * w + (C - 1) * np.log(1 - x_0 * w) - c <
                        np.log(u)):
                w_store[v] = w
                break

    # Step 3
    # Draw vectors_v uniformly on the sphere in C-1-space and combine with
    # w_store
    vectors_v = np.random.randn(V, C - 1)
    vectors_v = vectors_v / np.sqrt(np.sum(vectors_v**2, 1))[:, np.newaxis]
    preferences_utilities = np.concatenate((
        np.sqrt(1 - w_store**2)[:, np.newaxis] * vectors_v,
        w_store[:, np.newaxis]
    ), axis=1)
    # print(preferences_utilities)

    # At this stage (end of Wood's algorithm), we have used a law of pole
    # (0, ..., 0, 1). We are going to use a reflection that exchanges this
    # vector with 'pole'.
    old_pole = np.zeros(C)
    old_pole[-1] = 1
    unitary_parallel = _pole - old_pole
    # print('old_pole =', old_pole)
    # print('unitary_parallel =', unitary_parallel)
    if np.any(unitary_parallel != 0):
        # print('Change pole')
        unitary_parallel = unitary_parallel / np.sqrt(np.sum(
            unitary_parallel**2))
        # print(preferences_utilities * unitary_parallel[np.newaxis, :])
        # print(np.sum(preferences_utilities * unitary_parallel[np.newaxis,
        #                                      :], 1))
        # print('unitary_parallel =', unitary_parallel)
        x_parallel = np.outer(
            np.sum(preferences_utilities * unitary_parallel[np.newaxis, :], 1),
            unitary_parallel)
        # print('x_parallel =', x_parallel)
        preferences_utilities -= 2 * x_parallel
    return preferences_utilities