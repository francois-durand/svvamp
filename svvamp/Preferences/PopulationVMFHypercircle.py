# -*- coding: utf-8 -*-
"""
Created on oct. 31, 2014, 18:59 
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
from svvamp.Preferences.PopulationVMFAux import population_vmf_aux


class PopulationVMFHypercircle(Population):

    _layout_name = 'Von Mises-Fisher on C-2-sphere'

    def __init__(self, V, C, vmf_concentration, vmf_probability=None,
                 vmf_pole=None):
        """Population drawn with Von Mises-Fisher distributions on the
        C-2-sphere

        :param V: Integer. Number of voters.
        :param C: Integer. Number of candidates.
        :param vmf_concentration: 1d array. Let us note ``k`` its size (number
            of 'groups'). ``vmf_concentration[i]`` is the VMF concentration of
            group ``i``.
        :param vmf_probability: 1d array of size k. ``vmf_probability[i]`` is
            the probability, for a voter, to be in group ``i`` (up to
            normalization). If ``None``, then groups have equal probabilities.
        :param vmf_pole: 2d array of size ``(k, C)``. ``vmf_pole[i, :]`` is the
            pole of the VMF distribution for group ``i``.

        :return: A :class:`~svvamp.Population` object.

        We work on the ``C-2``-sphere: vectors of :math:`\mathbb{R}^C` with
        Euclidean norm equal to 1 and that are orthogonal to
        [1, ...,  1]. It is a representation of the classical Von
        Neumann-Morgenstern utility space. Cf. working paper Durand et al.
        'Geometry on the Utility Sphere'.

        Before all computations, the poles are projected onto the hyperplane
        and normalized. So, the only source of concentration for group ``i`` is
        ``vmf_concentration[i]``, not the norm of ``vmf_pole[i]``. If ``pole``
        is None, then each pole is drawn independently and uniformly on the
        ``C-2``-sphere.

        For each voter ``c``, we draw a group ``i`` at random, according to
        ``vmf_probability`` (normalized beforehand if necessary). Then, ``v``'s
        utility vector is drawn according to a Von Mises-Fisher distribution
        of pole ``vmf_pole[i, :]`` and concentration ``vmf_concentration[i]``,
        using Ulrich's method modified by Wood.

        Once group ``i`` is chosen, then up to a normalization constant, the
        density of probability for a unit vector ``x`` is
        ``exp(vmf_concentration[i] vmf.pole[i, :].x)``,
        where ``vmf.pole[i, :].x`` is a dot product.

        References:

            Ulrich (1984) - Computer Generation of Distributions on the m-Sphere

            Wood (1994) - Simulation of the von Mises Fisher distribution
        """
        # Ensure that _vmf_concentration is an np.array
        # Compute k, its size.
        self._log_depth = 0
        self._log_identity = "Population_VMF_Hypersphere"
        if (isinstance(vmf_concentration, list) or
                isinstance(vmf_concentration, np.ndarray)):
            _vmf_concentration = np.array(vmf_concentration)
        else:
            _vmf_concentration = np.array([vmf_concentration])
        k = len(_vmf_concentration)
        self._mylogv('_vmf_concentration =', _vmf_concentration, 3)
        self._mylogv('k =', k, 3)
        # Ensure that _vmf_probability is an np.array
        # Ensure that it is normalized.
        if vmf_probability is None:
            _vmf_probability = np.full(k, 1/k)
        elif (isinstance(vmf_probability, list) or
                isinstance(vmf_probability, np.ndarray)):
            _vmf_probability = np.array(vmf_probability) / np.sum(
                vmf_probability)
        else:
            _vmf_probability = np.array([1])
        self._mylogv('_vmf_probability =', _vmf_probability, 3)
        # Ensure that _vmf_pole is a 2d array
        if vmf_pole is None:
            _vmf_pole = np.random.randn(k, C)
        else:
            _vmf_pole = np.array(vmf_pole)
            if _vmf_pole.ndim == 1:
                _vmf_pole = _vmf_pole[np.newaxis, :]
        # Not necessary to normalize, it is done in population_vmf_aux.
        self._mylogv('_vmf_pole =', _vmf_pole, 3)

        # Compute the number of voters in each group
        group_v = np.random.choice(
            k, size=V, replace=True, p=_vmf_probability)
        cardinal_group_i = np.bincount(group_v, minlength=k)
        cum_cardinal_i = np.concatenate(([0], np.cumsum(cardinal_group_i)))
        self._mylogv('cardinal_group_i =', cardinal_group_i, 3)
        self._mylogv('cum_cardinal_i =', cum_cardinal_i, 3)

        # We have three interesting spaces:
        # Space 1: unit sphere of R^(C-1)
        # Space 2: unit sphere of R^C orthogonal to [0, ..., 0, 1]
        # Space 3: unit sphere of R^C orthogonal to [1, ..., 1, 1]
        # We use two isometries:
        # 1 => 2: add a 0 as last coordinate.
        # 2 => 1: remove the last coordinate (which is 0).
        # 2 <=> 3: symmetry by a hyperplane that exchanges [0, ..., 0, 1] and
        # [1, ..., 1] / sqrt(C).
        unitary_parallel = np.full(C, 1/np.sqrt(C)) - (
            np.array(range(C)) == C - 1)
        unitary_parallel = unitary_parallel / np.sqrt(np.sum(
            unitary_parallel**2))

        # Project poles on the hyperplane H orthogonal to [1,..., 1] (space 3).
        _vmf_pole = _vmf_pole - np.sum(_vmf_pole, 1)[:, np.newaxis] / C
        # Send poles to space 2 ~= space 1.
        pole_parallel = np.outer(
            np.sum(_vmf_pole * unitary_parallel[np.newaxis, :], 1),
            unitary_parallel)
        _vmf_pole -= 2 * pole_parallel

        # Use Von-Mises Fisher in space 1 ~= space 2
        preferences_utilities = np.zeros((V, C))
        for i in range(k):
            preferences_utilities[
                cum_cardinal_i[i]:cum_cardinal_i[i+1], :-1] = (
                    population_vmf_aux(C=C-1, V=cardinal_group_i[i],
                                       concentration=_vmf_concentration[i],
                                       pole=_vmf_pole[i, :-1]))
        # Now send back to space 3.
        preferences_parallel = np.outer(
            np.sum(preferences_utilities * unitary_parallel[np.newaxis, :], 1),
            unitary_parallel)
        preferences_utilities -= 2 * preferences_parallel

        # Conclude
        log_creation = ['VMFHypercircle', C, V,
                        'VMF Concentration', vmf_concentration,
                        'VMF Probability', vmf_probability,
                        'VMF Pole', vmf_pole]
        super().__init__(preferences_utilities=preferences_utilities,
                         log_creation=log_creation)

    @staticmethod
    def iterator(C, V, culture_parameters, nb_populations):
        for i in range(nb_populations):
            yield PopulationVMFHypercircle(V=V, C=C, **culture_parameters)

    @staticmethod
    def meta_iterator(C_list, V_list, culture_parameters_list, nb_populations):
        for C, V, culture_parameters in itertools.product(
                C_list, V_list, culture_parameters_list):
            log_csv = [
                'VMFHypercircle',
                'VMF Concentration', culture_parameters['vmf_concentration'],
                'VMF Probability', culture_parameters['vmf_probability'],
                'VMF Pole', culture_parameters['vmf_pole']
            ]
            log_print = (
                'VMFHypercircle, V = ' + str(V) + ', C = ' + str(C) +
                ', vmf_concentration = ' +
                format(culture_parameters['vmf_probability']) +
                ', vmf_probability = ' +
                format(culture_parameters['vmf_probability']) +
                ', vmf_pole = ' + format(culture_parameters['vmf_pole'])
            )
            yield log_csv, log_print, PopulationVMFHypercircle.iterator(
                C, V, culture_parameters, nb_populations)


if __name__ == '__main__':
    # A quick demo
    # pop = PopulationVMFHypercircle(
    #     V=50, C=3,
    #     vmf_concentration=[50, 50],
    #     vmf_probability=None,
    #     vmf_pole=[[2, -2, 0], [0, 0, 1]])
    pop = PopulationVMFHypercircle(
        V=100, C=4,
        vmf_concentration=[10],
        vmf_pole=[4, 1, 2, 3])
    pop.demo()
    pop.plot4()