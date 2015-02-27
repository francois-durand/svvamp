# -*- coding: utf-8 -*-
"""
Created on oct. 31, 2014, 10:59 
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


class PopulationVMFHypersphere(Population):

    _layout_name = 'Von Mises-Fisher on C-1-sphere'

    def __init__(self, V, C, vmf_concentration, vmf_probability=None,
                 vmf_pole=None, stretching=1):
        """Population drawn with Von Mises-Fisher distributions on the
        C-1-sphere

        :param V: Integer. Number of voters.
        :param C: Integer. Number of candidates.
        :param stretching: Number between 0 and ``numpy.inf`` (both included).
        :param vmf_concentration: 1d array. Let us note ``k`` its size (number
            of 'groups'). ``vmf_concentration[i]`` is the VMF concentration of
            group ``i``.
        :param vmf_probability: 1d array. ``vmf_probability[i]`` is the
            probability, for a voter, to be in group ``i`` (up to
            normalization). If None, then groups have equal probabilities.
        :param vmf_pole: 2d array of size ``(k, C)``. ``vmf_pole[i, :]`` is the
            pole of the VMF distribution for group ``i``.

        :return: A :class:`~svvamp.Population` object.

        For each voter ``c``, we draw a group ``i`` at random, according to
        ``vmf_probability`` (normalized beforehand if necessary). Then, ``v``'s
        utility vector is drawn according to a Von Mises-Fisher distribution
        of pole ``vmf_pole[i, :]`` and concentration ``vmf_concentration[i]``,
        using Ulrich's method modified by Wood.

        Once group ``i`` is chosen, then up to a normalization constant, the
        density of probability for a unit vector ``x`` is
        ``exp(vmf_concentration[i] vmf.pole[i, :].x)``,
        ``where vmf.pole[i, :].x`` is a dot product.

        Then, ``v``'s utility vector is sent onto the spheroid that is the
        image of the sphere by a dilatation of factor ``stretching`` along the
        direction [1, ..., 1]. For example, if ``stretching = 1``, we stay on
        the unit sphere of :math:`\\mathbb{R}^C`. Cf. working paper Durand et
        al. 'Geometry on the Utility Sphere'.

        N.B.: if ``stretching != 1``, it amounts to move the poles. For
        example, if the pole [1, 0, 0, 0] is given and ``stretching = 0``,
        then the actual pole will be [0.75, - 0.25, - 0.25, - 0.25]
        (up to a multiplicative constant).

        poles are normalized before being used. So, the only source of
        concentration for group ``i`` is ``vmf_concentration[i]``, not the
        norm of ``vmf_pole[i]``. If ``vmf_pole`` is ``None``, then each pole
        is drawn independently and uniformly on the sphere.

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
        # Not necessary to normalize, it is done in population_vmf_aux.
        if vmf_pole is None:
            _vmf_pole = np.random.randn(k, C)
        else:
            _vmf_pole = np.array(vmf_pole)
            if _vmf_pole.ndim == 1:
                _vmf_pole = _vmf_pole[np.newaxis, :]
        self._mylogv('_vmf_pole =', _vmf_pole, 3)

        # Compute the number of voters in each group
        group_v = np.random.choice(
            k, size=V, replace=True, p=_vmf_probability)
        cardinal_group_i = np.bincount(group_v, minlength=k)
        cum_cardinal_i = np.concatenate(([0], np.cumsum(cardinal_group_i)))
        self._mylogv('cardinal_group_i =', cardinal_group_i, 3)
        self._mylogv('cum_cardinal_i =', cum_cardinal_i, 3)
        # Use Von-Mises Fisher
        preferences_utilities = np.zeros((V, C))
        for i in range(k):
            preferences_utilities[cum_cardinal_i[i]:cum_cardinal_i[i+1], :] = (
                population_vmf_aux(C=C, V=cardinal_group_i[i],
                                   concentration=_vmf_concentration[i],
                                   pole=_vmf_pole[i, :]))
        # Apply stretching
        # Cf. comments in PopulationSpheroid
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
        # Conclude
        log_creation = ['VMFHypersphere', C, V,
                        'VMF Concentration', vmf_concentration,
                        'VMF Probability', vmf_probability,
                        'VMF Pole', vmf_pole,
                        'Stretching', stretching]
        super().__init__(preferences_ut=preferences_utilities,
                         log_creation=log_creation)

    @staticmethod
    def iterator(C, V, culture_parameters, nb_populations):
        for i in range(nb_populations):
            yield PopulationVMFHypersphere(V=V, C=C, **culture_parameters)

    @staticmethod
    def meta_iterator(C_list, V_list, culture_parameters_list, nb_populations):
        for C, V, culture_parameters in itertools.product(
                C_list, V_list, culture_parameters_list):
            log_csv = [
                'VMFHypersphere',
                'VMF Concentration', culture_parameters['vmf_concentration'],
                'VMF Probability', culture_parameters['vmf_probability'],
                'VMF Pole', culture_parameters['vmf_pole'],
                'Stretching', culture_parameters['stretching']
            ]
            log_print = (
                'VMFHypersphere, V = ' + str(V) + ', C = ' + str(C) +
                ', vmf_concentration = ' +
                format(culture_parameters['vmf_probability']) +
                ', vmf_probability = ' +
                format(culture_parameters['vmf_probability']) +
                ', vmf_pole = ' + format(culture_parameters['vmf_pole']) +
                ', stretching = ' + format(culture_parameters['stretching'])
            )
            yield log_csv, log_print, PopulationVMFHypersphere.iterator(
                C, V, culture_parameters, nb_populations)


if __name__ == '__main__':
    # A quick demo
    pop = PopulationVMFHypersphere(
        V=1000, C=4,
        vmf_concentration=[50, 50],
        vmf_probability=None,
        vmf_pole=[[2, -2, 0, 1], [0, 0, 1, 0]],
        stretching=1)
    pop.demo()
    pop.plot3()
    pop.plot4()