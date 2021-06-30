# -*- coding: utf-8 -*-
"""
Created on nov. 28, 2018, 15:44
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
from svvamp.Utils.Cache import cached_property
from svvamp.Utils.VonMisesFisher import profile_vmf_aux
from svvamp.Preferences.GeneratorProfile import GeneratorProfile
from svvamp.Preferences.Profile import Profile


class GeneratorProfileVMFHypersphere(GeneratorProfile):
    """Profile generator using the Von Mises-Fisher distribution on the ``n_c - 1``-sphere

    :param n_v: Integer. Number of voters.
    :param n_c: Integer. Number of candidates.
    :param stretching: Number between 0 and ``numpy.inf`` (both included).
    :param vmf_concentration: 1d array. Denote ``k`` its size (number of 'groups'). ``vmf_concentration[i]`` is
        the VMF concentration of group ``i``.
    :param vmf_probability: 1d array of size ``k``. ``vmf_probability[i]`` is the probability, for a voter,
        to be in group ``i`` (up to normalization). If ``None``, then the groups have equal probabilities.
    :param vmf_pole: 2d array of size ``(k, n_c)``. ``vmf_pole[i, :]`` is the pole of the VMF distribution for group
        ``i``.

    For each voter ``c``, we draw a group ``i`` at random, according to ``vmf_probability`` (normalized beforehand if
    necessary). Then, ``v``'s utility vector is drawn according to a Von Mises-Fisher distribution of pole
    ``vmf_pole[i, :]`` and concentration ``vmf_concentration[i]``, using Ulrich's method modified by Wood.

    Once group ``i`` is chosen, then up to a normalization constant, the density of probability for a unit vector
    ``x`` is ``exp(vmf_concentration[i] vmf.pole[i, :].x)``, ``where vmf.pole[i, :].x`` is a dot product.

    Then, ``v``'s utility vector is sent onto the spheroid that is the image of the sphere by a dilatation of factor
    ``stretching`` along the direction [1, ..., 1]. For example, if ``stretching = 1``, we stay on the unit sphere of
    :math:`\\mathbb{R}^n_c`. Cf. Durand et al. 'Geometry on the Utility Sphere'.

    N.B.: if ``stretching != 1``, it amounts to move the poles. For example, if the pole [1, 0, 0, 0] is given and
    ``stretching = 0``, then the actual pole will be [0.75, - 0.25, - 0.25, - 0.25] (up to a multiplicative constant).

    poles are normalized before being used. So, the only source of concentration for group ``i`` is
    ``vmf_concentration[i]``, not the norm of ``vmf_pole[i]``. If ``vmf_pole`` is ``None``, then each pole is drawn
    independently and uniformly on the sphere.

    References:

        Ulrich (1984) - Computer Generation of Distributions on the m-Sphere

        Wood (1994) - Simulation of the von Mises Fisher distribution

    >>> generator = GeneratorProfileVMFHypersphere(n_v=10, n_c=3, vmf_concentration=5)
    >>> generator().profile_.preferences_rk.shape
    (10, 3)
    """

    def __init__(self, n_v, n_c, vmf_concentration, vmf_probability=None, vmf_pole=None, stretching=1):
        self.n_v = n_v
        self.n_c = n_c
        # Ensure that _vmf_concentration is an np.array. Compute k, its size.
        if isinstance(vmf_concentration, list) or isinstance(vmf_concentration, np.ndarray):
            self.vmf_concentration = np.array(vmf_concentration)
        else:
            self.vmf_concentration = np.array([vmf_concentration])
        self.k = len(self.vmf_concentration)
        # Ensure that _vmf_probability is an np.array. Ensure that it is normalized.
        if vmf_probability is None:
            self.vmf_probability = np.full(self.k, 1 / self.k)
        elif isinstance(vmf_probability, list) or isinstance(vmf_probability, np.ndarray):
            self.vmf_probability = np.array(vmf_probability) / np.sum(vmf_probability)
        else:
            self.vmf_probability = np.array([1])
        # Ensure that _vmf_pole is a 2d array. It is not necessary to normalize: it is done in profile_vmf_aux.
        if vmf_pole is None:
            self.vmf_pole = np.random.randn(self.k, self.n_c)
        else:
            self.vmf_pole = np.array(vmf_pole)
            if self.vmf_pole.ndim == 1:
                self.vmf_pole = self.vmf_pole[np.newaxis, :]
        self.stretching = stretching
        self.log_creation = ['VMFHypersphere', n_c, n_v, 'VMF Concentration', vmf_concentration,
                             'VMF Probability', vmf_probability, 'VMF Pole', vmf_pole, 'Stretching', stretching]
        super().__init__()

    @cached_property
    def profile_(self):
        # Compute the number of voters in each group
        group_v = np.random.choice(self.k, size=self.n_v, replace=True, p=self.vmf_probability)
        cardinal_group_i = np.bincount(group_v, minlength=self.k)
        cum_cardinal_i = np.concatenate(([0], np.cumsum(cardinal_group_i)))
        # Use Von-Mises Fisher
        preferences_ut = np.zeros((self.n_v, self.n_c))
        for i in range(self.k):
            preferences_ut[cum_cardinal_i[i]:cum_cardinal_i[i+1], :] = profile_vmf_aux(
                n_c=self.n_c, n_v=cardinal_group_i[i], concentration=self.vmf_concentration[i],
                pole=self.vmf_pole[i, :])
        # Apply stretching. Cf. comments in PopulationSpheroid.
        if self.stretching < 1:
            preferences_ut = (preferences_ut
                              + np.sum(preferences_ut, 1)[:, np.newaxis] * (self.stretching - 1) / self.n_c)
        elif self.stretching > 1:
            preferences_ut = (preferences_ut / self.stretching
                              + np.sum(preferences_ut, 1)[:, np.newaxis] * (1 - 1 / self.stretching) / self.n_c)
        # Conclude
        return Profile(preferences_ut=preferences_ut, log_creation=self.log_creation)


if __name__ == '__main__':
    # A quick demo
    profile = GeneratorProfileVMFHypersphere(
        n_v=1000, n_c=4, vmf_concentration=[50, 50], vmf_probability=None, vmf_pole=[[2, -2, 0, 1], [0, 0, 1, 0]],
        stretching=1
    )().profile_
    profile.demo_()
    profile.plot3()
    profile.plot4()
