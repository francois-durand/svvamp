# -*- coding: utf-8 -*-
"""
Created on nov. 28, 2018, 15:19
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
from svvamp.utils.misc import initialize_random_seeds
from svvamp.utils.von_mises_fisher import profile_vmf_aux
from svvamp.preferences.generator_profile import GeneratorProfile
from svvamp.preferences.profile import Profile


class GeneratorProfileVMFHypercircle(GeneratorProfile):
    """Profile generator using the Von Mises-Fisher distribution on the ``n_c - 2``-sphere

    Parameters
    ----------
    n_v : int
        Number of voters.
    n_c : int
        Number of candidates.
    vmf_concentration : number or list or ndarray
        1d array. Denote ``k`` its size (number of 'groups'). ``vmf_concentration[i]`` is the VMF concentration of
        group ``i``.
    vmf_probability : number or list or ndarray
        1d array of size ``k``. ``vmf_probability[i]`` is the probability, for a voter, to be in group ``i``
        (up to normalization). If ``None``, then the groups have equal probabilities.
    vmf_pole : list or ndarray
        2d array of size ``(k, n_c)``. ``vmf_pole[i, :]`` is the pole of the VMF distribution for group ``i``.

    Notes
    -----
    We work on the ``n_c - 2``-sphere: vectors of :math:`\mathbb{R}^n_c` with Euclidean norm equal to 1 and that are
    orthogonal to [1, ...,  1]. It is a representation of the classical Von Neumann-Morgenstern utility space. Cf.
    Durand et al., 'Geometry on the Utility Sphere'.

    Before all computations, the poles are projected onto the hyperplane and normalized. So, the only source of
    concentration for group ``i`` is ``vmf_concentration[i]``, not the norm of ``vmf_pole[i]``. If ``pole`` is None,
    then each pole is drawn independently and uniformly on the ``n_c - 2``-sphere.

    For each voter ``c``, we draw a group ``i`` at random, according to ``vmf_probability`` (normalized beforehand if
    necessary). Then, ``v``'s utility vector is drawn according to a Von Mises-Fisher distribution of pole
    ``vmf_pole[i, :]`` and concentration ``vmf_concentration[i]``, using Ulrich's method modified by Wood.

    Once group ``i`` is chosen, then up to a normalization constant, the density of probability for a unit vector
    ``x`` is ``exp(vmf_concentration[i] vmf.pole[i, :].x)``, where ``vmf.pole[i, :].x`` is a dot product.

    Examples
    --------
    Typical usage:

        >>> initialize_random_seeds()
        >>> generator = GeneratorProfileVMFHypercircle(n_v=5, n_c=3, vmf_concentration=10)
        >>> profile = generator()
        >>> profile.preferences_ut
        array([[ 0.67886167, -0.73231774,  0.05345607],
               [ 0.62834837, -0.76570906,  0.1373607 ],
               [ 0.71859954, -0.6950244 , -0.02357514],
               [ 0.49333272, -0.81010857,  0.31677584],
               [ 0.63042345, -0.76457205,  0.1341486 ]])

    You can specify a pole:

        >>> initialize_random_seeds()
        >>> generator = GeneratorProfileVMFHypercircle(n_v=5, n_c=3, vmf_concentration=10, vmf_pole=[.7, 0, -.7])
        >>> profile = generator()
        >>> profile.preferences_ut
        array([[ 0.8147627 , -0.36132374, -0.45343896],
               [ 0.44917711,  0.36590272, -0.81507983],
               [ 0.71021983, -0.00626776, -0.70395207],
               [ 0.68206757,  0.04766644, -0.72973402],
               [ 0.81591082, -0.38117577, -0.43473505]])

    With several poles:

        >>> initialize_random_seeds()
        >>> generator = GeneratorProfileVMFHypercircle(n_v=5, n_c=3, vmf_concentration=[np.inf, np.inf],
        ...                                            vmf_probability=[.9, .1])
        >>> profile = generator()
        >>> profile.preferences_ut
        array([[ 1.19716876,  0.82383355, -2.02100232],
               [ 0.71640317, -0.64749197, -0.0689112 ],
               [ 0.71640317, -0.64749197, -0.0689112 ],
               [ 0.71640317, -0.64749197, -0.0689112 ],
               [ 0.71640317, -0.64749197, -0.0689112 ]])

    If the probabilities are not explicitly given, poles have equal probabilities:

        >>> initialize_random_seeds()
        >>> generator = GeneratorProfileVMFHypercircle(n_v=5, n_c=3, vmf_concentration=[np.inf, np.inf])
        >>> profile = generator()
        >>> profile.preferences_ut
        array([[ 1.19716876,  0.82383355, -2.02100232],
               [ 1.19716876,  0.82383355, -2.02100232],
               [ 1.19716876,  0.82383355, -2.02100232],
               [ 0.71640317, -0.64749197, -0.0689112 ],
               [ 0.71640317, -0.64749197, -0.0689112 ]])

    References
    ----------
    Ulrich (1984) - Computer Generation of Distributions on the m-Sphere

    Wood (1994) - Simulation of the von Mises Fisher distribution
    """

    def __init__(self, n_v, n_c, vmf_concentration, vmf_probability=None, vmf_pole=None):
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
        self.log_creation = ['VMFHypercircle', n_c, n_v, 'VMF Concentration', vmf_concentration,
                             'VMF Probability', vmf_probability, 'VMF Pole', vmf_pole]
        # We have three interesting spaces:
        # Space 1: unit sphere of R^(n_c-1)
        # Space 2: unit sphere of R^n_c orthogonal to [0, ..., 0, 1]
        # Space 3: unit sphere of R^n_c orthogonal to [1, ..., 1, 1]
        # We use two isometric transformations:
        # 1 => 2: add a 0 as last coordinate.
        # 2 => 1: remove the last coordinate (which is 0).
        # 2 <=> 3: symmetry by a hyperplane that exchanges [0, ..., 0, 1] and [1, ..., 1] / sqrt(n_c).
        self.unitary_parallel = np.full(self.n_c, 1 / np.sqrt(self.n_c)) - (np.array(range(self.n_c)) == self.n_c - 1)
        self.unitary_parallel = self.unitary_parallel / np.sqrt(np.sum(self.unitary_parallel**2))
        # Project poles on the hyperplane H orthogonal to [1,..., 1] (space 3).
        self.vmf_pole = self.vmf_pole - np.sum(self.vmf_pole, 1)[:, np.newaxis] / self.n_c
        # Send poles to space 2 ~= space 1.
        pole_parallel = np.outer(np.sum(self.vmf_pole * self.unitary_parallel[np.newaxis, :], 1),
                                 self.unitary_parallel)
        self.vmf_pole -= 2 * pole_parallel
        super().__init__()

    def __call__(self):
        # Compute the number of voters in each group
        group_v = np.random.choice(self.k, size=self.n_v, replace=True, p=self.vmf_probability)
        cardinal_group_i = np.bincount(group_v, minlength=self.k)
        cum_cardinal_i = np.concatenate(([0], np.cumsum(cardinal_group_i)))
        # Use Von-Mises Fisher in space 1 ~= space 2
        preferences_ut = np.zeros((self.n_v, self.n_c))
        for i in range(self.k):
            preferences_ut[cum_cardinal_i[i]:cum_cardinal_i[i+1], :-1] = profile_vmf_aux(
                n_c=self.n_c - 1, n_v=cardinal_group_i[i], concentration=self.vmf_concentration[i],
                pole=self.vmf_pole[i, :-1])
        # Now send back to space 3.
        preferences_parallel = np.outer(np.sum(preferences_ut * self.unitary_parallel[np.newaxis, :], 1),
                                        self.unitary_parallel)
        preferences_ut -= 2 * preferences_parallel
        # Conclude
        return Profile(preferences_ut=preferences_ut, log_creation=self.log_creation)
