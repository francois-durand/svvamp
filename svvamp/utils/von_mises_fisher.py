# -*- coding: utf-8 -*-
"""
Created on oct. 31, 2014, 10:15
Copyright François Durand 2014-2018
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


def profile_vmf_aux(n_c, n_v, concentration, pole=None):
    """Von Mises-Fisher distribution.

    Parameters
    ----------
    n_c : int
        Number of candidates.
    n_v : int
        Number of voters.
    concentration : number
        Number >= 0. Concentration of the VMF distribution.
    pole : list or ndarray
        1d array of size ``n_c``. Center of the VMF distribution.

    Returns
    -------
    ndarray
        2d array of size (``n_v``, ``n_c``). Preferences (utilities) of the population.

    Notes
    -----
    Each row (voter) of ``preferences_ut`` is drawn according to a Von Mises-Fisher law, using Ulrich's method modified
    by Wood. We work on the ``n_c - 1``-sphere (in ``R^n_c``). Up to a normalization constant, the density of
    probability for a unit vector ``x`` is ``exp(concentration pole.x)``, where ``pole.x`` is the dot product of
    ``pole`` and ``x``.

    ``pole`` is normalized before being used. So, the only source of concentration is the argument ``concentration``,
    not the norm of ``pole``. If ``pole`` is not given, it is drawn at random on the sphere (uniformly), then the
    algorithm is launched (i.e. all voters are drawn with the same pole).

    Examples
    --------
    Typical usage:

        >>> preferences_ut = profile_vmf_aux(n_v=10, n_c=3, concentration=5, pole=[1, 0, 0])
        >>> preferences_ut.shape
        (10, 3)

    If the pole is not given, it is drawn at random:

        >>> preferences_ut = profile_vmf_aux(n_v=10, n_c=3, concentration=5)
        >>> preferences_ut.shape
        (10, 3)

    With a concentration of 0, it amounts to the uniform model:

        >>> preferences_ut = profile_vmf_aux(n_v=10, n_c=3, concentration=0, pole=[1, 0, 0])
        >>> preferences_ut.shape
        (10, 3)

    With an infinite concentration, all voters are on the pole:

        >>> profile_vmf_aux(n_v=5, n_c=3, concentration=np.inf, pole=[1, 0, 0])
        array([[1, 0, 0],
               [1, 0, 0],
               [1, 0, 0],
               [1, 0, 0],
               [1, 0, 0]])

    References
    ----------
    Ulrich (1984) - Computer Generation of Distributions on the m-Sphere.
    Wood (1994) - Simulation of the von Mises Fisher distribution.
    """
    if concentration == 0:
        # Uniform model: the normal algorithm would work, but we can go faster.
        preferences_ut = np.random.randn(n_v, n_c)
        preferences_ut = preferences_ut / np.sum(preferences_ut**2, 1)[:, np.newaxis]
        return preferences_ut
    if np.isposinf(concentration):
        return np.tile(pole, (n_v, 1))
    if pole is None:
        _pole = np.random.randn(n_c)
        _pole = _pole / np.sqrt(np.sum(_pole**2))
    else:
        _pole = np.array(pole) / np.sqrt(np.sum(np.array(pole)**2))

    w_store = np.zeros(n_v)

    # Step 0 of the algorithm (compute constants)
    b = (- 2 * concentration + np.sqrt(4 * concentration ** 2 + (n_c - 1) ** 2)) / (n_c - 1)
    x_0 = (1 - b) / (1 + b)
    c = concentration * x_0 + (n_c - 1) * np.log(1 - x_0 ** 2)

    for v in range(n_v):
        while True:
            # Step 1
            z = np.random.beta((n_c - 1) / 2, (n_c - 1) / 2)
            u = np.random.random()
            w = (1 - (1 + b) * z) / (1 - (1 - b) * z)

            # Step 2: test if found
            if not (concentration * w + (n_c - 1) * np.log(1 - x_0 * w) - c < np.log(u)):
                w_store[v] = w
                break

    # Step 3
    # Draw vectors_v uniformly on the sphere in `n_c - 1`-space and combine with w_store
    vectors_v = np.random.randn(n_v, n_c - 1)
    vectors_v = vectors_v / np.sqrt(np.sum(vectors_v**2, 1))[:, np.newaxis]
    preferences_ut = np.concatenate((np.sqrt(1 - w_store**2)[:, np.newaxis] * vectors_v, w_store[:, np.newaxis]),
                                    axis=1)

    # At this stage (end of Wood's algorithm), we have used a distribution of pole (0, ..., 0, 1). We are going to
    # use a reflection that exchanges this vector with 'pole'.
    old_pole = np.zeros(n_c)
    old_pole[-1] = 1
    unitary_parallel = _pole - old_pole
    if np.any(unitary_parallel != 0):
        unitary_parallel = unitary_parallel / np.sqrt(np.sum(unitary_parallel**2))
        x_parallel = np.outer(np.sum(preferences_ut * unitary_parallel[np.newaxis, :], 1), unitary_parallel)
        preferences_ut -= 2 * x_parallel
    return preferences_ut
