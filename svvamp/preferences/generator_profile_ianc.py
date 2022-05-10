# -*- coding: utf-8 -*-
"""
Created on mar. 24, 2022, 6:48
Copyright François Durand 2022
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
from collections import Counter
from math import factorial

import numpy as np

from svvamp.preferences.generator_profile import GeneratorProfile
from svvamp.preferences.profile import Profile


class GeneratorProfileIanc(GeneratorProfile):
    """Profile generator: Impartial, Anonymous and Neutral Culture.

    Cf. Eğecioğlu, Ömer & Giritligil, Ayça E. (2013). The impartial, anonymous,
    and neutral culture model: a probability model for sampling public
    preference structures. The Journal of mathematical sociology, 37(4), 203-222.

    The code is adapted from:

    * A Mathematica program by Ömer Eğecioğlu: https://sites.cs.ucsb.edu/~omer/randomvote.nb.
    * The Python function `accel_asc` by Jerome Kelleher: https://jeromekelleher.net/generating-integer-partitions.html.

    Parameters
    ----------
    n_v : int
        Number of voters.
    n_c : int
        Number of candidates.
    sort_voters : bool
        This argument is passed to :class:`Profile`.

    Examples
    --------
        >>> np.random.seed(42)
        >>> generator = GeneratorProfileIanc(n_v=10, n_c=3)
        >>> profile = generator()
        >>> profile.preferences_rk.shape
        (10, 3)
    """

    def __init__(self, n_v, n_c, sort_voters=False):
        self.n_v = n_v
        self.n_c = n_c
        self.log_creation = ['IANC', n_c, n_v]
        super().__init__(sort_voters=sort_voters)
        self.pairs_lam_mu, self.probabilities = _pairs_lam_mu_and_probabilities(n=self.n_v, m=self.n_c)

    def __call__(self):
        preferences_rk = _random_profile(
            n=self.n_v, m=self.n_c,
            pairs_lam_mu=self.pairs_lam_mu, probabilities=self.probabilities,
            shuffle_voters=not self.sort_voters
        )
        return Profile(preferences_rk=preferences_rk, log_creation=self.log_creation,
                       sort_voters=self.sort_voters)


def _z_partition(lam: list):
    """Number `z` associated to a cycle type, i.e. to a partition of an integer.

    Defined by Eğecioğlu & Giritligil (2013), equation (4).

    Parameters
    ----------
    lam: list
        A cycle type, i.e. a partition of an integer `n` (cf. example below).

    Returns
    -------
    z: int
        The number `z` associated to `lam`.

    Examples
    --------
    Consider `n` = 7` and the permutation `sigma = (142)(35)(67)`. It has cycle type `2**2 * 3**1` (2 cycles
    of length 2, 1 cycle of length 3). So we consider the partition `lam = [3, 2, 2]` of `n`. The corresponding
    number `z_lambda` is:

        >>> _z_partition([3, 2, 2])
        24

    In the symmetric group `S_7`, the number of permutations having the same cycle type as `sigma` is:

        >>> factorial(7) // _z_partition([3, 2, 2])
        210
    """
    return np.prod([
        (cycle_length ** n_occurrences) * factorial(n_occurrences)
        for cycle_length, n_occurrences in Counter(lam).items()
    ])


def _enumerate_partitions(n: int):
    """Enumerate the partitions of an integer.

    This is the function `accel_asc` from https://jeromekelleher.net/generating-integer-partitions.html.

    Parameters
    ----------
    n: int
        Total number of elements.

    Yields
    ------
    lam: list
        A partition of `n` (cf. example below), i.e. a list of integers whose sum is `n`.
        Each partition is given once, with the elements of the list given in increasing order.

    Examples
    --------
    Enumerate all the partitions of 4:

        >>> for lam in _enumerate_partitions(4):
        ...     print(lam)
        [1, 1, 1, 1]
        [1, 1, 2]
        [1, 3]
        [2, 2]
        [4]
    """
    a = [0 for _ in range(n + 1)]
    k = 1
    y = n - 1
    while k != 0:
        x = a[k - 1] + 1
        k -= 1
        while 2 * x <= y:
            a[k] = x
            y -= x
            k += 1
        ell = k + 1
        while x <= y:
            a[k] = x
            a[ell] = y
            yield a[:k + 2]
            x += 1
            y -= 1
        a[k] = x + y
        y = x + y - 1
        yield a[:k + 1]


def lcm(*args):
    """Least common multiple.

    Built-in in the math module from Python 3.9. We add it here to be compatible with older versions.

    Examples
    --------
        >>> lcm(3, 12, 20)
        60
    """
    # noinspection PyUnresolvedReferences
    return np.lcm.reduce(args)


def gcd(*args):
    """Least common multiple.

    The version of the math module accepts more than 2 arguments since Python 3.9.
    We add it here to be compatible with older versions.

    Examples
    --------
        >>> gcd(9, 12, 21)
        3
    """
    # noinspection PyUnresolvedReferences
    return np.gcd.reduce(args)


def _pairs_lam_mu_and_probabilities(n: int, m: int):
    # noinspection PyShadowingNames
    """Pairs (lambda, mu) and their probabilities.

    Cf. Eğecioğlu & Giritligil (2013), Theorem 1.

    Parameters
    ----------
    n : int
        Number of voters.
    m : int
        Number of candidates

    Returns
    -------
    pairs_lam_mu: list of tuple of list
        Each element is a tuple `(lam, mu)`, where `lam` (resp. `mu`) is a list representing a
        partition of `n` (resp. `m`), i.e. a list of integers whose sum is `n` (resp. `m`).
        A tuple is given only if `gcd(*lam)` is divisible by `lcm(*mu)`.
    probabilities: np.ndarray
        For each tuple `(lam, mu)` in `pairs_lam_mu`, this gives the corresponding probability.

    Examples
    --------
        >>> pairs_lam_mu, probabilities = _pairs_lam_mu_and_probabilities(n=3, m=3)
        >>> pairs_lam_mu
        [([1, 1, 1], [1, 1, 1]), ([1, 2], [1, 1, 1]), ([3], [1, 1, 1]), ([3], [3])]
        >>> probabilities
        array([0.6       , 0.3       , 0.03333333, 0.06666667])

        >>> pairs_lam_mu, probabilities = _pairs_lam_mu_and_probabilities(n=3, m=2)
        >>> pairs_lam_mu
        [([1, 1, 1], [1, 1]), ([1, 2], [1, 1]), ([3], [1, 1])]
        >>> probabilities
        array([0.33333333, 0.5       , 0.16666667])
    """
    pairs_lam_mu = [
        (lam, mu)
        for lam in _enumerate_partitions(n)
        for mu in _enumerate_partitions(m)
        if gcd(*lam) % lcm(*mu) == 0
    ]
    probabilities = np.array([
        factorial(m) ** len(lam) / (_z_partition(lam) * _z_partition(mu))
        for (lam, mu) in pairs_lam_mu
    ])
    probabilities /= np.sum(probabilities)
    return pairs_lam_mu, probabilities


def _random_pair_lam_mu(pairs_lam_mu, probabilities):
    # noinspection PyShadowingNames
    """Random pair (lambda, mu).

    Parameters
    ----------
    pairs_lam_mu: list of tuple of list
        Cf. :func:`_pairs_lam_mu_and_probabilities`.
    probabilities: np.ndarray
        Cf. :func:`_pairs_lam_mu_and_probabilities`.

    Returns
    -------
    lam: list
        A partition of the number of voters `n`, i.e. a list of integers whose sum is `n`.
    mu: list
        A partition of the number of candidates `m`, i.e. a list of integers whose sum is `m`.

    Examples
    --------
        >>> np.random.seed(42)
        >>> pairs_lam_mu, probabilities = _pairs_lam_mu_and_probabilities(n=3, m=3)
        >>> _random_pair_lam_mu(pairs_lam_mu, probabilities)
        ([1, 1, 1], [1, 1, 1])
    """
    random_index = np.random.choice(len(probabilities), p=probabilities)
    return pairs_lam_mu[random_index]


def _canonical_permutation(lam):
    """Canonical permutation associated to a cycle type, i.e. a partition of `n`.

    Parameters
    ----------
    lam: list
        A partition of `n` (cf. example below), i.e. a list of integers whose sum is `n`.
        The elements of the list (cycle lengths) are given in increasing order.

    Returns
    -------
    permutation: np.ndarray
        The canonical permutation associated to `lam`.

    Examples
    --------
        >>> _canonical_permutation(lam=[3, 3, 5, 6])
        array([ 1,  2,  0,  4,  5,  3,  7,  8,  9, 10,  6, 12, 13, 14, 15, 16, 11])

    The 3 first elements are cycled, then the 3 next ones, then the 5 next ones, then the 6 next ones,
    as specified by the cycle type `lam`.
    """
    cum = np.concatenate(([0], np.cumsum(lam)))
    return np.concatenate([
        np.concatenate((np.arange(start + 1, end), [start]))
        for start, end in zip(cum[:-1], cum[1:])
    ])


def _add_voter_cycle(profile, size_voter_cycle, tau, ranking_first_voter):
    # noinspection PyShadowingNames
    """Add a "voter cycle" to a profile.

    The parameter `profile` will be modified in place.

    Parameters
    ----------
    profile: List of List
        Each row represents the ranking of a voter over the candidates.
    size_voter_cycle: int
        Number of voters that we want to add.
    tau: np.ndarray
        A permutation over the candidates. Will be applied multiple times to generate the cycle.
        When iterated `size_voter_cycle` times, it should be equal to the identity.
    ranking_first_voter: np.ndarray
        The ranking of the first voter over the candidates.

    Examples
    --------
        >>> profile = []
        >>> size_voter_cycle = 6
        >>> tau = np.array([1, 0, 3, 4, 2])  # A 2-cycle and a 3-cycle
        >>> ranking_first_voter = np.array([0, 1, 2, 3, 4])
        >>> _add_voter_cycle(profile, size_voter_cycle, tau, ranking_first_voter)
        >>> np.array(profile)
        array([[0, 1, 2, 3, 4],
               [1, 0, 3, 4, 2],
               [0, 1, 4, 2, 3],
               [1, 0, 2, 3, 4],
               [0, 1, 3, 4, 2],
               [1, 0, 4, 2, 3]])
    """
    # First voter: initial permutation
    ranking = ranking_first_voter
    profile.append(ranking)
    # Other voters: iterate `tau` over `ranking`
    for _ in range(size_voter_cycle - 1):
        ranking = tau[ranking]
        profile.append(ranking)


def _random_profile(n, m, pairs_lam_mu, probabilities, shuffle_voters=False):
    # noinspection PyShadowingNames
    """Random profile in IANC.

    Parameters
    ----------
    n: int
        Number of voters
    m: int
        Number of candidates
    pairs_lam_mu: list of tuple of list
        Cf. :func:`_pairs_lam_mu_and_probabilities`.
    probabilities: np.ndarray
        Cf. :func:`_pairs_lam_mu_and_probabilities`.
    shuffle_voters: bool
        If True, then voters are randomly shuffled at the end of the procedure.

    Returns
    -------
    profile: np.ndarray
        A random profile drawn from IANC. Shape `n` * `m`.

    Examples
    --------
        >>> n = 3
        >>> m = 3
        >>> pairs_lam_mu, probabilities = _pairs_lam_mu_and_probabilities(n, m)
        >>> np.random.seed(51)
        >>> _random_profile(n, m, pairs_lam_mu, probabilities)
        array([[2, 0, 1],
               [0, 2, 1],
               [0, 2, 1]])
        >>> np.random.seed(51)
        >>> _random_profile(n, m, pairs_lam_mu, probabilities, shuffle_voters=True)
        array([[0, 2, 1],
               [0, 2, 1],
               [2, 0, 1]])
    """
    lam, mu = _random_pair_lam_mu(pairs_lam_mu, probabilities)
    tau = _canonical_permutation(mu)
    profile = []
    for size_voter_cycle, n_voter_cycles_this_size in Counter(lam).items():
        for _ in range(n_voter_cycles_this_size):
            _add_voter_cycle(profile, size_voter_cycle, tau, ranking_first_voter=np.random.permutation(m))
    profile = np.array(profile)
    if shuffle_voters:
        profile = profile[np.random.permutation(n), :]
    return profile
