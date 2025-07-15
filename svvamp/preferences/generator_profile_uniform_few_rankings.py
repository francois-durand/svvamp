# -*- coding: utf-8 -*-
"""
Created on jul. 11, 2025, 10:27
Copyright François Durand 2025
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
import random
from math import factorial
import numpy as np
from svvamp.preferences.generator_profile import GeneratorProfile
from svvamp.preferences.generator_profile_ic import GeneratorProfileIc
from svvamp.preferences.profile import Profile
from svvamp.utils.misc import preferences_ut_to_preferences_rk


class GeneratorProfileUniformFewRankings(GeneratorProfile):
    """Profile generator drawing rankings uniformly in a random subset of all possible rankings.

    Parameters
    ----------
    n_v : int
        Number of voters.
    n_c : int
        Number of candidates.
    n_max_rankings : int
        Maximum number of rankings to draw. If ``n_max_rankings`` is greater than the total number of rankings,
        then all rankings are drawn.
    sort_voters : bool
        This argument is passed to :class:`Profile`.

    Notes
    -----
    First, a subset of `n_max_rankings` rankings is drawn uniformly at random from the set of all possible rankings.
    Then the ranking of each voter is drawn uniformly from this subset. The whole process is performed at each
    generation of a profile.

    This class is especially useful to find examples and counter-examples that are simple to write, visualize and
    understand.

    Examples
    --------
        >>> generator = GeneratorProfileUniformFewRankings(n_v=10, n_c=3, n_max_rankings=4)
        >>> profile = generator()
        >>> profile.preferences_rk.shape
        (10, 3)
    """

    def __init__(self, n_v, n_c, n_max_rankings, sort_voters=False):
        self.n_v = n_v
        self.n_c = n_c
        self.n_max_rankings = n_max_rankings
        self.use_impartial_culture = (self.n_max_rankings >= factorial(self.n_c))
        self.log_creation = ['UniformFewRankings', n_c, n_v, n_max_rankings]
        super().__init__(sort_voters=sort_voters)

    def __call__(self):
        if self.use_impartial_culture:
            return GeneratorProfileIc(n_v=self.n_v, n_c=self.n_c, sort_voters=self.sort_voters)()
        # Draw a subset of different rankings uniformly at random, of size n_max_rankings
        rankings = set()
        n_rankings = 0
        while n_rankings < self.n_max_rankings:
            ranking = tuple(np.random.permutation(self.n_c))
            if ranking not in rankings:
                rankings.add(ranking)
                n_rankings += 1
        # Draw the preferences of each voter uniformly at random from the subset of rankings
        preferences_rk = np.array(random.choices(list(rankings), k=self.n_v))
        return Profile(
            preferences_rk=preferences_rk,
            log_creation=self.log_creation, sort_voters=self.sort_voters
        )
