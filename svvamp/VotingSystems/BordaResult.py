# -*- coding: utf-8 -*-
"""
Created on Tue Sep 30 20:40:10 2014
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

from svvamp.VotingSystems.ElectionResult import ElectionResult
from svvamp.Preferences.Population import Population


class BordaResult(ElectionResult):
    """Results of an election using Borda count.

    Voter v gives C - 1 points to her top-ranked candidate, C - 2 to the
    second, ..., 0 to the last.

    Ties are broken by natural order on the candidates (lower index wins).
    """
    
    _options_parameters = ElectionResult._options_parameters.copy()

    def __init__(self, population, **kwargs):
        super().__init__(population, **kwargs)
        self._log_identity = "BORDA_RESULT"
    
    @property
    def ballots(self):
        """2d array of integers. ``ballots[v, c]`` is the Borda score
        attributed to candidate ``c`` by voter ``v`` (between 0 and ``C - 1``).
        """
        if self._ballots is None:
            self._mylog("Compute ballots", 1)
            self._ballots = self.pop.preferences_borda_rk
        return self._ballots
        
    @property
    def scores(self):
        """1d array of integers. ``scores[c]`` is the total Borda score for
        candidate ``c``.
        """
        if self._scores is None:
            self._mylog("Compute scores", 1)
            self._scores = self.pop.borda_score_c_rk
        return self._scores


if __name__ == '__main__':
    # A quick demo
    preferences_utilities = np.random.randint(-5, 5, (10, 5))
    pop = Population(preferences_utilities)
    election = BordaResult(pop)
    election.demo(log_depth=3)