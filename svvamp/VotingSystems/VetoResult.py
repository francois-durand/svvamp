# -*- coding: utf-8 -*-
"""
Created on Mon Sep 29 14:32:51 2014
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


class VetoResult(ElectionResult):
    """Results of an election using Veto (antiplurality).

    Each voter votes again her last-ranked candidate. The candidate with least
    votes against her is declared the winner.

    Ties are broken by natural order on the candidates (lower index wins).
    """
    
    _options_parameters = ElectionResult._options_parameters.copy()

    def __init__(self, population, **kwargs):
        super().__init__(population, **kwargs)
        self._log_identity = "VETO_RESULT"
    
    @property
    def ballots(self):
        """1d array of integers. ballots[v] is the candidate on voter v's 
        ballot.
        """
        if self._ballots is None:
            self._mylog("Compute ballots", 1)
            self._ballots = self.pop.preferences_ranking[:, -1]
        return self._ballots
        
    @property
    def scores(self):
        """1d array of integers. scores(c) is minus one times the number of 
        voters who vote against candidate c.
        """
        if self._scores is None:
            self._mylog("Compute scores", 1)
            self._scores = - np.bincount(self.ballots, minlength=self.pop.C)
        return self._scores


if __name__ == '__main__':
    # A quick demo
    preferences_utilities = np.random.randint(-5, 5, (10, 5))
    pop = Population(preferences_utilities)
    election = VetoResult(pop)
    election.demo(log_depth=3)