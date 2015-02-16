# -*- coding: utf-8 -*-
"""
Created on Thu Oct  9 11:44:40 2014
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

from svvamp.VotingSystems.ExhaustiveBallotResult import ExhaustiveBallotResult
from svvamp.Preferences.Population import Population


class IRVResult(ExhaustiveBallotResult):
    """Results of an election using Instant-Runoff Voting (IRV).

    In case of a tie, candidates with lowest index are privileged.
    """
    
    _options_parameters = ExhaustiveBallotResult._options_parameters.copy()

    def __init__(self, population, **kwargs):
        super().__init__(population, **kwargs)
        self._log_identity = "IRV_RESULT"


if __name__ == '__main__':
    # A quick demo
    import numpy as np

    preferences_utilities = np.random.randint(-5, 5, (10, 5))
    pop = Population(preferences_utilities)
    election = IRVResult(pop)
    election.demo(log_depth=3)