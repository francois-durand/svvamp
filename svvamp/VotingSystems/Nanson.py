# -*- coding: utf-8 -*-
"""
Created on Sun Oct  5 22:38:22 2014
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

from svvamp.VotingSystems.Election import Election
from svvamp.VotingSystems.NansonResult import NansonResult
from svvamp.Preferences.Population import Population


class Nanson(NansonResult, Election):
    """Nanson method.
    
    At each round, all candidates with a Borda score strictly lower than 
    average are simultaneously eliminated. When all remaining candidates have 
    the same Borda score, it means that the matrix of duels (for this subset
    of candidates) has only ties. Then the candidate with lowest index is 
    declared the winner.
    
    Since a Condorcet winner has always a Borda score higher than average,
    Nanson method meets the Condorcet criterion.
    """
    
    _layout_name = 'Nanson'
    _options_parameters = Election._options_parameters.copy()
    _options_parameters.update(NansonResult._options_parameters)
    _options_parameters['ICM_option'] = {'allowed': ['exact'],
                                           'default': 'exact'}

    def __init__(self, population, **kwargs):
        super().__init__(population, **kwargs)
        self._log_identity = "NANSON"
        self._class_result = NansonResult
        self._with_two_candidates_reduces_to_plurality = True
        self._is_based_on_strict_rankings = True
        self._meets_Condorcet_c_vtb_ctb = True
        self._precheck_ICM = False


if __name__ == '__main__':
    # A quick demo
    import numpy as np

    preferences_utilities = np.random.randint(-5, 5, (8, 4))
    pop = Population(preferences_utilities)
    election = Nanson(pop)
    election.CM_option = 'exact'
    election.demo(log_depth=3)