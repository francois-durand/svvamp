# -*- coding: utf-8 -*-
"""
Created on oct. 16, 2014, 11:35 
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
from svvamp.VotingSystems.IRVDuelsResult import IRVDuelsResult
from svvamp.Preferences.Population import Population


class IRVDuels(IRVDuelsResult, Election):
    """IRV with elimination duels.

    Even round r (including round 0): the two non-eliminated candidates
    who are ranked first (among the non-eliminated candidates) by least voters
    are selected for the elimination duels that is held in round r+1.
    Odd round r: voters vote for the selected candidate they like most in the
    duel. The candidate with least votes is eliminated.

    Obviously, this method meets the Condorcet criterion.

    Our thanks to Laurent Viennot for the idea of voting system.

    See also 'Condorcet-IRV' and 'ICRV' for other Condorcet variants of IRV.
    """
    
    _layout_name = 'IRV Duels'
    _options_parameters = Election._options_parameters.copy()
    _options_parameters.update(IRVDuelsResult._options_parameters)
    _options_parameters['ICM_option'] = {'allowed': ['exact'],
                                           'default': 'exact'}

    def __init__(self, population, **kwargs):
        super().__init__(population, **kwargs)
        self._log_identity = "IRV_DUELS"
        self._class_result = IRVDuelsResult
        self._with_two_candidates_reduces_to_plurality = True
        self._is_based_on_strict_rankings = True
        self._meets_Condorcet_c_vtb_ctb = True
        self._precheck_ICM = False


if __name__ == '__main__':
    # A quick demo
    import numpy as np

    preferences_utilities = np.random.randint(-5, 5, (10, 5))
    pop = Population(preferences_utilities)
    election = IRVDuels(pop)
    election.demo(log_depth=3)