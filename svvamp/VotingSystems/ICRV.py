# -*- coding: utf-8 -*-
"""
Created on Wed Oct 15 17:46:00 2014
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
from svvamp.VotingSystems.ICRVResult import ICRVResult
from svvamp.Preferences.Population import Population


class ICRV(ICRVResult, Election):
    """Instant-Condorcet Runoff Voting (ICRV).

    Even round r (including round 0): if a candidate w has only victories
    against all other non-eliminated candidates (i.e. is a Condorcet winner
    in this subset, in the sense of matrix_victories_vtb), then w is declared
    the winner.
    Odd round r: the candidate who is ranked first (among non-eliminated
    candidates) by least voters is eliminated, like in IRV.

    See also 'Condorcet-IRV' for another Condorcet variant of IRV.
    """
    
    _layout_name = 'ICRV'
    _options_parameters = Election._options_parameters.copy()
    _options_parameters.update(ICRVResult._options_parameters)
    _options_parameters['ICM_option'] = {'allowed': ['exact'],
                                           'default': 'exact'}

    def __init__(self, population, **kwargs):
        super().__init__(population, **kwargs)
        self._log_identity = "ICRV"
        self._class_result = ICRVResult
        self._with_two_candidates_reduces_to_plurality = True
        self._is_based_on_strict_rankings = True
        self._meets_majority_favorite_c_vtb_ctb = True
        self._meets_Condorcet_c_vtb = True
        self._precheck_ICM = False


if __name__ == '__main__':
    # A quick demo
    import numpy as np

    preferences_utilities = np.random.randint(-5, 5, (10, 5))
    pop = Population(preferences_utilities)
    election = ICRV(pop)
    election.demo(log_depth=3)