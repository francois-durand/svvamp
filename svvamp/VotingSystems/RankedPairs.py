# -*- coding: utf-8 -*-
"""
Created on Wed Oct  8 11:36:03 2014
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
from svvamp.VotingSystems.RankedPairsResult import RankedPairsResult
from svvamp.Preferences.Population import Population


class RankedPairs(RankedPairsResult, Election):
    """Tideman's Ranked Pairs.
    
    In the matrix of duels, victories (and ties) are sorted by decreasing 
    amplitude. If two duels have the same score, we take first the one where
    the winner has the smallest index; if there is still a choice to make,
    we take first the duel where the loser has the highest index.
    
    Starting with the largest victory, we build a directed graph whose 
    nodes are the candidates and edges are victories. But if a victory
    creates a cycle in the graph, it is not validated and the edge is not
    added.
    
    At the end, we has a transitive connected directed graph, whose adjacency
    relation is included in the relation of victories (with ties broken), 
    matrix_victories_vtb_ctb. The maximal node of this graph (by topological
    order) is declared the winner.
    """
    
    _layout_name = 'Ranked Pairs'
    _options_parameters = Election._options_parameters.copy()
    _options_parameters.update(RankedPairsResult._options_parameters)
    _options_parameters['ICM_option'] = {'allowed': ['exact'],
                                           'default': 'exact'}

    def __init__(self, population, **kwargs):
        super().__init__(population, **kwargs)
        self._log_identity = "RANKED_PAIRS"
        self._class_result = RankedPairsResult
        self._with_two_candidates_reduces_to_plurality = True
        self._is_based_on_strict_rankings = True
        self._meets_Condorcet_c_vtb_ctb = True
        self._precheck_ICM = False
        

if __name__ == '__main__':
    # A quick demo
    import numpy as np

    preferences_utilities = np.random.randint(-5, 5, (10, 5))
    pop = Population(preferences_utilities)
    election = RankedPairs(pop)
    election.demo(log_depth=3)