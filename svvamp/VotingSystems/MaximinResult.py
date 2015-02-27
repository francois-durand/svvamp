# -*- coding: utf-8 -*-
"""
Created on Sat Oct  4 19:03:07 2014
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


class MaximinResult(ElectionResult):
    """Results of an election using Maximin method.

<<<<<<< HEAD
    scores[c] is the minimum of the row matrix_duels_ut[c, :] (except the
=======
    scores[c] is the minimum of the row matrix_duels_vtb[c, :] (except the
>>>>>>> origin/master
    diagonal term), i.e. the result of candidate c for her worst duel. The
    candidate with highest score is declared the winner. In case of a tie,
    the candidate with lowest index wins.
    """
    
    _options_parameters = ElectionResult._options_parameters.copy()

    def __init__(self, population, **kwargs):
        super().__init__(population, **kwargs)
        self._log_identity = "MAXIMIN_RESULT"
        
    @property
    def scores(self):
<<<<<<< HEAD
        """1d array of integers. scores[c] is the minimum of the row
        matrix_duels_ut[c, :] (except the diagonal term), i.e. the result of
        candidate c for her worst duel.
=======
        """1d array of integers. ``scores[c]`` is the minimum of the row
        :attr:`~svvamp.Population.matrix_duels_vtb`\ ``[c, :]`` (except the
        diagonal term), i.e. the result of candidate ``c`` for her worst duel.
>>>>>>> origin/master
        """
        if self._scores is None:
            self._mylog("Compute scores", 1)
            self._scores = np.add(
                self.pop.V,
                - np.max(self.pop.matrix_duels_rk, 0))
        return self._scores


if __name__ == '__main__':
    # A quick demo
    preferences_utilities = np.random.randint(-5, 5, (10, 5))
    pop = Population(preferences_utilities)
    election = MaximinResult(pop)
    election.demo(log_depth=3)