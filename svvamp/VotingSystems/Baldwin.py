# -*- coding: utf-8 -*-
"""
Created on Tue Sep 30 10:52:47 2014
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
from svvamp.VotingSystems.BaldwinResult import BaldwinResult
from svvamp.Preferences.Population import Population


class Baldwin(BaldwinResult, Election):
    """Baldwin method.

    Inherits functions and optional parameters from superclasses
    :class:`~svvamp.ElectionResult` and :class:`~svvamp.Election`.

    :Example:

    >>> import svvamp
    >>> pop = svvamp.PopulationSpheroid(V=100, C=5)
    >>> election = svvamp.Baldwin(pop)

    Each voter provides a strict order of preference. The candidate with
    lowest Borda score is eliminated. Then the new Borda scores are computed.
    Etc. Ties are broken in favor of lower-index candidates: in case of a tie,
    the candidate with highest index is eliminated.

    Since a Condorcet winner has always a Borda score higher than average,
    Baldwin method meets the Condorcet criterion.

    :meth:`~svvamp.Election.CM`: Non-polynomial or non-exact algorithms
    from superclass :class:`~svvamp.Election`.

    :meth:`~svvamp.Election.ICM`: Exact in polynomial time.

    :meth:`~svvamp.Election.IM`: Deciding IM is NP-complete. Non-polynomial
    or non-exact algorithms from superclass :class:`~svvamp.Election`.

    :meth:`~svvamp.Election.not_IIA`: Exact in polynomial time.

    :meth:`~svvamp.Election.TM`: Exact in polynomial time.

    :meth:`~svvamp.Election.UM`: Non-polynomial or non-exact algorithms from
    superclass :class:`~svvamp.Election`.

    References:

        'Complexity of and algorithms for the manipulation of Borda,
        Nanson's and Baldwin's voting rules', Jessica Davies,
        George Katsirelos, Nina Narodytska, Toby Walsh and Lirong Xia, 2014.
    """

    _layout_name = 'Baldwin'
    _options_parameters = Election._options_parameters.copy()
    _options_parameters.update(BaldwinResult._options_parameters)
    _options_parameters['ICM_option'] = {'allowed': ['exact'],
                                           'default': 'exact'}

    def __init__(self, population, **kwargs):
        super().__init__(population, **kwargs)
        self._log_identity = "BALDWIN"
        self._class_result = Baldwin
        self._with_two_candidates_reduces_to_plurality = True
        self._is_based_on_strict_rankings = True
        # Consider vtb case. If c is a Condorcet winner with vtb and ctb, then
        # she has at worst ties in matrix_duels_vtb, so she has at least the
        # average Borda score. Can she be eliminated? For that, it is
        # necessary that all other candidates have exactly the average,
        # i.e. matrix_duels_vtb is a general tie.
        # Since she is Condorcet winner vtb/ctb, she must be candidate 0,
        # so our tie-breaking rule eliminates another candidate.
        # Conclusion: this voting system meets Condorcet criterion vtb/ctb.
        self._meets_Condorcet_c_vtb_ctb = True
        # Let us consider the following example.
        # preferences_borda_novtb:
        # [ 1   0 ]
        # [0.5 0.5]
        # [0.5 0.5]
        # ==> candidate 0 is a relative Condorcet winner.
        # preferences_borda_vtb (with vtb):
        # [ 1   0 ]
        # [ 0   1 ]
        # [ 0   1 ]
        # ==> candidate 1 wins.
        # Hence self._meets_Condorcet_c_rel = False
        self._precheck_ICM = False


if __name__ == '__main__':
    # A quick demo
    import numpy as np

    preferences_utilities = np.random.randint(-5, 5, (10, 5))
    pop = Population(preferences_utilities)
    election = Baldwin(pop)
    election.demo(log_depth=3)