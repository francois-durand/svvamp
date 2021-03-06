# -*- coding: utf-8 -*-
"""
Created on oct. 21, 2014, 09:54 
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

from svvamp.Preferences.Population import \
    preferences_ut_to_preferences_rk
from svvamp.Preferences.Population import Population


class PopulationSubsetCandidates(Population):
    """Sub-population for a subset of the original candidates"""

    def __init__(self, mother, candidates_subset):
        """Create a sub-population for a subset of the original candidates

        Arguments:
        mother -- A Population object.
        candidates_subset -- Normally a list of candidates indexes, like
        [0, 2, 3], but a list of booleans like [True False True True False]
        should work as well. Candidates belonging to the subset.

        N.B.: if candidates_subset is a list of integers, it must be sorted in
        ascending order.

        N.B.: in this object, candidates are re-numbered, for example
        [0, 1, 2]. So, if the winner of an election is w in the
        sub-population, it corresponds to candidates_subset[w] in the mother
        population (supposing that candidates_subset is given as a list of
        indexes, not as a list of booleans).
        """
        Population.__init__(
            self,
            preferences_ut=mother.preferences_ut[
                :, candidates_subset]
        )
        self._mother = mother
        self._candidates_subset = candidates_subset
        if self.mother._voters_sorted_by_rk:
            self._voters_sorted_by_rk = True

    @property
    def mother(self):
        """'Mother' population (with all candidates).
        """
        return self._mother

    @property
    def candidates_subset(self):
        """Subset of candidates used to define the sub-population.

        Normally a list of candidates indexes, like [0, 2, 3], but a list
        of boolean like [True False True True False] should work as well.
        """
        return self._candidates_subset

    @property
    def preferences_rk(self):
        """This is the essential function of this class: instead of using a
        new voter tie-breaking rule, we use the sub-rankings of the mother
        population.
        """
        if self._preferences_rk is None:
            self._preferences_rk = (
                preferences_ut_to_preferences_rk(
                    self.mother.preferences_borda_rk[
                        :, self.candidates_subset]))
        return self._preferences_rk

    #%% Matrix of duels

    @property
    def matrix_duels_ut(self):
        if self._matrix_duels_ut is None:
            self._matrix_duels_ut = self.mother.matrix_duels_ut[
                self.candidates_subset, :][:, self.candidates_subset]
        return self._matrix_duels_ut

    @property
    def matrix_duels_rk(self):
        if self._matrix_duels_rk is None:
            self._matrix_duels_rk = self.mother.matrix_duels_rk[
                self.candidates_subset, :][:, self.candidates_subset]
        return self._matrix_duels_rk

    @property
    def matrix_victories_ut_abs(self):
        if self._matrix_victories_ut_abs is None:
            self._matrix_victories_ut_abs = self.mother.matrix_victories_ut_abs[
                self.candidates_subset, :][:, self.candidates_subset]
        return self._matrix_victories_ut_abs

    @property
    def matrix_victories_ut_abs_ctb(self):
        if self._matrix_victories_ut_abs_ctb is None:
            self._matrix_victories_ut_abs_ctb = \
                self.mother.matrix_victories_abs_vtb[
                    self.candidates_subset, :][:, self.candidates_subset]
        return self._matrix_victories_ut_abs_ctb

    @property
    def matrix_victories_ut_rel(self):
        if self._matrix_victories_ut_rel is None:
            self._matrix_victories_ut_rel = self.mother.matrix_victories_ut_rel[
                self.candidates_subset, :][:, self.candidates_subset]
        return self._matrix_victories_ut_rel

    @property
    def matrix_victories_ut_rel_ctb(self):
        if self._matrix_victories_ut_rel_ctb is None:
            self._matrix_victories_ut_rel_ctb = \
                self.mother.matrix_victories_ut_rel_ctb[
                    self.candidates_subset, :][:, self.candidates_subset]
        return self._matrix_victories_ut_rel_ctb

    @property
    def matrix_victories_rk(self):
        if self._matrix_victories_rk is None:
            self._matrix_victories_rk = self.mother.matrix_victories_rk[
                self.candidates_subset, :][:, self.candidates_subset]
        return self._matrix_victories_rk

    @property
    def matrix_victories_rk_ctb(self):
        if self._matrix_victories_rk_ctb is None:
            self._matrix_victories_rk_ctb = \
                self.mother.matrix_victories_rk_ctb[
                    self.candidates_subset, :][:, self.candidates_subset]
        return self._matrix_victories_rk_ctb

    #%% Total utilities

    @property
    def total_utility_c(self):
        if self._total_utility_c is None:
            self._total_utility_c = self.mother.total_utility_c[
                self.candidates_subset]
        return self._total_utility_c