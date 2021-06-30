# -*- coding: utf-8 -*-
"""
Created on oct. 21, 2014, 09:54
Copyright François Durand 2014-2018
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
from svvamp.Utils.Cache import cached_property
from svvamp.Preferences.Profile import preferences_ut_to_preferences_rk
from svvamp.Preferences.Profile import Profile


class ProfileSubsetCandidates(Profile):
    """Sub-profile for a subset of the original candidates

    :param parent_profile: A Profile.
    :param candidates_subset: Normally a list of candidates indexes, like [0, 2, 3], but a list of booleans like
        [True False True True False] should work as well. Candidates belonging to the subset.

    N.B.: if candidates_subset is a list of integers, it must be sorted in ascending order.

    N.B.: in this object, candidates are re-numbered, for example [0, 1, 2]. So, if the winner of an election is w in
    the sub-profile, it corresponds to candidates_subset[w] in the parent profile (supposing that candidates_subset is
    given as a list of indexes, not as a list of booleans).
    """

    def __init__(self, parent_profile, candidates_subset):
        self.parent_profile = parent_profile
        self.candidates_subset = candidates_subset
        super().__init__(preferences_ut=parent_profile.preferences_ut[:, candidates_subset],
                         preferences_rk=preferences_ut_to_preferences_rk(
                             self.parent_profile.preferences_borda_rk[:, candidates_subset]),
                         sort_voters=False)

    # %% Matrix of duels

    @cached_property
    def matrix_duels_ut(self):
        return self.parent_profile.matrix_duels_ut[self.candidates_subset, :][:, self.candidates_subset]

    @cached_property
    def matrix_duels_rk(self):
        return self.parent_profile.matrix_duels_rk[self.candidates_subset, :][:, self.candidates_subset]

    @cached_property
    def matrix_victories_ut_abs(self):
        return self.parent_profile.matrix_victories_ut_abs[self.candidates_subset, :][:, self.candidates_subset]

    @cached_property
    def matrix_victories_ut_abs_ctb(self):
        return self.parent_profile.matrix_victories_abs_vtb[self.candidates_subset, :][:, self.candidates_subset]

    @cached_property
    def matrix_victories_ut_rel(self):
        return self.parent_profile.matrix_victories_ut_rel[self.candidates_subset, :][:, self.candidates_subset]

    @cached_property
    def matrix_victories_ut_rel_ctb(self):
        return self.parent_profile.matrix_victories_ut_rel_ctb[self.candidates_subset, :][:, self.candidates_subset]

    @cached_property
    def matrix_victories_rk(self):
        return self.parent_profile.matrix_victories_rk[self.candidates_subset, :][:, self.candidates_subset]

    @cached_property
    def matrix_victories_rk_ctb(self):
        return self.parent_profile.matrix_victories_rk_ctb[self.candidates_subset, :][:, self.candidates_subset]

    # %% Total utilities

    @cached_property
    def total_utility_c(self):
        return self.parent_profile.total_utility_c[self.candidates_subset]
