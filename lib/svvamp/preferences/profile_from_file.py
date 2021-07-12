# -*- coding: utf-8 -*-
"""
Created on nov. 28, 2018, 16:57
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

import numpy as np
import pandas as pd

from svvamp.utils.pref_lib import preflib_to_preferences_ut
from svvamp.preferences.profile import Profile


class ProfileFromFile(Profile):

    def __init__(self, file_name, sort_voters=True):
        """Profile from a file.

        Parameters
        ----------
        file_name : str
            The name of the file.
        sort_voters : bool
            Whether the voters should be sorted. Cf. :class:`Profile`.

        Notes
        -----
        * If the file name ends with '.t.csv' (t = transposed format): simple table of utilities with candidates
          declared in the first column and voters declared in the first row.
        * If the file name ends with '.csv' (but not '.t.csv'), candidates must be declared in the first row and voters
          in the first column.
        * Otherwise, the file is considered as a PrefLib file. In that case, since information is ordinal only,
          ``preferences_ut[v, c]`` is set to the Borda score (with no vtb) minus ``(C - 1) / 2``. This way,
          utilities are between ``- (C - 1) / 2`` and ``(C - 1) / 2``.
        """
        if file_name[-4:] == '.csv':
            df = pd.read_csv(filepath_or_buffer=file_name, sep=';', index_col=0)
            if file_name[-6:-4] == '.t':
                preferences_ut = df.transpose().values
                labels_candidates = df.index.values.astype(np.str)
            else:
                preferences_ut = df.values
                labels_candidates = df.columns.values.astype(np.str)
            pop_temp = Profile(preferences_ut)
            nb_victories_temp = np.sum(pop_temp.matrix_victories_ut_rel, 1)
            scores_temp = nb_victories_temp + pop_temp.borda_score_c_ut / pop_temp.n_c / pop_temp.n_v
            candidates_best_to_worst = np.argsort(- scores_temp, kind='mergesort')
            preferences_ut = preferences_ut[:, candidates_best_to_worst]
            labels_candidates = labels_candidates[candidates_best_to_worst]
        else:
            preferences_ut, labels_candidates = preflib_to_preferences_ut(file_name)
        super().__init__(preferences_ut=preferences_ut,
                         log_creation=['From file', 'File name', file_name],
                         labels_candidates=labels_candidates, sort_voters=sort_voters)
