# -*- coding: utf-8 -*-
"""
Created on nov. 03, 2014, 20:36
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

import itertools
import numpy as np

from svvamp.preferences.profile import Profile
from svvamp.utils.misc import indent


class StudyProfileCriteria():

    def __init__(self, boolean_criteria=None, numerical_criteria=None,
                 special_candidates_criteria=None, array_criteria=None,
                 matrix_criteria=None):
        """A set of criteria to study for the simulator about the profiles.

        Parameters
        ----------
        boolean_criteria : List of str
            Each string must correspond to a boolean criterion defined in Profile.
        numerical_criteria : List of tuple
            Each tuple is of the form (criterion, func, name), where:
                criterion: the numerical data to be processed.
                func: an aggregating function.
                name: the name chosen for this pair criterion / aggregation function.
        special_candidates_criteria : List of str
            Each string must correspond to an attribute defined in Profile that returns the
            index of a candidate or NaN.
        array_criteria : List of str
            Each string must correspond to a attribute defined in Profile that returns an 1d array of size n_c
            (the number of candidates).
        matrix_criteria : List of str
            Each string must correspond to a attribute defined in Profile that returns an array (whatever
            its size).

        Examples
        --------
        For each argument, if None (default), then a quite extensive pre-defined list is used. Hence, the typical usage
        is simply:

            >>> study_profile_criteria = StudyProfileCriteria()

        Here is an example of manually defined `numerical_criteria`:

            >>> study_profile_criteria = StudyProfileCriteria(numerical_criteria=[
            ...     ('total_utility_mean', np.mean, 'total_u_mean_mean')
            ... ])

        For each profile, this will compute `total_utility_mean` (mean over the candidates). Then it will use `np.mean`
        to calculate the mean over the profiles. The result will be called 'total_u_mean_mean'.
        """
        if boolean_criteria is None:
            boolean_criteria = [
                'exists_condorcet_admissible',
                'exists_weak_condorcet_winner',
                'exists_condorcet_winner_rk_ctb',
                'exists_condorcet_winner_rk',
                'exists_condorcet_winner_ut_rel_ctb',
                'exists_condorcet_winner_ut_rel',
                'exists_condorcet_winner_ut_abs_ctb',
                'exists_condorcet_winner_ut_abs',
                'exists_resistant_condorcet_winner',
                'exists_condorcet_order_rk_ctb',
                'exists_condorcet_order_rk',
                'exists_condorcet_order_ut_rel_ctb',
                'exists_condorcet_order_ut_rel',
                'exists_condorcet_order_ut_abs_ctb',
                'exists_condorcet_order_ut_abs',
                'exists_majority_favorite_rk_ctb',
                'exists_majority_favorite_rk',
                'exists_majority_favorite_ut_ctb',
                'exists_majority_favorite_ut',
                'exists_irv_immune_candidate',
                'not_exists_condorcet_admissible',
                'not_exists_weak_condorcet_winner',
                'not_exists_condorcet_winner_rk_ctb',
                'not_exists_condorcet_winner_rk',
                'not_exists_condorcet_winner_ut_rel_ctb',
                'not_exists_condorcet_winner_ut_rel',
                'not_exists_condorcet_winner_ut_abs_ctb',
                'not_exists_condorcet_winner_ut_abs',
                'not_exists_resistant_condorcet_winner',
                'not_exists_condorcet_order_rk_ctb',
                'not_exists_condorcet_order_rk',
                'not_exists_condorcet_order_ut_rel_ctb',
                'not_exists_condorcet_order_ut_rel',
                'not_exists_condorcet_order_ut_abs_ctb',
                'not_exists_condorcet_order_ut_abs',
                'not_exists_majority_favorite_rk_ctb',
                'not_exists_majority_favorite_rk',
                'not_exists_majority_favorite_ut_ctb',
                'not_exists_majority_favorite_ut',
            ]
        self.boolean_criteria = boolean_criteria
        if numerical_criteria is None:
            numerical_criteria = [
                ('mean_utility_min', np.min, 'mean_u_min_min'),
                ('mean_utility_min', np.max, 'mean_u_min_max'),
                ('mean_utility_min', np.mean, 'mean_u_min_mean'),
                ('mean_utility_min', np.std, 'mean_u_min_std'),
                ('mean_utility_max', np.min, 'mean_u_max_min'),
                ('mean_utility_max', np.max, 'mean_u_max_max'),
                ('mean_utility_max', np.mean, 'mean_u_max_mean'),
                ('mean_utility_max', np.std, 'mean_u_max_std'),
                ('mean_utility_mean', np.min, 'mean_u_mean_min'),
                ('mean_utility_mean', np.max, 'mean_u_mean_max'),
                ('mean_utility_mean', np.mean, 'mean_u_mean_mean'),
                ('mean_utility_mean', np.std, 'mean_u_mean_std'),
                ('nb_condorcet_admissible', np.mean,
                 'nb_condorcet_admissible_mean'),
                ('nb_weak_condorcet_winners', np.mean,
                 'nb_weak_condorcet_winners_mean')
            ]
        self.numerical_criteria = numerical_criteria
        if special_candidates_criteria is None:
            special_candidates_criteria = [
                'condorcet_winner_rk_ctb',
                'condorcet_winner_rk',
                'condorcet_winner_ut_rel_ctb',
                'condorcet_winner_ut_rel',
                'condorcet_winner_ut_abs_ctb',
                'condorcet_winner_ut_abs',
                'resistant_condorcet_winner',
                'majority_favorite_rk_ctb',
                'majority_favorite_rk',
                'majority_favorite_ut_ctb',
                'majority_favorite_ut',
            ]
        self.special_candidates_criteria = special_candidates_criteria
        if array_criteria is None:
            array_criteria = [
                'plurality_scores_rk',
                'plurality_scores_ut',
                'condorcet_admissible_candidates',
                'weak_condorcet_winners',
                'borda_score_c_rk',
                'borda_score_c_ut',
                'total_utility_c',
                'mean_utility_c',
                'relative_social_welfare_c',
            ]
        self.array_criteria = array_criteria
        if matrix_criteria is None:
            matrix_criteria = [
                'preferences_ut',
                'preferences_borda_rk',
                'preferences_borda_ut',
                'matrix_duels_ut',
                'matrix_duels_rk',
                'matrix_victories_ut_abs',
                'matrix_victories_ut_abs_ctb',
                'matrix_victories_ut_rel',
                'matrix_victories_ut_rel_ctb',
                'matrix_victories_rk',
                'matrix_victories_rk_ctb'
            ]
        self.matrix_criteria = matrix_criteria

    def __str__(self):
        """
            >>> study_profile_criteria = StudyProfileCriteria(
            ...     boolean_criteria=['exists_condorcet_winner_rk'],
            ...     numerical_criteria=[('mean_utility_min', np.min, 'mean_u_min_min')],
            ...     special_candidates_criteria=['condorcet_winner_rk'],
            ...     array_criteria=['total_utility_c'],
            ...     matrix_criteria=['matrix_duels_rk']
            ... )
            >>> print(study_profile_criteria)  # doctest: +ELLIPSIS
            StudyProfileCriteria with:
                boolean_criteria:
                    exists_condorcet_winner_rk
                numerical_criteria:
                    ('mean_utility_min', <function amin at ...>, 'mean_u_min_min')
                special_candidates_criteria:
                    condorcet_winner_rk
                array_criteria:
                    total_utility_c
                matrix_criteria:
                    matrix_duels_rk

            >>> study_profile_criteria = StudyProfileCriteria(
            ...     boolean_criteria=[],
            ...     numerical_criteria=[],
            ...     special_candidates_criteria=[],
            ...     array_criteria=[],
            ...     matrix_criteria=[]
            ... )
            >>> print(study_profile_criteria)
            StudyProfileCriteria with:
                boolean_criteria: None
                numerical_criteria: None
                special_candidates_criteria: None
                array_criteria: None
                matrix_criteria: None
        """
        # noinspection PyListCreation
        lines = []
        for name in ['boolean_criteria', 'numerical_criteria', 'special_candidates_criteria',
                     'array_criteria', 'matrix_criteria']:
            criteria = getattr(self, name)
            if criteria:
                lines.append(f'{name}:')
                lines.extend(['    ' + str(criterion) for criterion in criteria])
            else:
                lines.append(f'{name}: None')
        return 'StudyProfileCriteria with:\n' + indent('\n'.join(lines))

    def check_sanity(self):
        """Check sanity of the object

        Preforms some basic checks. It is recommended to use this method before launching big simulations.

        Examples
        --------
        Detect an illegal criterion:

            >>> study_profile_criteria = StudyProfileCriteria(boolean_criteria=['unexpected_criterion'])
            >>> study_profile_criteria.check_sanity()
            Traceback (most recent call last):
            ValueError: Attribute 'unexpected_criterion' is unknown for Profile.

        Detect when a aggregation function is not provided:

            >>> study_profile_criteria = StudyProfileCriteria(numerical_criteria=[
            ...     ('mean_utility_min', 'this_is_not_callable', 'mean_u_min_min')
            ... ])
            >>> study_profile_criteria.check_sanity()
            Traceback (most recent call last):
            TypeError: Expected: callable. Got: 'this_is_not_callable'.
        """
        for criterion in itertools.chain(self.boolean_criteria, self.special_candidates_criteria,
                                         self.array_criteria, self.matrix_criteria):
            if criterion not in dir(Profile):
                raise ValueError(f"Attribute {repr(criterion)} is unknown for Profile.")
        for (criterion, func, name) in self.numerical_criteria:
            if criterion not in dir(Profile):
                raise ValueError(f"Attribute {repr(criterion)} is unknown for Profile.")
            if not callable(func):
                raise TypeError(f"Expected: callable. Got: {repr(func)}.")
        print('StudyProfileCriteria: Sanity check was successful.')

    def remove(self, criterion):
        """Remove a criterion

        Parameters
        ----------
        criterion : str

        Examples
        --------
        Remove all occurrences of `criterion` (generally zero or one):

            >>> study_profile_criteria = StudyProfileCriteria(
            ...     boolean_criteria=['exists_condorcet_admissible', 'exists_weak_condorcet_winner'],
            ...     numerical_criteria=[],
            ...     special_candidates_criteria=[],
            ...     array_criteria=[],
            ...     matrix_criteria=[]
            ... )
            >>> study_profile_criteria.remove('exists_weak_condorcet_winner')
            >>> print(study_profile_criteria)
            StudyProfileCriteria with:
                boolean_criteria:
                    exists_condorcet_admissible
                numerical_criteria: None
                special_candidates_criteria: None
                array_criteria: None
                matrix_criteria: None

        For numerical criteria, use the last element of the triple, i.e. the name:

            >>> study_profile_criteria = StudyProfileCriteria(
            ...     boolean_criteria=[],
            ...     numerical_criteria=[('mean_utility_min', np.min, 'mean_u_min_min'),
            ...                         ('mean_utility_min', np.max, 'mean_u_min_max')],
            ...     special_candidates_criteria=[],
            ...     array_criteria=[],
            ...     matrix_criteria=[]
            ... )
            >>> study_profile_criteria.remove('mean_u_min_min')
            >>> print(study_profile_criteria)  # doctest: +ELLIPSIS
            StudyProfileCriteria with:
                boolean_criteria: None
                numerical_criteria:
                    ('mean_utility_min', <function amax at ...>, 'mean_u_min_max')
                special_candidates_criteria: None
                array_criteria: None
                matrix_criteria: None

        """
        for criteria in [self.boolean_criteria, self.special_candidates_criteria,
                         self.array_criteria, self.matrix_criteria]:
            criteria[:] = [c for c in criteria if c != criterion]
        self.numerical_criteria[:] = [c for c in self.numerical_criteria if c[2] != criterion]

    def append_boolean_criterion(self, criterion):
        """Add a boolean criterion

        Parameters
        ----------
        criterion : str

        Examples
        --------
            >>> study_profile_criteria = StudyProfileCriteria(
            ...     boolean_criteria=['exists_condorcet_admissible'],
            ...     numerical_criteria=[],
            ...     special_candidates_criteria=[],
            ...     array_criteria=[],
            ...     matrix_criteria=[]
            ... )
            >>> study_profile_criteria.append_boolean_criterion('exists_weak_condorcet_winner')
            >>> print(study_profile_criteria)
            StudyProfileCriteria with:
                boolean_criteria:
                    exists_condorcet_admissible
                    exists_weak_condorcet_winner
                numerical_criteria: None
                special_candidates_criteria: None
                array_criteria: None
                matrix_criteria: None
        """
        if criterion not in self.boolean_criteria:
            self.boolean_criteria.append(criterion)

    def append_numerical_criterion(self, criterion, func, name):
        """Add a numerical criterion

        Parameters
        ----------
        criterion : str
            The numerical data to be processed.
        func : callable
            An aggregating function.
        name : str
            The name chosen for this pair criterion / aggregation function.

        Examples
        --------
            >>> study_profile_criteria = StudyProfileCriteria(
            ...     boolean_criteria=[],
            ...     numerical_criteria=[],
            ...     special_candidates_criteria=[],
            ...     array_criteria=[],
            ...     matrix_criteria=[]
            ... )
            >>> study_profile_criteria.append_numerical_criterion(
            ...     'total_utility_mean', np.mean, 'total_u_mean_mean'
            ... )
            >>> print(study_profile_criteria)  # doctest: +ELLIPSIS
            StudyProfileCriteria with:
                boolean_criteria: None
                numerical_criteria:
                    ('total_utility_mean', <function mean at ...>, 'total_u_mean_mean')
                special_candidates_criteria: None
                array_criteria: None
                matrix_criteria: None

        For each profile, this will compute `total_utility_mean` (mean over the candidates). Then use `np.mean` to
        calculate the mean over the profiles. The result will be called 'total_u_mean_mean'.
        """
        if name not in [n for (c, f, n) in self.numerical_criteria]:
            self.numerical_criteria.append((criterion, func, name))

    def rename_numerical_criterion(self, old_name, new_name):
        """Rename a numerical criterion

        Parameters
        ----------
        old_name : str
        new_name : str

        Examples
        --------
            >>> study_profile_criteria = StudyProfileCriteria(
            ...     boolean_criteria=[],
            ...     numerical_criteria=[('total_utility_mean', np.mean, 'total_u_mean_mean')],
            ...     special_candidates_criteria=[],
            ...     array_criteria=[],
            ...     matrix_criteria=[]
            ... )
            >>> study_profile_criteria.rename_numerical_criterion('total_u_mean_mean', 'the_new_name')
            >>> print(study_profile_criteria)  # doctest: +ELLIPSIS
            StudyProfileCriteria with:
                boolean_criteria: None
                numerical_criteria:
                    ('total_utility_mean', <function mean at ...>, 'the_new_name')
                special_candidates_criteria: None
                array_criteria: None
                matrix_criteria: None
        """
        self.numerical_criteria[:] = [
            (c, f, new_name) if n == old_name else (c, f, n)
            for (c, f, n) in self.numerical_criteria
        ]

    def append_special_candidates_criterion(self, criterion):
        """Add a 'special candidates' criterion

        Parameters
        ----------
        criterion : str

        Examples
        --------
            >>> study_profile_criteria = StudyProfileCriteria(
            ...     boolean_criteria=[],
            ...     numerical_criteria=[],
            ...     special_candidates_criteria=['condorcet_winner_rk_ctb'],
            ...     array_criteria=[],
            ...     matrix_criteria=[]
            ... )
            >>> study_profile_criteria.append_special_candidates_criterion('condorcet_winner_rk')
            >>> print(study_profile_criteria)
            StudyProfileCriteria with:
                boolean_criteria: None
                numerical_criteria: None
                special_candidates_criteria:
                    condorcet_winner_rk_ctb
                    condorcet_winner_rk
                array_criteria: None
                matrix_criteria: None
        """
        if criterion not in self.special_candidates_criteria:
            self.special_candidates_criteria.append(criterion)

    def append_array_criterion(self, criterion):
        """Add an array criterion

        Parameters
        ----------
        criterion : str

        Examples
        --------
            >>> study_profile_criteria = StudyProfileCriteria(
            ...     boolean_criteria=[],
            ...     numerical_criteria=[],
            ...     special_candidates_criteria=[],
            ...     array_criteria=['plurality_scores_rk'],
            ...     matrix_criteria=[]
            ... )
            >>> study_profile_criteria.append_array_criterion('plurality_scores_ut')
            >>> print(study_profile_criteria)
            StudyProfileCriteria with:
                boolean_criteria: None
                numerical_criteria: None
                special_candidates_criteria: None
                array_criteria:
                    plurality_scores_rk
                    plurality_scores_ut
                matrix_criteria: None
        """
        if criterion not in self.array_criteria:
            self.array_criteria.append(criterion)

    def append_matrix_criterion(self, criterion):
        """Add a matrix criterion

        Parameters
        ----------
        criterion : str

        Examples
        --------
            >>> study_profile_criteria = StudyProfileCriteria(
            ...     boolean_criteria=[],
            ...     numerical_criteria=[],
            ...     special_candidates_criteria=[],
            ...     array_criteria=[],
            ...     matrix_criteria=['preferences_ut']
            ... )
            >>> study_profile_criteria.append_matrix_criterion('preferences_borda_rk')
            >>> print(study_profile_criteria)
            StudyProfileCriteria with:
                boolean_criteria: None
                numerical_criteria: None
                special_candidates_criteria: None
                array_criteria: None
                matrix_criteria:
                    preferences_ut
                    preferences_borda_rk
        """
        if criterion not in self.matrix_criteria:
            self.matrix_criteria.append(criterion)
