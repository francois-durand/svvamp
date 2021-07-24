# -*- coding: utf-8 -*-
"""
Created on jul. 21, 2021, 15:26
Copyright François Durand 2014-2021
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
from svvamp.utils.misc import indent
from svvamp.rules.rule_irv import RuleIRV


class StudyRuleCriteria:
    """A set of criteria to study for the simulator about one or several voting rules.

    Parameters
    ----------
    manipulation_criteria : List of str
        Manipulation criteria that need to be studied. If None (default), then all criteria related to
        collective manipulation are taken: ['is_tm_', 'is_um_', 'is_icm_', 'is_cm_']. If [], then no criterion is taken.
        Note that by default, 'is_iia_' and 'is_im_' are NOT included.
    manipulation_criteria_c : List of str
        This is where you put criteria such as 'candidates_cm_', 'candidates_tm_', etc.
    manipulation_only : bool
        If True, then result_criteria and utility criteria are ignored, i.e. set to [].
    result_criteria : List of str
        Result criteria that need to be studied. If None(default), then all criteria are taken:
        ['w_is_weak_condorcet_winner_', 'w_missed_weak_condorcet_winner_', ...].
    utility_criteria : List
        List of tuples  (criterion, func, name), where:
        criterion: the numerical data to be processed.
        func: an aggregating function.
        name: the name chosen for this pair criterion / aggregation function.
        Example: ('total_utility_w_', np.mean, 'total_u_mean').
        Will compute the mean of total_utility_w over the populations and the result will be called 'total_u_mean'.
        If utility_criteria is None (default), then use:
        [('total_utility_w_', np.max, 'total_u_max'),
         ('total_utility_w_', np.min, 'total_u_min'),
         ('total_utility_w_', np.mean, 'total_u_mean'),
         ('total_utility_w_', np.std, 'total_u_std')].
    numerical_criteria : List of str
        Other numerical criteria that need to be studied. These criteria must correspond to attributes in the rule
        that output a pair (inf, sup), for example `nb_candidates_cm_` or `worst_relative_welfare_with_cm_`.
    """

    def __init__(self, manipulation_criteria=None, manipulation_criteria_c=None, manipulation_only=False,
                 result_criteria=None, utility_criteria=None, numerical_criteria=None):
        # Manipulation criteria
        if manipulation_criteria is None:
            manipulation_criteria = [
                'is_tm_',
                'is_um_',
                'is_icm_',
                'is_cm_',
                'is_tm_or_um_',
                'elects_condorcet_winner_rk_even_with_cm_',
            ]
        self.manipulation_criteria = manipulation_criteria
        if manipulation_criteria_c is None:
            manipulation_criteria_c = []
        self.manipulation_criteria_c = manipulation_criteria_c
        # Result criteria
        if manipulation_only:
            self.result_criteria = []
        elif result_criteria is None:
            self.result_criteria = [
                'w_is_condorcet_admissible_',
                'w_is_not_condorcet_admissible_',
                'w_missed_condorcet_admissible_',
                'w_is_weak_condorcet_winner_',
                'w_is_not_weak_condorcet_winner_',
                'w_missed_weak_condorcet_winner_',
                'w_is_condorcet_winner_rk_ctb_',
                'w_is_not_condorcet_winner_rk_ctb_',
                'w_missed_condorcet_winner_rk_ctb_',
                'w_is_condorcet_winner_rk_',
                'w_is_not_condorcet_winner_rk_',
                'w_missed_condorcet_winner_rk_',
                'w_is_condorcet_winner_ut_rel_ctb_',
                'w_is_not_condorcet_winner_ut_rel_ctb_',
                'w_missed_condorcet_winner_ut_rel_ctb_',
                'w_is_condorcet_winner_ut_rel_',
                'w_is_not_condorcet_winner_ut_rel_',
                'w_missed_condorcet_winner_ut_rel_',
                'w_is_condorcet_winner_ut_abs_ctb_',
                'w_is_not_condorcet_winner_ut_abs_ctb_',
                'w_missed_condorcet_winner_ut_abs_ctb_',
                'w_is_condorcet_winner_ut_abs_',
                'w_is_not_condorcet_winner_ut_abs_',
                'w_missed_condorcet_winner_ut_abs_',
                'w_is_resistant_condorcet_winner_',
                'w_is_not_resistant_condorcet_winner_',
                'w_missed_resistant_condorcet_winner_'
            ]
        else:
            self.result_criteria = result_criteria
        # Utility criteria
        if manipulation_only:
            self.utility_criteria = []
        elif utility_criteria is None:
            self.utility_criteria = [
                ('mean_utility_w_', np.max, 'mean_u_max'),
                ('mean_utility_w_', np.min, 'mean_u_min'),
                ('mean_utility_w_', np.mean, 'mean_u_mean'),
                ('mean_utility_w_', np.std, 'mean_u_std'),
                ('relative_social_welfare_w_', np.mean, 'relative_social_welfare_mean'),
            ]
        else:
            self.utility_criteria = utility_criteria
        # Numerical criteria
        if numerical_criteria is None:
            self.numerical_criteria = [
                'nb_candidates_cm_',
                'worst_relative_welfare_with_cm_',
            ]
        else:
            self.numerical_criteria = numerical_criteria

    def __str__(self):
        """
            >>> study_rule_criteria = StudyRuleCriteria(
            ...     manipulation_criteria=['is_cm_'],
            ...     manipulation_criteria_c=['candidates_cm_'],
            ...     result_criteria=['w_is_condorcet_admissible_', 'w_is_weak_condorcet_winner_'],
            ...     utility_criteria=[('mean_utility_w_', np.max, 'mean_u_max'),
            ...                       ('mean_utility_w_', np.min, 'mean_u_min')],
            ...     numerical_criteria=['nb_candidates_cm_', 'worst_relative_welfare_with_cm_'],
            ... )
            >>> print(study_rule_criteria)  # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
            StudyRuleCriteria with:
                manipulation_criteria:
                    is_cm_
                manipulation_criteria_c:
                    candidates_cm_
                result_criteria:
                    w_is_condorcet_admissible_
                    w_is_weak_condorcet_winner_
                utility_criteria:
                    ('mean_utility_w_', <function amax at ...>, 'mean_u_max')
                    ('mean_utility_w_', <function amin at ...>, 'mean_u_min')
                numerical_criteria:
                    nb_candidates_cm_
                    worst_relative_welfare_with_cm_

            >>> study_rule_criteria = StudyRuleCriteria(
            ...     manipulation_criteria=[],
            ...     result_criteria=[],
            ...     utility_criteria=[],
            ...     numerical_criteria=[],
            ... )
            >>> print(study_rule_criteria)  # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
            StudyRuleCriteria with:
                manipulation_criteria: None
                manipulation_criteria_c: None
                result_criteria: None
                utility_criteria: None
                numerical_criteria: None
        """
        # noinspection PyListCreation
        lines = []
        for name in ['manipulation_criteria', 'manipulation_criteria_c', 'result_criteria',
                     'utility_criteria', 'numerical_criteria']:
            criteria = getattr(self, name)
            if criteria:
                lines.append(f'{name}:')
                lines.extend(['    ' + str(criterion) for criterion in criteria])
            else:
                lines.append(f'{name}: None')
        return 'StudyRuleCriteria with:\n' + indent('\n'.join(lines))

    def remove(self, criterion):
        """Remove a criterion

        Remove all occurrences of a criterion (generally zero or one) from this `StudyRuleCriteria` object.

        Parameters
        ----------
        criterion : str

        Returns
        -------
        StudyRuleCriteria
            The object itself.

        Examples
        --------
            >>> study_rule_criteria = StudyRuleCriteria(
            ...     manipulation_criteria=['is_cm_', 'is_tm_'],
            ...     result_criteria=['w_is_condorcet_admissible_', 'w_is_weak_condorcet_winner_'],
            ...     utility_criteria=[('mean_utility_w_', np.max, 'mean_u_max'),
            ...                       ('mean_utility_w_', np.min, 'mean_u_min')],
            ...     numerical_criteria=['nb_candidates_cm_', 'worst_relative_welfare_with_cm_'],
            ... )
            >>> study_rule_criteria.remove('is_cm_')
            >>> study_rule_criteria.remove('w_is_weak_condorcet_winner_')
            >>> study_rule_criteria.remove('mean_u_max')
            >>> study_rule_criteria.remove('worst_relative_welfare_with_cm_')
            >>> print(study_rule_criteria)  # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
            StudyRuleCriteria with:
                manipulation_criteria:
                    is_tm_
                manipulation_criteria_c: None
                result_criteria:
                    w_is_condorcet_admissible_
                utility_criteria:
                    ('mean_utility_w_', <function amin at ...>, 'mean_u_min')
                numerical_criteria:
                    nb_candidates_cm_
        """
        self.manipulation_criteria = [c for c in self.manipulation_criteria if c != criterion]
        self.manipulation_criteria_c = [c for c in self.manipulation_criteria_c if c != criterion]
        self.result_criteria = [c for c in self.result_criteria if c != criterion]
        self.utility_criteria = [c for c in self.utility_criteria if c[2] != criterion]
        self.numerical_criteria = [c for c in self.numerical_criteria if c != criterion]

    def append_manipulation_criterion(self, criterion):
        """Add a manipulation criterion

        Parameters
        ----------
        criterion : str

        Examples
        --------
            >>> study_rule_criteria = StudyRuleCriteria(
            ...     manipulation_criteria=[],
            ...     result_criteria=[],
            ...     utility_criteria=[],
            ...     numerical_criteria=[],
            ... )
            >>> study_rule_criteria.append_manipulation_criterion('is_im_')
            >>> study_rule_criteria.append_manipulation_criterion('is_iia_')
            >>> print(study_rule_criteria)  # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
            StudyRuleCriteria with:
                manipulation_criteria:
                    is_im_
                    is_iia_
                manipulation_criteria_c: None
                result_criteria: None
                utility_criteria: None
                numerical_criteria: None
        """
        if criterion not in self.manipulation_criteria:
            self.manipulation_criteria.append(criterion)

    def append_manipulation_criterion_c(self, criterion):
        """Add a manipulation criterion (for candidates)

        Parameters
        ----------
        criterion : str

        Examples
        --------
            >>> study_rule_criteria = StudyRuleCriteria(
            ...     manipulation_criteria=[],
            ...     manipulation_criteria_c=[],
            ...     result_criteria=[],
            ...     utility_criteria=[],
            ...     numerical_criteria=[],
            ... )
            >>> study_rule_criteria.append_manipulation_criterion_c('candidates_cm_')
            >>> study_rule_criteria.append_manipulation_criterion_c('candidates_tm_')
            >>> print(study_rule_criteria)  # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
            StudyRuleCriteria with:
                manipulation_criteria: None
                manipulation_criteria_c:
                    candidates_cm_
                    candidates_tm_
                result_criteria: None
                utility_criteria: None
                numerical_criteria: None
        """
        if criterion not in self.manipulation_criteria_c:
            self.manipulation_criteria_c.append(criterion)

    def append_result_criterion(self, criterion):
        """Add a result criterion

        Parameters
        ----------
        criterion : str

        Examples
        --------
            >>> study_rule_criteria = StudyRuleCriteria(
            ...     manipulation_criteria=[],
            ...     result_criteria=[],
            ...     utility_criteria=[],
            ...     numerical_criteria=[],
            ... )
            >>> study_rule_criteria.append_result_criterion('w_is_condorcet_admissible_')
            >>> print(study_rule_criteria)  # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
            StudyRuleCriteria with:
                manipulation_criteria: None
                manipulation_criteria_c: None
                result_criteria:
                    w_is_condorcet_admissible_
                utility_criteria: None
                numerical_criteria: None
        """
        if criterion not in self.result_criteria:
            self.result_criteria.append(criterion)

    def append_utility_criterion(self, criterion, func, name):
        """Add a utility criterion

        Parameters
        ----------
        criterion: str
            The numerical data to be processed.
        func: callable
            An aggregating function.
        name: str
            The name chosen for this pair criterion / aggregation function.

        Examples
        --------
            >>> study_rule_criteria = StudyRuleCriteria(
            ...     manipulation_criteria=[],
            ...     result_criteria=[],
            ...     utility_criteria=[],
            ...     numerical_criteria=[],
            ... )
            >>> study_rule_criteria.append_utility_criterion('total_utility_w_', np.sum, 'total_u_sum')
            >>> print(study_rule_criteria)  # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
            StudyRuleCriteria with:
                manipulation_criteria: None
                manipulation_criteria_c: None
                result_criteria: None
                utility_criteria:
                    ('total_utility_w_', <function sum at ...>, 'total_u_sum')
                numerical_criteria: None

        In the above example, the criterion will be used to compute the sum of `total_utility_w_` over all the profiles,
        and the result will be called 'total_u_sum'.
        """
        if name not in [n for (c, f, n) in self.utility_criteria]:
            self.utility_criteria.append((criterion, func, name))

    def append_numerical_criterion(self, criterion):
        """Add a numerical criterion

        Parameters
        ----------
        criterion : str

        Examples
        --------
            >>> study_rule_criteria = StudyRuleCriteria(
            ...     manipulation_criteria=[],
            ...     result_criteria=[],
            ...     utility_criteria=[],
            ...     numerical_criteria=[],
            ... )
            >>> study_rule_criteria.append_numerical_criterion('nb_candidates_cm_')
            >>> print(study_rule_criteria)  # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
            StudyRuleCriteria with:
                manipulation_criteria: None
                manipulation_criteria_c: None
                result_criteria: None
                utility_criteria: None
                numerical_criteria:
                    nb_candidates_cm_
        """
        if criterion not in self.numerical_criteria:
            self.numerical_criteria.append(criterion)

    def check_sanity(self, rule_class):
        """Check sanity of the object

        Preform some basic checks. It is recommended to use this method before launching big simulations.

        Parameters
        ----------
        rule_class : class
            Subclass of :class:`Rule`.

        Examples
        --------
        Detect an illegal manipulation criterion:

            >>> study_rule_criteria = StudyRuleCriteria(manipulation_criteria=['unexpected_criterion'])
            >>> study_rule_criteria.check_sanity(rule_class=RuleIRV)
            Traceback (most recent call last):
            ValueError: Criterion 'unexpected_criterion' is unknown for RuleIRV.

        Detect an illegal manipulation criterion (for candidates):

            >>> study_rule_criteria = StudyRuleCriteria(manipulation_criteria_c=['unexpected_criterion'])
            >>> study_rule_criteria.check_sanity(rule_class=RuleIRV)
            Traceback (most recent call last):
            ValueError: Criterion 'unexpected_criterion' is unknown for RuleIRV.

        Detect an illegal result criterion:

            >>> study_rule_criteria = StudyRuleCriteria(
            ...     manipulation_criteria=[],
            ...     result_criteria=['unexpected_criterion'])
            >>> study_rule_criteria.check_sanity(rule_class=RuleIRV)
            Traceback (most recent call last):
            ValueError: Criterion 'unexpected_criterion' is unknown for RuleIRV.

        Detect an illegal utility criterion:

            >>> study_rule_criteria = StudyRuleCriteria(
            ...     manipulation_criteria=[],
            ...     utility_criteria=[('unexpected_criterion', np.sum, 'the_name')])
            >>> study_rule_criteria.check_sanity(rule_class=RuleIRV)
            Traceback (most recent call last):
            ValueError: Criterion 'unexpected_criterion' is unknown for RuleIRV.

        Detect when a aggregation function is not provided:

            >>> study_rule_criteria = StudyRuleCriteria(
            ...     manipulation_criteria=[],
            ...     utility_criteria=[('mean_utility_w_', 'this_is_not_callable', 'the_name')])
            >>> study_rule_criteria.check_sanity(rule_class=RuleIRV)
            Traceback (most recent call last):
            TypeError: Expected: callable. Got: 'this_is_not_callable'.

        Detect an illegal numerical criterion:

            >>> study_rule_criteria = StudyRuleCriteria(
            ...     manipulation_criteria=[],
            ...     numerical_criteria=['unexpected_criterion'])
            >>> study_rule_criteria.check_sanity(rule_class=RuleIRV)
            Traceback (most recent call last):
            ValueError: Criterion 'unexpected_criterion' is unknown for RuleIRV.
        """
        # Check if manipulation criteria are legal.
        for criterion in self.manipulation_criteria:
            if criterion not in dir(rule_class):
                raise ValueError(f"Criterion '{criterion}' is unknown for {rule_class.__name__}.")
        # Check if manipulation criteria (candidates) are legal.
        for criterion in self.manipulation_criteria_c:
            if criterion not in dir(rule_class):
                raise ValueError(f"Criterion '{criterion}' is unknown for {rule_class.__name__}.")
        # Check if result criteria are legal.
        for criterion in self.result_criteria:
            if criterion not in dir(rule_class):
                raise ValueError(f"Criterion '{criterion}' is unknown for {rule_class.__name__}.")
        # Check if utility criteria are legal.
        for (criterion, func, name) in self.utility_criteria:
            if criterion not in dir(rule_class):
                raise ValueError(f"Criterion '{criterion}' is unknown for {rule_class.__name__}.")
            if not hasattr(func, '__call__'):
                raise TypeError(f"Expected: callable. Got: {repr(func)}.")
        # Check if numerical criteria are legal.
        for criterion in self.numerical_criteria:
            if criterion not in dir(rule_class):
                raise ValueError(f"Criterion '{criterion}' is unknown for {rule_class.__name__}.")
