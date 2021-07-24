# -*- coding: utf-8 -*-
"""
Created on jul. 21, 2021, 21:16
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
import inspect
import copy
from svvamp.meta.study_rule_criteria import StudyRuleCriteria
from svvamp.rules.all_rule_classes import ALL_RULE_CLASSES
from svvamp.rules.rule_approval import RuleApproval
from svvamp.rules.rule_plurality import RulePlurality
from svvamp.rules.rule_ranked_pairs import RuleRankedPairs
from svvamp.rules.rule_schulze import RuleSchulze
from svvamp.rules.rule_exhaustive_ballot import RuleExhaustiveBallot
from svvamp.rules.rule_irv import RuleIRV
from svvamp.rules.rule_condorcet_vtb_irv import RuleCondorcetVtbIRV
from svvamp.rules.rule_icrv import RuleICRV
from svvamp.utils.misc import indent


class VotingRuleTasks(list):
    """A set of tasks for the simulator, i.e. which voting rules with which options and which criteria about them.

    A "task" consists of: a voting rule, a dictionary of options, and a :class:`StudyRuleCriteria` object. A
    `VotingRuleTasks` is essentially a list of tasks.

    In the following, what we call a "voting system" consists of a voting rule and a dictionary of options (i.e.
    like a task, but with no :class:`StudyRuleCriteria` object specified).

    Parameters
    ----------
    voting_systems: List
        A list of pairs (rule_class, options), where `rule_class` is a subclass of :class:`Rule` and `options` is a
        dictionary of options. Instead of (rule_class, options), one may provide a singleton `rule_class`
        (without parentheses), which is equivalent to (rule_class, {}).
    study_rule_criteria: StudyRuleCriteria
        A :class:`StudyRuleCriteria` object.
    detailed_tasks: object
        Either a list of triples (rule_class, options, study_rule_criteria), or a VotingRuleTasks object.

    Examples
    --------
    Basically, there are two non-exclusive ways to define a VotingRuleTasks object:

        * Exhaustively with detailed_tasks.
        * As a Cartesian product by voting_systems and study_rule_criteria.

        >>> voting_rule_tasks1 = VotingRuleTasks(
        ...     voting_systems=[
        ...         RulePlurality,
        ...         RuleExhaustiveBallot,
        ...         (RuleIRV, {'cm_option': 'fast'}),
        ...         (RuleApproval, {'approval_threshold': 10}),
        ...     ],
        ...     study_rule_criteria=StudyRuleCriteria(
        ...         manipulation_criteria=['is_cm_', 'is_tm_', 'is_um_'],
        ...     ),
        ...     detailed_tasks=[
        ...         (
        ...             RuleIRV,
        ...             {'cm_option': 'exact'},
        ...             StudyRuleCriteria(
        ...                 manipulation_criteria=['is_cm_'],
        ...                 manipulation_only=True
        ...             )
        ...         ),
        ...         (
        ...             RuleApproval,
        ...             {'approval_threshold': 5},
        ...             StudyRuleCriteria(
        ...                 manipulation_criteria=['is_cm_', 'is_tm_', 'is_um_']
        ...             )
        ...         ),
        ...     ]
        ... )

        >>> voting_rule_tasks2 = VotingRuleTasks(
        ...     voting_systems=[RuleIRV, RuleExhaustiveBallot],
        ...     study_rule_criteria=StudyRuleCriteria(),
        ...     detailed_tasks=VotingRuleTasks(
        ...         voting_systems=[
        ...             (RuleIRV, {'cm_option': 'exact'}),
        ...             (RuleExhaustiveBallot, {'cm_option': 'exact'})
        ...         ],
        ...         study_rule_criteria=StudyRuleCriteria(
        ...             manipulation_criteria=['is_cm_'],
        ...             manipulation_only=True
        ...         )
        ...     )
        ... )

    In examples 1 and 2 above, all elements of `voting_systems` will be
    assigned the same set of criteria, the argument `study_rule_criteria`. The
    argument `detailed_tasks` allows to give different criteria for voting
    systems needing a specific treatment.

        * When voting_systems is None (default), it is normally set to [].
          But if detailed_tasks is empty, or study_rule_criteria is not empty,
          then voting_systems is set to a quite extensive list of
          voting systems.
        * When study_rule_criteria is None (default), it is normally set to [].
          But if detailed_tasks is empty, or voting_systems is not empty,
          then study_rule_criteria is set to StudyRuleCriteria(), which leads to study
          an quite extensive list of criteria.

    Study :class:`RuleSchulze` and :class:`Ranked Pairs` with a quite extensive list of criteria:

        >>> voting_rule_tasks = VotingRuleTasks(
        ...     voting_systems=[RuleSchulze, RuleRankedPairs]
        ... )

    Study a quite extensive list of voting systems, with CM being the only manipulation criterion:

        >>> voting_rule_tasks = VotingRuleTasks(
        ...     study_rule_criteria=StudyRuleCriteria(manipulation_criteria=['is_cm_'])
        ... )

    Study a quite extensive list of voting systems with a quite extensive list of criteria:

        >>> voting_rule_tasks = VotingRuleTasks()
    """

    def __init__(self, voting_systems=None, study_rule_criteria=None, detailed_tasks=None):
        # Dealing with default arguments...
        if detailed_tasks is None:
            detailed_tasks = []
        if voting_systems is None:
            if not detailed_tasks or study_rule_criteria is not None:
                voting_systems = ALL_RULE_CLASSES
            else:
                voting_systems = []
        if study_rule_criteria is None:
            study_rule_criteria = StudyRuleCriteria()
        # Here we go
        super().__init__([])
        for voting_system in voting_systems:
            if inspect.isclass(voting_system):
                self.append(
                    (voting_system, {}, copy.deepcopy(study_rule_criteria))
                )
            else:
                self.append(
                    (voting_system[0], voting_system[1], copy.deepcopy(study_rule_criteria))
                )
        self.extend(detailed_tasks)

    def __str__(self):
        """
            >>> voting_rule_tasks = VotingRuleTasks(detailed_tasks=[
            ...     (
            ...         RuleIRV,
            ...         {'cm_option': 'exact', 'um_option': 'exact'},
            ...         StudyRuleCriteria(manipulation_criteria=['is_cm_'], manipulation_only=True,
            ...                           numerical_criteria=[])
            ...     ),
            ...     (
            ...         RuleCondorcetVtbIRV,
            ...         {'cm_option': 'very_slow', 'um_option': 'exact'},
            ...         StudyRuleCriteria(manipulation_criteria=['is_cm_'], manipulation_only=True,
            ...                           numerical_criteria=[])
            ...     )
            ... ])
            >>> print(voting_rule_tasks)
            VotingRuleTasks with:
                voting_system: RuleIRV
                options: {'cm_option': 'exact', 'um_option': 'exact'}
                StudyRuleCriteria with:
                    manipulation_criteria:
                        is_cm_
                    manipulation_criteria_c: None
                    result_criteria: None
                    utility_criteria: None
                    numerical_criteria: None
                voting_system: RuleCondorcetVtbIRV
                options: {'cm_option': 'very_slow', 'um_option': 'exact'}
                StudyRuleCriteria with:
                    manipulation_criteria:
                        is_cm_
                    manipulation_criteria_c: None
                    result_criteria: None
                    utility_criteria: None
                    numerical_criteria: None
        """
        lines = []
        for (rule_class, options, study_rule_criteria) in self:
            lines.append(f'voting_system: {rule_class.__name__}')
            lines.append(f'options: {options}')
            lines.append(str(study_rule_criteria))
        return 'VotingRuleTasks with:\n' + indent('\n'.join(lines))

    def remove_rule(self, rule_class):
        """Remove a rule

        Removes all occurrences of `rule_class` from this `VotingRuleTasks` object.

        Parameters
        ----------
        rule_class : class
            Subclass of :class:`Rule`.

        Examples
        --------
            >>> voting_rule_tasks = VotingRuleTasks(detailed_tasks=[
            ...     (
            ...         RuleIRV,
            ...         {'cm_option': 'fast', 'um_option': 'fast'},
            ...         StudyRuleCriteria(manipulation_criteria=['is_cm_'], manipulation_only=True,
            ...                           numerical_criteria=[])
            ...     ),
            ...     (
            ...         RuleIRV,
            ...         {'cm_option': 'exact', 'um_option': 'exact'},
            ...         StudyRuleCriteria(manipulation_criteria=['is_cm_'], manipulation_only=True,
            ...                           numerical_criteria=[])
            ...     ),
            ...     (
            ...         RuleCondorcetVtbIRV,
            ...         {'cm_option': 'very_slow', 'um_option': 'exact'},
            ...         StudyRuleCriteria(manipulation_criteria=['is_cm_'], manipulation_only=True,
            ...                           numerical_criteria=[])
            ...     )
            ... ])
            >>> voting_rule_tasks.remove_rule(RuleIRV)
            >>> print(voting_rule_tasks)
            VotingRuleTasks with:
                voting_system: RuleCondorcetVtbIRV
                options: {'cm_option': 'very_slow', 'um_option': 'exact'}
                StudyRuleCriteria with:
                    manipulation_criteria:
                        is_cm_
                    manipulation_criteria_c: None
                    result_criteria: None
                    utility_criteria: None
                    numerical_criteria: None
        """
        self[:] = [element for element in self if element[0] != rule_class]

    def append_task(self, rule_class, options=None, study_rule_criteria=None):
        """Append a task to the list.

        Parameters
        ----------
        rule_class : class
            Subclass of :class:`Rule`.
        options : dict
            A dictionary of options. If None (default), then options will be an empty dictionary, which amounts to
            using the default options for this voting rule.
        study_rule_criteria : StudyRuleCriteria
            If None (default), then use the default StudyRuleCriteria() (with a quite extensive list of criteria).

        Examples
        --------
            >>> voting_rule_tasks = VotingRuleTasks(detailed_tasks=[
            ...     (
            ...         RuleIRV,
            ...         {'cm_option': 'fast'},
            ...         StudyRuleCriteria(manipulation_criteria=['is_cm_'], manipulation_only=True,
            ...                           numerical_criteria=[])
            ...     ),
            ... ])
            >>> voting_rule_tasks.append_task(
            ...     RuleIRV,
            ...     {'cm_option': 'exact'},
            ...     StudyRuleCriteria(manipulation_criteria=['is_cm_'], manipulation_only=True,
            ...                       numerical_criteria=[])
            ... )
            >>> print(voting_rule_tasks)
            VotingRuleTasks with:
                voting_system: RuleIRV
                options: {'cm_option': 'fast'}
                StudyRuleCriteria with:
                    manipulation_criteria:
                        is_cm_
                    manipulation_criteria_c: None
                    result_criteria: None
                    utility_criteria: None
                    numerical_criteria: None
                voting_system: RuleIRV
                options: {'cm_option': 'exact'}
                StudyRuleCriteria with:
                    manipulation_criteria:
                        is_cm_
                    manipulation_criteria_c: None
                    result_criteria: None
                    utility_criteria: None
                    numerical_criteria: None
        """
        if options is None:
            options = {}
        if study_rule_criteria is None:
            study_rule_criteria = StudyRuleCriteria()
        self.append((rule_class, options, copy.deepcopy(study_rule_criteria)))

    def extend(self, iterable):
        """Extend a `VotingRuleTasks` object.

        Note that the usual command `extend` can be used to concatenate two `VotingRuleTasks` objects.

        Examples
        --------
            >>> voting_rule_tasks = VotingRuleTasks(
            ...     voting_systems=[RuleExhaustiveBallot, RuleIRV],
            ...     study_rule_criteria=StudyRuleCriteria(manipulation_criteria=['is_cm_'], manipulation_only=True,
            ...                                           numerical_criteria=[])
            ... )
            >>> voting_rule_tasks_2 = VotingRuleTasks(
            ...     voting_systems=[RuleICRV, RuleCondorcetVtbIRV],
            ...     study_rule_criteria=StudyRuleCriteria(manipulation_criteria=['is_um_'], manipulation_only=True,
            ...                                           numerical_criteria=[])
            ... )
            >>> voting_rule_tasks.extend(voting_rule_tasks_2)
            >>> print(voting_rule_tasks)
            VotingRuleTasks with:
                voting_system: RuleExhaustiveBallot
                options: {}
                StudyRuleCriteria with:
                    manipulation_criteria:
                        is_cm_
                    manipulation_criteria_c: None
                    result_criteria: None
                    utility_criteria: None
                    numerical_criteria: None
                voting_system: RuleIRV
                options: {}
                StudyRuleCriteria with:
                    manipulation_criteria:
                        is_cm_
                    manipulation_criteria_c: None
                    result_criteria: None
                    utility_criteria: None
                    numerical_criteria: None
                voting_system: RuleICRV
                options: {}
                StudyRuleCriteria with:
                    manipulation_criteria:
                        is_um_
                    manipulation_criteria_c: None
                    result_criteria: None
                    utility_criteria: None
                    numerical_criteria: None
                voting_system: RuleCondorcetVtbIRV
                options: {}
                StudyRuleCriteria with:
                    manipulation_criteria:
                        is_um_
                    manipulation_criteria_c: None
                    result_criteria: None
                    utility_criteria: None
                    numerical_criteria: None
        """
        super().extend(iterable)

    def remove_criterion(self, criterion, rule_class=None):
        """Remove a criterion for one or all voting rules.

        Parameters
        ----------
        criterion : str
        rule_class : class
            Subclass of :class:`Rule`.

        Examples
        --------
        Removes all occurrences of `criterion` for all occurrences of `rule_class`:

            >>> voting_rule_tasks = VotingRuleTasks(
            ...     voting_systems=[RuleExhaustiveBallot, RuleIRV, (RuleIRV, {'cm_option': 'exact'})],
            ...     study_rule_criteria=StudyRuleCriteria(
            ...         manipulation_criteria=['is_cm_', 'is_um_'],
            ...         manipulation_only=True,
            ...         numerical_criteria=[]
            ...     )
            ... )
            >>> voting_rule_tasks.remove_criterion(criterion='is_um_', rule_class=RuleIRV)
            >>> print(voting_rule_tasks)
            VotingRuleTasks with:
                voting_system: RuleExhaustiveBallot
                options: {}
                StudyRuleCriteria with:
                    manipulation_criteria:
                        is_cm_
                        is_um_
                    manipulation_criteria_c: None
                    result_criteria: None
                    utility_criteria: None
                    numerical_criteria: None
                voting_system: RuleIRV
                options: {}
                StudyRuleCriteria with:
                    manipulation_criteria:
                        is_cm_
                    manipulation_criteria_c: None
                    result_criteria: None
                    utility_criteria: None
                    numerical_criteria: None
                voting_system: RuleIRV
                options: {'cm_option': 'exact'}
                StudyRuleCriteria with:
                    manipulation_criteria:
                        is_cm_
                    manipulation_criteria_c: None
                    result_criteria: None
                    utility_criteria: None
                    numerical_criteria: None

        If `rule_class` is None (default), then `criterion` is removed for all voting rules:

            >>> voting_rule_tasks = VotingRuleTasks(
            ...     voting_systems=[RuleExhaustiveBallot, RuleIRV],
            ...     study_rule_criteria=StudyRuleCriteria(
            ...         manipulation_criteria=['is_cm_', 'is_um_'],
            ...         manipulation_only=True,
            ...         numerical_criteria=[]
            ...     )
            ... )
            >>> voting_rule_tasks.remove_criterion('is_um_')
            >>> print(voting_rule_tasks)
            VotingRuleTasks with:
                voting_system: RuleExhaustiveBallot
                options: {}
                StudyRuleCriteria with:
                    manipulation_criteria:
                        is_cm_
                    manipulation_criteria_c: None
                    result_criteria: None
                    utility_criteria: None
                    numerical_criteria: None
                voting_system: RuleIRV
                options: {}
                StudyRuleCriteria with:
                    manipulation_criteria:
                        is_cm_
                    manipulation_criteria_c: None
                    result_criteria: None
                    utility_criteria: None
                    numerical_criteria: None
        """
        for (rc, options, study_rule_criteria) in self:
            if rule_class is None or rule_class == rc:
                study_rule_criteria.remove(criterion)

    def append_manipulation_criterion(self, criterion, rule_class=None):
        """Add a manipulation criterion for one or all voting rules.

        Parameters
        ----------
        criterion : str
        rule_class : class
            Subclass of :class:`Rule`.

        Examples
        --------
        Add `criterion` for all occurrences of `rule_class`:

            >>> voting_rule_tasks = VotingRuleTasks(
            ...     voting_systems=[RuleExhaustiveBallot, RuleIRV, (RuleIRV, {'cm_option': 'exact'})],
            ...     study_rule_criteria=StudyRuleCriteria(manipulation_criteria=['is_cm_'], manipulation_only=True,
            ...                                           numerical_criteria=[])
            ... )
            >>> voting_rule_tasks.append_manipulation_criterion(criterion='is_um_', rule_class=RuleIRV)
            >>> print(voting_rule_tasks)
            VotingRuleTasks with:
                voting_system: RuleExhaustiveBallot
                options: {}
                StudyRuleCriteria with:
                    manipulation_criteria:
                        is_cm_
                    manipulation_criteria_c: None
                    result_criteria: None
                    utility_criteria: None
                    numerical_criteria: None
                voting_system: RuleIRV
                options: {}
                StudyRuleCriteria with:
                    manipulation_criteria:
                        is_cm_
                        is_um_
                    manipulation_criteria_c: None
                    result_criteria: None
                    utility_criteria: None
                    numerical_criteria: None
                voting_system: RuleIRV
                options: {'cm_option': 'exact'}
                StudyRuleCriteria with:
                    manipulation_criteria:
                        is_cm_
                        is_um_
                    manipulation_criteria_c: None
                    result_criteria: None
                    utility_criteria: None
                    numerical_criteria: None

        If `rule_class` is None (default), then add `criterion` for all voting rules:

            >>> voting_rule_tasks = VotingRuleTasks(
            ...     voting_systems=[RuleExhaustiveBallot, RuleIRV],
            ...     study_rule_criteria=StudyRuleCriteria(manipulation_criteria=['is_cm_'], manipulation_only=True,
            ...                                           numerical_criteria=[])
            ... )
            >>> voting_rule_tasks.append_manipulation_criterion('is_um_')
            >>> print(voting_rule_tasks)
            VotingRuleTasks with:
                voting_system: RuleExhaustiveBallot
                options: {}
                StudyRuleCriteria with:
                    manipulation_criteria:
                        is_cm_
                        is_um_
                    manipulation_criteria_c: None
                    result_criteria: None
                    utility_criteria: None
                    numerical_criteria: None
                voting_system: RuleIRV
                options: {}
                StudyRuleCriteria with:
                    manipulation_criteria:
                        is_cm_
                        is_um_
                    manipulation_criteria_c: None
                    result_criteria: None
                    utility_criteria: None
                    numerical_criteria: None
        """
        for (rc, options, study_rule_criteria) in self:
            if rule_class is None or rule_class == rc:
                study_rule_criteria.append_manipulation_criterion(criterion)

    def append_manipulation_criterion_c(self, criterion, rule_class=None):
        """Add a manipulation criterion (candidates) for one or all voting rules.

        Parameters
        ----------
        criterion : str
        rule_class : class
            Subclass of :class:`Rule`.

        Examples
        --------
        Add `criterion` for all occurrences of `rule_class`:

            >>> voting_rule_tasks = VotingRuleTasks(
            ...     voting_systems=[RuleExhaustiveBallot, RuleIRV, (RuleIRV, {'cm_option': 'exact'})],
            ...     study_rule_criteria=StudyRuleCriteria(manipulation_criteria=['is_cm_'], manipulation_only=True,
            ...                                           numerical_criteria=[])
            ... )
            >>> voting_rule_tasks.append_manipulation_criterion_c(criterion='candidates_cm_', rule_class=RuleIRV)
            >>> print(voting_rule_tasks)
            VotingRuleTasks with:
                voting_system: RuleExhaustiveBallot
                options: {}
                StudyRuleCriteria with:
                    manipulation_criteria:
                        is_cm_
                    manipulation_criteria_c: None
                    result_criteria: None
                    utility_criteria: None
                    numerical_criteria: None
                voting_system: RuleIRV
                options: {}
                StudyRuleCriteria with:
                    manipulation_criteria:
                        is_cm_
                    manipulation_criteria_c:
                        candidates_cm_
                    result_criteria: None
                    utility_criteria: None
                    numerical_criteria: None
                voting_system: RuleIRV
                options: {'cm_option': 'exact'}
                StudyRuleCriteria with:
                    manipulation_criteria:
                        is_cm_
                    manipulation_criteria_c:
                        candidates_cm_
                    result_criteria: None
                    utility_criteria: None
                    numerical_criteria: None

        If `rule_class` is None (default), then add `criterion` for all voting rules:

            >>> voting_rule_tasks = VotingRuleTasks(
            ...     voting_systems=[RuleExhaustiveBallot, RuleIRV],
            ...     study_rule_criteria=StudyRuleCriteria(manipulation_criteria=['is_cm_'], manipulation_only=True,
            ...                                           numerical_criteria=[])
            ... )
            >>> voting_rule_tasks.append_manipulation_criterion_c('candidates_cm_')
            >>> print(voting_rule_tasks)
            VotingRuleTasks with:
                voting_system: RuleExhaustiveBallot
                options: {}
                StudyRuleCriteria with:
                    manipulation_criteria:
                        is_cm_
                    manipulation_criteria_c:
                        candidates_cm_
                    result_criteria: None
                    utility_criteria: None
                    numerical_criteria: None
                voting_system: RuleIRV
                options: {}
                StudyRuleCriteria with:
                    manipulation_criteria:
                        is_cm_
                    manipulation_criteria_c:
                        candidates_cm_
                    result_criteria: None
                    utility_criteria: None
                    numerical_criteria: None
        """
        for (rc, options, study_rule_criteria) in self:
            if rule_class is None or rule_class == rc:
                study_rule_criteria.append_manipulation_criterion_c(criterion)

    def append_result_criterion(self, criterion, rule_class=None):
        """Add a result criterion for one or all voting rules

        Parameters
        ----------
        criterion : str
        rule_class : class
            Subclass of :class:`Rule`.

        Examples
        --------
        Add `criterion` for all occurrences of `rule_class`:

            >>> voting_rule_tasks = VotingRuleTasks(
            ...     voting_systems=[RuleExhaustiveBallot, RuleIRV, (RuleIRV, {'cm_option': 'exact'})],
            ...     study_rule_criteria=StudyRuleCriteria(manipulation_criteria=['is_cm_'], manipulation_only=True,
            ...                                           numerical_criteria=[])
            ... )
            >>> voting_rule_tasks.append_result_criterion(criterion='w_is_condorcet_admissible_', rule_class=RuleIRV)
            >>> print(voting_rule_tasks)
            VotingRuleTasks with:
                voting_system: RuleExhaustiveBallot
                options: {}
                StudyRuleCriteria with:
                    manipulation_criteria:
                        is_cm_
                    manipulation_criteria_c: None
                    result_criteria: None
                    utility_criteria: None
                    numerical_criteria: None
                voting_system: RuleIRV
                options: {}
                StudyRuleCriteria with:
                    manipulation_criteria:
                        is_cm_
                    manipulation_criteria_c: None
                    result_criteria:
                        w_is_condorcet_admissible_
                    utility_criteria: None
                    numerical_criteria: None
                voting_system: RuleIRV
                options: {'cm_option': 'exact'}
                StudyRuleCriteria with:
                    manipulation_criteria:
                        is_cm_
                    manipulation_criteria_c: None
                    result_criteria:
                        w_is_condorcet_admissible_
                    utility_criteria: None
                    numerical_criteria: None

        If `rule_class` is None (default), then add `criterion` for all voting rules:

            >>> voting_rule_tasks = VotingRuleTasks(
            ...     voting_systems=[RuleExhaustiveBallot, RuleIRV],
            ...     study_rule_criteria=StudyRuleCriteria(manipulation_criteria=['is_cm_'], manipulation_only=True,
            ...                                           numerical_criteria=[])
            ... )
            >>> voting_rule_tasks.append_result_criterion(criterion='w_is_condorcet_admissible_')
            >>> print(voting_rule_tasks)
            VotingRuleTasks with:
                voting_system: RuleExhaustiveBallot
                options: {}
                StudyRuleCriteria with:
                    manipulation_criteria:
                        is_cm_
                    manipulation_criteria_c: None
                    result_criteria:
                        w_is_condorcet_admissible_
                    utility_criteria: None
                    numerical_criteria: None
                voting_system: RuleIRV
                options: {}
                StudyRuleCriteria with:
                    manipulation_criteria:
                        is_cm_
                    manipulation_criteria_c: None
                    result_criteria:
                        w_is_condorcet_admissible_
                    utility_criteria: None
                    numerical_criteria: None
        """
        for (rc, options, study_rule_criteria) in self:
            if rule_class is None or rule_class == rc:
                study_rule_criteria.append_result_criterion(criterion)

    def append_utility_criterion(self, criterion, func, name, rule_class=None):
        """Add a utility criterion for one or all voting rules

        Parameters
        ----------
        criterion : str
            The numerical data to be processed.
        func : callable
            An aggregating function.
        name : str
            The name chosen for this pair criterion / aggregation function. Cf.
            :meth:`StudyRuleCriteria.append_utility_criterion`.
        rule_class : class
            Subclass of :class:`Rule`.

        Examples
        --------
        Add `criterion` for all occurrences of `rule_class`:

            >>> voting_rule_tasks = VotingRuleTasks(
            ...     voting_systems=[RuleExhaustiveBallot, RuleIRV, (RuleIRV, {'cm_option': 'exact'})],
            ...     study_rule_criteria=StudyRuleCriteria(manipulation_criteria=['is_cm_'], manipulation_only=True,
            ...                                           numerical_criteria=[])
            ... )
            >>> voting_rule_tasks.append_utility_criterion(
            ...     criterion='total_utility_w_', func=np.sum, name='total_u_sum', rule_class=RuleIRV)
            >>> print(voting_rule_tasks)  # doctest: +ELLIPSIS
            VotingRuleTasks with:
                voting_system: RuleExhaustiveBallot
                options: {}
                StudyRuleCriteria with:
                    manipulation_criteria:
                        is_cm_
                    manipulation_criteria_c: None
                    result_criteria: None
                    utility_criteria: None
                    numerical_criteria: None
                voting_system: RuleIRV
                options: {}
                StudyRuleCriteria with:
                    manipulation_criteria:
                        is_cm_
                    manipulation_criteria_c: None
                    result_criteria: None
                    utility_criteria:
                        ('total_utility_w_', <function sum at ...>, 'total_u_sum')
                    numerical_criteria: None
                voting_system: RuleIRV
                options: {'cm_option': 'exact'}
                StudyRuleCriteria with:
                    manipulation_criteria:
                        is_cm_
                    manipulation_criteria_c: None
                    result_criteria: None
                    utility_criteria:
                        ('total_utility_w_', <function sum at ...>, 'total_u_sum')
                    numerical_criteria: None

        If `rule_class` is None (default), then add `criterion` for all voting rules:

            >>> voting_rule_tasks = VotingRuleTasks(
            ...     voting_systems=[RuleExhaustiveBallot, RuleIRV],
            ...     study_rule_criteria=StudyRuleCriteria(manipulation_criteria=['is_cm_'], manipulation_only=True,
            ...                                           numerical_criteria=[])
            ... )
            >>> voting_rule_tasks.append_utility_criterion(
            ...     criterion='total_utility_w_', func=np.sum, name='total_u_sum')
            >>> print(voting_rule_tasks)  # doctest: +ELLIPSIS
            VotingRuleTasks with:
                voting_system: RuleExhaustiveBallot
                options: {}
                StudyRuleCriteria with:
                    manipulation_criteria:
                        is_cm_
                    manipulation_criteria_c: None
                    result_criteria: None
                    utility_criteria:
                        ('total_utility_w_', <function sum at ...>, 'total_u_sum')
                    numerical_criteria: None
                voting_system: RuleIRV
                options: {}
                StudyRuleCriteria with:
                    manipulation_criteria:
                        is_cm_
                    manipulation_criteria_c: None
                    result_criteria: None
                    utility_criteria:
                        ('total_utility_w_', <function sum at ...>, 'total_u_sum')
                    numerical_criteria: None
        """
        for (rc, options, study_rule_criteria) in self:
            if rule_class is None or rule_class == rc:
                study_rule_criteria.append_utility_criterion(criterion, func, name)

    def append_numerical_criterion(self, criterion, rule_class=None):
        """Add a numerical criterion for one or all voting rules

        Parameters
        ----------
        criterion : str
        rule_class : class
            Subclass of :class:`Rule`.

        Examples
        --------
        Add `criterion` for all occurrences of `rule_class`:

            >>> voting_rule_tasks = VotingRuleTasks(
            ...     voting_systems=[RuleExhaustiveBallot, RuleIRV, (RuleIRV, {'cm_option': 'exact'})],
            ...     study_rule_criteria=StudyRuleCriteria(manipulation_criteria=['is_cm_'], manipulation_only=True,
            ...                                           numerical_criteria=[])
            ... )
            >>> voting_rule_tasks.append_numerical_criterion(criterion='nb_candidates_cm_', rule_class=RuleIRV)
            >>> print(voting_rule_tasks)
            VotingRuleTasks with:
                voting_system: RuleExhaustiveBallot
                options: {}
                StudyRuleCriteria with:
                    manipulation_criteria:
                        is_cm_
                    manipulation_criteria_c: None
                    result_criteria: None
                    utility_criteria: None
                    numerical_criteria: None
                voting_system: RuleIRV
                options: {}
                StudyRuleCriteria with:
                    manipulation_criteria:
                        is_cm_
                    manipulation_criteria_c: None
                    result_criteria: None
                    utility_criteria: None
                    numerical_criteria:
                        nb_candidates_cm_
                voting_system: RuleIRV
                options: {'cm_option': 'exact'}
                StudyRuleCriteria with:
                    manipulation_criteria:
                        is_cm_
                    manipulation_criteria_c: None
                    result_criteria: None
                    utility_criteria: None
                    numerical_criteria:
                        nb_candidates_cm_

        If `rule_class` is None (default), then add `criterion` for all voting rules:

            >>> voting_rule_tasks = VotingRuleTasks(
            ...     voting_systems=[RuleExhaustiveBallot, RuleIRV],
            ...     study_rule_criteria=StudyRuleCriteria(manipulation_criteria=['is_cm_'], manipulation_only=True,
            ...                                           numerical_criteria=[])
            ... )
            >>> voting_rule_tasks.append_numerical_criterion(criterion='nb_candidates_cm_')
            >>> print(voting_rule_tasks)
            VotingRuleTasks with:
                voting_system: RuleExhaustiveBallot
                options: {}
                StudyRuleCriteria with:
                    manipulation_criteria:
                        is_cm_
                    manipulation_criteria_c: None
                    result_criteria: None
                    utility_criteria: None
                    numerical_criteria:
                        nb_candidates_cm_
                voting_system: RuleIRV
                options: {}
                StudyRuleCriteria with:
                    manipulation_criteria:
                        is_cm_
                    manipulation_criteria_c: None
                    result_criteria: None
                    utility_criteria: None
                    numerical_criteria:
                        nb_candidates_cm_
        """
        for (rc, options, study_rule_criteria) in self:
            if rule_class is None or rule_class == rc:
                study_rule_criteria.append_numerical_criterion(criterion)

    def remove_option(self, option, rule_class=None):
        """Remove an option and its value for one or all voting rules

        Parameters
        ----------
        option : str
        rule_class : class
            Subclass of :class:`Rule`.

        Examples
        --------
        Remove `option` for all occurrences of `rule_class`:

            >>> voting_rule_tasks = VotingRuleTasks(
            ...     voting_systems=[
            ...         (RuleExhaustiveBallot, {'cm_option': 'exact'}),
            ...         (RuleIRV, {'cm_option': 'exact'})
            ...     ],
            ...     study_rule_criteria=StudyRuleCriteria(manipulation_criteria=['is_cm_'], manipulation_only=True,
            ...                                           numerical_criteria=[])
            ... )
            >>> voting_rule_tasks.remove_option(option='cm_option', rule_class=RuleIRV)
            >>> print(voting_rule_tasks)
            VotingRuleTasks with:
                voting_system: RuleExhaustiveBallot
                options: {'cm_option': 'exact'}
                StudyRuleCriteria with:
                    manipulation_criteria:
                        is_cm_
                    manipulation_criteria_c: None
                    result_criteria: None
                    utility_criteria: None
                    numerical_criteria: None
                voting_system: RuleIRV
                options: {}
                StudyRuleCriteria with:
                    manipulation_criteria:
                        is_cm_
                    manipulation_criteria_c: None
                    result_criteria: None
                    utility_criteria: None
                    numerical_criteria: None

        This means that from now on, `RuleIRV` will use its default value for `cm_option`.

        If `rule_class` is None (default), then remove the option for all voting rules:

            >>> voting_rule_tasks = VotingRuleTasks(
            ...     voting_systems=[
            ...         (RuleExhaustiveBallot, {'cm_option': 'exact'}),
            ...         (RuleIRV, {'cm_option': 'exact'})
            ...     ],
            ...     study_rule_criteria=StudyRuleCriteria(manipulation_criteria=['is_cm_'], manipulation_only=True,
            ...                                           numerical_criteria=[])
            ... )
            >>> voting_rule_tasks.remove_option(option='cm_option')
            >>> print(voting_rule_tasks)
            VotingRuleTasks with:
                voting_system: RuleExhaustiveBallot
                options: {}
                StudyRuleCriteria with:
                    manipulation_criteria:
                        is_cm_
                    manipulation_criteria_c: None
                    result_criteria: None
                    utility_criteria: None
                    numerical_criteria: None
                voting_system: RuleIRV
                options: {}
                StudyRuleCriteria with:
                    manipulation_criteria:
                        is_cm_
                    manipulation_criteria_c: None
                    result_criteria: None
                    utility_criteria: None
                    numerical_criteria: None
        """
        for (rc, options, study_rule_criteria) in self:
            if rule_class is None or rule_class == rc:
                if option in options:
                    del(options[option])

    def set_option(self, option, value, rule_class=None):
        """Set an option for one or all voting rules

        Parameters
        ----------
        option : str
        value : object
            New value for the option.
        rule_class : class
            Subclass of :class:`Rule`.

        Examples
        --------
        Set `option` to `value` for all occurrences of `rule_class`:

            >>> voting_rule_tasks = VotingRuleTasks(
            ...     voting_systems=[RuleExhaustiveBallot, RuleIRV],
            ...     study_rule_criteria=StudyRuleCriteria(manipulation_criteria=['is_cm_'], manipulation_only=True,
            ...                                           numerical_criteria=[])
            ... )
            >>> voting_rule_tasks.set_option('cm_option', 'exact', rule_class=RuleIRV)
            >>> print(voting_rule_tasks)
            VotingRuleTasks with:
                voting_system: RuleExhaustiveBallot
                options: {}
                StudyRuleCriteria with:
                    manipulation_criteria:
                        is_cm_
                    manipulation_criteria_c: None
                    result_criteria: None
                    utility_criteria: None
                    numerical_criteria: None
                voting_system: RuleIRV
                options: {'cm_option': 'exact'}
                StudyRuleCriteria with:
                    manipulation_criteria:
                        is_cm_
                    manipulation_criteria_c: None
                    result_criteria: None
                    utility_criteria: None
                    numerical_criteria: None

        If `rule_class` is None (default), do this for all voting rules:

            >>> voting_rule_tasks = VotingRuleTasks(
            ...     voting_systems=[RuleExhaustiveBallot, RuleIRV],
            ...     study_rule_criteria=StudyRuleCriteria(manipulation_criteria=['is_cm_'], manipulation_only=True,
            ...                                           numerical_criteria=[])
            ... )
            >>> voting_rule_tasks.set_option('cm_option', 'exact')
            >>> print(voting_rule_tasks)
            VotingRuleTasks with:
                voting_system: RuleExhaustiveBallot
                options: {'cm_option': 'exact'}
                StudyRuleCriteria with:
                    manipulation_criteria:
                        is_cm_
                    manipulation_criteria_c: None
                    result_criteria: None
                    utility_criteria: None
                    numerical_criteria: None
                voting_system: RuleIRV
                options: {'cm_option': 'exact'}
                StudyRuleCriteria with:
                    manipulation_criteria:
                        is_cm_
                    manipulation_criteria_c: None
                    result_criteria: None
                    utility_criteria: None
                    numerical_criteria: None
        """
        for (rc, options, study_rule_criteria) in self:
            if rule_class is None or rule_class == rc:
                # print(rule_class)
                # print('Change option to '+ str(value))
                options[option] = value

    def check_sanity(self):
        """Check sanity of the object

        Preform some basic checks. It is recommended to use this method before launching big simulations.

        Examples
        --------
        Detect an illegal option:

            >>> voting_rule_tasks = VotingRuleTasks(
            ...     voting_systems=[(RuleIRV, {'cm_option': 'unexpected_option'})],
            ... )
            >>> voting_rule_tasks.check_sanity()
            Traceback (most recent call last):
            ValueError: 'cm_option' = 'unexpected_option' is not allowed in RuleIRV.

        Detect an illegal :class:`StudyRuleCriteria`:

            >>> voting_rule_tasks = VotingRuleTasks(
            ...     voting_systems=[RuleIRV],
            ...     study_rule_criteria=StudyRuleCriteria(manipulation_criteria=['unexpected_criterion'])
            ... )
            >>> voting_rule_tasks.check_sanity()
            Traceback (most recent call last):
            ValueError: Criterion 'unexpected_criterion' is unknown for RuleIRV.
        """
        for (rule_class, options, study_rule_criteria) in self:
            # print(rule_class)
            for option, value in options.items():
                rule_class.check_option_allowed(option, value)
            study_rule_criteria.check_sanity(rule_class)
        print('VotingRuleTasks: Sanity check was successful.')
