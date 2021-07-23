# -*- coding: utf-8 -*-
"""
Created on nov. 03, 2014, 23:33
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

import time

import numpy as np
import pandas as pd
import svvamp

from svvamp.meta.study_profile_criteria import StudyProfileCriteria
from svvamp.meta.voting_rule_tasks import VotingRuleTasks
from svvamp.preferences.profile_from_file import ProfileFromFile
from svvamp.utils.pseudo_bool import equal_true


class ExperimentAnalyzer:

    def __init__(self, profile_iterator, study_profile_criteria=None, voting_rule_tasks=None,
                 file_suffix='', output_dir='out', log_csv=None, ping_period=10):
        self.profile_iterator = profile_iterator
        if study_profile_criteria is None:
            study_profile_criteria = StudyProfileCriteria()
        self.study_profile_criteria = study_profile_criteria
        if voting_rule_tasks is None:
            voting_rule_tasks = VotingRuleTasks()
        self.voting_rule_tasks = voting_rule_tasks
        if not file_suffix:
            self.file_suffix = ''
        else:
            self.file_suffix = '_' + file_suffix
        self.output_dir = output_dir
        if log_csv is None:
            log_csv = []
        self.log_csv = log_csv
        self.ping_period = ping_period

    def _prepare_csv(self):
        prefix = self.output_dir + '/Results_' + time.strftime('%Y-%m-%d_%H-%M-%S')
        self.file_m = open(f"{prefix}_m{self.file_suffix}.csv", 'a')
        self.file_m_fr = open(f"{prefix}_m_fr{self.file_suffix}.csv", 'a')
        self.file_u = open(f"{prefix}_u{self.file_suffix}.csv", 'a')
        self.file_u_fr = open(f"{prefix}_u_fr{self.file_suffix}.csv", 'a')
        self.file_d = open(f"{prefix}_d{self.file_suffix}.csv", 'a')
        self.file_d_fr = open(f"{prefix}_d_fr{self.file_suffix}.csv", 'a')
        headers_m = pd.DataFrame([
            'Voting system', 'Criterion', 'Candidate', 'Candidate name',
            'Algorithm parameters',
            'Computation time',
            'Rate (lower bound)',
            'Rate (upper bound)',
            'Number of profiles',
            'V', 'C', 'Culture type',
            'Culture parameter 1', 'Culture parameter 1 value',
            'Culture parameter 2', 'Culture parameter 2 value',
            'Culture parameter 3', 'Culture parameter 3 value',
            'Culture parameter 4', 'Culture parameter 4 value'
        ]).T
        headers_m.to_csv(path_or_buf=self.file_m, index=False, header=False, line_terminator='\n')
        headers_m.to_csv(path_or_buf=self.file_m_fr, index=False, header=False, sep=';', line_terminator='\n')
        headers_u = pd.DataFrame([
            'Voting system', 'Criterion',
            'Result',
            'Number of profiles',
            'V', 'C', 'Culture type',
            'Culture parameter 1', 'Culture parameter 1 value',
            'Culture parameter 2', 'Culture parameter 2 value',
            'Culture parameter 3', 'Culture parameter 3 value',
            'Culture parameter 4', 'Culture parameter 4 value'
        ]).T
        headers_u.to_csv(path_or_buf=self.file_u, index=False, header=False, line_terminator='\n')
        headers_u.to_csv(path_or_buf=self.file_u_fr, index=False, header=False, sep=';', line_terminator='\n')

    def _df_to_csv_m(self, df):
        return _df_to_csv(df, file_en=self.file_m, file_fr=self.file_m_fr)

    def _df_to_csv_u(self, df):
        return _df_to_csv(df, file_en=self.file_u, file_fr=self.file_u_fr)

    def _df_to_csv_d(self, df):
        return _df_to_csv(df, file_en=self.file_d, file_fr=self.file_d_fr)

    def run(self):
        self.study_profile_criteria.check_sanity()
        self.voting_rule_tasks.check_sanity()
        self._prepare_csv()
        n_iterations = 0
        n_c, n_v, labels_candidates = None, None, None
        results_p_boolean_criteria = None
        results_p_numerical_criteria = None
        results_p_special_candidates_criteria = None
        results_p_array_criteria = None
        results_p_matrix_criteria = None
        results_vs_winner = None
        results_vs_result_criteria = None
        results_vs_manipulation_criteria = None
        results_vs_manipulation_criteria_c = None
        results_vs_utility_criteria = None
        # Heavy lifting
        for profile in self.profile_iterator:
            n_iterations += 1
            # Initialization (based on the first profile)
            if n_iterations == 1:
                n_c = profile.n_c
                n_v = profile.n_v
                labels_candidates = profile.labels_candidates
                # Initialize records for profile results
                results_p_boolean_criteria = [dict(inf=0, sup=0) for _ in self.study_profile_criteria.boolean_criteria]
                results_p_numerical_criteria = [[] for _ in self.study_profile_criteria.numerical_criteria]
                results_p_special_candidates_criteria = [
                    np.zeros(n_c + 1) for _ in self.study_profile_criteria.special_candidates_criteria
                ]  # The last coefficient will be used for cases np.nan
                results_p_array_criteria = [np.zeros(n_c) for _ in self.study_profile_criteria.array_criteria]
                results_p_matrix_criteria = [
                    np.zeros(getattr(profile, criterion).shape)
                    for criterion in self.study_profile_criteria.matrix_criteria
                ]
                # Initialize record for voting systems results
                results_vs_winner = [np.zeros(n_c) for task in self.voting_rule_tasks]
                results_vs_result_criteria = [
                    [
                        dict(inf=0, sup=0)
                        for _ in study_rule_criteria.result_criteria
                    ]
                    for (rule_class, options, study_rule_criteria) in self.voting_rule_tasks
                ]
                results_vs_manipulation_criteria = [
                    [
                        dict(inf=0, sup=0, log='', time=0)
                        for _ in study_rule_criteria.manipulation_criteria
                    ]
                    for (rule_class, options, study_rule_criteria) in self.voting_rule_tasks
                ]
                results_vs_manipulation_criteria_c = [
                    [
                        dict(inf=np.zeros(n_c), sup=np.zeros(n_c), log='', time=0)
                        for _ in study_rule_criteria.manipulation_criteria
                    ]
                    for (rule_class, options, study_rule_criteria) in self.voting_rule_tasks
                ]
                results_vs_utility_criteria = [
                    [
                        []
                        for _ in study_rule_criteria.utility_criteria
                    ]
                    for (rule_class, options, study_rule_criteria) in self.voting_rule_tasks
                ]
            # Tests on the profile itself
            for i_criterion, criterion in enumerate(self.study_profile_criteria.boolean_criteria):
                answer = getattr(profile, criterion)
                if equal_true(answer):
                    results_p_boolean_criteria[i_criterion]['inf'] += 1
                    results_p_boolean_criteria[i_criterion]['sup'] += 1
                elif np.isnan(answer):
                    results_p_boolean_criteria[i_criterion]['sup'] += 1
            for i_criterion, (criterion, func, name) in enumerate(self.study_profile_criteria.numerical_criteria):
                answer = getattr(profile, criterion)
                results_p_numerical_criteria[i_criterion].append(answer)
            for i_criterion, criterion in enumerate(self.study_profile_criteria.special_candidates_criteria):
                answer = getattr(profile, criterion)
                if np.isnan(answer):
                    results_p_special_candidates_criteria[i_criterion][-1] += 1
                else:
                    results_p_special_candidates_criteria[i_criterion][answer] += 1
            for i_criterion, criterion in enumerate(self.study_profile_criteria.array_criteria):
                answer = getattr(profile, criterion)
                results_p_array_criteria[i_criterion] += answer
            for i_criterion, criterion in enumerate(self.study_profile_criteria.matrix_criteria):
                answer = getattr(profile, criterion)
                results_p_matrix_criteria[i_criterion] += answer
            # Tests on the voting systems
            for i_task, (rule_class, options, study_rule_criteria) in enumerate(self.voting_rule_tasks):
                election = rule_class(**options)(profile)
                # The winner
                results_vs_winner[i_task][election.w_] += 1
                # Result criteria
                for i_criterion, criterion in enumerate(study_rule_criteria.result_criteria):
                    answer = getattr(election, criterion)
                    if equal_true(answer):
                        results_vs_result_criteria[i_task][i_criterion]['inf'] += 1
                        results_vs_result_criteria[i_task][i_criterion]['sup'] += 1
                    elif np.isnan(answer):
                        results_vs_result_criteria[i_task][i_criterion]['sup'] += 1
                # Manipulation criteria (Boolean)
                for i_criterion, criterion in enumerate(study_rule_criteria.manipulation_criteria):
                    t1 = time.time()
                    answer = getattr(election, criterion)
                    t2 = time.time()
                    results_vs_manipulation_criteria[i_task][i_criterion]['time'] += t2 - t1
                    if equal_true(answer):
                        results_vs_manipulation_criteria[i_task][i_criterion]['inf'] += 1
                        results_vs_manipulation_criteria[i_task][i_criterion]['sup'] += 1
                    elif np.isnan(answer):
                        results_vs_manipulation_criteria[i_task][i_criterion]['sup'] += 1
                    if n_iterations == 1:
                        log = election.log_(criterion)
                        results_vs_manipulation_criteria[i_task][i_criterion]['log'] += log
                # Manipulation criteria (by candidate)
                for i_criterion, criterion in enumerate(study_rule_criteria.manipulation_criteria_c):
                    t1 = time.time()
                    answer = getattr(election, criterion)
                    t2 = time.time()
                    results_vs_manipulation_criteria_c[i_task][i_criterion]['time'] += t2 - t1
                    results_vs_manipulation_criteria[i_task][i_criterion]['inf'] += np.equal(answer, True)
                    results_vs_manipulation_criteria[i_task][i_criterion]['sup'] += np.equal(answer, True)
                    results_vs_manipulation_criteria[i_task][i_criterion]['sup'] += np.isnan(answer)
                    if n_iterations == 1:
                        log = election.log_(criterion)
                        results_vs_manipulation_criteria_c[i_task][i_criterion]['log'] += log
                # Utility criteria
                for i_criterion, (criterion, func, name) in enumerate(study_rule_criteria.utility_criteria):
                    answer = getattr(election, criterion)
                    results_vs_utility_criteria[i_task][i_criterion].append(answer)
            # Report progression (or not)
            if n_iterations % self.ping_period == 0:
                print('n_iterations =', n_iterations)

        # Write results in csv
        # Results on profiles
        for i_criterion, criterion in enumerate(self.study_profile_criteria.boolean_criteria):
            lower = results_p_boolean_criteria[i_criterion]['inf'] / n_iterations
            upper = results_p_boolean_criteria[i_criterion]['sup'] / n_iterations
            self._df_to_csv_m(
                ['Profile', criterion, '', '', '', '', lower, upper, n_iterations, n_v, n_c] + self.log_csv)
        for i_criterion, (criterion, func, name) in enumerate(self.study_profile_criteria.numerical_criteria):
            answer = func(results_p_numerical_criteria[i_criterion])
            self._df_to_csv_u(['Profile', name, answer, n_iterations, n_v, n_c] + self.log_csv)
        for i_criterion, criterion in enumerate(self.study_profile_criteria.special_candidates_criteria):
            rate = results_p_special_candidates_criteria[i_criterion] / n_iterations
            for c in range(n_c):
                self._df_to_csv_m(
                    ['Profile', criterion, c, labels_candidates[c], '', '', rate[c], rate[c], n_iterations, n_v, n_c]
                    + self.log_csv
                )
            self._df_to_csv_m(
                ['Profile', criterion, 'None', 'None', '', '', rate[n_c], rate[n_c], n_iterations, n_v, n_c]
                + self.log_csv
            )
        for i_criterion, criterion in enumerate(self.study_profile_criteria.array_criteria):
            aver = results_p_array_criteria[i_criterion] / n_iterations
            for c in range(n_c):
                self._df_to_csv_m(
                    ['Profile', criterion, c, labels_candidates[c], '', '', aver[c], aver[c], n_iterations, n_v, n_c]
                    + self.log_csv
                )
        for i_criterion, criterion in enumerate(self.study_profile_criteria.matrix_criteria):
            self._df_to_csv_d([criterion])
            self._df_to_csv_d(results_p_matrix_criteria[i_criterion] / n_iterations)
        # Results on voting systems
        for i_task, (rule_class, options, study_rule_criteria) in enumerate(self.voting_rule_tasks):
            # Winner
            rate_win = results_vs_winner[i_task] / n_iterations
            for c in range(n_c):
                self._df_to_csv_m(
                    [
                        rule_class.__name__, 'Winner', c, labels_candidates[c], '', '',
                        rate_win[c], rate_win[c], n_iterations, n_v, n_c
                    ]
                    + self.log_csv
                )
            # Result criteria
            for i_criterion, criterion in enumerate(study_rule_criteria.result_criteria):
                lower = results_vs_result_criteria[i_task][i_criterion]['inf'] / n_iterations
                upper = results_vs_result_criteria[i_task][i_criterion]['sup'] / n_iterations
                self._df_to_csv_m(
                    [rule_class.__name__, criterion, '', '', '', '', lower, upper, n_iterations, n_v, n_c]
                    + self.log_csv
                )
            # Manipulation criteria
            for i_criterion, criterion in enumerate(study_rule_criteria.manipulation_criteria):
                lower = results_vs_manipulation_criteria[i_task][i_criterion]['inf'] / n_iterations
                upper = results_vs_manipulation_criteria[i_task][i_criterion]['sup'] / n_iterations
                the_time = results_vs_manipulation_criteria[i_task][i_criterion]['time']
                the_log = results_vs_manipulation_criteria[i_task][i_criterion]['log']
                self._df_to_csv_m(
                    [
                        rule_class.__name__, criterion, 'any', '', the_log, the_time, lower, upper,
                        n_iterations, n_v, n_c
                    ]
                    + self.log_csv
                )
            for i_criterion, criterion in enumerate(study_rule_criteria.manipulation_criteria_c):
                lower = results_vs_manipulation_criteria_c[i_task][i_criterion]['inf'] / n_iterations
                upper = results_vs_manipulation_criteria_c[i_task][i_criterion]['sup'] / n_iterations
                the_time = results_vs_manipulation_criteria_c[i_task][i_criterion]['time']
                the_log = results_vs_manipulation_criteria_c[i_task][i_criterion]['log']
                for c in range(n_c):
                    self._df_to_csv_m(
                        [
                            rule_class.__name__, criterion, c, labels_candidates[c], the_log, '', lower[c], upper[c],
                            n_iterations, n_v, n_c
                        ]
                        + self.log_csv
                    )
                # Indicate the time only once
                self._df_to_csv_m(
                    [rule_class.__name__, criterion, 'any', '', the_log, the_time, '', '', n_iterations, n_v, n_c]
                    + self.log_csv
                )
            # Utility criteria
            for i_criterion, (criterion, func, name) in enumerate(study_rule_criteria.utility_criteria):
                answer = func(results_vs_utility_criteria[i_task][i_criterion])
                self._df_to_csv_u([rule_class.__name__, name, answer, n_iterations, n_v, n_c] + self.log_csv)
        self.file_m.close()
        self.file_m_fr.close()
        self.file_u.close()
        self.file_u_fr.close()
        self.file_d.close()
        self.file_d_fr.close()
        print('Simulation finished')


def my_float(x):
    if isinstance(x, float):
        return ('%.6f' % x).replace('.', ',')
    else:
        return x


def _df_to_csv(df, file_en, file_fr):
    df = pd.DataFrame(df)
    if df.shape[1] == 1:
        df = df.T
    df.to_csv(path_or_buf=file_en, index=False, header=False, line_terminator='\n')
    df.apply(my_float).to_csv(path_or_buf=file_fr, index=False, header=False, sep=';', line_terminator='\n')
