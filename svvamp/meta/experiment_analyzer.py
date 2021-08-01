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

from svvamp.meta.study_profile_criteria import StudyProfileCriteria
from svvamp.meta.voting_rule_tasks import VotingRuleTasks
from svvamp.preferences.generator_profile_noise import GeneratorProfileNoise
from svvamp.utils.pseudo_bool import equal_true


class ExperimentAnalyzer:

    def __init__(self,
                 base_profile=None,
                 n_samples=1,
                 relative_noise=0.,
                 absolute_noise=0.,
                 study_profile_criteria=None,
                 voting_rule_tasks=None,
                 output_dir='out',
                 output_file_suffix='',
                 log_csv=None,
                 ping_period=10):
        # Check sanity (or not)
        if study_profile_criteria:
            study_profile_criteria.check_sanity()
        if voting_rule_tasks:
            voting_rule_tasks.check_sanity()
        # Default options
        if study_profile_criteria is None:
            study_profile_criteria = StudyProfileCriteria()
        if voting_rule_tasks is None:
            voting_rule_tasks = VotingRuleTasks()
        if log_csv is None:
            log_csv = []
        # Set the default attributes
        self.default_base_profile = base_profile
        self.default_n_samples = n_samples
        self.default_relative_noise = relative_noise
        self.default_absolute_noise = absolute_noise
        self.default_study_profile_criteria = study_profile_criteria
        self.default_voting_rule_tasks = voting_rule_tasks
        self.default_output_dir = output_dir
        self.default_output_file_suffix = output_file_suffix
        self.default_log_csv = log_csv
        self.default_ping_period = ping_period
        # Declare the attributes
        self.base_profile = None
        self.n_samples = None
        self.relative_noise = None
        self.absolute_noise = None
        self.study_profile_criteria = None
        self.voting_rule_tasks = None
        self.output_dir = None
        self.output_file_suffix = None
        self.log_csv = None
        self.ping_period = None

    def check_sanity(self):
        self.study_profile_criteria.check_sanity()
        self.voting_rule_tasks.check_sanity()

    def _prepare_csv(self):
        prefix = self.output_dir + '/Results_' + time.strftime('%Y-%m-%d_%H-%M-%S')
        suffix = '_' + self.output_file_suffix if self.output_file_suffix else ''
        self.file_d_en = open(f"{prefix}_d_en{suffix}.csv", 'a')
        self.file_d_fr = open(f"{prefix}_d_fr{suffix}.csv", 'a')
        self.file_m_en = open(f"{prefix}_m_en{suffix}.csv", 'a')
        self.file_m_fr = open(f"{prefix}_m_fr{suffix}.csv", 'a')
        headers_m = pd.DataFrame([
            'Rule', 'Rule (abbr)', 'Rule (class)',
            'Criterion', 'Candidate', 'Candidate name',
            'Algorithm parameters',
            'Computation time',
            'Rate (lower bound)',
            'Rate (upper bound)',
            'Rate (uncertainty)',
            'Number of profiles',
            'V', 'C', 'Culture type',
            'Culture parameter 1', 'Culture parameter 1 value',
            'Culture parameter 2', 'Culture parameter 2 value',
            'Culture parameter 3', 'Culture parameter 3 value',
            'Culture parameter 4', 'Culture parameter 4 value'
        ]).T
        headers_m.to_csv(path_or_buf=self.file_m_en, index=False, header=False, line_terminator='\n')
        headers_m.to_csv(path_or_buf=self.file_m_fr, index=False, header=False, sep=';', line_terminator='\n')

    def _df_to_csv_m(self, df):
        return _df_to_csv(df, file_en=self.file_m_en, file_fr=self.file_m_fr)

    def _df_to_csv_d(self, df):
        return _df_to_csv(df, file_en=self.file_d_en, file_fr=self.file_d_fr)

    def __call__(
        self,
        base_profile=None,
        n_samples=None,
        relative_noise=None,
        absolute_noise=None,
        study_profile_criteria=None,
        voting_rule_tasks=None,
        output_dir=None,
        output_file_suffix=None,
        log_csv=None,
        ping_period=None
    ):
        # Set the parameters
        self.base_profile = self.default_base_profile if base_profile is None else base_profile
        self.n_samples = self.default_n_samples if n_samples is None else n_samples
        self.relative_noise = self.default_relative_noise if relative_noise is None else relative_noise
        self.absolute_noise = self.default_absolute_noise if absolute_noise is None else absolute_noise
        self.study_profile_criteria = (self.default_study_profile_criteria if study_profile_criteria is None
                                       else study_profile_criteria)
        self.voting_rule_tasks = self.default_voting_rule_tasks if voting_rule_tasks is None else voting_rule_tasks
        self.output_dir = self.default_output_dir if output_dir is None else output_dir
        self.output_file_suffix = self.default_output_file_suffix if output_file_suffix is None else output_file_suffix
        self.log_csv = self.default_log_csv if log_csv is None else log_csv
        self.ping_period = self.default_ping_period if ping_period is None else ping_period
        # Check sanity (or not)
        if study_profile_criteria:
            study_profile_criteria.check_sanity()
        if voting_rule_tasks:
            voting_rule_tasks.check_sanity()
        # Initialize
        n_c = self.base_profile.n_c
        n_v = self.base_profile.n_v
        labels_candidates = self.base_profile.labels_candidates
        # - Initialize records for profile results
        results_p_boolean_criteria = [dict(inf=0, sup=0) for _ in self.study_profile_criteria.boolean_criteria]
        results_p_numerical_criteria = [[] for _ in self.study_profile_criteria.numerical_criteria]
        results_p_special_candidates_criteria = [
            np.zeros(n_c + 1) for _ in self.study_profile_criteria.special_candidates_criteria
        ]  # The last coefficient will be used for cases np.nan
        results_p_array_criteria = [np.zeros(n_c) for _ in self.study_profile_criteria.array_criteria]
        results_p_matrix_criteria = [
            np.zeros(getattr(self.base_profile, criterion).shape)
            for criterion in self.study_profile_criteria.matrix_criteria
        ]
        # - Initialize record for voting systems results
        results_vs_winner = [np.zeros(n_c) for _ in self.voting_rule_tasks]
        results_vs_result_criteria = [
            [
                dict(inf=0, sup=0)
                for _ in study_rule_criteria.result_criteria
            ]
            for (rule_class, options, study_rule_criteria) in self.voting_rule_tasks
        ]
        results_vs_numerical_criteria = [
            [
                dict(inf=0, sup=0, time=0)
                for _ in study_rule_criteria.numerical_criteria
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
                for _ in study_rule_criteria.manipulation_criteria_c
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
        self._prepare_csv()
        generator_profile = GeneratorProfileNoise(
            base_profile=self.base_profile, relative_noise=self.relative_noise, absolute_noise=self.absolute_noise)
        # Heavy lifting
        print('Compute...')
        for n_iterations in range(self.n_samples):
            profile = generator_profile()
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
                # Utility criteria
                for i_criterion, (criterion, func, name) in enumerate(study_rule_criteria.utility_criteria):
                    answer = getattr(election, criterion)
                    results_vs_utility_criteria[i_task][i_criterion].append(answer)
                # Numerical criteria
                for i_criterion, criterion in enumerate(study_rule_criteria.numerical_criteria):
                    t1 = time.time()
                    inf, sup = getattr(election, criterion)
                    t2 = time.time()
                    results_vs_numerical_criteria[i_task][i_criterion]['time'] += t2 - t1
                    results_vs_numerical_criteria[i_task][i_criterion]['inf'] += inf
                    results_vs_numerical_criteria[i_task][i_criterion]['sup'] += sup
                # Manipulation criteria (by candidate)
                for i_criterion, criterion in enumerate(study_rule_criteria.manipulation_criteria_c):
                    t1 = time.time()
                    answer = getattr(election, criterion)
                    t2 = time.time()
                    results_vs_manipulation_criteria_c[i_task][i_criterion]['time'] += t2 - t1
                    results_vs_manipulation_criteria_c[i_task][i_criterion]['inf'] += np.equal(answer, True)
                    results_vs_manipulation_criteria_c[i_task][i_criterion]['sup'] += np.equal(answer, True)
                    results_vs_manipulation_criteria_c[i_task][i_criterion]['sup'] += np.isnan(answer)
                    if n_iterations == 0:
                        log = election.log_(criterion)
                        results_vs_manipulation_criteria_c[i_task][i_criterion]['log'] += log
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
                    if n_iterations == 0:
                        log = election.log_(criterion)
                        results_vs_manipulation_criteria[i_task][i_criterion]['log'] += log
            # Report progression (or not)
            if (n_iterations + 1) % self.ping_period == 0:
                print(f'{n_iterations + 1} profiles analyzed')

        # Write results in csv
        print('Write results in csv files...')
        # Results on profiles
        for i_criterion, criterion in enumerate(self.study_profile_criteria.boolean_criteria):
            lower = results_p_boolean_criteria[i_criterion]['inf'] / self.n_samples
            upper = results_p_boolean_criteria[i_criterion]['sup'] / self.n_samples
            self._df_to_csv_m(
                ['Profile', '', '', criterion, '', '', '', '', lower, upper, upper - lower, self.n_samples, n_v, n_c]
                + self.log_csv)
        for i_criterion, (criterion, func, name) in enumerate(self.study_profile_criteria.numerical_criteria):
            answer = func(results_p_numerical_criteria[i_criterion])
            self._df_to_csv_m(['Profile', '', '', name, '', '', '', '', answer, answer, 0, self.n_samples, n_v, n_c]
                              + self.log_csv)
        for i_criterion, criterion in enumerate(self.study_profile_criteria.special_candidates_criteria):
            rate = results_p_special_candidates_criteria[i_criterion] / self.n_samples
            for c in range(n_c):
                self._df_to_csv_m(
                    ['Profile', '', '', criterion, c, labels_candidates[c], '', '', rate[c], rate[c], 0, self.n_samples,
                     n_v, n_c]
                    + self.log_csv
                )
            self._df_to_csv_m(
                ['Profile', '', '', criterion, 'None', 'None', '', '', rate[n_c], rate[n_c], 0, self.n_samples,
                 n_v, n_c]
                + self.log_csv
            )
        for i_criterion, criterion in enumerate(self.study_profile_criteria.array_criteria):
            aver = results_p_array_criteria[i_criterion] / self.n_samples
            for c in range(n_c):
                self._df_to_csv_m(
                    ['Profile', '', '', criterion, c, labels_candidates[c], '', '', aver[c], aver[c], 0, self.n_samples,
                     n_v, n_c]
                    + self.log_csv
                )
        for i_criterion, criterion in enumerate(self.study_profile_criteria.matrix_criteria):
            self._df_to_csv_d([criterion])
            self._df_to_csv_d(results_p_matrix_criteria[i_criterion] / self.n_samples)
        # Results on voting systems
        for i_task, (rule_class, options, study_rule_criteria) in enumerate(self.voting_rule_tasks):
            # Winner
            rate_win = results_vs_winner[i_task] / self.n_samples
            for c in range(n_c):
                self._df_to_csv_m(
                    [
                        rule_class.full_name, rule_class.abbreviation, rule_class.__name__,
                        'Winner', c, labels_candidates[c], '', '',
                        rate_win[c], rate_win[c], 0, self.n_samples, n_v, n_c
                    ]
                    + self.log_csv
                )
            # Result criteria
            for i_criterion, criterion in enumerate(study_rule_criteria.result_criteria):
                lower = results_vs_result_criteria[i_task][i_criterion]['inf'] / self.n_samples
                upper = results_vs_result_criteria[i_task][i_criterion]['sup'] / self.n_samples
                self._df_to_csv_m(
                    [rule_class.full_name, rule_class.abbreviation, rule_class.__name__,
                     criterion, '', '', '', '', lower, upper, upper - lower, self.n_samples, n_v, n_c]
                    + self.log_csv
                )
            # Manipulation criteria
            for i_criterion, criterion in enumerate(study_rule_criteria.manipulation_criteria):
                lower = results_vs_manipulation_criteria[i_task][i_criterion]['inf'] / self.n_samples
                upper = results_vs_manipulation_criteria[i_task][i_criterion]['sup'] / self.n_samples
                the_time = results_vs_manipulation_criteria[i_task][i_criterion]['time']
                the_log = results_vs_manipulation_criteria[i_task][i_criterion]['log']
                self._df_to_csv_m(
                    [
                        rule_class.full_name, rule_class.abbreviation, rule_class.__name__,
                        criterion, 'any', '', the_log, the_time, lower, upper, upper - lower,
                        self.n_samples, n_v, n_c
                    ]
                    + self.log_csv
                )
            for i_criterion, criterion in enumerate(study_rule_criteria.manipulation_criteria_c):
                lower = results_vs_manipulation_criteria_c[i_task][i_criterion]['inf'] / self.n_samples
                upper = results_vs_manipulation_criteria_c[i_task][i_criterion]['sup'] / self.n_samples
                the_time = results_vs_manipulation_criteria_c[i_task][i_criterion]['time']
                the_log = results_vs_manipulation_criteria_c[i_task][i_criterion]['log']
                for c in range(n_c):
                    self._df_to_csv_m(
                        [
                            rule_class.full_name, rule_class.abbreviation, rule_class.__name__,
                            criterion, c, labels_candidates[c], the_log, '', lower[c], upper[c], upper[c] - lower[c],
                            self.n_samples, n_v, n_c
                        ]
                        + self.log_csv
                    )
                # Indicate the time only once
                self._df_to_csv_m(
                    [rule_class.full_name, rule_class.abbreviation, rule_class.__name__,
                     criterion, 'any', '', the_log, the_time, '', '', '', self.n_samples, n_v, n_c]
                    + self.log_csv
                )
            # Utility criteria
            for i_criterion, (criterion, func, name) in enumerate(study_rule_criteria.utility_criteria):
                answer = func(results_vs_utility_criteria[i_task][i_criterion])
                self._df_to_csv_m(
                    [rule_class.full_name, rule_class.abbreviation, rule_class.__name__,
                     name, '', '', '', '', answer, answer, 0, self.n_samples, n_v, n_c]
                    + self.log_csv
                )
            # Numerical criteria
            for i_criterion, criterion in enumerate(study_rule_criteria.numerical_criteria):
                lower = results_vs_numerical_criteria[i_task][i_criterion]['inf'] / self.n_samples
                upper = results_vs_numerical_criteria[i_task][i_criterion]['sup'] / self.n_samples
                the_time = results_vs_numerical_criteria[i_task][i_criterion]['time']
                try:
                    self._df_to_csv_m(
                        [
                            rule_class.full_name, rule_class.abbreviation, rule_class.__name__,
                            criterion, 'any', '', '', the_time, lower, upper, upper - lower,
                            self.n_samples, n_v, n_c
                        ]
                        + self.log_csv
                    )
                except ValueError:
                    print(criterion)
                    print(lower)
                    print(upper)
                    raise ValueError
        self.file_m_en.close()
        self.file_m_fr.close()
        self.file_d_en.close()
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
