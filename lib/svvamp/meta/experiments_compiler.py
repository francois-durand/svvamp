# -*- coding: utf-8 -*-
"""
Created on jul. 26, 2021, 08:43
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

import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pathlib import Path
import tikzplotlib


class ExperimentsCompiler:

    def __init__(self, prefix_tikz_file, tikz_directory='tikz', results_directory='out', figsize=(16, 8)):
        # Record parameters
        self.prefix_tikz_file = prefix_tikz_file
        self.results_directory = Path(results_directory)
        self.tikz_directory = Path(tikz_directory)
        self.figsize = figsize
        # Create tikz folder if it does not exist
        if not os.path.isdir(tikz_directory):
            os.mkdir(tikz_directory)
        # Create the mother of all dataframes
        dataframes = []
        for f in self.results_directory.iterdir():
            if '_m_en_' not in f.name:
                continue
            with f.open() as buffer:
                try:
                    new_df = pd.read_csv(buffer)
                except UnicodeDecodeError:
                    print(f)
                    raise UnicodeDecodeError
            new_df['file_name'] = f.name
            dataframes.append(new_df)
        self.df = pd.concat(dataframes, ignore_index=True)
        # Default order of the rules
        df_temp = self.df[self.df['Criterion'] == 'Winner'].pivot_table(
            values=['Rate (lower bound)'],
            index='Rule (abbr)'
        )

        def order_rules(rule_abbreviation):
            rules = [
                'Ben', 'SIRV', 'Tid', 'Woo',
                'CIRV', 'IRV', 'EB',
                'STAR', 'AV', 'RV'
            ]
            try:
                # noinspection PyUnresolvedReferences
                return rules.index(rule_abbreviation)
            except ValueError:
                return len(rules) + 1

        self.rules_order = sorted(list(df_temp.index), key=order_rules)
        # Number of rules
        self.n_rules = len(self.rules_order)
        # Corrected abbreviations of the rules
        self.d_abbr_new = {'CIRV': 'CI', 'SIRV': 'SI', 'STAR': 'Sta', 'IRVD': 'Vie'}

    def profiles_scatter_plot(self, tikz_file='profiles_scatter_plot.tex'):
        df_scatter = self.df[
            self.df['Criterion'] == 'exists_condorcet_admissible'  # No matter the criterion
            ]
        fig, ax = plt.subplots(figsize=self.figsize)
        plt.scatter(x=df_scatter['V'], y=df_scatter['C'])
        plt.grid(axis='y')
        ax.set_axisbelow(True)
        plt.xlabel('Number of voters $V$')
        plt.xscale('log')
        plt.xlim(xmin=1000)
        plt.ylabel('Number of candidates $C$')
        y_min = np.min(df_scatter['C'])
        y_max = np.max(df_scatter['C'])
        plt.yticks(range(y_min, y_max + 1))
        self.my_tikzplotlib_save(tikz_file)
        return df_scatter

    def profile_features_bar_plot(self, tikz_file='profile_features_bar_plot.tex'):
        d_criterion_legend = {
            'exists_condorcet_winner_rk': 'CW',
            'exists_condorcet_order_rk': 'CO',
            # 'exists_irv_immune_candidate': 'IRV-immune candidate',
            'exists_resistant_condorcet_winner': 'RCW',
            'exists_majority_favorite_rk': 'MF'
        }
        d_criterion_rate = {}
        for criterion in d_criterion_legend.keys():
            d_criterion_rate[criterion] = self.df[
                self.df['Criterion'] == criterion
                ].pivot_table(
                values=['Rate (lower bound)'],
                index='Rule'
            ).iloc[0, 0]
        x = list(d_criterion_legend.values())
        y = [d_criterion_rate[criterion] for criterion in d_criterion_legend.keys()]
        df_plot = pd.DataFrame(y, index=x, columns=['Rate'])
        # Plot
        fig, ax = plt.subplots(figsize=self.figsize)
        plt.bar(x, y)
        plt.grid(axis='y')
        ax.set_axisbelow(True)
        plt.ylim(0, 1.05)
        plt.yticks(np.arange(0, 1.1, step=0.1))
        plt.ylabel('Proportion of profiles')
        self.my_tikzplotlib_save(tikz_file, x_ticks_labels=d_criterion_legend.values())
        return df_plot

    def rate_bar_plot(self, criterion, ylabel, tikz_file, draw_rcw_line=False):
        # Create the pivot table
        df_plot = self.df[self.df['Criterion'] == criterion].pivot_table(
            values=['Rate (lower bound)', 'Rate (upper bound)', 'Rate (uncertainty)'],
            index='Rule (abbr)'
        )
        df_plot = df_plot.loc[self.rules_order, ['Rate (lower bound)', 'Rate (upper bound)', 'Rate (uncertainty)']]
        df_plot.sort_values(by=['Rate (lower bound)', 'Rate (upper bound)', ], inplace=True)
        # Plot
        fig, ax = plt.subplots(figsize=self.figsize)
        if draw_rcw_line:
            self.resistant_condorcet_line()
        plt.bar(
            df_plot.index,
            df_plot['Rate (lower bound)'],
            yerr=(np.zeros(len(df_plot.index)), df_plot['Rate (uncertainty)'])
        )
        plt.grid(axis='y')
        ax.set_axisbelow(True)
        plt.xlim(-1, self.n_rules)
        plt.ylim(0, 1.05)
        plt.yticks(np.arange(0, 1.1, step=0.1))
        plt.ylabel(ylabel)
        self.my_tikzplotlib_save(tikz_file, x_ticks_labels=df_plot.index)
        return df_plot

    def cm_rate_bar_plot(self, tikz_file='cm_rate_bar_plot.tex'):
        return self.rate_bar_plot(criterion='is_cm_', ylabel='CM rate', tikz_file=tikz_file,
                                  draw_rcw_line=True)

    def tm_rate_bar_plot(self, tikz_file='tm_rate_bar_plot.tex'):
        return self.rate_bar_plot(criterion='is_tm_', ylabel='TM rate', tikz_file=tikz_file)

    def um_rate_bar_plot(self, tikz_file='um_rate_bar_plot.tex'):
        return self.rate_bar_plot(criterion='is_um_', ylabel='UM rate', tikz_file=tikz_file)

    def cm_tm_um_rate_bar_plot(self, tikz_file='cm_tm_um_rate_bar_plot.tex'):
        # Create the pivot table
        df_tm = self.df[self.df['Criterion'] == 'is_tm_'].pivot_table(
            values=['Rate (lower bound)', 'Rate (upper bound)', 'Rate (uncertainty)'],
            index='Rule (abbr)'
        )
        df_tm = df_tm.loc[self.rules_order, ['Rate (lower bound)', 'Rate (upper bound)', 'Rate (uncertainty)']]

        df_um = self.df[self.df['Criterion'] == 'is_um_'].pivot_table(
            values=['Rate (lower bound)', 'Rate (upper bound)', 'Rate (uncertainty)'],
            index='Rule (abbr)'
        )
        df_um = df_um.loc[self.rules_order, ['Rate (lower bound)', 'Rate (upper bound)', 'Rate (uncertainty)']]

        df_cm = self.df[self.df['Criterion'] == 'is_cm_'].pivot_table(
            values=['Rate (lower bound)', 'Rate (upper bound)', 'Rate (uncertainty)'],
            index='Rule (abbr)'
        )
        df_cm = df_cm.loc[self.rules_order, ['Rate (lower bound)', 'Rate (upper bound)', 'Rate (uncertainty)']]

        df_plot = pd.concat([df_tm, df_um, df_cm], axis=1, keys=['TM', 'UM', 'CM'])
        df_plot.sort_values(by=[
            ('CM', 'Rate (lower bound)'), ('CM', 'Rate (upper bound)'),
            ('TM', 'Rate (lower bound)'), ('TM', 'Rate (upper bound)'),
            ('UM', 'Rate (lower bound)'), ('UM', 'Rate (upper bound)'),
        ], inplace=True)
        # Plot
        x = np.arange(len(df_plot.index))
        bar_width = 0.25
        fig, ax = plt.subplots(figsize=self.figsize)
        ax.bar(x - bar_width, df_plot[('TM', 'Rate (lower bound)')], bar_width, label='TM rate',
               yerr=(np.zeros(len(df_plot.index)), df_plot[('TM', 'Rate (uncertainty)')]))
        ax.bar(x, df_plot[('UM', 'Rate (lower bound)')], bar_width, label='UM rate',
               yerr=(np.zeros(len(df_plot.index)), df_plot[('UM', 'Rate (uncertainty)')]))
        ax.bar(x + bar_width, df_plot[('CM', 'Rate (lower bound)')], bar_width, label='CM rate',
               yerr=(np.zeros(len(df_plot.index)), df_plot[('CM', 'Rate (uncertainty)')]))
        self.resistant_condorcet_line()
        plt.grid(axis='y')
        ax.set_axisbelow(True)
        ax.set_xticks(x)
        ax.set_xticklabels(df_plot.index)
        plt.xlim(-1, self.n_rules)
        plt.ylim(0, 1.05)
        plt.yticks(np.arange(0, 1.1, step=0.1))
        plt.ylabel('Rate')
        ax.legend(loc='upper left')
        self.my_tikzplotlib_save(tikz_file, x_ticks_labels=df_plot.index)
        return df_plot

    def condorcet_consistency_bar_plot(self, tikz_file='condorcet_consistency_bar_plot.tex'):
        # Cumulative rate of Condorcet winner existence
        cum_rate_condorcet_exists = self.df[
            (self.df['Rule'] == 'Profile') & (self.df['Criterion'] == 'exists_condorcet_winner_rk')
            ].pivot_table(
            values='Rate (lower bound)',
            index='Rule',
            aggfunc=np.sum
        ).iloc[0, 0]
        # Sincere Condorcet consistency
        df_condorcet_consistency = self.df[
            self.df['Criterion'] == 'w_is_condorcet_winner_rk_'
            ].pivot_table(
            values='Rate (lower bound)',
            index='Rule (abbr)',
            aggfunc=np.sum
        )
        # Condorcet consistency with CM
        df_condorcet_consistency_despite_cm = self.df[
            self.df['Criterion'] == 'elects_condorcet_winner_rk_even_with_cm_'
            ].pivot_table(
            values=['Rate (lower bound)', 'Rate (upper bound)', 'Rate (uncertainty)'],
            index='Rule (abbr)',
            aggfunc=np.sum
        )
        df_condorcet_consistency_despite_cm = df_condorcet_consistency_despite_cm.loc[
                                              :, ['Rate (lower bound)', 'Rate (upper bound)', 'Rate (uncertainty)']]
        # Condorcet violation (sincere / with CM)
        df_violation = pd.DataFrame()
        df_violation[('Sincere', 'Rate (lower bound)')] = - df_condorcet_consistency[
            'Rate (lower bound)'] / cum_rate_condorcet_exists + 1
        df_violation[('CM', 'Rate (lower bound)')] = - df_condorcet_consistency_despite_cm[
            'Rate (upper bound)'] / cum_rate_condorcet_exists + 1
        df_violation[('CM', 'Rate (upper bound)')] = - df_condorcet_consistency_despite_cm[
            'Rate (lower bound)'] / cum_rate_condorcet_exists + 1
        df_violation[('CM', 'Rate (uncertainty)')] = df_condorcet_consistency_despite_cm[
                                                         'Rate (uncertainty)'] / cum_rate_condorcet_exists
        df_violation.columns = pd.MultiIndex.from_tuples(df_violation.columns)
        df_violation = df_violation.loc[self.rules_order, :]
        df_violation.sort_values(by=[
            ('CM', 'Rate (lower bound)'), ('CM', 'Rate (upper bound)'), ('Sincere', 'Rate (lower bound)')
        ], inplace=True)
        # Plot
        x = np.arange(len(df_violation.index))
        bar_width = 0.35
        fig, ax = plt.subplots(figsize=self.figsize)
        self.resistant_condorcet_line()
        ax.bar(x - bar_width / 2, df_violation[('Sincere', 'Rate (lower bound)')], bar_width, label='Sincere')
        ax.bar(x + bar_width / 2, df_violation[('CM', 'Rate (lower bound)')], bar_width, label='With CM',
               yerr=(np.zeros(len(df_violation.index)), df_violation[('CM', 'Rate (uncertainty)')]))
        plt.grid(axis='y')
        ax.set_axisbelow(True)
        ax.set_xticks(x)
        ax.set_xticklabels(df_violation.index)
        plt.xlim(-1, self.n_rules)
        plt.ylim(0, 1.05)
        plt.yticks(np.arange(0, 1.1, step=0.1))
        plt.ylabel('Condorcet violation rate')
        ax.legend(loc='upper left')
        self.my_tikzplotlib_save(tikz_file, x_ticks_labels=df_violation.index)
        return df_violation

    def loss_social_welfare_bar_plot(self, tikz_file='loss_social_welfare_bar_plot.tex'):
        # Sincere social welfare
        df_social_welfare_sincere = self.df[
            self.df['Criterion'] == 'relative_social_welfare_mean'
        ].pivot_table(
            values=['Rate (lower bound)'],
            index='Rule (abbr)',
        )
        # Social welfare with CM
        df_social_welfare_with_cm = self.df[
            self.df['Criterion'] == 'worst_relative_welfare_with_cm_'
        ].pivot_table(
            values=['Rate (lower bound)', 'Rate (upper bound)', 'Rate (uncertainty)'],
            index='Rule (abbr)',
        )
        # Loss of social welfare (sincere / with CM)
        df_loss_sw = pd.DataFrame()
        df_loss_sw[('Sincere', 'Rate (lower bound)')] = - df_social_welfare_sincere['Rate (lower bound)'] + 1
        df_loss_sw[('CM', 'Rate (lower bound)')] = - df_social_welfare_with_cm['Rate (upper bound)'] + 1
        df_loss_sw[('CM', 'Rate (upper bound)')] = - df_social_welfare_with_cm['Rate (lower bound)'] + 1
        df_loss_sw[('CM', 'Rate (uncertainty)')] = df_social_welfare_with_cm['Rate (uncertainty)']
        df_loss_sw.columns = pd.MultiIndex.from_tuples(df_loss_sw.columns)
        df_loss_sw = df_loss_sw.loc[self.rules_order, :]
        df_loss_sw.sort_values(by=[
            ('CM', 'Rate (lower bound)'), ('CM', 'Rate (upper bound)'), ('Sincere', 'Rate (lower bound)')
        ], inplace=True)
        # Plot
        x = np.arange(len(df_loss_sw.index))
        bar_width = 0.35
        fig, ax = plt.subplots(figsize=self.figsize)
        ax.bar(x - bar_width / 2, df_loss_sw[('Sincere', 'Rate (lower bound)')], bar_width, label='Sincere')
        ax.bar(x + bar_width / 2, df_loss_sw[('CM', 'Rate (lower bound)')], bar_width, label='With CM',
               yerr=(np.zeros(len(df_loss_sw.index)), df_loss_sw[('CM', 'Rate (uncertainty)')]))
        plt.grid(axis='y')
        ax.set_axisbelow(True)
        ax.set_xticks(x)
        ax.set_xticklabels(df_loss_sw.index)
        plt.xlim(-1, self.n_rules)
        plt.ylim(0, 1.05)
        plt.yticks(np.arange(0, 1.1, step=0.1))
        plt.ylabel('Loss of normalized social welfare')
        ax.legend(loc='upper left')
        self.my_tikzplotlib_save(tikz_file, x_ticks_labels=df_loss_sw.index)
        return df_loss_sw

    def nb_candidates_cm_line_plot(self, rules=None, tikz_file='nb_candidates_cm_line_plot.tex'):
        # Generate the df
        df_nb_candidates_cm = self.df[
            self.df['Criterion'] == 'nb_candidates_cm_'
        ].pivot_table(
            values=['Rate (lower bound)'],
            index='Rule (abbr)',
            columns='C',
        )
        df_nb_candidates_cm = df_nb_candidates_cm.loc[self.rules_order, 'Rate (lower bound)']
        if rules is not None:
            df_nb_candidates_cm = df_nb_candidates_cm.loc[
                [rule for rule in df_nb_candidates_cm.index if rule in rules], :]
        # Sort the df
        # df_nb_candidates_cm['mean'] = df_nb_candidates_cm.mean(axis=1)
        # df_nb_candidates_cm.sort_values('mean', inplace=True, ascending=False)
        # df_nb_candidates_cm.drop('mean', axis=1, inplace=True)
        df_nb_candidates_cm.sort_values(df_nb_candidates_cm.columns[-1], inplace=True, ascending=False)
        # Replace the names of the rules
        df_nb_candidates_cm.index = self.replace_rule_names(df_nb_candidates_cm.index)
        # Plot
        fig, ax = plt.subplots(figsize=self.figsize)
        df_nb_candidates_cm.T.plot(ax=ax)
        plt.grid()
        ax.set_axisbelow(True)
        plt.xlabel('Number of candidates')
        plt.xlim(3, df_nb_candidates_cm.columns[-1])
        plt.yticks(range(df_nb_candidates_cm.columns[-1]))
        plt.ylabel('Av. number of challengers who can win by CM')
        plt.ylim(0, df_nb_candidates_cm.columns[-1] - .5)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        self.my_tikzplotlib_save(tikz_file, axis_width=r'\axisSmallerWidth', axis_height=r'\axisHeight')
        return df_nb_candidates_cm

    def nb_candidates_cm_bar_plot(self, tikz_file='nb_candidates_cm_bar_plot.tex'):
        # Raw data
        df_nb_candidates_cm = self.df[
            self.df['Criterion'] == 'nb_candidates_cm_'
        ]
        # Raw data: proportion of candidates
        df_proportion_candidates_cm = pd.DataFrame()
        df_proportion_candidates_cm['Rule (abbr)'] = df_nb_candidates_cm['Rule (abbr)']
        df_proportion_candidates_cm['Rate (lower bound)'] = df_nb_candidates_cm['Rate (lower bound)'] / (
                df_nb_candidates_cm['C'] - 1)
        df_proportion_candidates_cm['Rate (upper bound)'] = df_nb_candidates_cm['Rate (upper bound)'] / (
                df_nb_candidates_cm['C'] - 1)
        df_proportion_candidates_cm['Rate (uncertainty)'] = df_nb_candidates_cm['Rate (uncertainty)'] / (
                df_nb_candidates_cm['C'] - 1)
        # Pivot table
        df_plot = df_proportion_candidates_cm.pivot_table(
            values=['Rate (lower bound)', 'Rate (upper bound)', 'Rate (uncertainty)'],
            index='Rule (abbr)',
        )
        df_plot = df_plot.loc[self.rules_order, ['Rate (lower bound)', 'Rate (upper bound)', 'Rate (uncertainty)']]
        df_plot.sort_values(by=['Rate (lower bound)', 'Rate (upper bound)'], inplace=True)
        # Plot
        fig, ax = plt.subplots(figsize=self.figsize)
        self.resistant_condorcet_line()
        plt.bar(
            df_plot.index,
            df_plot['Rate (lower bound)'],
            yerr=(np.zeros(len(df_plot.index)), df_plot['Rate (uncertainty)'])
        )
        plt.grid(axis='y')
        ax.set_axisbelow(True)
        plt.xlim(-1, self.n_rules)
        plt.ylim(0, 1.05)
        plt.yticks(np.arange(0, 1.1, step=0.1))
        plt.ylabel('Av. ratio of challengers who can win by CM')
        self.my_tikzplotlib_save(tikz_file, x_ticks_labels=df_plot.index)
        return df_plot

    def cm_power_index_bar_plot(self, tikz_file='cm_power_index_bar_plot.tex'):
        # Create the pivot table
        df_plot = self.df[self.df['Criterion'] == 'cm_power_index_'].pivot_table(
            values=['Rate (lower bound)', 'Rate (upper bound)', 'Rate (uncertainty)'],
            index='Rule (abbr)'
        )
        df_plot = df_plot.loc[self.rules_order, ['Rate (lower bound)', 'Rate (upper bound)', 'Rate (uncertainty)']]
        df_plot.sort_values(by=['Rate (lower bound)', 'Rate (upper bound)', ], inplace=True)
        # Set y_max
        y_max = df_plot['Rate (lower bound)'].max()
        y_quantum = 1 if y_max > 4.25 else 0.5
        y_max = (np.ceil(y_max / y_quantum - 0.5) + 0.51) * y_quantum
        # Plot
        uncertainty_to_plot = np.minimum(df_plot['Rate (upper bound)'], y_max) - df_plot['Rate (lower bound)']
        fig, ax = plt.subplots(figsize=self.figsize)
        plt.bar(
            df_plot.index,
            df_plot['Rate (lower bound)'],
            yerr=(np.zeros(len(df_plot.index)), uncertainty_to_plot)
        )
        plt.grid(axis='y')
        ax.set_axisbelow(True)
        plt.xlim(-1, self.n_rules)
        plt.ylim(0, y_max)
        plt.ylabel('CM power index')
        self.my_tikzplotlib_save(tikz_file, x_ticks_labels=df_plot.index)
        return df_plot

    def cm_complexity_index_bar_plot(self, tikz_file='cm_complexity_index_bar_plot.tex'):
        df_tm_or_um = self.df[
            self.df['Criterion'] == 'is_tm_or_um_'
        ].pivot_table(
            values=['Rate (lower bound)', 'Rate (upper bound)', 'Rate (uncertainty)'],
            index='Rule (abbr)',
            aggfunc=np.sum,
        )
        df_cm = self.df[
            self.df['Criterion'] == 'is_cm_'
        ].pivot_table(
            values=['Rate (lower bound)', 'Rate (upper bound)', 'Rate (uncertainty)'],
            index='Rule (abbr)',
            aggfunc=np.sum,
        )
        df_complexity_index = pd.DataFrame()
        df_complexity_index['Simplicity (lower bound)'] = df_tm_or_um['Rate (lower bound)'] / df_cm[
            'Rate (upper bound)']
        df_complexity_index['Simplicity (upper bound)'] = np.minimum(
            df_tm_or_um['Rate (upper bound)'] / df_cm['Rate (lower bound)'], 1)
        df_complexity_index['Rate (upper bound)'] = - df_complexity_index['Simplicity (lower bound)'] + 1
        df_complexity_index['Rate (lower bound)'] = - df_complexity_index['Simplicity (upper bound)'] + 1
        df_complexity_index['Rate (uncertainty)'] = (df_complexity_index['Rate (upper bound)']
                                                     - df_complexity_index['Rate (lower bound)'])
        df_complexity_index.drop('Simplicity (lower bound)', axis=1, inplace=True)
        df_complexity_index.drop('Simplicity (upper bound)', axis=1, inplace=True)
        df_complexity_index = df_complexity_index.loc[
            self.rules_order, ['Rate (lower bound)', 'Rate (upper bound)', 'Rate (uncertainty)']]
        df_complexity_index.sort_values(by=['Rate (lower bound)', 'Rate (upper bound)'], ascending=False, inplace=True)
        # Plot
        fig, ax = plt.subplots(figsize=self.figsize)
        plt.bar(
            df_complexity_index.index,
            df_complexity_index['Rate (lower bound)'],
            yerr=(np.zeros(len(df_complexity_index.index)), df_complexity_index['Rate (uncertainty)'])
        )
        plt.grid(axis='y')
        ax.set_axisbelow(True)
        plt.xlim(-1, self.n_rules)
        plt.ylim(0, 1.05)
        plt.yticks(np.arange(0, 1.1, step=0.1))
        plt.ylabel('CM complexity index')
        self.my_tikzplotlib_save(tikz_file, x_ticks_labels=df_complexity_index.index)
        return df_complexity_index

    def df_computation_time(self):
        df_sorted = self.df.sort_values(by='Computation time', ascending=False)
        df_sorted.drop(['Rule (abbr)', 'Rule (class)', 'Candidate name', 'Culture type'], axis=1, inplace=True)
        for i in range(1, 5):
            df_sorted.drop([f'Culture parameter {i}', f'Culture parameter {i} value'], axis=1, inplace=True)
        return df_sorted

    def df_computation_time_cumulative(self):
        df_cum_time = self.df.pivot_table(
            values='Computation time',
            index=['Rule (abbr)', 'Criterion'],
            aggfunc=np.sum,
        )
        df_cum_time.reset_index(inplace=True)
        df_cum_time.sort_values(by='Computation time', ascending=False, inplace=True)
        return df_cum_time

    def resistant_condorcet_line(self):
        rcw_rate = self.df[
            self.df['Criterion'] == 'exists_resistant_condorcet_winner'
        ].pivot_table(
            values=['Rate (lower bound)'],
            index='Rule'
        ).iloc[0, 0]
        nb_rules = self.df[
            self.df['Criterion'] == 'is_cm_'
        ].pivot_table(
            values=['Rate (lower bound)'],
            index='Rule'
        ).index.size
        plt.hlines(1 - rcw_rate, -1, nb_rules, 'purple', linestyles='dashed', zorder=-1)
        plt.text(8.5, 1 - rcw_rate + 0.03, 'RCW bound', color='purple',
                 horizontalalignment='center', verticalalignment='bottom', fontsize='medium')

    def my_tikzplotlib_save(self, tikz_file, x_ticks_labels=None,
                            axis_width=r'\axisWidth', axis_height=r'\axisHeight'):
        tikzplotlib.save(self.tikz_directory / (self.prefix_tikz_file + tikz_file),
                         axis_width=axis_width, axis_height=axis_height)
        with open(self.tikz_directory / (self.prefix_tikz_file + tikz_file), 'r') as f:
            file_data = f.read()
        # Set 'fill opacity' of the legend to 1
        file_data = file_data.replace('fill opacity=0.8,', 'fill opacity=1,')
        # Add yticks as they are in the matplotlib plot
        file_data = file_data.replace(
            'ytick style={',
            'ytick={' + ', '.join([str(y) for y in plt.yticks()[0]]) + '},\n'
            + 'ytick style={'
        )
        # Prevent from scaling down the plt.text
        file_data = file_data.replace('scale=0.5,', 'scale=1.0,')
        # Fix the x ticks in the tikz file
        if x_ticks_labels is not None:
            file_data = file_data.replace(
                    'y grid style={',
                    'xtick={' + ', '.join([str(i) for i in range(len(x_ticks_labels))]) + '},\n'
                    + 'xticklabels = {' + ', '.join(self.replace_rule_names(x_ticks_labels)) + '},\n'
                    + 'y grid style={'
                )
        with open(self.tikz_directory / (self.prefix_tikz_file + tikz_file), 'w') as f:
            f.write(file_data)

    def replace_rule_names(self, x_ticks_labels):
        return [self.d_abbr_new[abbr] if abbr in self.d_abbr_new.keys() else abbr
                for abbr in x_ticks_labels]
