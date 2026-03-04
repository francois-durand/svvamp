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
from matplotlib import colormaps
import matplotlib.backends.backend_pgf
matplotlib.backends.backend_pgf.common_texification = matplotlib.backends.backend_pgf._tex_escape
import matplotlib.legend
def get_legend_handles(legend):
    return legend.legend_handles
matplotlib.legend.Legend.legendHandles = property(get_legend_handles)
import webcolors

def integer_rgb_to_hex(rgb):
    return "#{:02x}{:02x}{:02x}".format(*rgb)

webcolors.CSS3_HEX_TO_NAMES = {integer_rgb_to_hex(webcolors.name_to_rgb(name)): name
                               for name in webcolors.names("css3")}
import tikzplotlib
from pathlib import Path
from svvamp.utils.tikzplotlib_fix_ncols import tikzplotlib_fix_ncols


class ExperimentsCompiler:
    """
    Compile the results of several experiments.

    Each experiment should be analyzed with `ExperimentAnalyzer`. Then `ExperimentsCompiler` compiles the
    analyses.

    Parameters
    ----------
        prefix_tikz_file: str
            Prefix for the names of the tikz files.
        tikz_directory: str or Path
            Directory where to save the tikz files.
        results_directory: str or Path
            Directory where to fetch the results obtained with `ExperimentAnalyzer`.
        figsize: tuple
            Size of the figures.
        ignore_rules_abbr: list of str
            List of rules to ignore, given as abbreviations.
    """

    def __init__(self, prefix_tikz_file, tikz_directory='tikz', results_directory='out', figsize=(16, 8),
                 ignore_rules_abbr=None):
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
            if '_m_en' not in f.name:
                continue
            with f.open() as buffer:
                try:
                    new_df = pd.read_csv(buffer)
                except UnicodeDecodeError:  # pragma: no cover
                    # This should not happen.
                    print(f)
                    raise UnicodeDecodeError
            new_df['file_name'] = f.name
            dataframes.append(new_df)
        self.df = pd.concat(dataframes, ignore_index=True)
        # Remove the ignored rules
        if ignore_rules_abbr is not None:
            self.df = self.df[~self.df['Rule (abbr)'].isin(ignore_rules_abbr)]
        # Default order of the rules
        df_temp = self.df[self.df['Criterion'] == 'Winner'].pivot_table(
            values=['Rate (lower bound)'],
            index='Rule (abbr)'
        )

        def order_rules(rule_abbreviation):
            rules = [
                'Ben', 'SI', 'Tid', 'Woo',
                'CI', 'IRV', 'EB',
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
        # self.d_abbr_new = {'CIRV': 'CI', 'SIRV': 'SI', 'STAR': 'Sta', 'IRVD': 'Vie'}
        self.d_abbr_new = {}

    def profiles_scatter_plot(self, tikz_file='profiles_scatter_plot.tex'):
        """Scatter plot: Number of voters and candidates of the profiles."""
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
        """Bar plot: Features of the profiles (existence of Condorcet winner, etc)."""
        d_criterion_legend = {
            'exists_condorcet_winner_rk': 'CW',
            'exists_condorcet_order_rk': 'CO',
            'exists_super_condorcet_winner': 'SCW',
            'exists_pair_safe_condorcet_winner': 'PSCW',
            'exists_set_safe_condorcet_winner': 'SSCW',
            'exists_resistant_condorcet_winner': 'RCW',
            'exists_majority_favorite_rk': 'MF',
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

    def rate_bar_plot(self, criterion, ylabel, tikz_file,
                      n_c_equal=None, n_c_ge=None, n_c_le=None,
                      f_rule_label=None, f_rule_value_used_for_color=None,
                      d_rule_color=None, d_bound_color=None,
                      draw_rcw_line=False, draw_pscw_line=False, draw_sscw_line=False, draw_scw_line=False,
                      d_old_new=None):
        """Bar plot: Rate of some criterion (auxiliary function)."""
        # Create the pivot table
        constraint = self.df['Criterion'] == criterion
        if n_c_equal is not None:
            constraint &= (self.df['C'] == n_c_equal)
        if n_c_ge is not None:
            constraint &= (self.df['C'] >= n_c_ge)
        if n_c_le is not None:
            constraint &= (self.df['C'] <= n_c_le)
        df_plot = self.df[constraint].pivot_table(
            values=['Rate (lower bound)', 'Rate (upper bound)', 'Rate (uncertainty)'],
            index='Rule (abbr)'
        )
        df_plot = df_plot.loc[self.rules_order, ['Rate (lower bound)', 'Rate (upper bound)', 'Rate (uncertainty)']]
        df_plot.sort_values(by=['Rate (lower bound)', 'Rate (upper bound)', ], inplace=True)
        # Plot
        fig, ax = plt.subplots(figsize=self.figsize)
        if draw_rcw_line:
            self.resistant_condorcet_line(n_c_equal=n_c_equal, n_c_ge=n_c_ge, n_c_le=n_c_le,
                                          d_bound_color=d_bound_color)
        if draw_pscw_line:
            self.pscw_line(n_c_equal=n_c_equal, n_c_ge=n_c_ge, n_c_le=n_c_le,
                           d_bound_color=d_bound_color)
        if draw_sscw_line:
            self.sscw_line(n_c_equal=n_c_equal, n_c_ge=n_c_ge, n_c_le=n_c_le,
                           d_bound_color=d_bound_color)
        if draw_scw_line:
            self.scw_line(n_c_equal=n_c_equal, n_c_ge=n_c_ge, n_c_le=n_c_le,
                          d_bound_color=d_bound_color)
        if f_rule_value_used_for_color is not None:
            colormap = colormaps['Spectral_r']
            colors = [colormap(min(f_rule_value_used_for_color(rule), .9999)) for rule in df_plot.index]
        elif d_rule_color is not None:
            colors = [d_rule_color[rule] for rule in df_plot.index]
        else:
            colors = None
        plt.bar(
            df_plot.index,
            df_plot['Rate (lower bound)'],
            yerr=(np.zeros(len(df_plot.index)), df_plot['Rate (uncertainty)']),
            color=colors
        )
        plt.grid(axis='y')
        ax.set_axisbelow(True)
        plt.xlim(-1, self.n_rules)
        plt.ylim(0, 1.05)
        plt.yticks(np.arange(0, 1.1, step=0.1))
        plt.ylabel(ylabel)
        if f_rule_label is not None:
            plt.xticks(
                ticks=np.arange(len(df_plot.index)),
                labels=[rule + '\n' + f_rule_label(rule) for rule in df_plot.index],
                rotation=0, ha='center'
            )
        self.my_tikzplotlib_save(tikz_file, x_ticks_labels=df_plot.index, d_old_new=d_old_new)
        return df_plot

    def cm_rate_bar_plot(self, n_c_equal=None, n_c_ge=None, n_c_le=None,
                         f_rule_label=None, d_rule_color=None, d_bound_color=None,
                         f_rule_value_used_for_color=None, tikz_file='cm_rate_bar_plot.tex', d_old_new=None):
        """Bar plot: CM rate."""
        return self.rate_bar_plot(criterion='is_cm_', ylabel='CM rate', tikz_file=tikz_file,
                                  n_c_equal=n_c_equal, n_c_ge=n_c_ge, n_c_le=n_c_le,
                                  f_rule_label=f_rule_label, d_rule_color=d_rule_color,
                                  d_bound_color=d_bound_color,
                                  f_rule_value_used_for_color=f_rule_value_used_for_color,
                                  draw_rcw_line=True, draw_pscw_line=True, draw_sscw_line=True, draw_scw_line=True,
                                  d_old_new=d_old_new)

    def tm_rate_bar_plot(self, n_c_equal=None, n_c_ge=None, n_c_le=None,
                         f_rule_label=None, d_rule_color=None, d_bound_color=None,
                         f_rule_value_used_for_color=None, tikz_file='tm_rate_bar_plot.tex', d_old_new=None):
        """Bar plot: TM rate."""
        return self.rate_bar_plot(criterion='is_tm_', ylabel='TM rate', tikz_file=tikz_file,
                                  n_c_equal=n_c_equal, n_c_ge=n_c_ge, n_c_le=n_c_le,
                                  f_rule_label=f_rule_label, d_rule_color=d_rule_color,
                                  d_bound_color=d_bound_color,
                                  f_rule_value_used_for_color=f_rule_value_used_for_color, d_old_new=d_old_new)

    def um_rate_bar_plot(self, n_c_equal=None, n_c_ge=None, n_c_le=None,
                         f_rule_label=None, d_rule_color=None, d_bound_color=None,
                         f_rule_value_used_for_color=None, tikz_file='um_rate_bar_plot.tex', d_old_new=None):
        """Bar plot: UM rate."""
        return self.rate_bar_plot(criterion='is_um_', ylabel='UM rate', tikz_file=tikz_file,
                                  n_c_equal=n_c_equal, n_c_ge=n_c_ge, n_c_le=n_c_le,
                                  f_rule_label=f_rule_label, d_rule_color=d_rule_color,
                                  d_bound_color=d_bound_color,
                                  f_rule_value_used_for_color=f_rule_value_used_for_color, d_old_new=d_old_new)

    def cm_xm_rate_bar_plot(self, n_c_equal=None, n_c_ge=None, n_c_le=None,
                            tikz_file='cm_xm_rate_bar_plot.tex'):
        """Bar plot: CM and XM rates. [Beta feature]"""
        # Create the pivot table
        constraint_cm = self.df['Criterion'] == 'is_cm_'
        constraint_xm = self.df['Criterion'] == 'is_xm_'
        if n_c_equal is not None:
            constraint_cm &= (self.df['C'] == n_c_equal)
            constraint_xm &= (self.df['C'] == n_c_equal)
        if n_c_ge is not None:
            constraint_cm &= (self.df['C'] >= n_c_ge)
            constraint_xm &= (self.df['C'] >= n_c_ge)
        if n_c_le is not None:
            constraint_cm &= (self.df['C'] <= n_c_le)
            constraint_xm &= (self.df['C'] <= n_c_le)
        df_cm = self.df[constraint_cm].pivot_table(
            values=['Rate (lower bound)', 'Rate (upper bound)', 'Rate (uncertainty)'],
            index='Rule (abbr)'
        )
        df_xm = self.df[constraint_xm].pivot_table(
            values=['Rate (lower bound)', 'Rate (upper bound)', 'Rate (uncertainty)'],
            index='Rule (abbr)'
        )
        df_cm = df_cm.loc[self.rules_order, ['Rate (lower bound)', 'Rate (upper bound)', 'Rate (uncertainty)']]
        df_xm = df_xm.loc[self.rules_order, ['Rate (lower bound)', 'Rate (upper bound)', 'Rate (uncertainty)']]
        df_plot = pd.concat([df_cm, df_xm], axis=1, keys=['CM', 'XM'])
        df_plot.sort_values(by=[
            ('CM', 'Rate (lower bound)'), ('CM', 'Rate (upper bound)'),
            ('XM', 'Rate (lower bound)'), ('XM', 'Rate (upper bound)'),
        ], inplace=True)
        # Plot
        x = np.arange(len(df_plot.index))
        bar_width = 0.25
        fig, ax = plt.subplots(figsize=self.figsize)
        ax.bar(x - bar_width / 2, df_plot[('CM', 'Rate (lower bound)')], bar_width, label='CM rate',
               yerr=(np.zeros(len(df_plot.index)), df_plot[('CM', 'Rate (uncertainty)')]))
        ax.bar(x + bar_width / 2, df_plot[('XM', 'Rate (lower bound)')], bar_width, label='XM rate',
               yerr=(np.zeros(len(df_plot.index)), df_plot[('XM', 'Rate (uncertainty)')]))
        self.resistant_condorcet_line(n_c_equal=n_c_equal, n_c_ge=n_c_ge, n_c_le=n_c_le)
        self.sscw_line(n_c_equal=n_c_equal, n_c_ge=n_c_ge, n_c_le=n_c_le)
        self.pscw_line(n_c_equal=n_c_equal, n_c_ge=n_c_ge, n_c_le=n_c_le)
        self.scw_line(n_c_equal=n_c_equal, n_c_ge=n_c_ge, n_c_le=n_c_le)
        plt.grid(axis='y')
        ax.set_axisbelow(True)
        ax.set_xticks(x)
        ax.set_xticklabels(df_plot.index)
        plt.xlim(-1, self.n_rules)
        plt.ylim(0, 1.05)
        plt.yticks(np.arange(0, 1.1, step=0.1))
        plt.ylabel('Rate')
        ax.legend(loc='upper left')
        tikzplotlib_fix_ncols(ax)
        self.my_tikzplotlib_save(tikz_file, x_ticks_labels=df_plot.index)
        return df_plot

    def cm_tm_um_rate_bar_plot(self, tikz_file='cm_tm_um_rate_bar_plot.tex'):
        """Bar plot: CM, TM and UM rates."""
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
        tikzplotlib_fix_ncols(ax)
        self.my_tikzplotlib_save(tikz_file, x_ticks_labels=df_plot.index)
        return df_plot

    def condorcet_consistency_bar_plot(self, tikz_file='condorcet_consistency_bar_plot.tex'):
        """Bar plot: Condorcet consistency."""
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
        tikzplotlib_fix_ncols(ax)
        self.my_tikzplotlib_save(tikz_file, x_ticks_labels=df_violation.index)
        return df_violation

    def loss_social_welfare_bar_plot(self, tikz_file='loss_social_welfare_bar_plot.tex'):
        """Bar plot: loss of social welfare."""
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
        tikzplotlib_fix_ncols(ax)
        self.my_tikzplotlib_save(tikz_file, x_ticks_labels=df_loss_sw.index)
        return df_loss_sw

    def nb_candidates_rate_line_plot(self, criterion='is_cm_', ylabel='CM rate',
                                     rules=None, remove_rules_with_high_uncertainty=False,
                                     d_rule_color=None,
                                     exponent_y_axis=1, tikz_file='nb_candidates_rate_line_plot.tex',
                                     d_old_new=None):
        """Line plot representing a rate as a function of the number of candidates."""
        # Pivot table for uncertainty
        df_uncertainty = self.df[(self.df['Criterion'] == criterion)].pivot_table(
            values=['Rate (uncertainty)'],
            index='Rule (abbr)',
            columns='C'
        )
        rules_with_high_uncertainty = df_uncertainty.index[np.any(df_uncertainty > 0.01, axis=1)]
        rules_with_low_uncertainty = df_uncertainty.index[np.all(df_uncertainty <= 0.01, axis=1)]
        # Create the pivot table for the line plot
        df_plot = self.df[(self.df['Criterion'] == criterion)].pivot_table(
            values=['Rate (lower bound)'],
            index='Rule (abbr)',
            columns='C'
        )
        df_plot = df_plot.loc[self.rules_order, 'Rate (lower bound)']
        if rules is not None:
            df_plot = df_plot.loc[
                      [rule for rule in df_plot.index if rule in rules], :]
        if remove_rules_with_high_uncertainty:
            df_plot = df_plot.loc[
                      [rule for rule in df_plot.index if rule in rules_with_low_uncertainty], :]
        else:
            df_plot.rename(index={rule: rule + "*" for rule in rules_with_high_uncertainty}, inplace=True)
        df_plot.sort_values(df_plot.columns[-1], inplace=True, ascending=False)
        df_plot.index = self.replace_rule_names(df_plot.index)
        fig, ax = plt.subplots(figsize=self.figsize)
        df_plot_exponentiated = df_plot ** exponent_y_axis
        if d_rule_color is None:
            df_plot_exponentiated.T.plot(ax=ax)
        else:
            x_values = df_plot_exponentiated.columns
            print(x_values)
            for rule in df_plot_exponentiated.T:
                try:
                    y_values = df_plot_exponentiated.loc[rule, x_values]
                    color = d_rule_color[rule[:-1] if rule[-1] == '*' else rule]
                    plt.plot(x_values, y_values, label=rule, color=color)
                except KeyError:
                    print(f'Warning for {rule=}')
        plt.grid()
        ax.set_axisbelow(True)
        plt.xlabel('Number of candidates $m$')
        plt.xlim(df_plot.columns[0], df_plot.columns[-1])
        k = exponent_y_axis
        plt.yticks(
            ticks=[x ** k for x in np.arange(0., 1.05, .1)],
            labels=[f"{x:.1f}" if x == 0 or x >= .45 else "" for x in np.arange(0., 1.05, .1)]
        )
        plt.ylabel('CM rate')
        plt.ylim(-0.05, 1.05)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        tikzplotlib_fix_ncols(ax)
        self.my_tikzplotlib_save(tikz_file, axis_width=r'\axisSmallerWidth', axis_height=r'\axisHeight',
                                 d_old_new=d_old_new)
        return df_plot

    def nb_candidates_cm_line_plot(self, rules=None, tikz_file='nb_candidates_cm_line_plot.tex'):
        """Line plot: Number of CM winners."""
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
        plt.ylabel('Average number of CM winners')
        plt.ylim(0, df_nb_candidates_cm.columns[-1] - .5)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        tikzplotlib_fix_ncols(ax)
        self.my_tikzplotlib_save(tikz_file, axis_width=r'\axisSmallerWidth', axis_height=r'\axisHeight')
        return df_nb_candidates_cm

    def nb_candidates_cm_bar_plot(self, tikz_file='nb_candidates_cm_bar_plot.tex'):
        """Bar plot: Number of CM winners."""
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
        plt.ylabel('Average ratio of CM winners')
        self.my_tikzplotlib_save(tikz_file, x_ticks_labels=df_plot.index)
        return df_plot

    def cm_power_index_bar_plot(self, tikz_file='cm_power_index_bar_plot.tex'):
        """Bar plot: CM power index."""
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
        """Bar plot: CM complexity index."""
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
        """Dataframe: Computation time."""
        df_sorted = self.df.sort_values(by='Computation time', ascending=False)
        df_sorted.drop(['Rule (abbr)', 'Rule (class)', 'Candidate name', 'Culture type'], axis=1, inplace=True)
        for i in range(1, 5):
            df_sorted.drop([f'Culture parameter {i}', f'Culture parameter {i} value'], axis=1, inplace=True)
        return df_sorted

    def df_computation_time_cumulative(self):
        """Dataframe: Cumulative computation time."""
        df_cum_time = self.df.pivot_table(
            values='Computation time',
            index=['Rule (abbr)', 'Criterion'],
            aggfunc=np.sum,
        )
        df_cum_time.reset_index(inplace=True)
        df_cum_time.sort_values(by='Computation time', ascending=False, inplace=True)
        return df_cum_time

    def line(self, criterion, label, label_above=True, n_c_equal=None, n_c_ge=None, n_c_le=None, color=None):
        """Draw a line for some criterion on the profiles."""
        if color is None:
            color = 'purple'
        constraint = (self.df['Criterion'] == criterion)
        if n_c_equal is not None:
            constraint &= (self.df['C'] == n_c_equal)
        if n_c_ge is not None:
            constraint &= (self.df['C'] >= n_c_ge)
        if n_c_le is not None:
            constraint &= (self.df['C'] <= n_c_le)
        rate = self.df[constraint].pivot_table(
            values=['Rate (lower bound)'],
            index='Rule'
        ).iloc[0, 0]
        nb_rules = self.df[
            self.df['Criterion'] == 'is_cm_'
        ].pivot_table(
            values=['Rate (lower bound)'],
            index='Rule'
        ).index.size
        plt.hlines(1 - rate, -1, nb_rules, color=color, linestyles='dashed', zorder=-1)
        label_yshift = -0.01 if label_above else 0.01
        verticalalignment = 'bottom' if label_above else 'top'
        plt.text(-.5, 1 - rate + label_yshift, label, color=color,
                 horizontalalignment='left', verticalalignment=verticalalignment, fontsize='medium')

    def resistant_condorcet_line(self, n_c_equal=None, n_c_ge=None, n_c_le=None, d_bound_color=None):
        """Draw the 'Resistant Condorcet' line (upper bound of CM rate for Condorcet rules)."""
        color = None if d_bound_color is None else d_bound_color.get('rcw')
        return self.line(criterion='exists_resistant_condorcet_winner', label='RCW bound', label_above=True,
                         n_c_equal=n_c_equal, n_c_ge=n_c_ge, n_c_le=n_c_le, color=color)

    def sscw_line(self, n_c_equal=None, n_c_ge=None, n_c_le=None, d_bound_color=None):
        """Draw the 'SSCW' line."""
        color = None if d_bound_color is None else d_bound_color.get('sscw')
        return self.line(criterion='exists_set_safe_condorcet_winner', label='SSCW bound', label_above=True,
                         n_c_equal=n_c_equal, n_c_ge=n_c_ge, n_c_le=n_c_le, color=color)

    def pscw_line(self, n_c_equal=None, n_c_ge=None, n_c_le=None, d_bound_color=None):
        """Draw the 'PSCW' line."""
        color = None if d_bound_color is None else d_bound_color.get('pscw')
        return self.line(criterion='exists_pair_safe_condorcet_winner', label='PSCW bound', label_above=True,
                         n_c_equal=n_c_equal, n_c_ge=n_c_ge, n_c_le=n_c_le, color=color)

    def scw_line(self, n_c_equal=None, n_c_ge=None, n_c_le=None, d_bound_color=None):
        """Draw the 'SCW' line."""
        color = None if d_bound_color is None else d_bound_color.get('scw')
        return self.line(criterion='exists_super_condorcet_winner', label='SCW bound', label_above=True,
                         n_c_equal=n_c_equal, n_c_ge=n_c_ge, n_c_le=n_c_le, color=color)

    def my_tikzplotlib_save(self, tikz_file, x_ticks_labels=None,
                            axis_width=r'\axisWidth', axis_height=r'\axisHeight',
                            d_old_new=None, transform_string=None, with_prefix=True):
        """Save a figure in tikz."""
        file_name = (self.prefix_tikz_file if with_prefix else '') + tikz_file
        tikzplotlib.save(self.tikz_directory / file_name,
                         axis_width=axis_width, axis_height=axis_height)
        with open(self.tikz_directory / file_name, 'r') as f:
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
        if d_old_new is not None:
            for k, v in d_old_new.items():
                file_data = file_data.replace(k, v)
        if transform_string is not None:
            file_data = transform_string(file_data)
        with open(self.tikz_directory / file_name, 'w') as f:
            f.write(file_data)

    def replace_rule_names(self, x_ticks_labels):
        """Replace the x-labels with relevant abbreviations (auxiliary function)."""
        return [self.d_abbr_new[abbr] if abbr in self.d_abbr_new.keys() else abbr
                for abbr in x_ticks_labels]
