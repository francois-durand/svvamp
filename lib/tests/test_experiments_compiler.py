import os
from svvamp import ExperimentAnalyzer, ProfileFromFile, VotingRuleTasks, RuleRangeVoting, ExperimentsCompiler, \
    RulePlurality


def test():
    """
    Some housework first:

        >>> this_dir = os.path.dirname(__file__)
        >>> output_dir = os.path.join(this_dir, 'out')
        >>> tikz_directory = os.path.join(this_dir, 'tikz')
        >>> if not os.path.exists(output_dir):
        ...     os.mkdir(output_dir)
        >>> if not os.path.exists(tikz_directory):
        ...     os.mkdir(tikz_directory)
        >>> file_paths = [os.path.join(output_dir, file) for file in os.listdir(output_dir)]
        >>> file_paths += [os.path.join(tikz_directory, file) for file in os.listdir(tikz_directory)]
        >>> for file_path in file_paths:
        ...     if os.path.isfile(file_path):
        ...         os.remove(file_path)
        >>> os.rmdir(tikz_directory)  # We let the analyzer create it, just for test coverage.

    Define and run the experiment analyzers:

        >>> experiment_analyzer = ExperimentAnalyzer(
        ...     voting_rule_tasks=VotingRuleTasks(
        ...         voting_systems=[RulePlurality, RuleRangeVoting],
        ...     ),
        ...     output_dir=output_dir,
        ... )
        VotingRuleTasks: Sanity check was successful.
        >>> experiment_analyzer(
        ...     base_profile=ProfileFromFile(os.path.join(this_dir, 'ED-00004-00000040.soc')),
        ...     output_file_suffix='ED-00004-00000040',
        ... )
        Compute...
        Write results in csv files...
        Simulation finished
        >>> experiment_analyzer(
        ...     base_profile=ProfileFromFile(os.path.join(this_dir, 'ED-00004-00000067.soc')),
        ...     output_file_suffix='ED-00004-00000067',
        ... )
        Compute...
        Write results in csv files...
        Simulation finished

    Define the experiment compiler:

        >>> experiment_compiler = ExperimentsCompiler(
        ...     prefix_tikz_file='foo',
        ...     tikz_directory=tikz_directory,
        ...     results_directory=output_dir,
        ... )

    And enjoy!

        >>> experiment_compiler.profiles_scatter_plot()  # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
                Rule  ...       file_name
        ...  Profile  ...       Results_...
        ...  Profile  ...       Results_...
        <BLANKLINE>
        [2 rows x 24 columns]
        >>> experiment_compiler.profile_features_bar_plot()  # doctest: +NORMALIZE_WHITESPACE
             Rate
        CW    1.0
        CO    1.0
        RCW   1.0
        MF    1.0
        >>> experiment_compiler.cm_rate_bar_plot()  # doctest: +NORMALIZE_WHITESPACE
                     Rate (lower bound)  Rate (upper bound)  Rate (uncertainty)
        Rule (abbr)
        RV                          0.0                 0.0                 0.0
        Plu                         0.0                 0.0                 0.0
        >>> experiment_compiler.cm_rate_bar_plot()  # doctest: +NORMALIZE_WHITESPACE
                     Rate (lower bound)  Rate (upper bound)  Rate (uncertainty)
        Rule (abbr)
        RV                          0.0                 0.0                 0.0
        Plu                         0.0                 0.0                 0.0
        >>> experiment_compiler.tm_rate_bar_plot()  # doctest: +NORMALIZE_WHITESPACE
                     Rate (lower bound)  Rate (upper bound)  Rate (uncertainty)
        Rule (abbr)
        RV                          0.0                 0.0                 0.0
        Plu                         0.0                 0.0                 0.0
        >>> experiment_compiler.um_rate_bar_plot()  # doctest: +NORMALIZE_WHITESPACE
                     Rate (lower bound)  Rate (upper bound)  Rate (uncertainty)
        Rule (abbr)
        RV                          0.0                 0.0                 0.0
        Plu                         0.0                 0.0                 0.0
        >>> experiment_compiler.cm_tm_um_rate_bar_plot()  # doctest: +NORMALIZE_WHITESPACE
                                    TM  ...                 CM
                    Rate (lower bound)  ... Rate (uncertainty)
        Rule (abbr)                     ...
        RV                         0.0  ...                0.0
        Plu                        0.0  ...                0.0
        <BLANKLINE>
        [2 rows x 9 columns]
        >>> experiment_compiler.condorcet_consistency_bar_plot()  # doctest: +NORMALIZE_WHITESPACE
                               Sincere  ...                 CM
                    Rate (lower bound)  ... Rate (uncertainty)
        Rule (abbr)                     ...
        RV                         0.0  ...                0.0
        Plu                        0.0  ...                0.0
        <BLANKLINE>
        [2 rows x 4 columns]
        >>> experiment_compiler.loss_social_welfare_bar_plot()  # doctest: +NORMALIZE_WHITESPACE
                               Sincere  ...                 CM
                    Rate (lower bound)  ... Rate (uncertainty)
        Rule (abbr)                     ...
        RV                         0.0  ...                0.0
        Plu                        0.0  ...                0.0
        <BLANKLINE>
        [2 rows x 4 columns]
        >>> experiment_compiler.nb_candidates_cm_line_plot(rules=['RV', 'Plu'])  # doctest: +NORMALIZE_WHITESPACE
        C      3
        RV   0.0
        Plu  0.0
        >>> experiment_compiler.nb_candidates_cm_bar_plot()  # doctest: +NORMALIZE_WHITESPACE
                     Rate (lower bound)  Rate (upper bound)  Rate (uncertainty)
        Rule (abbr)
        RV                          0.0                 0.0                 0.0
        Plu                         0.0                 0.0                 0.0
        >>> experiment_compiler.cm_power_index_bar_plot()  # doctest: +NORMALIZE_WHITESPACE
                     Rate (lower bound)  Rate (upper bound)  Rate (uncertainty)
        Rule (abbr)
        Plu                    1.322229            1.322229                 0.0
        RV                     1.668547            1.668547                 0.0
        >>> experiment_compiler.cm_complexity_index_bar_plot()  # doctest: +NORMALIZE_WHITESPACE
                     Rate (lower bound)  Rate (upper bound)  Rate (uncertainty)
        Rule (abbr)
        RV                          NaN                 NaN                 NaN
        Plu                         NaN                 NaN                 NaN
        >>> experiment_compiler.df_computation_time()  # doctest: +NORMALIZE_WHITESPACE
                     Rule  ...  file_name
        ...     Plurality  ...  ...
        ...
        >>> experiment_compiler.df_computation_time_cumulative()  # doctest: +NORMALIZE_WHITESPACE
           Rule (abbr)                              Criterion  Computation time
        ...

    We finish with some housework:

        >>> file_paths = [os.path.join(output_dir, file) for file in os.listdir(output_dir)]
        >>> file_paths += [os.path.join(tikz_directory, file) for file in os.listdir(tikz_directory)]
        >>> for file_path in file_paths:
        ...     if os.path.isfile(file_path):
        ...         os.remove(file_path)
        >>> os.rmdir(output_dir)
        >>> os.rmdir(tikz_directory)
    """
    pass
