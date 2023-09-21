import os
from svvamp import ExperimentAnalyzer, ProfileFromFile, VotingRuleTasks, RuleCondorcetAbsIRV, StudyProfileCriteria, \
    StudyRuleCriteria, RuleRangeVoting


def test():
    """
    Some housework first:

        >>> this_dir = os.path.dirname(__file__)
        >>> output_dir = os.path.join(this_dir, 'out_analyze_only')
        >>> if not os.path.exists(output_dir):
        ...     os.mkdir(output_dir)
        >>> file_paths = [os.path.join(output_dir, file) for file in os.listdir(output_dir)]
        >>> for file_path in file_paths:
        ...     if os.path.isfile(file_path):
        ...         os.remove(file_path)

    Define and run the experiment analyzer:

        >>> file = os.path.join(this_dir, 'ED-00017-00000001.toi')
        >>> base_profile = ProfileFromFile(file)
        >>> experiment_analyzer = ExperimentAnalyzer(output_dir=output_dir, ping_period=1)
        >>> experiment_analyzer(
        ...     base_profile=base_profile,
        ...     study_profile_criteria=StudyProfileCriteria(boolean_criteria=['exists_condorcet_winner_rk']),
        ...     voting_rule_tasks=VotingRuleTasks(
        ...         voting_systems=[RuleCondorcetAbsIRV, RuleRangeVoting],
        ...         study_rule_criteria=StudyRuleCriteria(manipulation_criteria_c=['candidates_cm_'])
        ...     ),
        ... )
        StudyProfileCriteria: Sanity check was successful.
        VotingRuleTasks: Sanity check was successful.
        Compute...
        1 profiles analyzed
        Write results in csv files...
        Simulation finished

    We finish with some housework:

        >>> file_paths = [os.path.join(output_dir, file) for file in os.listdir(output_dir)]
        >>> for file_path in file_paths:
        ...     if os.path.isfile(file_path):
        ...         os.remove(file_path)
        >>> os.rmdir(output_dir)
    """
    pass
