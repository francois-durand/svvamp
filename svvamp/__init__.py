"""Top-level package for SVVAMP."""

__author__ = """François Durand"""
__email__ = 'fradurand@gmail.com'
__version__ = '0.9.1'


# Utils
from svvamp.utils.misc import initialize_random_seeds
from svvamp.utils.constants import OPTIONS

# Profile
from svvamp.preferences.profile import Profile
from svvamp.preferences.profile_from_file import ProfileFromFile
from svvamp.preferences.profile_subset_candidates import ProfileSubsetCandidates

# Profile Generator
from svvamp.preferences.generator_profile import GeneratorProfile
from svvamp.preferences.generator_profile_cubic_uniform import GeneratorProfileCubicUniform
from svvamp.preferences.generator_profile_euclidean_box import GeneratorProfileEuclideanBox
from svvamp.preferences.generator_profile_gaussian_well import GeneratorProfileGaussianWell
from svvamp.preferences.generator_profile_ianc import GeneratorProfileIanc
from svvamp.preferences.generator_profile_ladder import GeneratorProfileLadder
from svvamp.preferences.generator_profile_noise import GeneratorProfileNoise
from svvamp.preferences.generator_profile_noised_file import GeneratorProfileNoisedFile
from svvamp.preferences.generator_profile_spheroid import GeneratorProfileSpheroid
from svvamp.preferences.generator_profile_vmf_hypercircle import GeneratorProfileVMFHypercircle
from svvamp.preferences.generator_profile_vmf_hypersphere import GeneratorProfileVMFHypersphere

# Voting Rule
from svvamp.rules.rule import Rule
from svvamp.rules.rule_approval import RuleApproval
from svvamp.rules.rule_baldwin import RuleBaldwin
from svvamp.rules.rule_black import RuleBlack
from svvamp.rules.rule_borda import RuleBorda
from svvamp.rules.rule_bucklin import RuleBucklin
from svvamp.rules.rule_condorcet_sum_defeats import RuleCondorcetSumDefeats
from svvamp.rules.rule_coombs import RuleCoombs
from svvamp.rules.rule_copeland import RuleCopeland
from svvamp.rules.rule_irv_duels import RuleIRVDuels
from svvamp.rules.rule_iterated_bucklin import RuleIteratedBucklin
from svvamp.rules.rule_kemeny import RuleKemeny
from svvamp.rules.rule_kim_roush import RuleKimRoush
from svvamp.rules.rule_majority_judgment import RuleMajorityJudgment
from svvamp.rules.rule_maximin import RuleMaximin
from svvamp.rules.rule_nanson import RuleNanson
from svvamp.rules.rule_plurality import RulePlurality
from svvamp.rules.rule_range_voting import RuleRangeVoting
from svvamp.rules.rule_ranked_pairs import RuleRankedPairs
from svvamp.rules.rule_schulze import RuleSchulze
from svvamp.rules.rule_split_cycle import RuleSplitCycle
from svvamp.rules.rule_star import RuleSTAR
from svvamp.rules.rule_two_round import RuleTwoRound
from svvamp.rules.rule_veto import RuleVeto

# Voting Rule: IRV Family
from svvamp.rules.rule_exhaustive_ballot import RuleExhaustiveBallot
from svvamp.rules.rule_irv import RuleIRV
from svvamp.rules.rule_condorcet_abs_irv import RuleCondorcetAbsIRV
from svvamp.rules.rule_condorcet_vtb_irv import RuleCondorcetVtbIRV
from svvamp.rules.rule_icrv import RuleICRV
from svvamp.rules.rule_irv_average import RuleIRVAverage
from svvamp.rules.rule_smith_irv import RuleSmithIRV
from svvamp.rules.rule_tideman import RuleTideman
from svvamp.rules.rule_woodall import RuleWoodall

# All rules
from svvamp.rules.all_rule_classes import ALL_RULE_CLASSES

# Meta
from svvamp.meta.study_profile_criteria import StudyProfileCriteria
from svvamp.meta.study_rule_criteria import StudyRuleCriteria
from svvamp.meta.voting_rule_tasks import VotingRuleTasks
from svvamp.meta.experiment_analyzer import ExperimentAnalyzer
from svvamp.meta.experiments_compiler import ExperimentsCompiler
