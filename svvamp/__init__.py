"""Top-level package for SVVAMP."""

__author__ = """François Durand"""
__email__ = 'fradurand@gmail.com'
__version__ = '0.1.0'


# Profile
from .Preferences.Profile import Profile
from .Preferences.ProfileFromFile import ProfileFromFile
from .Preferences.ProfileSubsetCandidates import ProfileSubsetCandidates

# Profile Generator
from .Preferences.GeneratorProfile import GeneratorProfile
from .Preferences.GeneratorProfileCubicUniform import GeneratorProfileCubicUniform
from .Preferences.GeneratorProfileEuclideanBox import GeneratorProfileEuclideanBox
from .Preferences.GeneratorProfileGaussianWell import GeneratorProfileGaussianWell
from .Preferences.GeneratorProfileLadder import GeneratorProfileLadder
from .Preferences.GeneratorProfileNoise import GeneratorProfileNoise
from .Preferences.GeneratorProfileNoisedFile import ProfileGeneratorNoisedFile
from .Preferences.GeneratorProfileSpheroid import GeneratorProfileSpheroid
from .Preferences.GeneratorProfileVMFHypercircle import GeneratorProfileVMFHypercircle
from .Preferences.GeneratorProfileVMFHypersphere import GeneratorProfileVMFHypersphere

# Voting Rule
from .Rules.Rule import Rule
from .Rules.RuleApproval import RuleApproval
from .Rules.RuleBaldwin import RuleBaldwin
from .Rules.RuleBorda import RuleBorda
from .Rules.RuleBucklin import RuleBucklin
from .Rules.RuleCondorcetSumDefeats import RuleCondorcetSumDefeats
from .Rules.RuleCoombs import RuleCoombs
from .Rules.RuleICRV import RuleICRV
from .Rules.RuleIRVAverage import RuleIRVAverage
from .Rules.RuleIRVDuels import RuleIRVDuels
from .Rules.RuleIteratedBucklin import RuleIteratedBucklin
from .Rules.RuleKemeny import RuleKemeny
from .Rules.RuleKimRoush import RuleKimRoush
from .Rules.RuleMajorityJudgment import RuleMajorityJudgment
from .Rules.RuleMaximin import RuleMaximin
from .Rules.RuleNanson import RuleNanson
from .Rules.RulePlurality import RulePlurality
from .Rules.RuleRangeVoting import RuleRangeVoting
from .Rules.RuleRankedPairs import RuleRankedPairs
from .Rules.RuleSchulze import RuleSchulze
from .Rules.RuleTwoRound import RuleTwoRound
from .Rules.RuleVeto import RuleVeto

# Voting Rule: IRV Family
from .Rules.RuleExhaustiveBallot import RuleExhaustiveBallot
from .Rules.RuleIRV import RuleIRV
from .Rules.RuleCondorcetAbsIRV import RuleCondorcetAbsIRV
from .Rules.RuleCondorcetVtbIRV import RuleCondorcetVtbIRV
