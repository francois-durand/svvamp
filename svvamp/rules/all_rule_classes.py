# Voting Rule
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

ALL_RULE_CLASSES = [
    RuleApproval,
    RuleBaldwin, RuleBlack, RuleBorda, RuleBucklin,
    RuleCondorcetSumDefeats, RuleCoombs, RuleCopeland,
    RuleIRVDuels, RuleIteratedBucklin,
    RuleKemeny, RuleKimRoush,
    RuleMajorityJudgment, RuleMaximin,
    RuleNanson,
    RulePlurality,
    RuleRangeVoting, RuleRankedPairs,
    RuleSchulze, RuleSplitCycle, RuleSTAR,
    RuleTwoRound,
    RuleVeto,
    RuleExhaustiveBallot, RuleIRV, RuleCondorcetAbsIRV, RuleCondorcetVtbIRV,
    RuleICRV, RuleIRVAverage, RuleSmithIRV, RuleTideman, RuleWoodall,
]
