# -*- coding: utf-8 -*-

__author__ = 'François Durand'
__email__ = 'fradurand@gmail.com'
__version__ = '0.0.4'

from .Preferences.Population import Population
from .Preferences.PopulationCubicUniform import PopulationCubicUniform
from .Preferences.PopulationEuclideanBox import PopulationEuclideanBox
from .Preferences.PopulationGaussianWell import PopulationGaussianWell
from .Preferences.PopulationLadder import PopulationLadder
from .Preferences.PopulationSpheroid import PopulationSpheroid
from .Preferences.PopulationVMFHypercircle import PopulationVMFHypercircle
from .Preferences.PopulationVMFHypersphere import PopulationVMFHypersphere
from .Preferences.PopulationFromFile import PopulationFromFile

from .VotingSystems.ElectionResult import ElectionResult
from .VotingSystems.Election import Election
from .VotingSystems.Approval import Approval
from .VotingSystems.Baldwin import Baldwin
from .VotingSystems.Borda import Borda
from .VotingSystems.Bucklin import Bucklin
from .VotingSystems.CondorcetAbsIRV import CondorcetAbsIRV
from .VotingSystems.CondorcetSumDefeats import CondorcetSumDefeats
from .VotingSystems.CondorcetVtbIRV import CondorcetVtbIRV
from .VotingSystems.Coombs import Coombs
from .VotingSystems.ExhaustiveBallot import ExhaustiveBallot
from .VotingSystems.ICRV import ICRV
from .VotingSystems.IRV import IRV
from .VotingSystems.IRVDuels import IRVDuels
from .VotingSystems.IteratedBucklin import IteratedBucklin
from .VotingSystems.Kemeny import Kemeny
from .VotingSystems.MajorityJudgment import MajorityJudgment
from .VotingSystems.Maximin import Maximin
from .VotingSystems.Nanson import Nanson
from .VotingSystems.Plurality import Plurality
from .VotingSystems.RangeVotingAverage import RangeVotingAverage
from .VotingSystems.RankedPairs import RankedPairs
from .VotingSystems.Schulze import Schulze
from .VotingSystems.TwoRound import TwoRound
from .VotingSystems.Veto import Veto
