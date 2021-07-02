import numpy as np
from svvamp import GeneratorProfileLadder
from svvamp.utils.misc import initialize_random_seeds
from svvamp import RuleApproval


def test_approval_threshold_setter():
    """
        >>> rule = RuleApproval()
        >>> rule.approval_threshold = 0
        >>> rule.approval_threshold
        0.0
        >>> rule.approval_threshold = 0.1
        >>> rule.approval_threshold
        0.1
        >>> rule.approval_threshold = 'rosebud'
        Traceback (most recent call last):
        ValueError: Unknown value for approval_threshold: rosebud
    """
    pass


def test_approval_comparator_setter():
    """
        >>> rule = RuleApproval()
        >>> rule.approval_comparator = '>'
        >>> rule.approval_comparator
        '>'
        >>> rule.approval_comparator = '>='
        >>> rule.approval_comparator
        '>='
        >>> rule.approval_comparator = '<'
        Traceback (most recent call last):
        ValueError: Unknown option for approval_comparator: <
    """
    pass


def test_comparator_greater_or_equal():
    """
        >>> initialize_random_seeds()
        >>> profile = GeneratorProfileLadder(n_v=5, n_c=3, n_rungs=5)().profile_
        >>> rule = RuleApproval(approval_comparator='>=')(profile)
        >>> rule.w_
        0
    """


def test_tm_um_c():
    """
        >>> initialize_random_seeds()
        >>> profile = GeneratorProfileLadder(n_v=5, n_c=3, n_rungs=5)().profile_
        >>> rule = RuleApproval()(profile)
        >>> rule.is_tm_c_(0)
        False
        >>> rule.is_um_c_(0)
        False
    """
    pass
