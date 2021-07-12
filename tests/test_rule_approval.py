from svvamp import RuleApproval, Profile


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
        >>> profile = Profile(preferences_ut=[
        ...     [ 0. , -0.5, -1. ],
        ...     [ 1. , -1. ,  0.5],
        ...     [ 0.5,  0.5, -0.5],
        ...     [ 0.5,  0. ,  1. ],
        ...     [-1. , -1. ,  1. ],
        ... ], preferences_rk=[
        ...     [0, 1, 2],
        ...     [0, 2, 1],
        ...     [1, 0, 2],
        ...     [2, 0, 1],
        ...     [2, 1, 0],
        ... ])
        >>> rule = RuleApproval(approval_comparator='>=')(profile)
        >>> rule.w_
        0
    """


def test_tm_um_c():
    """
        >>> profile = Profile(preferences_ut=[
        ...     [ 0. , -0.5, -1. ],
        ...     [ 1. , -1. ,  0.5],
        ...     [ 0.5,  0.5, -0.5],
        ...     [ 0.5,  0. ,  1. ],
        ...     [-1. , -1. ,  1. ],
        ... ], preferences_rk=[
        ...     [0, 1, 2],
        ...     [0, 2, 1],
        ...     [1, 0, 2],
        ...     [2, 0, 1],
        ...     [2, 1, 0],
        ... ])
        >>> rule = RuleApproval()(profile)
        >>> rule.is_tm_c_(0)
        False
        >>> rule.is_um_c_(0)
        False
    """
    pass
