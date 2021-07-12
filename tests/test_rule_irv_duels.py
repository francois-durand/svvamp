from svvamp import RuleIRVDuels
from svvamp import Profile


def test_loser_equals_selected_two():
    """
        >>> profile = Profile(preferences_rk=[
        ...     [0, 2, 1],
        ...     [0, 2, 1],
        ...     [1, 0, 1],
        ... ])
        >>> rule = RuleIRVDuels()(profile)
        >>> rule.w_
        0
    """
    pass
