from svvamp import RulePlurality, Profile


def test():
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
        >>> rule = RulePlurality()(profile)
        >>> rule.is_tm_c_(0)
        False
        >>> rule.is_um_c_(0)
        False
        >>> rule.relative_social_welfare_w_
        1.0
        >>> rule.elects_condorcet_winner_rk_even_with_cm_
        True
        >>> rule.nb_candidates_cm_
        (0, 0)
        >>> rule.worst_relative_welfare_with_cm_
        (1.0, 1.0)
        >>> rule.cm_power_index_
        (2.5, 2.5)
        >>> rule.is_tm_or_um_
        False

        >>> profile = Profile(preferences_rk=[
        ...     [2, 0, 1],
        ...     [0, 1, 2],
        ...     [1, 0, 2],
        ...     [2, 1, 0],
        ... ])
        >>> rule = RulePlurality()(profile)
        >>> rule.is_tm_or_um_
        True

        >>> profile = Profile(preferences_rk=[
        ...     [2, 1, 0],
        ...     [2, 0, 1],
        ...     [0, 2, 1],
        ...     [0, 1, 2],
        ... ])
        >>> rule = RulePlurality()(profile)
        >>> rule.elects_condorcet_winner_rk_even_with_cm_
        False
    """
    pass
