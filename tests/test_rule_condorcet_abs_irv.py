from svvamp import RuleCondorcetAbsIRV, Profile


def test_cm_slow():
    """
        >>> cm_option = 'slow'
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
        >>> rule = RuleCondorcetAbsIRV(cm_option=cm_option)(profile)
        >>> rule.candidates_cm_
        array([0., 0., 0.])
    """
    pass


def test_cm_almost_exact():
    """
        >>> cm_option = 'almost_exact'
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
        >>> rule = RuleCondorcetAbsIRV(cm_option=cm_option)(profile)
        >>> rule.candidates_cm_
        array([0., 0., 0.])
    """
    pass


def test_cm_exact():
    """
        >>> cm_option = 'exact'
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
        >>> rule = RuleCondorcetAbsIRV(cm_option=cm_option)(profile)
        >>> rule.candidates_cm_
        array([0., 0., 0.])
    """
    pass
