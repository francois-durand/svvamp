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


def test_improve_coverage_eb():
    """
        >>> profile = Profile(preferences_rk=[
        ...     [0, 1, 2, 3, 4],
        ...     [0, 1, 3, 4, 2],
        ...     [0, 2, 4, 3, 1],
        ...     [0, 3, 1, 2, 4],
        ...     [0, 4, 1, 2, 3],
        ...     [0, 4, 2, 1, 3],
        ...     [0, 4, 2, 1, 3],
        ...     [1, 0, 4, 2, 3],
        ...     [1, 2, 4, 0, 3],
        ...     [1, 3, 0, 4, 2],
        ...     [1, 3, 4, 0, 2],
        ...     [1, 3, 4, 0, 2],
        ...     [1, 4, 2, 3, 0],
        ...     [2, 0, 3, 1, 4],
        ...     [2, 1, 0, 4, 3],
        ...     [2, 1, 3, 0, 4],
        ...     [2, 3, 0, 4, 1],
        ...     [2, 4, 1, 0, 3],
        ...     [2, 4, 1, 3, 0],
        ...     [2, 4, 3, 1, 0],
        ...     [3, 0, 2, 1, 4],
        ...     [3, 0, 4, 2, 1],
        ...     [3, 0, 4, 2, 1],
        ...     [3, 1, 2, 0, 4],
        ...     [3, 1, 2, 4, 0],
        ...     [3, 2, 1, 4, 0],
        ...     [3, 4, 2, 1, 0],
        ...     [4, 1, 0, 3, 2],
        ...     [4, 1, 3, 2, 0],
        ...     [4, 1, 3, 2, 0],
        ...     [4, 3, 0, 1, 2],
        ...     [4, 3, 1, 2, 0],
        ... ])
        >>> RuleCondorcetAbsIRV(cm_option='almost_exact')(profile).is_cm_
        True
    """
    pass
