from svvamp import RuleIRV, Profile


def test_fast_algo_setter():
    """
        >>> rule = RuleIRV()
        >>> rule.fast_algo = 'foo'
        Traceback (most recent call last):
        ValueError: Unknown value for option fast_algo: foo
    """
    pass


def test_log_um_():
    """
        >>> rule = RuleIRV(um_option='exact')
        >>> rule.log_um_
        'um_option = exact'
    """
    pass


def test_log_cm_():
    """
        >>> rule = RuleIRV(cm_option='exact')
        >>> rule.log_cm_
        'cm_option = exact'
    """
    pass


def test_v_might_be_pivotal():
    """
        >>> profile = Profile(preferences_rk=[
        ...     [0, 1, 2],
        ...     [0, 2, 1],
        ...     [1, 0, 2],
        ...     [2, 0, 1],
        ...     [2, 1, 0],
        ... ])
        >>> rule = RuleIRV()(profile)
        >>> rule.v_might_be_pivotal_
        array([ True,  True,  True,  True,  True])
    """
    pass


def test_v_might_im_for_c():
    """
        >>> profile = Profile(preferences_rk=[
        ...     [0, 1, 2],
        ...     [0, 2, 1],
        ...     [1, 0, 2],
        ...     [2, 0, 1],
        ...     [2, 1, 0],
        ... ])
        >>> rule = RuleIRV()(profile)
        >>> rule.v_might_im_for_c_
        array([[ True,  True,  True],
               [ True,  True,  True],
               [ True,  True,  True],
               [ True,  True,  True],
               [ True,  True,  True]])
    """
    pass


def test_is_not_iia():
    """
        >>> profile = Profile(preferences_rk=[
        ...     [0, 1, 2],
        ...     [0, 2, 1],
        ...     [1, 0, 2],
        ...     [2, 0, 1],
        ...     [2, 1, 0],
        ... ])
        >>> rule = RuleIRV()(profile)
        >>> rule.is_not_iia_
        False
    """
    pass


def test_is_icm_c():
    """
    Examples
    --------
        >>> profile = Profile(preferences_rk=[
        ...     [1, 0, 2],
        ...     [1, 2, 0],
        ...     [1, 2, 0],
        ...     [2, 1, 0],
        ...     [2, 1, 0],
        ... ])
        >>> rule = RuleIRV()(profile)
        >>> rule.is_icm_c_(0)
        False
        >>> rule.is_icm_c_with_bounds_(0)
        (False, 5.0, 5.0)
    """
    pass
