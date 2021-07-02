from svvamp import RuleCoombs, Profile


def test_um_option_setter():
    """
        >>> rule = RuleCoombs()
        >>> rule.um_option = 'fast'
        >>> rule.um_option
        'fast'
        >>> rule.um_option = 'exact'
        >>> rule.um_option
        'exact'
        >>> rule.um_option = 'foo'
        Traceback (most recent call last):
        ValueError: Unknown value for um_option: foo
    """
    pass


def test_cm_option_setter():
    """
        >>> rule = RuleCoombs()
        >>> rule.cm_option = 'fast'
        >>> rule.cm_option
        'fast'
        >>> rule.cm_option = 'exact'
        >>> rule.cm_option
        'exact'
        >>> rule.cm_option = 'foo'
        Traceback (most recent call last):
        ValueError: Unknown value for cm_option: foo
    """
    pass


def test_is_um_c():
    """
        >>> profile = Profile(preferences_rk=[
        ...     [0, 1, 2],
        ...     [0, 2, 1],
        ...     [0, 2, 1],
        ...     [0, 2, 1],
        ...     [2, 0, 1],
        ... ])
        >>> rule = RuleCoombs()(profile)
        >>> rule.is_um_c_(2)
        nan
    """
    pass
