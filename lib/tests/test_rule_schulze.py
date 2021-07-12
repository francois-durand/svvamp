from svvamp import RuleSchulze


def test_um_option_setter():
    """
        >>> rule = RuleSchulze()
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
        >>> rule = RuleSchulze()
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
