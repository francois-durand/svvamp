from svvamp import RuleExhaustiveBallot


def test_fast_algo_setter():
    """
        >>> rule = RuleExhaustiveBallot()
        >>> rule.fast_algo = 'foo'
        Traceback (most recent call last):
        ValueError: Unknown value for option fast_algo: foo
    """
    pass


def test_log_um_():
    """
        >>> rule = RuleExhaustiveBallot(um_option='exact')
        >>> rule.log_um_
        'um_option = exact'
    """
    pass


def test_log_cm_():
    """
        >>> rule = RuleExhaustiveBallot(cm_option='exact')
        >>> rule.log_cm_
        'cm_option = exact'
    """
    pass
