from svvamp.utils.misc import initialize_random_seeds
from svvamp import RuleCondorcetAbsIRV, GeneratorProfileLadder


def test_cm_slow():
    """
        >>> cm_option = 'slow'
        >>> initialize_random_seeds()
        >>> profile = GeneratorProfileLadder(n_v=5, n_c=3, n_rungs=5)().profile_
        >>> rule = RuleCondorcetAbsIRV(cm_option=cm_option)(profile)
        >>> rule.candidates_cm_
        array([0., 0., 0.])
    """
    pass


def test_cm_almost_exact():
    """
        >>> cm_option = 'almost_exact'
        >>> initialize_random_seeds()
        >>> profile = GeneratorProfileLadder(n_v=5, n_c=3, n_rungs=5)().profile_
        >>> rule = RuleCondorcetAbsIRV(cm_option=cm_option)(profile)
        >>> rule.candidates_cm_
        array([0., 0., 0.])
    """
    pass


def test_cm_exact():
    """
        >>> cm_option = 'exact'
        >>> initialize_random_seeds()
        >>> profile = GeneratorProfileLadder(n_v=5, n_c=3, n_rungs=5)().profile_
        >>> rule = RuleCondorcetAbsIRV(cm_option=cm_option)(profile)
        >>> rule.candidates_cm_
        array([0., 0., 0.])
    """
    pass
