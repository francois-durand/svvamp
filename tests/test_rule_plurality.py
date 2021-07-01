from svvamp import RulePlurality
from svvamp import GeneratorProfileLadder
from svvamp.utils.misc import initialize_random_seeds


def test():
    """
        >>> initialize_random_seeds()
        >>> profile = GeneratorProfileLadder(n_v=5, n_c=3, n_rungs=5)().profile_
        >>> rule = RulePlurality()(profile)
        >>> rule.is_tm_c_(0)
        False
        >>> rule.is_um_c_(0)
        False
    """
    pass
