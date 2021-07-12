from svvamp.utils.misc import initialize_random_seeds
from svvamp import GeneratorProfileVMFHypersphere


def test():
    """
    When there is one group only, you may specify `vmf_probability` but it is ignored:

        >>> initialize_random_seeds()
        >>> generator = GeneratorProfileVMFHypersphere(n_v=5, n_c=3, vmf_concentration=10, vmf_probability=1)
        >>> profile = generator()
        >>> profile.preferences_ut
        array([[ 0.95891992, -0.12067454,  0.25672991],
               [ 0.86494714,  0.06265355,  0.49793673],
               [ 0.94128919, -0.27215259,  0.19976892],
               [ 0.80831535, -0.13959226,  0.57196179],
               [ 0.60411981, -0.07180745,  0.79365165]])
    """
    pass
