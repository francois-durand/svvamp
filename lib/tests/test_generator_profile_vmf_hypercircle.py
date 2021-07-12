from svvamp.utils.misc import initialize_random_seeds
from svvamp import GeneratorProfileVMFHypercircle


def test():
    """
    When there is one group only, you may specify `vmf_probability` but it is ignored:

        >>> initialize_random_seeds()
        >>> generator = GeneratorProfileVMFHypercircle(n_v=5, n_c=3, vmf_concentration=10, vmf_probability=1)
        >>> profile = generator()
        >>> profile.preferences_ut
        array([[ 0.67886167, -0.73231774,  0.05345607],
               [ 0.62834837, -0.76570906,  0.1373607 ],
               [ 0.71859954, -0.6950244 , -0.02357514],
               [ 0.49333272, -0.81010857,  0.31677584],
               [ 0.63042345, -0.76457205,  0.1341486 ]])
    """
    pass
