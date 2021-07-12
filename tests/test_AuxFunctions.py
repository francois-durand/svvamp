import numpy as np
from svvamp.utils.misc import is_resistant_condorcet


def test():
    assert not is_resistant_condorcet(w=np.nan, preferences_ut=np.array([[5, 1, 2], [4, 10, 1]]))
    assert is_resistant_condorcet(w=0, preferences_ut=np.array([[5, 1, 2], [10, 4, 1]]))
