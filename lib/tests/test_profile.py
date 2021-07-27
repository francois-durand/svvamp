import pytest
from svvamp import Profile


def test_not_enough_candidates():
    with pytest.raises(ValueError):
        profile = Profile(preferences_ut=[[1], [0]])
        _ = profile.n_c


def test_set_labels_candidates():
    """
        >>> profile = Profile(preferences_ut=[[1, 0], [1, 0]])
        >>> profile.labels_candidates
        ['0', '1']
        >>> profile.labels_candidates = ['Alice', 'Bob']
        >>> profile.labels_candidates
        ['Alice', 'Bob']
    """
    pass
