from svvamp import RuleCondorcetDuel, Profile
from tests.cm_brute_force import check_cm_against_brute_force


def test_cm_fast():
    """
    >>> profile = Profile(preferences_ut=[
    ...     [ 0. , -0.5, -1. ],
    ...     [ 1. , -1. ,  0.5],
    ...     [ 0.5,  0.5, -0.5],
    ...     [ 0.5,  0. ,  1. ],
    ...     [-1. , -1. ,  1. ],
    ... ], preferences_rk=[
    ...     [0, 1, 2],
    ...     [0, 2, 1],
    ...     [1, 0, 2],
    ...     [2, 0, 1],
    ...     [2, 1, 0],
    ... ])
    >>> rule = RuleCondorcetDuel(cm_option='fast')(profile)
    >>> rule.candidates_cm_
    array([0., 0., 0.])
    """
    pass


def test_cm_exact():
    """
    >>> profile = Profile(preferences_ut=[
    ...     [ 0. , -0.5, -1. ],
    ...     [ 1. , -1. ,  0.5],
    ...     [ 0.5,  0.5, -0.5],
    ...     [ 0.5,  0. ,  1. ],
    ...     [-1. , -1. ,  1. ],
    ... ], preferences_rk=[
    ...     [0, 1, 2],
    ...     [0, 2, 1],
    ...     [1, 0, 2],
    ...     [2, 0, 1],
    ...     [2, 1, 0],
    ... ])
    >>> rule = RuleCondorcetDuel(cm_option='exact')(profile)
    >>> rule.candidates_cm_
    array([0., 0., 0.])
    """
    pass


def test_cm_against_brute_force():
    """The CM algorithms never contradict the brute force (which keeps each voter at her position in the profile)."""
    check_cm_against_brute_force(RuleCondorcetDuel, n_profiles=15, n_v_max=5, n_c_max=4, seed=0, cm_option="fast")
    n_undecided = check_cm_against_brute_force(RuleCondorcetDuel, n_profiles=15, n_v_max=5, n_c_max=4, seed=0, cm_option="exact")
    assert n_undecided == 0
