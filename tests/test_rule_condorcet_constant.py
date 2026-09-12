import numpy as np

from svvamp import Profile, Rule, RuleCondorcetConstant
from tests.cm_brute_force import check_cm_against_brute_force, check_um_against_brute_force


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
    >>> rule = RuleCondorcetConstant(cm_option='fast')(profile)
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
    >>> rule = RuleCondorcetConstant(cm_option='exact')(profile)
    >>> rule.candidates_cm_
    array([0., 0., 0.])
    """
    pass


def test_cm_against_brute_force():
    """The CM algorithms never contradict the brute force (which keeps each voter at her position in the profile)."""
    check_cm_against_brute_force(RuleCondorcetConstant, n_profiles=15, n_v_max=5, n_c_max=4, seed=0, cm_option="fast")
    n_undecided = check_cm_against_brute_force(
        RuleCondorcetConstant, n_profiles=15, n_v_max=5, n_c_max=4, seed=0, cm_option="exact"
    )
    assert n_undecided == 0


def test_um_against_brute_force():
    """The UM algorithm never contradicts the brute force (which keeps each voter at her position in the profile)."""
    n_undecided = check_um_against_brute_force(RuleCondorcetConstant, n_profiles=60, n_v_max=7, n_c_max=4, seed=1)
    assert n_undecided == 0


def test_iia_against_brute_force():
    """The IIA algorithm agrees with the exhaustive algorithm of the superclass on random profiles."""
    from tests.cm_brute_force import random_profiles

    for profile in random_profiles(100, n_v_max=7, n_c_max=4, seed=2):
        rule_exhaustive = RuleCondorcetConstant(iia_subset_maximum_size=profile.n_c - 1)(profile)
        # noinspection PyUnresolvedReferences
        expected = Rule.__dict__["_compute_iia_"].fget(rule_exhaustive)["is_iia"]
        found = RuleCondorcetConstant()(profile).is_iia_
        assert not np.isnan(expected)
        assert found == expected, profile.to_doctest_string()
