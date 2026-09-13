"""Brute-force manipulation, used to cross-check the CM and UM algorithms of some voting rules.

Contrary to the exhaustive algorithm of :class:`svvamp.Rule`, this brute force keeps each voter at her position in
the profile, which matters for non-anonymous rules such as :class:`svvamp.RuleCondorcetDictatorship`.

Limitations: the rule must be based on strict rankings (the manipulators' ballots are enumerated as rankings), which
is checked with :attr:`svvamp.Rule.is_based_on_rk`. The options of the rule that affect the result of the election
(such as ``tie_break_rule`` for :class:`svvamp.RuleCopeland`) must be passed in ``rule_options``: they are transmitted
to the virtual elections through :attr:`svvamp.Rule._copy`.
"""

import itertools

import numpy as np

from svvamp import Profile


def candidates_cm_brute_force(rule_class, profile, n_m_max=None, **rule_options):
    """Decide CM for each candidate by trying all possible ballots of the manipulators.

    Parameters
    ----------
    rule_class : class
        A subclass of :class:`svvamp.Rule`, based on strict rankings.
    profile : Profile
    n_m_max : int or None
        If given, candidates with more manipulators than this are skipped (to bound the computation time).
    rule_options
        Options passed to the rule (cf. the module docstring).

    Returns
    -------
    ndarray
        ``candidates_cm[c]`` is 1. if the voters who prefer ``c`` to the sincere winner can make ``c`` win, 0.
        otherwise. ``nan`` for skipped candidates.
    """
    rule = rule_class(**rule_options)(profile)
    if not rule.is_based_on_rk:
        raise ValueError(f"{rule_class.__name__} is not based on strict rankings.")
    w = rule.w_
    n_c = profile.n_c
    candidates_cm = np.zeros(n_c)
    for c in range(n_c):
        if c == w:
            continue
        manipulators = np.where(rule.v_wants_to_help_c_[:, c])[0]
        if manipulators.shape[0] == 0:
            continue
        if n_m_max is not None and manipulators.shape[0] > n_m_max:
            candidates_cm[c] = np.nan
            continue
        rankings = list(itertools.permutations(range(n_c)))
        for ballots in itertools.product(rankings, repeat=manipulators.shape[0]):
            preferences_rk = np.copy(profile.preferences_rk)
            preferences_rk[manipulators, :] = np.array(ballots)
            w_test = rule._copy(profile=Profile(preferences_rk=preferences_rk, sort_voters=False)).w_
            if w_test == c:
                candidates_cm[c] = 1.0
                break
    return candidates_cm


def random_profiles(n_profiles, n_v_max=6, n_c_max=4, seed=0):
    """Generate random profiles (impartial culture) with few voters and candidates.

    Parameters
    ----------
    n_profiles : int
    n_v_max : int
    n_c_max : int
    seed : int

    Yields
    ------
    Profile
    """
    rng = np.random.RandomState(seed)
    for _ in range(n_profiles):
        n_v = rng.randint(1, n_v_max + 1)
        n_c = rng.randint(2, n_c_max + 1)
        preferences_rk = np.array([rng.permutation(n_c) for _ in range(n_v)])
        yield Profile(preferences_rk=preferences_rk)


def check_cm_against_brute_force(rule_class, n_profiles=60, n_v_max=6, n_c_max=4, seed=0, n_m_max=None, **rule_options):
    """Check that the CM algorithm of a rule agrees with the brute force on random profiles.

    Parameters
    ----------
    rule_class : class
    n_profiles : int
    n_v_max : int
    n_c_max : int
    seed : int
    n_m_max : int or None
        Cf. :func:`candidates_cm_brute_force`.
    rule_options
        Options passed to the rule (typically ``cm_option``).

    Returns
    -------
    n_undecided : int
        Number of (profile, candidate) pairs where the rule could not decide.

    Raises
    ------
    AssertionError
        If the rule decides CM to True (resp. False) for a candidate whereas the brute force says False (resp. True).
    """
    n_undecided = 0
    for profile in random_profiles(n_profiles, n_v_max=n_v_max, n_c_max=n_c_max, seed=seed):
        expected = candidates_cm_brute_force(rule_class, profile, n_m_max=n_m_max, **rule_options)
        rule = rule_class(**rule_options)(profile)
        found = rule.candidates_cm_
        for c in range(profile.n_c):
            if np.isnan(expected[c]):
                continue
            if np.isnan(found[c]):
                n_undecided += 1
            elif found[c] != expected[c]:
                raise AssertionError(
                    f"{rule_class.__name__}, c = {c}: found {found[c]}, expected {expected[c]}.\n"
                    f"{profile.to_doctest_string()}"
                )
    return n_undecided


def candidates_um_brute_force(rule_class, profile, **rule_options):
    """Decide UM for each candidate by trying all possible (common) ballots of the manipulators.

    Parameters
    ----------
    rule_class : class
        A subclass of :class:`svvamp.Rule`, based on strict rankings.
    profile : Profile
    rule_options
        Options passed to the rule (cf. the module docstring).

    Returns
    -------
    ndarray
        ``candidates_um[c]`` is 1. if the voters who prefer ``c`` to the sincere winner can make ``c`` win by casting
        the same ballot, 0. otherwise.
    """
    rule = rule_class(**rule_options)(profile)
    if not rule.is_based_on_rk:
        raise ValueError(f"{rule_class.__name__} is not based on strict rankings.")
    w = rule.w_
    n_c = profile.n_c
    candidates_um = np.zeros(n_c)
    for c in range(n_c):
        if c == w:
            continue
        manipulators = np.where(rule.v_wants_to_help_c_[:, c])[0]
        if manipulators.shape[0] == 0:
            continue
        for ballot in itertools.permutations(range(n_c)):
            preferences_rk = np.copy(profile.preferences_rk)
            preferences_rk[manipulators, :] = ballot
            w_test = rule._copy(profile=Profile(preferences_rk=preferences_rk, sort_voters=False)).w_
            if w_test == c:
                candidates_um[c] = 1.0
                break
    return candidates_um


def check_um_against_brute_force(rule_class, n_profiles=60, n_v_max=6, n_c_max=4, seed=0, **rule_options):
    """Check that the UM algorithm of a rule agrees with the brute force on random profiles.

    Parameters
    ----------
    rule_class : class
    n_profiles : int
    n_v_max : int
    n_c_max : int
    seed : int
    rule_options
        Options passed to the rule (typically ``um_option``).

    Returns
    -------
    n_undecided : int
        Number of (profile, candidate) pairs where the rule could not decide.

    Raises
    ------
    AssertionError
        If the rule decides UM to True (resp. False) for a candidate whereas the brute force says False (resp. True).
    """
    n_undecided = 0
    for profile in random_profiles(n_profiles, n_v_max=n_v_max, n_c_max=n_c_max, seed=seed):
        expected = candidates_um_brute_force(rule_class, profile, **rule_options)
        rule = rule_class(**rule_options)(profile)
        found = rule.candidates_um_
        for c in range(profile.n_c):
            if np.isnan(found[c]):
                n_undecided += 1
            elif found[c] != expected[c]:
                raise AssertionError(
                    f"{rule_class.__name__}, c = {c}: found {found[c]}, expected {expected[c]}.\n"
                    f"{profile.to_doctest_string()}"
                )
    return n_undecided
