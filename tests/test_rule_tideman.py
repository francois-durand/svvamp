from svvamp import RuleTideman
from tests.cm_brute_force import check_cm_against_brute_force


def test_cm_against_brute_force():
    """The CM algorithms never contradict the brute force (which keeps each voter at her position in the profile)."""
    for cm_option in ("fast", "slow", "very_slow"):
        check_cm_against_brute_force(
            RuleTideman, n_profiles=40, n_v_max=7, n_c_max=4, seed=5, n_m_max=2, cm_option=cm_option
        )
