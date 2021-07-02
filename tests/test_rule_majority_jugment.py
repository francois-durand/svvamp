import numpy as np
from svvamp import GeneratorProfileLadder
from svvamp.utils.misc import initialize_random_seeds
from svvamp import RuleMajorityJudgment


def test_min_grade_setter():
    """
        >>> rule = RuleMajorityJudgment()
        >>> rule.min_grade = 0
        >>> rule.min_grade
        0
        >>> rule.min_grade = -42
        >>> rule.min_grade
        -42
        >>> rule.min_grade = -np.inf
        Traceback (most recent call last):
        ValueError: Unknown value for min_grade: -inf
    """
    pass


def test_max_grade_setter():
    """
        >>> rule = RuleMajorityJudgment()
        >>> rule.max_grade = 1
        >>> rule.max_grade
        1
        >>> rule.max_grade = 42
        >>> rule.max_grade
        42
        >>> rule.max_grade = np.inf
        Traceback (most recent call last):
        ValueError: Unknown value for max_grade: inf
    """
    pass


def test_step_grade_setter():
    """
        >>> rule = RuleMajorityJudgment()
        >>> rule.step_grade = 0
        >>> rule.step_grade
        0
        >>> rule.step_grade = 0.1
        >>> rule.step_grade
        0.1
        >>> rule.step_grade = np.inf
        Traceback (most recent call last):
        ValueError: Unknown value for step_grade: inf
    """
    pass


def test_rescale_grades_setter():
    """
        >>> rule = RuleMajorityJudgment()
        >>> rule.rescale_grades = True
        >>> rule.rescale_grades
        True
        >>> rule.rescale_grades = False
        >>> rule.rescale_grades
        False
        >>> rule.rescale_grades = 'unexpected value'
        Traceback (most recent call last):
        ValueError: Unknown value for rescale_grades: unexpected value
    """
    pass


def test_no_rescale_grades():
    """
        >>> initialize_random_seeds()
        >>> profile = GeneratorProfileLadder(n_v=5, n_c=3, n_rungs=5)().profile_
        >>> rule = RuleMajorityJudgment(rescale_grades=False)(profile)
        >>> rule.w_
        0
    """
    pass


def test_step_grade_not_equal_to_zero():
    """
        >>> initialize_random_seeds()
        >>> profile = GeneratorProfileLadder(n_v=5, n_c=3, n_rungs=5)().profile_
        >>> rule = RuleMajorityJudgment(step_grade=0.1)(profile)
        >>> rule.w_
        0
    """
    pass


def test_tm_um_c():
    """
        >>> initialize_random_seeds()
        >>> profile = GeneratorProfileLadder(n_v=5, n_c=3, n_rungs=5)().profile_
        >>> rule = RuleMajorityJudgment()(profile)
        >>> rule.is_tm_c_(0)
        False
        >>> rule.is_um_c_(0)
        False
    """
    pass
