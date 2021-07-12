import numpy as np
from svvamp import RuleMajorityJudgment, Profile


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
        >>> rule = RuleMajorityJudgment(rescale_grades=False)(profile)
        >>> rule.w_
        0
    """
    pass


def test_step_grade_not_equal_to_zero():
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
        >>> rule = RuleMajorityJudgment(step_grade=0.1)(profile)
        >>> rule.w_
        0
    """
    pass


def test_tm_um_c():
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
        >>> rule = RuleMajorityJudgment()(profile)
        >>> rule.is_tm_c_(0)
        False
        >>> rule.is_um_c_(0)
        False
    """
    pass
