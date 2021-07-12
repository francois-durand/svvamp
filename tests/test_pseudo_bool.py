import pytest
from svvamp.utils.pseudo_bool import pseudo_bool


def test_pseudo_bool():
    with pytest.raises(ValueError):
        pseudo_bool(42)
