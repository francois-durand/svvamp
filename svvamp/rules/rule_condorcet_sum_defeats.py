# -*- coding: utf-8 -*-
"""
Created on 4 dec. 2018, 16:00
Copyright François Durand 2014-2018
fradurand@gmail.com

This file is part of SVVAMP.

    SVVAMP is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    SVVAMP is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with SVVAMP.  If not, see <http://www.gnu.org/licenses/>.
"""
import numpy as np
from svvamp.rules.rule import Rule
from svvamp.utils.cache import cached_property
from svvamp.preferences.profile import Profile


# noinspection PySimplifyBooleanCheck
class RuleCondorcetSumDefeats(Rule):
    """Condorcet with sum of defeats.

    >>> import svvamp
    >>> profile = svvamp.Profile(preferences_rk=[[0, 2, 1], [0, 2, 1], [2, 0, 1], [2, 1, 0], [2, 1, 0]])
    >>> rule = svvamp.RuleCondorcetSumDefeats()(profile)
    >>> print(rule.scores_)
    [-1. -4. -0.]
    >>> print(rule.candidates_by_scores_best_to_worst_)
    [2 0 1]
    >>> rule.w_
    2

    An *elementary move* consists of reversing a voter's preference about a pair of candidate ``(c, d)`` (without
    demanding that her whole relation of preference stays transitive). The score for candidate ``c`` is minus the
    number of *elementary moves* needed so that ``c`` becomes a Condorcet winner. It is the same principle as
    Dodgson's method, but without looking for a transitive profile.

    In practice:

    .. math::

        \\texttt{scores}[c] = - \\sum_{c \\text{ does not beat } d}\\left(
        \\left\\lfloor\\frac{V}{2}\\right\\rfloor + 1 - \\texttt{matrix_duels_rk}[c, d]
        \\right)

    In particular, for :attr:`n_v` odd:

    .. math::

        \\texttt{scores}[c] = - \\sum_{c \\text{ does not beat } d}\\left(
        \\left\\lceil\\frac{V}{2}\\right\\rceil - \\texttt{matrix_duels_rk}[c, d]
        \\right)

    * :meth:`is_cm_`: Non-polynomial or non-exact algorithms from superclass :class:`Rule`.
    * :meth:`is_icm_`: Algorithm from superclass :class:`Rule`. It is polynomial and has a window of error of 1
      manipulator.
    * :meth:`is_im_`: Non-polynomial or non-exact algorithms from superclass :class:`Rule`.
    * :meth:`is_iia_`: Non-polynomial or non-exact algorithms from superclass :class:`Rule`. If
      :attr:`iia_subset_maximum_size` = 2, it runs in polynomial time and is exact up to ties (which can occur only if
      :attr:`n_v` is even).
    * :meth:`is_tm_`: Exact in polynomial time.
    * :meth:`is_um_`: Non-polynomial or non-exact algorithms from superclass :class:`Rule`.
    """

    def __init__(self, **kwargs):
        super().__init__(
            with_two_candidates_reduces_to_plurality=True, is_based_on_rk=True,
            precheck_icm=False,
            log_identity="CONDORCET_SUM_DEFEATS", **kwargs
        )

    @cached_property
    def scores_(self):
        """1d array of integers.

        .. math::

            \\texttt{scores}[c] = - \\sum_{c \\text{ does not beat } d}\\left(
            \\left\\lfloor\\frac{V}{2}\\right\\rfloor + 1 - \\texttt{matrix_duels_rk}[c, d]
            \\right)
        """
        self.mylog("Compute scores", 1)
        scores = np.zeros(self.profile_.n_c)
        for c in range(self.profile_.n_c):
            scores[c] = - np.sum(np.floor(self.profile_.n_v / 2) + 1
                                 - self.profile_.matrix_duels_rk[c, self.profile_.matrix_victories_rk[:, c] > 0])
        return scores

    @cached_property
    def meets_condorcet_c_rk(self):
        return True

    @cached_property
    def meets_infmc_c_ctb(self):
        return True


if __name__ == '__main__':
    RuleCondorcetSumDefeats()(Profile(preferences_ut=np.random.randint(-5, 5, (5, 3)))).demo_(log_depth=3)
