# -*- coding: utf-8 -*-
"""
Created on 4 dec. 2018, 11:42
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


class RuleBaldwin(Rule):
    """Baldwin method.

    >>> import svvamp
    >>> profile = svvamp.Profile(preferences_rk=[[0, 2, 1], [0, 2, 1], [2, 0, 1], [2, 1, 0], [2, 1, 0]])
    >>> rule = svvamp.RuleBaldwin()(profile)
    >>> print(rule.scores_)
    [[ 5.  2.  8.]
     [ 2. inf  3.]]
    >>> print(rule.candidates_by_scores_best_to_worst_)
    [2 0 1]
    >>> rule.w_
    2

    Each voter provides a strict order of preference. The candidate with lowest Borda score is eliminated. Then the
    new Borda scores are computed. Etc. Ties are broken in favor of lower-index candidates: in case of a tie,
    the candidate with highest index is eliminated.

    Since a Condorcet winner has always a Borda score higher than average, Baldwin method meets the Condorcet
    criterion.

    * :meth:`is_cm_`: Non-polynomial or non-exact algorithms from superclass :class:`Rule`.
    * :meth:`is_icm_`: Exact in polynomial time.
    * :meth:`is_im_`: Deciding IM is NP-complete. Non-polynomial or non-exact algorithms from superclass :class:`Rule`.
    * :meth:`is_iia_`: Exact in polynomial time.
    * :meth:`is_tm_`: Exact in polynomial time.
    * :meth:`is_um_`: Non-polynomial or non-exact algorithms from superclass :class:`Rule`.

    References:

        'Complexity of and algorithms for the manipulation of Borda, Nanson's and Baldwin's voting rules',
        Jessica Davies, George Katsirelos, Nina Narodytska, Toby Walsh and Lirong Xia, 2014.
    """
    def __init__(self, **kwargs):
        super().__init__(
            options_parameters={'icm_option': {'allowed': ['exact'], 'default': 'exact'}},
            with_two_candidates_reduces_to_plurality=True, is_based_on_rk=True,
            precheck_icm=False, log_identity="BALDWIN", **kwargs
        )

    @cached_property
    def _count_ballots_(self):
        """
        :return: a dictionary with ``scores``, ``w``, ``candidates_by_scores_best_to_worst``
        """
        self.mylog("Count ballots", 1)
        scores = np.zeros((self.profile_.n_c - 1, self.profile_.n_c))
        worst_to_best = []
        preferences_borda_temp = self.profile_.preferences_borda_rk.astype(float)
        w = None
        for r in range(self.profile_.n_c - 1):  # Round r
            scores[r, :] = np.sum(preferences_borda_temp, 0)
            loser = np.where(scores[r, :] == np.min(scores[r, :]))[0][-1]  # Tie-breaking: higher index
            worst_to_best.append(loser)
            if r == self.profile_.n_c - 2:
                preferences_borda_temp[0, loser] = np.inf
                w = np.argmin(preferences_borda_temp[0, :])
                worst_to_best.append(w)
                break
            # Prepare for next round
            # If v was ranking c above loser, then c loses 1 point:
            preferences_borda_temp[preferences_borda_temp > preferences_borda_temp[:, loser][:, np.newaxis]] -= 1
            preferences_borda_temp[:, loser] = np.inf
        candidates_by_scores_best_to_worst = np.array(worst_to_best[::-1])
        return {'scores': scores, 'w': w, 'candidates_by_scores_best_to_worst': candidates_by_scores_best_to_worst}

    @cached_property
    def scores_(self):
        """2d array of integers. ``scores[r, c]`` is the Borda score of candidate ``c`` at elimination round ``r``.
        By convention, if candidate ``c`` does not participate to round ``r``, then ``scores[r, c] = numpy.inf``.
        """
        return self._count_ballots_['scores']

    @cached_property
    def w_(self):
        return self._count_ballots_['w']

    @cached_property
    def candidates_by_scores_best_to_worst_(self):
        """1d array of integers. ``candidates_by_scores_best_to_worst[-r]`` is the candidate eliminated at
        elimination round ``r``. By definition / convention, ``candidates_by_scores_best_to_worst[0]`` = :attr:`w_`.
        """
        return self._count_ballots_['candidates_by_scores_best_to_worst']

    @cached_property
    def meets_condorcet_c_rk_ctb(self):
        # Consider vtb case. If ``c`` is a Condorcet winner with vtb and ctb, then she has at worst ties in
        # ``matrix_duels_rk``, so she has at least the average Borda score. Can she be eliminated? For that, it is
        # necessary that all other candidates have exactly the average, i.e. ``matrix_duels_rk`` is a general tie.
        # Since she is Condorcet winner vtb/ctb, she must be candidate 0, so our tie-breaking rule eliminates another
        # candidate. Conclusion: this voting system meets Condorcet criterion vtb/ctb.
        #
        # However, let us consider the following example.
        # preferences_borda_ut:
        # [ 1   0 ]
        # [0.5 0.5]
        # [0.5 0.5]
        # ==> candidate 0 is a relative Condorcet winner.
        # preferences_borda_rk (with vtb):
        # [ 1   0 ]
        # [ 0   1 ]
        # [ 0   1 ]
        # ==> candidate 1 wins.
        # Conclusion: this voting system does not meet Condorcet criterion (ut/rel).
        return True


if __name__ == '__main__':
    RuleBaldwin()(Profile(preferences_ut=np.random.randint(-5, 5, (5, 3)))).demo_(log_depth=3)
