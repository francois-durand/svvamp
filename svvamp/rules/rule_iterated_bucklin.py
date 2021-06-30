# -*- coding: utf-8 -*-
"""
Created on 7 dec. 2018, 16:35
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
from svvamp.utils.util_cache import cached_property
from svvamp.preferences.profile import Profile


class RuleIteratedBucklin(Rule):
    """Iterated Bucklin method.

    Examples
    --------
        >>> import svvamp
        >>> profile = svvamp.Profile(preferences_rk=[[0, 2, 1], [0, 2, 1], [2, 0, 1], [2, 1, 0], [2, 1, 0]])
        >>> rule = svvamp.RuleIteratedBucklin()(profile)
        >>> print(rule.scores_)
        [[0.66666667 0.         1.66666667]
         [1.                inf 1.66666667]]
        >>> print(rule.candidates_by_scores_best_to_worst_)
        [2, 0, 1]
        >>> rule.w_
        2

    Notes
    -----
    The candidate with least *adjusted median Borda score* (cf. below) is eliminated. Then the new Borda scores are
    computed. Etc. Ties are broken in favor of lower-index candidates: in case of a tie, the candidate with highest
    index is eliminated.

    Adjusted median Borda score:

        Let ``med_c`` be the median Borda score for candidate ``c``. Let ``x_c`` the number of voters who put a lower
        Borda score to ``c``. Then ``c``'s adjusted median is ``med_c - x_c / (n_v + 1)``.

        If ``med_c > med_d``, then it is also true for the adjusted median. If ``med_c = med_d``, then ``c`` has a
        better adjusted median iff ``x_c < x_d``, i.e. if more voters give to ``c`` the Borda score ``med_c`` or
        higher.

        So, the best candidate by adjusted median is the :class:`~Bucklin` winner. Here, at each round,
        we eliminate the candidate with lowest adjusted median Borda score, which justifies the name of "Iterated
        Bucklin method".

    Unlike Baldwin method (= Iterated Borda), Iterated Bucklin does not meet the Condorcet criterion. Indeed,
    a Condorcet winner may have the (strictly) worst median ranking.

    * :meth:`is_cm_`: Non-polynomial or non-exact algorithms from superclass :class:`Rule`.
    * :meth:`is_icm_`: The algorithm from superclass :class:`Rule` is polynomial and has a window of error of 1
      manipulator.
    * :meth:`is_im_`: Non-polynomial or non-exact algorithms from superclass :class:`Rule`.
    * :meth:`is_iia`: Non-polynomial or non-exact algorithms from superclass :class:`Rule`.
    * :meth:`is_tm_`: Exact in polynomial time.
    * :meth:`is_um_`: Non-polynomial or non-exact algorithms from superclass :class:`Rule`.
    """

    def __init__(self, **kwargs):
        super().__init__(
            with_two_candidates_reduces_to_plurality=True, is_based_on_rk=True,
            log_identity="ITERATED_BUCKLIN", **kwargs
        )

    @cached_property
    def _count_ballots_(self):
        self.mylog("Count ballots", 1)
        scores = np.zeros((self.profile_.n_c - 1, self.profile_.n_c))
        worst_to_best = []
        is_eliminated = np.zeros(self.profile_.n_c, dtype=np.bool)
        preferences_borda_temp = np.copy(self.profile_.preferences_borda_rk)
        for r in range(self.profile_.n_c - 1):
            for c in range(self.profile_.n_c):
                if is_eliminated[c]:
                    scores[r, c] = np.inf
                    continue
                median = np.median(preferences_borda_temp[:, c])
                x = np.sum(np.less(preferences_borda_temp[:, c], median))
                scores[r, c] = median - x / (self.profile_.n_v + 1)
            loser = np.where(scores[r, :] == np.min(scores[r, :]))[0][-1]  # Tie-breaking: last index
            is_eliminated[loser] = True
            worst_to_best.append(loser)
            preferences_borda_temp[
                np.less(preferences_borda_temp, preferences_borda_temp[:, loser][:, np.newaxis])
            ] += 1
        w = np.argmin(is_eliminated)
        worst_to_best.append(w)
        candidates_by_scores_best_to_worst = worst_to_best[::-1]
        return {'scores': scores, 'w': w, 'candidates_by_scores_best_to_worst': candidates_by_scores_best_to_worst}

    @cached_property
    def scores_(self):
        """2d array of integers. ``scores[r, c]`` is the adjusted median Borda score of candidate ``c`` at
        elimination round ``r``.
        """
        return self._count_ballots_['scores']

    @cached_property
    def w_(self):
        return self._count_ballots_['w']

    @cached_property
    def candidates_by_scores_best_to_worst_(self):
        """1d array of integers. Candidates are sorted according to their order of elimination.
        """
        return self._count_ballots_['candidates_by_scores_best_to_worst']

    @cached_property
    def _one_v_might_be_pivotal_(self):
        scores_r = np.zeros(self.profile_.n_c)
        is_eliminated = np.zeros(self.profile_.n_c, dtype=np.bool)
        preferences_borda_temp = np.copy(self.profile_.preferences_borda_rk)
        for r in range(self.profile_.n_c - 1):
            self.mylogv("Pre-testing pivotality: round r =", r, 3)
            loser_sincere = (self.candidates_by_scores_best_to_worst_[-1 - r])
            self.mylogv("Pre-testing pivotality: loser_sincere =", loser_sincere, 3)
            # At worst, one manipulator can put one ``c`` from top to bottom or from bottom to top. Can this change the
            # loser of this round?
            for c in range(self.profile_.n_c):
                if is_eliminated[c]:
                    scores_r[c] = np.inf
                    continue
                if c == loser_sincere:
                    median = np.median(
                        np.concatenate((preferences_borda_temp[:, c], [self.profile_.n_c, self.profile_.n_c])))
                    x = sum(preferences_borda_temp[:, c] < median) - 1
                else:
                    median = np.median(np.concatenate((preferences_borda_temp[:, c], [1, 1])))
                    x = sum(preferences_borda_temp[:, c] < median) + 1
                scores_r[c] = median - x / (self.profile_.n_v + 1)
            loser = np.where(scores_r == scores_r.min())[0][-1]
            self.mylogv("Pre-testing pivotality: loser =", loser, 3)
            if loser != loser_sincere:
                self.mylog("There might be a pivotal voter", 3)
                return True
            is_eliminated[loser] = True
            preferences_borda_temp[
                np.less(preferences_borda_temp, preferences_borda_temp[:, loser][:, np.newaxis])
            ] += 1
        else:
            self.mylog("There is no pivotal voter", 3)
            return False

    @cached_property
    def v_might_im_for_c_(self):
        return np.full((self.profile_.n_v, self.profile_.n_c), self._one_v_might_be_pivotal_)

    @cached_property
    def meets_majority_favorite_c_rk(self):
        return True


if __name__ == '__main__':
    RuleIteratedBucklin()(Profile(preferences_ut=np.random.randint(-5, 5, (5, 3)))).demo_()
