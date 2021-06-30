# -*- coding: utf-8 -*-
"""
Created on 7 dec. 2018, 15:32
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


class RuleIRVAverage(Rule):
    """Instant Runoff Voting based on Average Score (Nanson-like).

    Examples
    --------
        >>> import svvamp
        >>> profile = svvamp.Profile(preferences_rk=[[0, 2, 1], [0, 2, 1], [2, 0, 1], [2, 1, 0], [2, 1, 0]])
        >>> rule = svvamp.RuleIRVAverage()(profile)
        >>> print(rule.scores_)
        [[ 2.  0.  3.]
         [ 2. inf  3.]
         [inf inf  5.]]
        >>> print(rule.candidates_by_scores_best_to_worst_)
        [2 0 1]
        >>> rule.w_
        2

    Notes
    -----
    At each round, all candidates with a Plurality score strictly lower than average are simultaneously eliminated.
    When all remaining candidates have the same Plurality score, the candidate with lowest index is declared the
    winner.

    This method meets MajFav criterion.

    * :meth:`is_cm_`: Non-polynomial or non-exact algorithms from superclass :class:`Rule`.
    * :meth:`is_icm_`: Exact in polynomial time.
    * :meth:`is_im_`: Non-polynomial or non-exact algorithms from superclass :class:`Rule`.
    * :meth:`is_iia`: Non-polynomial or non-exact algorithms from superclass :class:`Rule`.
    * :meth:`is_tm_`: Exact in polynomial time.
    * :meth:`is_um_`: Non-polynomial or non-exact algorithms from superclass :class:`Rule`.
    """

    def __init__(self, **kwargs):
        super().__init__(
            with_two_candidates_reduces_to_plurality=True, is_based_on_rk=True,
            precheck_icm=True,
            log_identity="IRV_AVERAGE", **kwargs
        )
        # TODO The parameter precheck_icm = True was in the original SVVAMP version. Is is useful? It does not hurt
        # much anyway, but we could think about it.

    @cached_property
    def _count_ballots_(self):
        self.mylog("Count ballots", 1)
        scores = []
        candidates_worst_to_best = []
        one_v_might_be_pivotal = False
        # score_temp[c] = plurality score
        # score_temp[c] will be inf when c is eliminated
        score_temp = np.copy(self.profile_.plurality_scores_rk).astype(float)
        score_average = self.profile_.n_v / self.profile_.n_c
        candidates = np.array(range(self.profile_.n_c))
        is_alive = np.ones(self.profile_.n_c, dtype=np.bool)
        while True:  # This is a round
            scores.append(np.copy(score_temp))
            least_score = np.min(score_temp)
            self.mylogv("scores =", scores, 2)
            self.mylogv("score_average =", score_average, 2)
            if least_score == score_average:
                # Then all candidates have the average. This also includes the case where only one candidate is
                # remaining. We eliminate them by decreasing order of index (highest indexes first).
                losing_candidates = np.where(np.isfinite(score_temp))[0]
                candidates_worst_to_best.extend(losing_candidates[::-1])
                if len(losing_candidates) > 1:
                    one_v_might_be_pivotal = True
                    self.mylog("One voter might be pivotal", 2)
                self.mylogv("losing_candidates[::-1] =", losing_candidates[::-1], 2)
                break
            # Remove all candidates with score < average. Lowest score first, highest indexes first.
            while least_score < score_average:
                if least_score + 1 >= score_average:
                    one_v_might_be_pivotal = True
                    self.mylog("One voter might be pivotal", 2)
                losing_candidates = np.where(score_temp == least_score)[0]
                candidates_worst_to_best.extend(losing_candidates[::-1])
                self.mylogv("losing_candidates[::-1] =", losing_candidates[::-1], 2)
                is_alive[losing_candidates] = False
                score_temp[losing_candidates] = np.inf
                least_score = np.min(score_temp)
            if least_score - 1 < score_average:
                one_v_might_be_pivotal = True
                self.mylog("One voter might be pivotal", 2)
            candidates_alive = candidates[is_alive]
            score_temp = np.bincount(
                candidates_alive[np.argmax(self.profile_.preferences_borda_rk[:, is_alive], 1)],
                minlength=self.profile_.n_c
            ).astype(float)
            score_temp[np.logical_not(is_alive)] = np.inf
            score_average = self.profile_.n_v / np.sum(is_alive)
        candidates_by_scores_best_to_worst = np.array(candidates_worst_to_best[::-1])
        w = candidates_by_scores_best_to_worst[0]
        scores = np.array(scores)
        return {'scores': scores, 'w': w, 'one_v_might_be_pivotal': one_v_might_be_pivotal,
                'candidates_by_scores_best_to_worst': candidates_by_scores_best_to_worst}

    @cached_property
    def scores_(self):
        """2d array of integers. ``scores[r, c]`` is the Plurality score of candidate ``c`` at elimination round ``r``.
        By convention, if candidate ``c`` does not participate to round ``r``, then ``scores[r, c] = numpy.inf``.
        """
        return self._count_ballots_['scores']

    @cached_property
    def w_(self):
        return self._count_ballots_['w']

    @cached_property
    def candidates_by_scores_best_to_worst_(self):
        """1d array of integers. Candidates are sorted according to their order of elimination. When several
        candidates are eliminated during the same round, they are sorted by Plurality score at that round (less votes
        are eliminated first) and, in case of a tie, by their index (highest indexes are eliminated first).
        """
        return self._count_ballots_['candidates_by_scores_best_to_worst']

    @cached_property
    def _one_v_might_be_pivotal_(self):
        return self._count_ballots_['one_v_might_be_pivotal']

    @cached_property
    def v_might_im_for_c_(self):
        return np.full((self.profile_.n_v, self.profile_.n_c), self._one_v_might_be_pivotal_)

    @cached_property
    def meets_majority_favorite_c_rk_ctb(self):
        return True


if __name__ == '__main__':
    RuleIRVAverage()(Profile(preferences_ut=np.random.randint(-5, 5, (5, 3)))).demo_()
