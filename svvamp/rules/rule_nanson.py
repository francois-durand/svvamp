# -*- coding: utf-8 -*-
"""
Created on 10 dec. 2018, 15:46
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


class RuleNanson(Rule):
    """Nanson method.

    Examples
    --------
        >>> import svvamp
        >>> profile = svvamp.Profile(preferences_rk=[[0, 2, 1], [0, 2, 1], [2, 0, 1], [2, 1, 0], [2, 1, 0]])
        >>> rule = svvamp.RuleNanson()(profile)
        >>> print(rule.scores_)
        [[ 5.  2.  8.]
         [ 2. inf  3.]
         [inf inf  0.]]
        >>> print(rule.candidates_by_scores_best_to_worst_)
        [2 0 1]
        >>> rule.w_
        2

    Notes
    -----
    At each round, all candidates with a Borda score strictly lower than average are simultaneously eliminated. When
    all remaining candidates have the same Borda score, it means that the matrix of duels (for this subset of
    candidates) has only ties. Then the candidate with lowest index is declared the winner. Since a Condorcet winner
    has always a Borda score higher than average, Nanson method meets the Condorcet criterion.

    * :meth:`is_cm_`: Deciding CM is NP-complete. Non-polynomial or non-exact algorithms from superclass :class:`Rule`.
    * :meth:`is_icm_`: Exact in polynomial time.
    * :meth:`is_im_`: Deciding IM is NP-complete. Non-polynomial or non-exact algorithms from superclass :class:`Rule`.
    * :meth:`is_iia_`: Exact in polynomial time.
    * :meth:`is_tm_`: Exact in polynomial time.
    * :meth:`is_um_`: Non-polynomial or non-exact algorithms from superclass :class:`Rule`.

    References
    ----------
    'Complexity of and algorithms for the manipulation of Borda, Nanson's and Baldwin's voting rules',
    Jessica Davies, George Katsirelos, Nina Narodytska, Toby Walsh and Lirong Xia, 2014.
    """

    def __init__(self, **kwargs):
        super().__init__(
            options_parameters={
                'icm_option': {'allowed': ['exact'], 'default': 'exact'}
            },
            with_two_candidates_reduces_to_plurality=True, is_based_on_rk=True,
            precheck_icm=False,
            log_identity="NANSON", **kwargs
        )

    @cached_property
    def _count_ballots_(self):
        self.mylog("Count ballots", 1)
        scores = []
        candidates_worst_to_best = []
        one_v_might_be_pivotal = False
        # borda_temp[c] will be inf when c is eliminated
        borda_temp = np.copy(self.profile_.borda_score_c_rk).astype(float)
        borda_average = self.profile_.n_v * (self.profile_.n_c - 1) / 2
        while True:  # This is a round
            scores.append(np.copy(borda_temp))
            least_borda = np.min(borda_temp)
            self.mylogv("scores =", scores, 2)
            # self.mylogv("least_borda =", least_borda, 2)
            self.mylogv("borda_average =", borda_average, 2)
            if least_borda == borda_average:
                # Then all candidates have the average. This means that all duels are ties. This also includes the
                # case where only one candidate is remaining. We eliminate them by decreasing order of index (highest
                # indexes first).
                losing_candidates = np.where(np.isfinite(borda_temp))[0]
                candidates_worst_to_best.extend(losing_candidates[::-1])
                if len(losing_candidates) > 1:
                    one_v_might_be_pivotal = True
                    self.mylog("One voter might be pivotal", 2)
                self.mylogv("losing_candidates[::-1] =", losing_candidates[::-1], 2)
                break
            # Remove all candidates with borda score < average. Lowest Borda score first, highest indexes first.
            nb_candidates_alive = np.sum(np.isfinite(borda_temp))
            # (alive at beginning of round)
            borda_for_next_round = np.copy(borda_temp).astype(float)
            borda_average_for_next_round = borda_average
            while least_borda < borda_average:
                if least_borda + (nb_candidates_alive-1) >= borda_average:
                    one_v_might_be_pivotal = True
                    self.mylog("One voter might be pivotal", 2)
                losing_candidates = np.where(borda_temp == least_borda)[0]
                candidates_worst_to_best.extend(losing_candidates[::-1])
                self.mylogv("losing_candidates[::-1] =", losing_candidates[::-1], 2)
                # self.mylogv("candidates_worst_to_best =", candidates_worst_to_best, 2)
                borda_average_for_next_round -= (len(losing_candidates) * self.profile_.n_v / 2)
                borda_for_next_round[losing_candidates] = np.inf
                borda_for_next_round = borda_for_next_round - np.sum(
                    self.profile_.matrix_duels_rk[:, losing_candidates], 1)
                borda_temp[losing_candidates] = np.inf
                least_borda = np.min(borda_temp)
                # self.mylogv("borda_temp =", borda_temp, 2)
                # self.mylogv("least_borda =", least_borda, 2)
                # self.mylogv("borda_average =", borda_average, 2)
            if least_borda - (nb_candidates_alive-1) < borda_average:
                one_v_might_be_pivotal = True
                self.mylog("One voter might be pivotal", 2)
            borda_temp = borda_for_next_round
            borda_average = borda_average_for_next_round
        candidates_by_scores_best_to_worst = np.array(candidates_worst_to_best[::-1])
        w = candidates_by_scores_best_to_worst[0]
        scores = np.array(scores)
        return {'scores': scores, 'w': w, 'candidates_by_scores_best_to_worst': candidates_by_scores_best_to_worst,
                'one_v_might_be_pivotal': one_v_might_be_pivotal}

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
        """1d array of integers. Candidates are sorted according to their order of elimination. When several
        candidates are eliminated during the same round, they are sorted by Borda score at that round and,
        in case of a tie, by their index (highest indexes are eliminated first).
        """
        return self._count_ballots_['candidates_by_scores_best_to_worst']

    @cached_property
    def v_might_im_for_c_(self):
        return np.full((self.profile_.n_v, self.profile_.n_c), self._count_ballots_['one_v_might_be_pivotal'])

    # %% Manipulation criteria of the voting system

    def meets_condorcet_c_rk_ctb(self):
        return True


if __name__ == '__main__':
    RuleNanson()(Profile(preferences_ut=np.random.randint(-5, 5, (5, 3)))).demo_()
