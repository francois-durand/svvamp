# -*- coding: utf-8 -*-
"""
Created on 7 dec. 2018, 16:17
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


class RuleIRVDuels(Rule):
    """IRV with elimination duels.

    >>> import svvamp
    >>> profile = svvamp.Profile(preferences_rk=[[0, 2, 1], [0, 2, 1], [2, 0, 1], [2, 1, 0], [2, 1, 0]])
    >>> rule = svvamp.RuleIRVDuels()(profile)
    >>> print(rule.scores_)
    [[ 2.  0.  3.]
     [ 3.  2. nan]
     [ 2. nan  3.]
     [ 2. nan  3.]]
    >>> print(rule.candidates_by_scores_best_to_worst_)
    [2 0 1]
    >>> rule.w_
    2

    Principle: each round, perform a duel between the two least-favorite candidates and eliminate the loser of this
    duel.

    * Even round ``r`` (including round 0): the two non-eliminated candidates who are ranked first (among the
      non-eliminated candidates) by least voters are selected for the elimination duels that is held in round
      ``r + 1``.
    * Odd round ``r``: voters vote for the selected candidate they like most in the duel. The candidate with least
      votes is eliminated.

    This method meets the Condorcet criterion.

    We thank Laurent Viennot for the idea of this voting system.

    * :meth:`is_cm_`: Non-polynomial or non-exact algorithms from superclass :class:`Rule`.
    * :meth:`is_icm_`: Exact in polynomial time.
    * :meth:`is_im_`: Non-polynomial or non-exact algorithms from superclass :class:`Rule`.
    * :meth:`~svvamp.Election.not_iia`: Exact in polynomial time.
    * :meth:`is_tm_`: Exact in polynomial time.
    * :meth:`is_um_`: Non-polynomial or non-exact algorithms from superclass :class:`Rule`.

    .. seealso:: :class:`RuleExhaustiveBallot`, :class:`RuleIRV`, :class:`RuleICRV`, :class:`RuleCondorcetAbsIRV`,
                 :class:`RuleCondorcetVtbIRV`.
    """

    def __init__(self, **kwargs):
        super().__init__(
            options_parameters={
                'icm_option': {'allowed': ['exact'], 'default': 'exact'}
            },
            with_two_candidates_reduces_to_plurality=True, is_based_on_rk=True,
            precheck_icm=False,
            log_identity="IRV_DUELS", **kwargs
        )

    # %% Counting the ballots

    @cached_property
    def _counts_ballots_(self):
        self.mylog("Count ballots", 1)
        scores = np.full((2 * self.profile_.n_c - 2, self.profile_.n_c), np.nan)
        is_candidate_alive = np.ones(self.profile_.n_c, dtype=np.bool)
        worst_to_best = []
        for r in range(self.profile_.n_c - 1):
            # Select the two worst scores
            scores[2*r, is_candidate_alive] = np.sum(
                self.profile_.preferences_borda_rk[:, is_candidate_alive]
                == np.max(self.profile_.preferences_borda_rk[:, is_candidate_alive], 1)[:, np.newaxis],
                0)
            selected_one = np.where(scores[2 * r, :] == np.nanmin(scores[2 * r, :]))[0][-1]
            score_selected_one = scores[2*r, selected_one]
            scores[2*r, selected_one] = np.inf
            selected_two = np.where(scores[2 * r, :] == np.nanmin(scores[2 * r, :]))[0][-1]
            scores[2*r, selected_one] = score_selected_one
            self.mylogv("selected_one =", selected_one, 3)
            self.mylogv("selected_two =", selected_two, 3)
            # Do the duel
            scores[2*r + 1, selected_one] = self.profile_.matrix_duels_rk[selected_one, selected_two]
            scores[2*r + 1, selected_two] = self.profile_.matrix_duels_rk[selected_two, selected_one]
            if self.profile_.matrix_victories_rk_ctb[selected_one, selected_two] == 1:
                loser = selected_two
            else:
                loser = selected_one
            is_candidate_alive[loser] = False
            worst_to_best.append(loser)
        w = np.argmax(is_candidate_alive)
        worst_to_best.append(w)
        candidates_by_scores_best_to_worst = np.array(worst_to_best[::-1])
        return {'scores': scores, 'w': w, 'candidates_by_scores_best_to_worst': candidates_by_scores_best_to_worst}

    @cached_property
    def w_(self):
        return self._counts_ballots_['w']

    @cached_property
    def scores_(self):
        """2d array.

        * For even rounds ``r`` (including round 0), ``scores[r, c]`` is the number of voters who rank ``c`` first
          (among non-eliminated candidates).
        * For odd rounds ``r``, only the two candidates who are selected for the elimination duels get scores.
          ``scores[r, c]`` is the number of voters who vote for ``c`` in this elimination duel.
        """
        return self._counts_ballots_['scores']

    @cached_property
    def candidates_by_scores_best_to_worst_(self):
        """1d array of integers. Candidates are sorted in the reverse order of their elimination.
        """
        return self._counts_ballots_['candidates_by_scores_best_to_worst']

    # TODO: self.v_might_im_for_c

    @cached_property
    def meets_condorcet_c_rk_ctb(self):
        return True


if __name__ == '__main__':
    RuleIRVDuels()(Profile(preferences_ut=np.random.randint(-5, 5, (5, 3)))).demo_()
