# -*- coding: utf-8 -*-
"""
Created on 7 dec. 2018, 15:01
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


class RuleICRV(Rule):
    """Instant-Condorcet Runoff Voting (ICRV).

    Examples
    --------
        >>> import svvamp
        >>> profile = svvamp.Profile(preferences_rk=[[0, 2, 1], [0, 2, 1], [2, 0, 1], [2, 1, 0], [2, 1, 0]])
        >>> rule = svvamp.RuleICRV()(profile)
        >>> print(rule.scores_)
        [[1. 0. 2.]]
        >>> print(rule.candidates_by_scores_best_to_worst_)
        [2 0 1]
        >>> rule.w_
        2

    Notes
    -----
    Principle: eliminate candidates as in IRV; stop as soon as there is a Condorcet winner.

    * Even round ``r`` (including round 0): if a candidate ``w`` has only victories against all other non-eliminated
      candidates (i.e. is a Condorcet winner in this subset, in the sense of :attr:`matrix_victories_rk`), then ``w``
      is declared the winner.
    * Odd round ``r``: the candidate who is ranked first (among non-eliminated candidates) by least voters is
    eliminated, like in :class:`RuleIRV`.

    This method meets the Condorcet criterion.

    * :meth:`is_cm_`: Non-polynomial or non-exact algorithms from superclass :class:`Rule`.
    * :meth:`is_icm_`: Exact in polynomial time.
    * :meth:`is_im_`: Non-polynomial or non-exact algorithms from superclass :class:`Rule`.
    * :meth:`~svvamp.Election.not_iia`: Exact in polynomial time.
    * :meth:`is_tm_`: Exact in polynomial time.
    * :meth:`is_um_`: Non-polynomial or non-exact algorithms from superclass :class:`Rule`.

    References
    ----------
    'Four Condorcet-Hare Hybrid Methods for Single-Winner Elections', James Green-Armytage, 2011.

    See Also
    --------
    :class:`RuleExhaustiveBallot`, :class:`RuleIRV`, :class:`RuleIRVDuels`, :class:`RuleCondorcetAbsIRV`,
    :class:`RuleCondorcetVtbIRV`.
    """

    def __init__(self, **kwargs):
        super().__init__(
            options_parameters={
                'icm_option': {'allowed': ['exact'], 'default': 'exact'},
            },
            with_two_candidates_reduces_to_plurality=True, is_based_on_rk=True,
            precheck_icm=False,
            log_identity="ICRV", **kwargs
        )

    # %% Counting the ballots

    @cached_property
    def _counts_ballots_(self):
        self.mylog("Count ballots", 1)
        scores = []
        is_candidate_alive = np.ones(self.profile_.n_c, dtype=np.bool)
        worst_to_best = []
        r = 0
        while True:
            # Here, scores_r is the number of victories (when restricting to alive candidates).
            scores_r = np.full(self.profile_.n_c, np.nan)
            scores_r[is_candidate_alive] = np.sum(
                self.profile_.matrix_victories_rk[is_candidate_alive, :][:, is_candidate_alive], 1)
            scores.append(scores_r)
            best_c = np.nanargmax(scores_r)
            self.mylogv("best_c =", best_c, 3)
            if scores_r[best_c] == self.profile_.n_c - 1 - r:
                w = best_c
                self.mylogm("scores (before) =", scores, 3)
                scores = np.array(scores)
                self.mylogm("scores (after) =", scores, 3)
                candidates_alive = np.array(range(self.profile_.n_c))[is_candidate_alive]
                candidates_by_scores_best_to_worst = np.concatenate((
                    candidates_alive[np.argsort(-scores_r[is_candidate_alive], kind='mergesort')],
                    worst_to_best[::-1]
                )).astype(dtype=np.int)
                return {'scores': scores, 'w': w,
                        'candidates_by_scores_best_to_worst': candidates_by_scores_best_to_worst}

            # Now, scores_r is the Plurality score (IRV-style)
            scores_r = np.full(self.profile_.n_c, np.nan)
            scores_r[is_candidate_alive] = np.sum(
                self.profile_.preferences_borda_rk[:, is_candidate_alive]
                == np.max(self.profile_.preferences_borda_rk[:, is_candidate_alive], 1)[:, np.newaxis],
                0)
            scores.append(scores_r)
            loser = np.where(scores_r == np.nanmin(scores_r))[0][-1]
            self.mylogv("loser =", loser, 3)
            is_candidate_alive[loser] = False
            worst_to_best.append(loser)

            r += 1

    @cached_property
    def w_(self):
        return self._counts_ballots_['w']

    @cached_property
    def scores_(self):
        """2d array.

        For even rounds ``r`` (including round 0), ``scores[r, c]`` is the number of victories for ``c`` in
        :attr:`matrix_victories_rk` (restricted to non-eliminated candidates). Ties count for 0.5.

        For odd rounds ``r``, ``scores[r, c]`` is the number of voters who rank ``c`` first (among non-eliminated
        candidates).
        """
        return self._counts_ballots_['scores']

    @cached_property
    def candidates_by_scores_best_to_worst_(self):
        """1d array of integers.

        Candidates that are not eliminated at the moment a Condorcet winner is detected are sorted by their number of
        victories in :attr:`matrix_victories_rk` (restricted to candidates that are not eliminated at that time).

        Other candidates are sorted in the reverse order of their IRV elimination.
        """
        return self._counts_ballots_['candidates_by_scores_best_to_worst']

    # TODO: self._v_might_im_for_c

    @cached_property
    def meets_majority_favorite_c_rk_ctb(self):
        return True

    @cached_property
    def meets_condorcet_c_rk(self):
        return True


if __name__ == '__main__':
    RuleICRV()(Profile(preferences_ut=np.random.randint(-5, 5, (5, 3)))).demo_()
