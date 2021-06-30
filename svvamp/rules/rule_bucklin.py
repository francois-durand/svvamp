# -*- coding: utf-8 -*-
"""
Created on 4 dec. 2018, 15:20
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
class RuleBucklin(Rule):
    """Bucklin method.

    Examples
    --------
    >>> import svvamp
    >>> profile = svvamp.Profile(preferences_rk=[[0, 2, 1], [0, 2, 1], [2, 0, 1], [2, 1, 0], [2, 1, 0]])
    >>> rule = svvamp.RuleBucklin()(profile)
    >>> print(rule.scores_)
    [[2. 0. 3.]
     [3. 2. 5.]
     [5. 5. 5.]]
    >>> print(rule.candidates_by_scores_best_to_worst_)
    [2 0 1]
    >>> rule.w_
    2

    Notes
    -----
    At counting round ``r``, all voters who rank candidate ``c`` in ``r``\ :sup:`th` position gives her an additional
    vote. As soon as at least one candidate has more than :attr:`n_v`/2 votes (accrued with previous rounds),
    the candidate with most votes is declared the winner. In case of a tie, the candidate with lowest index wins.

    * :meth:`is_cm_`: Exact in polynomial time.
    * :meth:`is_icm_`: Exact in polynomial time.
    * :meth:`is_im_`: Exact in polynomial time.
    * :meth:`is_iia_`: Non-polynomial or non-exact algorithms from superclass :class:`Rule`.
    * :meth:`is_tm_`: Exact in polynomial time.
    * :meth:`is_um_`: Exact in polynomial time.

    References
    ----------
    'The Majoritarian Compromise in large societies', Arkadii Slinko, 2002.

    'Complexity of Unweighted Coalitional Manipulation under Some Common Voting Rules', Lirong Xia,
    Michael Zuckerman, Ariel D. Procaccia, Vincent Conitzer and Jeffrey S. Rosenschein, 2009.
    """

    def __init__(self, **kwargs):
        super().__init__(
            options_parameters={
                'im_option': {'allowed': ['exact'], 'default': 'exact'},
                'tm_option': {'allowed': ['exact'], 'default': 'exact'},
                'um_option': {'allowed': ['exact'], 'default': 'exact'},
                'icm_option': {'allowed': ['exact'], 'default': 'exact'},
                'cm_option': {'allowed': ['exact'], 'default': 'exact'},
            },
            with_two_candidates_reduces_to_plurality=True, is_based_on_rk=True,
            precheck_icm=False,  # Bucklin does not meet infmc_c_ctb, but precheck on ICM is not interesting anyway.
            log_identity="BUCKLIN", **kwargs
        )

    @cached_property
    def _count_ballots_(self):
        self.mylog("Count ballots", 1)
        scores = np.zeros((self.profile_.n_c, self.profile_.n_c))
        scores_r = np.zeros(self.profile_.n_c, dtype=np.int)
        w = None
        candidates_by_scores_best_to_worst = None
        for r in range(self.profile_.n_c):
            scores_r += np.bincount(self.profile_.preferences_rk[:, r], minlength=self.profile_.n_c)
            scores[r, :] = np.copy(scores_r)
            if w is None:
                w_r = np.argmax(scores_r)
                if scores_r[w_r] > self.profile_.n_v / 2:
                    w = w_r
                    candidates_by_scores_best_to_worst = np.argsort(- scores_r, kind='mergesort')
        return {'scores': scores, 'w': w, 'candidates_by_scores_best_to_worst': candidates_by_scores_best_to_worst}

    @property
    def scores_(self):
        """2d array of integers. ``scores[r, c]`` is the accrued score of candidate ``c`` at elimination round ``r``.
        It is the number of voters who rank ``c`` between 0\ :sup:`th` and ``r``\ :sup:`th` rank on their ballot.

        For information, ballot are still counted after the round where the decision is made (it is used for
        manipulation algorithms).
        """
        return self._count_ballots_['scores']

    @property
    def w_(self):
        """Integer (winning candidate). When at least one candidate has more than :attr:`n_v`/2 votes, the candidate
        with most votes gets elected. In case of a tie, the candidate with highest index wins.
        """
        return self._count_ballots_['w']

    @property
    def candidates_by_scores_best_to_worst_(self):
        """1d array of integers. Candidates are sorted according to their scores during the counting round during
        which at least one candidate reaches majority.

        By definition, ``candidates_by_scores_best_to_worst_[0]`` = :attr:`w_`.
        """
        return self._count_ballots_['candidates_by_scores_best_to_worst']

    @cached_property
    def v_might_im_for_c_(self):
        # If a voter improves ``c`` from last to first on her ballot, then ``c``'s score gain 1 point each round (except
        # last round ``n_c -  1``, but the decision is always done before). Conversely, if a voter bury ``w`` from first
        # to last, then ``w`` loses 1 point each round (except last round ``n_c - 1``). Modifications by one elector
        # are bounded by that. With such modifications, can a challenger ``c`` do better than ``w`` ? I.e., either
        # reach the majority before, or have more votes than ``w`` when the majority is reached?
        pseudo_scores = np.copy(self.scores_)
        pseudo_scores[:, np.array(range(self.profile_.n_c)) != self.w_] += 1
        pseudo_scores[:, self.w_] -= 1
        self.mylogm('Pseudo-scores =', pseudo_scores)
        r = np.argmax(pseudo_scores[self.w_] > self.profile_.n_v / 2)
        c_has_a_chance = np.zeros(self.profile_.n_c)
        for c in range(self.profile_.n_c):
            if c == self.w_:
                continue
            if r != 0:
                if pseudo_scores[r - 1, c] > self.profile_.n_v / 2:
                    c_has_a_chance[c] = True
            if pseudo_scores[r, c] > pseudo_scores[r, self.w_]:
                c_has_a_chance[c] = True
        return np.tile(np.logical_not(c_has_a_chance), (self.profile_.n_v, 1))

    @cached_property
    def meets_majority_favorite_c_rk(self):
        return True
        # N.B.: majority_favorite_c_ctb is not met.

    # %% Individual manipulation (IM)

    def _im_main_work_v_(self, v, c_is_wanted, nb_wanted_undecided, stop_if_true):
        scores_without_v = np.copy(self.scores_)
        for k in range(self.profile_.n_c):
            scores_without_v[range(k, self.profile_.n_c), self.profile_.preferences_rk[v, k]] -= 1
        for c in range(self.profile_.n_c):
            if not c_is_wanted[c]:
                continue
            if not np.isneginf(self._v_im_for_c[v, c]):
                continue
            nb_wanted_undecided -= 1
            # r : round where c will have majority (with the manipulator).
            r = np.where(scores_without_v[:, c] + 1 > self.profile_.n_v / 2)[0][0]
            if r == 0:
                self._v_im_for_c[v, c] = True
                self._candidates_im[c] = True
                self._voters_im[v] = True
                self._is_im = True
                self.mylog("IM found", 3)
                if stop_if_true:
                    return
                if nb_wanted_undecided == 0:
                    return
                continue
            scores_prev = np.copy(scores_without_v[r - 1, :])
            scores_prev[c] += 1
            scores_r = np.copy(scores_without_v[r, :])
            scores_r[c] += 1
            # Obvious problems
            if np.argmax(scores_r) != c:
                # One d has a better score than c!
                self._v_im_for_c[v, c] = False
                if nb_wanted_undecided == 0:
                    return
                continue
            if np.max(scores_prev) > self.profile_.n_v / 2:
                # One d reaches majority before c!
                self._v_im_for_c[v, c] = False
                if nb_wanted_undecided == 0:
                    return
                continue

            # Now, attribute other ranks in manipulator's ballots. For ``d`` to be added safely at rank ``r``
            # (corresponding to last round), we just need that ``d`` will not outperform ``c`` at rank ``r``.
            d_can_be_added = np.less(scores_r + 1, scores_r[c] + (c < np.array(range(self.profile_.n_c))))
            d_can_be_added[c] = False
            # For ``d`` to be added safely before rank ``r``, we need also that ``d`` will not have a majority before
            # round ``r``.
            d_can_be_added_before_last_round = np.logical_and(d_can_be_added, scores_prev + 1 <= self.profile_.n_v / 2)

            # We can conclude.
            self._v_im_for_c[v, c] = (np.sum(d_can_be_added) >= r - 1
                                      and np.sum(d_can_be_added_before_last_round) >= r - 2)
            if self._v_im_for_c[v, c] == True:
                self._candidates_im[c] = True
                self._voters_im[v] = True
                self._is_im = True
                self.mylog("IM found", 3)
                if stop_if_true:
                    return
            else:
                self._v_im_for_c[v, c] = False
            if nb_wanted_undecided == 0:
                return

    # %% Trivial Manipulation (TM)

    # Defined in the superclass Rule.

    # %% Unison manipulation (UM)

    def _um_main_work_c_(self, c):
        n_m = self.profile_.matrix_duels_ut[c, self.w_]
        scores_r = np.zeros(self.profile_.n_c)
        # Manipulators put ``c`` in first position anyway.
        scores_r[c] = n_m
        # Look for the round ``r`` where ``c`` has a majority. Compute how many sincere votes each candidate will
        # have at that time, + the ``n_m`` manipulating votes that we know for ``c``.
        r = None
        scores_prev = None
        for r in range(self.profile_.n_c):
            scores_prev = np.copy(scores_r)
            scores_r += np.bincount(self.profile_.preferences_rk[np.logical_not(self.v_wants_to_help_c_[:, c]), r],
                                    minlength=self.profile_.n_c)
            if scores_r[c] > self.profile_.n_v / 2:  # It is the last round
                if np.argmax(scores_r) != c:
                    # One ``d`` has a better score than ``c``!
                    self._candidates_um[c] = False
                    return
                break
            if np.max(scores_r) > self.profile_.n_v / 2:
                # One d reaches majority before c!
                self._candidates_um[c] = False
                return

        # Now, attribute other ranks in manipulator's ballots. For ``d`` to be added safely at rank ``r``
        # (corresponding to last round), we just need that ``d`` will not outperform ``c`` at rank ``r``.
        d_can_be_added = np.less(scores_r + n_m, scores_r[c] + (c < np.array(range(self.profile_.n_c))))
        d_can_be_added[c] = False
        # For ``d`` to be added safely before rank ``r``, we need also that ``d`` will not have a majority before
        # round ``r``.
        d_can_be_added_before_last_round = np.logical_and(d_can_be_added, scores_prev + n_m <= self.profile_.n_v / 2)

        # We can conclude.
        self._candidates_um[c] = (np.sum(d_can_be_added) >= r - 1
                                  and np.sum(d_can_be_added_before_last_round) >= r - 2)

    # %% Ignorant-Coalition Manipulation (ICM)

    def _icm_main_work_c_exact_(self, c, complete_mode=True):
        # The only question is when we have exactly ``n_v / 2`` manipulators. If the counter-manipulators put ``c``
        # last, then ``c`` cannot be elected (except if there are 2 candidates and ``c == 0``). So exactly ``n_v / 2``
        # manipulators is not enough.
        n_s = self.profile_.n_v - self.profile_.matrix_duels_ut[c, self.w_]
        if self.profile_.n_c == 2 and c == 0:
            self._update_sufficient(self._sufficient_coalition_size_icm, c, n_s,
                                    'ICM: Tie-breaking: sufficient_coalition_size_icm = n_s =')
        else:
            self._update_necessary(self._necessary_coalition_size_icm, c, n_s + 1,
                                   'ICM: Tie-breaking: necessary_coalition_size_icm = n_s + 1 =')

    # %% Coalition Manipulation (CM)

    def _cm_main_work_c_exact_(self, c, optimize_bounds):
        # We do not try to find optimal bounds. We just check whether it is possible to manipulate with the number of
        #  manipulators that we have.
        n_m = self.profile_.matrix_duels_ut[c, self.w_]
        if n_m < self._necessary_coalition_size_cm[c]:
            # This algorithm will not do better (so, this is not a quick escape).
            return
        if n_m >= self._sufficient_coalition_size_cm[c]:
            # Idem.
            return
        scores_r = np.zeros(self.profile_.n_c)
        # Manipulators put ``c`` in first position anyway.
        scores_r[c] = n_m
        # Look for the round ``r`` where ``c`` has a majority. Compute how many sincere votes each candidate will have
        # at that time, + the ``n_m`` manipulating votes that we know for ``c``.
        r = None
        scores_prev = None
        for r in range(self.profile_.n_c):
            scores_prev = np.copy(scores_r)
            scores_r += np.bincount(self.profile_.preferences_rk[np.logical_not(self.v_wants_to_help_c_[:, c]), r],
                                    minlength=self.profile_.n_c)
            if scores_r[c] > self.profile_.n_v / 2:
                break

        votes_can_be_added_before_last_r = np.zeros(self.profile_.n_c)
        votes_can_be_added = np.zeros(self.profile_.n_c)
        one_d_beats_c_anyway = False
        for d in range(self.profile_.n_c):
            if d == c:
                continue
            if scores_r[d] + (d < c) > scores_r[c] or scores_prev[d] > self.profile_.n_v / 2:
                one_d_beats_c_anyway = True
                break
            votes_can_be_added[d] = min(scores_r[c] - (d < c) - scores_r[d], n_m)
            votes_can_be_added_before_last_r[d] = min(np.floor(self.profile_.n_v / 2) - scores_prev[d],
                                                      votes_can_be_added[d])

        if (not one_d_beats_c_anyway and sum(votes_can_be_added) >= (r - 1) * n_m
                and sum(votes_can_be_added_before_last_r) >= (r - 2) * n_m):
            self._update_sufficient(self._sufficient_coalition_size_cm, c, n_m,
                                    'CM: Exact: Manipulation found for n_m manipulators =>\n'
                                    '    sufficient_coalition_size_cm = n_m =')
        else:
            self._update_necessary(self._necessary_coalition_size_cm, c, n_m + 1,
                                   'CM: Exact: Manipulation proven impossible for n_m manipulators =>\n'
                                   '    necessary_coalition_size_cm[c] = n_m + 1 =')


if __name__ == '__main__':
    RuleBucklin()(Profile(preferences_ut=np.random.randint(-5, 5, (5, 3)))).demo_(log_depth=3)
