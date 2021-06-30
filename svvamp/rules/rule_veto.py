# -*- coding: utf-8 -*-
"""
Created on 11 dec. 2018, 14:06
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


class RuleVeto(Rule):
    """Veto. Also called Antiplurality.

    >>> import svvamp
    >>> profile = svvamp.Profile(preferences_rk=[[0, 2, 1], [0, 2, 1], [2, 0, 1], [2, 1, 0], [2, 1, 0]])
    >>> rule = svvamp.RuleVeto()(profile)
    >>> print(rule.scores_)
    [-2 -3  0]
    >>> print(rule.candidates_by_scores_best_to_worst_)
    [2 0 1]
    >>> rule.w_
    2

    Each voter votes against one candidate (veto). The candidate with least vetos is declared the winner. In case of
    a tie, the tied candidate with lowest index wins. Sincere voters vote against their least-liked candidate.

    * :meth:`is_iia`: Non-polynomial or non-exact algorithms from superclass :class:`Rule`.
    * :meth:`is_cm_`, :meth:`is_icm_`, :meth:`is_im_`, :meth:`is_tm_`, :meth:`is_um_`: Exact in polynomial time.
    """

    def __init__(self, **kwargs):
        super().__init__(
            options_parameters={
                'im_option': {'allowed': ['exact'], 'default': 'exact'},
                'tm_option': {'allowed': ['exact'], 'default': 'exact'},
                'um_option': {'allowed': ['exact'], 'default': 'exact'},
                'icm_option': {'allowed': ['exact'], 'default': 'exact'},
                'cm_option': {'allowed': ['exact'], 'default': 'exact'}
            },
            with_two_candidates_reduces_to_plurality=True, is_based_on_rk=True,
            precheck_um=False, precheck_icm=False, precheck_tm=False,
            log_identity="VETO", **kwargs
        )

    @cached_property
    def ballots_(self):
        """1d array of integers. ``ballots_[v]`` is the candidate on voter ``v``'s ballot.
        """
        self.mylog("Compute ballots", 1)
        return self.profile_.preferences_rk[:, -1]

    @cached_property
    def scores_(self):
        """1d array of integers. ``scores_[c]`` is minus one times the number of vetos against candidate ``c``.
        """
        self.mylog("Compute scores", 1)
        return - np.bincount(self.ballots_, minlength=self.profile_.n_c)

    # %% Independence of Irrelevant Alternatives (IIA)

    def _compute_winner_of_subset_(self, candidates_r):
        self.mylogv("IIA: Compute winner of subset ", candidates_r, 3)
        scores_r = - np.bincount(np.argmin(self.profile_.preferences_borda_rk[:, candidates_r], 1),
                                 minlength=candidates_r.shape[0])
        index_w_r_in_subset = np.argmax(scores_r)
        w_r = candidates_r[index_w_r_in_subset]
        self.mylogv("IIA: Winner =", w_r, 3)
        return w_r

    # %% Individual manipulation (IM)

    def _im_main_work_v_(self, v, c_is_wanted, nb_wanted_undecided, stop_if_true):
        # If voter ``v`` strictly prefers some ``c_test`` to ``w``, let us note that she cannot have voted against
        # ``c_test``. So, the only thing she can do better is to vote against ``w`` (if it is not already the case),
        # because otherwise ``w`` will still keep a better score than ``c_test``. This strategy does not depend on
        # ``c_test``!
        scores_with_v_manip = np.copy(self.scores_)
        # Remove ``v``'s sincere vote:
        scores_with_v_manip[self.ballots_[v]] += 1
        # Vote against ``w`` instead:
        scores_with_v_manip[self.w_] -= 1
        new_winner = np.argmax(scores_with_v_manip)
        self._v_im_for_c[v, :] = False
        if self.v_wants_to_help_c_[v, new_winner]:
            self._v_im_for_c[v, new_winner] = True
            self._candidates_im[new_winner] = True
            self._voters_im[v] = True
            self._is_im = True

    # %% Trivial Manipulation (TM)

    def _tm_main_work_c_(self, c):
        # Sincere voters:
        scores_test = np.array(- np.bincount(
            self.profile_.preferences_rk[np.logical_not(self.v_wants_to_help_c_[:, c]), -1],
            minlength=self.profile_.n_c))
        # Manipulators vote against ``w``.
        # Remark: for Veto, the trivial strategy is far from the best one!
        scores_test[self.w_] -= self.profile_.matrix_duels_ut[c, self.w_]
        w_test = np.argmax(scores_test)
        self._candidates_tm[c] = (w_test == c)

    # %% Unison Manipulation (UM)
    # In their sincere ballots, manipulators were not voting against ``c``, but against ``w`` or another ``d``. If
    # now they vote against some ``d``, then ``w``'s score might get better, while ``c``'s score will not change:
    # this strategy cannot succeed. So, their only hope is to vote against ``w``. This is precisely the trivial
    # strategy!

    @cached_property
    def is_um_(self):
        return self.is_tm_

    def is_um_c_(self, c):
        return self.is_tm_c_(c)

    @cached_property
    def candidates_um_(self):
        return self.candidates_tm_

    # %% Ignorant-Coalition Manipulation (ICM)

    def _icm_main_work_c_(self, c, optimize_bounds):
        # At worst, 'sincere' manipulators may respond by all voting against ``c``. We need to give as many vetos to
        # all other candidates, which makes ``(n_c - 1) * n_s`` (where ``n_s`` = sincere voters). For each candidate
        # of index lower than ``c``, we need to give an additional veto (because of the tie-breaking rule). This
        # makes ``c`` additional manipulators needed. Hence ``sufficient_...[c] = (n_c - 1) * n_s + c``.
        n_s = self.profile_.n_v - self.profile_.matrix_duels_ut[c, self.w_]
        self._sufficient_coalition_size_icm[c] = (self.profile_.n_c - 1) * n_s + c
        self._necessary_coalition_size_icm[c] = self._sufficient_coalition_size_icm[c]

    # %% Coalition Manipulation (CM)

    def _cm_main_work_c_(self, c, optimize_bounds):
        # Sincere voters:
        scores_test = np.array(- np.bincount(
            self.profile_.preferences_rk[np.logical_not(self.v_wants_to_help_c_[:, c]), -1],
            minlength=self.profile_.n_c))
        # Make each other candidate ``d`` have a lower score than ``c``:
        # ``manipulators against d = scores_test[d] - scores_test[c] + (d < c)``
        self._sufficient_coalition_size_cm[c] = np.sum(
            np.maximum(scores_test - scores_test[c] + (np.array(range(self.profile_.n_c)) < c), 0))
        self._necessary_coalition_size_cm[c] = self._sufficient_coalition_size_cm[c]


if __name__ == '__main__':
    RuleVeto()(Profile(preferences_ut=np.random.randint(-5, 5, (5, 3)))).demo_()
