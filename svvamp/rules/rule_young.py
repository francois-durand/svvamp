# -*- coding: utf-8 -*-
"""
Created on 27 june 2025, 15:33
Copyright François Durand 2014-2025
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
from svvamp.utils.util_cache import cached_property, DeleteCacheMixin
from svvamp.utils import my_log
from svvamp.utils.pseudo_bool import equal_true
from svvamp.preferences.profile import Profile


class RuleYoung(DeleteCacheMixin, my_log.MyLog):
    """Young method.

    This emulates a part of `Rule` functionalities, but it is not a subclass of `Rule`. The core difference is that
    `self.w_` may be `numpy.nan`, which means that the winner is not known. For that reason, many other methods from
    `Rule` are not implemented here. As of now, the main purpose of this class is to implement `is_cm_` to evaluate
    the coalitional manipulability of Young method.

    Consider that the implementation of `is_cm_` is trustworthy only if voters have no indifference.
    """

    def __init__(self, log_identity='YOUNG'):
        # Log
        super().__init__()
        self.log_identity = log_identity
        # Initialize the computed variables
        self.profile_ = None

    def __call__(self, profile):
        """
        Parameters
        ----------
        profile : Profile
        """
        self.delete_cache(suffix='_')
        self.profile_ = profile
        return self

    @cached_property
    def w_(self):
        """Integer (winning candidate).
        """
        self.mylog("Compute winner", 1)
        if self.profile_.exists_condorcet_winner_rk:
            return self.profile_.condorcet_winner_rk
        else:
            return np.nan

    @cached_property
    def losing_candidates_(self):
        """1d of Integers. List of losing candidates, in a decreasing order of (heuristic) dangerousness
        """
        self.mylog("Compute ordered list of losing candidates", 1)
        if np.isnan(self.w_):
            # If the winner is not known, we cannot compute the losing candidates.
            return np.isnan
        result = np.concatenate((
            np.array(range(0, self.w_), dtype=int),
            np.array(range(self.w_ + 1, self.profile_.n_c), dtype=int)
        ))
        return result[np.argsort(- self.profile_.matrix_duels_ut[result, self.w_], kind='mergesort')]

    # %% Coalition Manipulation (CM)

    @cached_property
    def is_cm_(self):
        """Boolean (or ``numpy.nan``). ``True`` if a CM is possible, ``False`` otherwise. If the algorithm cannot
        decide, then ``numpy.nan``.
        """
        if not self.profile_.exists_weak_condorcet_winner:
            # For any candidate (and in particular the winner), there exists an opponent strictly preferred
            # by more than half of the voters. Hence, it is CM.
            return True
        if np.isnan(self.w_):
            # There is no Condorcet winner (_rk), hence the winner is not clear. However, there is at least one
            # weak Condorcet winner (in the sense of utilities), hence it is not obviously CM.
            # If preferences are strict, this can happen only with an even number of voters, if there are one or several
            # weak Condorcet winners, but none of them is a Condorcet winner. We do not try to cover this edge case
            # for the moment.
            return np.nan
        if self.profile_.exists_semi_resistant_condorcet_winner:
            # TODO: In cases with indifference, check if this condition is correct.
            # If there is a semi-resistant Condorcet winner, then it is not CM
            return False
        # Now we are in the case where the winner is known, it is a Condorcet winner (rk), it is a weak Condorcet
        # winner (in the sense of utilities) and it is not semi-resistant.
        for c in self.losing_candidates_:
            if equal_true(self._is_cm_for_c_(c)):
                return True
        return np.nan

    def _is_cm_for_c_(self, c):
        """
        Check if it is CM in favor of `c`.

        Assume we are in the last case in the code of `is_cm_`: there is a Condorcet winner (rk), it is a weak
        Condorcet winner (in the sense of utilities) and it is not semi-resistant. We try to prove the success of
        a manipulation where all manipulators put `c` first, `w` last, and all the other candidates in between, in an
        arbitrary order.

        Parameters
        ----------
        c: int
            Candidate. Must be different from `self.w_`.

        Returns
        -------
        bool or numpy.nan
            True if it is CM in favor of `c`, False otherwise. Can be `numpy.nan` if the algorithm cannot decide.
        """
        n_manipulators = self.profile_.matrix_duels_ut[c, self.w_]
        score_c_cm_lower_bound = 2 * n_manipulators - 1
        # Note : We could add the sincere voters who rank `c` first (and thus have `c` indifferent with `w`), but
        # this is a very marginal case, so we do not waste computation time on it.

        # Now, compute an upper bound for the score of w. Initialize this value with an easy upper bound
        # obtained against `c` herself:
        score_w_cm_upper_bound = 2 * self.profile_.matrix_duels_rk[self.w_, c] - 1
        # Let us improve this. Let `d` be a third candidate. Among the sincere voters, we have some `w > d` and some
        # `w < d` (but no `w ~ d`, because we consider _rk). We can keep all the voters |w > d|_sincere. On top of
        # that, we will add alpha |d > w| voters (either sincere voters, manipulators, or virtual voters). We must have:
        # |w > d|_sincere > alpha
        # score(w) <= |w > d|_sincere + alpha < 2 |w > d|_sincere <= 2 |w > d|_sincere - 1
        v_sincere = (self.profile_.preferences_ut[:, self.w_] >= self.profile_.preferences_ut[:, c])
        for d in self.losing_candidates_:
            if d == c:
                continue
            v_ranks_w_above_d = (self.profile_.preferences_borda_rk[:, self.w_]
                                 > self.profile_.preferences_borda_rk[:, d])
            new_upper_bound = 2 * np.sum(np.logical_and(v_sincere, v_ranks_w_above_d)) - 1
            score_w_cm_upper_bound = min(score_w_cm_upper_bound, new_upper_bound)
        if not score_c_cm_lower_bound > score_w_cm_upper_bound:
            # We failed to prove that `c` can surpass `w`, but it might still be possible (especially if we have
            # indifference).
            return np.nan

        # We manage to defeat w. Now, we need to check that we defeat all the d's. Let d be a third candidate.
        # Among the sincere voters, we have some `c > d` and some `c < d` (but no `c ~ d`, because we consider _rk).
        # We can keep all the voters |d > c|_sincere. On top of that, we will add alpha |c > d| voters.
        # |d > c|_sincere > alpha
        # score(d) <= |d > c|_sincere + alpha < 2 |d > c|_sincere <= 2 |d > c|_sincere - 1
        for d in self.losing_candidates_:
            if d == c:
                continue
            v_ranks_d_above_c = (self.profile_.preferences_borda_rk[:, d]
                                 > self.profile_.preferences_borda_rk[:, c])
            score_d_upper_bound = 2 * np.sum(np.logical_and(v_sincere, v_ranks_d_above_c)) - 1
            if not score_c_cm_lower_bound > score_d_upper_bound:
                # We failed to prove that `c` can surpass `d`, but it might still be possible (especially if we have
                # indifference).
                return np.nan

        # We managed to prove that `c` can surpass `w` and all the d's.
        return True
