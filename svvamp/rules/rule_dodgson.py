# -*- coding: utf-8 -*-
"""
Created on 15 july 2025, 10:41
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
from math import floor
import numpy as np

from svvamp.utils.util_cache import cached_property, DeleteCacheMixin
from svvamp.utils import my_log
from svvamp.utils.pseudo_bool import equal_true
from svvamp.preferences.profile import Profile


class RuleDodgson(DeleteCacheMixin, my_log.MyLog):
    """Dodgson method.

    This emulates a part of `Rule` functionalities, but it is not a subclass of `Rule`. The core difference is that
    `self.w_` may be `numpy.nan`, which means that the winner is not known. For that reason, many other methods from
    `Rule` are not implemented here. As of now, the main purpose of this class is to implement `is_cm_` to evaluate
    the coalitional manipulability of the Dodgson method.

    Consider that the implementation of `is_cm_` is trustworthy only if voters have no indifference.
    """

    full_name = 'Dodgson'
    abbreviation = 'Dod'

    def __init__(self):
        super().__init__()
        self.log_identity = 'DODGSON'
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

    def log_(self, method_name):
        """Log corresponding to a particular manipulation method.
        """
        if '_cm_' in method_name:
            return "cm_option = fast"
        elif '_xm_' in method_name:
            return "xm_option = exact"
        else:
            raise NotImplementedError()

    @cached_property
    def is_cm_(self):
        """Boolean (or ``numpy.nan``). ``True`` if a CM is possible, ``False`` otherwise. If the algorithm cannot
        decide, then ``numpy.nan``.
        """
        self.mylog("CM: Compute is_cm_", 1)
        if not self.profile_.exists_condorcet_admissible:
            # For any candidate (and in particular the winner), there exists an opponent strictly preferred
            # by more than half of the voters. Hence, it is CM.
            return True
        if np.isnan(self.w_):
            # There is no Condorcet winner (_rk), hence the winner is not clear. However, there is at least one
            # Condorcet-admissible candidate, hence it is not obviously CM.
            # If preferences are strict, this can happen only with an even number of voters, if there are one or several
            # weak Condorcet winners, but none of them is a Condorcet winner. We do not try to cover this edge case
            # for the moment.
            return np.nan
        if self.profile_.exists_resistant_condorcet_winner:
            return False
        # Now we are in the case where the winner is known, it is a Condorcet winner (rk), it is
        # Condorcet-admissible, but it is not resistant.
        result = False
        for c in self.losing_candidates_:
            is_cm_for_c = self._is_cm_for_c_(c)
            if equal_true(is_cm_for_c):
                return True
            if np.isnan(is_cm_for_c):
                result = np.nan
        return result

    def _is_cm_for_c_(self, c):
        """
        Check if it is CM in favor of `c`.

        Assume we are in the last case in the code of `is_cm_`: there is a Condorcet winner (rk), it is Condorcet
        admissible, but it is not resistant. We try to prove the success of the trivial manipulation,
        or to prove the failure of any attempt of coalitional manipulation.

        Parameters
        ----------
        c: int
            Candidate. Must be different from `self.w_`.

        Returns
        -------
        bool or numpy.nan
            True if it is CM in favor of `c`, False otherwise. Can be `numpy.nan` if the algorithm cannot decide.
        """
        self.mylogv("CM: Compute _is_cm_for_c_ for c =", c, 1)
        strict_majority = floor(self.profile_.n_v / 2) + 1
        strict_minority = self.profile_.n_v - strict_majority
        v_sincere = (self.profile_.preferences_ut[:, self.w_] >= self.profile_.preferences_ut[:, c])

        # First, we try to prove that manipulation is impossible by showing that the penalty of `c` will necessarily
        # be higher than the one of `w` (typical case of CM failure).
        # Bound for `c`
        penalty_c_cm_lower_bound = 0
        n_voters_who_are_sincere_and_rank_d_above_c = np.zeros(self.profile_.n_c, dtype=int)
        c_does_not_win_against_some_d = False
        for d in self.losing_candidates_:
            if d == c:
                continue
            n_voters_who_are_sincere_and_rank_d_above_c[d] = np.sum(np.logical_and(
                v_sincere,
                self.profile_.preferences_borda_rk[:, d] > self.profile_.preferences_borda_rk[:, c]
            ))
            if n_voters_who_are_sincere_and_rank_d_above_c[d] > strict_minority:
                self.mylogv("CM: c does not win against d =", d, 2)
                c_does_not_win_against_some_d = True
                penalty_c_cm_lower_bound += n_voters_who_are_sincere_and_rank_d_above_c[d] - strict_minority
        penalty_c_cm_lower_bound += self.profile_.matrix_duels_rk[self.w_, c] - strict_minority
        self.mylogv("CM: penalty_c_cm_lower_bound =", penalty_c_cm_lower_bound, 2)
        # Bound for `w`: Swaps needed to become majority winner, only with sincere voters. This is necessarily possible,
        # because `w` is Condorcet-admissible (hence the manipulators are a minority).
        n_w_first = self.profile_.plurality_scores_rk[self.w_]
        points_needed_for_w_to_be_majority_winner = max(0, strict_majority - n_w_first)
        # Quick check with loose bound
        penalty_w_cm_upper_bound = points_needed_for_w_to_be_majority_winner * (self.profile_.n_c - 1)
        self.mylogv("CM: penalty_w_cm_upper_bound (loose) =", penalty_w_cm_upper_bound, 2)
        if penalty_c_cm_lower_bound > penalty_w_cm_upper_bound:
            self.mylogv("CM: Impossible for c =", c, 2)
            return False
        # Slower check with improved bound
        penalty_w_cm_upper_bound = 0
        l_rank_n_sincere_voters_rank_w_at_this_rank = np.sum(self.profile_.preferences_rk[v_sincere, :] == self.w_, 0)
        for rank in range(1, self.profile_.n_c - 1):
            n_sincere_voters_rank_w_at_this_rank = l_rank_n_sincere_voters_rank_w_at_this_rank[rank]
            if n_sincere_voters_rank_w_at_this_rank >= points_needed_for_w_to_be_majority_winner:
                penalty_w_cm_upper_bound += points_needed_for_w_to_be_majority_winner * rank
                break
            else:
                penalty_w_cm_upper_bound += n_sincere_voters_rank_w_at_this_rank * rank
                points_needed_for_w_to_be_majority_winner -= n_sincere_voters_rank_w_at_this_rank
        self.mylogv("CM: penalty_w_cm_upper_bound (improved) =", penalty_w_cm_upper_bound, 2)
        if penalty_c_cm_lower_bound > penalty_w_cm_upper_bound:
            self.mylogv("CM: Impossible for c =", c, 2)
            return False
        self.mylogv("CM: Failed to prove non-CM for c =", c, 2)

        # Now, we have failed to prove non-CM.
        # If `c` does not win against some `d`, then TM might be possible, but we won't bother with that case and
        # just say that we do not know.
        if c_does_not_win_against_some_d:
            self.mylogv("CM: c does not win against some d => Failed to prove TM for c =", c, 2)
            return np.nan

        # Now, we are in the case where `c` wins against all the `d`'s. We just need to take care of the swaps used to
        # defeat `w`. Note that `penalty_c_tm` is computed exactly in that case.
        penalty_c_tm = 0
        points_needed = self.profile_.matrix_duels_rk[self.w_, c] - strict_minority
        for gap in range(1, self.profile_.n_c):
            n_voters_with_this_gap = np.sum(
                self.profile_.preferences_borda_rk[:, self.w_] == self.profile_.preferences_borda_rk[:, c] + gap
            )
            if n_voters_with_this_gap >= points_needed:
                penalty_c_tm += points_needed * gap
                break
            else:
                penalty_c_tm += n_voters_with_this_gap * gap
                points_needed -= n_voters_with_this_gap
        self.mylogv("CM: penalty_c_tm =", penalty_c_tm, 2)
        # `penalty_c_tm` is also a lower bound of the penalty of `c` for any kind of manipulation. Hence in passing,
        # we get another chance to prove non-CM.
        if penalty_c_tm > penalty_w_cm_upper_bound:
            self.mylogv("CM: Impossible for c =", c, 2)
            return False

        # Now, compute a lower bound for the penalty of the winner `w_`.
        penalty_w_cm_lower_bound = 0
        for d in self.losing_candidates_:
            if d == c:
                continue
            score_d_against_w = self.profile_.n_v - self.profile_.size_bicoalition[c, d]
            penalty_w_cm_lower_bound += max(0, score_d_against_w - strict_minority)
        self.mylogv("CM: penalty_w_cm_lower_bound =", penalty_w_cm_lower_bound, 2)
        if not penalty_c_tm < penalty_w_cm_lower_bound:
            # We failed to prove that `c` can surpass `w`, but it might still be possible.
            self.mylogv("CM: Failed to prove TM for c =", c, 2)
            return np.nan

        # We manage to defeat w. Now, we need to check that we defeat all the d's. Let d be a third candidate.
        # In the case of a trivial manipulation, all duels not implying `c` or `w_` are not modified.
        # So, for candidate `d`:
        # * Against `c`, `d` has np.sum(v_sincere and v_rank_d_before_c).
        # * Against `w`, `d` has np.sum(v_sincere and v_rank_d_before_w) + n_m.
        # * All duels against another `e` are the same as in the original profile.
        for d in self.losing_candidates_:
            if d == c:
                continue
            penalty_d_cm_lower_bound = 0
            for e in range(self.profile_.n_c):
                if e == c:
                    score_d_vs_e = n_voters_who_are_sincere_and_rank_d_above_c[d]
                elif e == self.w_:
                    # np.sum(v_sincere and v_rank_d_before_w) + n_m
                    # = np.sum(v_sincere) - np.sum(v_sincere and v_rank_w_before_d) + n_m
                    # = n_v - np.sum(v_sincere and v_rank_w_before_d)
                    # = n_v - size_bicoalition[c, d]
                    score_d_vs_e = self.profile_.n_v - self.profile_.size_bicoalition[c, d]
                else:
                    score_d_vs_e = self.profile_.matrix_duels_rk[d, e]
                penalty_d_cm_lower_bound += max(0, strict_majority - score_d_vs_e)
            self.mylogv("CM: penalty_d_cm_lower_bound =", penalty_d_cm_lower_bound, 2)
            if not penalty_c_tm < penalty_d_cm_lower_bound:
                # We failed to prove that `c` can surpass `d`, but it might still be possible.
                self.mylogv("CM: Failed to prove TM for c =", c, 2)
                return np.nan
        return True

    # %% Estimation of CM via empirical theta (XM)
    @cached_property
    def theta_critical_(self):
        n_c = self.profile_.n_c
        return (n_c - 2) / (4 * n_c - 5)

    @cached_property
    def is_xm_(self):
        if self.profile_.theta_empirical > self.theta_critical_:
            self.mylog("XM: Theta > theta_c, so XM is False", 2)
            return False
        elif self.profile_.theta_empirical < self.theta_critical_:
            self.mylog("XM: Theta < theta_c, so XM is True", 2)
            return True
        else:
            self.mylog("XM: Theta == theta_c, so XM is unknown", 2)
            return np.nan
