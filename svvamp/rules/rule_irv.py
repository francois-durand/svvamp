# -*- coding: utf-8 -*-
"""
Created on 12 dec. 2018, 10:22
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
from svvamp.utils.pseudo_bool import neginf_to_zero
from svvamp.rules.rule_exhaustive_ballot import RuleExhaustiveBallot


# noinspection PySimplifyBooleanCheck
class RuleIRV(Rule):
    """Instant-Runoff Voting (IRV). Also known as Single Transferable Voting, Alternative Vote, Hare method.

    >>> import svvamp
    >>> profile = svvamp.Profile(preferences_rk=[[0, 2, 1], [0, 2, 1], [2, 0, 1], [2, 1, 0], [2, 1, 0]])
    >>> rule = svvamp.RuleIRV()(profile)
    >>> print(rule.scores_)
    [[ 2.  0.  3.]
     [ 2. nan  3.]]
    >>> print(rule.candidates_by_scores_best_to_worst_)
    [2 0 1]
    >>> rule.w_
    2

    The candidate who is ranked first by least voters is eliminated. Then we iterate. Ties are broken in favor of
    lower-index candidates: in case of a tie, the tied candidate with highest index is eliminated.

    * :meth:`is_cm_`: Deciding CM is NP-complete.

        * :attr:`cm_option` = ``'fast'``: Polynomial heuristic. Can prove CM but unable to decide non-CM (except in
          rare obvious cases).
        * :attr:`cm_option` = ``'slow'``: Rely on :class:`~svvamp.ExhaustiveBallot`'s exact algorithm. Non-polynomial
          heuristic (:math:`2^{n_c}`). Quite efficient to prove CM or non-CM.
        * :attr:`cm_option` = ``'exact'``: Non-polynomial algorithm (:math:`n_c!`) adapted from Walsh, 2010.

    * :meth:`is_icm_`: Exact in polynomial time.
    * :meth:`is_im_`: Deciding IM is NP-complete.

        * :attr:`im_option` = ``'lazy'``: Lazy algorithm from superclass :class:`Rule`.
        * :attr:`im_option` = ``'exact'``: Non-polynomial algorithm (:math:`n_c!`) adapted from Walsh, 2010.

    * :meth:`is_iia_`: Non-polynomial or non-exact algorithms from superclass :class:`Rule`.
    * :meth:`is_tm_`: Exact in polynomial time.
    * :meth:`is_um_`: Deciding UM is NP-complete.

        * :attr:`um_option` = ``'fast'``: Polynomial heuristic. Can prove UM but unable to decide non-UM (except in
          rare obvious cases).
        * :attr:`um_option` = ``'exact'``: Non-polynomial algorithm (:math:`n_c!`) adapted from Walsh, 2010.

    References:

        'Single transferable vote resists strategic voting', John J. Bartholdi and James B. Orlin, 1991.

        'On The Complexity of Manipulating Elections', Tom Coleman and Vanessa Teague, 2007.

        'Manipulability of Single Transferable Vote', Toby Walsh, 2010.
    """
    # Exceptionally, for this voting system, we establish a pointer from the Profile object, so that the
    # manipulation results can be used by Condorcet-IRV.

    def __init__(self, **kwargs):
        self._fast_algo = None
        self.eb_ = None
        super().__init__(
            options_parameters={
                'um_option': {'allowed': {'fast', 'exact'}, 'default': 'fast'},
                'cm_option': {'allowed': {'fast', 'slow', 'exact'}, 'default': 'fast'},
                'tm_option': {'allowed': ['exact'], 'default': 'exact'},
                'icm_option': {'allowed': {'exact'}, 'default': 'exact'},
                'fast_algo': {'allowed': {'c_minus_max', 'minus_max', 'hardest_first'}, 'default': 'c_minus_max'}
            },
            with_two_candidates_reduces_to_plurality=True, is_based_on_rk=True,
            precheck_um=False, precheck_icm=False,
            log_identity="IRV", **kwargs
        )

    def __call__(self, profile):
        """
        :param profile: A :class:`~svvamp.Profile`.
        """
        # Unplug this irv from the old profile
        if self.profile_ is not None:
            delattr(self.profile_, 'irv')
        # See if new profile has an irv
        if hasattr(profile, 'irv'):
            # Update the profile's irv with my options
            for option_name in self.options_parameters.keys():
                setattr(profile.irv, option_name, getattr(self, option_name))
            # Copy all its inner variables into mine.
            self.__dict__.update(profile.irv.__dict__)
        else:
            self.delete_cache(suffix='_')
        # Bind this irv with the new profile
        self.profile_ = profile
        profile.irv = self
        # Grab the exhaustive ballot of the profile (or create it)
        eb_options = {}
        if self.cm_option in {'slow', 'exact'}:
            eb_options['cm_option'] = 'exact'
        if self.im_option == 'exact':
            eb_options['im_option'] = 'exact'
        if self.um_option == 'exact':
            eb_options['um_option'] = 'exact'
        self.eb_ = RuleExhaustiveBallot(**eb_options)(self.profile_)
        return self

    # %% Dealing with the options

    @property
    def fast_algo(self):
        """String. Algorithm used for CM (resp. UM) when cm_option (resp. um_option) is 'fast'. Mostly for developers.
        """
        return self._fast_algo

    @fast_algo.setter
    def fast_algo(self, value):
        if self._fast_algo == value:
            return
        if value in self.options_parameters['fast_algo']['allowed']:
            self.mylogv("Setting fast_algo =", value, 1)
            self._fast_algo = value
            self.delete_cache(contains='_um_', suffix='_')
            self.delete_cache(contains='_cm_', suffix='_')
        else:
            raise ValueError("Unknown value for option fast_algo: " + format(value))

    @cached_property
    def log_um_(self):
        if self.um_option == 'exact':
            return "um_option = exact"
        else:
            return "um_option = " + self.um_option + ", fast_algo = " + self.fast_algo

    @cached_property
    def log_cm_(self):
        if self.cm_option == 'exact':
            return "cm_option = exact"
        else:
            return ("cm_option = " + self.cm_option + ", fast_algo = " + self.fast_algo +
                    ", " + self.log_icm_ + ", " + self.log_tm_)

    # %% Counting the ballots

    @cached_property
    def ballots_(self):
        """2d array of integers. ``ballots[v, r]`` is the candidate for which voter ``v`` votes at round ``r``.
        """
        return self.eb_.ballots_

    @cached_property
    def w_(self):
        return self.eb_.w_

    @cached_property
    def scores_(self):
        """2d array. ``scores[r, c]`` is the number of voters who vote for candidate ``c`` at round ``r``.

        For eliminated candidates, ``scores[r, c] = numpy.nan``. In contrast, ``scores[r, c] = 0`` means that
        ``c`` is present at round ``r`` but no voter votes for ``c``.
        """
        return self.eb_.scores_

    @cached_property
    def margins(self):
        """2d array. ``margins[r, c]`` is the number of votes that ``c`` must lose to be eliminated at round ``r``
        (all other things being equal). The candidate who is eliminated at round ``r`` is the only one for which
        ``margins[r, c] = 0``.

        For eliminated candidates, ``margins[r, c] = numpy.nan``.
        """
        return self.eb_.margins_

    @cached_property
    def candidates_by_scores_best_to_worst_(self):
        """1d array of integers. ``candidates_by_scores_best_to_worst`` is the list of all candidates in the reverse
        order of their elimination.
        """
        return self.eb_.candidates_by_scores_best_to_worst_

    @cached_property
    def elimination_path(self):
        """1d array of integers. Same as :attr:`~svvamp.ExhaustiveBallot.candidates_by_scores_best_to_worst`,
        but in the reverse order.
        """
        return self.eb_.elimination_path_

    @cached_property
    def v_might_be_pivotal_(self):
        return self.eb_.v_might_be_pivotal_

    @cached_property
    def v_might_im_for_c_(self):
        return self.eb_.v_might_im_for_c_

    # %% Manipulation criteria of the voting system

    @cached_property
    def meets_majority_favorite_c_rk_ctb(self):
        return True

    # %% For these manipulation variables, we identify with Exhaustive Ballot

    @cached_property
    def v_wants_to_help_c_(self):
        return self.eb_.v_wants_to_help_c_

    @cached_property
    def c_has_supporters_(self):
        return self.eb_.c_has_supporters_

    @cached_property
    def losing_candidates_(self):
        return self.eb_.losing_candidates_

    # %% Independence of Irrelevant Alternatives (IIA)

    @cached_property
    def is_iia_(self):
        self.eb_.iia_subset_maximum_size = self.iia_subset_maximum_size
        return self.eb_.is_iia_

    @cached_property
    def is_not_iia_(self):
        self.eb_.iia_subset_maximum_size = self.iia_subset_maximum_size
        return self.eb_.is_not_iia_

    @cached_property
    def example_winner_iia_(self):
        self.eb_.iia_subset_maximum_size = self.iia_subset_maximum_size
        return self.eb_.example_winner_iia_

    @cached_property
    def example_subset_iia_(self):
        self.eb_.iia_subset_maximum_size = self.iia_subset_maximum_size
        return self.eb_.example_subset_iia_

    # %% Individual manipulation (IM)

    def _im_main_work_v_exact_(self, v, c_is_wanted, nb_wanted_undecided, stop_if_true):
        self.mylogv("self._v_im_for_c[v, :] =", self._v_im_for_c[v, :], 3)
        other_voters = (np.array(range(self.profile_.n_v)) != v)
        n_s = self.profile_.n_v - 1
        r = 0
        is_candidate_alive_begin_r = np.zeros((self.profile_.n_c - 1, self.profile_.n_c), dtype=np.bool)
        is_candidate_alive_begin_r[0, :] = np.ones(self.profile_.n_c)
        ballot_m_begin_r = np.array(- np.ones(self.profile_.n_c - 1, dtype=np.int))
        scores_tot_begin_r = np.zeros((self.profile_.n_c - 1, self.profile_.n_c))
        scores_tot_begin_r[0, :] = np.sum(np.equal(
            self.profile_.preferences_borda_rk[other_voters, :],
            np.max(self.profile_.preferences_borda_rk[other_voters, :], 1)[:, np.newaxis]
        ), 0)
        self.mylogv("im_aux_exact: r =", r, 3)
        self.mylogv("im_aux_exact: scores_tot_begin_r[r] =", scores_tot_begin_r[0, :], 3)
        # ``strategy_r[r]`` is False (0) if we keep our ballot, True (1) if we change it to vote for the natural
        # loser. If ``strategy_r == 2``, we have tried everything.
        strategy_r = np.zeros(self.profile_.n_c - 1, dtype=np.int)
        natural_loser_r = np.zeros(self.profile_.n_c - 1, dtype=np.int)
        natural_loser_r[0] = np.where(scores_tot_begin_r[0, :] == np.nanmin(scores_tot_begin_r[0, :]))[0][-1]
        eliminated_d_r = np.zeros(self.profile_.n_c - 1)
        self.mylogv("im_aux_exact: natural_loser_r[r] =", natural_loser_r[r], 3)
        # If ``w`` has too many votes, then manipulation is not possible.
        if scores_tot_begin_r[0, self.w_] + (self.w_ == 0) > n_s + 1 - scores_tot_begin_r[0, self.w_]:
            self.mylog("im_aux_exact: Manipulation impossible by this path (w has too many votes)", 3)
            r = -1
        while True:
            if r < 0:
                self.mylog("im_aux_exact: End of exploration", 3)
                neginf_to_zero(self._v_im_for_c[v, :])
                return
            if strategy_r[r] > 1:
                r -= 1
                self.mylogv("im_aux_exact: Tried everything for round r, go back to r =", r, 3)
                self.mylogv("im_aux_exact: r =", r, 3)
                if r >= 0:
                    strategy_r[r] += 1
                continue
            self.mylogv("im_aux_exact: strategy_r[r] =", strategy_r[r], 3)
            if strategy_r[r] == 0:
                ballot_m_temp = ballot_m_begin_r[r]
                d = natural_loser_r[r]
            else:
                if ballot_m_begin_r[r] != -1:
                    # We cannot change our ballot.
                    self.mylog("im_aux_exact: Cannot change our ballot.", 3)
                    strategy_r[r] += 1
                    continue
                else:
                    ballot_m_temp = natural_loser_r[r]
                    scores_tot_temp = np.copy(scores_tot_begin_r[r, :])
                    scores_tot_temp[ballot_m_temp] += 1
                    self.mylogv("im_aux_exact: scores_tot_temp =", scores_tot_temp, 3)
                    d = np.where(scores_tot_temp == np.nanmin(scores_tot_temp))[0][-1]
                    if d == natural_loser_r[r]:
                        self.mylog("im_aux_exact: Cannot save natural_loser_r.", 3)
                        strategy_r[r] += 1
                        continue
            self.mylogv("im_aux_exact: d =", d, 3)
            eliminated_d_r[r] = d
            if r == self.profile_.n_c - 2:
                is_candidate_alive_end = np.copy(is_candidate_alive_begin_r[r, :])
                is_candidate_alive_end[d] = False
                c = np.argmax(is_candidate_alive_end)
                self.mylogv("im_aux_exact: Winner =", c, 3)
                if np.isneginf(self._v_im_for_c[v, c]):
                    self._v_im_for_c[v, c] = True
                    self._candidates_im[c] = True
                    self._voters_im[v] = True
                    self._is_im = True
                    self.mylogv("IM found for c =", c, 3)
                    if c_is_wanted[c]:
                        if stop_if_true:
                            return
                        nb_wanted_undecided -= 1
                    if nb_wanted_undecided == 0:
                        return  # We know everything we want for this voter
                strategy_r[r] += 1
                continue
                # Calculate scores for next round
            is_candidate_alive_begin_r[r + 1, :] = is_candidate_alive_begin_r[r, :]
            is_candidate_alive_begin_r[r + 1, d] = False
            self.mylogv("im_aux_exact: is_candidate_alive_begin_r[r+1, :] =", is_candidate_alive_begin_r[r + 1, :], 3)
            scores_tot_begin_r[r + 1, :] = np.full(self.profile_.n_c, np.nan)
            scores_tot_begin_r[r + 1, is_candidate_alive_begin_r[r + 1, :]] = (
                np.sum(np.equal(
                    self.profile_.preferences_borda_rk[other_voters, :][:, is_candidate_alive_begin_r[r + 1, :]],
                    np.max(self.profile_.preferences_borda_rk[other_voters, :][:, is_candidate_alive_begin_r[r + 1, :]],
                           1)[:, np.newaxis]
                ), 0))
            self.mylogv("im_aux_exact: scores_s_begin_r[r+1, :] =", scores_tot_begin_r[r + 1, :], 3)
            if ballot_m_temp == d:
                ballot_m_begin_r[r + 1] = -1
            else:
                ballot_m_begin_r[r + 1] = ballot_m_temp
            self.mylogv("im_aux_exact: ballot_m_begin_r[r+1] =", ballot_m_begin_r[r + 1], 3)
            if ballot_m_begin_r[r + 1] != -1:
                scores_tot_begin_r[r + 1, ballot_m_begin_r[r + 1]] += 1
            self.mylogv("im_aux_exact: scores_tot_begin_r[r+1, :] =", scores_tot_begin_r[r + 1, :], 3)

            # If an opponent has too many votes, then manipulation is not possible.
            if scores_tot_begin_r[r + 1, self.w_] + (self.w_ == 0) > n_s + 1 - scores_tot_begin_r[r + 1, self.w_]:
                self.mylog("im_aux_exact: Manipulation impossible by this path (w will have too many votes)", 3)
                strategy_r[r] += 1
                continue

            # Update other variables for next round
            strategy_r[r + 1] = 0
            natural_loser_r[r + 1] = np.where(
                scores_tot_begin_r[r + 1, :] == np.nanmin(scores_tot_begin_r[r + 1, :])
            )[0][-1]
            r += 1
            self.mylogv("im_aux_exact: r =", r, 3)
            self.mylogv("im_aux_exact: natural_loser_r[r] =", natural_loser_r[r], 3)

    def _im_preliminary_checks_general_subclass_(self):
        if np.all(np.equal(self._v_im_for_c, False)):
            return
        if self.im_option == "exact":
            # In that case, we check Exhaustive Ballot first.
            self.eb_.im_option = "exact"
            if self.eb_.is_im_()[0] == False:
                self.mylog("IM impossible (since it is impossible for Exhaustive Ballot)", 2)
                self._v_im_for_c[:] = False
                # Other variables will be updated in ``_im_preliminary_checks_general``.

    def _im_preliminary_checks_v_subclass_(self, v):
        # Pretest based on Exhaustive Ballot
        if self.im_option == "exact":
            if np.any(np.isneginf(self._v_im_for_c[v, :])):
                # ``self.eb_.im_option = exact``
                candidates_im_v = self.eb_.is_im_v_with_candidates_(v)[2]
                self.mylogv("IM: Preliminary checks: EB._v_im_for_c[v, :] =", candidates_im_v, 3)
                self._v_im_for_c[v, candidates_im_v == False] = False

    # %% Trivial Manipulation (TM)

    @cached_property
    def is_tm_(self):
        # self.eb_.tm_option = self.tm_option
        return self.eb_.is_tm_

    def is_tm_c_(self, c):
        # self.eb_.tm_option = self.tm_option
        return self.eb_.is_tm_c_(c)

    @cached_property
    def candidates_tm_(self):
        # self.eb_.tm_option = self.tm_option
        return self.eb_.candidates_tm_

    @cached_property
    def example_path_tm_(self):
        # self.eb_.tm_option = self.tm_option
        return self.eb_.example_path_tm_

    @cached_property
    def sufficient_coalition_size_tm_(self):
        # self.eb_.tm_option = self.tm_option
        return self.eb_.sufficient_coalition_size_tm_

    def example_path_tm_c_(self, c):
        # self.eb_.tm_option = self.tm_option
        return self.eb_.example_path_tm_c_(c)

    def sufficient_coalition_size_tm_c_(self, c):
        # self.eb_.tm_option = self.tm_option
        return self.eb_.sufficient_coalition_size_tm_c_(c)

    # %% Ignorant-Coalition Manipulation (ICM)

    @cached_property
    def is_icm_(self):
        # self.eb_.icm_option = self.icm_option
        return self.eb_.is_icm_

    def is_icm_c_(self, c):
        # self.eb_.icm_option = self.icm_option
        return self.eb_.is_icm_c_(c)

    def is_icm_c_with_bounds_(self, c):
        # self.eb_.icm_option = self.icm_option
        return self.eb_.is_icm_c_with_bounds_(c)

    @cached_property
    def candidates_icm_(self):
        # self.eb_.icm_option = self.icm_option
        return self.eb_.candidates_icm_

    @cached_property
    def necessary_coalition_size_icm_(self):
        # self.eb_.icm_option = self.icm_option
        return self.eb_.necessary_coalition_size_icm_

    @cached_property
    def sufficient_coalition_size_icm_(self):
        # self.eb_.icm_option = self.icm_option
        return self.eb_.sufficient_coalition_size_icm_

    # %% Unison manipulation (UM)
    #
    # TODO: implement UM slow (use EB exact, but not IRV exact).

    @cached_property
    def _um_is_initialized_general_subclass_(self):
        self._example_path_um = {c: None for c in range(self.profile_.n_c)}
        return True

    def _um_aux_fast(self, c, n_m, preferences_borda_s):
        """Fast algorithm used for UM.

        Arguments:
        :param c: Integer. Candidate for which we want to manipulate.
        :param n_m: Integer. Number of manipulators.
        :param preferences_borda_s: 2d integer. Preferences of the sincere voters (in Borda format).
        :return: tuple ``(manip_found_fast, example_path_fast)``.

            * ``manip_found_fast``: Boolean. Whether a manipulation was found or not.
            * ``example_path_fast``: An example of elimination path that realizes the manipulation with ``n_m``
              manipulators. ``example_path_fast[k]`` is the ``k``-th candidate eliminated. If no manipulation is
              found, ``example_path`` is NaN.
        """
        # At each round, we examine all candidates ``d`` that we can eliminate. If several ``d`` are possible,
        # we choose the one that maximize a heuristic parameter denoted ``situation_c``.
        candidates = np.array(range(self.profile_.n_c))
        n_s = preferences_borda_s.shape[0]
        example_path_fast = []
        is_candidate_alive = np.ones(self.profile_.n_c, dtype=np.bool)
        # Sincere scores (eliminated candidates will have nan)
        scores_s = np.sum(np.equal(preferences_borda_s, np.max(preferences_borda_s, 1)[:, np.newaxis]), 0)
        # Manipulators' ballot: if blocked, index of the candidate. Else, -1
        ballot_m = -1
        # Total scores (eliminated candidates will have nan)
        scores_tot = scores_s
        for r in range(self.profile_.n_c - 1):
            self.mylogv("um_aux_fast: Round r =", r, 3)
            self.mylogv("um_aux_fast: scores_s =", scores_s, 3)
            self.mylogv("um_aux_fast: ballot_m =", ballot_m, 3)
            self.mylogv("um_aux_fast: scores_tot =", scores_tot, 3)
            # If an opponent has too many votes, then manipulation is not possible.
            max_score = np.nanmax(scores_tot[candidates != c])
            most_serious_opponent = np.where(scores_tot == max_score)[0][0]
            if max_score + (most_serious_opponent < c) > n_s + n_m - max_score:
                self.mylogv("um_aux_fast: most_serious_opponent =", most_serious_opponent, 3)
                self.mylog("um_aux_fast: Manipulation impossible by this path (an opponent has too many votes)", 3)
                return False, np.nan
            # Initialize the loop on ``d``
            best_d = -1  # ``d`` which is eliminated with the 'best' strategy
            best_situation_for_c = -np.inf
            ballot_free_r = False  # Will be updated
            scores_s_end_r = np.nan
            ballot_m_end_r = -1  # Will be updated
            scores_tot_end_r = np.nan
            natural_loser = np.where(scores_tot == np.nanmin(scores_tot))[0][-1]
            self.mylogv("um_aux_fast: natural_loser =", natural_loser, 3)
            for vote_for_natural_loser in [False, True]:
                # Which candidate will lose?
                if not vote_for_natural_loser:
                    # In fact, it means that we do not change ``ballot_m``. It includes the case where we vote
                    # already for ``natural_loser``.
                    self.mylog("um_aux_fast: Strategy: keep our ballot", 3)
                    ballot_m_temp = ballot_m
                    d = natural_loser
                else:
                    if ballot_m != -1:
                        self.mylog("um_aux_fast: No other strategy (cannot change our ballot).", 3)
                        continue
                    else:
                        self.mylog("um_aux_fast: Strategy: vote for natural_loser", 3)
                        scores_tot_temp = np.copy(scores_tot)
                        ballot_m_temp = natural_loser
                        scores_tot_temp[natural_loser] += n_m
                        d = np.where(scores_tot_temp == np.nanmin(scores_tot_temp))[0][-1]
                        self.mylogv("um_aux_fast: ballot_m_temp =", ballot_m_temp, 3)
                        self.mylogv("um_aux_fast: scores_tot_temp =", scores_tot_temp, 3)
                self.mylogv("um_aux_fast: d =", d, 3)
                if d == c:
                    self.mylog("um_aux_fast: This eliminates c", 3)
                    continue
                # Compute heuristic: "Situation" for ``c``
                # Now we compute the tables at beginning of next round
                if r == self.profile_.n_c - 2:
                    best_d = d
                    break
                is_candidate_alive_temp = np.copy(is_candidate_alive)
                is_candidate_alive_temp[d] = False
                scores_s_temp = np.full(self.profile_.n_c, np.nan)
                scores_s_temp[is_candidate_alive_temp] = np.sum(np.equal(
                    preferences_borda_s[:, is_candidate_alive_temp],
                    np.max(preferences_borda_s[:, is_candidate_alive_temp], 1)[:, np.newaxis]
                ), 0)
                if ballot_m_temp == d:
                    ballot_m_temp = -1
                ballot_free_temp = (ballot_m_temp == -1)
                scores_tot_temp = np.copy(scores_s_temp)
                if ballot_m_temp != -1:
                    scores_tot_temp[ballot_m_temp] += n_m
                if self.fast_algo == "c_minus_max":
                    situation_for_c = scores_s_temp[c] - np.nanmax(scores_tot_temp[candidates != c])
                elif self.fast_algo == "minus_max":
                    situation_for_c = - np.nanmax(scores_tot_temp[candidates != c])
                elif self.fast_algo == "hardest_first":
                    situation_for_c = (ballot_m_temp != -1)
                else:
                    raise ValueError("Unknown fast algorithm: " + format(self.fast_algo))
                self.mylogv("um_aux_fast: scores_s_temp =", scores_s_temp, 3)
                self.mylogv("um_aux_fast: ballot_m_temp =", ballot_m_temp, 3)
                self.mylogv("um_aux_fast: scores_tot_temp =", scores_tot_temp, 3)
                self.mylogv("um_aux_fast: situation_for_c =", situation_for_c, 3)
                # Is the the best ``d`` so far?
                # Lexicographic comparison on three criteria (highest ``situation``, lowest number of manipulators,
                # lowest index).
                if [situation_for_c, ballot_free_temp, -d] > [best_situation_for_c, ballot_free_r, -best_d]:
                    best_d = d
                    best_situation_for_c = situation_for_c
                    ballot_free_r = ballot_free_temp
                    scores_s_end_r, scores_s_temp = scores_s_temp, None
                    ballot_m_end_r = ballot_m_temp
                    scores_tot_end_r, scores_tot_temp = scores_tot_temp, None
            self.mylogv("um_aux_fast: best_d =", best_d, 3)
            if best_d == -1:
                return False, np.nan
            # Update variables for next round
            example_path_fast.append(best_d)
            is_candidate_alive[best_d] = False
            scores_s, scores_s_end_r = scores_s_end_r, None
            ballot_m = ballot_m_end_r
            scores_tot, scores_tot_end_r = scores_tot_end_r, None
        self.mylog("um_aux_fast: Conclusion: manipulation found", 3)
        example_path_fast.append(c)
        self.mylogv("um_aux_fast: example_path_fast =", example_path_fast, 3)
        return True, np.array(example_path_fast)

    def _um_aux_exact(self, c, n_m, preferences_borda_s):
        """Exact algorithm used for UM.

        :param c: Integer. Candidate for which we want to manipulate.
        :param n_m: Integer. Number of manipulators.
        :param preferences_borda_s: 2d integer. Preferences of the sincere voters (in Borda format).
        :return: tuple ``(manip_found_exact, example_path)``.

            * ``manip_found_exact``: Boolean. Whether a manipulation was found or not.
            * ``example_path``: An example of elimination path that realizes the manipulation with ``n_m`` manipulators.
              ``example_path[k]`` is the ``k``-th candidate eliminated. If the manipulation is impossible,
              ``example_path`` is NaN.
        """
        candidates = np.array(range(self.profile_.n_c))
        n_s = preferences_borda_s.shape[0]
        r = 0
        is_candidate_alive_begin_r = np.zeros((self.profile_.n_c - 1, self.profile_.n_c), dtype=np.bool)
        is_candidate_alive_begin_r[0, :] = np.ones(self.profile_.n_c)
        ballot_m_begin_r = np.array(- np.ones(self.profile_.n_c - 1, dtype=np.int))
        scores_tot_begin_r = np.zeros((self.profile_.n_c - 1, self.profile_.n_c))
        scores_tot_begin_r[0, :] = np.sum(np.equal(
            preferences_borda_s, np.max(preferences_borda_s, 1)[:, np.newaxis]
        ), 0)
        self.mylogv("um_aux_exact: r =", r, 3)
        self.mylogv("um_aux_exact: scores_tot_begin_r[r] =", scores_tot_begin_r[0, :], 3)
        # If an opponent has too many votes, then manipulation is not possible.
        max_score = np.nanmax(scores_tot_begin_r[0, candidates != c])
        most_serious_opponent = np.where(scores_tot_begin_r[0, :] == max_score)[0][0]
        if max_score + (most_serious_opponent < c) > n_s + n_m - max_score:
            self.mylogv("um_aux_exact: most_serious_opponent =", most_serious_opponent, 3)
            self.mylog("um_aux_exact: Manipulation impossible by this path (an opponent has too many votes)", 3)
            r = -1
        # ``strategy_r[r]`` is False if we keep our ballot, True if we change it to vote for the natural loser. If
        # ``strategy_r == 2``, we have tried everything.
        strategy_r = np.zeros(self.profile_.n_c - 1, dtype=np.int)
        natural_loser_r = np.zeros(self.profile_.n_c - 1, dtype=np.int)
        natural_loser_r[0] = np.where(scores_tot_begin_r[0, :] == np.nanmin(scores_tot_begin_r[0, :]))[0][-1]
        eliminated_d_r = np.zeros(self.profile_.n_c - 1)
        self.mylogv("um_aux_exact: natural_loser_r[r] =", natural_loser_r[r], 3)
        while True:
            if r < 0:
                self.mylog("um_aux_exact: End of exploration", 3)
                return False, np.nan
            if strategy_r[r] > 1:
                r -= 1
                self.mylogv("um_aux_exact: Tried everything for round r, go back to r =", r, 3)
                self.mylogv("um_aux_exact: r =", r, 3)
                if r >= 0:
                    strategy_r[r] += 1
                continue
            self.mylogv("um_aux_exact: strategy_r[r] =", strategy_r[r], 3)
            if strategy_r[r] == 0:
                ballot_m_temp = ballot_m_begin_r[r]
                d = natural_loser_r[r]
            else:
                if ballot_m_begin_r[r] != -1:
                    # We cannot change our ballot.
                    self.mylog("um_aux_exact: Cannot change our ballot.", 3)
                    strategy_r[r] += 1
                    continue
                else:
                    ballot_m_temp = natural_loser_r[r]
                    scores_tot_temp = np.copy(scores_tot_begin_r[r, :])
                    scores_tot_temp[ballot_m_temp] += n_m
                    self.mylogv("um_aux_exact: scores_tot_temp =", scores_tot_temp, 3)
                    d = np.where(scores_tot_temp == np.nanmin(scores_tot_temp))[0][-1]
                    if d == natural_loser_r[r]:
                        self.mylog("um_aux_exact: Cannot save natural_loser_r.", 3)
                        strategy_r[r] += 1
                        continue
            self.mylogv("um_aux_exact: d =", d, 3)
            if d == c:
                self.mylog("um_aux_exact: This eliminates c.", 3)
                strategy_r[r] += 1
                continue
            eliminated_d_r[r] = d
            if r == self.profile_.n_c - 2:
                example_path = np.concatenate((eliminated_d_r, np.array([c])))
                self.mylog("um_aux_exact: UM found", 3)
                self.mylogv("um_aux_exact: example_path =", example_path, 3)
                return True, example_path

            # Calculate scores for next round
            is_candidate_alive_begin_r[r + 1, :] = is_candidate_alive_begin_r[r, :]
            is_candidate_alive_begin_r[r + 1, d] = False
            self.mylogv("um_aux_exact: is_candidate_alive_begin_r[r+1, :] =", is_candidate_alive_begin_r[r + 1, :], 3)
            scores_tot_begin_r[r + 1, :] = np.full(self.profile_.n_c, np.nan)
            scores_tot_begin_r[r + 1, is_candidate_alive_begin_r[r + 1, :]] = np.sum(np.equal(
                preferences_borda_s[:, is_candidate_alive_begin_r[r + 1, :]],
                np.max(preferences_borda_s[:, is_candidate_alive_begin_r[r + 1, :]], 1)[:, np.newaxis]
            ), 0)
            self.mylogv("um_aux_exact: scores_s_begin_r[r+1, :] =", scores_tot_begin_r[r + 1, :], 3)
            if ballot_m_temp == d:
                ballot_m_begin_r[r + 1] = -1
            else:
                ballot_m_begin_r[r + 1] = ballot_m_temp
            self.mylogv("um_aux_exact: ballot_m_begin_r[r+1] =", ballot_m_begin_r[r + 1], 3)
            if ballot_m_begin_r[r + 1] != -1:
                scores_tot_begin_r[r + 1, ballot_m_begin_r[r + 1]] += n_m
            self.mylogv("um_aux_exact: scores_tot_begin_r[r+1, :] =", scores_tot_begin_r[r + 1, :], 3)

            # If an opponent has too many votes, then manipulation is not possible.
            max_score = np.nanmax(scores_tot_begin_r[r + 1, candidates != c])
            most_serious_opponent = np.where(scores_tot_begin_r[r + 1, :] == max_score)[0][0]
            if max_score + (most_serious_opponent < c) > n_s + n_m - max_score:
                self.mylogv("um_aux_exact: most_serious_opponent =", most_serious_opponent, 3)
                self.mylog(
                    "um_aux_exact: Manipulation impossible by this path (an opponent will have too many votes)", 3)
                strategy_r[r] += 1
                continue

            # Update other variables for next round
            strategy_r[r + 1] = 0
            natural_loser_r[r + 1] = np.where(
                scores_tot_begin_r[r + 1, :] == np.nanmin(scores_tot_begin_r[r + 1, :])
            )[0][-1]
            r += 1
            self.mylogv("um_aux_exact: r =", r, 3)
            self.mylogv("um_aux_exact: natural_loser_r[r] =", natural_loser_r[r], 3)

    def _um_preliminary_checks_general_subclass_(self):
        if np.all(np.equal(self._candidates_um, False)):
            return
        if self.um_option == "exact":
            # In that case, we check Exhaustive Ballot first.
            self.eb_.um_option = "exact"
            if self.eb_.is_um_()[0] == False:
                self.mylog("UM impossible (since it is impossible for Exhaustive Ballot)", 2)
                self._candidates_um[:] = False
                # Other variables will be updated in ``_um_preliminary_checks_general``.

    def _um_main_work_c_(self, c):
        exact = (self.um_option == "exact")
        n_m = self.profile_.matrix_duels_ut[c, self.w_]
        manip_found_fast, example_path_fast = self._um_aux_fast(
            c, n_m,
            preferences_borda_s=self.profile_.preferences_borda_rk[np.logical_not(self.v_wants_to_help_c_[:, c]), :])
        self.mylogv("UM: manip_found_fast =", manip_found_fast, 3)
        if manip_found_fast:
            self._candidates_um[c] = True
            if self._example_path_um[c] is None:
                self._example_path_um[c] = example_path_fast
            return
        if not exact:
            self._candidates_um[c] = np.nan
            return

        # From this point, we have necessarily the 'exact' option (and have not found a manipulation for ``c`` yet).
        if self.eb_.candidates_um_c_(c) == False:
            self.mylog("UM impossible for c (since it is impossible for Exhaustive Ballot)", 2)
            self._candidates_um[c] = False
            return
        manip_found_exact, example_path_exact = self._um_aux_exact(
            c, n_m,
            preferences_borda_s=self.profile_.preferences_borda_rk[np.logical_not(self.v_wants_to_help_c_[:, c]), :])
        self.mylogv("UM: manip_found_exact =", manip_found_exact)
        if manip_found_exact:
            self._candidates_um[c] = True
            if self._example_path_um[c] is None:
                self._example_path_um[c] = example_path_exact
        else:
            self._candidates_um[c] = False

    # %% Coalition Manipulation (CM)

    @cached_property
    def example_path_cm_(self):
        _ = self.candidates_cm_
        return self._example_path_cm

    def example_path_cm_c_(self, c):
        _ = self.is_cm_c_(c)
        return self._example_path_cm[c]

    @cached_property
    def _cm_is_initialized_general_subclass_(self):
        self._example_path_cm = {c: None for c in range(self.profile_.n_c)}
        # Rule: each time ``self._sufficient_coalition_size_cm[c]`` is decreased, the corresponding elimination path
        # must be stored.
        return True

    def _cm_aux_fast(self, c, n_max, preferences_borda_s):
        """Fast algorithm used for CM.

        Arguments:
        :param c: Integer. Candidate for which we want to manipulate.
        :param n_max: Integer. Maximum number of manipulators allowed.

            * CM, complete and exact --> put the current value of ``sufficient_coalition_size[c] - 1`` (we want to find
              the best value for ``sufficient_coalition_size[c]``, even if it exceeds the number of manipulators)
            * CM, otherwise --> put the number of manipulators.

        :param preferences_borda_s: 2d integer. Preferences of the sincere voters (in Borda format).
        :return: tuple ``(n_manip_fast, example_path_fast)``.

            * ``n_manip_fast``: Integer or inf. If a manipulation is found, a sufficient number of manipulators is
              returned. If no manipulation is found, it is +inf.
            * ``example_path_fast``: An example of elimination path that realizes the manipulation with
              ``n_manip_fast`` manipulators. ``example_path_fast[k]`` is the ``k``-th candidate eliminated. If no
              manipulation is found, ``example_path`` is NaN.
        """
        # At each round, we examine all candidates ``d`` that we can eliminate. If several ``d`` are possible,
        # we choose the one that maximize a heuristic parameter denoted ``situation_c``.
        candidates = np.array(range(self.profile_.n_c))
        n_s = preferences_borda_s.shape[0]
        n_manip_fast = 0
        example_path_fast = []
        is_candidate_alive = np.ones(self.profile_.n_c, dtype=np.bool)
        # Sincere scores (eliminated candidates will have nan)
        scores_s = np.sum(np.equal(preferences_borda_s, np.max(preferences_borda_s, 1)[:, np.newaxis]), 0)
        # Manipulators' scores (eliminated candidates will have 0)
        scores_m = np.zeros(self.profile_.n_c)
        # Total scores (eliminated candidates will have nan)
        scores_tot = scores_s
        for r in range(self.profile_.n_c - 1):
            self.mylogv("cm_aux_fast: Round r =", r, 3)
            self.mylogv("cm_aux_fast: scores_s =", scores_s, 3)
            self.mylogv("cm_aux_fast: scores_m =", scores_m, 3)
            self.mylogv("cm_aux_fast: scores_tot =", scores_tot, 3)
            # If an opponent has too many votes, then manipulation is not possible.
            max_score = np.nanmax(scores_tot[candidates != c])
            most_serious_opponent = np.where(scores_tot == max_score)[0][0]
            if max_score + (most_serious_opponent < c) > n_s + n_max - max_score:
                self.mylogv("cm_aux_fast: most_serious_opponent =", most_serious_opponent, 3)
                self.mylog("cm_aux_fast: Manipulation impossible by this path (an opponent has too many votes)", 3)
                n_manip_fast = np.inf  # by convention
                return n_manip_fast, np.nan
            # Initialize the loop on ``d``
            best_d = -1
            best_situation_for_c = -np.inf
            n_manip_r = np.inf
            scores_s_end_r = np.nan
            scores_m_end_r = np.nan
            scores_tot_end_r = np.nan
            for d in candidates[is_candidate_alive]:
                if d == c:
                    continue
                self.mylogv("cm_aux_fast: d =", d, 3)
                # Is it possible to eliminate ``d`` now?
                scores_m_new = np.zeros(self.profile_.n_c)
                scores_m_new[is_candidate_alive] = np.maximum(
                    0, scores_tot[d] - scores_tot[is_candidate_alive] + (candidates[is_candidate_alive] > d))
                self.mylogv("cm_aux_fast: scores_m_new =", scores_m_new, 3)
                scores_m_tot_d = scores_m + scores_m_new
                n_manip_d = np.sum(scores_m_tot_d)
                self.mylogv("cm_aux_fast: n_manip_d =", n_manip_d, 3)
                if n_manip_d > n_max:
                    self.mylogv("cm_aux_fast: n_manip_d > n_max =", n_max, 3)
                    continue
                # Compute heuristic: "Situation" for ``c``
                if r == self.profile_.n_c - 2:
                    best_d = d
                    n_manip_r = n_manip_d
                    break
                is_candidate_alive_temp = np.copy(is_candidate_alive)
                is_candidate_alive_temp[d] = False
                scores_s_temp = np.full(self.profile_.n_c, np.nan)
                scores_s_temp[is_candidate_alive_temp] = np.sum(np.equal(
                    preferences_borda_s[:, is_candidate_alive_temp],
                    np.max(preferences_borda_s[:, is_candidate_alive_temp], 1)[:, np.newaxis]
                ), 0)
                scores_m_temp = np.copy(scores_m_tot_d)
                scores_m_temp[d] = 0
                scores_tot_temp = scores_s_temp + scores_m_temp
                if self.fast_algo == "c_minus_max":
                    situation_for_c = scores_s_temp[c] - np.nanmax(scores_tot_temp[candidates != c])
                elif self.fast_algo == "minus_max":
                    situation_for_c = - np.nanmax(scores_tot_temp[candidates != c])
                elif self.fast_algo == "hardest_first":
                    situation_for_c = n_manip_d
                else:
                    raise ValueError("Unknown fast algorithm: " + format(self.fast_algo))
                self.mylogv("cm_aux_fast: scores_s_temp =", scores_s_temp, 3)
                self.mylogv("cm_aux_fast: scores_m_temp =", scores_m_temp, 3)
                self.mylogv("cm_aux_fast: scores_tot_temp =", scores_tot_temp, 3)
                self.mylogv("cm_aux_fast: situation_for_c =", situation_for_c, 3)
                # Is the the best ``d`` so far? Lexicographic comparison on three criteria (highest 'situation',
                # lowest number of manipulators, lowest index).
                if [situation_for_c, -n_manip_d, -d] > [best_situation_for_c, -n_manip_r, -best_d]:
                    best_d = d
                    best_situation_for_c = situation_for_c
                    n_manip_r = n_manip_d
                    scores_s_end_r, scores_s_temp = scores_s_temp, None
                    scores_m_end_r, scores_m_temp = scores_m_temp, None
                    scores_tot_end_r, scores_tot_temp = scores_tot_temp, None
            self.mylogv("cm_aux_fast: best_d =", best_d, 3)
            self.mylogv("cm_aux_fast: n_manip_r =", n_manip_r, 3)
            # Update variables for next round
            n_manip_fast = max(n_manip_fast, n_manip_r)
            if n_manip_fast > n_max:
                n_manip_fast = np.inf  # by convention
                break
            example_path_fast.append(best_d)
            is_candidate_alive[best_d] = False
            scores_s, scores_s_end_r = scores_s_end_r, None
            scores_m, scores_m_end_r = scores_m_end_r, None
            scores_tot, scores_tot_end_r = scores_tot_end_r, None
        self.mylogv("cm_aux_fast: Conclusion: n_manip_fast =", n_manip_fast, 3)
        if n_manip_fast <= n_max:
            example_path_fast.append(c)
            return n_manip_fast, np.array(example_path_fast)
        else:
            return n_manip_fast, np.nan

    def _cm_aux_slow(self, suggested_path, preferences_borda_s):
        """'Slow' algorithm used for CM. Checks only if suggested_path works.

        :param suggested_path: A suggested path of elimination.
        :param preferences_borda_s: 2d integer. Preferences of the sincere voters (in Borda format).
        :return: Integer, ``n_manip_slow``. Number of manipulators needed to manipulate with ``suggested_path``.
        """
        candidates = np.array(range(self.profile_.n_c))
        is_candidate_alive = np.ones(self.profile_.n_c, dtype=np.bool)
        # Manipulators' scores (eliminated candidates will have 0)
        scores_m = np.zeros(self.profile_.n_c)
        n_manip_slow = 0
        for r in range(self.profile_.n_c - 1):
            # Sincere scores (eliminated candidates will have nan)
            scores_s = np.full(self.profile_.n_c, np.nan)
            scores_s[is_candidate_alive] = np.sum(np.equal(
                preferences_borda_s[:, is_candidate_alive],
                np.max(preferences_borda_s[:, is_candidate_alive], 1)[:, np.newaxis]
            ), 0)
            # Total scores (eliminated candidates will have nan)
            scores_tot = scores_s + scores_m
            self.mylogv("cm_aux_slow: Round r =", r, 3)
            self.mylogv("cm_aux_slow: scores_s =", scores_s, 3)
            self.mylogv("cm_aux_slow: scores_m =", scores_m, 3)
            self.mylogv("cm_aux_slow: scores_tot =", scores_tot, 3)
            # Let us manipulate
            d = suggested_path[r]
            scores_m_new = np.zeros(self.profile_.n_c)
            scores_m_new[is_candidate_alive] = np.maximum(
                0, scores_tot[d] - scores_tot[is_candidate_alive] + (candidates[is_candidate_alive] > d))
            self.mylogv("cm_aux_slow: scores_m_new =", scores_m_new, 3)
            scores_m = scores_m + scores_m_new
            n_manip_r = np.sum(scores_m)
            self.mylogv("cm_aux_slow: n_manip_r =", n_manip_r, 3)
            # Prepare for next round
            is_candidate_alive[d] = False
            scores_m[d] = 0
            n_manip_slow = max(n_manip_slow, n_manip_r)
        self.mylogv("cm_aux_slow: Conclusion: n_manip_slow =", n_manip_slow, 3)
        return n_manip_slow

    def _cm_aux_exact(self, c, n_max, n_min, optimize_bounds, suggested_path, preferences_borda_s):
        """Exact algorithm used for CM.

        :param c: Integer. Candidate for which we want to manipulate.
        :param n_max: Integer. Maximum number of manipulators allowed.

            * CM, optimize_bounds and exact --> put the current value of ``sufficient_coalition_size[c] - 1`` (we want
              to find the best value for ``sufficient_coalition_size[c]``, even if it exceeds the number of
              manipulators)
            * CM, otherwise --> put the number of manipulators.

        :param n_min: Integer. When we know that ``n_min`` manipulators are needed (necessary coalition size).
        :param optimize_bounds: Boolean. True iff we need to continue, even after a manipulation is found.
        :param suggested_path: A suggested path of elimination.
        :param preferences_borda_s: 2d integer. Preferences of the sincere voters (in Borda format).
        :return: a tuple ``(n_manip_final, example_path, quick_escape)``.

            * ``n_manip_final``: Integer or +inf. If manipulation is impossible with ``<= n_max`` manipulators, it is
              +inf. If manipulation is possible (with ``<= n_max``):

                * If ``optimize_bounds`` is True, it is the minimal number of manipulators.
                * Otherwise, it is a number of manipulators that allow this manipulation (not necessarily minimal).

            * ``example_path``: An example of elimination path that realizes the manipulation with ``n_manip_final``
              manipulators. ``example_path[k]`` is the ``k``-th candidate eliminated. If the manipulation is impossible,
              ``example_path`` is NaN.
            * ``quick_escape``: Boolean. True if we get out without optimizing ``n_manip_final``.
        """
        candidates = np.array(range(self.profile_.n_c))
        n_s = preferences_borda_s.shape[0]
        n_max_updated = n_max  # Maximal number of manipulators allowed
        n_manip_final = np.inf  # Result: number of manipulators finally used
        example_path = np.nan  # Result: example of elimination path
        r = 0
        is_candidate_alive_begin_r = np.zeros((self.profile_.n_c - 1, self.profile_.n_c), dtype=np.bool)
        is_candidate_alive_begin_r[0, :] = np.ones(self.profile_.n_c)
        n_manip_used_before_r = np.zeros(self.profile_.n_c - 1, dtype=np.int)
        scores_m_begin_r = np.zeros((self.profile_.n_c - 1, self.profile_.n_c))
        scores_tot_begin_r = np.zeros((self.profile_.n_c - 1, self.profile_.n_c))
        scores_tot_begin_r[0, :] = np.sum(np.equal(
            preferences_borda_s, np.max(preferences_borda_s, 1)[:, np.newaxis]
        ), 0)
        # ``suggested_path_r[r]`` is a list with all opponents (``candidates != c``) who are alive at the beginning
        # of ``r``, given in a suggested order of elimination.
        self.mylogv('cm_aux_exact: suggested_path =', suggested_path, 3)
        self.mylogv('cm_aux_exact: c =', c, 3)
        suggested_path_r = {0: suggested_path[suggested_path != c]}
        # ``index_in_path_r[r]`` is the index of the candidate we eliminate at round ``r`` in the list
        # ``suggested_path_r[r]``.
        index_in_path_r = np.zeros(self.profile_.n_c - 1, dtype=np.int)
        self.mylogv("cm_aux_exact: r =", r, 3)
        # If an opponent has too many votes, then manipulation is not possible.
        max_score = np.nanmax(scores_tot_begin_r[0, candidates != c])
        most_serious_opponent = np.where(scores_tot_begin_r[0, :] == max_score)[0][0]
        if max_score + (most_serious_opponent < c) > n_s + n_max_updated - max_score:
            self.mylogv("cm_aux_exact: scores_tot_begin_r =", scores_tot_begin_r[0, :], 3)
            self.mylogv("cm_aux_exact: most_serious_opponent =", most_serious_opponent, 3)
            self.mylog("cm_aux_exact: Manipulation impossible by this path (an opponent has too many votes)", 3)
            r = -1
        while True:
            if r < 0:
                self.mylog("cm_aux_exact: End of exploration", 3)
                return n_manip_final, np.array(example_path), False
            if index_in_path_r[r] >= suggested_path_r[r].shape[0] or n_manip_used_before_r[r] > n_max_updated:
                # The second condition above may happen in ``optimize_bounds`` exact mode, if we have found a
                # solution and updated ``n_max_updated``.
                r -= 1
                self.mylogv("cm_aux_exact: Tried everything for round r, go back to r =", r, 3)
                self.mylogv("cm_aux_exact: r =", r, 3)
                if r >= 0:
                    index_in_path_r[r] += 1
                continue
            d = suggested_path_r[r][index_in_path_r[r]]
            self.mylogv("cm_aux_exact: suggested_path_r[r] =", suggested_path_r[r], 3)
            self.mylogv("cm_aux_exact: index_in_path_r[r] =", index_in_path_r[r], 3)
            self.mylogv("cm_aux_exact: d =", d, 3)

            # What manipulators are needed to make ``d`` lose?
            self.mylogv("cm_aux_exact: scores_tot_begin_r[r, :] =", scores_tot_begin_r[r, :], 3)
            scores_m_new_r = np.zeros(self.profile_.n_c)
            scores_m_new_r[is_candidate_alive_begin_r[r, :]] = np.maximum(
                0,
                scores_tot_begin_r[r, d] - scores_tot_begin_r[r, is_candidate_alive_begin_r[r, :]]
                + (candidates[is_candidate_alive_begin_r[r, :]] > d))
            scores_m_end_r = scores_m_begin_r[r, :] + scores_m_new_r
            n_manip_r_and_before = max(n_manip_used_before_r[r], np.sum(scores_m_end_r))
            self.mylogv("cm_aux_exact: scores_m_new_r =", scores_m_new_r, 3)
            self.mylogv("cm_aux_exact: scores_m_end_r =", scores_m_end_r, 3)
            self.mylogv("cm_aux_exact: n_manip_r_and_before =", n_manip_r_and_before, 3)

            if n_manip_r_and_before > n_max_updated:
                self.mylog("cm_aux_exact: Cannot eliminate d, try another one.", 3)
                index_in_path_r[r] += 1
                continue

            if r == self.profile_.n_c - 2:
                n_manip_final = n_manip_r_and_before
                example_path = []
                for r in range(self.profile_.n_c - 1):
                    example_path.append(suggested_path_r[r][index_in_path_r[r]])
                example_path.append(c)
                self.mylog("cm_aux_exact: CM found", 3)
                self.mylogv("cm_aux_exact: n_manip_final =", n_manip_final, 3)
                self.mylogv("cm_aux_exact: example_path =", example_path, 3)
                if n_manip_final == n_min:
                    self.mylogv("cm_aux_exact: End of exploration: it is not possible to do better than n_min =",
                                n_min, 3)
                    return n_manip_final, np.array(example_path), False
                if not optimize_bounds:
                    return n_manip_final, np.array(example_path), True
                n_max_updated = n_manip_r_and_before - 1
                self.mylogv("cm_aux_exact: n_max_updated =", n_max_updated, 3)
                index_in_path_r[r] += 1
                continue

            # Calculate scores for next round
            n_manip_used_before_r[r + 1] = n_manip_r_and_before
            is_candidate_alive_begin_r[r + 1, :] = is_candidate_alive_begin_r[r, :]
            is_candidate_alive_begin_r[r + 1, d] = False
            self.mylogv("cm_aux_exact: is_candidate_alive_begin_r[r+1, :] =", is_candidate_alive_begin_r[r + 1, :], 3)
            scores_tot_begin_r[r + 1, :] = np.full(self.profile_.n_c, np.nan)
            scores_tot_begin_r[r + 1, is_candidate_alive_begin_r[r + 1, :]] = (
                np.sum(np.equal(
                    preferences_borda_s[:, is_candidate_alive_begin_r[r + 1, :]],
                    np.max(preferences_borda_s[:, is_candidate_alive_begin_r[r + 1, :]], 1)[:, np.newaxis]
                ), 0))
            self.mylogv("cm_aux_exact: scores_s_begin_r[r+1, :] =", scores_tot_begin_r[r + 1, :], 3)
            scores_m_begin_r[r + 1, :] = scores_m_end_r
            scores_m_begin_r[r + 1, d] = 0
            self.mylogv("cm_aux_exact: scores_m_begin_r[r+1, :] =", scores_m_begin_r[r + 1, :], 3)
            scores_tot_begin_r[r + 1, :] += scores_m_begin_r[r + 1, :]
            self.mylogv("cm_aux_exact: scores_tot_begin_r[r+1, :] =", scores_tot_begin_r[r + 1, :], 3)

            # If an opponent has too many votes, then manipulation is not possible.
            max_score = np.nanmax(scores_tot_begin_r[r + 1, candidates != c])
            most_serious_opponent = np.where(scores_tot_begin_r[r + 1, :] == max_score)[0][0]
            if max_score + (most_serious_opponent < c) > n_s + n_max_updated - max_score:
                self.mylogv("cm_aux_exact: most_serious_opponent =", most_serious_opponent, 3)
                self.mylog(
                    "cm_aux_exact: Manipulation impossible by this path (an opponent will have too many votes)", 3)
                index_in_path_r[r] += 1
                continue

            # Update other variables for next round
            suggested_path_r[r + 1] = suggested_path_r[r][suggested_path_r[r][:] != d]
            index_in_path_r[r + 1] = 0
            r += 1
            self.mylogv("cm_aux_exact: r =", r, 3)

    def _cm_preliminary_checks_general_subclass_(self):
        if self.cm_option == "slow" or self.cm_option == 'exact':
            # In that case, we check Exhaustive Ballot first
            self.eb_.cm_option = "exact"
            if self.eb_.is_cm_ == False:
                self.mylog("CM impossible (since it is impossible for Exhaustive Ballot)", 2)
                self._is_cm = False
                self._candidates_cm[:] = False
                self._cm_was_computed_with_candidates = True

    def _cm_preliminary_checks_c_(self, c, optimize_bounds):
        # A mandatory additional precheck, to ensure that ``_example_path_cm[c]`` is updated if
        # ``sufficient_coalition_size_cm[c]`` has been updated. We use the following syntax (instead of
        # ``_cm_preliminary_checks_c_subclass``) because we want the test on TM here to be done, even if another one
        # succeeded.
        super()._cm_preliminary_checks_c_(c, optimize_bounds)
        # As a test for sufficient size, this one is better (lower) than all the other ones in
        # ``_cm_preliminary_checks_c``. So, as soon as one of them updates sufficient size, this one will provide an
        # example of path.
        if self.sufficient_coalition_size_tm_c_(c) <= self._sufficient_coalition_size_cm[c]:
            # The <= is not a typo.
            self._update_sufficient(self._sufficient_coalition_size_cm, c, self.sufficient_coalition_size_tm_c_(c),
                                    'CM: Preliminary checks: TM improved => \n    sufficient_coalition_size_cm[c] = ')
            self.mylogv('CM: Preliminary checks: Update _example_path_cm[c] = _example_path_tm[c] =',
                        self.example_path_tm_c_(c), 3)
            self._example_path_cm[c] = self.example_path_tm_c_(c)
        n_m = self.profile_.matrix_duels_ut[c, self.w_]  # Number of manipulators
        if not optimize_bounds and n_m >= self._sufficient_coalition_size_cm[c]:
            return
        if not optimize_bounds and self._necessary_coalition_size_cm[c] > n_m:
            return
        # Another pretest, based on Exhaustive Ballot
        if self.cm_option == "slow" or self.cm_option == "exact":
            self.eb_.cm_option = "exact"
            if optimize_bounds:
                self.eb_.is_cm_c_with_bounds_(c)
            else:
                # This will at least tell us when EB.CM is impossible for ``c``, and store an example path for EB when
                # EB.CM is possible for ``c``.
                self.eb_.is_cm_c_(c)
            # noinspection PyProtectedMember
            self._update_necessary(self._necessary_coalition_size_cm, c, self.eb_._necessary_coalition_size_cm[c],
                                   'CM: Preliminary checks: Use EB =>\n'
                                   '    necessary_coalition_size_cm[c] = EB._necessary_coalition_size_cm[c] =')

    def _cm_main_work_c_(self, c, optimize_bounds):
        exact = (self.cm_option == "exact")
        slow = (self.cm_option == "slow")
        fast = (self.cm_option == "fast")
        if optimize_bounds and exact:
            n_max = self._sufficient_coalition_size_cm[c] - 1
        else:
            n_max = self.profile_.matrix_duels_ut[c, self.w_]
        self.mylogv("CM: n_max =", n_max, 3)
        if fast and self._necessary_coalition_size_cm[c] > n_max:
            self.mylog("CM: Fast algorithm will not do better than what we already know", 3)
            return False
        n_manip_fast, example_path_fast = self._cm_aux_fast(
            c, n_max,
            preferences_borda_s=self.profile_.preferences_borda_rk[np.logical_not(self.v_wants_to_help_c_[:, c]), :])
        self.mylogv("CM: n_manip_fast =", n_manip_fast, 3)
        if n_manip_fast < self._sufficient_coalition_size_cm[c]:
            self._sufficient_coalition_size_cm[c] = n_manip_fast
            self._example_path_cm[c] = example_path_fast
            self.mylogv('CM: Update sufficient_coalition_size_cm[c] = n_manip_fast =', n_manip_fast, 3)
        if fast:
            # With fast algo, we stop here anyway. It is not a "quick escape" (if we'd try again with
            # ``optimize_bounds``, we would not try better).
            return False

        # From this point, we have necessarily the 'slow' or 'exact' option
        if self._sufficient_coalition_size_cm[c] == self._necessary_coalition_size_cm[c]:
            return False
        if not optimize_bounds and self.profile_.matrix_duels_ut[c, self.w_] >= self._sufficient_coalition_size_cm[c]:
            # This is a quick escape: since we have the option 'exact', if we come back with ``optimize_bounds``,
            # we will try to be more precise.
            return True

        # Either we're with ``optimize_bounds`` (and might have succeeded), or without (and we have failed).
        n_max_updated = min(n_manip_fast - 1, n_max)
        self.mylogv("CM: n_max_updated =", n_max_updated, 3)

        # EB should always suggest an elimination path. But just as precaution, we use the fact that we know that
        # ``self._example_path_cm[c]`` provides a path (thanks to 'improved' TM).
        if self.eb_.example_path_cm_c_(c) is None:
            suggested_path = self._example_path_cm[c]
            self.mylog('***************************************************', 0)
            self.mylog('CM: WARNING: EB did not provide an elimination path', 0)
            self.mylog('***************************************************', 0)
            self.mylogv('CM: Use self._example_path_cm[c] =', suggested_path, 3)
        else:
            suggested_path = self.eb_.example_path_cm_c_(c)
            self.mylogv('CM: Use eb_.example_path_cm_c_(c) =', suggested_path, 3)
        if slow:
            n_manip_slow = self._cm_aux_slow(
                suggested_path,
                preferences_borda_s=self.profile_.preferences_borda_rk[np.logical_not(self.v_wants_to_help_c_[:, c]), :]
            )
            self.mylogv("CM: n_manip_slow =", n_manip_slow, 3)
            if n_manip_slow < self._sufficient_coalition_size_cm[c]:
                self._sufficient_coalition_size_cm[c] = n_manip_slow
                self._example_path_cm[c] = suggested_path
                self.mylogv('CM: Update sufficient_coalition_size_cm[c] = n_manip_slow =')
            return False
        else:
            n_manip_exact, example_path_exact, quick_escape = self._cm_aux_exact(
                    c, n_max_updated, self._necessary_coalition_size_cm[c], optimize_bounds, suggested_path,
                    preferences_borda_s=self.profile_.preferences_borda_rk[
                                        np.logical_not(self.v_wants_to_help_c_[:, c]), :]
            )
            self.mylogv("CM: n_manip_exact =", n_manip_exact, 3)
            if n_manip_exact < self._sufficient_coalition_size_cm[c]:
                self._sufficient_coalition_size_cm[c] = n_manip_exact
                self._example_path_cm[c] = example_path_exact
                self.mylogv('CM: Update sufficient_coalition_size_cm[c] = n_manip_exact =')
            # Update necessary coalition and return
            if optimize_bounds:
                self._update_necessary(self._necessary_coalition_size_cm, c, self._sufficient_coalition_size_cm[c],
                                       'CM: Update necessary_coalition_size_cm[c] = sufficient_coalition_size_cm[c] =')
                return False
            else:
                if self.profile_.matrix_duels_ut[c, self.w_] >= self._sufficient_coalition_size_cm[c]:
                    # Manipulation worked. By design of ``_cm_aux_exact`` when running without ``optimize_bounds``,
                    # we have not explored everything (quick escape).
                    return True
                else:
                    # We have explored everything with ``n_max = n_m`` but manipulation failed. However, we have not
                    # optimized ``sufficient_size`` (which must be higher than ``n_m``), so it is a quick escape.
                    self._update_necessary(
                        self._necessary_coalition_size_cm, c, self.profile_.matrix_duels_ut[c, self.w_] + 1,
                        'CM: Update necessary_coalition_size_cm[c] = n_m + 1 =')
                    return True


if __name__ == '__main__':
    RuleIRV()(Profile(preferences_ut=np.random.randint(-5, 5, (5, 3)))).demo_()
