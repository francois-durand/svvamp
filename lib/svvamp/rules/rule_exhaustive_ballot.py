# -*- coding: utf-8 -*-
"""
Created on 11 dec. 2018, 15:20
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
from svvamp.utils.pseudo_bool import equal_false
from svvamp.preferences.profile import Profile


class RuleExhaustiveBallot(Rule):
    """Exhaustive Ballot.

    Examples
    --------
        >>> profile = Profile(preferences_ut=[
        ...     [ 0. , -0.5, -1. ],
        ...     [ 1. , -1. ,  0.5],
        ...     [ 0.5,  0.5, -0.5],
        ...     [ 0.5,  0. ,  1. ],
        ...     [-1. , -1. ,  1. ],
        ... ], preferences_rk=[
        ...     [0, 1, 2],
        ...     [0, 2, 1],
        ...     [1, 0, 2],
        ...     [2, 0, 1],
        ...     [2, 1, 0],
        ... ])
        >>> rule = RuleExhaustiveBallot()(profile)
        >>> rule.demo_results_(log_depth=0)  # doctest: +NORMALIZE_WHITESPACE
        <BLANKLINE>
        ************************
        *                      *
        *   Election Results   *
        *                      *
        ************************
        <BLANKLINE>
        ***************
        *   Results   *
        ***************
        profile_.preferences_ut (reminder) =
        [[ 0.  -0.5 -1. ]
         [ 1.  -1.   0.5]
         [ 0.5  0.5 -0.5]
         [ 0.5  0.   1. ]
         [-1.  -1.   1. ]]
        profile_.preferences_rk (reminder) =
        [[0 1 2]
         [0 2 1]
         [1 0 2]
         [2 0 1]
         [2 1 0]]
        ballots =
        [[0 0]
         [0 0]
         [1 0]
         [2 2]
         [2 2]]
        scores =
        [[ 2.  1.  2.]
         [ 3. nan  2.]]
        candidates_by_scores_best_to_worst
        [0 2 1]
        scores_best_to_worst
        [[ 2.  2.  1.]
         [ 3.  2. nan]]
        w = 0
        score_w = [2. 3.]
        total_utility_w = 1.0
        <BLANKLINE>
        *********************************
        *   Condorcet efficiency (rk)   *
        *********************************
        w (reminder) = 0
        <BLANKLINE>
        condorcet_winner_rk_ctb = 0
        w_is_condorcet_winner_rk_ctb = True
        w_is_not_condorcet_winner_rk_ctb = False
        w_missed_condorcet_winner_rk_ctb = False
        <BLANKLINE>
        condorcet_winner_rk = 0
        w_is_condorcet_winner_rk = True
        w_is_not_condorcet_winner_rk = False
        w_missed_condorcet_winner_rk = False
        <BLANKLINE>
        ***************************************
        *   Condorcet efficiency (relative)   *
        ***************************************
        w (reminder) = 0
        <BLANKLINE>
        condorcet_winner_ut_rel_ctb = 0
        w_is_condorcet_winner_ut_rel_ctb = True
        w_is_not_condorcet_winner_ut_rel_ctb = False
        w_missed_condorcet_winner_ut_rel_ctb = False
        <BLANKLINE>
        condorcet_winner_ut_rel = 0
        w_is_condorcet_winner_ut_rel = True
        w_is_not_condorcet_winner_ut_rel = False
        w_missed_condorcet_winner_ut_rel = False
        <BLANKLINE>
        ***************************************
        *   Condorcet efficiency (absolute)   *
        ***************************************
        w (reminder) = 0
        <BLANKLINE>
        condorcet_admissible_candidates =
        [ True False False]
        w_is_condorcet_admissible = True
        w_is_not_condorcet_admissible = False
        w_missed_condorcet_admissible = False
        <BLANKLINE>
        weak_condorcet_winners =
        [ True False False]
        w_is_weak_condorcet_winner = True
        w_is_not_weak_condorcet_winner = False
        w_missed_weak_condorcet_winner = False
        <BLANKLINE>
        condorcet_winner_ut_abs_ctb = 0
        w_is_condorcet_winner_ut_abs_ctb = True
        w_is_not_condorcet_winner_ut_abs_ctb = False
        w_missed_condorcet_winner_ut_abs_ctb = False
        <BLANKLINE>
        condorcet_winner_ut_abs = 0
        w_is_condorcet_winner_ut_abs = True
        w_is_not_condorcet_winner_ut_abs = False
        w_missed_condorcet_winner_ut_abs = False
        <BLANKLINE>
        resistant_condorcet_winner = nan
        w_is_resistant_condorcet_winner = False
        w_is_not_resistant_condorcet_winner = True
        w_missed_resistant_condorcet_winner = False
        >>> rule.demo_manipulation_(log_depth=0)  # doctest: +NORMALIZE_WHITESPACE
        <BLANKLINE>
        *****************************
        *                           *
        *   Election Manipulation   *
        *                           *
        *****************************
        <BLANKLINE>
        *********************************************
        *   Basic properties of the voting system   *
        *********************************************
        with_two_candidates_reduces_to_plurality =  True
        is_based_on_rk =  True
        is_based_on_ut_minus1_1 =  False
        meets_iia =  False
        <BLANKLINE>
        ****************************************************
        *   Manipulation properties of the voting system   *
        ****************************************************
        Condorcet_c_ut_rel_ctb (False)     ==>     Condorcet_c_ut_rel (False)
         ||                                                               ||
         ||     Condorcet_c_rk_ctb (False) ==> Condorcet_c_rk (False)     ||
         ||           ||               ||       ||             ||         ||
         V            V                ||       ||             V          V
        Condorcet_c_ut_abs_ctb (False)     ==>     Condorcet_ut_abs_c (False)
         ||                            ||       ||                        ||
         ||                            V        V                         ||
         ||       maj_fav_c_rk_ctb (True)  ==> maj_fav_c_rk (True)        ||
         ||           ||                                       ||         ||
         V            V                                        V          V
        majority_favorite_c_ut_ctb (True)  ==> majority_favorite_c_ut (True)
         ||                                                               ||
         V                                                                V
        IgnMC_c_ctb (True)                 ==>                IgnMC_c (True)
         ||                                                               ||
         V                                                                V
        InfMC_c_ctb (True)                 ==>                InfMC_c (True)
        <BLANKLINE>
        *****************************************************
        *   Independence of Irrelevant Alternatives (IIA)   *
        *****************************************************
        w (reminder) = 0
        is_iia = True
        log_iia: iia_subset_maximum_size = 2.0
        example_winner_iia = nan
        example_subset_iia = nan
        <BLANKLINE>
        **********************
        *   c-Manipulators   *
        **********************
        w (reminder) = 0
        preferences_ut (reminder) =
        [[ 0.  -0.5 -1. ]
         [ 1.  -1.   0.5]
         [ 0.5  0.5 -0.5]
         [ 0.5  0.   1. ]
         [-1.  -1.   1. ]]
        v_wants_to_help_c =
        [[False False False]
         [False False False]
         [False False False]
         [False False  True]
         [False False  True]]
        <BLANKLINE>
        ************************************
        *   Individual Manipulation (IM)   *
        ************************************
        is_im = nan
        log_im: im_option = lazy
        candidates_im =
        [ 0.  0. nan]
        <BLANKLINE>
        *********************************
        *   Trivial Manipulation (TM)   *
        *********************************
        is_tm = False
        log_tm: tm_option = exact
        candidates_tm =
        [0. 0. 0.]
        <BLANKLINE>
        ********************************
        *   Unison Manipulation (UM)   *
        ********************************
        is_um = nan
        log_um: um_option = fast, fast_algo = c_minus_max
        candidates_um =
        [ 0.  0. nan]
        <BLANKLINE>
        *********************************************
        *   Ignorant-Coalition Manipulation (ICM)   *
        *********************************************
        is_icm = False
        log_icm: icm_option = exact
        candidates_icm =
        [0. 0. 0.]
        necessary_coalition_size_icm =
        [0. 6. 4.]
        sufficient_coalition_size_icm =
        [0. 6. 4.]
        <BLANKLINE>
        ***********************************
        *   Coalition Manipulation (CM)   *
        ***********************************
        is_cm = nan
        log_cm: cm_option = fast, fast_algo = c_minus_max, icm_option = exact, tm_option = exact
        candidates_cm =
        [ 0.  0. nan]
        necessary_coalition_size_cm =
        [0. 0. 2.]
        sufficient_coalition_size_cm =
        [0. 2. 4.]

    Notes
    -----
    At each round, voters vote for one non-eliminated candidate. The candidate with least votes is eliminated. Then
    the next round is held. Unlike :class:`RuleIRV`, voters actually vote at each round. This does not change
    anything for sincere voting, but offers a bit more possibilities for the manipulators. In case of a tie,
    the candidate with highest index is eliminated.

    * :meth:`is_cm_`:

        * :attr:`cm_option` = ``'fast'``: Polynomial heuristic. Can prove CM but unable to decide non-CM (except in
          rare obvious cases).
        * :attr:`cm_option` = ``'exact'``: Non-polynomial algorithm (:math:`2^{n_c}`) adapted from Walsh, 2010.

    * :meth:`is_icm_`: Exact in polynomial time.
    * :meth:`is_im_`:

        * :attr:`im_option` = ``'lazy'``: Lazy algorithm from superclass :class:`Rule`.
        * :attr:`im_option` = ``'exact'``: Non-polynomial algorithm (:math:`2^{n_c}`) adapted from Walsh, 2010.

    * :meth:`is_iia`: Non-polynomial or non-exact algorithms from superclass :class:`Rule`.
    * :meth:`is_tm_`: Exact in polynomial time.
    * :meth:`is_um_`:

        * :attr:`um_option` = ``'fast'``: Polynomial heuristic. Can prove UM but unable to decide non-UM (except in
          rare obvious cases).
        * :attr:`um_option` = ``'exact'``: Non-polynomial algorithm (:math:`2^{n_c}`) adapted from Walsh, 2010.

    References
    ----------
    'Single transferable vote resists strategic voting', John J. Bartholdi and James B. Orlin, 1991.

    'On The Complexity of Manipulating Elections', Tom Coleman and Vanessa Teague, 2007.

    'Manipulability of Single Transferable Vote', Toby Walsh, 2010.
    """
    # Exceptionally, for this voting system, we establish a pointer from the Profile object, so that the
    # manipulation results can be used by IRV.

    full_name = 'Exhaustive Ballot'
    abbreviation = 'EB'

    options_parameters = Rule.options_parameters.copy()
    options_parameters.update({
        'um_option': {'allowed': {'fast', 'exact'}, 'default': 'fast'},
        'cm_option': {'allowed': {'fast', 'exact'}, 'default': 'fast'},
        'tm_option': {'allowed': {'exact'}, 'default': 'exact'},
        'icm_option': {'allowed': {'exact'}, 'default': 'exact'},
        'fast_algo': {'allowed': {'c_minus_max', 'minus_max', 'hardest_first'}, 'default': 'c_minus_max'}
    })

    def __init__(self, **kwargs):
        self._fast_algo = None
        super().__init__(
            with_two_candidates_reduces_to_plurality=True, is_based_on_rk=True,
            precheck_um=False, precheck_icm=False,
            log_identity="EXHAUSTIVE_BALLOT", **kwargs
        )

    def __call__(self, profile):
        """
        Parameters
        ----------
        profile : Profile

        Examples
        --------
        Same 'rule' object, different profiles:

            >>> rule = RuleExhaustiveBallot()
            >>> profile = Profile(preferences_rk=[
            ...     [0, 1, 2],
            ...     [0, 2, 1],
            ...     [1, 0, 2],
            ...     [2, 0, 1],
            ...     [2, 1, 0],
            ... ])
            >>> rule(profile).w_
            0
            >>> profile = Profile(preferences_rk=[
            ...     [0, 2, 1],
            ...     [2, 0, 1],
            ...     [2, 0, 1],
            ...     [2, 0, 1],
            ...     [2, 1, 0],
            ... ])
            >>> rule(profile).w_
            2

        Same profile, different 'rule' objects:

            >>> profile = Profile(preferences_rk=[
            ...     [0, 1, 2],
            ...     [0, 2, 1],
            ...     [1, 0, 2],
            ...     [2, 0, 1],
            ...     [2, 1, 0],
            ... ])
            >>> rule_1 = RuleExhaustiveBallot()(profile)
            >>> rule_1.w_
            0
            >>> rule_2 = RuleExhaustiveBallot()(profile)
            >>> rule_2.w_
            0

        Check that the new rule does not change the options of the old one (which used to be a bug):

            >>> profile = Profile(preferences_rk=[
            ...     [0, 1, 2],
            ...     [0, 2, 1],
            ...     [1, 0, 2],
            ...     [2, 0, 1],
            ...     [2, 1, 0],
            ... ])
            >>> rule_eb_fast = RuleExhaustiveBallot(cm_option='fast')
            >>> rule_eb_exact = RuleExhaustiveBallot(cm_option='exact')
            >>> _ = rule_eb_fast(profile)
            >>> _ = rule_eb_exact(profile)
            >>> rule_eb_fast.cm_option
            'fast'
        """
        # Unplug this exhaustive_ballot from the old profile
        if self.profile_ is not None:
            delattr(self.profile_, 'exhaustive_ballot')
        # See if new profile has an exhaustive_ballot, with the same options
        if hasattr(profile, 'exhaustive_ballot'):
            # Save my options
            my_options = self.options
            # Copy all its inner variables into mine
            self.__dict__.update(profile.exhaustive_ballot.__dict__)
            # Reload my options
            self.update_options(my_options)
        else:
            # 'Usual' behavior
            self.delete_cache(suffix='_')
            self.profile_ = profile
        # Bind this exhaustive_ballot with the new profile
        profile.exhaustive_ballot = self
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

    def _count_ballots_aux_(self, compute_v_might_be_pivotal):
        self.mylog("Count ballots", 1)
        ballots = np.zeros((self.profile_.n_v, self.profile_.n_c - 1), dtype=np.int)
        scores = np.zeros((self.profile_.n_c - 1, self.profile_.n_c))
        margins = np.zeros((self.profile_.n_c - 1, self.profile_.n_c))
        v_might_be_pivotal = None
        if compute_v_might_be_pivotal:
            v_might_be_pivotal = np.zeros(self.profile_.n_v)
        is_alive = np.ones(self.profile_.n_c, dtype=np.bool)
        worst_to_best = []
        preferences_borda_rk_temp = self.profile_.preferences_borda_rk.copy()  # We will put -1 when a candidate loses
        for r in range(self.profile_.n_c - 1):
            # Compute the result of Plurality voting
            self.mylogv("is_alive =", is_alive, 3)
            self.mylogv("worst_to_best =", worst_to_best, 3)
            ballots[:, r] = np.argmax(preferences_borda_rk_temp, axis=1)
            self.mylogv("ballots[:, r] =", ballots[:, r], 3)
            scores[r, :] = np.bincount(ballots[:, r], minlength=self.profile_.n_c)
            scores[r, np.logical_not(is_alive)] = np.nan
            self.mylogv("scores[r, :] =", scores[r, :], 3)
            # Who gets eliminated?
            loser = np.where(scores[r, :] == np.nanmin(scores[r, :]))[0][-1]  # Tie-breaking: the last index
            self.mylogv("loser =", loser, 3)
            margins[r, :] = (scores[r, :] - scores[r, loser] + (np.array(range(self.profile_.n_c)) < loser))
            min_margin_r = np.nanmin(margins[r, margins[r, :] != 0])
            self.mylogv("margins_[r, :] =", margins[r, :], 3)
            # Are there pivot voters?
            if compute_v_might_be_pivotal:
                if min_margin_r == 1:
                    # Any voter who has not voted for ``loser`` can save her: she just needs to vote for her and a
                    # candidate who had a margin of 1 (or 2) will be eliminated.
                    v_might_be_pivotal = np.logical_or(v_might_be_pivotal, ballots[:, r] != loser)
                elif min_margin_r == 2:
                    # To change the result of the round by one voter, it is necessary and sufficient one voter who voted
                    # for a candidate with margin 2 votes for ``loser`` instead.
                    v_might_be_pivotal = np.logical_or(v_might_be_pivotal, margins[r, ballots[:, r]] == 2)
                self.mylogv("v_might_be_pivotal =", v_might_be_pivotal, 3)
            # Update tables
            is_alive[loser] = False
            worst_to_best.append(loser)
            preferences_borda_rk_temp[:, loser] = -1
        w = np.argmax(is_alive)
        worst_to_best.append(w)
        elimination_path = np.array(worst_to_best)
        candidates_by_scores_best_to_worst = np.array(worst_to_best[::-1])
        return {'ballots': ballots, 'scores': scores, 'margins': margins, 'v_might_be_pivotal': v_might_be_pivotal,
                'w': w, 'elimination_path': elimination_path,
                'candidates_by_scores_best_to_worst': candidates_by_scores_best_to_worst}

    @cached_property
    def _count_ballots_(self):
        return self._count_ballots_aux_(compute_v_might_be_pivotal=False)

    @cached_property
    def ballots_(self):
        """2d array of integers. ``ballots[v, r]`` is the candidate for which voter ``v`` votes at round ``r``.
        """
        return self._count_ballots_['ballots']

    @cached_property
    def w_(self):
        self.mylog("Compute w", 1)
        plurality_elimination_engine = self.profile_.plurality_elimination_engine()
        for r in range(self.profile_.n_c - 1):
            if r != 0:
                plurality_elimination_engine.update_scores()
            scores_r = plurality_elimination_engine.scores
            # Result of Plurality voting
            self.mylogv("scores_r =", scores_r, 3)
            # Does someone win immediately?
            best_candidate = np.nanargmax(scores_r)
            best_score = scores_r[best_candidate]
            if best_score > self.profile_.n_v / 2:
                return best_candidate
            # Who gets eliminated?
            loser = np.where(scores_r == np.nanmin(scores_r))[0][-1]  # Tie-breaking: the last index
            plurality_elimination_engine.eliminate_candidate(loser)
            self.mylogv("loser =", loser, 3)
        # After the last round...
        return plurality_elimination_engine.candidates_alive[0]

    @cached_property
    def scores_(self):
        """2d array. ``scores[r, c]`` is the number of voters who vote for candidate ``c`` at round ``r``.

        For eliminated candidates, ``scores[r, c] = numpy.nan``. In contrast, ``scores[r, c] = 0`` means that
        ``c`` is present at round ``r`` but no voter votes for ``c``.
        """
        return self._count_ballots_['scores']

    @cached_property
    def margins_(self):
        """2d array. ``margins_[r, c]`` is the number of votes that ``c`` must lose to be eliminated at round ``r``
        (all other things being equal). The candidate who is eliminated at round ``r`` is the only one for which
        ``margins_[r, c] = 0``.

        For eliminated candidates, ``margins_[r, c] = numpy.nan``.

        Examples
        --------
            >>> profile = Profile(preferences_rk=[
            ...     [0, 1, 2],
            ...     [0, 2, 1],
            ...     [0, 2, 1],
            ...     [0, 2, 1],
            ...     [2, 0, 1],
            ... ])
            >>> rule = RuleExhaustiveBallot()(profile)
            >>> rule.margins_
            array([[ 5.,  0.,  1.],
                   [ 4., nan,  0.]])
        """
        return self._count_ballots_['margins']

    @cached_property
    def candidates_by_scores_best_to_worst_(self):
        """1d array of integers. ``candidates_by_scores_best_to_worst`` is the list of all candidates in the reverse
        order of their elimination.
        """
        return self._count_ballots_['candidates_by_scores_best_to_worst']

    @cached_property
    def elimination_path_(self):
        """1d array of integers. Same as :attr:`~svvamp.ExhaustiveBallot.candidates_by_scores_best_to_worst`,
        but in the reverse order.

        Examples
        --------
            >>> profile = Profile(preferences_rk=[
            ...     [0, 1, 2],
            ...     [0, 2, 1],
            ...     [0, 2, 1],
            ...     [0, 2, 1],
            ...     [2, 0, 1],
            ... ])
            >>> rule = RuleExhaustiveBallot()(profile)
            >>> list(rule.elimination_path_)
            [1, 2, 0]
        """
        return self._count_ballots_['elimination_path']

    @cached_property
    def v_might_be_pivotal_(self):
        return self._count_ballots_aux_(compute_v_might_be_pivotal=True)['v_might_be_pivotal']

    @cached_property
    def v_might_im_for_c_(self):
        return np.tile(self.v_might_be_pivotal_[:, np.newaxis], (1, self.profile_.n_c))

    # %% Manipulation criteria of the voting system

    @cached_property
    def meets_majority_favorite_c_rk_ctb(self):
        return True

    # %% Independence of Irrelevant Alternatives (IIA)

    # TODO: A faster algorithm that the one implemented in superclass Rule. Simply check that for each subset of
    # candidates including ``w``, ``w`` cannot be a Plurality loser.

    # %% Individual manipulation (IM)

    def _im_aux(self, anti_voter_allowed, preferences_borda_s):
        """
        Parameters
        ----------
        anti_voter_allowed : bool
            If True, we manipulate with one voter but also an anti-voter (who may decrease one candidate's score by
            1 point at each round). If False, we manipulate only with one voter.
        preferences_borda_s : ndarray
            Preferences of the sincere voters (in Borda format with vtb).

        Returns
        -------
        list of bool
            1d array of booleans, ``candidates_im_aux``. ``candidates_im_aux[c]`` is True if manipulation for
            ``c`` is possible (whether desired or not).

        Examples
        --------
            >>> profile = Profile(preferences_rk=[
            ...     [0, 2, 1],
            ...     [0, 2, 1],
            ...     [1, 0, 2],
            ...     [1, 2, 0],
            ...     [2, 0, 1],
            ... ])
            >>> rule = RuleExhaustiveBallot(im_option='exact')(profile)
            >>> rule.is_im_
            False
        """
        # Explore subsets that are reachable. ``situations_begin_r`` is a dictionary.
        #
        #   * keys: ``is_candidate_alive_begin_r``, tuple of booleans.
        #   * values: just a ``None`` for the moment. If we want to store an elimination path in a future version,
        #     it will be here.
        candidates = np.array(range(self.profile_.n_c))
        n_s = preferences_borda_s.shape[0]
        situations_begin_r = {tuple(np.ones(self.profile_.n_c, dtype=np.bool)): (0, [])}
        situations_end_r = {}
        for r in range(self.profile_.n_c - 1):
            self.mylogv("im_aux: Round r =", r, 3)
            situations_end_r = {}
            for is_candidate_alive_begin_r, _ in (situations_begin_r.items()):
                self.mylogv("im_aux: is_candidate_alive_begin_r =", is_candidate_alive_begin_r, 3)
                # Sincere ballots
                is_candidate_alive_begin_r = np.array(is_candidate_alive_begin_r)
                scores_s = np.full(self.profile_.n_c, np.nan)
                scores_s[is_candidate_alive_begin_r] = np.sum(np.equal(
                    preferences_borda_s[:, is_candidate_alive_begin_r],
                    np.max(preferences_borda_s[:, is_candidate_alive_begin_r], 1)[:, np.newaxis]
                ), 0)
                self.mylogv("im_aux: scores_s =", scores_s, 3)
                # If ``w`` has too many votes, then manipulation is not possible.
                if scores_s[self.w_] + (self.w_ == 0) - anti_voter_allowed > n_s - scores_s[self.w_] + 1:
                    self.mylog("im_aux: Manipulation impossible by this " + "path (w has too many votes)", 3)
                    continue
                # Try to limit the possible ``d``'s
                natural_loser = np.where(scores_s == np.nanmin(scores_s))[0][-1]
                self.mylogv("im_aux: natural_loser =", natural_loser, 3)
                if not anti_voter_allowed:
                    # ``v`` can change the result of the round by only one strategy: vote for ``natural_loser``. Does
                    #  it lead to another candidate losing?
                    scores_temp = np.copy(scores_s)
                    scores_temp[natural_loser] += 1
                    other_possible_loser = np.where(scores_temp == np.nanmin(scores_temp))[0][-1]
                    self.mylogv("im_aux: other_possible_loser =", other_possible_loser, 3)
                    if other_possible_loser == natural_loser:
                        losers_to_test = [natural_loser]
                    else:
                        losers_to_test = [natural_loser, other_possible_loser]
                else:
                    # With an anti-voter, who can be eliminated?
                    #
                    #   * The candidate with margin 0 (natural loser).
                    #   * Candidates with margin 1: -1 vote is enough (definition of the margin).
                    #   * Candidates with margin 2: it is necessary to remove a vote to her, and add a vote to
                    #     ``natural_loser``. But it  might not be sufficient, so we need to check if it works.
                    margins_s = scores_s - scores_s[natural_loser] + (candidates < natural_loser)
                    self.mylogv("im_aux: margins_s =", margins_s, 3)
                    is_eliminable = np.equal(margins_s, 1)
                    is_eliminable[natural_loser] = True
                    for d in candidates[margins_s == 2]:
                        scores_temp = np.copy(scores_s)
                        scores_temp[d] -= 1
                        scores_temp[natural_loser] += 1
                        self.mylogv("im_aux: scores_temp =", scores_temp, 3)
                        loser_temp = np.where(scores_temp == np.nanmin(scores_temp))[0][-1]
                        self.mylogv("im_aux: loser_temp =", loser_temp, 3)
                        if loser_temp == d:
                            is_eliminable[d] = True
                    losers_to_test = candidates[is_eliminable]
                self.mylogv("im_aux: losers_to_test = ", losers_to_test, 3)
                # Loop on ``d``
                for d in losers_to_test:
                    # At this point, we know that we can eliminate ``d``. Feed the dictionary ``situations_end_r``.
                    is_candidate_alive_end_r = np.copy(is_candidate_alive_begin_r)
                    is_candidate_alive_end_r[d] = False
                    if tuple(is_candidate_alive_end_r) not in situations_end_r:
                        situations_end_r[tuple(is_candidate_alive_end_r)] = None
            self.mylogv("im_aux: situations_end_r =", situations_end_r, 3)
            situations_begin_r = situations_end_r
        candidates_im_aux = np.zeros(self.profile_.n_c)
        for is_candidate_alive_end, foobar in situations_end_r.items():
            candidates_im_aux = np.logical_or(candidates_im_aux, is_candidate_alive_end)
        self.mylogv("im_aux: candidates_im_aux =", candidates_im_aux, 3)
        return candidates_im_aux

    def _im_preliminary_checks_general_subclass_(self):
        """
            >>> profile = Profile(preferences_rk=[
            ...     [0, 2, 1],
            ...     [0, 2, 1],
            ...     [1, 0, 2],
            ...     [1, 0, 2],
            ...     [1, 2, 0],
            ... ])
            >>> rule = RuleExhaustiveBallot(im_option='exact')(profile)
            >>> rule.is_im_
            False

            >>> profile = Profile(preferences_rk=[
            ...     [0, 1, 2],
            ...     [0, 1, 2],
            ...     [0, 1, 2],
            ...     [0, 2, 1],
            ...     [1, 0, 2],
            ... ])
            >>> rule = RuleExhaustiveBallot(im_option='exact')(profile)
            >>> rule.is_im_
            False
        """
        if self.im_option != "exact":
            return
        if np.all(np.logical_not(np.isneginf(self._v_im_for_c))):
            return
        self.mylog("IM: Test with a voter and an anti-voter...")
        # If it's impossible with a voter and an anti-voter on the whole population, then we'll know that IM is
        # impossible.
        candidates_im_aux = self._im_aux(anti_voter_allowed=True,
                                         preferences_borda_s=self.profile_.preferences_borda_rk)
        candidates_im_aux[self.w_] = False
        for c in self.losing_candidates_:
            if equal_false(candidates_im_aux[c]):
                self.mylogv("IM: Manipulation with voter and anti-voter failed for c =", c, 2)
                self._v_im_for_c[:, c] = False

    def _im_main_work_v_(self, v, c_is_wanted, nb_wanted_undecided, stop_if_true):
        """
            >>> profile = Profile(preferences_rk=[
            ...     [0, 2, 1],
            ...     [0, 2, 1],
            ...     [1, 2, 0],
            ...     [1, 2, 0],
            ...     [2, 1, 0],
            ... ])
            >>> rule = RuleExhaustiveBallot(im_option='exact')(profile)
            >>> rule.is_im_
            True
        """
        if self.im_option == "lazy":
            self._im_main_work_v_lazy_(v, c_is_wanted, nb_wanted_undecided, stop_if_true)
            return
        candidates_im_aux = self._im_aux(
            anti_voter_allowed=False,
            preferences_borda_s=self.profile_.preferences_borda_rk[np.array(range(self.profile_.n_v)) != v, :])
        for c in self.losing_candidates_:
            if not np.isneginf(self._v_im_for_c[v, c]):
                # ``v`` is not interested, or we already know for some reason
                continue
            if candidates_im_aux[c]:
                self._v_im_for_c[v, c] = True
                self._candidates_im[c] = True
                self._voters_im[v] = True
                self._is_im = True
                self.mylogv("IM found for c =", c, 3)
            else:
                self._v_im_for_c[v, c] = False

    # %% Collective manipulations (CM / UM): general routine

    def _cm_aux_fast(self, c, n_max, unison, preferences_borda_s):
        """Fast algorithm used for CM and UM.

        Parameters
        ----------
        c : int
            Candidate for which we want to manipulate.
        n_max : int
            Maximum number of manipulators allowed.

            * UM --> put the number of manipulators.
            * CM, with candidates and exact --> put the current value of ``sufficient_coalition_size[c] - 1`` (we want
              to find the best value for ``sufficient_coalition_size[c]``, even if it exceeds the number of
              manipulators)
            * CM, otherwise --> put the number of manipulators.

        unison : bool
            Must be True when computing UM, False for CM.
        preferences_borda_s : ndarray
            Preferences of the sincere voters (in Borda format with vtb).

        Returns
        -------
        tuple
            ``(n_manip_fast, example_path_fast)``.

            * ``n_manip_fast``: Integer or +inf. If a manipulation is found, a sufficient number of manipulators
              is returned (if unison is True, this number will always be ``n_max``). If no manipulation is found,
              it is +inf.
            * ``example_path_fast``: An example of elimination path that realizes the manipulation with
              ``n_manip_fast`` manipulators. ``example_path_fast[k]`` is the ``k``-th candidate eliminated. If no
              manipulation is found, ``example_path`` is NaN.

        Examples
        --------
            >>> profile = Profile(preferences_rk=[
            ...     [0, 1, 2],
            ...     [0, 2, 1],
            ...     [1, 0, 2],
            ...     [1, 2, 0],
            ...     [2, 1, 0],
            ... ])
            >>> rule = RuleExhaustiveBallot(fast_algo='c_minus_max')(profile)
            >>> rule.is_cm_
            nan
            >>> rule = RuleExhaustiveBallot(fast_algo='minus_max')(profile)
            >>> rule.is_cm_
            nan
            >>> rule = RuleExhaustiveBallot(fast_algo='hardest_first')(profile)
            >>> rule.is_cm_
            nan

            >>> profile = Profile(preferences_rk=[
            ...     [0, 1, 2],
            ...     [0, 1, 2],
            ...     [1, 0, 2],
            ...     [1, 2, 0],
            ...     [2, 1, 0],
            ... ])
            >>> rule = RuleExhaustiveBallot()(profile)
            >>> rule.is_um_
            nan
        """
        # At each round, we examine all candidates ``d`` that we can eliminate. If several ``d`` are possible,
        # we choose the one that maximize a heuristic parameter denoted ``situation_c``.
        candidates = np.array(range(self.profile_.n_c))
        n_s = preferences_borda_s.shape[0]
        n_manip_fast = 0  # Number of manipulators
        example_path_fast = []
        is_candidate_alive = np.ones(self.profile_.n_c, dtype=np.bool)
        # Sincere scores (eliminated candidates will have nan)
        scores_s = np.sum(np.equal(preferences_borda_s, np.max(preferences_borda_s, 1)[:, np.newaxis]), 0)
        for r in range(self.profile_.n_c - 1):
            self.mylogv("cm_aux_fast: Round r =", r, 3)
            self.mylogv("cm_aux_fast: scores_s =", scores_s, 3)
            # If an opponent has too many votes, then manipulation is not possible.
            max_score = np.nanmax(scores_s[candidates != c])
            most_serious_opponent = np.where(scores_s == max_score)[0][0]
            if max_score + (most_serious_opponent < c) > n_s + n_max - max_score:
                self.mylogv("cm_aux_fast: most_serious_opponent =", most_serious_opponent, 3)
                self.mylog("cm_aux_fast: Manipulation impossible by this path (an opponent has too many votes)", 3)
                n_manip_fast = np.inf  # by convention
                return n_manip_fast, np.nan
            # Try to limit the possible ``d``'s (for unison)
            if unison:
                natural_loser = np.where(scores_s == np.nanmin(scores_s))[0][-1]
                self.mylogv("cm_aux_fast: natural_loser =", natural_loser, 3)
                # In unison, if we want to change the result of the round, the only thing we can do is to vote for
                # the natural loser.
                scores_temp = np.copy(scores_s)
                scores_temp[natural_loser] += n_max
                other_possible_loser = np.where(scores_temp == np.nanmin(scores_temp))[0][-1]
                self.mylogv("cm_aux_fast: other_possible_loser =", other_possible_loser, 3)
                if other_possible_loser == natural_loser:
                    losers_to_test = [natural_loser]
                else:
                    losers_to_test = [natural_loser, other_possible_loser]
            else:
                losers_to_test = candidates[is_candidate_alive]
            self.mylogv("cm_aux_fast: losers_to_test =", losers_to_test, 3)
            self.mylogv("cm_aux_fast: but do not test ", c, 3)
            # Initialize the loop on ``d``
            best_d = -1
            best_situation_for_c = -np.inf
            n_manip_r = np.inf
            scores_s_end_r = np.nan
            for d in losers_to_test:
                if d == c:
                    continue
                self.mylogv("cm_aux_fast: d =", d, 3)
                # Is it possible to eliminate ``d`` now?
                if unison:
                    # We already know that it is possible.
                    n_manip_d = n_max
                else:
                    scores_m = np.maximum(
                        0, scores_s[d] - scores_s[is_candidate_alive] + (candidates[is_candidate_alive] > d))
                    n_manip_d = np.sum(scores_m)
                    self.mylogv("cm_aux_fast: n_manip_d =", n_manip_d, 3)
                    if n_manip_d > n_max:
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
                if self.fast_algo == "c_minus_max":
                    situation_for_c = (scores_s_temp[c] - np.nanmax(scores_s_temp[candidates != c]))
                elif self.fast_algo == "minus_max":
                    situation_for_c = - np.nanmax(scores_s_temp[candidates != c])
                elif self.fast_algo == "hardest_first":
                    situation_for_c = n_manip_d
                else:
                    raise NotImplementedError("Unknown fast algorithm: " + format(self.fast_algo))
                self.mylogv("cm_aux_fast: scores_s_temp =", scores_s_temp, 3)
                self.mylogv("cm_aux_fast: situation_for_c =", situation_for_c, 3)
                # Is the the best ``d`` so far?
                # Lexicographic comparison on three criteria (highest ``situation``, lowest number of manipulators,
                # lowest index).
                if [situation_for_c, -n_manip_d, -d] > [best_situation_for_c, -n_manip_r, -best_d]:
                    best_d = d
                    best_situation_for_c = situation_for_c
                    n_manip_r = n_manip_d
                    scores_s_end_r, scores_s_temp = scores_s_temp, None
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
        self.mylogv("cm_aux_fast: : n_manip_fast =", n_manip_fast, 3)
        if n_manip_fast <= n_max:
            example_path_fast.append(c)
            return n_manip_fast, np.array(example_path_fast)
        else:
            return n_manip_fast, np.nan

    def _cm_aux_exact(self, c, n_max, unison, preferences_borda_s):
        """Exact algorithm used for CM and UM.

        Parameters
        ----------
        c : int
            Candidate for which we want to manipulate.
        n_max : int
            Maximum number of manipulators allowed.

            * UM --> put the number of manipulators.
            * CM, with candidates and exact --> put the current value of ``sufficient_coalition_size[c] - 1`` (we want
              to find the best value for ``sufficient_coalition_size[c]``, even if it exceeds the number of
              manipulators)
            * CM, otherwise --> put the number of manipulators.

        unison : bool
            Must be True when computing UM, False for CM.
        preferences_borda_s : ndarray
            Preferences of the sincere voters (in Borda format).

        Returns
        -------
        tuple
            ``(n_manip_final, example_path)``.

            * ``n_manip_final``: Integer or +inf. If manipulation is impossible with ``<= n_max`` manipulators, it is
              +inf. If manipulation is possible (with ``<= n_max``):

                * If ``unison`` is False, it is the minimal number of manipulators.
                * If ``unison`` is True, it is ``n_max``.

            * ``example_path``: An example of elimination path that realizes the manipulation with ``n_manip_final``
              manipulators. ``example_path[k]`` is the ``k``-th candidate eliminated. If the manipulation is impossible,
              ``example_path`` is NaN.

        Examples
        --------
            >>> profile = Profile(preferences_rk=[
            ...     [0, 1, 2],
            ...     [1, 0, 2],
            ...     [1, 2, 0],
            ...     [2, 1, 0],
            ...     [2, 1, 0],
            ... ])
            >>> rule = RuleExhaustiveBallot(um_option='exact')(profile)
            >>> rule.is_um_
            False

            >>> profile = Profile(preferences_rk=[
            ...     [0, 1, 3, 2],
            ...     [0, 2, 3, 1],
            ...     [2, 0, 1, 3],
            ...     [3, 1, 0, 2],
            ...     [3, 2, 0, 1],
            ... ])
            >>> rule = RuleExhaustiveBallot(cm_option='exact')(profile)
            >>> rule.candidates_cm_
            array([0., 0., 0., 0.])

            >>> profile = Profile(preferences_rk=[
            ...     [0, 2, 1, 3],
            ...     [1, 0, 3, 2],
            ...     [1, 3, 2, 0],
            ...     [2, 3, 1, 0],
            ...     [3, 2, 0, 1],
            ... ])
            >>> rule = RuleExhaustiveBallot(cm_option='exact')(profile)
            >>> rule.candidates_cm_
            array([0., 1., 0., 1.])

            >>> profile = Profile(preferences_rk=[
            ...     [0, 1, 2, 3],
            ...     [1, 3, 0, 2],
            ...     [2, 1, 3, 0],
            ...     [3, 0, 1, 2],
            ...     [3, 1, 0, 2],
            ... ])
            >>> rule = RuleExhaustiveBallot(cm_option='exact')(profile)
            >>> rule.candidates_cm_
            array([0., 0., 0., 0.])
        """
        # Explore subsets that are reachable with less than the upper bound. ``situations_begin_r`` is a dictionary.
        #
        #   * keys: ``is_candidate_alive_begin_r``, tuple of booleans.
        #   * values: tuple ``(n_manip_used_before_r, example_path_before_r)``.
        #
        #       * ``n_manip_used_before_r``: number of manipulators used before round ``r`` to get to this subset of
        #         candidates.
        #       * ``example_path_before_r``: as it sounds.
        candidates = np.array(range(self.profile_.n_c))
        n_s = preferences_borda_s.shape[0]
        situations_begin_r = {tuple(np.ones(self.profile_.n_c, dtype=np.bool)): (0, [])}
        situations_end_r = {}
        for r in range(self.profile_.n_c - 1):
            self.mylogv("cm_aux_exact: Round r =", r, 3)
            situations_end_r = {}
            for is_candidate_alive_begin_r, (n_manip_used_before_r, example_path_before_r) in (
                    situations_begin_r.items()):
                self.mylogv("cm_aux_exact: is_candidate_alive_begin_r =", is_candidate_alive_begin_r, 3)
                self.mylogv("cm_aux_exact: n_manip_used_before_r =", n_manip_used_before_r, 3)
                self.mylogv("cm_aux_exact: example_path_before_r =", example_path_before_r, 3)
                # Sincere ballots
                is_candidate_alive_begin_r = np.array(is_candidate_alive_begin_r)
                scores_s = np.full(self.profile_.n_c, np.nan)
                scores_s[is_candidate_alive_begin_r] = np.sum(np.equal(
                    preferences_borda_s[:, is_candidate_alive_begin_r],
                    np.max(preferences_borda_s[:, is_candidate_alive_begin_r], 1)[:, np.newaxis]
                ), 0)
                self.mylogv("cm_aux_exact: scores_s =", scores_s, 3)
                # If an opponent has too many votes, then manipulation is not possible.
                max_score = np.nanmax(scores_s[candidates != c])
                most_serious_opponent = np.where(scores_s == max_score)[0][0]
                if max_score + (most_serious_opponent < c) > n_s + n_max - max_score:
                    self.mylogv("cm_aux_exact: most_serious_opponent =", most_serious_opponent, 3)
                    self.mylog("cm_aux_exact: Manipulation impossible by this path (an opponent has too many votes)", 3)
                    continue
                # Try to limit the possible ``d``'s
                if unison:
                    natural_loser = np.where(scores_s == np.nanmin(scores_s))[0][-1]
                    self.mylogv("cm_aux_exact: natural_loser =", natural_loser, 3)
                    scores_temp = np.copy(scores_s)
                    scores_temp[natural_loser] += n_max
                    other_possible_loser = np.where(scores_temp == np.nanmin(scores_temp))[0][-1]
                    self.mylogv("cm_aux_exact: other_possible_loser =", other_possible_loser, 3)
                    if other_possible_loser == natural_loser:
                        losers_to_test = [natural_loser]
                    else:
                        losers_to_test = [natural_loser, other_possible_loser]
                else:
                    losers_to_test = candidates[is_candidate_alive_begin_r]
                self.mylogv("cm_aux_exact: losers_to_test =", losers_to_test, 3)
                self.mylogv("cm_aux_exact: but do not test ", c, 3)
                # Loop on ``d``
                for d in losers_to_test:
                    if d == c:
                        continue
                    self.mylogv("cm_aux_exact: d =", d, 3)
                    # Is it possible to eliminate ``d`` now?
                    if unison:
                        # We already know that it is possible.
                        n_manip_r = n_max
                    else:
                        scores_m = np.maximum(0,
                                              scores_s[d] - scores_s[is_candidate_alive_begin_r]
                                              + (candidates[is_candidate_alive_begin_r] > d))
                        n_manip_r = np.sum(scores_m)
                    self.mylogv("cm_aux_exact: n_manip_r =", n_manip_r, 3)
                    if n_manip_r > n_max:
                        continue
                    # Feed the dictionary 'situations_end_r'
                    n_manip_r_and_before = max(n_manip_r, n_manip_used_before_r)
                    is_candidate_alive_end_r = np.copy(is_candidate_alive_begin_r)
                    is_candidate_alive_end_r[d] = False
                    example_path_end_r = example_path_before_r[:]
                    example_path_end_r.append(d)
                    if tuple(is_candidate_alive_end_r) in situations_end_r:
                        if n_manip_r_and_before < situations_end_r[tuple(is_candidate_alive_end_r)][0]:
                            situations_end_r[tuple(is_candidate_alive_end_r)] = (
                                n_manip_r_and_before, example_path_end_r)
                    else:
                        situations_end_r[tuple(is_candidate_alive_end_r)] = (
                            n_manip_r_and_before, example_path_end_r)
            self.mylogv("cm_aux_exact: situations_end_r =", situations_end_r, 3)
            if len(situations_end_r) == 0:
                self.mylog("cm_aux_exact: Manipulation is impossible with n_max manipulators.", 3)
                return np.inf, np.nan
            situations_begin_r = situations_end_r
        # If we reach this point, we know that all rounds were successful.
        self.mylogv("cm_aux_exact: situations_end_r:", situations_end_r, 3)
        is_candidate_alive_end, (n_manip_exact, example_path_exact) = situations_end_r.popitem()
        example_path_exact.append(c)
        self.mylogv("cm_aux_exact: is_candidate_alive_end:", is_candidate_alive_end, 3)
        self.mylogv("cm_aux_exact: example_path_exact:", example_path_exact, 3)
        self.mylogv("cm_aux_exact: Conclusion phase 2: n_manip_exact =", n_manip_exact, 3)
        return n_manip_exact, np.array(example_path_exact)

    # %% Trivial Manipulation (TM)

    @cached_property
    def example_path_tm_(self):
        """
        Examples
        --------
            >>> profile = Profile(preferences_rk=[
            ...     [1, 0, 2],
            ...     [1, 2, 0],
            ...     [1, 2, 0],
            ...     [2, 1, 0],
            ...     [2, 1, 0],
            ... ])
            >>> rule = RuleExhaustiveBallot()(profile)
            >>> rule.example_path_tm_
            {0: array([2, 1, 0]), 1: None, 2: array([0, 1, 2])}
        """
        _ = self.candidates_tm_
        return self._example_path_tm

    @cached_property
    def sufficient_coalition_size_tm_(self):
        """
        Examples
        --------
            >>> profile = Profile(preferences_rk=[
            ...     [1, 0, 2],
            ...     [1, 2, 0],
            ...     [1, 2, 0],
            ...     [2, 1, 0],
            ...     [2, 1, 0],
            ... ])
            >>> rule = RuleExhaustiveBallot()(profile)
            >>> rule.sufficient_coalition_size_tm_
            array([5., 0., 4.])
        """
        _ = self.candidates_tm_
        return self._sufficient_coalition_size_tm

    def example_path_tm_c_(self, c):
        _ = self.is_tm_c_(c)
        return self._example_path_tm[c]

    def sufficient_coalition_size_tm_c_(self, c):
        _ = self.is_tm_c_(c)
        return self._sufficient_coalition_size_tm[c]

    @cached_property
    def _tm_is_initialized_general_subclass_(self):
        self._sufficient_coalition_size_tm = np.full(self.profile_.n_c, np.inf)
        self._sufficient_coalition_size_tm[self.w_] = 0
        self._candidates_tm[self.w_] = False
        self._example_path_tm = {c: None for c in range(self.profile_.n_c)}
        return True

    def _tm_preliminary_checks_general_(self):
        # We remove the general preliminary checks, because we always want to run ``_tm_main_work_c`` to provide an
        # example of path.
        pass

    def _tm_initialize_c_(self, c):
        self.mylogv("TM: Candidate =", c, 2)
        # We remove the general preliminary checks on ``c`` for the same reason.

    def _tm_main_work_c_(self, c):
        # For Exhaustive Ballot, if TM works, then manipulators always vote for ``c``, so the rest of their ballot
        # has no impact on the result. So we can define a minimum coalition size for TM: minimal number of
        # manipulators such that, when always voting for ``c``, ``c`` gets elected. It will help us for CM: indeed,
        # it is a lower bound that is better than TM (= ``n_m``, when it works) and also than ICM (= ``n_s`` or
        # ``n_s + 1``).
        candidates_not_c = np.concatenate((range(c), range(c + 1, self.profile_.n_c))).astype(int)
        example_path = []
        is_alive = np.ones(self.profile_.n_c, dtype=np.bool)
        n_manip_used = 0
        for r in range(self.profile_.n_c - 1):
            scores_tot = np.full(self.profile_.n_c, np.nan)
            scores_tot[is_alive] = np.sum(np.equal(
                self.profile_.preferences_borda_rk[np.logical_not(self.v_wants_to_help_c_[:, c]), :][:, is_alive],
                np.max(self.profile_.preferences_borda_rk[
                       np.logical_not(self.v_wants_to_help_c_[:, c]), :][:, is_alive], 1)[:, np.newaxis]
            ), 0)
            loser = candidates_not_c[
                np.where(scores_tot[candidates_not_c] == np.nanmin(scores_tot[candidates_not_c]))[0][-1]]
            n_manip_used = max(n_manip_used, scores_tot[loser] - scores_tot[c] + (c > loser))
            is_alive[loser] = False
            example_path.append(loser)
        example_path.append(c)
        # Conclude for ``c``
        self._sufficient_coalition_size_tm[c] = n_manip_used
        self.mylogv('TM: sufficient_coalition_size_TM[c]', self._sufficient_coalition_size_tm[c], 2)
        self._example_path_tm[c] = np.array(example_path)
        self.mylogv('TM: example_path_TM[c] =', self._example_path_tm[c], 2)
        if self.profile_.matrix_duels_ut[c, self.w_] >= self._sufficient_coalition_size_tm[c]:
            self._candidates_tm[c] = True
            self._is_tm = True
        else:
            self._candidates_tm[c] = False

    # %% Unison manipulation (UM)
    #
    # Note: if pretests conclude that UM is True, no elimination path is computed. But in that cases, TM is True. So
    # we can use the elimination path of TM if we need to quickly provide one for CM.

    @cached_property
    def example_path_um_(self):
        """
            >>> profile = Profile(preferences_rk=[
            ...     [1, 0, 2],
            ...     [1, 2, 0],
            ...     [1, 2, 0],
            ...     [2, 1, 0],
            ...     [2, 1, 0],
            ... ])
            >>> rule = RuleExhaustiveBallot()(profile)
            >>> rule.example_path_um_
            {0: None, 1: None, 2: None}
        """
        _ = self.candidates_um_
        return self._example_path_um

    def example_path_um_c_(self, c):
        """
            >>> profile = Profile(preferences_rk=[
            ...     [1, 0, 2],
            ...     [1, 2, 0],
            ...     [1, 2, 0],
            ...     [2, 1, 0],
            ...     [2, 1, 0],
            ... ])
            >>> rule = RuleExhaustiveBallot()(profile)
            >>> print(rule.example_path_um_c_(0))
            None
        """
        _ = self.is_um_c_(c)
        return self._example_path_um[c]

    @cached_property
    def _um_is_initialized_general_subclass_(self):
        self._example_path_um = {c: None for c in range(self.profile_.n_c)}
        return True

    def _um_preliminary_checks_c_(self, c):
        if self.um_option not in {'fast', 'lazy'} or self.cm_option not in {'fast', 'lazy'}:
            if (
                self.w_ == self.profile_.condorcet_winner_rk_ctb
                and not self.profile_.c_might_be_there_when_cw_is_eliminated_irv_style[c]
            ):
                # Impossible to manipulate with n_m manipulators
                self._candidates_um[c] = False

    def _um_main_work_c_(self, c):
        """
            >>> profile = Profile(preferences_rk=[
            ...     [0, 3, 1, 2],
            ...     [1, 3, 0, 2],
            ...     [2, 0, 1, 3],
            ...     [2, 1, 0, 3],
            ...     [3, 0, 2, 1],
            ... ])
            >>> rule = RuleExhaustiveBallot(um_option='exact')(profile)
            >>> rule.candidates_um_
            array([0., 0., 0., 1.])

            >>> profile = Profile(preferences_ut=[
            ...     [-1. ,  0.5, -0.5, -1. ,  0.5],
            ...     [ 0. ,  1. ,  1. ,  0. ,  0.5],
            ...     [ 0. , -0.5, -0.5,  0.5,  0. ],
            ...     [-1. , -0.5,  0.5,  1. ,  0.5],
            ...     [-0.5, -0.5, -1. , -1. ,  1. ],
            ... ], preferences_rk=[
            ...     [1, 4, 2, 0, 3],
            ...     [2, 1, 4, 0, 3],
            ...     [3, 0, 4, 2, 1],
            ...     [3, 2, 4, 1, 0],
            ...     [4, 0, 1, 3, 2],
            ... ])
            >>> rule = RuleExhaustiveBallot(um_option='exact')(profile)
            >>> rule.candidates_um_
            array([0., 0., 1., 0., 1.])
        """
        exact = (self.um_option == "exact")
        n_m = self.profile_.matrix_duels_ut[c, self.w_]
        self.mylogv("UM: n_m =", n_m, 3)
        n_manip_fast, example_path_fast = self._cm_aux_fast(
            c, n_max=n_m, unison=True,
            preferences_borda_s=self.profile_.preferences_borda_rk[np.logical_not(self.v_wants_to_help_c_[:, c]), :])
        self.mylogv("UM: n_manip_fast =", n_manip_fast, 3)
        if n_manip_fast <= n_m:
            self._candidates_um[c] = True
            if self._example_path_um[c] is None:
                self._example_path_um[c] = example_path_fast
            return
        if not exact:
            self._candidates_um[c] = np.nan
            return

        # From this point, we have necessarily the 'exact' option (and have not found a manipulation for ``c`` yet).
        n_manip_exact, example_path_exact = self._cm_aux_exact(
            c, n_m, unison=True,
            preferences_borda_s=self.profile_.preferences_borda_rk[np.logical_not(self.v_wants_to_help_c_[:, c]), :])
        self.mylogv("UM: n_manip_exact =", n_manip_exact)
        if n_manip_exact <= n_m:
            self._candidates_um[c] = True
            if self._example_path_um[c] is None:
                self._example_path_um[c] = example_path_exact
        else:
            self._candidates_um[c] = False

    # %% Ignorant-Coalition Manipulation (ICM)

    # Use the methods from superclass.

    # %% Coalition Manipulation (CM)

    @cached_property
    def example_path_cm_(self):
        """
            >>> profile = Profile(preferences_rk=[
            ...     [0, 1, 2],
            ...     [0, 2, 1],
            ...     [1, 0, 2],
            ...     [2, 1, 0],
            ...     [2, 1, 0],
            ... ])
            >>> rule = RuleExhaustiveBallot()(profile)
            >>> rule.example_path_cm_
            {0: None, 1: array([2, 0, 1]), 2: array([1, 0, 2])}
        """
        _ = self.candidates_cm_
        return self._example_path_cm

    def example_path_cm_c_(self, c):
        _ = self.is_cm_c_(c)
        return self._example_path_cm[c]

    @cached_property
    def _cm_is_initialized_general_subclass_(self):
        self._example_path_cm = {c: None for c in range(self.profile_.n_c)}
        return True

    def _cm_preliminary_checks_c_(self, c, optimize_bounds):
        # A mandatory additional precheck, to ensure that ``_example_path_cm[c]`` is updated if
        # ``sufficient_coalition_size_cm[c]`` has been updated. We use the following syntax (instead of
        # ``_cm_preliminary_checks_c_subclass``) because we want the test here to be done, even if another one
        # succeeded.
        super()._cm_preliminary_checks_c_(c, optimize_bounds)
        # As a test for sufficient size, this one is better (lower) than all the other ones in
        # ``_cm_preliminary_checks_c``. So, as soon as one of them updates sufficient size, this one will provide an
        # example of path.
        if self.sufficient_coalition_size_tm_c_(c) <= self._sufficient_coalition_size_cm[c]:
            # The <= is not a typo.
            self._update_sufficient(
                self._sufficient_coalition_size_cm, c, self.sufficient_coalition_size_tm_c_(c),
                'CM: Preliminary checks: TM improved => \n    sufficient_coalition_size_cm[c] = ')
            self.mylogv('CM: Preliminary checks: Update _example_path_cm[c] = _example_path_TM[c] =',
                        self.example_path_tm_c_(c), 3)
            self._example_path_cm[c] = self.example_path_tm_c_(c)
        if self.cm_option not in {'fast', 'lazy'}:
            if self.w_ == self.profile_.condorcet_winner_rk_ctb:
                self._update_necessary(
                    self._necessary_coalition_size_cm, c,
                    self.profile_.necessary_coalition_size_to_break_irv_immunity[c],
                    'CM: Preliminary checks: IRV-Immunity => \n    necessary_coalition_size_cm[c] = '
                )

    def _cm_main_work_c_(self, c, optimize_bounds):
        """
            >>> profile = Profile(preferences_rk=[
            ...     [0, 1, 2, 3],
            ...     [0, 3, 2, 1],
            ...     [1, 3, 2, 0],
            ...     [2, 0, 3, 1],
            ...     [3, 2, 0, 1],
            ... ])
            >>> rule = RuleExhaustiveBallot(cm_option='exact')(profile)
            >>> rule.candidates_cm_
            array([1., 0., 0., 1.])

            >>> profile = Profile(preferences_ut=[
            ...     [ 1. ,  0. ,  0.5],
            ...     [ 0. , -0.5,  0.5],
            ... ], preferences_rk=[
            ...     [0, 2, 1],
            ...     [2, 0, 1],
            ... ])
            >>> rule = RuleExhaustiveBallot(cm_option='fast', fast_algo='c_minus_max', icm_option='exact',
            ...                             tm_option='exact')(profile)
            >>> rule.is_cm_c_with_bounds_(1)
            (False, 1.0, 3.0)
        """
        exact = (self.cm_option == "exact")
        if optimize_bounds and exact:
            n_max = self._sufficient_coalition_size_cm[c] - 1
        else:
            n_max = self.profile_.matrix_duels_ut[c, self.w_]
        self.mylogv("CM: n_max =", n_max, 3)
        if not exact and self._necessary_coalition_size_cm[c] > n_max:
            self.mylog("CM: Fast algorithm will not do better than what we already know", 3)
            return
        n_manip_fast, example_path_fast = self._cm_aux_fast(
            c, n_max, unison=False,
            preferences_borda_s=self.profile_.preferences_borda_rk[np.logical_not(self.v_wants_to_help_c_[:, c]), :])
        self.mylogv("CM: n_manip_fast =", n_manip_fast, 3)
        if n_manip_fast < self._sufficient_coalition_size_cm[c]:
            self._sufficient_coalition_size_cm[c] = n_manip_fast
            self._example_path_cm[c] = example_path_fast
            self.mylogv('CM: Update sufficient_coalition_size_cm[c] = n_manip_fast =', n_manip_fast, 3)
        if not exact:
            # With fast algo, we stop here anyway. It is not a "quick escape" (if we'd try again with
            # ``optimize_bounds``, we would not try better).
            return False

        # From this point, we have necessarily the ``exact`` option
        if self._sufficient_coalition_size_cm[c] == self._necessary_coalition_size_cm[c]:
            return False
        if not optimize_bounds and self.profile_.matrix_duels_ut[c, self.w_] >= self._sufficient_coalition_size_cm[c]:
            # This is a quick escape: since we have the option ``exact``, if we come back with ``optimize_bounds``,
            # we will try to be more precise.
            return True

        # Either we're with ``optimize_bounds`` (and might have succeeded), or in non-optimized mode (and we have
        # failed)
        n_max_updated = min(n_manip_fast - 1, n_max)
        self.mylogv("CM: n_max_updated =", n_max_updated)
        n_manip_exact, example_path_exact = self._cm_aux_exact(
            c, n_max_updated, unison=False,
            preferences_borda_s=self.profile_.preferences_borda_rk[np.logical_not(self.v_wants_to_help_c_[:, c]), :])
        self.mylogv("CM: n_manip_exact =", n_manip_exact)
        if n_manip_exact < self._sufficient_coalition_size_cm[c]:
            self._sufficient_coalition_size_cm[c] = n_manip_exact
            self._example_path_cm[c] = example_path_exact
            self.mylogv('CM: Update sufficient_coalition_size_cm[c] = n_manip_exact =')
        # Update necessary coalition and return
        if optimize_bounds:
            self._update_necessary(
                self._necessary_coalition_size_cm, c, self._sufficient_coalition_size_cm[c],
                'CM: Update necessary_coalition_size_cm[c] = sufficient_coalition_size_cm[c] =')
            return False
        else:
            if self.profile_.matrix_duels_ut[c, self.w_] >= self._sufficient_coalition_size_cm[c]:
                # We have optimized the size of the coalition.
                self._update_necessary(
                    self._necessary_coalition_size_cm, c, self._sufficient_coalition_size_cm[c],
                    'CM: Update necessary_coalition_size_cm[c] = sufficient_coalition_size_cm[c] =')
                return False
            else:
                # We have explored everything with ``n_max = n_m`` but manipulation failed. However, we have not
                # optimized ``sufficient_size`` (which must be higher than ``n_m``), so it is a quick escape.
                self._update_necessary(
                    self._necessary_coalition_size_cm, c, self.profile_.matrix_duels_ut[c, self.w_] + 1,
                    'CM: Update necessary_coalition_size_cm[c] = n_m + 1 =')
                return True
