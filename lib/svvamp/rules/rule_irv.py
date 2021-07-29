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
import itertools
import numpy as np
from svvamp.rules.rule import Rule
from svvamp.utils.util_cache import cached_property
from svvamp.preferences.profile import Profile
from svvamp.utils.pseudo_bool import neginf_to_zero, equal_false, equal_true
from svvamp.rules.rule_exhaustive_ballot import RuleExhaustiveBallot
from svvamp.utils.misc import preferences_ut_to_matrix_duels_ut


class RuleIRV(Rule):
    """Instant-Runoff Voting (IRV). Also known as Single Transferable Voting, Alternative Vote, Hare method.

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
        >>> rule = RuleIRV()(profile)
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

    References
    ----------
    'Single transferable vote resists strategic voting', John J. Bartholdi and James B. Orlin, 1991.

    'On The Complexity of Manipulating Elections', Tom Coleman and Vanessa Teague, 2007.

    'Manipulability of Single Transferable Vote', Toby Walsh, 2010.
    """
    # Exceptionally, for this voting system, we establish a pointer from the Profile object, so that the
    # manipulation results can be used by Condorcet-IRV.

    full_name = 'Instant-Runoff Voting'
    abbreviation = 'IRV'

    options_parameters = Rule.options_parameters.copy()
    options_parameters.update({
        'um_option': {'allowed': {'fast', 'exact'}, 'default': 'fast'},
        'cm_option': {'allowed': {'fast', 'slow', 'exact'}, 'default': 'fast'},
        'tm_option': {'allowed': ['exact'], 'default': 'exact'},
        'icm_option': {'allowed': {'exact'}, 'default': 'exact'},
        'fast_algo': {'allowed': {'c_minus_max', 'minus_max', 'hardest_first'}, 'default': 'c_minus_max'}
    })

    def __init__(self, **kwargs):
        self._fast_algo = None
        self.eb_ = None
        super().__init__(
            with_two_candidates_reduces_to_plurality=True, is_based_on_rk=True,
            precheck_um=False, precheck_icm=False,
            log_identity="IRV", **kwargs
        )
        self._example_ballots_cm_c = None
        self._example_ballots_cm_w_against = None

    def __call__(self, profile):
        """
        Parameters
        ----------
        profile : Profile

        Examples
        --------
        Same 'rule' object, different profiles:

            >>> rule = RuleIRV()
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
            >>> rule_1 = RuleIRV()(profile)
            >>> rule_1.w_
            0
            >>> rule_2 = RuleIRV()(profile)
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
            >>> rule_irv_fast = RuleIRV(cm_option='fast')
            >>> rule_irv_exact = RuleIRV(cm_option='exact')
            >>> _ = rule_irv_fast(profile)
            >>> _ = rule_irv_exact(profile)
            >>> rule_irv_fast.cm_option
            'fast'

        Same kind of example with EB/IRV:

            >>> profile = Profile(preferences_rk=[
            ...     [0, 1, 2],
            ...     [0, 2, 1],
            ...     [1, 0, 2],
            ...     [2, 0, 1],
            ...     [2, 1, 0],
            ... ])
            >>> rule_eb_fast = RuleExhaustiveBallot(cm_option='fast')
            >>> rule_irv_exact = RuleIRV(cm_option='exact')
            >>> _ = rule_eb_fast(profile)
            >>> _ = rule_irv_exact(profile)
            >>> rule_eb_fast.cm_option
            'fast'
        """
        # Unplug this irv from the old profile
        if self.profile_ is not None:
            delattr(self.profile_, 'irv')
        # See if new profile has an irv, with the same options
        if hasattr(profile, 'irv'):
            # Save my options
            my_options = self.options
            # Copy all its inner variables into mine
            self.__dict__.update(profile.irv.__dict__)
            # Reload my options
            self.update_options(my_options)
        else:
            # 'Usual' behavior
            self.delete_cache(suffix='_')
            self.profile_ = profile
        # Bind this irv with the new profile
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
        # Initialize examples of manipulating ballots
        self._example_ballots_cm_c = {c: None for c in range(profile.n_c)}
        self._example_ballots_cm_w_against = {w_other_rule: None for w_other_rule in range(profile.n_c)}
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
            ...     [1, 0, 2],
            ...     [2, 0, 1],
            ...     [2, 1, 0],
            ... ])
            >>> rule = RuleIRV()(profile)
            >>> rule.margins_
            array([[ 2.,  0.,  1.],
                   [ 2., nan,  0.]])
        """
        return self.eb_.margins_

    @cached_property
    def candidates_by_scores_best_to_worst_(self):
        """1d array of integers. ``candidates_by_scores_best_to_worst`` is the list of all candidates in the reverse
        order of their elimination.

        Examples
        --------
            >>> profile = Profile(preferences_rk=[
            ...     [0, 1, 2],
            ...     [0, 2, 1],
            ...     [1, 0, 2],
            ...     [2, 0, 1],
            ...     [2, 1, 0],
            ... ])
            >>> rule = RuleIRV()(profile)
            >>> list(rule.candidates_by_scores_best_to_worst_)
            [0, 2, 1]
        """
        return self.eb_.candidates_by_scores_best_to_worst_

    @cached_property
    def elimination_path_(self):
        """1d array of integers. Same as :attr:`~svvamp.ExhaustiveBallot.candidates_by_scores_best_to_worst`,
        but in the reverse order.

        Examples
        --------
            >>> profile = Profile(preferences_rk=[
            ...     [0, 1, 2],
            ...     [0, 2, 1],
            ...     [1, 0, 2],
            ...     [2, 0, 1],
            ...     [2, 1, 0],
            ... ])
            >>> rule = RuleIRV()(profile)
            >>> list(rule.elimination_path_)
            [1, 2, 0]
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
        """
            >>> profile = Profile(preferences_rk=[
            ...     [0, 1, 2],
            ...     [1, 0, 2],
            ...     [1, 2, 0],
            ...     [2, 0, 1],
            ...     [2, 0, 1],
            ... ])
            >>> rule = RuleIRV(im_option='exact')(profile)
            >>> rule.is_im_
            True

            >>> profile = Profile(preferences_rk=[
            ...     [0, 1, 2],
            ...     [0, 2, 1],
            ...     [1, 2, 0],
            ...     [1, 2, 0],
            ...     [2, 0, 1],
            ... ])
            >>> rule = RuleIRV(im_option='exact')(profile)
            >>> rule.is_im_
            True

            >>> profile = Profile(preferences_rk=[
            ...     [0, 1, 2, 3],
            ...     [1, 2, 0, 3],
            ...     [2, 0, 1, 3],
            ...     [3, 0, 2, 1],
            ...     [3, 1, 2, 0],
            ... ])
            >>> rule = RuleIRV(im_option='exact')(profile)
            >>> rule.is_im_
            True

            >>> profile = Profile(preferences_rk=[
            ...     [1, 0, 3, 2],
            ...     [1, 2, 0, 3],
            ...     [2, 0, 3, 1],
            ...     [3, 0, 1, 2],
            ...     [3, 0, 2, 1],
            ... ])
            >>> rule = RuleIRV(im_option='exact')(profile)
            >>> rule.is_im_
            True

            >>> profile = Profile(preferences_rk=[
            ...     [0, 2, 1, 3],
            ...     [1, 0, 2, 3],
            ...     [1, 0, 3, 2],
            ...     [2, 3, 0, 1],
            ...     [3, 0, 2, 1],
            ... ])
            >>> rule = RuleIRV(im_option='exact')(profile)
            >>> rule.is_im_
            False

            >>> profile = Profile(preferences_rk=[
            ...     [0, 1, 2],
            ...     [0, 1, 2],
            ...     [0, 1, 2],
            ...     [1, 0, 2],
            ...     [2, 1, 0],
            ... ])
            >>> rule = RuleIRV(im_option='exact')(profile)
            >>> rule.is_im_
            False

            >>> profile = Profile(preferences_rk=[
            ...     [1, 0, 2],
            ...     [1, 2, 0],
            ...     [1, 2, 0],
            ...     [1, 2, 0],
            ...     [1, 2, 0],
            ... ])
            >>> rule = RuleIRV(im_option='exact')(profile)
            >>> rule.is_im_
            False

            >>> profile = Profile(preferences_rk=[
            ...     [0, 3, 1, 2],
            ...     [1, 2, 0, 3],
            ...     [2, 0, 3, 1],
            ...     [3, 1, 0, 2],
            ...     [3, 2, 1, 0],
            ... ])
            >>> rule = RuleIRV(im_option='exact')(profile)
            >>> rule.is_im_
            True

            >>> profile = Profile(preferences_rk=[
            ...     [0, 2, 3, 1, 4],
            ...     [1, 2, 0, 4, 3],
            ...     [3, 2, 1, 4, 0],
            ...     [4, 2, 0, 3, 1],
            ...     [4, 2, 1, 3, 0],
            ... ])
            >>> rule = RuleIRV(im_option='exact')(profile)
            >>> rule.is_im_
            True

            >>> profile = Profile(preferences_rk=[
            ...     [0, 1, 2],
            ...     [1, 0, 2],
            ...     [1, 0, 2],
            ...     [2, 0, 1],
            ...     [2, 0, 1],
            ... ])
            >>> rule = RuleIRV(im_option='exact')(profile)
            >>> rule.is_im_v_with_candidates_(4)
            (True, array([1., 0., 0.]))
        """
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
        if (scores_tot_begin_r[0, self.w_] + (self.w_ == 0)
                > n_s + 1 - scores_tot_begin_r[0, self.w_]):  # pragma: no cover
            # TO DO: Investigate whether this case can actually happen.
            self._reached_uncovered_code()
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
            if equal_false(self.eb_.is_im_):
                self.mylog("IM impossible (since it is impossible for Exhaustive Ballot)", 2)
                self._v_im_for_c[:] = False
                # Other variables will be updated in ``_im_preliminary_checks_general``.

    def _im_preliminary_checks_v_subclass_(self, v):
        # Pretest based on Exhaustive Ballot
        if self.im_option == "exact":
            if np.any(np.isneginf(self._v_im_for_c[v, :])):
                # ``self.eb_.im_option = exact``
                candidates_im_v = self.eb_.is_im_v_with_candidates_(v)[1]
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
            >>> rule = RuleIRV()(profile)
            >>> rule.example_path_tm_
            {0: array([2, 1, 0]), 1: None, 2: array([0, 1, 2])}
        """
        # self.eb_.tm_option = self.tm_option
        return self.eb_.example_path_tm_

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
            >>> rule = RuleIRV()(profile)
            >>> rule.sufficient_coalition_size_tm_
            array([5., 0., 4.])
        """
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

    def _um_preliminary_checks_c_(self, c):
        if self.um_option not in {'fast', 'lazy'} or self.cm_option not in {'fast', 'lazy'}:
            if (
                self.w_ == self.profile_.condorcet_winner_rk_ctb
                and not self.profile_.c_might_be_there_when_cw_is_eliminated_irv_style[c]
            ):
                # Impossible to manipulate with n_m manipulators
                self._candidates_um[c] = False

    def _um_aux_fast(self, c, n_m, preferences_borda_s):
        """Fast algorithm used for UM.

        Parameters
        ----------
        c : int
            Candidate for which we want to manipulate.
        n_m : int
            Number of manipulators.
        preferences_borda_s : ndarray
            Preferences of the sincere voters (in Borda format).

        Returns
        -------
        tuple
            ``(manip_found_fast, example_path_fast)``.

            * ``manip_found_fast``: Boolean. Whether a manipulation was found or not.
            * ``example_path_fast``: An example of elimination path that realizes the manipulation with ``n_m``
              manipulators. ``example_path_fast[k]`` is the ``k``-th candidate eliminated. If no manipulation is
              found, ``example_path`` is NaN.

        Examples
        --------
            >>> profile = Profile(preferences_rk=[
            ...     [0, 2, 1],
            ...     [1, 2, 0],
            ...     [1, 2, 0],
            ...     [2, 0, 1],
            ...     [2, 1, 0],
            ... ])
            >>> rule = RuleIRV()(profile)
            >>> rule.candidates_um_
            array([nan, nan,  0.])

            >>> profile = Profile(preferences_rk=[
            ...     [0, 1, 3, 2],
            ...     [1, 0, 2, 3],
            ...     [1, 2, 0, 3],
            ...     [2, 0, 3, 1],
            ...     [2, 3, 1, 0],
            ... ])
            >>> rule = RuleIRV()(profile)
            >>> rule.candidates_um_
            array([nan,  0., nan, nan])

            >>> profile = Profile(preferences_rk=[
            ...     [0, 3, 1, 2],
            ...     [1, 0, 2, 3],
            ...     [1, 2, 0, 3],
            ...     [2, 3, 0, 1],
            ...     [3, 0, 2, 1],
            ... ])
            >>> rule = RuleIRV()(profile)
            >>> rule.candidates_um_
            array([ 0., nan, nan,  1.])

            >>> profile = Profile(preferences_ut=[
            ...     [ 0.5,  0.5, -1. ,  0.5],
            ...     [-1. ,  0. ,  0. ,  0. ],
            ...     [ 0. ,  1. ,  0. ,  0.5],
            ...     [ 0.5,  0. ,  1. , -0.5],
            ...     [-0.5, -0.5,  1. ,  0.5],
            ... ], preferences_rk=[
            ...     [0, 3, 1, 2],
            ...     [1, 2, 3, 0],
            ...     [1, 3, 2, 0],
            ...     [2, 0, 1, 3],
            ...     [2, 3, 0, 1],
            ... ])
            >>> rule = RuleIRV()(profile)
            >>> rule.candidates_um_
            array([ 1.,  0., nan, nan])

            >>> profile = Profile(preferences_rk=[
            ...     [0, 1, 2],
            ...     [0, 2, 1],
            ...     [1, 0, 2],
            ...     [2, 0, 1],
            ...     [2, 0, 1],
            ... ])
            >>> rule = RuleIRV(fast_algo='minus_max')(profile)
            >>> rule.candidates_um_
            array([ 0., nan, nan])

            >>> profile = Profile(preferences_rk=[
            ...     [0, 1, 2],
            ...     [0, 2, 1],
            ...     [1, 0, 2],
            ...     [2, 0, 1],
            ...     [2, 0, 1],
            ... ])
            >>> rule = RuleIRV(fast_algo='hardest_first')(profile)
            >>> rule.candidates_um_
            array([ 0., nan, nan])
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
                    raise NotImplementedError("Unknown fast algorithm: " + format(self.fast_algo))
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

        Parameters
        ----------
        c : int
            Candidate for which we want to manipulate.
        n_m : int
            Number of manipulators.
        preferences_borda_s : ndarray
            Preferences of the sincere voters (in Borda format).

        Returns
        -------
        tuple
            ``(manip_found_exact, example_path)``.

            * ``manip_found_exact``: Boolean. Whether a manipulation was found or not.
            * ``example_path``: An example of elimination path that realizes the manipulation with ``n_m`` manipulators.
              ``example_path[k]`` is the ``k``-th candidate eliminated. If the manipulation is impossible,
              ``example_path`` is NaN.

        Examples
        --------
            >>> profile = Profile(preferences_ut=[
            ...     [ 0. ,  0. , -1. ,  0. ],
            ...     [ 0. , -0.5,  0. , -1. ],
            ...     [ 1. ,  1. ,  1. ,  0.5],
            ...     [ 1. ,  0.5,  1. , -0.5],
            ...     [-1. , -0.5,  0. ,  0. ],
            ... ], preferences_rk=[
            ...     [0, 1, 3, 2],
            ...     [0, 2, 1, 3],
            ...     [1, 2, 0, 3],
            ...     [2, 0, 1, 3],
            ...     [3, 2, 1, 0],
            ... ])
            >>> rule = RuleIRV(um_option='exact')(profile)
            >>> rule.candidates_um_
            array([1., 0., 0., 0.])

            >>> profile = Profile(preferences_ut=[
            ...     [ 0.5,  0.5, -1. ,  0. , -0.5],
            ...     [ 0. ,  1. , -1. , -1. ,  1. ],
            ...     [-0.5,  0.5,  1. ,  0.5, -1. ],
            ...     [ 0. ,  0.5, -1. ,  1. ,  0. ],
            ...     [ 0.5, -0.5,  1. ,  0.5,  1. ],
            ... ], preferences_rk=[
            ...     [0, 1, 3, 4, 2],
            ...     [1, 4, 0, 3, 2],
            ...     [2, 3, 1, 0, 4],
            ...     [3, 1, 4, 0, 2],
            ...     [4, 2, 0, 3, 1],
            ... ])
            >>> rule = RuleIRV(um_option='exact')(profile)
            >>> rule.candidates_um_
            array([0., 0., 0., 1., 0.])

            >>> profile = Profile(preferences_ut=[
            ...     [ 1. ,  0.5,  1. , -1. , -1. ],
            ...     [ 0.5, -0.5, -0.5,  0.5, -0.5],
            ...     [ 0. ,  0. ,  0. , -0.5, -0.5],
            ...     [ 0.5,  0. ,  1. ,  0.5,  0.5],
            ...     [-1. , -0.5,  0.5,  0. ,  1. ],
            ... ], preferences_rk=[
            ...     [0, 2, 1, 4, 3],
            ...     [0, 3, 1, 4, 2],
            ...     [1, 2, 0, 3, 4],
            ...     [2, 3, 4, 0, 1],
            ...     [4, 2, 3, 1, 0],
            ... ])
            >>> rule = RuleIRV(um_option='exact')(profile)
            >>> rule.candidates_um_
            array([0., 0., 0., 0., 0.])

            >>> profile = Profile(preferences_ut=[
            ...     [ 0.5,  0.5, -0.5, -1. ],
            ...     [ 1. ,  1. ,  1. , -1. ],
            ...     [-1. ,  0. , -0.5, -1. ],
            ...     [ 0. ,  0. ,  1. , -0.5],
            ...     [ 1. ,  0. ,  1. ,  0. ],
            ...     [-1. , -0.5, -1. ,  0. ],
            ...     [ 0. , -1. ,  0.5,  1. ],
            ... ], preferences_rk=[
            ...     [0, 1, 2, 3],
            ...     [1, 0, 2, 3],
            ...     [1, 2, 3, 0],
            ...     [2, 0, 1, 3],
            ...     [2, 0, 1, 3],
            ...     [3, 1, 0, 2],
            ...     [3, 2, 0, 1],
            ... ])
            >>> rule = RuleIRV(um_option='exact')(profile)
            >>> rule.candidates_um_
            array([1., 0., 0., 0.])
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
        if max_score + (most_serious_opponent < c) > n_s + n_m - max_score:  # pragma: no cover
            # TO DO: Investigate whether this case can actually happen.
            self._reached_uncovered_code()
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
        """
            >>> profile = Profile(preferences_rk=[
            ...     [0, 1, 2],
            ...     [1, 0, 2],
            ...     [1, 0, 2],
            ...     [2, 0, 1],
            ...     [2, 1, 0],
            ... ])
            >>> rule = RuleIRV(um_option='exact')(profile)
            >>> rule.candidates_um_
            array([0., 0., 0.])

            >>> profile = Profile(preferences_ut=[
            ...     [-0.5, -0.5, -0.5],
            ...     [-1. ,  0.5, -0.5],
            ...     [-1. ,  0. , -1. ],
            ...     [ 0. ,  1. ,  1. ],
            ...     [ 0. ,  0.5,  0.5],
            ... ], preferences_rk=[
            ...     [0, 1, 2],
            ...     [1, 2, 0],
            ...     [1, 2, 0],
            ...     [2, 1, 0],
            ...     [2, 1, 0],
            ... ])
            >>> rule = RuleIRV()(profile)
            >>> rule.candidates_um_
            array([0., 0., 0.])
        """
        if np.all(np.equal(self._candidates_um, False)):
            return
        if self.um_option == "exact":
            # In that case, we check Exhaustive Ballot first.
            self.eb_.um_option = "exact"
            if equal_false(self.eb_.is_um_):
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
        if equal_false(self.eb_.is_um_c_(c)):
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
            >>> rule = RuleIRV()(profile)
            >>> rule.example_path_cm_
            {0: None, 1: None, 2: None}
        """
        _ = self.candidates_cm_
        return self._example_path_cm

    def example_path_cm_c_(self, c):
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
            >>> rule = RuleIRV()(profile)
            >>> print(rule.example_path_cm_c_(0))
            None
        """
        _ = self.is_cm_c_(c)
        return self._example_path_cm[c]

    def example_ballots_cm_c_(self, c):
        """
        To manipulate for c against w.
        """
        if c == self.w_:
            return None
        if self._example_ballots_cm_c[c] is None:
            if not equal_true(self.is_cm_c_(c)):
                return None
            suggested_path = self.example_path_cm_c_(c)
            n_m = self.profile_.matrix_duels_ut[c, self.w_]
            preferences_borda_s = self.profile_.preferences_borda_rk[np.logical_not(self.v_wants_to_help_c_[:, c]), :]
            matrix_duels_temp = (preferences_ut_to_matrix_duels_ut(preferences_borda_s))
            self._example_ballots_cm_c[c] = self._example_ballots_cm_c_aux(
                c, n_m, suggested_path, preferences_borda_s, matrix_duels_temp)
        return self._example_ballots_cm_c[c]

    def example_ballots_cm_w_against_(self, w_other_rule):
        """
        To manipulate for w when in a sibling rule, the winner is different (w_other_rule).
        """
        if w_other_rule == self.w_:
            return None
        if self._example_ballots_cm_w_against[w_other_rule] is None:
            suggested_path = self.elimination_path_
            n_m = self.profile_.matrix_duels_ut[self.w_, w_other_rule]
            preferences_borda_s = self.profile_.preferences_borda_rk[
                self.profile_.preferences_ut[:, w_other_rule] >= self.profile_.preferences_ut[:, self.w_], :]
            matrix_duels_temp = (preferences_ut_to_matrix_duels_ut(preferences_borda_s))
            self._example_ballots_cm_w_against[w_other_rule] = self._example_ballots_cm_c_aux(
                self.w_, n_m, suggested_path, preferences_borda_s, matrix_duels_temp)
        return self._example_ballots_cm_w_against[w_other_rule]

    def _example_ballots_cm_c_aux(self, c, n_m, suggested_path, preferences_borda_s, matrix_duels_temp):
        """Example of manipulating ballots (in rk format).

        Parameters
        ----------
        c : int
            Candidate.

        Returns
        -------
        ballot_rk_m : ndarray or None
            ``ballot_rk_m[v, k]`` is the candidate placed in k-th position on the v-th manipulator's ballot.
        """
        # print(c)
        # print(n_m)
        # print(suggested_path)
        # print(preferences_borda_s)
        # print(matrix_duels_temp)
        candidates = np.array(range(self.profile_.n_c))
        ballots_m = [[] for _ in range(n_m)]

        # Step 1: ensure the elimination path
        # And consequences on the majority matrix
        scores_m_begin_r = np.zeros(self.profile_.n_c)  # Score due to manipulators at the beginning of the round
        is_candidate_alive_begin_r = np.ones(self.profile_.n_c, dtype=np.bool)
        current_top_v = np.array(- np.ones(n_m))  # -1 means that manipulator v is available
        candidates_to_put_in_ballot = np.ones((n_m, self.profile_.n_c), dtype=np.bool)
        for r in range(self.profile_.n_c - 1):
            self.mylogv("cm_aux: r =", r, 3)
            scores_tot_begin_r = np.full(self.profile_.n_c, np.nan)
            scores_tot_begin_r[is_candidate_alive_begin_r] = np.sum(np.equal(
                preferences_borda_s[:, is_candidate_alive_begin_r],
                np.max(preferences_borda_s[:, is_candidate_alive_begin_r], 1)[:, np.newaxis]
            ), 0)
            self.mylogv("cm_aux: scores_s_begin_r =", scores_tot_begin_r, 3)
            self.mylogv("cm_aux: scores_m_begin_r =", scores_m_begin_r, 3)
            scores_tot_begin_r += scores_m_begin_r
            self.mylogv("cm_aux: scores_tot_begin_r =", scores_tot_begin_r, 3)
            d = suggested_path[r]
            self.mylogv("cm_aux: d =", d, 3)
            scores_m_new_r = np.zeros(self.profile_.n_c, dtype=np.int)
            scores_m_new_r[is_candidate_alive_begin_r] = np.maximum(
                0,
                scores_tot_begin_r[d] - scores_tot_begin_r[is_candidate_alive_begin_r]
                + (candidates[is_candidate_alive_begin_r] > d))
            self.mylogv("cm_aux: scores_m_new_r =", scores_m_new_r, 3)
            # Update variables for next round
            scores_m_begin_r = scores_m_begin_r + scores_m_new_r
            if np.sum(scores_m_begin_r) > n_m:
                raise NotImplementedError("Error: this should not happen.")
            scores_m_begin_r[d] = 0
            is_candidate_alive_begin_r[d] = False
            # We need to attribute manipulator's votes to specific manipulators. This is done arbitrarily,
            # and has consequences on the majority matrix.
            free_manipulators = np.where(current_top_v == -1)[0]
            i_manipulator = 0
            for e in range(self.profile_.n_c):
                n_manip_new_e = scores_m_new_r[e]
                for k in range(n_manip_new_e):
                    manipulator = free_manipulators[i_manipulator]
                    ballots_m[manipulator].append(e)
                    current_top_v[manipulator] = e
                    candidates_to_put_in_ballot[manipulator, e] = False
                    matrix_duels_temp[e, candidates_to_put_in_ballot[manipulator, :]] += 1
                    i_manipulator += 1
            current_top_v[current_top_v == d] = -1
            self.mylogv("cm_aux: current_top_v =", current_top_v, 3)
            self.mylogm("cm_aux: matrix_duels_temp =", matrix_duels_temp, 3)

        # Step 2
        # Ensure that no candidate ``!= c`` is Condorcet winner. If ``c`` is not yet in all ballots, put it.
        for manipulator in np.where(candidates_to_put_in_ballot[:, c])[0]:
            ballots_m[manipulator].append(c)
            candidates_to_put_in_ballot[manipulator, c] = False
            matrix_duels_temp[c, candidates_to_put_in_ballot[manipulator, :]] += 1
        self.mylogv("cm_aux: Adding to all ballots c =", c, 3)
        self.mylogm("cm_aux: matrix_duels_temp =", matrix_duels_temp, 3)
        # If some candidates already have some non-victories in the matrix of duels, they can safely be put in the
        # ballots. Maybe this will generate non-victories for other candidates, etc.
        candidates_ok = np.zeros(self.profile_.n_c, dtype=np.bool)
        candidates_ok[c] = True
        i_found_a_new_ok = True
        while i_found_a_new_ok:
            not_yet_ok = np.where(np.logical_not(candidates_ok))[0]
            if not_yet_ok.shape[0] == 0:
                self.mylog("cm_aux: Decondorcification succeeded", 3)
                self._example_ballots_cm_c[c] = np.array(ballots_m)
                return self._example_ballots_cm_c[c]
            csd_penalties = np.zeros(self.profile_.n_c)  # Sum of losses in absolute defeats
            for d in not_yet_ok:
                csd_penalties[d] = np.sum(np.maximum(
                    matrix_duels_temp[:, d] - self.profile_.n_v / 2, 0
                ))
            max_penalty = np.max(csd_penalties)
            if max_penalty == 0:
                i_found_a_new_ok = False
            else:
                i_found_a_new_ok = True
                d = np.where(csd_penalties == max_penalty)[0][-1]
                candidates_ok[d] = True
                for manipulator in np.where(candidates_to_put_in_ballot[:, d])[0]:
                    ballots_m[manipulator].append(d)
                    candidates_to_put_in_ballot[manipulator, d] = False
                    matrix_duels_temp[d, candidates_to_put_in_ballot[manipulator, :]] += 1
                self.mylogv("cm_aux: Found a non-Condorcet d =", d, 3)
                self.mylogm("cm_aux: matrix_duels_temp =", matrix_duels_temp, 3)

        # Step 3
        # Some candidates are left, who do not have non-victories yet. We will put them in the ballots,
        # while favoring a Condorcet cycle 0 > 1 > ... > n_c-1 > 0. N.B.: In practice, this step seems never necessary.
        self.mylog("cm_aux: Step 3 needed", 1)
        for manipulator in range(n_m):
            candidate_start = manipulator % self.profile_.n_c
            for d in itertools.chain(range(candidate_start, self.profile_.n_c), range(candidate_start)):
                if candidates_to_put_in_ballot[manipulator, d]:
                    ballots_m[manipulator].append(d)
                    candidates_to_put_in_ballot[manipulator, d] = False
                    matrix_duels_temp[d, candidates_to_put_in_ballot[manipulator, :]] += 1
        return np.array(ballots_m)

    @cached_property
    def _cm_is_initialized_general_subclass_(self):
        self._example_path_cm = {c: None for c in range(self.profile_.n_c)}
        # Rule: each time ``self._sufficient_coalition_size_cm[c]`` is decreased, the corresponding elimination path
        # must be stored.
        return True

    def _cm_aux_fast(self, c, n_max, preferences_borda_s):
        """Fast algorithm used for CM.

        Parameters
        ----------
        c : int
            Candidate for which we want to manipulate.
        n_max : int
            Maximum number of manipulators allowed.

            * CM, complete and exact --> put the current value of ``sufficient_coalition_size[c] - 1`` (we want to find
              the best value for ``sufficient_coalition_size[c]``, even if it exceeds the number of manipulators)
            * CM, otherwise --> put the number of manipulators.

        preferences_borda_s : ndarray
            Preferences of the sincere voters (in Borda format).

        Returns
        -------
        tuple
            ``(n_manip_fast, example_path_fast)``.

            * ``n_manip_fast``: Integer or inf. If a manipulation is found, a sufficient number of manipulators is
              returned. If no manipulation is found, it is +inf.
            * ``example_path_fast``: An example of elimination path that realizes the manipulation with
              ``n_manip_fast`` manipulators. ``example_path_fast[k]`` is the ``k``-th candidate eliminated. If no
              manipulation is found, ``example_path`` is NaN.

        Examples
        --------
            >>> profile = Profile(preferences_rk=[
            ...     [0, 1, 2],
            ...     [1, 0, 2],
            ...     [1, 2, 0],
            ...     [2, 0, 1],
            ...     [2, 1, 0],
            ... ])
            >>> rule = RuleIRV(fast_algo='minus_max')(profile)
            >>> rule.candidates_cm_
            array([nan,  0., nan])

            >>> profile = Profile(preferences_rk=[
            ...     [0, 1, 2],
            ...     [1, 0, 2],
            ...     [1, 2, 0],
            ...     [2, 0, 1],
            ...     [2, 1, 0],
            ... ])
            >>> rule = RuleIRV(fast_algo='hardest_first')(profile)
            >>> rule.candidates_cm_
            array([nan,  0., nan])
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
                    raise NotImplementedError("Unknown fast algorithm: " + format(self.fast_algo))
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

        Parameters
        ----------
        suggested_path : List
            A suggested path of elimination.
        preferences_borda_s : ndarray
            Preferences of the sincere voters (in Borda format).

        Returns
        -------
        n_manip_slow : int
            Number of manipulators needed to manipulate with ``suggested_path``.

        Examples
        --------
            >>> profile = Profile(preferences_rk=[
            ...     [0, 2, 3, 1],
            ...     [1, 2, 3, 0],
            ...     [1, 2, 3, 0],
            ...     [2, 1, 0, 3],
            ...     [3, 2, 0, 1],
            ... ])
            >>> rule = RuleIRV(cm_option='slow')(profile)
            >>> rule.is_cm_
            nan
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

        Parameters
        ----------
        c : int
            Candidate for which we want to manipulate.
        n_max : int
            Maximum number of manipulators allowed.

            * CM, optimize_bounds and exact --> put the current value of ``sufficient_coalition_size[c] - 1`` (we want
              to find the best value for ``sufficient_coalition_size[c]``, even if it exceeds the number of
              manipulators)
            * CM, otherwise --> put the number of manipulators.

        n_min : int
            When we know that ``n_min`` manipulators are needed (necessary coalition size).
        optimize_bounds : bool
            True iff we need to continue, even after a manipulation is found.
        suggested_path : List
            A suggested path of elimination.
        preferences_borda_s : ndarray
            Preferences of the sincere voters (in Borda format).

        Returns
        -------
        tuple
            ``(n_manip_final, example_path, quick_escape)``.

            * ``n_manip_final``: Integer or +inf. If manipulation is impossible with ``<= n_max`` manipulators, it is
              +inf. If manipulation is possible (with ``<= n_max``):

                * If ``optimize_bounds`` is True, it is the minimal number of manipulators.
                * Otherwise, it is a number of manipulators that allow this manipulation (not necessarily minimal).

            * ``example_path``: An example of elimination path that realizes the manipulation with ``n_manip_final``
              manipulators. ``example_path[k]`` is the ``k``-th candidate eliminated. If the manipulation is impossible,
              ``example_path`` is NaN.
            * ``quick_escape``: Boolean. True if we get out without optimizing ``n_manip_final``.

        Examples
        --------
            >>> profile = Profile(preferences_rk=[
            ...     [0, 1, 2, 3],
            ...     [1, 3, 0, 2],
            ...     [2, 3, 0, 1],
            ...     [2, 3, 1, 0],
            ...     [3, 0, 2, 1],
            ... ])
            >>> rule = RuleIRV(cm_option='exact')(profile)
            >>> rule.candidates_cm_
            array([0., 0., 0., 1.])

            >>> profile = Profile(preferences_rk=[
            ...     [1, 0, 4, 2, 3],
            ...     [1, 4, 2, 0, 3],
            ...     [2, 4, 3, 1, 0],
            ...     [3, 4, 1, 2, 0],
            ...     [4, 0, 3, 1, 2],
            ... ])
            >>> rule = RuleIRV(cm_option='exact')(profile)
            >>> rule.candidates_cm_
            array([1., 1., 1., 0., 1.])

            >>> profile = Profile(preferences_rk=[
            ...     [0, 3, 1, 4, 2],
            ...     [1, 0, 4, 2, 3],
            ...     [1, 2, 0, 3, 4],
            ...     [2, 1, 0, 4, 3],
            ...     [3, 4, 0, 2, 1],
            ...     [4, 2, 0, 1, 3],
            ... ])
            >>> rule = RuleIRV(cm_option='exact')(profile)
            >>> rule.candidates_cm_
            array([0., 1., 0., 0., 0.])

            >>> profile = Profile(preferences_rk=[
            ...     [0, 1, 2, 3],
            ...     [1, 2, 0, 3],
            ...     [2, 0, 3, 1],
            ...     [2, 3, 0, 1],
            ...     [3, 2, 1, 0],
            ... ])
            >>> rule = RuleIRV(cm_option='exact')(profile)
            >>> rule.necessary_coalition_size_cm_
            array([4., 3., 0., 4.])
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
        if max_score + (most_serious_opponent < c) > n_s + n_max_updated - max_score:  # pragma: no cover
            # TO DO: Investigate whether this case can actually happen.
            self._reached_uncovered_code()
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
            if equal_false(self.eb_.is_cm_):
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
        if not optimize_bounds and self._necessary_coalition_size_cm[c] > n_m:  # pragma: no cover
            # TO DO: Investigate whether this case can actually happen.
            self._reached_uncovered_code()
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
        """
            >>> profile = Profile(preferences_rk=[
            ...     [0, 2, 1, 3],
            ...     [1, 2, 0, 3],
            ...     [1, 3, 2, 0],
            ...     [2, 1, 0, 3],
            ...     [3, 2, 1, 0],
            ... ])
            >>> rule = RuleIRV(cm_option='exact')(profile)
            >>> rule.candidates_cm_
            array([0., 1., 0., 0.])

            >>> profile = Profile(preferences_rk=[
            ...     [0, 1, 2],
            ...     [0, 2, 1],
            ...     [1, 0, 2],
            ...     [1, 0, 2],
            ...     [2, 0, 1],
            ... ])
            >>> rule = RuleIRV(cm_option='exact')(profile)
            >>> rule.necessary_coalition_size_cm_
            array([0., 4., 5.])

            >>> profile = Profile(preferences_rk=[
            ...     [0, 1, 2],
            ...     [0, 2, 1],
            ...     [2, 1, 0],
            ...     [2, 1, 0],
            ...     [2, 1, 0],
            ... ])
            >>> rule = RuleIRV()(profile)
            >>> rule.necessary_coalition_size_cm_
            array([3., 2., 0.])

            >>> profile = Profile(preferences_rk=[
            ...     [1, 0, 2, 3, 4],
            ...     [2, 0, 1, 3, 4],
            ...     [2, 1, 3, 4, 0],
            ...     [3, 0, 4, 1, 2],
            ...     [4, 1, 3, 0, 2],
            ... ])
            >>> rule = RuleIRV(cm_option='slow')(profile)
            >>> rule.is_cm_
            nan
        """
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
        if self.eb_.example_path_cm_c_(c) is None:  # pragma: no cover - This should never happen.
            self._reached_uncovered_code()
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
