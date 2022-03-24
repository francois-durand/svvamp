# -*- coding: utf-8 -*-
"""
Created on 29 nov. 2018, 10:38
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
import random
import numpy as np
from svvamp.utils.constants import OPTIONS
from svvamp.utils.util_cache import cached_property, DeleteCacheMixin
from svvamp.utils import my_log, type_checker
from svvamp.utils.printing import printm, print_title, print_big_title
from svvamp.utils.misc import compute_next_subset_with_w, compute_next_borda_clever
from svvamp.utils.pseudo_bool import pseudo_bool, neginf_to_nan, neginf_to_zero, equal_true, equal_false, \
    pseudo_bool_not
from svvamp.preferences.profile import Profile
from svvamp.preferences.profile_um import ProfileUM
from svvamp.preferences.profile_subset_candidates import ProfileSubsetCandidates


class Rule(DeleteCacheMixin, my_log.MyLog):
    """A voting rule.

    Parameters
    ----------
    options_parameters : dict
        It is a dictionary of allowed and default options. Allowed is a minimal check that will
        be performed before launching big simulations, but other checks might be performed when setting the option.
        Allowed is either a list of values, like ['lazy', 'fast', 'exact'], or a function checking if the parameter
        is correct. Example: ``{'max_grade': dict(allowed=np.isfinite, default=1), 'option_example': dict(allowed=[42,
        51], default=42)}``.
    kwargs
        Additional keyword parameters. See :attr:`options_parameters` for allowed and default options.
    with_two_candidates_reduces_to_plurality : bool
        ``True`` iff, when using this voting system with only two candidates, it amounts to Plurality (with voter and
        candidate tie-breaking).
    is_based_on_rk : bool
        ``True`` iff this voting system is based only on strict rankings (no cardinal information, indifference not
        allowed).
    is_based_on_ut_minus1_1 : bool
        ``True`` iff:

            *   This voting system is based only on utilities (not rankings, i.e. does not depend on how voters break
                ties in their own preferences),
            *   And for a ``c``-manipulator (IM or CM), it is optimal to pretend that ``c`` has utility 1 and other
                candidates have utility -1.
    meets_iia : bool
        ``True`` iff this voting system meets Independence of Irrelevant Alternatives.
    precheck_um : bool
        If ``True``, then before computing CM, we check whether there is UM.
    precheck_tm : bool
        If ``True``, then before computing CM, we check whether there is TM.
    precheck_icm : bool
        If ``True``, then before computing CM, we check whether there is ICM. Remark: when the voting system meets
        InfMC_c_ctb, then precheck on ICM will not do better than other basic prechecks.
    log_identity : str
        Cf. :class:`MyLog`.

    Notes
    -----
    This is an 'abstract' class. As an end-user, you should always use its subclasses :attr:`~svvamp.Approval`,
    :attr:`~svvamp.Plurality`, etc.

    This class and its subclasses are suitable for voting rules that are deterministic and anonymous (treating all
    voters equally). As a consequence, they are not neutral (because they need to break ties in totally symmetric
    situations). As of now, SVVAMP does not support other kinds of voting systems.

    **Ties in a voter's utilities**

    When a sincere voter ``v`` must provide a strict order in a specific voting system, she uses
    :attr:`~svvamp.Profile.preferences_rk`\ ``[v, :]`` (which breaks possible ties in her utilities).

    In contrast, to know if a voter ``v`` wants to manipulate for a candidate ``c`` against ``w``, we always use her
    utilities :attr:`~svvamp.Profile.preferences_ut`\ ``[v, :]``. If she attributes the same utility to ``w`` and
    ``c``, she is not interested in this manipulation.

    Some ordinal voting systems in SVVAMP may be adapted to accept weak orders of preferences as ballots. This is
    future work.

    **Ties in the result of an election**

    The voting system itself may need to break ties, for example if candidates ``c`` and ``d`` have the same score in
    a score-based system. The standard tie-breaking in SVVAMP, referred to as Candidate Tie-Breaking (CTB),
    consists of breaking ties by lowest index: ``c`` is favored over ``d`` if ``c`` < ``d``. This tie-breaking rule
    is used for example in 'A note on manipulability of large voting schemes' (Peleg, 1979). Future voting rules
    implemented as a subclass of ``Rule`` may use another tie-breaking rule.

    **Options for manipulation**

    Attributes allow to choose the algorithm used to compute different kinds of manipulation:
    :attr:`~cm_option`, :attr:`~icm_option`, :attr:`~im_option`, :attr:`~tm_option` and :attr:`~um_option`.

    To know what options are accepted for a given voting system, use :attr:`~svvamp.Rule.options_parameters`. Here
    is a non-exhaustive list of typical values for these options.

            *   ``'exact'``: Exact algorithm. Can always decide manipulation: it answers ``True`` or ``False``. Other
                algorithms may also answer ``numpy.nan``, which is the SVVAMP convention meaning that the algorithm was
                not able to decide. For a given voting system, if the exact algorithm runs in polynomial time,
                then it is the only accepted option.
            *   ``'slow'``: Non-polynomial algorithm, but not exact. For voting systems accepting this option,
                it is however faster than 'exact' (in a little-o sense) and more precise than 'fast'.
            *   ``'fast'``: Polynomial algorithm, not exact. If the exact algorithm runs in polynomial time,
                this option is not available.
            *   ``'lazy'``: Perform only some preliminary checks. Run in polynomial time (unless deciding the winner
                of the election is not polynomial, like for :class:`~svvamp.Kemeny`). Like other non-exact
                algorithms, it can decide manipulation to ``True``, ``False`` or return ``numpy.nan`` (undecided).

    For a given voting system, the default option is the most precise algorithm running in polynomial time.

    **Option for Independence of Irrelevant Alternatives (IIA)**

    The default algorithm for :attr:`~svvamp.Rule.not_iia` first performs some preliminary checks based on the
    known properties of the voting system under study. For example, if it meets the Condorcet criterion,
    then the algorithm exploits if. If it meets the majority favorite criterion (see below) and if
    :attr:`w_` is a majority favorite, then it decides IIA immediately.

    If the preliminary checks do not allow to decide, the default algorithm then uses brute force to test subsets of
    candidates including the sincere winner :attr:`w_`. This can be non-polynomial or
    non-exact, depending on the attribute :attr:`iia_subset_maximum_size`.

    **Implication diagram between criteria**

    Cf. corresponding attributes below for the definition of these criteria. See Durand et al.,
    'Condorcet Criterion and Reduction in Coalitional Manipulability'.

    ::

        Condorcet_c_ut_rel_ctb            ==>            Condorcet_c_ut_rel
        ||             Condorcet_c_rk_ctb ==>      Condorcet_c_rk       ||
        ||           ||        ||                   ||         ||       ||
        V            V         ||                   ||         V        V
        Condorcet_c_ut_abs_ctb            ==>            Condorcet_c_ut_abs
        ||                     ||                   ||                  ||
        ||                     V                    V                   ||
        ||     majority_favorite_c_rk_ctb ==> majority_favorite_c_rk    ||
        ||            ||                                  ||            ||
        V             V                                   V             V
        majority_favorite_c_ut_ctb        ==>        majority_favorite_ut_c
        ||                                                              ||
        V                                                               V
        IgnMC_c_ctb                       ==>                       IgnMC_c
        ||                                                              ||
        V                                                               V
        InfMC_c_ctb                       ==>                       InfMC_c
    """

    # Notes for developers:
    # In the code of this class, some special values are used.
    # * -np.inf (or None) means "I have not started to compute this value".
    # * np.nan means "I tried to compute this value, and I decided that I don't know".
    # As for np.inf, it really means + Infinity.
    #
    # 1) Methods for IM have an architecture of their own.
    # 2) Methods for TM and UM have essentially the same architecture and work on variables _candidates_.. and _is_..
    # 3) Methods for ICM and CM have essentially the same structure and focus on _sufficient_coalition_size_.. and
    # _necessary_coalition_size_cm.., then on _candidates_.. and _is_..
    # However, there are subtle differences of architecture between 1, 2 and 3 (cf. their docstrings).

    # Guideline:
    # When there is a polynomial exact algorithm, it should be the only option. The default option should be the most
    # precise algorithm among those running in polynomial time.
    # Exception: for iia_subset_maximum_size, default option is 2.

    options_parameters = {
        'iia_subset_maximum_size': {'allowed': type_checker.is_number, 'default': 2},
        'im_option': {'allowed': ['lazy', 'exact'], 'default': 'lazy'},
        'tm_option': {'allowed': ['lazy', 'exact'], 'default': 'exact'},
        'um_option': {'allowed': ['lazy', 'exact'], 'default': 'lazy'},
        'icm_option': {'allowed': ['lazy'], 'default': 'lazy'},
        'cm_option': {'allowed': ['lazy', 'exact'], 'default': 'lazy'}
    }

    def __init__(self,
                 with_two_candidates_reduces_to_plurality=False, is_based_on_rk=False,
                 is_based_on_ut_minus1_1=False, meets_iia=False,
                 precheck_um=True, precheck_tm=True, precheck_icm=True,
                 log_identity='RULE', **kwargs):
        # Log
        super().__init__()
        self.log_identity = log_identity
        # Basic properties of the voting system
        self.with_two_candidates_reduces_to_plurality = with_two_candidates_reduces_to_plurality
        self.is_based_on_rk = is_based_on_rk
        self.is_based_on_ut_minus1_1 = is_based_on_ut_minus1_1
        self.meets_iia = meets_iia
        self.precheck_um = precheck_um
        self.precheck_tm = precheck_tm
        self.precheck_icm = precheck_icm
        # Options
        self._iia_subset_maximum_size = None
        self._im_option = None
        self._tm_option = None
        self._um_option = None
        self._icm_option = None
        self._cm_option = None
        self._result_options = {}
        # Each option used for the RESULT of the election must be stored in self._result_options, even if there is also
        # an attribute for the option.
        self._initialize_options(**kwargs)
        # Initialize the computed variables
        self.profile_ = None

    @cached_property
    def _copy(self):
        """A simple copy of the rule.

        It is used to compute the results of a virtual election (for tests like IIA, TM, IM...).
        """
        return self.__class__(**self._result_options)

    def _initialize_options(self, **kwargs):
        """Initialize the options.

        Arguments: option1=value1, option2=value2, etc. Checks iff option1, option2 are known in
        self.options_parameters. For each option in self.options_parameters:
        * If a value is given, option is set to this value.
        * Otherwise, the default value is used.
        N.B.: validity check on the value are not performed here, but in the setter method for the option.
        """
        self.mylog('Initialize options', 1)
        for option in self.options_parameters:
            try:
                setattr(self, option, kwargs.pop(option))
            except KeyError:
                setattr(self, option, self.options_parameters[option]['default'])
        if kwargs:
            raise ValueError('Unknown option:', list(kwargs.keys()).pop())

    @property
    def options(self):
        """dict: The options. Key: name of the option. Value: value of the option."""
        return {k: getattr(self, k) for k in self.options_parameters.keys()}

    def update_options(self, options):
        """Update options.

        For example, instead of ``rule.cm_option='exact'``, you can write ``rule.update_options{'cm_option': 'exact'}``.

        Parameters
        ----------
        options : dict
            Key: option name. Value: option value.
        """
        for k, v in options.items():
            setattr(self, k, v)

    @classmethod
    def check_option_allowed(cls, option, value):
        """Check whether a pair (option, value) is allowed.

        Parameters
        ----------
        option : str
        value : object

        Examples
        --------
        Raise an error if the option is not in :attr:`options_parameters`:

            >>> Rule.check_option_allowed('not_existing_option', 42)
            Traceback (most recent call last):
            ValueError: Option 'not_existing_option' is unknown for Rule.

        Raise an error if the value is not authorized for this option:

            >>> Rule.check_option_allowed('cm_option', 'unexpected_value')
            Traceback (most recent call last):
            ValueError: 'cm_option' = 'unexpected_value' is not allowed in Rule.
        """
        if option not in cls.options_parameters:
            raise ValueError(f"Option {repr(option)} is unknown for {cls.__name__}.")
        allowed = cls.options_parameters[option]['allowed']
        this_value_is_allowed = (callable(allowed) and allowed(value)) or (not callable(allowed) and value in allowed)
        if not this_value_is_allowed:
            raise ValueError(f"'{option}' = {repr(value)} is not allowed in {cls.__name__}.")

    def log_(self, method_name):
        """Log corresponding to a particular manipulation method.
        """
        if '_iia_' in method_name:
            return self.log_iia_
        if '_im_' in method_name:
            return self.log_im_
        if '_icm_' in method_name:
            return self.log_icm_
        if '_tm_' in method_name:
            return self.log_tm_
        if '_um_' in method_name:
            return self.log_um_
        if '_cm_' in method_name:
            return self.log_cm_

    def __call__(self, profile):
        """
        Parameters
        ----------
        profile : Profile
        """
        self.delete_cache(suffix='_')
        self.profile_ = profile
        return self

    # %% Manipulation criteria of the voting system
    # In the subclass corresponding to a specific voting system, it is sufficient to redefine to True only the
    # strongest criteria that are met by the voting system.

    @cached_property
    def meets_condorcet_c_ut_rel_ctb(self):
        """Boolean. ``True`` iff the voting system meets the 'relative Condorcet criterion with ctb'. I.e.: if a
        candidate is a :attr:`~svvamp.Profile.condorcet_winner_ut_rel_ctb`, she wins.

        Implies: :attr:`~meets_condorcet_c_ut_rel`, :attr:`~meets_condorcet_c_ut_abs_ctb`.
        """
        return False

    @cached_property
    def meets_condorcet_c_rk_ctb(self):
        """Boolean. ``True`` iff the voting system meets the 'Condorcet criterion (rk) with ctb'. I.e.: if a
        candidate is a :attr:`~svvamp.Profile.condorcet_winner_rk_ctb`, she wins.

        Implies: :attr:`~meets_condorcet_c_rk`, :attr:`~meets_condorcet_c_ut_abs_ctb`,
        :attr:`~meets_majority_favorite_c_rk_ctb`.
        """
        return False

    @cached_property
    def meets_condorcet_c_ut_abs_ctb(self):
        """Boolean. ``True`` iff the voting system meets the 'absolute Condorcet criterion with ctb'. I.e.: if a
        candidate is a :attr:`~svvamp.Profile.condorcet_winner_ut_abs_ctb`, she wins.

        Is implied by: :attr:`~meets_condorcet_c_rk_ctb`, :attr:`~meets_condorcet_c_ut_rel_ctb`.

        Implies: :attr:`~meets_condorcet_c_ut_abs`, :attr:`~meets_majority_favorite_c_ut_ctb`.
        """
        return self.meets_condorcet_c_rk_ctb or self.meets_condorcet_c_ut_rel_ctb

    @cached_property
    def meets_majority_favorite_c_rk_ctb(self):
        """Boolean. ``True`` iff the voting system meets the 'majority favorite criterion (rk) with ctb'. I.e.:

            *   It :attr:`~meets_majority_favorite_c_rk`,
            *   And if :attr:`~svvamp.Profile.n_v`/2 voters rank candidate 0 first (rk), she wins.

        Is implied by: :attr:`~meets_condorcet_c_rk_ctb`.

        Implies: :attr:`~meets_majority_favorite_c_ut_ctb`, :attr:`~meets_majority_favorite_c_rk`.
        """
        return self.meets_condorcet_c_rk_ctb

    @cached_property
    def meets_majority_favorite_c_ut_ctb(self):
        """Boolean. ``True`` iff the voting system meets the 'majority favorite criterion (ut) with ctb'. I.e.:

            *   It :attr:`~meets_majority_favorite_c_ut`,
            *   And if :attr:`~svvamp.Profile.n_v`/2 voters strictly prefer candidate 0 to all other candidates,
                she wins.

        Is implied by: :attr:`~meets_condorcet_c_ut_abs_ctb`, :attr:`~meets_majority_favorite_c_rk_ctb`.

        Implies: :attr:`~meets_IgnMC_c_ctb`, :attr:`~meets_majority_favorite_c_ut`.
        """
        return self.meets_condorcet_c_ut_abs_ctb or self.meets_majority_favorite_c_rk_ctb

    @cached_property
    def meets_ignmc_c_ctb(self):
        """Boolean. ``True`` iff the voting system meets the 'ignorant majority coalition criterion with ctb'. I.e.:

            *   It :attr:`~meets_IgnMC_c`,
            *   And any ignorant coalition of size :attr:`~svvamp.Profile.n_v`/2 can make candidate 0 win.

        Is implied by: :attr:`~meets_majority_favorite_c_ut_ctb`.

        Implies: :attr:`~meets_InfMC_c_ctb`, :attr:`~meets_IgnMC_c`.
        """
        return self.meets_majority_favorite_c_ut_ctb

    @cached_property
    def meets_infmc_c_ctb(self):
        """Boolean. ``True`` iff the voting system meets the 'informed majority coalition criterion with ctb'. I.e.:

            *   It :attr:`~meets_InfMC_c`,
            *   And any informed coalition of size :attr:`~svvamp.Profile.n_v`/2 can make candidate 0 win.

        Is implied by: :attr:`~meets_IgnMC_c_ctb`.

        Implies: :attr:`~meets_InfMC_c`.
        """
        return self.meets_ignmc_c_ctb

    @cached_property
    def meets_condorcet_c_ut_rel(self):
        """Boolean. ``True`` iff the voting system meets the relative Condorcet criterion. I.e. if a candidate is a
        :attr:`~svvamp.Profile.condorcet_winner_ut_rel`, then she wins.

        Is implied by: :attr:`~meets_condorcet_c_ut_rel_ctb`.

        Implies: :attr:`~meets_condorcet_c_ut_abs`.
        """
        return self.meets_condorcet_c_ut_rel_ctb

    @cached_property
    def meets_condorcet_c_rk(self):
        """Boolean. ``True`` iff the voting system meets the Condorcet criterion (rk). I.e. if a candidate is a
        :attr:`~svvamp.Profile.condorcet_winner_rk`, then she wins.

        Is implied by: :attr:`~meets_condorcet_c_rk_ctb`.

        Implies: :attr:`~meets_condorcet_c_ut_abs`, :attr:`~meets_majority_favorite_c_rk`.
        """
        return self.meets_condorcet_c_rk_ctb

    @cached_property
    def meets_condorcet_c_ut_abs(self):
        """Boolean. ``True`` iff the voting system meets the absolute Condorcet criterion. I.e. if a candidate is a
        :attr:`~svvamp.Profile.condorcet_winner_ut_abs`, then she wins.

        Is implied by: :attr:`~meets_condorcet_c_rk`, :attr:`~meets_condorcet_c_ut_rel`,
        :attr:`~meets_condorcet_c_ut_abs_ctb`.

        Implies: :attr:`~meets_majority_favorite_c_ut`.
        """
        return self.meets_condorcet_c_rk or self.meets_condorcet_c_ut_rel or self.meets_condorcet_c_ut_abs_ctb

    @cached_property
    def meets_majority_favorite_c_rk(self):
        """Boolean. ``True`` iff the voting system meets the majority favorite criterion (rk). I.e. if strictly more
        than :attr:`~svvamp.Profile.n_v`/2 voters rank a candidate first (rk), then she wins.

        Is implied by: :attr:`~meets_condorcet_c_rk`, :attr:`~meets_majority_favorite_c_rk_ctb`.

        Implies: :attr:`~_meets_majority_favorite_c_ut`.
        """
        return self.meets_condorcet_c_rk or self.meets_majority_favorite_c_rk_ctb

    @cached_property
    def meets_majority_favorite_c_ut(self):
        """Boolean. ``True`` iff the voting system meets the majority favorite criterion (ut). I.e. if strictly more
        than :attr:`~svvamp.Profile.n_v`/2 voters strictly prefer a candidate to all others (ut), she wins.

        Is implied by: :attr:`~meets_condorcet_c_ut_abs`, :attr:`~meets_majority_favorite_c_ut_ctb`,
        :attr:`~meets_majority_favorite_c_rk`.

        Implies: :attr:`~meets_IgnMC_c`.
        """
        return (self.meets_condorcet_c_ut_abs or self.meets_majority_favorite_c_rk
                or self.meets_majority_favorite_c_ut_ctb)

    @cached_property
    def meets_ignmc_c(self):
        """Boolean. ``True`` iff the voting system meets the ignorant majority coalition criterion. I.e. any ignorant
        coalition of size strictly more than :attr:`~svvamp.Profile.n_v`/2 can make any candidate win. See Durand et
        al.: 'Condorcet Criterion and Reduction in Coalitional Manipulability'.

        *Ignorant* means that they can choose their ballot without knowing what other voters will do.

        Is implied by: :attr:`~meets_majority_favorite_c_ut`, :attr:`~meets_IgnMC_c_ctb`.

        Implies: :attr:`~meets_InfMC_c`.
        """
        return self.meets_majority_favorite_c_ut or self.meets_ignmc_c_ctb

    @cached_property
    def meets_infmc_c(self):
        """Boolean. ``True`` iff the voting system meets the informed majority coalition criterion. I.e. any informed
        coalition of size strictly more than :attr:`~svvamp.Profile.n_v`/2 can make any candidate win. See Durand et
        al.: 'Condorcet Criterion and Reduction in Coalitional Manipulability'.

        *Informed* means that they know other voters' ballots before choosing their own.

        Is implied by: :attr:`~meets_IgnMC_c`, :attr:`~meets_InfMC_c_ctb`.
        """
        return self.meets_ignmc_c or self.meets_infmc_c_ctb

    # %% Counting ballots
    #    Attributes to be implemented in subclasses.

    @cached_property
    def scores_(self):
        """Scores of the candidates in the election.

        See specific documentation for each voting rule. Typical type in most subclasses: 1d or 2d array. Typical
        behavior in most subclasses:

        * If ``scores_`` is a 1d array, then ``scores_[c]`` is the numerical score for candidate ``c``.
        * If ``scores_`` is a 2d array, then ``scores_[:, c]`` is the score vector for candidate ``c``.

        It is not mandatory to follow this behavior.
        """
        raise NotImplementedError

    # %% Counting ballots
    #    Attributes with default, but frequently redefined in subclasses.

    @cached_property
    def ballots_(self):
        """Ballots cast by the voters.

        Default type: 2d array of integers. Default behavior:
        ``ballots[v, k]`` = :attr:`~svvamp.Profile.preferences_rk`\ ``[v, k]``.
        """
        # This can be overridden by specific voting systems. This general behavior is ok only for ordinal voting systems
        # (and even in this case, it can be redefined in favor of something more practical).
        self.mylog("Compute ballots", 1)
        return self.profile_.preferences_rk

    @cached_property
    def w_(self):
        """Integer (winning candidate).

        Default behavior: the candidate with highest value in vector :attr:`scores_` is declared the winner. In case
        of a tie, the tied candidate with lowest index wins.
        """
        # This general method works only if scores are scalar and the best score wins.
        self.mylog("Compute winner", 1)
        if self.scores_.ndim == 1:
            return int(np.argmax(self.scores_))
        else:
            raise NotImplementedError

    @cached_property
    def candidates_by_scores_best_to_worst_(self):
        """1d array of integers. All candidates, sorted from the winner to the last candidate in the election's result.

        Default behavior: ``candidates_by_scores_best_to_worst[k]`` is the candidate with ``k``\ :sup:`th` highest
        value in :attr:`scores_`. By definition, ``candidates_by_scores_best_to_worst[0]`` = :attr:`w_`.
        """
        # This can be overridden by specific voting systems. This general method works only if scores are scalar and
        # the best score wins. If the lowest score wins, then candidates_by_scores_best_to_worst need to be sorted by
        # ascending score...
        self.mylog("Compute candidates_by_scores_best_to_worst", 1)
        return np.argsort(- self.scores_, kind='mergesort')

    @cached_property
    def v_might_im_for_c_(self):
        """2d array of booleans. `

        `v_might_im_for_c_[v, c]`` is True unless it is clearly and easily excluded that ``v`` has an individual
        manipulation in favor of ``c``. Typically, if ``v`` is not pivotal (at all), then it is False. Specific
        implementations of this function in subclasses can also test if ``v`` is interested in manipulating for ``c``,
        but it is not mandatory.

        A non-trivial redefinition of this function is useful for voting systems where computing IM is costly. For
        voting systems where it is cheap, it it not worth the effort.
        """
        return np.ones((self.profile_.n_v, self.profile_.n_c), dtype=np.bool)

    # %% Counting ballots
    #    Attributes that are almost always computed here in the superclass.

    @cached_property
    def score_w_(self):
        """Score of the sincere winner.

        Default type: number or 1d array.

        Default behavior:

        * If :attr:`scores_` is a 1d array, then ``score_w_`` is :attr:`w_`'s numerical score.
        * If :attr:`scores_` is a 2d array, then ``score_w_`` is :attr:`w_`'s score vector:
          ``score_w_ = scores_[:, w_]``.
        """
        # Exception: if scores are read in rows (Schulze, Ranked pairs), this needs to be redefined.
        self.mylog("Compute winner's score", 1)
        if self.scores_.ndim == 1:
            return self.scores_[self.w_]
        elif self.scores_.ndim == 2:
            return self.scores_[:, self.w_]
        else:
            raise NotImplementedError

    @cached_property
    def scores_best_to_worst_(self):
        """Scores of the candidates, from the winner to the last candidate of the election.

        Default type: 1d or 2d array.

        Default behavior: ``scores_best_to_worst_`` is derived from :attr:`scores_` and
        :attr:`candidates_by_scores_best_to_worst_`.

        If :attr:`scores_` is a 1d array, then so is ``scores_best_to_worst_``. It is defined by
        ``scores_best_to_worst_`` = ``scores_[candidates_by_scores_best_to_worst_]``. Then by definition,
        ``scores_best_to_worst_[0]`` = :attr:`score_w_`.

        If :attr:`scores_` is a 2d array, then so is ``scores_best_to_worst_``. It is defined by
        ``scores_best_to_worst_`` = ``scores_[:, candidates_by_scores_best_to_worst_]``. Then by definition,
        ``scores_best_to_worst_[:, 0]`` = :attr:`score_w_`.
        """
        # Exception: if scores are read in rows (Schulze, Ranked pairs), this needs to be redefined.
        self.mylog("Compute scores_best_to_worst", 1)
        if self.scores_.ndim == 1:
            return self.scores_[self.candidates_by_scores_best_to_worst_]
        elif self.scores_.ndim == 2:
            return self.scores_[:, self.candidates_by_scores_best_to_worst_]
        else:
            raise NotImplementedError

    @cached_property
    def total_utility_w_(self):
        """Float. The total utility for the sincere winner :attr:`w_`."""
        return self.profile_.total_utility_c[self.w_]

    @cached_property
    def mean_utility_w_(self):
        """Float. The mean utility for the sincere winner :attr:`w_`."""
        return self.profile_.mean_utility_c[self.w_]

    @cached_property
    def relative_social_welfare_w_(self):
        """Float. The relative social welfare for the sincere winner :attr:`w_`."""
        return self.profile_.relative_social_welfare_c[self.w_]

    # %% Condorcet efficiency and variants

    @cached_property
    def w_is_condorcet_admissible_(self):
        """Boolean. ``True`` iff the sincere winner :attr:`w_` is Condorcet-admissible.
        Cf. :attr:`~svvamp.Profile.condorcet_admissible_candidates`.
        """
        return self.profile_.condorcet_admissible_candidates[self.w_]

    @cached_property
    def w_is_not_condorcet_admissible_(self):
        """Boolean. ``True`` iff the sincere winner :attr:`w_` is not a Condorcet-admissible candidate (whether some
        exist or not). Cf. :attr:`~svvamp.Profile.condorcet_admissible_candidates`.
        """
        return not self.w_is_condorcet_admissible_

    @cached_property
    def w_missed_condorcet_admissible_(self):
        """Boolean. True iff the sincere winner :attr:`w_` is not a Condorcet-admissible candidate, despite the fact
        that at least one exists. Cf. :attr:`~svvamp.Profile.condorcet_admissible_candidates`.
        """
        return self.profile_.nb_condorcet_admissible > 0 and not self.w_is_condorcet_admissible_

    @cached_property
    def w_is_weak_condorcet_winner_(self):
        """Boolean. True iff the sincere winner :attr:`w_` is a Weak Condorcet winner.
        Cf. :attr:`~svvamp.Profile.weak_condorcet_winners`.
        """
        return self.profile_.weak_condorcet_winners[self.w_]

    @cached_property
    def w_is_not_weak_condorcet_winner_(self):
        """Boolean. True iff the sincere winner :attr:`w_` is not a Weak Condorcet winner (whether some exist or not).
        Cf. :attr:`~svvamp.Profile.weak_condorcet_winners`.
        """
        return not self.w_is_weak_condorcet_winner_

    @cached_property
    def w_missed_weak_condorcet_winner_(self):
        """Boolean. True iff the sincere winner :attr:`w_` is not a Weak Condorcet winner, despite the fact that at
        least one exists. Cf. :attr:`~svvamp.Profile.weak_condorcet_winners`.
        """
        return self.profile_.nb_weak_condorcet_winners > 0 and not self.w_is_weak_condorcet_winner_

    @cached_property
    def w_is_condorcet_winner_rk_ctb_(self):
        """Boolean. True iff the sincere winner :attr:`w_` is a :attr:`~svvamp.Profile.condorcet_winner_rk_ctb`.
        """
        return self.w_ == self.profile_.condorcet_winner_rk_ctb

    @cached_property
    def w_is_not_condorcet_winner_rk_ctb_(self):
        """Boolean. True iff the sincere winner :attr:`w_` is not a :attr:`~svvamp.Profile.condorcet_winner_rk_ctb`
        (whether one exists or not).
        """
        return not self.w_is_condorcet_winner_rk_ctb_

    @cached_property
    def w_missed_condorcet_winner_rk_ctb_(self):
        """Boolean. True iff the sincere winner :attr:`w_` is not the :attr:`~svvamp.Profile.condorcet_winner_rk_ctb`,
        despite the fact that she exists.
        """
        return not np.isnan(self.profile_.condorcet_winner_rk_ctb) and not self.w_is_condorcet_winner_rk_ctb_

    @cached_property
    def w_is_condorcet_winner_rk_(self):
        """Boolean. True iff the sincere winner :attr:`w_` is a :attr:`~svvamp.Profile.condorcet_winner_rk`.
        """
        return self.w_ == self.profile_.condorcet_winner_rk

    @cached_property
    def w_is_not_condorcet_winner_rk_(self):
        """Boolean. True iff the sincere winner :attr:`w_` is not a :attr:`~svvamp.Profile.condorcet_winner_rk`
        (whether one exists or not).
        """
        return not self.w_is_condorcet_winner_rk_

    @cached_property
    def w_missed_condorcet_winner_rk_(self):
        """Boolean. True iff the sincere winner :attr:`w_` is not the :attr:`~svvamp.Profile.condorcet_winner_rk`,
        despite the fact that she exists.
        """
        return not np.isnan(self.profile_.condorcet_winner_rk) and not self.w_is_condorcet_winner_rk_

    @cached_property
    def w_is_condorcet_winner_ut_rel_ctb_(self):
        """Boolean. True iff the sincere winner :attr:`w_` is a :attr:`~svvamp.Profile.condorcet_winner_ut_abs_ctb`.
        """
        return self.w_ == self.profile_.condorcet_winner_ut_rel_ctb

    @cached_property
    def w_is_not_condorcet_winner_ut_rel_ctb_(self):
        """Boolean. True iff the sincere winner :attr:`w_` is not a :attr:`~svvamp.Profile.condorcet_winner_ut_abs_ctb`
        (whether one exists or not).
        """
        return not self.w_is_condorcet_winner_ut_rel_ctb_

    @cached_property
    def w_missed_condorcet_winner_ut_rel_ctb_(self):
        """Boolean. True iff the sincere winner :attr:`w_` is not the
        :attr:`~svvamp.Profile.condorcet_winner_ut_abs_ctb`, despite the fact that she exists.
        """
        return not np.isnan(self.profile_.condorcet_winner_ut_rel_ctb) and not self.w_is_condorcet_winner_ut_rel_ctb_

    @cached_property
    def w_is_condorcet_winner_ut_rel_(self):
        """Boolean. True iff the sincere winner :attr:`w_` is a :attr:`~svvamp.Profile.condorcet_winner_ut_rel`.
        """
        return self.w_ == self.profile_.condorcet_winner_ut_rel

    @cached_property
    def w_is_not_condorcet_winner_ut_rel_(self):
        """Boolean. True iff the sincere winner :attr:`w_` is not a :attr:`~svvamp.Profile.condorcet_winner_ut_rel`
        (whether one exists or not).
        """
        return not self.w_is_condorcet_winner_ut_rel_

    @cached_property
    def w_missed_condorcet_winner_ut_rel_(self):
        """Boolean. True iff the sincere winner :attr:`w_` is not the :attr:`~svvamp.Profile.condorcet_winner_ut_rel`,
        despite the fact that she exists.
        """
        return not np.isnan(self.profile_.condorcet_winner_ut_rel) and not self.w_is_condorcet_winner_ut_rel_

    @cached_property
    def w_is_condorcet_winner_ut_abs_ctb_(self):
        """Boolean. True iff the sincere winner :attr:`w_` is a :attr:`~svvamp.Profile.condorcet_winner_ut_abs_ctb`.
        """
        return self.w_ == self.profile_.condorcet_winner_ut_abs_ctb

    @cached_property
    def w_is_not_condorcet_winner_ut_abs_ctb_(self):
        """Boolean. True iff the sincere winner :attr:`w_` is not a :attr:`~svvamp.Profile.condorcet_winner_ut_abs_ctb`
        (whether one exists or not).
        """
        return not self.w_is_condorcet_winner_ut_abs_ctb_

    @cached_property
    def w_missed_condorcet_winner_ut_abs_ctb_(self):
        """Boolean. True iff the sincere winner :attr:`w_` is not the
        :attr:`~svvamp.Profile.condorcet_winner_ut_abs_ctb`, despite the fact that she exists.
        """
        return not np.isnan(self.profile_.condorcet_winner_ut_abs_ctb) and not self.w_is_condorcet_winner_ut_abs_ctb_

    @cached_property
    def w_is_condorcet_winner_ut_abs_(self):
        """Boolean. True iff the sincere winner :attr:`w_` is a :attr:`~svvamp.Profile.condorcet_winner_ut_abs`.
        """
        return self.w_ == self.profile_.condorcet_winner_ut_abs

    @cached_property
    def w_is_not_condorcet_winner_ut_abs_(self):
        """Boolean. True iff the sincere winner :attr:`w_` is not a :attr:`~svvamp.Profile.condorcet_winner_ut_abs`
        (whether one exists or not).
        """
        return not self.w_is_condorcet_winner_ut_abs_

    @cached_property
    def w_missed_condorcet_winner_ut_abs_(self):
        """Boolean. True iff the sincere winner :attr:`w_` is not the :attr:`~svvamp.Profile.condorcet_winner_ut_abs`,
        despite the fact that she exists.
        """
        return not np.isnan(self.profile_.condorcet_winner_ut_abs) and not self.w_is_condorcet_winner_ut_abs_

    @cached_property
    def w_is_resistant_condorcet_winner_(self):
        """Boolean. True iff the sincere winner :attr:`w_` is a :attr:`~svvamp.Profile.resistant_condorcet_winner`.
        """
        return self.w_ == self.profile_.resistant_condorcet_winner

    @cached_property
    def w_is_not_resistant_condorcet_winner_(self):
        """Boolean. True iff the sincere winner :attr:`w_` is not a :attr:`~svvamp.Profile.resistant_condorcet_winner`
        (whether one exists or not).
        """
        return not self.w_is_resistant_condorcet_winner_

    @cached_property
    def w_missed_resistant_condorcet_winner_(self):
        """Boolean. True iff the sincere winner :attr:`w_` is not the
        :attr:`~svvamp.Profile.resistant_condorcet_winner`, despite the fact that she exists.
        """
        return not np.isnan(self.profile_.resistant_condorcet_winner) and not self.w_is_resistant_condorcet_winner_

    def demo_profile_(self, log_depth=1):
        """Demonstrate the attributes of the loaded profile.

        Parameters
        ----------
        log_depth : int
            Integer from 0 (basic info) to 3 (verbose).
        """
        print_big_title('Profile Class')
        self.profile_.demo(log_depth=log_depth)

    def demo_results_(self, log_depth=1):
        """Demonstrate the methods related to the result of the election (without manipulation).

        Parameters
        ----------
        log_depth : int
            Integer from 0 (basic info) to 3 (verbose).
        """
        old_log_depth = self.log_depth
        self.log_depth = log_depth

        print_big_title('Election Results')

        print_title("Results")
        printm("profile_.preferences_ut (reminder) =", self.profile_.preferences_ut)
        printm("profile_.preferences_rk (reminder) =", self.profile_.preferences_rk)
        printm("ballots =", self.ballots_)
        printm("scores =", self.scores_)
        printm("candidates_by_scores_best_to_worst", self.candidates_by_scores_best_to_worst_)
        printm("scores_best_to_worst", self.scores_best_to_worst_)
        print("w =", self.w_)
        print("score_w =", self.score_w_)
        print("total_utility_w =", self.total_utility_w_)

        print_title("Condorcet efficiency (rk)")
        print("w (reminder) =", self.w_)
        print("")
        print("condorcet_winner_rk_ctb =", self.profile_.condorcet_winner_rk_ctb)
        print("w_is_condorcet_winner_rk_ctb =", self.w_is_condorcet_winner_rk_ctb_)
        print("w_is_not_condorcet_winner_rk_ctb =", self.w_is_not_condorcet_winner_rk_ctb_)
        print("w_missed_condorcet_winner_rk_ctb =", self.w_missed_condorcet_winner_rk_ctb_)
        print("")
        print("condorcet_winner_rk =", self.profile_.condorcet_winner_rk)
        print("w_is_condorcet_winner_rk =", self.w_is_condorcet_winner_rk_)
        print("w_is_not_condorcet_winner_rk =", self.w_is_not_condorcet_winner_rk_)
        print("w_missed_condorcet_winner_rk =", self.w_missed_condorcet_winner_rk_)

        print_title("Condorcet efficiency (relative)")
        print("w (reminder) =", self.w_)
        print("")
        print("condorcet_winner_ut_rel_ctb =", self.profile_.condorcet_winner_ut_rel_ctb)
        print("w_is_condorcet_winner_ut_rel_ctb =", self.w_is_condorcet_winner_ut_rel_ctb_)
        print("w_is_not_condorcet_winner_ut_rel_ctb =", self.w_is_not_condorcet_winner_ut_rel_ctb_)
        print("w_missed_condorcet_winner_ut_rel_ctb =", self.w_missed_condorcet_winner_ut_rel_ctb_)
        print("")
        print("condorcet_winner_ut_rel =", self.profile_.condorcet_winner_ut_rel)
        print("w_is_condorcet_winner_ut_rel =", self.w_is_condorcet_winner_ut_rel_)
        print("w_is_not_condorcet_winner_ut_rel =", self.w_is_not_condorcet_winner_ut_rel_)
        print("w_missed_condorcet_winner_ut_rel =", self.w_missed_condorcet_winner_ut_rel_)

        print_title("Condorcet efficiency (absolute)")
        print("w (reminder) =", self.w_)
        print("")
        printm("condorcet_admissible_candidates =", self.profile_.condorcet_admissible_candidates)
        print("w_is_condorcet_admissible =", self.w_is_condorcet_admissible_)
        print("w_is_not_condorcet_admissible =", self.w_is_not_condorcet_admissible_)
        print("w_missed_condorcet_admissible =", self.w_missed_condorcet_admissible_)
        print("")
        printm("weak_condorcet_winners =", self.profile_.weak_condorcet_winners)
        print("w_is_weak_condorcet_winner =", self.w_is_weak_condorcet_winner_)
        print("w_is_not_weak_condorcet_winner =", self.w_is_not_weak_condorcet_winner_)
        print("w_missed_weak_condorcet_winner =", self.w_missed_weak_condorcet_winner_)
        print("")
        print("condorcet_winner_ut_abs_ctb =", self.profile_.condorcet_winner_ut_abs_ctb)
        print("w_is_condorcet_winner_ut_abs_ctb =", self.w_is_condorcet_winner_ut_abs_ctb_)
        print("w_is_not_condorcet_winner_ut_abs_ctb =", self.w_is_not_condorcet_winner_ut_abs_ctb_)
        print("w_missed_condorcet_winner_ut_abs_ctb =", self.w_missed_condorcet_winner_ut_abs_ctb_)
        print("")
        print("condorcet_winner_ut_abs =", self.profile_.condorcet_winner_ut_abs)
        print("w_is_condorcet_winner_ut_abs =", self.w_is_condorcet_winner_ut_abs_)
        print("w_is_not_condorcet_winner_ut_abs =", self.w_is_not_condorcet_winner_ut_abs_)
        print("w_missed_condorcet_winner_ut_abs =", self.w_missed_condorcet_winner_ut_abs_)
        print("")
        print("resistant_condorcet_winner =", self.profile_.resistant_condorcet_winner)
        print("w_is_resistant_condorcet_winner =", self.w_is_resistant_condorcet_winner_)
        print("w_is_not_resistant_condorcet_winner =", self.w_is_not_resistant_condorcet_winner_)
        print("w_missed_resistant_condorcet_winner =", self.w_missed_resistant_condorcet_winner_)

        self.log_depth = old_log_depth

    # %% Setting the options

    @property
    def iia_subset_maximum_size(self):
        """Float or ``numpy.inf``. Maximum size of any subset of candidates that is used to compute
        :meth:`not_iia` (and related methods). For a given voting system, this attribute has no
        effect if there is an exact algorithm running in polynomial time implemented in SVVAMP.
        """
        return self._iia_subset_maximum_size

    @iia_subset_maximum_size.setter
    def iia_subset_maximum_size(self, value):
        if self._iia_subset_maximum_size == value:
            return
        try:
            self.mylogv("Setting iia_subset_maximum_size =", value, 1)
            self._iia_subset_maximum_size = float(value)
            self.delete_cache(contains='_iia_', suffix='_')
        except ValueError:
            raise ValueError("Unknown value for iia_subset_maximum_size: " + format(value) +
                             " (number or np.inf expected).")

    @property
    def im_option(self):
        """String. Option used to compute :meth:`is_im_` and related methods.

        To know what options are accepted for a given voting system, use :attr:`options_parameters`.
        """
        return self._im_option

    @im_option.setter
    def im_option(self, value):
        if self._im_option == value:
            return
        if value in self.options_parameters['im_option']['allowed']:
            self.mylogv("Setting im_option =", value, 1)
            self._im_option = value
            self.delete_cache(contains='_im_', suffix='_')
        else:
            raise ValueError("Unknown value for im_option: " + format(value))

    @property
    def tm_option(self):
        """String. Option used to compute :meth:`is_tm_` and related methods.

        To know what options are accepted for a given voting system, use :attr:`options_parameters`.
        """
        return self._tm_option

    @tm_option.setter
    def tm_option(self, value):
        if self._tm_option == value:
            return
        if value in self.options_parameters['tm_option']['allowed']:
            self.mylogv("Setting tm_option =", value, 1)
            self._tm_option = value
            self.delete_cache(contains='_tm_', suffix='_')
            if self.cm_option != 'exact' and self.precheck_tm:
                self.delete_cache(contains='_cm_', suffix='_')
        else:
            raise ValueError("Unknown value for tm_option: " + format(value))

    @property
    def um_option(self):
        """String. Option used to compute :meth:`is_um_` and related methods.

        To know what options are accepted for a given voting system, use :attr:`options_parameters`.
        """
        return self._um_option

    @um_option.setter
    def um_option(self, value):
        if self._um_option == value:
            return
        if value in self.options_parameters['um_option']['allowed']:
            self.mylogv("Setting um_option =", value, 1)
            self._um_option = value
            self.delete_cache(contains='_um_', suffix='_')
            if self.cm_option != 'exact' and self.precheck_um:
                self.delete_cache(contains='_cm_', suffix='_')
        else:
            raise ValueError("Unknown value for um_option: " + format(value))

    @property
    def icm_option(self):
        """String. Option used to compute :meth:`is_icm_` and related methods.

        To know what options are accepted for a given voting system, use :attr:`options_parameters`.
        """
        return self._icm_option

    @icm_option.setter
    def icm_option(self, value):
        if self._icm_option == value:
            return
        if value in self.options_parameters['icm_option']['allowed']:
            self.mylogv("Setting icm_option =", value, 1)
            self._icm_option = value
            self.delete_cache(contains='_icm_', suffix='_')
            if self.cm_option != 'exact' and self.precheck_icm:
                self.delete_cache(contains='_cm_', suffix='_')
        else:
            raise ValueError("Unknown value for icm_option: " + format(value))

    @property
    def cm_option(self):
        """String. Option used to compute :meth:`is_cm_` and related methods.

        To know what options are accepted for a given voting system, use :attr:`options_parameters`.
        """
        return self._cm_option

    @cm_option.setter
    def cm_option(self, value):
        if self._cm_option == value:
            return
        if value in self.options_parameters['cm_option']['allowed']:
            self.mylogv("Setting cm_option =", value, 1)
            self._cm_option = value
            self.delete_cache(contains='_cm_', suffix='_')
        else:
            raise ValueError("Unknown value for cm_option: " + format(value))

    # %% Independence of Irrelevant Alternatives (IIA)

    @cached_property
    def log_iia_(self):
        """String. Parameters used to compute :meth:`is_not_iia_` and related methods."""
        return "iia_subset_maximum_size = " + format(self.iia_subset_maximum_size)

    @cached_property
    def is_not_iia_(self):
        """Boolean. ``True`` if there exists a subset of candidates including the sincere winner
        :attr:`w_`, such that if the election is held with this subset of candidates, then :attr:`w_` is not the
        winner anymore. If the algorithm cannot decide, then the result is ``numpy.nan``.
        """
        if np.isnan(self.is_iia_):
            return np.nan
        else:
            return not self.is_iia_

    @cached_property
    def is_iia_(self):
        """Boolean. Cf. :attr:`is_not_iia`."""
        return self._compute_iia_['is_iia']

    @cached_property
    def example_winner_iia_(self):
        """Integer (candidate). If the election is not IIA, ``example_winner_iia`` is the winner corresponding to the
        counter-example ``example_subset_iia``. If the election is IIA (or if the algorithm cannot decide), then
        ``example_winner_iia = numpy.nan``.
        """
        return self._compute_iia_['example_winner_iia']

    @cached_property
    def example_subset_iia_(self):
        """1d array of booleans. If the election is not IIA, ``example_subset_iia`` provides a subset of candidates
        breaking IIA. ``example_subset_iia[c]`` is ``True`` iff candidate ``c`` belongs to the subset. If the
        election is IIA (or if the algorithm cannot decide), then ``example_subset_iia = numpy.nan``.
        """
        return self._compute_iia_['example_subset_iia']

    @cached_property
    def _compute_iia_(self):
        """Compute iia.

        :return: a dictionary whose keys are 'is_iia', 'example_subset_iia', 'example_winner_iia'.
        """
        self.mylog("Compute IIA", 1)
        if self.meets_iia:
            return self._compute_iia_aux_when_guaranteed_("IIA is guaranteed for this voting system.")
        if self.meets_condorcet_c_ut_abs and self.w_is_condorcet_winner_ut_abs_:
            return self._compute_iia_aux_when_guaranteed_("IIA guaranteed: w is a Condorcet winner.")
        if self.meets_condorcet_c_ut_abs_ctb and self.w_is_condorcet_winner_ut_abs_ctb_:
            return self._compute_iia_aux_when_guaranteed_("IIA guaranteed: w is a Condorcet winner (ctb).")
        if self.meets_condorcet_c_ut_rel and self.w_is_condorcet_winner_ut_rel_:  # pragma: no cover
            # TO DO: Investigate whether this case can actually happen.
            self._reached_uncovered_code()
            return self._compute_iia_aux_when_guaranteed_("IIA guaranteed: w is a relative Condorcet winner.")
        if self.meets_condorcet_c_ut_rel_ctb and self.w_is_condorcet_winner_ut_rel_ctb_:  # pragma: no cover
            # TO DO: Investigate whether this case can actually happen.
            self._reached_uncovered_code()
            return self._compute_iia_aux_when_guaranteed_("IIA guaranteed: w is a relative Condorcet winner (ctb).")
        if self.meets_condorcet_c_rk and self.w_is_condorcet_winner_rk_:
            return self._compute_iia_aux_when_guaranteed_("IIA guaranteed: w is a Condorcet winner (vtb).")
        if self.meets_condorcet_c_rk_ctb and self.w_is_condorcet_winner_rk_ctb_:
            return self._compute_iia_aux_when_guaranteed_("IIA guaranteed: w is a Condorcet winner (vtb, ctb).")
        if self.meets_majority_favorite_c_ut and self.profile_.plurality_scores_ut[self.w_] > self.profile_.n_v / 2:
            return self._compute_iia_aux_when_guaranteed_("IIA guaranteed: w is a majority favorite.")
        if self.meets_majority_favorite_c_rk and self.profile_.plurality_scores_rk[self.w_] > self.profile_.n_v / 2:
            return self._compute_iia_aux_when_guaranteed_("IIA guaranteed: w is a majority favorite (vtb).")
        if (self.meets_majority_favorite_c_ut_ctb and self.w_ == 0
                and self.profile_.plurality_scores_ut[self.w_] >= self.profile_.n_v / 2):
            return self._compute_iia_aux_when_guaranteed_("IIA guaranteed: w is a majority favorite (ctb) (w = 0).")
        if (self.meets_majority_favorite_c_rk_ctb and self.w_ == 0
                and self.profile_.plurality_scores_rk[self.w_] >= self.profile_.n_v / 2):
            return self._compute_iia_aux_when_guaranteed_("IIA guaranteed: w is a majority favorite (vtb, ctb)"
                                                          "(w = 0).")
        if self.with_two_candidates_reduces_to_plurality:
            if self.w_is_not_condorcet_winner_rk_ctb_:
                # For subsets of 2 candidates, we use the matrix of victories to gain time.
                self.mylog("IIA failure found by Condorcet failure (rk, ctb).", 2)
                is_iia = False
                example_winner_iia = np.nonzero(self.profile_.matrix_victories_rk_ctb[:, self.w_])[0][0]
                example_subset_iia = np.zeros(self.profile_.n_c, dtype=bool)
                example_subset_iia[self.w_] = True
                example_subset_iia[example_winner_iia] = True
                return {'is_iia': is_iia, 'example_subset_iia': example_subset_iia,
                        'example_winner_iia': example_winner_iia}
            else:
                self.mylog("IIA: subsets of size 2 are ok because w is a Condorcet winner (rk, ctb).", 2)
                return self._compute_iia_aux_(subset_minimum_size=3)
        else:
            return self._compute_iia_aux_(subset_minimum_size=2)

    def _compute_iia_aux_when_guaranteed_(self, message):
        """Auxiliary function of _compute_iia_, used when IIA is guaranteed.

        Parameters
        ----------
        message : str
            A log message explaining why IIA is guaranteed.

        Returns
        -------
        dict
            A dictionary whose keys are 'is_iia', 'example_subset_iia', 'example_winner_iia'.
        """
        self.mylog(message, 1)
        return {'is_iia': True, 'example_subset_iia': np.nan, 'example_winner_iia': np.nan}

    def _compute_iia_aux_(self, subset_minimum_size):
        """Auxiliary function of _compute_iia_: real work.

        Parameters
        ----------
        subset_minimum_size : int

        Returns
        -------
        dict
            A dictionary whose keys are 'is_iia', 'example_subset_iia', 'example_winner_iia'.

        Notes
        -----
        Tests all subsets from size ``subset_minimum_size`` to ``self.iia_subset_maximum_size``. If
        ``self.iia_subset_maximum_size`` < ``C - 1``, then the algorithm may not be able to decide whether
        the election is IIA or not: in this case, we may have is_iia = NaN.
        """
        self.mylogv("IIA: Use _compute_iia_aux_ with subset_minimum_size =", subset_minimum_size, 1)
        subset_maximum_size = int(min(self.profile_.n_c - 1, self.iia_subset_maximum_size))
        for n_c_reduced in range(subset_minimum_size, subset_maximum_size + 1):
            if self.w_ <= n_c_reduced - 1:
                candidates_r = np.array(range(n_c_reduced))
            else:
                candidates_r = np.concatenate((range(n_c_reduced - 1), [self.w_]))
            while candidates_r is not None:
                w_r = self._compute_winner_of_subset_(candidates_r)
                if w_r != self.w_:
                    self.mylog("IIA failure found", 2)
                    example_subset_iia = np.zeros(self.profile_.n_c, dtype=bool)
                    for c in candidates_r:
                        example_subset_iia[c] = True
                    return {'is_iia': False, 'example_winner_iia': w_r, 'example_subset_iia': example_subset_iia}
                candidates_r = compute_next_subset_with_w(candidates_r, self.profile_.n_c, n_c_reduced, self.w_)
        # We have not found a counter-example...
        if self.iia_subset_maximum_size < self.profile_.n_c - 1:
            self.mylog("IIA: I have found no counter-example, but we have not explored all possibilities", 2)
            return {'is_iia': np.nan, 'example_winner_iia': np.nan, 'example_subset_iia': np.nan}
        else:
            self.mylog("IIA is guaranteed.", 2)
            return {'is_iia': True, 'example_winner_iia': np.nan, 'example_subset_iia': np.nan}

    def _compute_winner_of_subset_(self, candidates_r):
        """Compute the winner for a subset of candidates.

        This function is internally used to compute Independence of Irrelevant Alternatives (IIA).

        Parameters
        ----------
        candidates_r : list or ndarray
            1d array of integers. ``candidates_r[k]`` is the ``k``-th candidate of the subset. This vector must be
            sorted in ascending order.

        Returns
        -------
        w_r : int
            Candidate who wins the sub-election defined by ``candidates_r``.
        """
        self.mylogv("IIA: Compute winner of subset ", candidates_r, 3)
        return candidates_r[self._copy(profile=ProfileSubsetCandidates(self.profile_, candidates_r)).w_]

    # %% Manipulation: common features

    @cached_property
    def v_wants_to_help_c_(self):
        """2d array of booleans. ``v_wants_to_help_c[v, c]`` is ``True`` iff voter ``v`` strictly prefers candidate
        ``c`` to the sincere winner :attr:`w_`. If ``v`` attributes the same utility to ``c`` and ``w_``, then ``v``
        is not interested.
        """
        self.mylog("Compute v_wants_to_help_c", 1)
        return self.profile_.preferences_ut > self.profile_.preferences_ut[:, self.w_][:, np.newaxis]

    @cached_property
    def losing_candidates_(self):
        """1d of Integers. List of losing candidates, in a decreasing order of (heuristic) dangerousness

        This attribute is mostly intended for SVVAMP developers. It is used in most manipulation algorithms. The idea
        is to try first the candidates for whom we think manipulation is more likely to succeed, in order to gain time.

        Losing candidates are sorted from the 'most dangerous' to the 'least dangerous' (for the sincere winner
        :attr:`w_`). By default, they are sorted by their score against :attr:`w_` in the
        :attr:`~svvamp.Profile.matrix_duels_ut` (which is the number of potential manipulators for a given
        candidate). This behavior can be redefined in the subclass implementing a specific voting system.
        """
        self.mylog("Compute ordered list of losing candidates", 1)
        result = np.concatenate((
            np.array(range(0, self.w_), dtype=int),
            np.array(range(self.w_ + 1, self.profile_.n_c), dtype=int)
        ))
        return result[np.argsort(- self.profile_.matrix_duels_ut[result, self.w_], kind='mergesort')]

    @cached_property
    def c_has_supporters_(self):
        """1d array of booleans. ``c_has_supporters_[c]`` is ``True`` iff at least one voter prefers candidate ``c`` to
        the sincere winner :attr:`w_`.
        """
        self.mylog("Compute c_has_supporters_", 1)
        return np.any(self.v_wants_to_help_c_, 0)

    def _update_sufficient(self, sufficient_array, c, value, message=None):
        """Update an array _sufficient_coalition_size_.. for candidate c.

        Parameters
        ----------
        sufficient_array : list or ndarray
            An array like ``_sufficient_coalition_size_cm`` or ``_sufficient_coalition_size_icm``.
        c : int
            Candidate.
        value : int
            If the number of manipulators is >= value, then manipulation (CM or ICM) is possible.
        message : str
            A message that can be displayed if ``sufficient_array[c]`` is actually updated.

        Notes
        -----
        Perform ``sufficient_array[c] = min(sufficient_array[c], value)``. If ``sufficient_array[c]`` is actually
        updated, i.e. iff ``value`` is strictly lower that the former value of ``sufficient_array[c]``, then:
        send ``message`` and ``value`` to ``self.mylogv (with detail level=3)``.
        """
        if value < sufficient_array[c]:
            sufficient_array[c] = value
            if message is not None:
                self.mylogv(message, value, 3)

    def _update_necessary(self, necessary_array, c, value, message=None):
        """Update an array _necessary_coalition_size_.. for candidate c.

        Parameters
        ----------
        necessary_array : list or ndarray
            An array like ``_necessary_coalition_size_cm`` or ``_necessary_coalition_size_icm``.
        c : int
            Candidate.
        value : int
            If the number of manipulators is < value, then manipulation (CM or ICM) is impossible.
        message : str
            A message that can be displayed if ``necessary_array[c]`` is actually updated.

        Notes
        -----
        Perform ``necessary_array[c] = max(necessary_array[c], value)``. If ``necessary_array[c]`` is actually updated,
        i.e. iff ``value`` is strictly greater that the former value of ``necessary_array[c]``, then: send ``message``
        and ``value`` to ``self.mylogv (with detail level = 3)``.
        """
        if value > necessary_array[c]:
            necessary_array[c] = value
            if message is not None:
                self.mylogv(message, value, 3)

    # %% Individual manipulation (IM)

    @cached_property
    def log_im_(self):
        """String. Parameters used to compute :meth:`is_im_` and related methods."""
        return "im_option = " + self.im_option

    @cached_property
    def is_im_(self):
        """Boolean. ``True`` if there exists a voter who can and wants to manipulate, ``False`` otherwise. If the
        algorithm cannot decide, then ``numpy.nan``.

        Voter ``v`` can and wants to manipulate for candidate ``c`` iff:

            *   ``v`` strictly prefers ``c`` to :attr:`w_` (in the sense of :attr:`~svvamp.Profile.preferences_ut`).
            *   And by changing her vote, she can make ``c`` win instead of :attr:`w_`.
        """
        _ = self._im_is_initialized_general_
        if np.isneginf(self._is_im):
            self._compute_im_(mode='is_im_')
        return pseudo_bool(self._is_im)

    def is_im_c_(self, c):
        """Individual manipulation, focus on one candidate.

        Parameters
        ----------
        c : int
            Candidate.

        Returns
        -------
        bool or nan
            ``candidates_im[c]``.
        """
        _ = self._im_is_initialized_general_
        if np.isneginf(self._candidates_im[c]):
            self._compute_im_(mode='is_im_c_', c=c)
        return pseudo_bool(self._candidates_im[c])

    def is_im_c_with_voters_(self, c):
        """Individual manipulation, focus on one candidate, with details.

        Parameters
        ----------
        c : int
            Candidate.

        Returns
        -------
        tuple
            (``candidates_im[c]``, ``v_im_for_c[:, c]``).
        """
        _ = self._im_is_initialized_general_
        if np.isneginf(self._candidates_im[c]) or np.any(np.isneginf(self._v_im_for_c[:, c])):
            self._compute_im_(mode='is_im_c_with_voters_', c=c)
        return pseudo_bool(self._candidates_im[c]), self._v_im_for_c[:, c]

    @cached_property
    def voters_im_(self):
        """1d array of booleans (or ``numpy.nan``). ``voters_im_[v]`` is ``True`` if voter ``v`` can and wants to
        manipulate for at least one candidate, ``False`` otherwise. If the algorithm cannot decide, then ``numpy.nan``.
        """
        _ = self._im_is_initialized_general_
        if not self._im_was_computed_with_voters:
            self._compute_im_(mode='im_with_voters_')
        return self._voters_im.astype(np.float)

    @cached_property
    def candidates_im_(self):
        """1d array of booleans (or ``numpy.nan``). ``candidates_im[c]`` is ``True`` if there exists a voter who can
        manipulate for candidate ``c``, ``False`` otherwise. If the algorithm cannot decide, then ``numpy.nan``. For
        the sincere winner :attr:`w_`, we have by convention ``candidates_im_[w_] = False``.
        """
        _ = self._im_is_initialized_general_
        if not self._im_was_computed_with_candidates:
            self._compute_im_(mode='im_with_candidates_')
        return self._candidates_im.astype(np.float)

    @cached_property
    def v_im_for_c_(self):
        """2d array of booleans. ``v_im_for_c_[v, c]`` is ``True`` if voter ``v`` can manipulate for candidate ``c``,
        ``False`` otherwise. If the algorithm cannot decide, then ``numpy.nan``. For the sincere winner :attr:w_`,
        we have by convention ``v_im_for_c_[v, w_] = False``.
        """
        _ = self._im_is_initialized_general_
        if not self._im_was_computed_full:
            self._compute_im_(mode='v_im_for_c_')
        return self._v_im_for_c.astype(np.float)

    def is_im_v_(self, v):
        """Individual manipulation, focus on one voter.

        Parameters
        ----------
        v : int
            Voter.

        Returns
        -------
        bool or nan
            ``voters_im[v]``.
        """
        _ = self._im_is_initialized_general_
        if np.isneginf(self._voters_im[v]):
            self._compute_im_v_(v, c_is_wanted=np.ones(self.profile_.n_c, dtype=np.bool), stop_if_true=True)
        return pseudo_bool(self._voters_im[v])

    def is_im_v_with_candidates_(self, v):
        """Individual manipulation, focus on one voter, with details.

        Parameters
        ----------
        v : int
            Voter.

        Returns
        -------
        tuple
            (``voters_im[v]``, ``v_im_for_c[v, :]``).
        """
        _ = self._im_is_initialized_general_
        if np.any(np.isneginf(self._v_im_for_c[v, :])):
            self._compute_im_v_(v, c_is_wanted=np.ones(self.profile_.n_c, dtype=np.bool), stop_if_true=False)
        return pseudo_bool(self._voters_im[v]), self._v_im_for_c[v, :].astype(np.float)

    @cached_property
    def _im_is_initialized_general_(self):
        """Initialize IM variables and do preliminary checks.

        Since it is a cached property, the code is run only the first time IM is launched (whatever the  mode).

        * ``_is_im`` --> False or True if we know, -inf otherwise.
        * ``_candidates_im[c]`` --> True of False if we know, -inf otherwise.
        * ``_voters_im[v]`` --> True of False if we know, -inf otherwise.
        * ``_v_im_for_c[v, c]`` --> True or False if we know, -inf otherwise.

        It is mandatory that ``_v_im_for_c[v, c]`` is False if voter ``c`` does not prefer ``c`` to the sincere winner
        ``w_``. Other kinds of checks are optional if this method is redefined in subclasses.

        If ``_candidates_im`` and ``_is_im`` are totally decided to True or False, then
        ``_im_was_computed_with_candidates`` should become True (not mandatory but recommended).
        """
        self.mylog("IM: Initialize", 2)
        self._im_was_computed_with_candidates = False
        self._im_was_computed_with_voters = False
        self._im_was_computed_full = False
        self._v_im_for_c = np.full((self.profile_.n_v, self.profile_.n_c), -np.inf)
        self._candidates_im = np.full(self.profile_.n_c, -np.inf)
        self._voters_im = np.full(self.profile_.n_v, -np.inf)
        self._is_im = -np.inf
        self._im_preliminary_checks_general_()
        return True

    def _im_preliminary_checks_general_(self):
        """Do preliminary checks for IM. Only first time IM is launched.

        Can update some ``_v_im_for_c[v, c]`` to True or False (instead of -inf). In particular, it is mandatory that
        it is updated to False if voter ``v`` does not prefer ``c`` to the sincere winner ``w``.

        * If ``_v_im_for_c[v, c]`` becomes True, then ``_candidates_im[c]`` and ``_voters_im[v]`` must become True. If
          ``_candidates_im[c]`` or ``_voters_im[v]`` becomes True, then ``_is_im`` must become True.
        * If ``_is_im`` becomes True, it is not necessary to update a specific ``_candidates_im[c]`` or
          ``_voters_im[v]``. If ``_candidates_im[c]`` or ``_voters_im[v]`` becomes True, it is not necessary to update
          a specific ``_v_im_for_c[v, c]``.
        * If ``_is_im`` becomes False, then ``_candidates_im[:]`` and ``_voters_im[:]`` must become False. If
          ``_candidates_im[c]`` becomes False, then ``_v_im_for_c[:, c]`` must become False. ``If _voters_im[v]``
           becomes False, then ``_v_im_for_c[v, :]`` must become False.
        * If for a candidate ``c`` and all voters ``v``, all ``_v_im_for_c[v, c]`` become False, then
          ``_candidates_im[c]`` must be updated to False. If for all candidates ``c``, ``_candidates_im[c]`` becomes
          False, then ``_is_im`` must be updated to False.
        * Similarly, if for a voter ``v`` and all candidates ``c``, all ``_v_im_for_c[v, c]`` become False,
          then ``_voters_im[v]`` must become False. If for all voters ``v``, ``_voters_im[v]`` becomes False, then
          ``_is_im`` must be updated to False.
        * If ``_v_im_for_c``, ``_candidates_im`` and ``_is_im`` are totally decided to True or False, then
          ``_im_was_computed_with_candidates``, ``_im_was_computed_with_voters`` and ``_im_was_computed_full`` should
          become True (not mandatory but recommended).
        """
        # Perform some preliminary checks
        self._v_im_for_c[np.logical_not(self.v_wants_to_help_c_)] = False
        self._v_im_for_c[np.logical_not(self.v_might_im_for_c_)] = False
        self._im_preliminary_checks_general_subclass_()
        # Update 'False' answers for _candidates_im, _voters_im and _is_im
        self._candidates_im[np.all(np.equal(self._v_im_for_c, False), 0)] = False
        self._voters_im[np.all(np.equal(self._v_im_for_c, False), 1)] = False
        if np.all(np.equal(self._candidates_im, False)):
            self.mylog("IM: preliminary checks: IM is impossible.", 2)
            self._is_im = False
            self._im_was_computed_with_candidates = True
            self._im_was_computed_with_voters = True
            self._im_was_computed_full = True
        self.mylogm('_v_im_for_c =', self._v_im_for_c, 3)

    def _im_preliminary_checks_general_subclass_(self):
        """Do preliminary checks for IM. Only first time IM is launched.

        Can update some ``_v_im_for_c[v, c]`` to True or False (instead of -inf).

        True must be propagated from specific to general, False must be propagated from general to specific. I.e.:
        * If ``_v_im_for_c[v, c]`` becomes True, then ``_candidates_im[c]`` and ``_voters_im[v]`` must become True.
          If ``_candidates_im[c]`` or ``_voters_im[v]`` becomes True, then ``_is_im must`` become True.
        * If ``_is_im`` becomes False, then ``_candidates_im[:]`` and ``_voters_im[v]`` must become False. If
          ``_candidates_im[c]`` or ``_voters_im[v]`` becomes False, then ``_v_im_for_c[:, c]`` must become False.

        If for a candidate ``c`` and all voters ``v``, all ``_v_im_for_c[v, c]`` become False, it is not necessary to
        update ``_candidates_im[c]`` to False (and it is not necessary to update ``_is_im``).
        """
        pass

    def _im_initialize_v_(self, v):
        """Initialize the IM loop for voter v and do preliminary checks. Launched every time we work on voter v.

        * If the voting system is ordinal and voter ``v`` has the same ordinal preferences as previous voter ``v - 1``,
          then update the line ``_v_im_for_c[v, :]`` with what we know for ``v - 1``.
        * Preliminary checks: try to decide some ``_v_im_for_c[v, c]``. If ``_v_im_for_c[v, c]`` becomes True, then
          ``_candidates_im[c]``, ``_voters_im[v]`` and ``_is_im`` must become True as well. In the other cases, it is
          not necessary to update ``_candidates_im[c]`` and ``_is_im``.
        """
        self.mylogv("IM: Voter =", v, 3)
        # Check if v is identical to previous voter
        if self.is_based_on_rk and self.profile_.v_has_same_ordinal_preferences_as_previous_voter[v]:
            self.mylog("IM: Identical to previous voter", 3)
            decided_previous_v = np.logical_not(np.isneginf(self._v_im_for_c[v - 1, :]))
            self._v_im_for_c[v, decided_previous_v] = self._v_im_for_c[v - 1, decided_previous_v]
            if equal_true(self._voters_im[v - 1]):
                self._voters_im[v] = True
        # Preliminary checks on v
        self._im_preliminary_checks_v_(v)

    def _im_preliminary_checks_v_(self, v):
        """IM: preliminary checks for voter v.

        Try to decide some ``_v_im_for_c[v, c]``. If ``_v_im_for_c[v, c]`` becomes True, then ``_candidates_im[c]``,
        ``_voters_im[v]`` and ``_is_im`` must become True as well. In the other cases, it is not necessary to update
        ``_candidates_im[c]``, ``_voters_im[c]`` and ``_is_im``.
        """
        # Nothing smart for the moment.
        self._im_preliminary_checks_v_subclass_(v)

    def _im_preliminary_checks_v_subclass_(self, v):
        """IM: preliminary checks for voter v.

        Try to decide some ``_v_im_for_c[v, c]``. If ``_v_im_for_c[v, c]`` becomes True, then ``_candidates_im[c]``,
        ``_voters_im[v]`` and ``_is_im`` must become True as well. In the other cases, it is not necessary to update
        ``_candidates_im[c]``, ``_voters_im[c]`` and ``_is_im``.
        """
        pass

    def _im_main_work_v_(self, v, c_is_wanted, nb_wanted_undecided, stop_if_true):
        """Do the main work in IM loop for voter v.

        Parameters
        ----------
        v : int
            Voter.
        c_is_wanted : list or ndarray
            1d array of booleans. If for all ``c`` such that ``c_is_wanted[c]`` is True, ``_v_im_for_c[v, c]`` is
            decided, then we are authorized to get out.
        nb_wanted_undecided : int
            Number of 'wanted' candidates ``c`` such that ``_v_im_for_c[v, c]`` is not decided yet.
        stop_if_true : bool
            If True, then as soon as a True is found for a 'wanted' candidate, we are authorized to get out.

        Notes
        -----
        Try to decide ``_v_im_for_c[v, :]``. At the end, ``_v_im_for_c[v, c]`` can be True, False, NaN or -inf (we may
        not have decided for all candidates).

        If ``_v_im_for_c[v, c]`` becomes True, then ``_candidates_im[c]``, ``_voters_im[v]`` and ``_is_im`` must become
        True. In the other cases, it is not necessary to update ``_candidates_im[c]``, ``_voters_im[v]`` and ``_is_im``
        (even if ``_v_im_for_c[v, c]`` becomes NaN).

        Each time a wanted candidate is decided (to True, False or NaN), decrement ``nb_wanted_undecided``. When it
        reaches 0, we may get out. If a wanted candidate is decided to True and ``stop_if_true``, then we may get out.
        """
        # N.B.: in some subclasses, it is possible to try one method, then another one if the first one fails,
        # etc. In this general class, we will simply do a switch between 'lazy' and 'exact'.
        getattr(self, '_im_main_work_v_' + self.im_option + '_')(v, c_is_wanted, nb_wanted_undecided, stop_if_true)
        # Launch a sub-method like _im_main_work_v_lazy, etc.

    # noinspection PyUnusedLocal
    def _im_main_work_v_lazy_(self, v, c_is_wanted, nb_wanted_undecided, stop_if_true):
        """Do the main work in IM loop for voter v, with option 'lazy'. Cf. :meth:`_im_main_work_v_`.
        """
        # When we don't know, we decide that we don't know!
        neginf_to_nan(self._v_im_for_c[v, :])

    def _im_main_work_v_exact_(self, v, c_is_wanted, nb_wanted_undecided, stop_if_true):
        """Do the main work in IM loop for voter v, with option 'exact'. Cf. :meth:`_im_main_work_v_`.
        """
        if self.is_based_on_rk:
            self._im_main_work_v_exact_rankings_(v, c_is_wanted, nb_wanted_undecided, stop_if_true)
        elif self.is_based_on_ut_minus1_1:  # pragma: no cover
            # As of now, all the voting rules concerned (Majority Judgement, Range Voting and Approval)
            # have their own `_im_main_work_v_` method, so they do not use this.
            self._reached_uncovered_code()
            self._im_main_work_v_exact_utilities_minus1_1_(v, c_is_wanted, nb_wanted_undecided, stop_if_true)
        else:
            raise NotImplementedError

    def _im_main_work_v_exact_rankings_(self, v, c_is_wanted, nb_wanted_undecided, stop_if_true):
        """Do the main work in IM loop for voter v, with option 'exact', for a voting system based only on strict
        rankings. Cf. :meth:`_im_main_work_v_`.
        """
        preferences_borda_test = np.copy(self.profile_.preferences_borda_rk)
        ballot = np.array(range(self.profile_.n_c))
        ballot_favorite = self.profile_.n_c - 1
        while ballot is not None:  # Loop on possible ballots
            self.mylogv("IM: Ballot =", ballot, 3)
            preferences_borda_test[v, :] = ballot
            w_test = self._copy(profile=Profile(preferences_ut=preferences_borda_test, sort_voters=False)).w_
            if np.isneginf(self._v_im_for_c[v, w_test]):
                # Implicitly, it also means that v prefers c to w (cf. specifications of _im_initialize_general).
                self._v_im_for_c[v, w_test] = True
                self._candidates_im[w_test] = True
                self._voters_im[v] = True
                self._is_im = True
                self.mylogv("IM found for c =", w_test, 3)
                if c_is_wanted[w_test]:
                    if stop_if_true:
                        return
                    nb_wanted_undecided -= 1
                if nb_wanted_undecided == 0:
                    return  # We know everything we want for this voter
            ballot, ballot_favorite = compute_next_borda_clever(ballot, ballot_favorite, self.profile_.n_c)
        # If we reach this point, we have tried all ballots, so if we have not found a manipulation for ``c``, it is
        # not possible. Next instruction replaces all -Inf with 0.
        neginf_to_zero(self._v_im_for_c[v, :])

    def _im_main_work_v_exact_utilities_minus1_1_(self, v, c_is_wanted,
                                                  nb_wanted_undecided, stop_if_true):  # pragma: no cover
        """Do the main work in IM loop for voter v, with option 'exact', for a voting system based only on utilities
        and where it is optimal for a c-manipulator to pretend that ``c`` has utility 1 and other candidates utility 0.
        Cf. :meth:`_im_main_work_v`.
        """
        # As of now, all the voting rules concerned (Majority Judgement, Range Voting and Approval)
        # have their own `_im_main_work_v_` method, so they do not use this.
        self._reached_uncovered_code()
        preferences_ut_test = np.copy(self.profile_.preferences_ut)
        for c in range(self.profile_.n_c):
            if not c_is_wanted[c]:
                continue
            if not np.isneginf(self._v_im_for_c[v, c]):
                continue
            # Implicitly, it also means that v prefers c to w (cf. specifications of _im_initialize_general).
            preferences_ut_test[v, :] = -1
            preferences_ut_test[v, c] = 1
            w_test = self._copy(Profile(preferences_ut=preferences_ut_test, sort_voters=False)).w_
            if w_test == c:
                self._v_im_for_c[v, c] = True
                self._candidates_im[c] = True
                self._voters_im[v] = True
                self._is_im = True
                self.mylog("IM found", 3)
                if stop_if_true:
                    return
            else:
                self._v_im_for_c[v, c] = False
            nb_wanted_undecided -= 1
            if nb_wanted_undecided == 0:
                return

    def _compute_im_(self, mode, c=None):
        """Compute IM.

        Parameters
        ----------
        mode : str
            Name of the method calling _compute_im.
        c : int or None
            If integer, then we only want to study IM for this candidate.
        """
        self.mylog("Compute IM", 1)
        for v in range(self.profile_.n_v):
            # Prepare work
            if mode == 'is_im_':
                c_is_wanted = np.ones(self.profile_.n_c, dtype=np.bool)
                stop_if_true = True
            elif mode in {'is_im_c_', 'is_im_c_with_voters_'}:
                c_is_wanted = np.zeros(self.profile_.n_c, dtype=np.bool)
                c_is_wanted[c] = True
                stop_if_true = True
            elif mode == 'im_with_voters_':
                c_is_wanted = np.ones(self.profile_.n_c, dtype=np.bool)
                stop_if_true = True
            elif mode == 'im_with_candidates_':
                c_is_wanted = np.isneginf(self._candidates_im)
                stop_if_true = False
            elif mode == 'v_im_for_c_':
                c_is_wanted = np.ones(self.profile_.n_c, dtype=np.bool)
                stop_if_true = False
            else:  # This should not happen.
                raise NotImplementedError
            # Work
            self._compute_im_v_(v, c_is_wanted, stop_if_true)
            # Conclude for v
            if mode == 'is_im_':
                if not np.isneginf(self._is_im):
                    return
            elif mode == 'is_im_c_':
                if not np.isneginf(self._candidates_im[c]):
                    return
            elif mode == 'im_with_candidates_':
                if not np.any(np.isneginf(self._candidates_im)):
                    self._im_was_computed_with_candidates = True
                    return
        # Conclude: update _candidates_im and _is_im if possible
        self._candidates_im[np.all(np.equal(self._v_im_for_c, False), 0)] = False
        for c in self.losing_candidates_:
            if np.isneginf(self._candidates_im[c]):
                if np.all(np.logical_not(np.isneginf(self._v_im_for_c[:, c]))):
                    self._candidates_im[c] = np.nan
        if np.isneginf(self._is_im):
            if np.all(np.equal(self._candidates_im, False)):
                self._is_im = False
            elif np.all(np.logical_not(np.isneginf(self._candidates_im))):
                self._is_im = np.nan
        if not np.any(np.isneginf(self._v_im_for_c)):
            self._im_was_computed_full = True
            self._im_was_computed_with_voters = True
            self._im_was_computed_with_candidates = True
        else:
            if not np.any(np.isneginf(self._voters_im)):
                self._im_was_computed_with_voters = True
            if not np.any(np.isneginf(self._candidates_im)):
                self._im_was_computed_with_candidates = True

    def _compute_im_v_(self, v, c_is_wanted, stop_if_true):
        """Compute IM for voter v.

        Parameters
        ----------
        v : int
            Voter.
        c_is_wanted : list or ndarray
            1d array of booleans. If for all ``c`` such that ``c_is_wanted[c]`` is True, ``_v_im_for_c[v, c]`` is
            decided, then we are authorized to get out.
        stop_if_true : bool
            If True, then as soon as a True is found for a 'wanted' candidate, we are authorized to get out.

        Notes
        -----
        Try to decide ``_v_im_for_c[v, :]``. At the end, ``_v_im_for_c[v, c]`` can be True, False, NaN or -inf (we may
        have not decided for all candidates). At the end, ``_voters_im[v]`` must be consistent with what we know about
        ``_v_im_for_c[v, :]`` (True, False, NaN or -inf).

        If ``_v_im_for_c[v, c]`` becomes True, then ``_candidates_im[c]`` and ``_is_im`` must become True. In the other
        cases, it is not necessary to update ``_candidates_im[c]``, and ``_is_im`` (even if ``_v_im_for_c[v, c]``
        becomes NaN).
        """
        self._im_initialize_v_(v)
        nb_wanted_undecided = np.isneginf(self._v_im_for_c[v, c_is_wanted]).sum()
        if nb_wanted_undecided == 0:
            self.mylog("IM: Job already done", 3)
        else:
            self.mylogv("IM: Preliminary checks: Still some work for v =", v, 3)
            self._im_main_work_v_(v, c_is_wanted, nb_wanted_undecided, stop_if_true)
        if np.isneginf(self._voters_im[v]):
            if np.all(np.equal(self._v_im_for_c[v, :], False)):
                self._voters_im[v] = False
            elif np.all(np.logical_not(np.isneginf(self._v_im_for_c[v, :]))):
                self._voters_im[v] = np.nan

    # %% Trivial Manipulation (TM)

    @cached_property
    def log_tm_(self):
        """String. Parameters used to compute :meth:`is_tm_` and related methods."""
        return "tm_option = " + self.tm_option

    @cached_property
    def is_tm_(self):
        """Boolean (or ``numpy.nan``). ``True`` if TM is possible, ``False`` otherwise. If the algorithm cannot decide,
        then ``numpy.nan`` (but as of now, this value is never used for TM).

        For ordinal voting systems, we call *trivial manipulation* for candidate ``c`` against :attr:`w_` the fact of
        putting ``c`` on top (compromising), :attr:`w_` at the bottom (burying), while keeping a sincere order on
        the other candidates.

        For cardinal voting systems, we call *trivial manipulation* for ``c`` (against :attr:`w_`) the fact of
        putting the maximum grade for ``c`` and the minimum grade for the other candidates.

        In both cases, the intuitive idea is the following: if I want to make ``c`` win and I only know that
        candidate :attr:`w_` is 'dangerous' (but I know nothing else), then trivial manipulation is my 'best'
        strategy.

        We say that a situation is *trivially manipulable* for ``c`` (implicitly: coalitionally) iff, when all voters
        preferring ``c`` to the sincere winner :attr:`w_` use trivial manipulation, candidate ``c`` wins.
        """
        _ = self._tm_is_initialized_general_
        if np.isneginf(self._is_tm):
            self._compute_tm_(with_candidates=False)
        return pseudo_bool(self._is_tm)

    def is_tm_c_(self, c):
        """Trivial manipulation, focus on one candidate.

        Parameters
        ----------
        c : int
            Candidate.

        Returns
        -------
        bool or nan
            ``candidates_tm[c]``.
        """
        _ = self._tm_is_initialized_general_
        if np.isneginf(self._candidates_tm[c]):
            self._compute_tm_c_(c)
        return pseudo_bool(self._candidates_tm[c])

    @cached_property
    def candidates_tm_(self):
        """1d array of booleans (or ``numpy.nan``). ``candidates_tm[c]`` is ``True`` if a TM for candidate ``c`` is
        possible, ``False`` otherwise. If the algorithm cannot decide, then ``numpy.nan`` (but as of now, this value
        is not never for TM). For the sincere winner :attr:`w_`, we have by convention ``candidates_tm[w] = False``.
        """
        _ = self._tm_is_initialized_general_
        if not self._tm_was_computed_with_candidates:
            self._compute_tm_(with_candidates=True)
        return self._candidates_tm.astype(np.float)

    @cached_property
    def _tm_is_initialized_general_(self):
        """Initialize TM variables and perform some preliminary checks. Used only the first time TM is launched
        (whatever the mode).
        _is_tm --> True or False if we know, -inf otherwise.
        _candidates_tm[c] --> True or False if we know, -inf otherwise.

        If _candidates_tm and _is_tm are totally decided to True, False or NaN, then _tm_was_computed_with_candidates
        should become True (not mandatory but recommended).
        """
        self.mylog("TM: Initialize", 2)
        self._tm_was_computed_with_candidates = False
        self._candidates_tm = np.full(self.profile_.n_c, -np.inf)
        self._is_tm = -np.inf
        _ = self._tm_is_initialized_general_subclass_
        self._tm_preliminary_checks_general_()
        return True

    @cached_property
    def _tm_is_initialized_general_subclass_(self):
        return True

    def _tm_preliminary_checks_general_(self):
        """Do preliminary checks for TM. Only first time TM is launched.

        Can update some _candidates_tm[c] to True or False (instead of -inf).

        * If some ``_candidates_tm[c]`` becomes True, then ``_is_tm`` must become True as well.
        * If ``_is_tm`` becomes True, it is not necessary to update a specific ``_candidates_tm[c]``.
        * If for all candidates ``c``, ``_candidates_tm[c]`` become False, then ``_is_tm`` must be updated to False.
        * If ``_is_tm`` becomes False, then all ``_candidates_tm[c]`` must become False.
        * If ``_candidates_tm`` and ``_is_tm`` are totally decided to True or False, then
          ``_tm_was_computed_with_candidates`` should become True (not mandatory but recommended).

        N.B.: Be careful, if a pretest deciding TM to True is added, then some modifications may be needed for
        Exhaustive ballot.
        """
        # 1) Preliminary checks that may improve _candidates_tm (all must be done, except if everything is decided).
        # Majority favorite criterion
        if self.meets_majority_favorite_c_ut and self.profile_.plurality_scores_ut[self.w_] > self.profile_.n_v / 2:
            self.mylog("TM impossible (w is a majority favorite).", 2)
            self._is_tm = False
            self._candidates_tm[:] = False
            self._tm_was_computed_with_candidates = True
            return
        if (self.meets_majority_favorite_c_ut_ctb and self.w_ == 0
                and self.profile_.plurality_scores_ut[self.w_] >= self.profile_.n_v / 2):
            self.mylog("TM impossible (w=0 is a majority favorite with candidate tie-breaking).", 2)
            self._is_tm = False
            self._candidates_tm[:] = False
            self._tm_was_computed_with_candidates = True
            return
        if self.meets_majority_favorite_c_rk and self.profile_.plurality_scores_rk[self.w_] > self.profile_.n_v / 2:
            self.mylog("TM impossible (w is a majority favorite with voter tie-breaking).", 2)
            self._is_tm = False
            self._candidates_tm[:] = False
            self._tm_was_computed_with_candidates = True
            return
        if (self.meets_majority_favorite_c_rk_ctb and self.w_ == 0
                and self.profile_.plurality_scores_rk[self.w_] >= self.profile_.n_v / 2):
            self.mylog("TM impossible (w=0 is a majority favorite with voter and candidate tie-breaking).", 2)
            self._is_tm = False
            self._candidates_tm[:] = False
            self._tm_was_computed_with_candidates = True
            return
        # Having supporters
        self._candidates_tm[np.logical_not(self.c_has_supporters_)] = False
        # 2) Additional preliminary checks from the subclass.
        self._tm_preliminary_checks_general_subclass_()
        if not np.any(self.c_has_supporters_):
            self.mylog("TM impossible (all voters like w best)", 2)
            self._is_tm = False
            self._candidates_tm[:] = False
            self._tm_was_computed_with_candidates = True
            return
        if not np.isneginf(self._is_tm):  # pragma: no cover
            # For the moment, this cannot happen, because no voting rule overrides
            # `_tm_preliminary_checks_general_subclass_` (which does nothing by default).
            self._reached_uncovered_code()
            return
        # 3) Preliminary checks that gives only global information on _is_tm
        # (may return as soon as decision is made).
        # Nothing

    def _tm_preliminary_checks_general_subclass_(self):
        """Do preliminary checks for TM. Only first time TM is launched.

        Can update some ``_candidates_tm[c]`` to True or False (instead of -inf).

        True must be propagated from specific to general, False must be propagated from general to specific.
        * If some ``_candidates_tm[c]`` becomes True, then ``_is_tm`` must become True as well.
        * If ``_is_tm`` becomes True, it is not necessary to update a specific ``_candidates_tm[c]``.
        * If ``_is_tm`` becomes False, then all ``_candidates_tm[c]`` must become False.
        * If for all candidates ``c``, ``_candidates_tm[c]`` becomes False, it is not necessary to update ``_is_tm``.
        * If ``_candidates_tm`` and ``_is_tm`` are totally decided to True or False, then
          ``_tm_was_computed_with_candidates`` should become True (not mandatory but recommended).

        Put first the checks that may improve ``_candidates_tm`` (all must be done, except if everything is decided).
        Then the checks that gives only global information on ``_is_tm`` (which may return as soon as decision is made).
        """
        pass

    def _tm_initialize_c_(self, c):
        """Initialize the TM loop for candidate ``c`` and may do preliminary checks.

        * If ``_candidates_tm[c]`` is decided (True/False/NaN), it means that all the work for ``c`` has been done
          before. Then get out.
        * Preliminary checks: try to decide ``_candidates_tm[c]``. If it becomes True, then ``_is_tm`` must become True
          as well. In other cases, do not update ``_is_tm``.
        """
        self.mylogv("TM: Candidate =", c, 2)
        # Check if job is done for c
        if not np.isneginf(self._candidates_tm[c]):
            self.mylog("TM: Job already done", 2)
            return
        # Preliminary checks
        self._tm_preliminary_checks_c_(c)
        # Conclude what we can
        if equal_true(self._candidates_tm[c]):  # pragma: no cover
            # For the moment, this cannot happen, because no voting rule overrides
            # `_tm_preliminary_checks_general_subclass_` (which does nothing by default).
            self._reached_uncovered_code()
            self.mylogv("TM: Preliminary checks: TM is True for c =", c, 2)
            self._is_tm = True
        elif equal_false(self._candidates_tm[c]):  # pragma: no cover
            # For the moment, this cannot happen, because no voting rule overrides
            # `_tm_preliminary_checks_general_subclass_` (which does nothing by default).
            self._reached_uncovered_code()
            self.mylogv("TM: Preliminary checks: TM is False for c =", c, 2)
        else:
            self.mylogv("TM: Preliminary checks: TM is unknown for c =", c, 3)

    def _tm_preliminary_checks_c_(self, c):
        """TM: preliminary checks for challenger ``c``.

        Try to decide ``_candidates_tm[c]`` to True or False (instead of -inf). Do not update ``_is_tm``.
        """
        # We not do run any preliminary test for the moment, since computing TM is generally very easy (by design).
        self._tm_preliminary_checks_c_subclass_(c)

    def _tm_preliminary_checks_c_subclass_(self, c):
        """TM: preliminary checks for challenger ``c``.

        Try to decide ``_candidates_tm[c]`` to True or False (instead of -inf). Do not update ``_is_tm``.
        """
        pass

    def _tm_main_work_c_(self, c):
        """ Do the main work in TM loop for candidate ``c``. Must decide ``_candidates_tm[c]`` (to True, False or NaN).
        Do not update ``_is_tm``.
        """
        # N.B.: in some subclasses, it is possible to try one method, then another one if the first one fails,
        # etc. In this general class, we will simply do a switch between 'lazy' and 'exact'.
        getattr(self, '_tm_main_work_c_' + self.tm_option + '_')(c)
        # Launch a sub-method like _tm_main_work_c_lazy, etc.

    def _tm_main_work_c_lazy_(self, c):
        """Do the main work in TM loop for candidate ``c``, with option 'lazy'. Must decide ``_candidates_tm[c]`` (to
        True, False or NaN). Do not update ``_is_tm``.
        """
        self._candidates_tm[c] = neginf_to_nan(self._candidates_tm[c])

    def _tm_main_work_c_exact_(self, c):
        """Do the main work in TM loop for candidate ``c``, with option 'exact'. Must decide ``_candidates_tm[c]`` (to
        True, False or NaN). Do not update ``_is_tm``.
        """
        if self.is_based_on_rk:
            self._tm_main_work_c_exact_rankings_(c)
        elif self.is_based_on_ut_minus1_1:  # pragma: no cover
            # As of now, all the voting rules concerned (Majority Judgement, Range Voting and Approval)
            # have other ways to compute TM, so they do not use this.
            self._reached_uncovered_code()
            self._tm_main_work_c_exact_utilities_minus1_1_(c)
        else:
            raise NotImplementedError

    def _tm_main_work_c_exact_rankings_(self, c):
        """Do the main work in TM loop for candidate ``c``, with option 'exact', for a voting system based only on
        strict rankings. Must decide ``_candidates_tm[c]`` (to True, False or NaN). Do not update ``_is_tm``.
        """
        # Manipulators put c on top and w at bottom.
        w_test = self._copy(profile=self._compute_trivial_strategy_ordinal_(c)).w_
        self.mylogv("TM: w_test =", w_test)
        self._candidates_tm[c] = (w_test == c)

    def _tm_main_work_c_exact_utilities_minus1_1_(self, c):  # pragma: no cover
        """Do the main work in TM loop for candidate ``c``, with option 'exact', for a voting system based only on
        utilities and where it is optimal for a ``c``-manipulator to pretend that ``c`` has utility 1 and other
        candidates utility 0. Must decide ``_candidates_tm[c]`` (to True, False or NaN). Do not update ``_is_tm``.
        """
        # As of now, all the voting rules concerned (Majority Judgement, Range Voting and Approval)
        # have other ways to compute TM, so they do not use this.
        #
        # Manipulators give -1 to all candidates, except 1 for c.
        self._reached_uncovered_code()
        preferences_test = np.copy(self.profile_.preferences_ut)
        preferences_test[self.v_wants_to_help_c_[:, c], :] = -1
        preferences_test[self.v_wants_to_help_c_[:, c], c] = 1
        w_test = self._copy(profile=Profile(preferences_ut=preferences_test, sort_voters=False)).w_
        self._candidates_tm[c] = (w_test == c)

    def _tm_conclude_c_(self, c):
        """Conclude the TM loop for candidate ``c``, according to the value of ``_candidates_tm[c]``.

        ``_is_tm``:

        * If ``_candidates_tm[c]`` is True, then ``_is_tm = True``.
        * Otherwise, do not update ``_is_tm``.
        """
        if equal_true(self._candidates_tm[c]):
            self.mylogv("TM: Final answer: TM is True for c =", c, 2)
            self._is_tm = True
        elif equal_false(self._candidates_tm[c]):
            self.mylogv("TM: Final answer: TM is False for c =", c, 2)
        else:
            self.mylogv("TM: Final answer: TM is unknown for c =", c, 2)

    def _compute_tm_(self, with_candidates):
        """Compute TM: is_tm.

        Note that this method is launched by TM only if _is_tm is not decided, and by tm_with_candidates only if not
        ``_tm_was_computed_with_candidates``. So, it is not necessary to do a preliminary check on these variables.

        If ``with_candidates`` is False:
        * At the end, ``_is_tm`` must be decided to True, False or NaN.
        * ``_candidates_tm`` must be at least initialized (to an array of -inf). It can be partially decided to True,
          False or NaN (to avoid some computations if we come back later), but it is not mandatory.
        * Consistence is not mandatory: notably, if ``_is_tm`` is decided to True, it is not necessary to update a
          specific ``_candidates_tm[c]``.
        * If ``_is_tm`` and ``_candidates_tm`` are totally decided to True, False or NaN, then
          ``_tm_was_computed_with_candidates`` should become True (not mandatory but recommended).

        If ``with_candidates`` is True:
        * ``_is_tm`` and ``_candidates_tm`` must be decided to True, False or NaN.
        * ``_tm_was_computed_with_candidates`` must become True.
        """
        # We start with _is_tm = -Inf (undecided).
        # If we find a candidate for which _candidates_tm[c] = NaN, then _is_tm becomes NaN too ("at least maybe").
        # If we find a candidate for which _candidates_tm[c] = True, then _is_tm becomes True ("surely yes").
        for c in self.losing_candidates_:
            self._compute_tm_c_(c)
            if not with_candidates and equal_true(self._is_tm):
                return
            if np.isneginf(self._is_tm) and np.isnan(self._candidates_tm[c]):
                self._is_tm = np.nan
        # If we reach this point, we have decided all _candidates_tm to True, False or NaN.
        self._tm_was_computed_with_candidates = True  # even if with_candidates = False
        self._is_tm = neginf_to_zero(self._is_tm)

    def _compute_tm_c_(self, c):
        """Compute TM for candidate c.

        Note that this method checks if ``_candidates_tm[c]`` is already decided. So, it is not necessary to do this
        check before calling the method.

        During this method:

        * ``_candidates_tm[c]`` must be decided to True, False or NaN.
        * If it becomes True, then ``_is_tm`` must become True as well. Otherwise, do not update ``_is_tm``.
        """
        self._tm_initialize_c_(c)
        if np.isfinite(self._candidates_tm[c]):
            return
        self._tm_main_work_c_(c)
        self._tm_conclude_c_(c)

    def _compute_trivial_strategy_ordinal_(self, c):
        """Compute trivial strategy for a voting system based on strict rankings.

        Parameters
        ----------
        c : int
            The candidate for whom we want to manipulate.

        Returns
        -------
        Profile
            For each voter preferring ``c`` to ``w``, she now puts ``c`` on top, ``w`` at the bottom, and other
            Borda scores are modified accordingly.
        """
        preferences_rk = np.copy(self.profile_.preferences_rk)
        self.mylogm("Rankings (sincere) =", preferences_rk, 3)
        preferences_rk_manipulators = preferences_rk[self.v_wants_to_help_c_[:, c], :]
        sorting_array = np.array(preferences_rk_manipulators == self.w_, dtype=int) - (preferences_rk_manipulators == c)
        indexes = np.argsort(sorting_array)
        preferences_rk_manipulators = np.take_along_axis(preferences_rk_manipulators, indexes, axis=1)
        preferences_rk[self.v_wants_to_help_c_[:, c], :] = preferences_rk_manipulators
        self.mylogm("Rankings (with trivial strategy) =", preferences_rk, 3)
        return Profile(preferences_rk=preferences_rk, sort_voters=False)

    # %% Unison Manipulation (UM)

    @cached_property
    def log_um_(self):
        """String. Parameters used to compute :meth:`is_um_` and related methods.
        """
        return "um_option = " + self.um_option

    @cached_property
    def is_um_(self):
        """Boolean (or ``numpy.nan``). ``True`` if UM is possible, ``False`` otherwise. If the algorithm cannot decide,
        then ``numpy.nan``.

        We say that a situation is *unison-manipulable* for a candidate ``c`` ``!=`` :attr:`w_`  iff all voters who
        prefer ``c`` to the sincere winner :attr:`w_` can cast the **same** ballot so that ``c`` is elected (while
        other voters still vote sincerely).
        """
        _ = self._um_is_initialized_general_
        if np.isneginf(self._is_um):
            self._compute_um_(with_candidates=False)
        return pseudo_bool(self._is_um)

    def is_um_c_(self, c):
        """Unison manipulation, focus on one candidate.

        Parameters
        ----------
        c : int
            Candidate.

        Returns
        -------
        bool or nan
            ``candidates_um[c]``
        """
        _ = self._um_is_initialized_general_
        if np.isneginf(self._candidates_um[c]):
            self._compute_um_c_(c)
        return pseudo_bool(self._candidates_um[c])

    @cached_property
    def candidates_um_(self):
        """1d array of booleans (or ``numpy.nan``). ``candidates_um_[c]`` is ``True`` if UM for candidate ``c`` is
        possible, ``False`` otherwise. If the algorithm cannot decide, then ``numpy.nan``. For the sincere winner
        :attr:`w_`, we have by convention ``candidates_um_[w_] = False``.
        """
        _ = self._um_is_initialized_general_
        if not self._um_was_computed_with_candidates:
            self._compute_um_(with_candidates=True)
        return self._candidates_um.astype(np.float)

    @cached_property
    def _um_is_initialized_general_(self):
        """Initialize UM variables and do preliminary checks. Used only the first time UM is launched (whatever the
        mode).
        _is_um --> True or False if we know, -inf otherwise.
        _candidates_um[c] --> True or False if we know, -inf otherwise.

        If ``_candidates_um`` and ``_is_um`` are totally decided to True, False or NaN, then
        ``_um_was_computed_with_candidates`` should become True (not mandatory but recommended).
        """
        self.mylog("UM: Initialize", 2)
        self._um_was_computed_with_candidates = False
        self._candidates_um = np.full(self.profile_.n_c, -np.inf)
        self._is_um = -np.inf
        _ = self._um_is_initialized_general_subclass_
        self._um_preliminary_checks_general_()
        return True

    @cached_property
    def _um_is_initialized_general_subclass_(self):
        return True

    def _um_preliminary_checks_general_(self):
        """Do preliminary checks for UM. Only first time UM is launched.

        Can update some ``_candidates_um[c]`` to True or False (instead of -inf).

        * If some ``_candidates_um[c]`` becomes True, then ``_is_um`` must become True as well.
        * If ``_is_um`` becomes True, it is not necessary to update a specific ``_candidates_um[c].``
        * If for all candidates ``c``, ``_candidates_um[c]`` become False, then ``_is_um`` must be updated to False.
        * If ``_is_um`` becomes False, then all ``_candidates_um[c]`` must become False.
        * If ``_candidates_um`` and ``_is_um`` are totally decided to True or False, then
          ``_um_was_computed_with_candidates`` should become True (not mandatory but recommended).
        """
        # 1) Preliminary checks that may improve ``_candidates_um`` (all must be done, except if everything is decided).
        # Majority favorite criterion
        if self.meets_majority_favorite_c_ut and self.profile_.plurality_scores_ut[self.w_] > self.profile_.n_v / 2:
            self.mylog("UM impossible (w is a majority favorite).", 2)
            self._is_um = False
            self._candidates_um[:] = False
            self._um_was_computed_with_candidates = True
            return
        if (self.meets_majority_favorite_c_ut_ctb and self.w_ == 0
                and self.profile_.plurality_scores_ut[self.w_] >= self.profile_.n_v / 2):
            self.mylog("UM impossible (w=0 is a majority favorite with candidate tie-breaking).", 2)
            self._is_um = False
            self._candidates_um[:] = False
            self._um_was_computed_with_candidates = True
            return
        if self.meets_majority_favorite_c_rk and self.profile_.plurality_scores_rk[self.w_] > self.profile_.n_v / 2:
            self.mylog("UM impossible (w is a majority favorite with voter tie-breaking).", 2)
            self._is_um = False
            self._candidates_um[:] = False
            self._um_was_computed_with_candidates = True
            return
        if (self.meets_majority_favorite_c_rk_ctb and self.w_ == 0
                and self.profile_.plurality_scores_rk[self.w_] >= self.profile_.n_v / 2):
            self.mylog("UM impossible (w=0 is a majority favorite with voter and candidate tie-breaking).", 2)
            self._is_um = False
            self._candidates_um[:] = False
            self._um_was_computed_with_candidates = True
            return
        # Condorcet resistance
        if self.meets_condorcet_c_ut_abs and self.w_is_resistant_condorcet_winner_:
            self.mylog("UM impossible (w is a Resistant Condorcet winner)", 2)
            self._is_um = False
            self._candidates_um[:] = False
            self._um_was_computed_with_candidates = True
            return
        # Having supporters
        self._candidates_um[np.logical_not(self.c_has_supporters_)] = False
        # 2) Additional preliminary checks from the subclass.
        self._um_preliminary_checks_general_subclass_()
        if np.all(np.equal(self._candidates_um, False)):
            self.mylog("UM: preliminary checks: UM is impossible.", 2)
            self._is_um = False
            self._um_was_computed_with_candidates = True
            return
        if not np.isneginf(self._is_um):
            return
        # 3) Preliminary checks that gives only global information on _is_um (may return as soon as decision is made).
        if self.meets_majority_favorite_c_rk and self.w_is_not_condorcet_admissible_:
            self.mylog("UM found (w is not Condorcet-admissible)", 2)
            self._is_um = True
            return

    def _um_preliminary_checks_general_subclass_(self):
        """Do preliminary checks for UM. Only first time UM is launched.

        Can update some ``_candidates_um[c]`` to True or False (instead of -inf). True must be propagated from
        specific to general, False must be propagated from general to specific.

        * If some ``_candidates_um[c]`` becomes True, then ``_is_um`` must become True as well.
        * If ``_is_um`` becomes True, it is not necessary to update a specific ``_candidates_um[c]``.
        * If ``_is_um`` becomes False, then all ``_candidates_um[c]`` must become False.
        * If for all candidates ``c``, ``_candidates_um[c]`` becomes False, it is not necessary to update ``_is_um``.
        * If ``_candidates_um`` and ``_is_um`` are totally decided to True or False, then
          ``_um_was_computed_with_candidates`` should become True (not mandatory but recommended).

        Put first the checks that may improve ``_candidates_um`` (all must be done, except if everything is decided).
        Then the checks that gives only global information on ``_is_um`` (which may return as soon as decision is made).
        """
        pass

    def _um_initialize_c_(self, c):
        """Initialize the UM loop for candidate ``c`` and may do preliminary checks.

        * If ``_candidates_um[c]`` is decided (True/False/NaN), it means that all the work for ``c`` has been done
          before. Then get out.
        * Preliminary checks: try to decide ``_candidates_um[c]``. If it becomes True, then ``_is_um`` must become True
          as well. In other cases, do not update ``_is_um``.
        """
        self.mylogv("UM: Candidate =", c, 2)
        # Check if job is done for c
        if not np.isneginf(self._candidates_um[c]):
            self.mylog("UM: Job already done", 2)
            return
        # Preliminary checks
        self._um_preliminary_checks_c_(c)
        # Conclude what we can
        if equal_true(self._candidates_um[c]):
            self.mylogv("UM: Preliminary checks: UM is True for c =", c, 2)
            self._is_um = True
        elif equal_false(self._candidates_um[c]):
            self.mylogv("UM: Preliminary checks: UM is False for c =", c, 2)
        else:
            self.mylogv("UM: Preliminary checks: UM is unknown for c =", c, 2)

    def _um_preliminary_checks_c_(self, c):
        """UM: preliminary checks for challenger ``c``.

        Try to decide _candidates_um[c]`` to True or False (instead of -inf). Do not update ``_is_um``.
        """
        n_m = self.profile_.matrix_duels_ut[c, self.w_]  # Number of manipulators
        n_s = self.profile_.n_v - n_m                    # Number of sincere voters
        # Positive pretest based on the majority favorite criterion
        if self.meets_majority_favorite_c_rk and n_m > self.profile_.n_v / 2:
            self.mylog('UM: Preliminary checks: n_m > n_v / 2', 3)
            self._candidates_um[c] = True
            return
        if self.meets_majority_favorite_c_rk_ctb and c == 0 and n_m >= self.profile_.n_v / 2:
            self.mylog('UM: Preliminary checks: n_m >= n_v / 2 and c == 0', 3)
            self._candidates_um[c] = True
            return
        # Negative pretest based on the majority favorite criterion
        # If ``plurality_scores_ut[w] > (n_s + n_m) / 2``, then CM impossible.
        # Necessary condition: ``n_m >= 2 * plurality_scores_ut[w] - n_s``.
        if self.meets_majority_favorite_c_ut:
            if n_m < 2 * self.profile_.plurality_scores_ut[self.w_] - n_s:  # pragma: no cover
                # TO DO: Investigate whether this case can actually happen.
                self._reached_uncovered_code()
                self.mylog('UM: Preliminary checks: even with n_m manipulators, w stays plurality winner (ut)', 3)
                self._candidates_um[c] = False
                return
        if self.meets_majority_favorite_c_ut_ctb and self.w_ == 0:
            if n_m < 2 * self.profile_.plurality_scores_ut[self.w_] - n_s + 1:  # pragma: no cover
                # TO DO: Investigate whether this case can actually happen.
                self._reached_uncovered_code()
                self.mylog('UM: Preliminary checks: even with n_m manipulators, w stays plurality winner (ut, ctb)', 3)
                self._candidates_um[c] = False
                return
        if self.meets_majority_favorite_c_rk:
            if n_m < 2 * self.profile_.plurality_scores_rk[self.w_] - n_s:  # pragma: no cover
                # TO DO: Investigate whether this case can actually happen.
                self._reached_uncovered_code()
                self.mylog('UM: Preliminary checks: even with n_m manipulators, w stays plurality winner (rk)', 3)
                self._candidates_um[c] = False
                return
        if self.meets_majority_favorite_c_rk_ctb and self.w_ == 0:
            if n_m < 2 * self.profile_.plurality_scores_rk[self.w_] - n_s + 1:  # pragma: no cover
                # TO DO: Investigate whether this case can actually happen.
                self._reached_uncovered_code()
                self.mylog('UM: Preliminary checks: even with n_m manipulators, w stays plurality winner (rk, ctb)', 3)
                self._candidates_um[c] = False
                return
        # Pretest based on the same idea as Condorcet resistance
        if self.meets_condorcet_c_ut_abs:
            if n_m < self.profile_.threshold_c_prevents_w_condorcet_ut_abs[c, self.w_]:
                self.mylog('UM: Preliminary checks: c-manipulators cannot prevent w from being a Condorcet winner', 3)
                self._candidates_um[c] = False
                return
        # Other pretests
        self._um_preliminary_checks_c_subclass_(c)

    def _um_preliminary_checks_c_subclass_(self, c):
        """UM: preliminary checks for challenger ``c``.

        Try to decide ``_candidates_um[c]`` to True or False (instead of -inf). Do not update ``_is_um``.
        """
        pass

    def _um_main_work_c_(self, c):
        """ Do the main work in UM loop for candidate ``c``. Must decide ``_candidates_um[c]`` (to True,
        False or NaN). Do not update ``_is_um``.
        """
        # N.B.: in some subclasses, it is possible to try one method, then another one if the first one fails,
        # etc. In this general class, we will simply do a switch between 'lazy' and 'exact'.
        getattr(self, '_um_main_work_c_' + self.um_option + '_')(c)
        # Launch a sub-method like _um_main_work_c_lazy, etc.

    def _um_main_work_c_lazy_(self, c):
        """Do the main work in UM loop for candidate c, with option 'lazy'. Must decide ``_candidates_um[c]`` (to True,
        False or NaN). Do not update ``_is_um``.
        """
        self._candidates_um[c] = neginf_to_nan(
            self._candidates_um[c])

    def _um_main_work_c_exact_(self, c):
        """Do the main work in UM loop for candidate ``c``, with option 'exact'. Must decide ``_candidates_um[c]`` (to
        True, False or NaN). Do not update ``_is_um``.
        """
        self.mylogv("UM: Compute UM for c =", c, 1)
        if self.is_based_on_rk:
            self._um_main_work_c_exact_rankings_(c)
        elif self.is_based_on_ut_minus1_1:  # pragma: no cover
            # As of now, all the voting rules concerned (Majority Judgement, Range Voting and Approval)
            # have other ways to compute UM, so they do not use this.
            self._reached_uncovered_code()
            self._um_main_work_c_exact_utilities_minus1_1_(c)
        else:
            raise NotImplementedError

    def _um_main_work_c_exact_rankings_(self, c):
        """Do the main work in UM loop for candidate ``c``, with option 'exact', for a voting system based only on
        strict rankings. Must decide ``_candidates_um[c]`` (to True, False or NaN). Do not update ``_is_um``.
        """
        profile_s = Profile(
            preferences_rk=self.profile_.preferences_rk[np.logical_not(self.v_wants_to_help_c_[:, c]), :],
            preferences_borda_rk=self.profile_.preferences_borda_rk[np.logical_not(self.v_wants_to_help_c_[:, c]), :],
            sort_voters=False
        )
        base_ballot = [c] + list([i for i in range(self.profile_.n_c) if i != c])  # Put c first for the first try...
        n_m = self.profile_.matrix_duels_ut[c, self.w_]
        for ballot in itertools.permutations(base_ballot):
            self.mylogv("UM: Ballot =", ballot, 3)
            profile_um = ProfileUM(profile_s=profile_s, n_m=n_m, ballot_rk=ballot)
            w_test = self._copy(profile=profile_um).w_
            self.mylogv("UM: w_test =", w_test, 3)
            if w_test == c:
                self._candidates_um[c] = True
                return
        else:
            self._candidates_um[c] = False

    def _um_main_work_c_exact_utilities_minus1_1_(self, c):  # pragma: no cover
        """Do the main work in UM loop for candidate ``c``, with option 'exact', for a voting system based only on
        utilities and where it is optimal for a c-manipulator to pretend that ``c`` has utility 1 and other
        candidates utility 0. Must decide ``_candidates_um[c]`` (to True, False or NaN). Do not update ``_is_um``.
        """
        # As of now, all the voting rules concerned (Majority Judgement, Range Voting and Approval)
        # have other ways to compute UM, so they do not use this.
        self._reached_uncovered_code()
        self._candidates_um[c] = self.is_tm_c_(c)

    def _um_conclude_c_(self, c):
        """Conclude the UM loop for candidate ``c``, according to the value of ``_candidates_um[c]``.
        ``_is_um`` -->
            * If ``_candidates_um[c]`` is True, then ``_is_um = True``.
            * Otherwise, do not update ``_is_um``.
        """
        if equal_true(self._candidates_um[c]):
            self.mylogv("UM: Final answer: UM is True for c =", c, 2)
            self._is_um = True
        elif equal_false(self._candidates_um[c]):
            self.mylogv("UM: Final answer: UM is False for c =", c, 2)
        else:
            self.mylogv("UM: Final answer: UM is unknown for c =", c, 2)

    def _compute_um_(self, with_candidates):
        """Compute UM: ``is_um``.

        Note that this method is launched by ``is_um_`` only if ``_is_um`` is not decided, and by ``candidates_um_``
        only if not ``_um_was_computed_with_candidates``. So, it is not necessary to do a preliminary check on these
        variables.

        If ``with_candidates`` is False:

        * At the end, ``_is_um`` must be decided to True, False or NaN.
        * ``_candidates_um`` must be at least initialized (to an array of -inf). It can be partially decided to True,
          False or NaN (to avoid some computations if we come back later), but it is not mandatory.
        * Consistence is not mandatory: notably, if ``_is_um`` is decided to True, it is not necessary to update a
          specific ``_candidates_um[c]``.
        * If ``_is_um`` and ``_candidates_um`` are totally decided to True, False or NaN, then
          ``_um_was_computed_with_candidates`` should become True (not mandatory but recommended).

        If ``with_candidates`` is True:

        * ``_is_um`` and ``_candidates_um`` must be decided to True, False or NaN.
        * ``_um_was_computed_with_candidates`` must become True.
        """
        # We start with ``_is_um = -Inf`` (undecided).
        # If we find a candidate for which ``_candidates_um[c] = NaN``, then ``_is_um`` becomes NaN too ("at least
        #   maybe").
        # If we find a candidate for which ``_candidates_um[c] = True``, then ``_is_um`` becomes True ("surely yes").
        for c in self.losing_candidates_:
            self._compute_um_c_(c)
            if not with_candidates and equal_true(self._is_um):
                return
            if np.isneginf(self._is_um) and np.isnan(self._candidates_um[c]):
                self._is_um = np.nan
        # If we reach this point, we have decided all _candidates_um to True, False or NaN.
        self._um_was_computed_with_candidates = True  # even if with_candidates = False
        self._is_um = neginf_to_zero(self._is_um)

    def _compute_um_c_(self, c):
        """Compute UM for candidate ``c``.

        Note that this method checks if ``_candidates_um[c]`` is already decided. So, it is not necessary to do this
        check before calling the method.

        During this method:

        * ``_candidates_um[c]`` must be decided to True, False or NaN.
        * If it becomes True, then ``_is_um`` must become True as well. Otherwise, do not update ``_is_um``.
        """
        self._um_initialize_c_(c)
        if np.isfinite(self._candidates_um[c]):
            return
        self._um_main_work_c_(c)
        self._um_conclude_c_(c)

    # %% Ignorant-Coalition Manipulation (ICM)
    # When the voting systems meets IgnMC with ctb, it is very easy, and it is managed at the beginning of
    # ``_compute_icm``. So, for most subroutines, we can suppose that the voting system does not meet IgnMC with ctb.

    @cached_property
    def log_icm_(self):
        """String. Parameters used to compute :meth:`is_icm_` and related methods.
        """
        return "icm_option = " + self.icm_option

    @cached_property
    def is_icm_(self):
        """Boolean (or ``numpy.nan``). ``True`` if an ICM is possible, ``False`` otherwise. If the algorithm cannot
        decide, then ``numpy.nan``.

        We say that a situation is *Ignorant-Coalition Manipulable* (ICM) for ``c`` ``!=`` :attr:`w_` iff the voters
        who prefer ``c`` to :attr:`w_` can cast ballots so that, whatever the other voters do, ``c`` is elected.
        """
        _ = self._icm_is_initialized_general_
        if np.isneginf(self._is_icm):
            self._compute_icm_(with_candidates=False, optimize_bounds=False)
        return pseudo_bool(self._is_icm)

    def is_icm_c_(self, c):
        """Ignorant-Coalition Manipulation, focus on one candidate.

        Parameters
        ----------
        c : int
            Candidate.

        Returns
        -------
        bool or nan
            ``candidates_icm[c]_``.
        """
        _ = self._icm_is_initialized_general_
        if np.isneginf(self._candidates_icm[c]):
            self._compute_icm_c_(c, optimize_bounds=False)
        return pseudo_bool(self._candidates_icm[c])

    def is_icm_c_with_bounds_(self, c):
        """Ignorant-Coalition Manipulation, focus on one candidate, with bounds.

        Parameters
        ----------
        c : int
            Candidate.

        Returns
        -------
        tuple
            (``candidates_icm_[c]``, ``necessary_coalition_size_icm_[c]``, ``sufficient_coalition_size_icm_[c]``).
        """
        _ = self._icm_is_initialized_general_
        if equal_false(self._bounds_optimized_icm[c]):
            self._compute_icm_c_(c, optimize_bounds=True)
        return (pseudo_bool(self._candidates_icm[c]), np.float(self._necessary_coalition_size_icm[c]),
                np.float(self._sufficient_coalition_size_icm[c]))

    @cached_property
    def candidates_icm_(self):
        """1d array of booleans (or ``numpy.nan``). ``candidates_icm_[c]`` is ``True`` if ICM for candidate ``c`` is
        possible, ``False`` otherwise. If the algorithm cannot decide, then ``numpy.nan``. For the sincere winner
        :attr:`w_`, we have by convention ``candidates_icm[w_] = False``.
        """
        _ = self._icm_is_initialized_general_
        if not self._icm_was_computed_with_candidates:
            self._compute_icm_(with_candidates=True, optimize_bounds=False)
        return self._candidates_icm.astype(np.float)

    @cached_property
    def _coalition_sizes_icm_(self):
        _ = self._icm_is_initialized_general_
        if not self._icm_was_computed_full:
            self._compute_icm_(with_candidates=True, optimize_bounds=True)
        return {'necessary_coalition_size_icm': self._necessary_coalition_size_icm.astype(np.float),
                'sufficient_coalition_size_icm': self._sufficient_coalition_size_icm.astype(np.float)}

    @cached_property
    def necessary_coalition_size_icm_(self):
        """1d array of integers. ``necessary_coalition_size_icm[c]`` is the lower bound found by the algorithm for
        :math:`x_c` (see below). For the sincere winner :attr:`w_`, we have by convention
        ``necessary_coalition_size_icm_[w_] = 0``.

        Internally, to decide the problem of ICM, SVVAMP studies the following question. When considering the
        sub-population of voters who do not prefer ``c`` to :attr:`w_` (sincere voters), what is the minimal number
        :math:`x_c` of ``c``-manipulators needed to perform ICM? For all voting system currently implemented in
        SVVAMP, it means that ICM is possible iff there are :math:`x_c` voters or more who prefer ``c`` to :attr:`w_`.

        For information only, the result of SVVAMP's computations about :math:`x_c` is given in outputs
        ``necessary_coalition_size_icm_`` and ``sufficient_coalition_size_icm_``. By definition, we have
        ``necessary_coalition_size_icm_[c]`` :math:`\leq x_c \leq` ``sufficient_coalition_size_icm_[c]``.

        When :attr:`icm_option` = ``'exact'``, the exactness concerns the ICM decision problems (boolean results below),
        but not the numerical evaluation of :math:`x_c`. It means that for all boolean answers, SVVAMP will not answer
        ``numpy.nan`` ( undecided). But it is possible that ``necessary_coalition_size_icm_[c]`` <
        ``sufficient_coalition_size_icm_[c]``.
        """
        return self._coalition_sizes_icm_['necessary_coalition_size_icm']

    @cached_property
    def sufficient_coalition_size_icm_(self):
        """1d of integers or ``numpy.inf``. ``sufficient_coalition_size_icm[c]`` is the upper bound found by the
        algorithm for :math:`x_c` (see :meth:`necessary_coalition_size_icm_`). For the sincere winner :attr:`w_`,
        we have by convention ``sufficient_coalition_size_icm_[w_] = 0``.
        """
        return self._coalition_sizes_icm_['sufficient_coalition_size_icm']

    @cached_property
    def _icm_is_initialized_general_(self):
        """Initialize ICM variables an do preliminary checks. Used each time ICM is launched (whatever the mode).
        ``_is_icm`` --> False or True if we know, -inf otherwise.
        ``_candidates_icm[c]`` --> True or False if we know, -inf otherwise.

        ``_sufficient_coalition_size_icm[c]`` --> +inf (except for w).
        ``_necessary_coalition_size_icm`` --> 0.
        ``_bounds_optimized_icm[c]`` --> False.
        For ``_sufficient_coalition_size_icm`` and ``_necessary_coalition_size_icm``, it is not recommended to do better
        here.
        """
        self.mylog("ICM: Initialize", 2)
        self._icm_was_computed_with_candidates = False
        self._icm_was_computed_full = False
        self._candidates_icm = np.full(self.profile_.n_c, -np.inf)
        self._candidates_icm[self.w_] = False
        self._sufficient_coalition_size_icm = np.full(self.profile_.n_c, np.inf)
        self._sufficient_coalition_size_icm[self.w_] = 0
        self._necessary_coalition_size_icm = np.zeros(self.profile_.n_c)
        self._bounds_optimized_icm = np.zeros(self.profile_.n_c)
        self._bounds_optimized_icm[self.w_] = True
        self._is_icm = -np.inf
        _ = self._icm_is_initialized_general_subclass_
        self._icm_preliminary_checks_general_()
        return True

    @cached_property
    def _icm_is_initialized_general_subclass_(self):
        return True

    def _icm_preliminary_checks_general_(self):
        """Do preliminary checks for ICM. Only first time ICM is launched.

        Can update some ``_candidates_icm[c]`` to True or False (instead of -inf).

        * If some ``_candidates_icm[c]`` becomes True, then ``_is_icm`` must become True as well.
        * If ``_is_icm`` becomes True, it is not necessary to update a specific ``_candidates_icm[c]``.
        * If for all candidates ``c``, ``_candidates_icm[c]`` become False, then ``_is_icm`` must be updated to False.
        * If ``_is_icm`` becomes False, then all ``_candidates_icm[c]`` must become False.
        * If ``_candidates_icm`` and ``_is_icm`` are totally decided to True or False, then
          ``_icm_was_computed_with_candidates`` should become True (not mandatory but recommended).
        """
        if self.meets_infmc_c and self.w_is_condorcet_winner_ut_abs_:
            self.mylog("ICM impossible (w is a Condorcet winner)", 2)
            self._is_icm = False
            self._candidates_icm[:] = False
            self._icm_was_computed_with_candidates = True
            return
        self._candidates_icm[np.logical_not(self.c_has_supporters_)] = False
        if np.all(np.equal(self._candidates_icm, False)):
            self.mylog("ICM: preliminary checks: ICM is impossible.", 2)
            self._is_icm = False
            self._icm_was_computed_with_candidates = True
            return
        if self.meets_ignmc_c and self.w_is_not_condorcet_admissible_:
            self.mylog("ICM found (w is not Condorcet-admissible)", 2)
            self._is_icm = True
            return
        # Other checks
        self._icm_preliminary_checks_general_subclass_()

    def _icm_preliminary_checks_general_subclass_(self):
        """Do preliminary checks for ICM. Only first time ICM is launched.

        Can update ``_is_icm`` to True or False (instead of -inf).

        * If ``_is_icm`` becomes True, it is not necessary to update a specific ``_candidates_icm[c]``.
        * If ``_is_icm`` becomes False, then all ``_candidates_icm[c]`` must become False. And it is recommended that
          ``_icm_was_computed_with_candidates`` becomes True.

        For ``_sufficient_coalition_size_icm`` and ``_necessary_coalition_size_icm``, it is not recommended to do
        better here.
        """
        pass

    def _icm_initialize_c(self, c, optimize_bounds):
        """Initialize the ICM loop for candidate ``c`` and do preliminary checks.

        * If ``_bounds_optimized_icm[c]`` is True, it means that all the work for ``c`` has been done before. Then get
          out.
        * If ``_candidates_icm[c]`` is decided (True/False/NaN) and ``optimize_bounds`` is False, then get out.
        * Preliminary checks to improve bounds ``_sufficient_coalition_size_icm[c]`` and
          ``_necessary_coalition_size_icm[c]``.
        * If the two bounds are equal, then ``_bounds_optimized_icm[c]`` becomes True.
        * Update ``_candidates_icm[c]`` to True or False if possible.
        * If we can decide ``_is_icm`` to True, do it.

        :return: Boolean, ``job_done``. True iff we have done all the job for ``c`` (with bounds if ``optimize_bounds``
        is True, only for ``_candidates_icm[c]`` otherwise).
        """
        self.mylogv("ICM: Candidate =", c, 2)
        # Check if job is done for c
        if equal_true(self._bounds_optimized_icm[c]):
            self.mylog("ICM: Job already done", 2)
            return True
        if equal_false(optimize_bounds) and not np.isneginf(self._candidates_icm[c]):
            self.mylog("ICM: Job already done", 2)
            return True
        # Improve bounds
        self._icm_preliminary_checks_c_(c, optimize_bounds)
        # Conclude what we can
        # Some log
        n_m = self.profile_.matrix_duels_ut[c, self.w_]
        self.mylogv("ICM: Preliminary checks: necessary_coalition_size_icm[c] =",
                    self._necessary_coalition_size_icm[c], 3)
        self.mylogv("ICM: Preliminary checks: sufficient_coalition_size_icm[c] =",
                    self._sufficient_coalition_size_icm[c], 3)
        self.mylogv("ICM: Preliminary checks: n_m =", n_m, 3)
        # Conclude
        if self._sufficient_coalition_size_icm[c] == self._necessary_coalition_size_icm[c]:
            self.mylog("ICM: Preliminary checks: Bounds are equal", 2)
            self._bounds_optimized_icm[c] = True
        if n_m >= self._sufficient_coalition_size_icm[c]:
            self.mylogv("ICM: Preliminary checks: ICM is True for c =", c, 2)
            self._candidates_icm[c] = True
            self._is_icm = True
            if equal_false(optimize_bounds) or self._bounds_optimized_icm[c]:
                return True
        elif n_m < self._necessary_coalition_size_icm[c]:
            self.mylogv("ICM: Preliminary checks: ICM is False for c =", c, 2)
            self._candidates_icm[c] = False
            if equal_false(optimize_bounds) or self._bounds_optimized_icm[c]:
                return True
        else:
            self.mylogv("ICM: Preliminary checks: ICM is unknown for c =", c, 2)
        return False

    def _icm_preliminary_checks_c_(self, c, optimize_bounds):
        """ICM: preliminary checks for challenger ``c``.

        Try to improve bounds ``_sufficient_coalition_size_icm[c]`` and ``_necessary_coalition_size_icm[c]``. Do not
        update the other variables.

        If ``optimize_bounds`` is False, then return as soon as ``n_m >= _sufficient_coalition_size_icm[c]``, or
        ``_necessary_coalition_size_icm[c] > n_m`` (where ``n_m`` is the number or manipulators).
        """
        n_m = self.profile_.matrix_duels_ut[c, self.w_]  # Number of manipulators
        n_s = self.profile_.n_v - n_m  # Number of sincere voters
        if self.meets_infmc_c_ctb and c != 0:
            self._update_necessary(self._necessary_coalition_size_icm, c, n_s + 1,
                                   'ICM: InfMC_c_ctb => necessary_coalition_size_icm[c] = n_s + 1 =')
            if not optimize_bounds and self._necessary_coalition_size_icm[c] > n_m:
                return
        if self.meets_infmc_c:
            self._update_necessary(self._necessary_coalition_size_icm, c, n_s,
                                   'ICM: InfMC_c => necessary_coalition_size_icm[c] = n_s =')
            if not optimize_bounds and self._necessary_coalition_size_icm[c] > n_m:
                return
        if self.meets_ignmc_c_ctb and c == 0:
            self._update_sufficient(self._sufficient_coalition_size_icm, c, n_s,
                                    'ICM: IgnMC_c => sufficient_coalition_size_icm[c] = n_s =')
            if not optimize_bounds and n_m >= self._sufficient_coalition_size_icm[c]:
                return
        if self.meets_ignmc_c:
            self._update_sufficient(self._sufficient_coalition_size_icm, c, n_s + 1,
                                    'ICM: IgnMC_c => sufficient_coalition_size_icm[c] = n_s + 1 =')
            if not optimize_bounds and n_m >= self._sufficient_coalition_size_icm[c]:
                return
        # Other preliminary checks
        self._icm_preliminary_checks_c_subclass_(c, optimize_bounds)

    def _icm_preliminary_checks_c_subclass_(self, c, optimize_bounds):
        """ICM: preliminary checks for challenger ``c``.

        Try to improve bounds ``_sufficient_coalition_size_icm[c]`` and ``_necessary_coalition_size_icm[c]``. Do not
        update the other variables.

        If ``optimize_bounds`` is False, then return as soon as ``n_m >= _sufficient_coalition_size_icm[c]``, or
        ``_necessary_coalition_size_icm[c] > n_m`` (where ``n_m`` is the number or manipulators).

        If a test is especially costly, it is recommended to test first if ``_sufficient_coalition_size_icm[c] ==
        _necessary_coalition_size_icm[c]`` and to return immediately in that case.
        """
        pass

    def _icm_main_work_c_(self, c, optimize_bounds):
        """Do the main work in ICM loop for candidate ``c``.

        * Try to improve bounds ``_sufficient_coalition_size_icm[c]`` and ``_necessary_coalition_size_icm[c]``.
        * Do not update other variables (``_is_icm``, ``_candidates_icm``, etc.).

        If ``optimize_bounds`` is False, can return as soon as ``n_m >= _sufficient_coalition_size_icm[c]``, or
        ``_necessary_coalition_size_icm[c] > n_m`` (where ``n_m`` is the number or manipulators).

        :return: Boolean, ``is_quick_escape``. True if we did not improve the bound the best we could. (Allowed to be
            None or False otherwise).
        """
        # N.B.: in some subclasses, it is possible to try one method, then another one if the first one fails,
        # etc. In this general class, we will simply do a switch between 'lazy' and 'exact'.
        return getattr(self, '_icm_main_work_c_' + self.icm_option + '_')(c, optimize_bounds)
        # Launch a sub-method like _icm_main_work_c_lazy_, etc.

    # noinspection PyMethodMayBeStatic,PyUnusedLocal
    def _icm_main_work_c_lazy_(self, c, optimize_bounds):
        """Do the main work in ICM loop for candidate ``c``, with option 'lazy'. Same specifications as
        ``_icm_main_work_c_``.
        """
        # With option 'lazy', there is nothing to do! And this is not a 'quick escape': we did the best we could
        # (considering laziness). N.B.: for most voting system, lazy is actually quite good for ICM! In fact,
        # as soon as _meets_IgnMC_c_ctb, this lazy method is exact!
        return False

    # noinspection PyUnusedLocal
    def _icm_main_work_c_exact_(self, c, optimize_bounds):
        """Do the main work in ICM loop for candidate ``c``, with option 'exact'. Same specifications as
        ``_icm_main_work_c``.
        """
        if self.meets_ignmc_c_ctb:  # pragma: no cover
            # TO DO: Investigate whether this case can actually happen.
            self._reached_uncovered_code()
            return False
        else:
            raise NotImplementedError

    def _icm_conclude_c(self, c, is_quick_escape):
        """Conclude the ICM loop for candidate ``c``.
        ``_bounds_optimized_icm[c]`` --> if not ``quick_escape``, becomes True.
        ``_candidates_icm[c]`` --> True, False or NaN according to the bounds ``_sufficient_coalition_size_icm[c]`` and
            ``_necessary_coalition_size_icm[c]``.
        ``_is_icm`` -->
            * If ``_candidates_icm[c]`` is True, then ``_is_icm`` = True.
            * Otherwise, do not update ``_is_icm``.
        """
        if not is_quick_escape:
            self._bounds_optimized_icm[c] = True
        n_m = self.profile_.matrix_duels_ut[c, self.w_]
        if n_m >= self._sufficient_coalition_size_icm[c]:
            self.mylogv("ICM: Final answer: ICM is True for c =", c, 2)
            self._candidates_icm[c] = True
            self._is_icm = True
        elif n_m < self._necessary_coalition_size_icm[c]:
            self.mylogv("ICM: Final answer: ICM is False for c =", c, 2)
            self._candidates_icm[c] = False
        else:
            self.mylogv("ICM: Final answer: ICM is unknown for c =", c, 2)
            self._candidates_icm[c] = np.nan

    def _compute_icm_(self, with_candidates, optimize_bounds):
        """Compute ICM.

        Note that this method is launched by ``is_icm_`` only if ICM was not initialized, and by ``candidates_icm_``
        only if not ``_icm_was_computed_with_candidates``. So, it is not necessary to do a preliminary check on these
        variables.
        """
        self.mylog("Compute ICM", 1)
        # We start with ``_is_icm`` = -Inf (undecided).
        # If we find a candidate for which ``_candidates_icm[c] = NaN``, then ``_is_icm`` becomes NaN too ("at least
        #   maybe").
        # If we find a candidate for which ``_candidates_icm[c] = True``, then ``_is_icm`` becomes True ("surely yes").
        for c in self.losing_candidates_:
            self._compute_icm_c_(c, optimize_bounds)
            if not with_candidates and equal_true(self._is_icm):
                return
            if np.isneginf(self._is_icm) and np.isnan(self._candidates_icm[c]):
                self._is_icm = np.nan
        # If we reach this point, we have decided all ``_candidates_icm`` to True, False or NaN.
        self._icm_was_computed_with_candidates = True
        self._is_icm = neginf_to_zero(self._is_icm)
        if optimize_bounds:
            self._icm_was_computed_full = True

    def _compute_icm_c_(self, c, optimize_bounds):
        job_done = self._icm_initialize_c(c, optimize_bounds)
        if job_done:
            return
        if not optimize_bounds and not np.isneginf(self._candidates_icm[c]):  # pragma: no cover
            # TO DO: Investigate whether this case can actually happen.
            self._reached_uncovered_code()
            return
        is_quick_escape = self._icm_main_work_c_(c, optimize_bounds)
        self._icm_conclude_c(c, is_quick_escape)

    # %% Coalition Manipulation (CM)

    @cached_property
    def log_cm_(self):
        """String. Parameters used to compute :meth:`is_cm_` and related methods.
        """
        if self.cm_option == 'exact':
            return "cm_option = exact"
        else:
            return ("cm_option = " + self.cm_option +
                    self.precheck_um * (", " + self.log_um_) +
                    self.precheck_icm * (", " + self.log_icm_) +
                    self.precheck_tm * (", " + self.log_tm_))

    @cached_property
    def is_cm_(self):
        """Boolean (or ``numpy.nan``). ``True`` if a CM is possible, ``False`` otherwise. If the algorithm cannot
        decide, then ``numpy.nan``.

        We say that a situation is *Coalitionally Manipulable* (CM) for ``c`` ``!=`` :attr:`w_` iff voters who prefer
        ``c`` to :attr:`w_` can cast ballots so that ``c`` is elected (while other voters still vote sincerely).
        """
        _ = self._cm_is_initialized_general_
        if np.isneginf(self._is_cm):
            self._compute_cm_(with_candidates=False, optimize_bounds=False)
        return pseudo_bool(self._is_cm)

    def is_cm_c_(self, c):
        """Coalition Manipulation, focus on one candidate.

        Parameters
        ----------
        c : int
            Candidate.

        Returns
        -------
        bool or nan
            ``candidates_cm[c]``.
        """
        _ = self._cm_is_initialized_general_
        if np.isneginf(self._candidates_cm[c]):
            self._compute_cm_c_(c, optimize_bounds=False)
        return pseudo_bool(self._candidates_cm[c])

    def is_cm_c_with_bounds_(self, c):
        """Coalition Manipulation, focus on one candidate, with bounds.

        Parameters
        ----------
        c : int
            Candidate.

        Returns
        -------
        tuple
            (``candidates_cm[c]``, ``necessary_coalition_size_cm[c]``, ``sufficient_coalition_size_cm[c]``).
        """
        _ = self._cm_is_initialized_general_
        if equal_false(self._bounds_optimized_cm[c]):
            self._compute_cm_c_(c, optimize_bounds=True)
        return (pseudo_bool(self._candidates_cm[c]), np.float(self._necessary_coalition_size_cm[c]),
                np.float(self._sufficient_coalition_size_cm[c]))

    @cached_property
    def candidates_cm_(self):
        """1d array of booleans (or ``numpy.nan``). ``candidates_cm_[c]`` is ``True`` if CM for candidate ``c`` is
        possible, ``False`` otherwise. If the algorithm cannot decide, then ``numpy.nan``. For the sincere winner
        :attr:`w_`, we have by convention ``candidates_cm_[w_] = False``.
        """
        _ = self._cm_is_initialized_general_
        if not self._cm_was_computed_with_candidates:
            self._compute_cm_(with_candidates=True, optimize_bounds=False)
        return self._candidates_cm.astype(np.float)

    @cached_property
    def _coalition_sizes_cm_(self):
        _ = self._cm_is_initialized_general_
        if not self._cm_was_computed_full:
            self._compute_cm_(with_candidates=True, optimize_bounds=True)
        return {'necessary_coalition_size_cm': self._necessary_coalition_size_cm.astype(np.float),
                'sufficient_coalition_size_cm': self._sufficient_coalition_size_cm.astype(np.float)}

    @cached_property
    def necessary_coalition_size_cm_(self):
        """1d array of integers. ``necessary_coalition_size_cm_[c]`` is the lower bound found by the algorithm for
        :math:`x_c` (cf. below). For the sincere winner :attr:`w_`, we have by convention
        ``necessary_coalition_size_cm_[w_] = 0``.

        Internally, to decide the problem of CM, SVVAMP studies the following question. When considering the
        sub-population of voters who do not prefer ``c`` to :attr:`w_` (sincere voters), what is the minimal number
        :math:`x_c` of ``c``-manipulators needed to perform CM? For all voting system currently implemented in SVVAMP,
        it means that CM is possible iff there are :math:`x_c` voters or more who prefer ``c`` to :attr:`w_`. (A
        sufficient condition on the voting system is that, if a population elects ``c``, then an additional voter may
        always cast a ballot so that ``c`` stays elected)

        For information only, the result of SVVAMP's computations about :math:`x_c` is given in outputs
        ``necessary_coalition_size_cm_`` and ``sufficient_coalition_size_cm_``. By definition, we have
        ``necessary_coalition_size_cm_[c]`` :math:`\leq x_c \leq` ``sufficient_coalition_size_cm_[c]``.

        When :attr:`cm_option` = ``'exact'``, the exactness concerns the CM decision problems (Boolean results), not
        the numerical evaluation of :math:`x_c`. It means that for all Boolean answers, SVVAMP will not answer
        ``numpy.nan`` ( undecided). But it is possible that ``necessary_coalition_size_cm_[c]`` <
        ``sufficient_coalition_size_cm_[c]``.
        """
        return self._coalition_sizes_cm_['necessary_coalition_size_cm']

    @cached_property
    def sufficient_coalition_size_cm_(self):
        """1d array of integers or ``numpy.inf``. ``sufficient_coalition_size_cm_[c]`` is the upper bound found by the
        algorithm for :math:`x_c` (cf. :attr:`necessary_coalition_size_cm_`). For the sincere winner :attr:`w_`, we
        have by convention ``sufficient_coalition_size_cm_[w_] = 0``.
        """
        return self._coalition_sizes_cm_['sufficient_coalition_size_cm']

    @cached_property
    def _cm_is_initialized_general_(self):
        """Initialize CM variables and do preliminary checks. Used only the first time CM is launched (whatever the
        mode).
        ``_is_cm`` --> False or True if we know, -inf otherwise.
        ``_candidates_cm[c]`` --> True or False if we know, -inf otherwise.

        ``_sufficient_coalition_size_cm[c]`` --> +inf (except for ``w_``).
        ``_necessary_coalition_size_cm[c]`` --> 0.
        ``_bounds_optimized_cm[c]`` --> False.
        For ``_sufficient_coalition_size_cm`` and ``_necessary_coalition_size_cm``, it is not recommended to do
        better here.
        """
        self.mylog("CM: Initialize", 2)
        self._cm_was_computed_with_candidates = False
        self._cm_was_computed_full = False
        self._candidates_cm = np.full(self.profile_.n_c, -np.inf)
        self._candidates_cm[self.w_] = False
        self._sufficient_coalition_size_cm = np.full(self.profile_.n_c, np.inf)
        self._sufficient_coalition_size_cm[self.w_] = 0
        self._necessary_coalition_size_cm = np.zeros(self.profile_.n_c)
        self._bounds_optimized_cm = np.zeros(self.profile_.n_c)
        self._bounds_optimized_cm[self.w_] = True
        self._is_cm = -np.inf
        _ = self._cm_is_initialized_general_subclass_
        self._cm_preliminary_checks_general_()
        return True

    @cached_property
    def _cm_is_initialized_general_subclass_(self):
        return True

    def _cm_preliminary_checks_general_(self):
        """Do preliminary checks for CM. Only first time CM is launched.

        Can update some ``_candidates_cm[c]`` to True or False (instead of -inf).

        * If some ``_candidates_cm[c]`` becomes True, then ``_is_cm`` must become True as well.
        * If ``_is_cm`` becomes True, it is not necessary to update a specific ``_candidates_cm[c]``.
        * If for all candidates ``c``, ``_candidates_cm[c]`` becomes False, then ``_is_cm`` must be updated to False.
        * If ``_is_cm`` becomes False, then all ``_candidates_cm[c]`` must become False.
        * If ``_candidates_cm`` and ``_is_cm`` are totally decided to True or False, then
        ``_cm_was_computed_with_candidates`` should become True (not mandatory but recommended).
        """
        # 1) Preliminary checks that may improve _candidates_cm (all must be done, except if everything is decided).
        # Majority favorite criterion
        if self.meets_majority_favorite_c_ut and self.profile_.plurality_scores_ut[self.w_] > self.profile_.n_v / 2:
            self.mylog("CM impossible (w is a majority favorite).", 2)
            self._is_cm = False
            self._candidates_cm[:] = False
            self._cm_was_computed_with_candidates = True
            return
        if (self.meets_majority_favorite_c_ut_ctb and self.w_ == 0 and
                self.profile_.plurality_scores_ut[self.w_] >= self.profile_.n_v / 2):
            self.mylog("CM impossible (w=0 is a majority favorite with candidate tie-breaking).", 2)
            self._is_cm = False
            self._candidates_cm[:] = False
            self._cm_was_computed_with_candidates = True
            return
        if self.meets_majority_favorite_c_rk and self.profile_.plurality_scores_rk[self.w_] > self.profile_.n_v / 2:
            self.mylog("CM impossible (w is a majority favorite with voter tie-breaking).", 2)
            self._is_cm = False
            self._candidates_cm[:] = False
            self._cm_was_computed_with_candidates = True
            return
        if (self.meets_majority_favorite_c_rk_ctb and self.w_ == 0 and
                self.profile_.plurality_scores_rk[self.w_] >= self.profile_.n_v / 2):
            self.mylog("CM impossible (w=0 is a majority favorite with voter and candidate tie-breaking).", 2)
            self._is_cm = False
            self._candidates_cm[:] = False
            self._cm_was_computed_with_candidates = True
            return
        # Condorcet resistance
        if self.meets_condorcet_c_ut_abs and self.w_is_resistant_condorcet_winner_:
            self.mylog("CM impossible (w is a Resistant Condorcet winner)", 2)
            self._is_cm = False
            self._candidates_cm[:] = False
            self._cm_was_computed_with_candidates = True
            return
        # Having supporters
        self._candidates_cm[np.logical_not(self.c_has_supporters_)] = False
        if np.all(np.equal(self._candidates_cm, False)):
            self.mylog("CM: preliminary checks: CM is impossible.", 2)
            self._is_cm = False
            self._cm_was_computed_with_candidates = True
            return
        # 2) Preliminary checks that gives only global information on _is_cm (may return as soon as decision is made).
        # InfMC
        if self.meets_infmc_c and self.w_is_not_condorcet_admissible_:
            self.mylog("CM found (w is not Condorcet-admissible)", 2)
            self._is_cm = True
            return
        # ICM
        if self.precheck_icm:
            if equal_true(self.is_icm_):
                self.mylog("CM found (thanks to ICM)", 2)
                self._is_cm = True
                return
        # TM
        if self.precheck_tm:
            if equal_true(self.is_tm_):
                self.mylog("CM found (thanks to TM)", 2)
                self._is_cm = True
                return
        # UM
        if self.precheck_um:
            if equal_true(self.is_um_):
                self.mylog("CM found (thanks to UM)", 2)
                self._is_cm = True
                return
        # 3) Other checks
        self._cm_preliminary_checks_general_subclass_()

    def _cm_preliminary_checks_general_subclass_(self):
        """Do preliminary checks for CM. Only first time CM is launched.

        Can update ``_is_cm`` to True or False (instead of -inf).

        * If ``_is_cm`` becomes True, it is not necessary to update a specific ``_candidates_cm[c]``.
        * If ``_is_cm`` becomes False, then all ``_candidates_cm[c]`` must become False. And it is recommended that
        ``_cm_was_computed_with_candidates`` becomes True.

        For ``_sufficient_coalition_size_cm`` and ``_necessary_coalition_size_cm``, it is not recommended to do
        better here.
        """
        pass

    def _cm_initialize_c_(self, c, optimize_bounds):
        """Initialize the CM loop for candidate ``c`` and do preliminary checks.

        * If ``_bounds_optimized_cm[c]`` is True, it means that all the work for ``c`` has been done before. Then get
          out.
        * If ``_candidates_cm[c]`` is decided (True/False/NaN) and ``optimize_bounds`` is False, then get out.
        * Preliminary checks to improve bounds ``_sufficient_coalition_size_cm[c]`` and
          ``_necessary_coalition_size_cm[c]``.
        * If the two bounds are equal, then ``_bounds_optimized_cm[c]`` becomes True.
        * Update ``_candidates_cm[c]`` to True or False if possible.
        * If we can decide ``_is_cm`` to True, do it.

        :return: Boolean, ``job_done``. True iff we have done all the job for ``c`` (with bounds if ``optimize_bounds``
        is True, only for ``_candidates_cm[c]`` otherwise).
        """
        self.mylogv("CM: Candidate =", c, 2)
        # Check  if job is done for c
        if equal_true(self._bounds_optimized_cm[c]):
            self.mylog("CM: Job already done", 2)
            return True
        if equal_false(optimize_bounds) and not (np.isneginf(self._candidates_cm[c])):
            self.mylog("CM: Job already done", 2)
            return True
        # Improve bounds
        self._cm_preliminary_checks_c_(c, optimize_bounds)
        # Conclude what we can
        # Some log
        n_m = self.profile_.matrix_duels_ut[c, self.w_]
        self.mylogv("CM: Preliminary checks: necessary_coalition_size_cm[c] =",
                    self._necessary_coalition_size_cm[c], 3)
        self.mylogv("CM: Preliminary checks: sufficient_coalition_size_cm[c] =",
                    self._sufficient_coalition_size_cm[c], 3)
        self.mylogv("CM: Preliminary checks: n_m =", n_m, 3)
        # Conclude
        if self._sufficient_coalition_size_cm[c] == self._necessary_coalition_size_cm[c]:
            self.mylog("CM: Preliminary checks: Bounds are equal", 2)
            self._bounds_optimized_cm[c] = True
        if n_m >= self._sufficient_coalition_size_cm[c]:
            self.mylogv("CM: Preliminary checks: CM is True for c =", c, 2)
            self._candidates_cm[c] = True
            self._is_cm = True
            if equal_false(optimize_bounds) or self._bounds_optimized_cm[c]:
                return True
        elif n_m < self._necessary_coalition_size_cm[c]:
            self.mylogv("CM: Preliminary checks: CM is False for c =", c, 2)
            self._candidates_cm[c] = False
            if equal_false(optimize_bounds) or self._bounds_optimized_cm[c]:
                return True
        else:
            self.mylogv("CM: Preliminary checks: CM is unknown for c =", c, 2)
        return False

    def _cm_preliminary_checks_c_(self, c, optimize_bounds):
        """CM: preliminary checks for challenger ``c``.

        Try to improve bounds ``_sufficient_coalition_size_cm[c]`` and ``_necessary_coalition_size_cm[c]``. Do not
        update the other variables.

        If ``optimize_bounds`` is False, then return as soon as ``n_m >= _sufficient_coalition_size_cm[c]``,
        or ``_necessary_coalition_size_cm[c] > n_m`` (where ``n_m`` is the number or manipulators).
        """
        n_m = self.profile_.matrix_duels_ut[c, self.w_]  # Number of manipulators
        n_s = self.profile_.n_v - n_m  # Number of sincere voters
        # Pretest based on Informed Majority Coalition Criterion
        if self.meets_infmc_c_ctb and c == 0:
            self._update_sufficient(
                self._sufficient_coalition_size_cm, c, n_s,
                'CM: Preliminary checks: InfMC_c_ctb => \n    sufficient_coalition_size_cm[c] = n_s =')
            if not optimize_bounds and n_m >= self._sufficient_coalition_size_cm[c]:
                return
        if self.meets_infmc_c:
            self._update_sufficient(
                self._sufficient_coalition_size_cm, c, n_s + 1,
                'CM: Preliminary checks: InfMC_c => \n    sufficient_coalition_size_cm[c] = n_s + 1 =')
            if not optimize_bounds and n_m >= self._sufficient_coalition_size_cm[c]:
                return
        # Pretest based on the majority favorite criterion
        # If ``plurality_scores_ut[w] > (n_s + n_m) / 2``, then CM impossible.
        # Necessary condition: ``n_m >= 2 * plurality_scores_ut[w_] - n_s``.
        if self.meets_majority_favorite_c_rk_ctb and self.w_ == 0:
            self._update_necessary(
                self._necessary_coalition_size_cm, c, 2 * self.profile_.plurality_scores_rk[self.w_] - n_s + 1,
                'CM: Preliminary checks: majority_favorite_c_rk_ctb => \n    '
                'necessary_coalition_size_cm[c] = 2 * plurality_scores_rk[w] - n_s + 1 =')
            if not optimize_bounds and self._necessary_coalition_size_cm[c] > n_m:  # pragma: no cover
                # TO DO: Investigate whether this case can actually happen.
                self._reached_uncovered_code()
                return
        if self.meets_majority_favorite_c_rk:
            self._update_necessary(
                self._necessary_coalition_size_cm, c, 2 * self.profile_.plurality_scores_rk[self.w_] - n_s,
                'CM: Preliminary checks: majority_favorite_c_rk => \n    '
                'necessary_coalition_size_cm[c] = 2 * plurality_scores_rk[w] - n_s =')
            if not optimize_bounds and self._necessary_coalition_size_cm[c] > n_m:  # pragma: no cover
                # TO DO: Investigate whether this case can actually happen.
                self._reached_uncovered_code()
                return
        if self.meets_majority_favorite_c_ut_ctb and self.w_ == 0:
            self._update_necessary(
                self._necessary_coalition_size_cm, c, 2 * self.profile_.plurality_scores_ut[self.w_] - n_s + 1,
                'CM: Preliminary checks: majority_favorite_c_ut_ctb => \n    '
                'necessary_coalition_size_cm[c] = 2 * plurality_scores_ut[w] - n_s + 1 =')
            if not optimize_bounds and self._necessary_coalition_size_cm[c] > n_m:  # pragma: no cover
                # TO DO: Investigate whether this case can actually happen.
                self._reached_uncovered_code()
                return
        if self.meets_majority_favorite_c_ut:
            self._update_necessary(
                self._necessary_coalition_size_cm, c, 2 * self.profile_.plurality_scores_ut[self.w_] - n_s,
                'CM: Preliminary checks: majority_favorite_c_ut => \n    '
                'necessary_coalition_size_cm[c] = 2 * plurality_scores_ut[w] - n_s =')
            if not optimize_bounds and self._necessary_coalition_size_cm[c] > n_m:  # pragma: no cover
                # TO DO: Investigate whether this case can actually happen.
                self._reached_uncovered_code()
                return
        # Pretest based on the same idea as Condorcet resistance
        if self.meets_condorcet_c_ut_abs:
            self._update_necessary(
                self._necessary_coalition_size_cm, c, self.profile_.threshold_c_prevents_w_condorcet_ut_abs[c, self.w_],
                'CM: Preliminary checks: Condorcet_c => \n    '
                'necessary_coalition_size_cm[c] = threshold_c_prevents_w_Condorcet_ut_abs[c, w] =')
            if not optimize_bounds and self._necessary_coalition_size_cm[c] > n_m:
                return
        # Pretests based on ICM, TM and UM
        if self.precheck_icm:
            _, _, suf_icm_c = self.is_icm_c_with_bounds_(c)
            self._update_sufficient(
                self._sufficient_coalition_size_cm, c, suf_icm_c,
                'CM: Preliminary checks: ICM => \n    '
                'sufficient_coalition_size_cm[c] = sufficient_coalition_size_icm[c] =')
            if not optimize_bounds and n_m >= self._sufficient_coalition_size_cm[c]:  # pragma: no cover
                # TO DO: Investigate whether this case can actually happen.
                self._reached_uncovered_code()
                return
        if self.precheck_tm and self._necessary_coalition_size_cm[c] <= n_m < self._sufficient_coalition_size_cm[c]:
            if equal_true(self.is_tm_c_(c)):
                self._update_sufficient(
                    self._sufficient_coalition_size_cm, c, n_m,
                    'CM: Preliminary checks: TM => \n    '
                    'sufficient_coalition_size_cm[c] = n_m =')
                if not optimize_bounds:
                    return
        if self.precheck_um and self._necessary_coalition_size_cm[c] <= n_m < self._sufficient_coalition_size_cm[c]:
            if equal_true(self.is_tm_c_(c)):  # pragma: no cover
                # TO DO: Investigate whether this case can actually happen.
                self._reached_uncovered_code()
                self._update_sufficient(
                    self._sufficient_coalition_size_cm, c, n_m,
                    'CM: Preliminary checks: UM => \n    '
                    'sufficient_coalition_size_cm[c] = n_m =')
                if not optimize_bounds:
                    return
        # Other preliminary checks
        self._cm_preliminary_checks_c_subclass_(c, optimize_bounds)
        if not optimize_bounds and (n_m >= self._sufficient_coalition_size_cm[c]
                                    or self._necessary_coalition_size_cm[c] > n_m):
            return
        # Try to improve bounds with heuristic
        self._cm_preliminary_optimize_bound_heuristic_(c, optimize_bounds)

    def _cm_preliminary_optimize_bound_heuristic_(self, c, optimize_bounds):
        """CM: Try to improve bounds with heuristic.

        Try to improve bound ``_sufficient_coalition_size_cm[c]``.

        If ``optimize_bounds`` is False, then return as soon as ``n_m >= _sufficient_coalition_size_cm[c]``
        (where ``n_m`` is the number or manipulators).

        This method can (and maybe should) be overridden in subclasses.
        """
        if not self.is_based_on_rk:
            return
        n_m = self.profile_.matrix_duels_ut[c, self.w_]
        profile_s = Profile(
            preferences_rk=self.profile_.preferences_rk[np.logical_not(self.v_wants_to_help_c_[:, c]), :],
            preferences_borda_rk=self.profile_.preferences_borda_rk[np.logical_not(self.v_wants_to_help_c_[:, c]), :],
            sort_voters=False
        )
        ballot_rk = profile_s.candidates_by_decreasing_borda_score_rk
        ballot_rk = np.array([c] + [d for d in ballot_rk if d != c and d != self.w_] + [self.w_])
        n_m_inf = int(self._necessary_coalition_size_cm[c] - 1)
        n_m_sup = self._sufficient_coalition_size_cm[c]
        if np.isinf(n_m_sup):
            return
        else:
            n_m_sup = int(n_m_sup)
        # Loop invariant: CM is always possible with n_m_sup manipulators, not proven possible with n_m_inf manipulators
        while n_m_sup - n_m_inf > 1:
            n_m_test = (n_m_inf + n_m_sup) // 2
            profile_um = ProfileUM(profile_s=profile_s, n_m=n_m_test, ballot_rk=ballot_rk)
            w_test = self._copy(profile=profile_um).w_
            if w_test == c:
                n_m_sup = n_m_test
                if not optimize_bounds and n_m >= n_m_test:
                    break
            else:
                n_m_inf = n_m_test
        self._update_sufficient(self._sufficient_coalition_size_cm, c, n_m_sup,
                                'CM: Heuristic Dichotomy => sufficient_coalition_size_cm[c] = ')

    def _cm_preliminary_checks_c_subclass_(self, c, optimize_bounds):
        """CM: preliminary checks for challenger ``c``.

        Try to improve bounds ``_sufficient_coalition_size_cm[c]`` and ``_necessary_coalition_size_cm[c]``. Do not
        update the other variables.

        If ``optimize_bounds`` is False, then return as soon as ``n_m >= _sufficient_coalition_size_cm[c]``,
        or ``_necessary_coalition_size_cm[c] > n_m`` (where ``n_m`` is the number or manipulators).

        If a test is especially costly, it is recommended to test first if ``_sufficient_coalition_size_cm[c] ==
        _necessary_coalition_size_cm[c]`` and to return immediately in that case.
        """
        pass

    def _cm_main_work_c_(self, c, optimize_bounds):
        """Do the main work in CM loop for candidate ``c``.

        * Try to improve bounds ``_sufficient_coalition_size_cm[c]`` and ``_necessary_coalition_size_cm[c]``.
        * Do not update other variables (``_is_cm``, ``_candidates_cm``, etc.).

        If ``optimize_bounds`` is False, can return as soon as ``n_m >= _sufficient_coalition_size_cm[c]``, or
        ``_necessary_coalition_size_cm[c] > n_m`` (where ``n_m`` is the number or manipulators).

        :return: Boolean, ``is_quick_escape``. True if we did not improve the bound the best we could. (Allowed to be
            None or False otherwise).
        """
        # N.B.: in some subclasses, it is possible to try one method, then another one if the first one fails,
        # etc. In this general class, we will simply do a switch between 'lazy' and 'exact'.
        return getattr(self, '_cm_main_work_c_' + self.cm_option + '_')(c, optimize_bounds)
        # Launch a sub-method like _cm_main_work_v_lazy_, etc.

    # noinspection PyUnusedLocal,PyMethodMayBeStatic
    def _cm_main_work_c_lazy_(self, c, optimize_bounds):
        """Do the main work in CM loop for candidate ``c``, with option 'lazy'. Same specifications as
        :meth:`_cm_main_work_c_`.
        """
        # With option 'lazy', there is nothing to do! And this is not a 'quick escape': we did the best we could
        # (considering laziness).
        return False

    # noinspection PyUnusedLocal
    def _cm_main_work_c_exact_(self, c, optimize_bounds):
        """Do the main work in CM loop for candidate ``c``, with option 'exact'. Same specifications as
        ``_cm_main_work_c``.
        """
        if self.is_based_on_ut_minus1_1:  # pragma: no cover
            # As of now, all the voting rules concerned (Majority Judgement, Range Voting and Approval)
            # have other ways to compute TM, so they do not use this.
            self._reached_uncovered_code()
            # TM was already checked during preliminary checks. If TM was not True, then CM impossible.
            self._update_necessary(self._necessary_coalition_size_cm, c, self.profile_.matrix_duels_ut[c, self.w_] + 1)
            return
        if not self.is_based_on_rk:
            raise NotImplementedError
        n_m = self.profile_.matrix_duels_ut[c, self.w_]
        if n_m < self._necessary_coalition_size_cm[c]:
            # This exhaustive algorithm will not do better (so, this is not a quick escape).
            return
        if n_m >= self._sufficient_coalition_size_cm[c]:
            # Idem.
            return
        preferences_borda_temp = np.concatenate((
            np.tile(range(self.profile_.n_c), (n_m, 1)),
            self.profile_.preferences_borda_rk[np.logical_not(self.v_wants_to_help_c_[:, c]), :],
        ))
        manipulator_favorite = np.full(n_m, self.profile_.n_c - 1)
        while preferences_borda_temp is not None:
            # self.mylogm('preferences_borda_temp =', preferences_borda_temp, 3)
            w_test = self._copy(profile=Profile(preferences_ut=preferences_borda_temp, sort_voters=False)).w_
            if w_test == c:
                self._update_sufficient(self._sufficient_coalition_size_cm, c, n_m,
                                        'CM: Manipulation found by exhaustive test =>\n'
                                        '    sufficient_coalition_size_cm = n_m =')
                break
            for i_manipulator in range(n_m-1, -1, -1):
                new_ballot, new_favorite = compute_next_borda_clever(
                    preferences_borda_temp[i_manipulator, :], manipulator_favorite[i_manipulator], self.profile_.n_c)
                # self.mylogv('new_ballot = ', new_ballot)
                if new_ballot is None:
                    continue
                preferences_borda_temp[i_manipulator:n_m, :] = new_ballot[np.newaxis, :]
                manipulator_favorite[i_manipulator:n_m] = new_favorite
                break
            else:
                preferences_borda_temp = None
        else:
            self._update_necessary(self._necessary_coalition_size_cm, c, n_m + 1,
                                   'CM: Manipulation proven impossible by exhaustive test =>\n'
                                   '    necessary_coalition_size_cm[c] = n_m + 1 =')

    def _cm_conclude_c_(self, c, is_quick_escape):
        """Conclude the CM loop for candidate ``c``.

        ``_bounds_optimized_cm[c]`` --> if not ``quick_escape``, becomes True.
        ``_candidates_cm[c]`` --> True, False or NaN according to the bounds ``_sufficient_coalition_size_cm[c]`` and
            ``_necessary_coalition_size_cm[c]``.
        ``_is_cm`` -->
            * If ``_candidates_cm[c]`` is True, then ``_is_cm = True``.
            * Otherwise, do not update ``_is_cm``.
        """
        if not is_quick_escape:
            self._bounds_optimized_cm[c] = True
        n_m = self.profile_.matrix_duels_ut[c, self.w_]
        if n_m >= self._sufficient_coalition_size_cm[c]:
            self.mylogv("CM: Final answer: CM is True for c =", c, 2)
            self._candidates_cm[c] = True
            self._is_cm = True
        elif n_m < self._necessary_coalition_size_cm[c]:
            self.mylogv("CM: Final answer: CM is False for c =", c, 2)
            self._candidates_cm[c] = False
        else:
            self.mylogv("CM: Final answer: CM is unknown for c =", c, 2)
            self._candidates_cm[c] = np.nan

    def _compute_cm_(self, with_candidates, optimize_bounds):
        """Compute CM.

        Note that this method is launched by ``is_cm_`` only if CM was not initialized, and by ``candidates_cm_``
        only if not ``_cm_was_computed_with_candidates``. So, it is not necessary to do a preliminary check on these
        variables.
        """
        # We start with ``_is_cm = -Inf`` (undecided).
        # If we find a candidate for which ``_candidates_cm[c] = NaN``, then ``_is_cm`` becomes NaN too ("at least
        #   maybe").
        # If we find a candidate for which ``_candidates_cm[c] = True``, then ``_is_cm`` becomes True ("surely yes").
        for c in self.losing_candidates_:
            self._compute_cm_c_(c, optimize_bounds)
            if not with_candidates and equal_true(self._is_cm):
                return
            if np.isneginf(self._is_cm) and np.isnan(self._candidates_cm[c]):
                self._is_cm = np.nan
        # If we reach this point, we have decided all ``_candidates_cm`` to True, False or NaN.
        self._cm_was_computed_with_candidates = True  # even if ``with_candidates = False``
        self._is_cm = neginf_to_zero(self._is_cm)
        if optimize_bounds:
            self._cm_was_computed_full = True

    def _compute_cm_c_(self, c, optimize_bounds):
        job_done = self._cm_initialize_c_(c, optimize_bounds)
        if job_done:
            return
        if not optimize_bounds and not np.isneginf(self._candidates_cm[c]):  # pragma: no cover
            # TO DO: Investigate whether this case can actually happen.
            self._reached_uncovered_code()
            return
        is_quick_escape = self._cm_main_work_c_(c, optimize_bounds)
        self._cm_conclude_c_(c, is_quick_escape)

    # %% Indicators with manipulation

    @cached_property
    def elects_condorcet_winner_rk_even_with_cm_(self):
        """bool : True iff there is a Condorcet winner, she is elected by sincere voting and it is not CM."""
        if self.profile_.exists_condorcet_winner_rk and self.w_is_condorcet_winner_rk_:
            return pseudo_bool_not(self.is_cm_)
        else:
            return False

    @cached_property
    def nb_candidates_cm_(self):
        """Number of candidates who can benefit from CM."""
        inf = np.sum(self.candidates_cm_ == True)
        sup = inf + np.sum(np.isnan(self.candidates_cm_))
        return inf, sup

    @cached_property
    def worst_relative_welfare_with_cm_(self):
        """Worst relative social welfare (sincere winner or candidate who can benefit from CM)."""
        possible_winners = self.candidates_cm_.copy()
        possible_winners[self.w_] = True
        inf = np.min(
            self.profile_.relative_social_welfare_c[(possible_winners == True) | np.isnan(possible_winners)]
        )
        sup = np.min(
            self.profile_.relative_social_welfare_c[possible_winners == True]
        )
        return inf, sup

    @cached_property
    def cm_power_index_(self):
        """CM power index. For each candidate c != w, it is n_s / x_c (where n_s is the number of sincere voters
        and `x_c` the minimal number of manipulators that can make `c` win (cf :attr:`necessary_coalition_size_cm_`).
        Globally, it is the max over the candidates c != w.
        """
        n_sincere_c = np.sum(np.logical_not(self.v_wants_to_help_c_), axis=0)
        inf = np.max(
            n_sincere_c[self.sufficient_coalition_size_cm_ != 0]
            / self.sufficient_coalition_size_cm_[self.sufficient_coalition_size_cm_ != 0]
        )
        necessary_coalition_size_cm = np.maximum(self.necessary_coalition_size_cm_.copy(), 1)
        necessary_coalition_size_cm[self.w_] = 0
        try:
            sup = np.max(
                n_sincere_c[necessary_coalition_size_cm != 0]
                / necessary_coalition_size_cm[necessary_coalition_size_cm != 0]
            )
        except ValueError:
            print(n_sincere_c)
            print(self.necessary_coalition_size_cm_)
            raise ValueError
        return inf, sup

    @cached_property
    def is_tm_or_um_(self):
        if equal_true(self.is_tm_) or equal_true(self.is_um_):
            return True
        elif np.isnan(self.is_tm_) or np.isnan(self.is_um_):
            return np.nan
        else:
            return False

    # %% Demo

    def demo_manipulation_(self, log_depth=1):
        """Demonstrate the manipulation methods of :class:`Rule` class.

        Parameters
        ----------
        log_depth : int
            Integer from 0 (basic info) to 3 (verbose).
        """
        old_log_depth = self.log_depth
        self.log_depth = log_depth

        print_big_title("Election Manipulation")

        print_title("Basic properties of the voting system")
        print("with_two_candidates_reduces_to_plurality = ", self.with_two_candidates_reduces_to_plurality)
        print("is_based_on_rk = ", self.is_based_on_rk)
        print("is_based_on_ut_minus1_1 = ", self.is_based_on_ut_minus1_1)
        print("meets_iia = ", self.meets_iia)

        print_title("Manipulation properties of the voting system")

        # Condorcet_c_ut_rel_ctb (False)     ==>     Condorcet_c_ut_rel (False)
        #  ||                                                               ||
        #  ||     Condorcet_c_rk_ctb (False) ==> Condorcet_c_rk (False)     ||
        #  ||           ||               ||       ||             ||         ||
        #  V            V                ||       ||             V          V
        # Condorcet_c_ut_abs_ctb (False)     ==>     Condorcet_ut_abs_c (False)
        #  ||                            ||       ||                        ||
        #  ||                            V        V                         ||
        #  ||       maj_fav_c_rk_ctb (False) ==> maj_fav_c_rk (False)       ||
        #  ||           ||                                       ||         ||
        #  V            V                                        V          V
        # majority_favorite_c_ut_ctb (False) ==> majority_favorite_c_ut (False)
        #  ||                                                               ||
        #  V                                                                V
        # IgnMC_c_ctb (False)                ==>                IgnMC_c (False)
        #  ||                                                               ||
        #  V                                                                V
        # InfMC_c_ctb (False)                ==>                InfMC_c (False)

        def display_bool(value):
            return '(True) ' if equal_true(value) else '(False)'

        print('Condorcet_c_ut_rel_ctb ' + display_bool(self.meets_condorcet_c_ut_rel_ctb) +
              '     ==>     Condorcet_c_ut_rel ' + display_bool(self.meets_condorcet_c_ut_rel))
        print(' ||                                                               ||')
        print(' ||     Condorcet_c_rk_ctb ' + display_bool(self.meets_condorcet_c_rk_ctb) +
              ' ==> Condorcet_c_rk ' + display_bool(self.meets_condorcet_c_rk) + '     ||')
        print(' ||           ||               ||       ||             ||         ||')
        print(' V            V                ||       ||             V          V')
        print('Condorcet_c_ut_abs_ctb ' + display_bool(self.meets_condorcet_c_ut_abs_ctb) +
              '     ==>     Condorcet_ut_abs_c ' + display_bool(self.meets_condorcet_c_ut_abs))
        print(' ||                            ||       ||                        ||')
        print(' ||                            V        V                         ||')
        print(' ||       maj_fav_c_rk_ctb ' + display_bool(self.meets_majority_favorite_c_rk_ctb) +
              ' ==> maj_fav_c_rk ' + display_bool(self.meets_majority_favorite_c_rk) + '       ||')
        print(' ||           ||                                       ||         ||')
        print(' V            V                                        V          V')
        print('majority_favorite_c_ut_ctb ' + display_bool(self.meets_majority_favorite_c_ut_ctb) +
              ' ==> majority_favorite_c_ut ' + display_bool(self.meets_majority_favorite_c_ut))
        print(' ||                                                               ||')
        print(' V                                                                V')
        print('IgnMC_c_ctb ' + display_bool(self.meets_ignmc_c_ctb) +
              '                ==>                IgnMC_c ' + display_bool(self.meets_ignmc_c))
        print(' ||                                                               ||')
        print(' V                                                                V')
        print('InfMC_c_ctb ' + display_bool(self.meets_infmc_c_ctb) +
              '                ==>                InfMC_c ' + display_bool(self.meets_infmc_c))

        print_title("Independence of Irrelevant Alternatives (IIA)")
        print("w (reminder) =", self.w_)
        print("is_iia =", self.is_iia_)
        print("log_iia:", self.log_iia_)
        print("example_winner_iia =", self.example_winner_iia_)
        print("example_subset_iia =", self.example_subset_iia_)

        print_title("c-Manipulators")
        print("w (reminder) =", self.w_)
        printm("preferences_ut (reminder) =", self.profile_.preferences_ut)
        printm("v_wants_to_help_c = ", self.v_wants_to_help_c_)

        print_title("Individual Manipulation (IM)")
        print("is_im =", self.is_im_)
        print("log_im:", self.log_im_)
        printm("candidates_im =", self.candidates_im_)

        print_title("Trivial Manipulation (TM)")
        print("is_tm =", self.is_tm_)
        print("log_tm:", self.log_tm_)
        printm("candidates_tm =", self.candidates_tm_)

        print_title("Unison Manipulation (UM)")
        print("is_um =", self.is_um_)
        print("log_um:", self.log_um_)
        printm("candidates_um =", self.candidates_um_)

        print_title("Ignorant-Coalition Manipulation (ICM)")
        print("is_icm =", self.is_icm_)
        print("log_icm:", self.log_icm_)
        printm("candidates_icm =", self.candidates_icm_)
        printm("necessary_coalition_size_icm =", self.necessary_coalition_size_icm_)
        printm("sufficient_coalition_size_icm =", self.sufficient_coalition_size_icm_)

        print_title('Coalition Manipulation (CM)')
        print("is_cm =", self.is_cm_)
        print("log_cm:", self.log_cm_)
        printm("candidates_cm =", self.candidates_cm_)
        printm("necessary_coalition_size_cm =", self.necessary_coalition_size_cm_)
        printm("sufficient_coalition_size_cm =", self.sufficient_coalition_size_cm_)

        self.log_depth = old_log_depth

    def _reached_uncovered_code(self):
        """
        Print a log message when some uncovered code is reached.

        We should call this method each time a portion of code is not covered by the tests.
        """
        import inspect
        current_frame = inspect.currentframe()
        calling_frame = inspect.getouterframes(current_frame, 2)
        caller_name = calling_frame[1][3]
        print("You reached a portion of code that is not covered by the tests. If you want to \n"
              "help SVVAMP's developers, please send an email to fradurand@gmail.com and \n"
              "copy-paste the following log message.\n")
        print(self.__class__.__name__)
        print(caller_name)
        if self.profile_ is not None:
            print('n_v =', self.profile_.n_v)
            print('n_c =', self.profile_.n_c)
            print(self.profile_.to_doctest_string())
        print('result_options =', self._result_options)
        print(self.log_iia_)
        print(self.log_im_)
        print(self.log_tm_)
        print(self.log_um_)
        print(self.log_icm_)
        print(self.log_cm_)
        if OPTIONS.ERROR_WHEN_UNCOVERED_CODE:
            raise AssertionError('Uncovered portion of code.')

    def _example_reached_uncovered_code(self):
        """
        Demonstrate :meth:`_reached_uncovered_code` (cf. the unit test in `test_rule.py`).
        """
        self._reached_uncovered_code()

    def _set_random_options(self):
        """Set random options.

        For each option where a set of values is allowed, select a random value.
        """
        for option, d in self.options_parameters.items():
            if isinstance(d['allowed'], list):
                value = random.choice(d['allowed'])
                setattr(self, option, value)

    @staticmethod
    def _random_instruction():
        """Random instruction (used for testing purposes, especially identify uncovered code).

        Returns
        -------
        str
            A random instruction.
        """
        instructions = [
            'is_im_',
            'is_im_c_(0)',
            'is_im_c_(1)',
            'is_im_c_with_voters_(0)',
            'is_im_c_with_voters_(1)',
            'voters_im_',
            'candidates_im_',
            'v_im_for_c_',
            'is_im_v_(0)',
            'is_im_v_(1)',
            'is_im_v_with_candidates_(0)',
            'is_im_v_with_candidates_(1)',
            'is_tm_',
            'is_tm_c_(0)',
            'is_tm_c_(1)',
            'candidates_tm_',
            'is_um_',
            'is_um_c_(0)',
            'is_um_c_(1)',
            'candidates_um_',
            'is_icm_',
            'is_icm_c_(0)',
            'is_icm_c_(1)',
            'is_icm_c_with_bounds_(0)',
            'is_icm_c_with_bounds_(1)',
            'candidates_icm_',
            'necessary_coalition_size_icm_',
            'sufficient_coalition_size_icm_',
            'is_cm_',
            'is_cm_c_(0)',
            'is_cm_c_(1)',
            'is_cm_c_with_bounds_(0)',
            'is_cm_c_with_bounds_(1)',
            'candidates_cm_',
            'necessary_coalition_size_cm_',
            'sufficient_coalition_size_cm_',
        ]
        return random.choice(instructions)
