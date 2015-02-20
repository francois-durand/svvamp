# -*- coding: utf-8 -*-
"""
Created on Fri Sep 26 14:42:57 2014
Copyright François Durand 2014, 2015
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

from svvamp.Preferences.PopulationSubsetCandidates import \
    PopulationSubsetCandidates
from svvamp.Utils import MyLog
from svvamp.Utils import TypeChecker
from svvamp.VotingSystems.ElectionResult import ElectionResult
from svvamp.Preferences.Population import Population


class Election(ElectionResult):

    # Notes for developers
    #
    # The subclass implementing voting system Foobar is called Foobar. It is a
    # subclass of:
    # * Election,
    # * FoobarResult, which is a subclass of ElectionResult.
    #
    # In the code of this class, some special values are used.
    # * -np.inf (or None) means "I have not started to compute this value".
    # * np.nan means "I tried to compute this value, and I decided that I don't
    # know".
    # As for np.inf, it really means + Infinity.
    #
    # 1) Methods for IM have an architecture of their own.
    # 2) Methods for TM and UM have essentially the same architecture and work
    # on variables _candidates_.. and _is_..
    # 3) Methods for ICM and CM have essentially the same structure and focus
    # on _sufficient_coalition_size_.. and _necessary_coalition_size_CM..,
    # then on _candidates_.. and _is_..
    # However, there are subtle differences of architecture between 1, 2 and 3
    # (cf. their docstrings).

    # This name should be redefined for each voting system. It is used in the
    # csv file when registering simulation results.
    _layout_name = 'Election'

    # Guideline:
    # When there is a polynomial exact algorithm, it should be the only option.
    # The default option should be the most precise algorithm among those
    # running in polynomial time.
    # Exception: for IIA_subset_maximum_size, default option is +inf.
    _options_parameters = {
        'IIA_subset_maximum_size': {'allowed': TypeChecker.is_number,
                                    'default': 2},
        'IM_option': {'allowed': ['lazy', 'exact'], 'default': 'lazy'},
        'TM_option': {'allowed': ['lazy', 'exact'], 'default': 'exact'},
        'UM_option': {'allowed': ['lazy', 'exact'], 'default': 'lazy'},
        'ICM_option': {'allowed': ['lazy'], 'default': 'lazy'},
        'CM_option': {'allowed': ['lazy', 'exact'], 'default': 'lazy'}
    }

    def __init__(self, population, **kwargs):
        """Create an election with possibilities of manipulation.

        Inherits functions from superclass :class:`~svvamp.ElectionResult`.

        This is an 'abstract' class. As an end-user, you should always use its
        subclasses :attr:`~svvamp.Approval`, :attr:`~svvamp.Plurality`, etc.

        :param population: A :class:`~svvamp.Population` object.
        :param kwargs: additional keyword parameters.

        :Example:

        >>> import svvamp
        >>> pop = svvamp.PopulationSpheroid(V=100, C=5)
        >>> election = svvamp.IRV(pop, CM_option='exact')

        This class and its subclasses are suitable for voting systems that
        are deterministic and anonymous (treating all voters equally).
        Hence, they are not neutral (because they need to break ties in
        totally symmetric situations). As of now, SVVAMP does not support
        other kinds of voting systems.

        **Tie-breaking issues**

        There are essentially 2 types of tie-breaking in SVVAMP.

            *   When a sincere voter ``v`` is forced to provide a strict order
                in a specific voting system, she uses
                :attr:`svvamp.Population.preferences_ranking`\ ``[v, :]``.
                As explained in :attr:`svvamp.Population`, if she has the same
                utility for candidates ``c`` and ``d``, then she uses Voter
                Tie-Breaking (VTB): she decides once and for all if she will
                provide ``c`` before ``d`` or ``d`` before ``c`` in such
                voting systems.
            *   The voting system itself may need to break ties, for example if
                candidates ``c`` and ``d`` have the same score in a score-based
                system. The standard tie-breaking in SVVAMP, referred to as
                Candidate Tie-Breaking (CTB), consists of breaking ties by
                lowest index: ``c`` is favored over ``d`` if ``c`` < ``d``.
                This tie-breaking rule is used for example in 'A note on
                manipulability of large voting schemes' (Peleg, 1979). Future
                voting systems implemented as a subclass of ``Election`` may
                use another tie-breaking rule.

        In contrast, to know if a voter ``v`` wants to manipulate for a
        candidate ``c`` against ``w``, we do not break ``v``'s possible tie
        in her preferences. We use her utilities
        :attr:`svvamp.Population.preferences_utilities` (or equivalently,
        :attr:`svvamp.Population.preferences_borda_novtb`). If she
        attributes the same utility to ``w`` and ``c``, she is not interested
        in this manipulation.

        Some ordinal voting systems in SVVAMP may be adapted to accept weak
        orders of preferences as ballots. This is future work.

        **Options for manipulation**

        Attributes allow to choose the algorithm used to compute different
        kinds of manipulation:
        :attr:`~svvamp.Election.CM_option`,
        :attr:`~svvamp.Election.ICM_option`,
        :attr:`~svvamp.Election.IM_option`,
        :attr:`~svvamp.Election.TM_option` and
        :attr:`~svvamp.Election.UM_option`.

        To know what options are accepted for a given voting system, use
        :attr:`~svvamp.Election.options_parameters`.

        :Example:

        ::

            import svvamp
            pop = svvamp.PopulationSpheroid(V=100, C=5)
            election = svvamp.IRV(pop, CM_option='exact')
            print(election.CM())
            print(election.options_parameters)
            election.CM_option = 'fast'
            print(election.CM())

        Here is a non-exhaustive list of typical values for these options.

            *   ``'exact'``: Exact algorithm. Can always decide
                manipulation: it answers ``True`` or ``False``. Other
                algorithms may also answer ``numpy.nan``, which is the
                SVVAMP convention meaning that the algorithm was not able to
                decide. For a given voting system, if the exact algorithm
                runs in polynomial time, then it is the only accepted option.
            *   ``'slow'``: Non-polynomial algorithm, but not exact. For
                voting systems accepting this option, it is however
                faster than 'exact' (in a little-o sense) and quite more
                precise than 'fast'.
            *   ``'fast'``: Polynomial algorithm, no exact.
            *   ``'lazy'``: Perform only some preliminary checks. Run in
                polynomial time (unless deciding the winner of the election
                is not polynomial, like for :class:`~svvamp.Kemeny`). Like
                other non-exact algorithms, it can decide manipulation to
                ``True``, ``False`` or return ``numpy.nan`` (undecided).

        For a given voting system, the default option is the most precise
        algorithm running in polynomial time.

        **Option for Independence of Irrelevant Alternatives (IIA)**

        The default algorithm for :attr:`~svvamp.Election.not_IIA` is a
        brute force algorithm. It can be non-polynomial or non-exact,
        depending on the attribute
        :attr:`svvamp.Election.IIA_subset_maximum_size`.

        **Implication diagram between criteria**

        Cf. corresponding attributes below for the definition of these
        criteria. See working paper, Durand et al. (2014):
        'Condorcet Criterion and Reduction in Coalitional Manipulability'.

        ::

            Condorcet_c_rel_ctb               ==>               Condorcet_c_rel
            ||           Condorcet_c_vtb_ctb ==>      Condorcet_c_vtb       ||
            ||           ||        ||                   ||         ||       ||
            V            V         ||                   ||         V        V
            Condorcet_c_ctb                   ==>                   Condorcet_c
            ||                     ||                   ||                  ||
            ||                     V                    V                   ||
            ||   majority_favorite_c_vtb_ctb ==> majority_favorite_c_vtb    ||
            ||            ||                                  ||            ||
            V             V                                   V             V
            majority_favorite_c_ctb           ==>           majority_favorite_c
            ||                                                              ||
            V                                                               V
            IgnMC_c_ctb                       ==>                       IgnMC_c
            ||                                                              ||
            V                                                               V
            InfMC_c_ctb                       ==>                       InfMC_c
        """
        # Whether we need to do a preliminary check on UM, TM, ICM before
        # computing CM.
        self._first_initialization = True
        # self._CM_option = None

        # Initialize as superclass
        super().__init__(population, **kwargs)
        # In particular, this sets the population and calls
        # 'forget_all_computations'.

        # Constant parameters that should be redefined in each subclass
        # corresponding to a specific voting system.

        # Log identity
        self._log_identity = "ELECTION"
        self._class_result = None

        # Basic properties of the voting system
        # For the definition of these criteria, see the corresponding getter
        # methods.
        self._with_two_candidates_reduces_to_plurality = False
        self._is_based_on_strict_rankings = False
        self._is_based_on_utilities_minus1_1 = False
        self._meets_IIA = False

        self._precheck_UM = True
        self._precheck_TM = True
        self._precheck_ICM = True
        # Remark: when the voting system meets InfMC_c_ctb, then precheck on
        # ICM will not do better than other basic prechecks.

        # Manipulation criteria for the voting system
        # In the subclass corresponding to a specific voting system,
        # it is sufficient to set to True only the strongest criteria that 
        # are met by the voting system. 
        self._meets_Condorcet_c_rel = None
        self._meets_Condorcet_c_vtb = None
        self._meets_Condorcet_c = None
        self._meets_majority_favorite_c_vtb = None
        self._meets_majority_favorite_c = None
        self._meets_IgnMC_c = None
        self._meets_InfMC_c = None
        self._meets_Condorcet_c_vtb_ctb = None
        self._meets_Condorcet_c_rel_ctb = None
        self._meets_Condorcet_c_ctb = None
        self._meets_majority_favorite_c_vtb_ctb = None
        self._meets_majority_favorite_c_ctb = None
        self._meets_IgnMC_c_ctb = None
        self._meets_InfMC_c_ctb = None

        # End of parameters that need to be redefined in all subclasses

        self._first_initialization = False

    def _forget_all_computations(self):
        """Initialize / forget all computations

        Typically used when the population is modified: all results of the
        election and all manipulation computations are initialized then.
        """
        self._forget_results()
        self._forget_manipulations()

    def _forget_manipulations(self):
        """Initialize / forget all manipulation computations.
        Also, ensures that voters are sorted by their ordinal preferences.

        Typically used when the population is modified: all manipulation
        computations are initialized then.
        """
        self.pop.ensure_voters_sorted_by_ordinal_preferences()
        self._v_wants_to_help_c = None
        self._c_has_supporters = None
        self._losing_candidates = None
        self._forget_IIA()
        self._forget_IM()
        self._forget_TM()
        self._forget_UM()
        self._forget_ICM()
        self._forget_CM()
        self._forget_manipulations_subclass()
        
    def _forget_manipulations_subclass(self):
        """Initialize / forget manipulation computations.

        This concerns only the manipulation computations, not the results of
        the election.
        """
        pass

    def _forget_IIA(self):
        """Initialize / forget the computations for IIA.
        Typically used when the option is changed for these computations.
        """
        self._is_IIA = None
        self._example_winner_IIA = None
        self._example_subset_IIA = None
        
    def _forget_IM(self):
        """Initialize / forget the computations for IM.
        Typically used when the option is changed for these computations.
        """
        self._IM_was_initialized = False
        self._IM_was_computed_with_candidates = False
        self._IM_was_computed_with_voters = False
        self._IM_was_computed_full = False

    def _forget_TM(self):
        """Initialize / forget the computations for TM.
        Typically used when the option is changed for these computations.
        """
        self._TM_was_initialized = False
        self._TM_was_computed_with_candidates = False

    def _forget_UM(self):
        """Initialize / forget the computations for UM.
        Typically used when the option is changed for these computations.
        """
        self._UM_was_initialized = False
        self._UM_was_computed_with_candidates = False

    def _forget_ICM(self):
        """Initialize / forget the computations for ICM.
        Typically used when the option is changed for these computations.
        """
        self._ICM_was_initialized = False
        self._ICM_was_computed_with_candidates = False
        self._ICM_was_computed_full = False

    def _forget_CM(self):
        """Initialize / forget the computations for CM.
        Typically used when the option is changed for these computations.
        """
        self._CM_was_initialized = False
        self._CM_was_computed_with_candidates = False
        self._CM_was_computed_full = False

    #%% Setting the options
           
    @property
    def IIA_subset_maximum_size(self):
        """Integer or ``numpy.inf``. Maximum size of any subset of candidates
        that is used to compute :meth:`~svvamp.Election.not_IIA` (and
        related methods). For a given voting system, has no
        effect if there is an exact algorithm running in polynomial time
        implemented in SVVAMP.
        """
        return self._IIA_subset_maximum_size
           
    @IIA_subset_maximum_size.setter
    def IIA_subset_maximum_size(self, value):
        try:
            if self._IIA_subset_maximum_size == value:
                return
        except AttributeError:
            pass
        try:
            self._mylogv("Setting IIA_subset_maximum_size =", value, 1)
            self._IIA_subset_maximum_size = float(value)
            self._forget_IIA()
        except ValueError:
            raise ValueError("Unknown option for IIA_subset_maximum_size: "
                             + format(value) + " (number or np.inf expected).")
       
    @property
    def IM_option(self):
        """String. Option used to compute :meth:`~svvamp.Election.IM` and
        related methods.

        To know what options are accepted for a given voting system, use
        :attr:`~svvamp.ElectionResult.options_parameters`.
        """
        return self._IM_option
           
    @IM_option.setter
    def IM_option(self, value):
        try:
            if self._IM_option == value:
                return
        except AttributeError:
            pass
        if value in self.options_parameters['IM_option']['allowed']:
            self._mylogv("Setting IM_option =", value, 1)
            self._IM_option = value
            self._forget_IM()
        else:
            raise ValueError("Unknown option for IM: " + format(value))
       
    @property
    def TM_option(self):
        """String. Option used to compute :meth:`~svvamp.Election.TM` and
        related methods.

        To know what options are accepted for a given voting system, use
        :attr:`~svvamp.ElectionResult.options_parameters`.
        """
        return self._TM_option
           
    @TM_option.setter
    def TM_option(self, value):
        try:
            if self._TM_option == value:
                return
        except AttributeError:
            pass
        if value in self.options_parameters['TM_option']['allowed']:
            self._mylogv("Setting TM_option =", value, 1)
            self._TM_option = value
            self._forget_TM()
            if not self._first_initialization:
                if self._CM_option != 'exact' and self._precheck_TM:
                    self._forget_CM()
        else:
            raise ValueError("Unknown option for TM: " + format(value))
       
    @property
    def UM_option(self):
        """String. Option used to compute :meth:`~svvamp.Election.UM` and
        related methods.

        To know what options are accepted for a given voting system, use
        :attr:`~svvamp.ElectionResult.options_parameters`.
        """
        return self._UM_option
           
    @UM_option.setter
    def UM_option(self, value):
        try:
            if self._UM_option == value:
                return
        except AttributeError:
            pass
        if value in self.options_parameters['UM_option']['allowed']:
            self._mylogv("Setting UM_option =", value, 1)
            self._UM_option = value
            self._forget_UM()
            if not self._first_initialization:
                if self._CM_option != 'exact' and self._precheck_UM:
                    self._forget_CM()
        else:
            raise ValueError("Unknown option for UM: " + format(value))
       
    @property
    def ICM_option(self):
        """String. Option used to compute :meth:`~svvamp.Election.ICM` and
        related methods.

        To know what options are accepted for a given voting system, use
        :attr:`~svvamp.ElectionResult.options_parameters`.
        """
        return self._ICM_option
           
    @ICM_option.setter
    def ICM_option(self, value):
        try:
            if self._ICM_option == value:
                return
        except AttributeError:
            pass
        if value in self.options_parameters['ICM_option']['allowed']:
            self._mylogv("Setting ICM_option =", value, 1)
            self._ICM_option = value
            self._forget_ICM()
            if not self._first_initialization:
                if self._CM_option != 'exact' and self._precheck_ICM:
                    self._forget_CM()
        else:
            raise ValueError("Unknown option for ICM: " + format(value))
       
    @property
    def CM_option(self):
        """String. Option used to compute :meth:`~svvamp.Election.CM` and
        related methods.

        To know what options are accepted for a given voting system, use
        :attr:`~svvamp.ElectionResult.options_parameters`.
        """
        return self._CM_option
           
    @CM_option.setter
    def CM_option(self, value):
        try:
            if self._CM_option == value:
                return
        except AttributeError:
            pass
        if value in self.options_parameters['CM_option']['allowed']:
            self._mylogv("Setting CM_option =", value, 1)
            self._CM_option = value
            self._forget_CM()
        else:
            raise ValueError("Unknown option for CM: " + format(value))

    #%% Manipulation criteria of the voting system

    @property
    def with_two_candidates_reduces_to_plurality(self):
        """Boolean. ``'True'`` iff, when using this voting system with only
        two candidates, it amounts to Plurality (with voter and candidate
        tie-breaking).
        """
        return self._with_two_candidates_reduces_to_plurality

    @property
    def is_based_on_strict_rankings(self):
        """Boolean. ``'True'`` iff this voting system is based only on
        strict rankings (no cardinal information, indifference not allowed).
        """
        return self._is_based_on_strict_rankings

    @property
    def is_based_on_utilities_minus1_1(self):
        """Boolean. ``'True'`` iff:
        * This voting system is based only on utilities (not strict ranking,
        i.e. does not depend on the voter's tie-breaking),
        * And for a ``c``-manipulator (IM or CM), it is optimal to pretend
        that c has utility 1 and other candidates have utility -1.
        """
        return self._is_based_on_utilities_minus1_1

    @property
    def meets_IIA(self):
        """Boolean. ``'True'`` iff this voting system meets Independence of
        Irrelevant Alternatives.
        """
        return self._meets_IIA
           
    @property
    def meets_Condorcet_c_rel_ctb(self):
        """Boolean. ``'True'`` iff the voting system meets
        the 'relative Condorcet criterion with ctb'. I.e.: if a
        candidate is 'relative Condorcet winner with ctb', she wins.
        Cf. :attr:`~svvamp.Population.condorcet_winner_rel_ctb`.

        Implies:
        :attr:`~svvamp.Election.meets_Condorcet_c_rel`,
        :attr:`~svvamp.Election.meets_Condorcet_c_ctb`.
        """
        if self._meets_Condorcet_c_rel_ctb is None:
            self._meets_Condorcet_c_rel_ctb = False
        return self._meets_Condorcet_c_rel_ctb

    @property
    def meets_Condorcet_c_vtb_ctb(self):
        """Boolean. ``'True'`` iff the voting system meets
        the 'Condorcet criterion with vtb and ctb'. I.e.: if a candidate is
        'Condorcet winner with vtb and ctb', she wins.
        Cf. :attr:`~svvamp.Population.condorcet_winner_vtb_ctb`.

        Implies:
        :attr:`~svvamp.Election.meets_Condorcet_c_vtb`,
        :attr:`~svvamp.Election.meets_Condorcet_c_ctb`,
        :attr:`~svvamp.Election.meets_majority_favorite_c_vtb_ctb`.
        """
        if self._meets_Condorcet_c_vtb_ctb is None:
            self._meets_Condorcet_c_vtb_ctb = False
        return self._meets_Condorcet_c_vtb_ctb

    @property
    def meets_Condorcet_c_ctb(self):
        """Boolean. ``'True'`` iff the voting system meets
        the 'Condorcet criterion with ctb'. I.e.: if a candidate is
        'Condorcet winner with ctb', she wins.
        Cf. :attr:`~svvamp.Population.condorcet_winner_ctb`.

        Is implied by:
        :attr:`~svvamp.Election.meets_Condorcet_c_vtb_ctb`,
        :attr:`~svvamp.Election.meets_Condorcet_c_rel_ctb`.

        Implies:
        :attr:`~svvamp.Election.meets_Condorcet_c`,
        :attr:`~svvamp.Election.meets_majority_favorite_c_ctb`.
        """
        if self._meets_Condorcet_c_ctb is None:
            self._meets_Condorcet_c_ctb = (
                self.meets_Condorcet_c_vtb_ctb or
                self.meets_Condorcet_c_rel_ctb)
        return self._meets_Condorcet_c_ctb
           
    @property
    def meets_majority_favorite_c_vtb_ctb(self):
        """Boolean. ``'True'`` iff the voting system meets
        the 'majority favorite criterion with vtb and ctb'. I.e.:

            *   It :attr:`~svvamp.Election.meets_majority_favorite_c_vtb`,
            *   And if :attr:`~svvamp.Population.V`/2 voters rank candidate 0
                first (with vtb), she wins.

        Is implied by:
        :attr:`~svvamp.Election.meets_Condorcet_c_vtb_ctb`.

        Implies:
        :attr:`~svvamp.Election.meets_majority_favorite_c_ctb`,
        :attr:`~svvamp.Election.meets_majority_favorite_c_vtb`.
        """
        if self._meets_majority_favorite_c_vtb_ctb is None:
            self._meets_majority_favorite_c_vtb_ctb = \
                self.meets_Condorcet_c_vtb_ctb
        return self._meets_majority_favorite_c_vtb_ctb

    @property
    def meets_majority_favorite_c_ctb(self):
        """Boolean. ``'True'`` iff the voting system meets
        the 'majority favorite criterion with ctb'. I.e.:

            *   It :attr:`~svvamp.Election.meets_majority_favorite_c`,
            *   And if :attr:`~svvamp.Population.V`/2 voters strictly prefer
                candidate 0 to all other candidates, she wins.

        Is implied by:
        :attr:`~svvamp.Election.meets_Condorcet_c_ctb`,
        :attr:`~svvamp.Election.meets_majority_favorite_c_vtb_ctb`.

        Implies:
        :attr:`~svvamp.Election.meets_IgnMC_c_ctb`,
        :attr:`~svvamp.Election.meets_majority_favorite_c`.
        """
        if self._meets_majority_favorite_c_ctb is None:
            self._meets_majority_favorite_c_ctb = (
                self.meets_Condorcet_c_ctb or
                self.meets_majority_favorite_c_vtb_ctb)
        return self._meets_majority_favorite_c_ctb

    @property
    def meets_IgnMC_c_ctb(self):
        """Boolean. ``'True'`` iff the voting system meets
        the 'ignorant majority coalition criterion with ctb'. I.e.:

            *   It :attr:`~svvamp.Election.meets_IgnMC_c`,
            *   And any ignorant coalition of size
                :attr:`~svvamp.Population.V`/2 can make candidate 0 win.

        Is implied by:
        :attr:`~svvamp.Election.meets_majority_favorite_c_ctb`.

        Implies:
        :attr:`~svvamp.Election.meets_InfMC_c_ctb`,
        :attr:`~svvamp.Election.meets_IgnMC_c`.
        """
        if self._meets_IgnMC_c_ctb is None:
            self._meets_IgnMC_c_ctb = self.meets_majority_favorite_c_ctb
        return self._meets_IgnMC_c_ctb

    @property
    def meets_InfMC_c_ctb(self):
        """Boolean. ``'True'`` iff the voting system meets
        the 'informed majority coalition criterion with ctb'. I.e.:

            *   It :attr:`~svvamp.Election.meets_InfMC_c`,
            *   And any informed coalition of size
                :attr:`~svvamp.Population.V`/2 can make candidate 0 win.
        
        Is implied by:
        :attr:`~svvamp.Election.meets_IgnMC_c_ctb`.

        Implies:
        :attr:`~svvamp.Election.meets_InfMC_c`.
        """
        if self._meets_InfMC_c_ctb is None:
            self._meets_InfMC_c_ctb = self.meets_IgnMC_c_ctb
        return self._meets_InfMC_c_ctb
        
    @property
    def meets_Condorcet_c_rel(self):
        """Boolean. ``'True'`` iff the voting system meets
        the relative Condorcet criterion. I.e. if a candidate is a
        relative Condorcet winner, then she wins.
        Cf. :attr:`~svvamp.Population.condorcet_winner_rel`.

        Is implied by:
        :attr:`~svvamp.Election.meets_Condorcet_c_rel_ctb`.

        Implies:
        :attr:`~svvamp.Election.meets_Condorcet_c`.
        """
        if self._meets_Condorcet_c_rel is None:
            self._meets_Condorcet_c_rel = self.meets_Condorcet_c_rel_ctb
        return self._meets_Condorcet_c_rel

    @property
    def meets_Condorcet_c_vtb(self):
        """Boolean. ``'True'`` iff the voting system meets
        the Condorcet criterion with vtb. I.e. if a candidate is a Condorcet
        winner with vtb, then she wins.
        Cf. :attr:`~svvamp.Population.condorcet_winner_vtb`.

        Is implied by:
        :attr:`~svvamp.Election.meets_Condorcet_c_vtb_ctb`.

        Implies:
        :attr:`~svvamp.Election.meets_Condorcet_c`,
        :attr:`~svvamp.Election.meets_majority_favorite_c_vtb`.
        """
        if self._meets_Condorcet_c_vtb is None:
            self._meets_Condorcet_c_vtb = self.meets_Condorcet_c_vtb_ctb
        return self._meets_Condorcet_c_vtb

    @property
    def meets_Condorcet_c(self):
        """Boolean. ``'True'`` iff the voting system meets
        the Condorcet criterion. I.e. if a candidate is a Condorcet winner,
        then she wins.
        Cf. :attr:`~svvamp.Population.condorcet_winner`.

        Is implied by:
        :attr:`~svvamp.Election.meets_Condorcet_c_vtb`,
        :attr:`~svvamp.Election.meets_Condorcet_c_rel`,
        :attr:`~svvamp.Election.meets_Condorcet_c_ctb`.

        Implies:
        :attr:`~svvamp.Election.meets_majority_favorite_c`.
        """
        if self._meets_Condorcet_c is None:
            self._meets_Condorcet_c = (
                self.meets_Condorcet_c_vtb or
                self.meets_Condorcet_c_rel or
                self.meets_Condorcet_c_ctb)
        return self._meets_Condorcet_c

    @property
    def meets_majority_favorite_c_vtb(self):
        """Boolean. ``'True'`` iff the voting system meets
        the majority favorite criterion with vtb. I.e. if strictly more than
        :attr:`~svvamp.Population.V`/2 voters rank a candidate first (with
        vtb), she wins.

        Is implied by:
        :attr:`~svvamp.Election.meets_Condorcet_c_vtb`,
        :attr:`~svvamp.Election.meets_majority_favorite_c_vtb_ctb`.

        Implies:
        :attr:`~svvamp.Election._meets_majority_favorite_c`.
        """
        if self._meets_majority_favorite_c_vtb is None:
            self._meets_majority_favorite_c_vtb = (
                self.meets_Condorcet_c_vtb or
                self.meets_majority_favorite_c_vtb_ctb)
        return self._meets_majority_favorite_c_vtb

    @property
    def meets_majority_favorite_c(self):
        """Boolean. ``'True'`` iff the voting system meets
        the majority favorite criterion. I.e. if strictly more than
        :attr:`~svvamp.Population.V`/2 voters strictly prefer a candidate to
        all others (without vtb), she wins.

        Is implied by:
        :attr:`~svvamp.Election.meets_Condorcet_c`,
        :attr:`~svvamp.Election.meets_majority_favorite_c_ctb`,
        :attr:`~svvamp.Election.meets_majority_favorite_c_vtb`.

        Implies:
        :attr:`~svvamp.Election.meets_IgnMC_c`.
        """
        if self._meets_majority_favorite_c is None:
            self._meets_majority_favorite_c = (
                self.meets_Condorcet_c or
                self.meets_majority_favorite_c_vtb or
                self.meets_majority_favorite_c_ctb)
        return self._meets_majority_favorite_c

    @property
    def meets_IgnMC_c(self):
        """Boolean. ``'True'`` iff the voting system meets
        the ignorant majority coalition criterion. I.e. any ignorant coalition 
        of size strictly more than :attr:`~svvamp.Population.V`/2 can make
        any candidate win. See working paper, Durand et al. (2014):
        'Condorcet Criterion and Reduction in Coalitional Manipulability'.

        *Ignorant* means that they can choose their ballot without knowing
        what other voters will do.
        
        Is implied by:
        :attr:`~svvamp.Election.meets_majority_favorite_c`,
        :attr:`~svvamp.Election.meets_IgnMC_c_ctb`.

        Implies:
        :attr:`~svvamp.Election.meets_InfMC_c`.
        """
        if self._meets_IgnMC_c is None:
            self._meets_IgnMC_c = (
                self.meets_majority_favorite_c or
                self.meets_IgnMC_c_ctb)
        return self._meets_IgnMC_c

    @property
    def meets_InfMC_c(self):
        """Boolean. ``'True'`` iff the voting system meets
        the informed majority coalition criterion. I.e. any informed coalition 
        of size strictly more than :attr:`~svvamp.Population.V`/2 can make
        any candidate win. See working paper, Durand et al. (2014):
        'Condorcet Criterion and Reduction in Coalitional Manipulability'.
        
        *Informed* means that they know other voters' ballots before
        choosing their own.

        Is implied by:
        :attr:`~svvamp.Election.meets_IgnMC_c`,
        :attr:`~svvamp.Election.meets_InfMC_c_ctb`.
        """
        if self._meets_InfMC_c is None:
            self._meets_InfMC_c = (
                self.meets_IgnMC_c or
                self.meets_InfMC_c_ctb)
        return self._meets_InfMC_c

    #%% Result of an election (for tests like TM, IM...)

    def _create_result(self, pop_test):
        """Create a FoobarResult object for a population, where Foobar is
        the voting system.

        Arguments:
        pop_test -- A Population object.

        Returns:
        election_result -- A FoobarResult object.
        """
        return self._class_result(pop_test)

    def _basic_version(self, **kwargs):

        class_result = self._class_result
        is_based_on_strict_rankings = self.is_based_on_strict_rankings
        is_based_on_utilities_minus1_1 = self.is_based_on_utilities_minus1_1

        class _BasicVersion(class_result, Election):

            _options_parameters = Election._options_parameters.copy()
            _options_parameters.update(class_result._options_parameters)

            def __init__(self, population, **kwargs):
                super().__init__(population, **kwargs)
                self._log_identity = "BASIC"
                self._is_based_on_strict_rankings = (
                    is_based_on_strict_rankings)
                self._is_based_on_utilities_minus1_1 = (
                    is_based_on_utilities_minus1_1)

            def _create_result(self, pop_test):
                return class_result(pop_test)

        return _BasicVersion(self.pop, **kwargs)

    #%% Independence of Irrelevant Alternatives (IIA)
        
    @property
    def log_IIA(self):
        """String. Parameters used to compute :meth:`~svvamp.not_IIA` and
        related methods.
        """
        return "IIA_subset_maximum_size = " + format(
            self.IIA_subset_maximum_size)

    def not_IIA(self):
        """Independence of Irrelevant Alternatives, incomplete mode.

        :return: (``is_not_IIA``, ``log_IIA``).

        Cf. :meth:`~svvamp.Election.not_IIA_complete` for more details.
        """
        if self._is_IIA is None:
            self._compute_IIA()
        if np.isnan(self._is_IIA):
            return np.nan, self.log_IIA
        else:
            return (not self._is_IIA), self.log_IIA

    def not_IIA_complete(self):
        """Independence of Irrelevant Alternatives, complete mode.

        :return: (``is_not_IIA``, ``log_IIA``, ``example_subset_IIA``,
                 ``example_winner_IIA``).

        ``is_not_IIA``: Boolean. ``True`` if there exists a subset of
        candidates including the sincere winner
        :attr:`~svvamp.ElectionResult.w`, such that if the election is held
        with this subset of candidates, then
        :attr:`~svvamp.ElectionResult.w` is not the winner anymore.
        If the algorithm cannot decide, then the result is ``numpy.nan``.

        ``log_IIA``: String. Parameters used to compute IIA.

        ``example_subset_IIA``: 1d array of booleans. If the election is
        not IIA, ``example_subset_IIA`` provides a subset of candidates
        breaking IIA. ``example_subset_IIA[c]`` is ``True`` iff candidate
        ``c`` belongs to the subset. If the election is IIA (or if the
        algorithm cannot decide), then ``example_subset_IIA = numpy.nan``.

        ``example_winner_IIA``: Integer (candidate). If the election is
        not IIA, ``example_winner_IIA`` is the winner corresponding to the
        counter-example ``example_subset_IIA``. If the election is IIA (or
        if the algorithm cannot decide), then
        ``example_winner_IIA = numpy.nan``.

        .. seealso::

            :meth:`~svvamp.Election.not_IIA`.
        """
        if self._is_IIA is None:
            self._compute_IIA()
        if np.isnan(self._is_IIA):
            return np.nan, self.log_IIA, \
                   self._example_subset_IIA, self._example_winner_IIA
        else:
            return (not self._is_IIA), self.log_IIA, \
                   self._example_subset_IIA, self._example_winner_IIA

    def _IIA_impossible(self, message):
        """Actions when IIA is impossible.
        Displays a message and sets the relevant variables.

        Arguments:
        message -- String. A log message.
        """
        self._mylog(message, 1)
        self._is_IIA = True
        self._example_subset_IIA = np.nan
        self._example_winner_IIA = np.nan

    def _compute_IIA(self):
        """Compute IIA: _is_IIA, _example_subset_IIA and _example_winner_IIA.
        """
        self._mylog("Compute IIA", 1)
        if self.meets_IIA:
            self._IIA_impossible("IIA is guaranteed for this voting system.")
            return
        if self.meets_Condorcet_c and self.w_is_condorcet_winner:
            self._IIA_impossible("IIA guaranteed: w is a Condorcet winner.")
            return
        if self.meets_Condorcet_c_ctb and self.w_is_condorcet_winner_ctb:
            self._IIA_impossible("IIA guaranteed: w is a Condorcet winner "
                                 "with candidate tie-breaking.")
            return
        if self.meets_Condorcet_c_rel and self.w_is_condorcet_winner_rel:
            self._IIA_impossible("IIA guaranteed: w is a relative Condorcet "
                                 "winner.")
            return
        if (self.meets_Condorcet_c_rel_ctb and
                self.w_is_condorcet_winner_rel_ctb):
            self._IIA_impossible("IIA guaranteed: w is a relative Condorcet "
                                 "winner with candidate tie-breaking.")
            return
        if self.meets_Condorcet_c_vtb and self.w_is_condorcet_winner_vtb:
            self._IIA_impossible("IIA guaranteed: w is a Condorcet winner "
                                 "with voter tie-breaking.")
            return
        if (self.meets_Condorcet_c_vtb_ctb and
                self.w_is_condorcet_winner_vtb_ctb):
            self._IIA_impossible("IIA guaranteed: w is a Condorcet winner "
                                 "with voter and candidate tie-breaking.")
            return
        if (self.meets_majority_favorite_c and
                self.pop.plurality_scores_novtb[self.w] > self.pop.V / 2):
            self._IIA_impossible("IIA guaranteed: w is a majority favorite.")
            return
        if (self.meets_majority_favorite_c_vtb and
                self.pop.plurality_scores_vtb[self.w] > self.pop.V / 2):
            self._IIA_impossible("IIA guaranteed: w is a majority favorite "
                                 "with voter tie-breaking.")
            return
        if (self.meets_majority_favorite_c_ctb and
                self.w == 0 and
                self.pop.plurality_scores_novtb[self.w] >= self.pop.V / 2):
            self._IIA_impossible("IIA guaranteed: w is a majority favorite "
                                 "with candidate tie-breaking (w = 0).")
            return
        if (self.meets_majority_favorite_c_vtb_ctb and
                self.w == 0 and
                self.pop.plurality_scores_vtb[self.w] >= self.pop.V / 2):
            self._IIA_impossible("IIA guaranteed: w is a majority favorite "
                                 "with voter and candidate tie-breaking "
                                 "(w = 0).")
            return
        if self._with_two_candidates_reduces_to_plurality:
            if self.w_is_not_condorcet_winner_vtb_ctb:
                # For subsets of 2 candidates, we use the matrix of victories 
                # to gain time.
                self._mylog("IIA failure found by Condorcet failure (with "
                            "vtb and ctb)", 2)
                self._is_IIA = False
                self._example_winner_IIA = np.nonzero(
                    self.pop.matrix_victories_vtb_ctb[:, self.w])[0][0]
                self._example_subset_IIA = np.zeros(self.pop.C, dtype=bool)
                self._example_subset_IIA[self.w] = True
                self._example_subset_IIA[self._example_winner_IIA] = True
            else:
                self._mylog("IIA: subsets of size 2 are ok because w is a "
                            "Condorcet winner (with vtb and ctb)", 2)
                self._compute_IIA_aux(subset_minimum_size=3)
        else:
            self._compute_IIA_aux(subset_minimum_size=2)

    def _compute_IIA_aux(self, subset_minimum_size):
        """Compute IIA: is_IIA, example_subset_IIA and example_winner_IIA.
        
        Arguments:
        subset_minimum_size -- Integer.
        
        Tests all subsets from size 'subset_minimum_size' to 
        'self.IIA_subset_maximum_size'. If self.IIA_subset_maximum_size < C-1,
        then the algorithm may not be able to decide whether election is IIA 
        or not: in this case, we may have is_IIA = NaN.
        """
        self._mylogv("IIA: Use _compute_IIA_aux with subset_minimum_size =", 
                     subset_minimum_size, 1)
        subset_maximum_size = int(min(
            self.pop.C - 1, self.IIA_subset_maximum_size))
        for C_r in range(subset_minimum_size, subset_maximum_size + 1):
            if self.w <= C_r - 1:
                candidates_r = np.array(range(C_r))
            else:
                candidates_r = np.concatenate((range(C_r - 1), [self.w]))
            while candidates_r is not None:
                w_r = self._compute_winner_of_subset(candidates_r)
                if w_r != self.w:
                    self._mylog("IIA failure found", 2)
                    self._is_IIA = False
                    self._example_winner_IIA = w_r
                    self._example_subset_IIA = np.zeros(self.pop.C, dtype=bool)
                    for c in candidates_r:
                        self._example_subset_IIA[c] = True
                    return
                candidates_r = compute_next_subset_with_w(
                    candidates_r, self.pop.C, C_r, self.w)
        # We have not found a counter-example...
        self._example_winner_IIA = np.nan
        self._example_subset_IIA = np.nan
        if self.IIA_subset_maximum_size < self.pop.C - 1:
            self._mylog("IIA: I have found no counter-example, but I have " +
                        "not explored all possibilities", 2)
            self._is_IIA = np.nan
        else:
            self._mylog("IIA is guaranteed.", 2)
            self._is_IIA = True
            
    def _compute_winner_of_subset(self, candidates_r):
        """Compute the winner for a subset of candidates.
        
        This function is internally used to compute Independence of Irrelevant 
        Alternatives (IIA).
        
        Arguments:
        candidates_r -- 1d array of integers. candidates_r(k) is the k-th
            candidate of the subset. This vector must be sorted in ascending
            order.
            
        Returns:
        w_r -- Integer. Candidate who wins the sub-election defined by 
            candidates_r.
        """
        self._mylogv("IIA: Compute winner of subset ", candidates_r, 3)
        pop_test = PopulationSubsetCandidates(self.pop, candidates_r)
        result_test = self._create_result(pop_test)
        w_r = candidates_r[result_test.w]
        return w_r

    #%% Manipulation: common features

    @property
    def v_wants_to_help_c(self):
        """2d array of booleans. ``v_wants_to_help_c[v, c]`` is ``True`` iff
        voter ``v`` strictly prefers candidate ``c`` to the sincere winner
        :attr:`~svvamp.ElectionResult.w`. If ``v`` attributes
        the same utility for ``c`` and ``w``, then ``v`` is not interested.
        """
        if self._v_wants_to_help_c is None:
            self._mylog("Compute v_wants_to_help_c", 1)
            self._v_wants_to_help_c = np.greater(
                self.pop.preferences_utilities,
                self.pop.preferences_utilities[:, self.w][:, np.newaxis]
            )
        return self._v_wants_to_help_c
        
    @property
    def losing_candidates(self):
        """1d of Integers. List of losing candidates, in an arbitrary order.

        This attribute is mostly for SVVAMP developers. It is used in
        manipulation algorithms.
        """
        # In fact, the order is not really arbitrary... Losing candidates are
        # sorted from the 'most dangerous' to the 'least dangerous' (for the
        # sincere winner :attr:`~svvamp.ElectionResult.w`).
        #
        # By default, they are sorted by their score against
        # :attr:`~svvamp.ElectionResult.w` in the
        # :attr:`~svvamp.Population.matrix_duels` (which is the number of
        # potential manipulators for a given candidate). This behavior can be
        # redefined in the subclass implementing a specific voting system.
        #
        # This attribute is used for most manipulation algorithms. The idea is
        # to try first the candidates for whom we think manipulation is more
        # likely to succeed, in order to gain time.
        if self._losing_candidates is None:
            self._mylog("Compute ordered list of losing candidates", 1)
            self._losing_candidates = np.concatenate((
                np.array(range(0, self.w), dtype=int),
                np.array(range(self.w+1, self.pop.C), dtype=int)
            ))
            self._losing_candidates = self._losing_candidates[np.argsort(
                -self.pop.matrix_duels[self._losing_candidates, self.w],
                kind='mergesort')]
        return self._losing_candidates
        
    @property
    def c_has_supporters(self):
        """1d array of booleans. ``c_has_supporters[c]`` is ``True`` iff at
        least one voter prefers candidate c to the sincere winner
        :attr:`~svvamp.ElectionResult.w`.
        """
        if self._c_has_supporters is None:
            self._mylog("Compute c_has_supporters", 1)
            self._c_has_supporters = np.any(self.v_wants_to_help_c, 0)
        return self._c_has_supporters

    def _update_sufficient(self, sufficient_array, c, value, message=None):
        """Update an array _sufficient_coalition_size_.. for candidate c.

        Arguments:
        sufficient_array -- An array like _sufficient_coalition_size_CM or
            _sufficient_coalition_size_ICM.
        c -- Integer (candidate).
        value -- Integer. If the number of manipulators is >= value, then
            manipulation (CM or ICM) is possible.
        message -- String. A message that can be displayed if
            sufficient_array[c] is actually updated.

        Perform sufficient_array[c] = min(sufficient_array[c], value).
        If sufficient_array[c] is actually, updated, i.e. iff value is
        strictly lower that the former value of sufficient_array[c], then:
        send message and value to self._mylogv (with detail level = 3).
        """
        if value < sufficient_array[c]:
            sufficient_array[c] = value
            if message is not None:
                self._mylogv(message, value, 3)

    def _update_necessary(self, necessary_array, c, value, message=None):
        """Update an array _necessary_coalition_size_.. for candidate c.

        Arguments:
        necessary_array -- An array like _necessary_coalition_size_CM or
            _necessary_coalition_size_ICM.
        c -- Integer (candidate).
        value -- Integer. If the number of manipulators is < value, then
            manipulation (CM or ICM) is impossible.
        message -- String. A message that can be displayed if
            necessary_array[c] is actually updated.

        Perform necessary_array[c] = max(necessary_array[c], value).
        If necessary_array[c] is actually, updated, i.e. iff value is
        strictly greater that the former value of necessary_array[c], then
        send message and value to self._mylogv (with detail level = 3).
        """
        if value > necessary_array[c]:
            necessary_array[c] = value
            if message is not None:
                self._mylogv(message, value, 3)

    #%% Individual manipulation (IM)

    @property
    def log_IM(self):
        """String. Parameters used to compute :meth:`~svvamp.Election.IM`
        and related methods.
        """
        # noinspection PyTypeChecker
        return "IM_option = " + self.IM_option
    
    def IM(self):
        """Individual manipulation.

        :returns: (``is_IM``, ``log_IM``).

        Cf. :meth:`~svvamp.Election.IM_full`.
        """
        if not self._IM_was_initialized:
            self._IM_initialize_general()
        if np.isneginf(self._is_IM):
            self._compute_IM(mode='IM')
        return display_pseudo_bool(self._is_IM), self.log_IM
            
    def IM_c(self, c):
        """Individual manipulation, focus on one candidate.

        :param c: Integer (candidate).

        :returns: (``candidates_IM[c]``, ``log_IM``).

        Cf. :meth:`~svvamp.Election.IM_full`.
        """
        if not self._IM_was_initialized:
            self._IM_initialize_general()
        if np.isneginf(self._candidates_IM[c]):
            self._compute_IM(mode='IM_c', c=c)
        return display_pseudo_bool(self._candidates_IM[c]), self.log_IM

    def IM_c_with_voters(self, c):
        """Individual manipulation, focus on one candidate, with details.

        :param c: Integer (candidate).

        :returns: (``candidates_IM[c]``, ``log_IM``, ``v_IM_for_c[:, c]``).

        Cf. :meth:`~svvamp.Election.IM_full`.
        """
        if not self._IM_was_initialized:
            self._IM_initialize_general()
        if np.any(np.isneginf(self._v_IM_for_c[:, c])):
            self._compute_IM(mode='IM_c_with_voters', c=c)
        return display_pseudo_bool(self._candidates_IM[c]), \
               self.log_IM, \
               display_pseudo_bool(self._v_IM_for_c[:, c])

    def IM_with_voters(self):
        """Individual manipulation, focus on voters.

        :returns: (``is_IM``, ``log_IM``, ``voters_IM``).

        Cf. :meth:`~svvamp.Election.IM_full`.
        """
        if not self._IM_was_initialized:
            self._IM_initialize_general()
        if not self._IM_was_computed_with_voters:
            self._compute_IM(mode='IM_with_voters')
        return display_pseudo_bool(self._is_IM), self.log_IM, \
               self._voters_IM.astype(np.float)

    def IM_with_candidates(self):
        """Individual manipulation, focus on candidates.

        :returns: (``is_IM``, ``log_IM``, ``candidates_IM``).

        Cf. :meth:`~svvamp.Election.IM_full`.
        """
        if not self._IM_was_initialized:
            self._IM_initialize_general()
        if not self._IM_was_computed_with_candidates:
            self._compute_IM(mode='IM_with_candidates')
        return display_pseudo_bool(self._is_IM), self.log_IM, \
               self._candidates_IM.astype(np.float)

    def IM_full(self):
        """Individual manipulation, full mode.

        Voter ``v`` can and wants to manipulate for candidate ``c`` iff:

            *   ``v`` strictly prefers ``c`` to
                :attr:`~svvamp.ElectionResult.w` (in the sense of
                :attr:`~svvamp.Population.preferences_utilities`).
            *   And by changing her vote, she can make ``c`` win instead of
                :attr:`~svvamp.ElectionResult.w`.

        :returns: (``is_IM``, ``log_IM``, ``candidates_IM``, ``voters_IM``,
                  ``v_IM_for_c``).

        ``is_IM``: Boolean. ``True`` if there exists a voter who can
        and wants to manipulate, ``False`` otherwise. If the algorithm cannot
        decide, then ``numpy.nan``.

        ``log_IM``: String. Parameters used to compute IM.

        ``candidates_IM``: 1d array of booleans (or ``numpy.nan``).
        ``candidates_IM[c]`` is ``True`` if there exists a voter who can
        manipulate for candidate ``c``, ``False`` otherwise. If the
        algorithm cannot decide, then ``numpy.nan``. For the sincere winner
        :attr:`~svvamp.ElectionResult.w`, we have by convention
        ``candidates_IM[w] = False``.

        ``voters_IM``: 1d array of booleans (or ``numpy.nan``).
        ``voters_IM[v]`` is ``True`` if voter ``v`` can and wants to
        manipulate for at least one candidate, ``False`` otherwise. If the
        algorithm cannot decide, then ``numpy.nan``.

        ``v_IM_for_c``: 2d array of booleans. ``v_IM_for_c[v, c]`` is ``True``
        if voter ``v`` can manipulate for candidate ``c``, ``False`` otherwise.
        If the algorithm cannot decide, then ``numpy.nan``. For the sincere
        winner :attr:`~svvamp.ElectionResult.w`, we have by convention
        ``v_IM_for_c[v, w] = False``.

        .. seealso::

            :meth:`~svvamp.Election.IM`,
            :meth:`~svvamp.Election.IM_c`,
            :meth:`~svvamp.Election.IM_c_with_voters`,
            :meth:`~svvamp.Election.IM_v`,
            :meth:`~svvamp.Election.IM_v_with_candidates`,
            :meth:`~svvamp.Election.IM_with_candidates`,
            :meth:`~svvamp.Election.IM_with_voters`.
        """
        if not self._IM_was_initialized:
            self._IM_initialize_general()
        if not self._IM_was_computed_full:
            self._compute_IM(mode='IM_full')
        return display_pseudo_bool(self._is_IM), self.log_IM, \
               self._candidates_IM.astype(np.float), \
               self._voters_IM.astype(np.float), \
               self._v_IM_for_c.astype(np.float)

    def IM_v(self, v):
        """Individual manipulation, focus on one voter.

        :param v: Integer (voter).

        :returns: (``voters_IM[v]``, ``log_IM``).

        Cf. :meth:`~svvamp.Election.IM_full`.
        """
        if not self._IM_was_initialized:
            self._IM_initialize_general()
        if np.isneginf(self._voters_IM[v]):
            self._compute_IM_v(v, c_is_wanted=np.ones(self.pop.C,
                                                      dtype=np.bool),
                               stop_if_true=True)
        return display_pseudo_bool(self._voters_IM[v]), self.log_IM

    def IM_v_with_candidates(self, v):
        """Individual manipulation, focus on one voter, with details.

        :param v: Integer (voter).

        :returns: (``voters_IM[v]``, ``log_IM``, ``v_IM_for_c[v, :]``).

        Cf. :meth:`~svvamp.Election.IM_full`.
        """
        if not self._IM_was_initialized:
            self._IM_initialize_general()
        if np.any(np.isneginf(self._v_IM_for_c[v, :])):
            self._compute_IM_v(v, c_is_wanted=np.ones(self.pop.C,
                                                      dtype=np.bool),
                               stop_if_true=False)
        return display_pseudo_bool(self._voters_IM[v]), self.log_IM, \
               self._v_IM_for_c[v, :].astype(np.float)

    def _IM_initialize_general(self):
        """Initialize IM variables and do preliminary checks. Used only the
        first time IM is launched (whatever the mode).
        _IM_was_initialized --> True
        _is_IM --> False or True if we know, -inf otherwise.
        _candidates_IM[c] --> True of False if we know, -inf otherwise.
        _voters_IM[v] --> True of False if we know, -inf otherwise.
        _v_IM_for_c[v, c] --> True or False if we know, -inf otherwise.

        It is mandatory that _v_IM_for_c[v, c] is False if voter c does
        not prefer c to the sincere winner w. Other kinds of checks are
        optional if this method is redefined in subclasses.

        If _candidates_IM and _is_IM are totally decided
        to True or False, then _IM_was_computed_with_candidates should become
        True (not mandatory but recommended).
        """
        self._mylog("IM: Initialize", 2)
        self._IM_was_initialized = True
        self._v_IM_for_c = np.full((self.pop.V, self.pop.C), -np.inf)
        self._candidates_IM = np.full(self.pop.C, -np.inf)
        self._voters_IM = np.full(self.pop.V, -np.inf)
        self._is_IM = -np.inf
        self._IM_preliminary_checks_general()

    def _IM_preliminary_checks_general(self):
        """Do preliminary checks for IM. Only first time IM is launched.

        Can update some _v_IM_for_c[v, c] to True or False (instead of
        -inf). In particular, it is mandatory that it is updated to False if
        voter c does not prefer c to the sincere winner w.

        * If _v_IM_for_c[v, c] becomes True, then _candidates_IM[c] and
        _voters_IM[v] must become True. If _candidates_IM[c] or
        _voters_IM[v] becomes True, then _is_IM must become True.
        * If _is_IM becomes True, it is not necessary to update a specific
        _candidates_IM[c] or _voters_IM[v]. If _candidates_IM[c] or
        _voters_IM[v] becomes True, it is not necessary to update a
        specific _v_IM_for_c[v, c].
        * If _is_IM becomes False, then _candidates_IM[:] and _voters_IM[:]
        must become False. If _candidates_IM[c] becomes False, then
        _v_IM_for_c[:, c] must become False. If _voters_IM[v] becomes False,
        then _v_IM_for_c[v, :] must become False.
        * If for a candidate c and all voters v, all _v_IM_for_c[v, c]
        become False, then _candidates_IM[c] must be updated to False. If for
        all candidates c, _candidates_IM[c] becomes False, then _is_IM must
        be updated to False.
        * Similarly, if for a voter v and all candidates c, all
        _v_IM_for_c[v, c] become False, then _voters_IM[v] must become False.
        If for all voters v, _voters_IM[v] becomes False, then _is_IM must
        be updated to False.
        * If _v_IM_for_c, _candidates_IM and _is_IM are totally decided
        to True or False, then _IM_was_computed_with_candidates,
        _IM_was_computed_with_voters and _IM_was_computed_full should become
        True (not mandatory but recommended).
        """
        # Perform some preliminary checks
        self._v_IM_for_c[np.logical_not(self.v_wants_to_help_c)] = False
        self._v_IM_for_c[np.logical_not(self._v_might_IM_for_c)] = False
        self._IM_preliminary_checks_general_subclass()
        # Update 'False' answers for _candidates_IM, _voters_IM and _is_IM
        self._candidates_IM[
            np.all(np.equal(self._v_IM_for_c, False), 0)] = False
        self._voters_IM[
            np.all(np.equal(self._v_IM_for_c, False), 1)] = False
        if np.all(np.equal(self._candidates_IM, False)):
            self._mylog("IM: preliminary checks: IM is impossible.", 2)
            self._is_IM = False
            self._IM_was_computed_with_candidates = True
            self._IM_was_computed_with_voters = True
            self._IM_was_computed_full = True
        self._mylogm('_v_IM_for_c =', self._v_IM_for_c, 3)

    def _IM_preliminary_checks_general_subclass(self):
        """Do preliminary checks for IM. Only first time IM is launched.

        Can update some _v_IM_for_c[v, c] to True or False (instead of
        -inf).

        True must be propagated from specific to general, False must be
        propagated from general to specific. I.e.:
        * If _v_IM_for_c[v, c] becomes True, then _candidates_IM[c] and
        _voters_IM[v] must become True. If _candidates_IM[c] or
        _voters_IM[v] becomes True, then _is_IM must become True.
        * If _is_IM becomes False, then _candidates_IM[:] and _voters_IM[v]
        must become False. If _candidates_IM[c] or _voters_IM[v] becomes
        False, then _v_IM_for_c[:, c] must become False.

        * If for a candidate c and all voters v, all _v_IM_for_c[v, c] become
        False, it is not necessary to update _candidates_IM[c] to False (and it
        is not necessary to update _is_IM).
        """
        pass

    def _IM_initialize_v(self, v):
        """Initialize the IM loop for voter v and do preliminary checks.
        Launched every time we work on voter v.

        * If the voting system is ordinal and voter v has the same
        ordinal preferences as previous voter v - 1, then update the line
        _v_IM_for_c[v, :] with what we know for v - 1.
        * Preliminary checks: try to decide some _v_IM_for_c[v, c]. If
        _v_IM_for_c[v, c] becomes True, then _candidates_IM[c], _voters_IM[v]
        and _is_IM must become True as well. In the other cases, it is not
        necessary to update _candidates_IM[c] and _is_IM.
        """
        self._mylogv("IM: Voter =", v, 3)
        # Check if v is identical to previous voter
        if (self._is_based_on_strict_rankings and
                self.pop.v_has_same_ordinal_preferences_as_previous_voter[v]):
            self._mylog("IM: Identical to previous voter", 3)
            decided_previous_v = np.logical_not(np.isneginf(
                self._v_IM_for_c[v - 1, :]))
            self._v_IM_for_c[v, decided_previous_v] = self._v_IM_for_c[
                v - 1, decided_previous_v]
            if self._voters_IM[v - 1] == True:
                self._voters_IM[v] = True
        # Preliminary checks on v
        self._IM_preliminary_checks_v(v)

    def _IM_preliminary_checks_v(self, v):
        """IM: preliminary checks for voter v.
        
        Try to decide some _v_IM_for_c[v, c]. If _v_IM_for_c[v, c] becomes
        True, then _candidates_IM[c], _voters_IM[v] and _is_IM must become True
        as well. In the other cases, it is not necessary to update
        _candidates_IM[c], _voters_IM[c] and _is_IM.
        """
        # Nothing smart for the moment.
        self._IM_preliminary_checks_v_subclass(v)
        pass

    def _IM_preliminary_checks_v_subclass(self, v):
        """IM: preliminary checks for voter v.

        Try to decide some _v_IM_for_c[v, c]. If _v_IM_for_c[v, c] becomes
        True, then _candidates_IM[c], _voters_IM[v] and _is_IM must become True
        as well. In the other cases, it is not necessary to update
        _candidates_IM[c], _voters_IM[c] and _is_IM.
        """
        pass

    def _IM_main_work_v(self, v, c_is_wanted,
                        nb_wanted_undecided, stop_if_true):
        """Do the main work in IM loop for voter v.

        Arguments:
        v -- Integer (voter).
        c_is_wanted -- 1d array of booleans. If for all c such that
            c_is_wanted[c] is True, _v_IM_for_c[v, c] is decided, then we
            are authorized to get out.
        nb_wanted_undecided -- Integer. Number of 'wanted' candidates c such
            that _v_IM_for_c[v, c] is not decided yet.
        stop_if_true -- Boolean. If True, then as soon as a True is found
            for a 'wanted' candidate, we are authorized to get out.

        Try to decide _v_IM_for_c[v, :]. At the end, _v_IM_for_c[v, c]
        can be True, False, NaN or -inf (we may not have decided for all
        candidates).

        If _v_IM_for_c[v, c] becomes True, then _candidates_IM[c],
        _voters_IM[v] and _is_IM must become True.
        In the other cases, it is not necessary to update _candidates_IM[c],
        _voters_IM[v] and _is_IM (even if _v_IM_for_c[v, c] becomes NaN).

        Each time a wanted candidate is decided (to True, False or NaN),
        decrement nb_wanted_undecided. When it reaches 0, we may get out.
        If a wanted candidate is decided to True and stop_if_true, then we
        may get out.
        """
        # N.B.: in some subclasses, it is possible to try one method,
        # then another one if the first one fails, etc. In this general class,
        # we will simply do a switch between 'lazy' and 'exact'.
        # noinspection PyTypeChecker
        getattr(self, '_IM_main_work_v_' + self.IM_option)(
            v, c_is_wanted, nb_wanted_undecided, stop_if_true
        )  # Launch a sub-method like _IM_main_work_v_lazy, etc.

    def _IM_main_work_v_lazy(self, v, c_is_wanted,
                             nb_wanted_undecided, stop_if_true):
        """Do the main work in IM loop for voter v, with option 'lazy'.

        Same specifications as _IM_main_work_v.
        """
        # When we don't know, we decide that we don't know!
        neginf_to_nan(self._v_IM_for_c[v, :])

    def _IM_main_work_v_exact(self, v, c_is_wanted,
                              nb_wanted_undecided, stop_if_true):
        """Do the main work in IM loop for voter v, with option 'exact'.

        Same specifications as _IM_main_work_v.
        """
        if self.is_based_on_strict_rankings:
            self._IM_main_work_v_exact_rankings(
                v, c_is_wanted, nb_wanted_undecided, stop_if_true)
        elif self.is_based_on_utilities_minus1_1:
            self._IM_main_work_v_exact_utilities_minus1_1(
                v, c_is_wanted, nb_wanted_undecided, stop_if_true)
        else:
            raise NotImplementedError("IM: Exact manipulation is not "
                                      "implemented for this voting system.")

    def _IM_main_work_v_exact_rankings(self, v, c_is_wanted,
                                       nb_wanted_undecided, stop_if_true):
        """Do the main work in IM loop for voter v, with option 'exact',
        for a voting system based only on strict rankings.

        Same specifications as _IM_main_work_v.
        """
        preferences_borda_test = np.copy(self.pop.preferences_borda_vtb)
        ballot = np.array(range(self.pop.C))
        ballot_favorite = self.pop.C - 1
        while ballot is not None:  # Loop on possible ballots
            self._mylogv("IM: Ballot =", ballot, 3)
            preferences_borda_test[v, :] = ballot
            pop_test = Population(preferences_utilities=preferences_borda_test)
            result_test = self._create_result(pop_test)
            w_test = result_test.w
            if np.isneginf(self._v_IM_for_c[v, w_test]):
                # Implicitly, it also means that v prefers c to w (cf.
                # specifications of _IM_initialize_general).
                self._v_IM_for_c[v, w_test] = True
                self._candidates_IM[w_test] = True
                self._voters_IM[v] = True
                self._is_IM = True
                self._mylogv("IM found for c =", w_test, 3)
                if c_is_wanted[w_test]:
                    if stop_if_true:
                        return
                    nb_wanted_undecided -= 1
                if nb_wanted_undecided == 0:
                    return  # We know everything we want for this voter
            ballot, ballot_favorite = compute_next_borda_clever(
                ballot, ballot_favorite, self.pop.C)
        # If we reach this point, we have tried all ballots, so if we have
        # not found a manipulation for c, it is not possible. Next
        # instruction replaces all -Inf with 0.
        neginf_to_zero(self._v_IM_for_c[v, :])

    def _IM_main_work_v_exact_utilities_minus1_1(self, v, c_is_wanted,
                                                 nb_wanted_undecided,
                                                 stop_if_true):
        """Do the main work in IM loop for voter v, with option 'exact',
        for a voting system based only on utilities and where it is optimal
        for a c-manipulator to pretend that c has utility 1 and other
        candidates utility 0.

        Same specifications as _IM_main_work_v.
        """
        preferences_utilities_test = np.copy(self.pop.preferences_utilities)
        for c in range(self.pop.C):
            if not c_is_wanted[c]:
                continue
            if not np.isneginf(self._v_IM_for_c[v, c]):
                continue
            # Implicitly, it also means that v prefers c to w (cf.
            # specifications of _IM_initialize_general).
            preferences_utilities_test[v, :] = -1
            preferences_utilities_test[v, c] = 1
            pop_test = Population(
                preferences_utilities=preferences_utilities_test)
            result_test = self._create_result(pop_test)
            w_test = result_test.w
            if w_test == c:
                self._v_IM_for_c[v, c] = True
                self._candidates_IM[c] = True
                self._voters_IM[v] = True
                self._is_IM = True
                self._mylog("IM found", 3)
                if stop_if_true:
                    return
            else:
                self._v_IM_for_c[v, c] = False
            nb_wanted_undecided -= 1
            if nb_wanted_undecided == 0:
                return

    def _compute_IM(self, mode, c=None):
        """Compute IM.

        Arguments:
        mode -- String. Name of the method calling _compute_IM.
        c -- Integer or None. If integer, then we only want to study IM for
            this candidate.
        """
        self._mylog("Compute IM", 1)
        for v in range(self.pop.V):
            # Prepare work
            if mode == 'IM':
                c_is_wanted = np.ones(self.pop.C, dtype=np.bool)
                stop_if_true = True
            elif mode in {'IM_c', 'IM_c_with_voters'}:
                c_is_wanted = np.zeros(self.pop.C, dtype=np.bool)
                c_is_wanted[c] = True
                stop_if_true = True
            elif mode == 'IM_with_voters':
                c_is_wanted = np.ones(self.pop.C, dtype=np.bool)
                stop_if_true = True
            elif mode == 'IM_with_candidates':
                c_is_wanted = np.isneginf(self._candidates_IM)
                stop_if_true = False
            else:  # mode == 'IM_full'
                c_is_wanted = np.ones(self.pop.C, dtype=np.bool)
                stop_if_true = False
            # Work
            self._compute_IM_v(v, c_is_wanted, stop_if_true)
            # Conclude for v
            if mode == 'IM':
                if not np.isneginf(self._is_IM):
                    return
            elif mode == 'IM_c':
                if not np.isneginf(self._candidates_IM[c]):
                    return
            elif mode == 'IM_with_candidates':
                if not np.any(np.isneginf(self._candidates_IM)):
                    self._IM_was_computed_with_candidates = True
                    return
        # Conclude: update _candidates_IM and _is_IM if possible
        self._candidates_IM[np.all(np.equal(self._v_IM_for_c, False),
                                   0)] = False
        for c in self.losing_candidates:
            if np.isneginf(self._candidates_IM[c]):
                if np.all(np.logical_not(np.isneginf(self._v_IM_for_c[:, c]))):
                    self._candidates_IM[c] = np.nan
        if np.isneginf(self._is_IM):
            if np.all(np.equal(self._candidates_IM, False)):
                self._is_IM = False
            elif np.all(np.logical_not(np.isneginf(self._candidates_IM))):
                self._is_IM = np.nan
        if not np.any(np.isneginf(self._v_IM_for_c)):
            self._IM_was_computed_full = True
            self._IM_was_computed_with_voters = True
            self._IM_was_computed_with_candidates = True
        else:
            if not np.any(np.isneginf(self._voters_IM)):
                self._IM_was_computed_with_voters = True
            if not np.any(np.isneginf(self._candidates_IM)):
                self._IM_was_computed_with_candidates = True

    def _compute_IM_v(self, v, c_is_wanted, stop_if_true):
        """Compute IM for voter v.

        Arguments:
        v -- Integer (voter).
        c_is_wanted -- 1d array of booleans. If for all c such that
            c_is_wanted[c] is True, _v_IM_for_c[v, c] is decided, then we
            are authorized to get out.
        stop_if_true -- Boolean. If True, then as soon as a True is found
            for a 'wanted' candidate, we are authorized to get out.

        Try to decide _v_IM_for_c[v, :]. At the end, _v_IM_for_c[v, c]
        can be True, False, NaN or -inf (we may not have decided for all
        candidates).
        At the end, _voters_IM[v] must be coherent with what we know about
        _v_IM_for_c[v, :] (True, False, NaN or -inf).

        If _v_IM_for_c[v, c] becomes True, then _candidates_IM[c] and
        _is_IM must become True.
        In the other cases, it is not necessary to update _candidates_IM[c],
        and _is_IM (even if _v_IM_for_c[v, c] becomes NaN).

        """
        self._IM_initialize_v(v)
        nb_wanted_undecided = np.sum(np.isneginf(
            self._v_IM_for_c[v, c_is_wanted]))
        if nb_wanted_undecided == 0:
            self._mylog("IM: Job already done", 3)
        else:
            self._mylogv("IM: Preliminary checks: Still some work for v =",
                         v, 3)
            self._IM_main_work_v(v, c_is_wanted, nb_wanted_undecided,
                                 stop_if_true)
        if np.isneginf(self._voters_IM[v]):
            if np.all(np.equal(self._v_IM_for_c[v, :], False)):
                self._voters_IM[v] = False
            elif np.all(np.logical_not(np.isneginf(self._v_IM_for_c[v, :]))):
                self._voters_IM[v] = np.nan

    #%% Trivial Manipulation (TM)

    @property
    def log_TM(self):
        """String. Parameters used to compute :meth:`~svvamp.Election.TM`
        and related methods.
        """
        # noinspection PyTypeChecker
        return "TM_option = " + self.TM_option
    
    def TM(self):
        """Trivial manipulation.

        :returns: (``is_TM``, ``log_TM``).

        Cf. :meth:`~svvamp.Election.TM_with_candidates`.
        """
        if not self._TM_was_initialized:
            self._TM_initialize_general(with_candidates=False)
        if np.isneginf(self._is_TM):
            self._compute_TM(with_candidates=False)
        return display_pseudo_bool(self._is_TM), self.log_TM
            
    def TM_c(self, c):
        """Trivial manipulation, focus on one candidate.

        :param c: Integer (candidate).

        :returns: (``candidates_TM[c]``, ``log_TM``).

        Cf. :meth:`~svvamp.Election.TM_with_candidates`.
        """
        if not self._TM_was_initialized:
            self._TM_initialize_general(with_candidates=False)
        if np.isneginf(self._candidates_TM[c]):
            self._compute_TM_c(c)
        return display_pseudo_bool(self._candidates_TM[c]), self.log_TM

    def TM_with_candidates(self):
        """Trivial manipulation, full mode.

        For ordinal voting systems, we call *trivial manipulation* for
        candidate ``c`` against :attr:`~svvamp.ElectionResult.w` the fact of
        putting ``c`` on top (compromising), :attr:`~svvamp.ElectionResult.w`
        at bottom (burying), while keeping a sincere order on other candidates.
        
        For cardinal voting systems, we call *trivial manipulation* for ``c``
        (against :attr:`~svvamp.ElectionResult.w`) the fact of putting the
        maximum grade for ``c`` and the minimum grade for other candidates.

        In both cases, the intuitive idea is the following: if I want to 
        make ``c`` win and I only know that candidate
        :attr:`~svvamp.ElectionResult.w` is 'dangerous' (but I know
        nothing else), then trivial manipulation is my 'best' strategy.
        
        We say that a situation is *trivially manipulable* for ``c``
        (implicitly: by coalition) iff, when all voters preferring ``c`` to
        the sincere winner :attr:`~svvamp.ElectionResult.w` use trivial
        manipulation, candidate ``c`` wins.
        
        :returns: (``is_TM``, ``log_TM``, ``candidates_TM``).

        ``is_TM``: Boolean (or ``numpy.nan``). ``True`` if TM is possible,
        ``False`` otherwise. If the algorithm cannot decide,
        then ``numpy.nan`` (but as of now, this value is never used for TM).

        ``log_TM``: String. Parameters used to compute TM.

        ``candidates_TM``: 1d array of booleans (or ``numpy.nan``).
        ``candidates_TM[c]`` is ``True`` if a TM for candidate ``c`` is
        possible, ``False`` otherwise. If the algorithm cannot decide, then
        ``numpy.nan`` (but as of now, this value is not never for TM). For the
        sincere winner :attr:`~svvamp.ElectionResult.w`, we have by convention
        ``candidates_TM[w] = False``.

        .. seealso::

            :meth:`~svvamp.Election.TM`,
            :meth:`~svvamp.Election.TM_c`.
        """
        if not self._TM_was_initialized:
            self._TM_initialize_general(with_candidates=True)
        if not self._TM_was_computed_with_candidates:
            self._compute_TM(with_candidates=True)
        return display_pseudo_bool(self._is_TM), self.log_TM, \
               self._candidates_TM.astype(np.float)

    def _TM_initialize_general(self, with_candidates):
        """Initialize TM variables and do preliminary checks. Used only the
        first time TM is launched (whatever the mode).
        _TM_was_initialized --> True
        _is_TM --> True or False if we know, -inf otherwise.
        _candidates_TM[c] --> True or False if we know, -inf otherwise.

        If _candidates_TM and _is_TM are totally decided to True, False or
        NaN, then _TM_was_computed_with_candidates should become True (not
        mandatory but recommended).
        """
        self._mylog("TM: Initialize", 2)
        self._TM_was_initialized = True
        self._candidates_TM = np.full(self.pop.C, -np.inf)
        self._is_TM = -np.inf
        self._TM_preliminary_checks_general()
        
    def _TM_preliminary_checks_general(self):
        """Do preliminary checks for TM. Only first time TM is launched.

        Can update some _candidates_TM[c] to True or False (instead of -inf).
        * If some _candidates_TM[c] becomes True, then _is_TM must become True
        as well.
        * If _is_TM becomes True, it is not necessary to update a specific
        _candidates_TM[c].
        * If for all candidates c, _candidates_TM[c] become False,
        then _is_TM must be updated to False.
        * If _is_TM becomes False, then all _candidates_TM[c] must become
        False.
        * If _candidates_TM and _is_TM are totally decided to True or False,
        then _TM_was_computed_with_candidates should become True (not mandatory but
        recommended).

        N.B.: Be careful, if a pretest deciding TM to True is added,
        then some modifications may be needed for Exhaustive ballot.
        """
        # 1) Preliminary checks that may improve _candidates_TM (all must be
        # done, except if everything is decided).
        # Majority favorite criterion
        if (self.meets_majority_favorite_c and
                self.pop.plurality_scores_novtb[self.w] > self.pop.V / 2):
            self._mylog("TM impossible (w is a majority favorite).", 2)
            self._is_TM = False
            self._candidates_TM[:] = False
            self._TM_was_computed_with_candidates = True
            return
        if (self.meets_majority_favorite_c_ctb and
                self.w == 0 and
                self.pop.plurality_scores_novtb[self.w] >= self.pop.V / 2):
            self._mylog("TM impossible (w=0 is a majority favorite with " +
                        "candidate tie-breaking).", 2)
            self._is_TM = False
            self._candidates_TM[:] = False
            self._TM_was_computed_with_candidates = True
            return
        if (self.meets_majority_favorite_c_vtb and
                self.pop.plurality_scores_vtb[self.w] > self.pop.V / 2):
            self._mylog("TM impossible (w is a majority favorite with "
                        "voter tie-breaking).", 2)
            self._is_TM = False
            self._candidates_TM[:] = False
            self._TM_was_computed_with_candidates = True
            return
        if (self.meets_majority_favorite_c_vtb_ctb and
                self.w == 0 and
                self.pop.plurality_scores_vtb[self.w] >= self.pop.V / 2):
            self._mylog("TM impossible (w=0 is a majority favorite with " +
                        "voter and candidate tie-breaking).", 2)
            self._is_TM = False
            self._candidates_TM[:] = False
            self._TM_was_computed_with_candidates = True
            return
        # Having supporters
        self._candidates_TM[np.logical_not(self.c_has_supporters)] = False
        # 2) Additional preliminary checks from the subclass.
        self._TM_preliminary_checks_general_subclass()
        if not np.any(self.c_has_supporters):
            self._mylog("TM impossible (all voters like w best)", 2)
            self._is_TM = False
            self._candidates_TM[:] = False
            self._TM_was_computed_with_candidates = True
            return
        if not np.isneginf(self._is_TM):
            return
        # 3) Preliminary checks that gives only global information on _is_TM
        # (may return as soon as decision is made).
        # Nothing

    def _TM_preliminary_checks_general_subclass(self):
        """Do preliminary checks for TM. Only first time TM is launched.

        Can update some _candidates_TM[c] to True or False (instead of -inf).

        True must be propagated from specific to general, False must be 
        propagated from general to specific.
        * If some _candidates_TM[c] becomes True, then _is_TM must become True
        as well.
        * If _is_TM becomes True, it is not necessary to update a specific
        _candidates_TM[c].
        * If _is_TM becomes False, then all _candidates_TM[c] must become
        False.
        * If for all candidates c, _candidates_TM[c] becomes False,
        it is not necessary to update _is_TM.
        * If _candidates_TM and _is_TM are totally decided to True or False,
        then _TM_was_computed_with_candidates should become True (not mandatory but
        recommended).

        Put first the checks that may improve _candidates_TM (all must be
        done, except if everything is decided).
        Then the checks that gives only global information on _is_TM (which may
        return as soon as decision is made).
        """
        pass

    def _TM_initialize_c(self, c):
        """Initialize the TM loop for candidate c and may do preliminary 
        checks.

        * If _candidates_TM[c] is decided (True/False/NaN), it means that 
            all the work for c has been done before. Then get out.
        * Preliminary checks: try to decide _candidates_TM[c]. If it becomes
            True, then _is_TM must become True as well. In other cases, do not
            update _is_TM.
        """
        self._mylogv("TM: Candidate =", c, 2)
        # Check if job is done for c
        if not np.isneginf(self._candidates_TM[c]):
            self._mylog("TM: Job already done", 2)
            return
        # Preliminary checks
        self._TM_preliminary_checks_c(c)
        # Conclude what we can
        if self._candidates_TM[c] == True:
            self._mylogv("TM: Preliminary checks: TM is True for c =", c, 2)
            self._is_TM = True
        elif self._candidates_TM[c] == False:
            self._mylogv("TM: Preliminary checks: TM is False for c =", c, 2)
        else:
            self._mylogv("TM: Preliminary checks: TM is unknown for c =", c, 3)

    def _TM_preliminary_checks_c(self, c):
        """TM: preliminary checks for challenger c.

        Try to decide _candidates_TM[c] to True or False (instead of -inf). Do
        not update _is_TM.
        """
        # We not do run any preliminary test for the moment, since computing TM
        # is generally very easy (by design).
        self._TM_preliminary_checks_c_subclass(c)

    def _TM_preliminary_checks_c_subclass(self, c):
        """TM: preliminary checks for challenger c.

        Try to decide _candidates_TM[c] to True or False (instead of -inf). Do
        not update _is_TM.
        """
        pass

    def _TM_main_work_c(self, c):
        """ Do the main work in TM loop for candidate c.
        Must decide _candidates_TM[c] (to True, False or NaN).
        Do not update _is_TM.
        """
        # N.B.: in some subclasses, it is possible to try one method,
        # then another one if the first one fails, etc. In this general class,
        # we will simply do a switch between 'lazy' and 'exact'.
        # noinspection PyTypeChecker
        getattr(self, '_TM_main_work_c_' + self.TM_option)(
            c
        )  # Launch a sub-method like _TM_main_work_c_lazy, etc.

    def _TM_main_work_c_lazy(self, c):
        """Do the main work in TM loop for candidate c, with option 'lazy'.
        Must decide _candidates_TM[c] (to True, False or NaN).
        Do not update _is_TM.
        """
        self._candidates_TM[c] = neginf_to_nan(
            self._candidates_TM[c])

    def _TM_main_work_c_exact(self, c):
        """Do the main work in TM loop for candidate c, with option 'exact'.
        Must decide _candidates_TM[c] (to True, False or NaN).
        Do not update _is_TM.
        """
        if self.is_based_on_strict_rankings:
            self._TM_main_work_c_exact_rankings(c)
        elif self.is_based_on_utilities_minus1_1:
            self._TM_main_work_c_exact_utilities_minus1_1(c)
        else:
            raise NotImplementedError("TM: Exact manipulation is not "
                                      "implemented for this voting system.")

    def _TM_main_work_c_exact_rankings(self, c):
        """Do the main work in TM loop for candidate c, with option 'exact',
        for a voting system based only on strict rankings.
        Must decide _candidates_TM[c] (to True, False or NaN).
        Do not update _is_TM.
        """
        # Manipulators put c on top and w at bottom.
        preferences_borda_test = self._compute_trivial_strategy_ordinal(c)
        pop_test = Population(preferences_utilities=preferences_borda_test)
        result_test = self._create_result(pop_test)
        w_test = result_test.w
        self._mylogv("TM: w_test =", w_test)
        self._candidates_TM[c] = (w_test == c)

    def _TM_main_work_c_exact_utilities_minus1_1(self, c):
        """Do the main work in TM loop for candidate c, with option 'exact',
        for a voting system based only on utilities and where it is optimal
        for a c-manipulator to pretend that c has utility 1 and other
        candidates utility 0.
        Must decide _candidates_TM[c] (to True, False or NaN).
        Do not update _is_TM.
        """
        # Manipulators give -1 to all candidates, except 1 for c.
        preferences_test = np.copy(self.pop.preferences_utilities)
        preferences_test[self.v_wants_to_help_c[:, c], :] = -1
        preferences_test[self.v_wants_to_help_c[:, c], c] = 1
        pop_test = Population(preferences_utilities=preferences_test)
        result_test = self._create_result(pop_test)
        w_test = result_test.w
        self._candidates_TM[c] = (w_test == c)

    def _TM_conclude_c(self, c):
        """Conclude the TM loop for candidate c, according to the value of 
        _candidates_TM[c].
        _is_TM -->
            * If _candidates_TM[c] is True, then _is_TM = True.
            * Otherwise, do not update _is_TM.
        """
        if self._candidates_TM[c] == True:
            self._mylogv("TM: Final answer: TM is True for c =", c, 2)
            self._is_TM = True
        elif self._candidates_TM[c] == False:
            self._mylogv("TM: Final answer: TM is False for c =", c, 2)
        else:
            self._mylogv("TM: Final answer: TM is unknown for c =", c, 2)

    def _compute_TM(self, with_candidates):
        """Compute TM: is_TM.
        
        Note that this method is launched by TM only if _is_TM is not decided,
        and by TM_with_candidates only if not _TM_was_computed_with_candidates. So,
        it is not necessary to do a preliminary check on these variables.

        If with_candidates is False:
        * At the end, _is_TM must be decided to True, False or NaN.
        * _candidates_TM must be at least initialized (to an array of -inf).
        It can be partially decided to True, False or NaN (to avoid some
        computations if we come back later), but it is not mandatory.
        * Coherence is not mandatory: notably, if _is_TM is decided to True,
        it is not necessary to update a specific _candidates_TM[c].
        * _TM_was_initialized must become True.
        * If _is_TM and _candidates_TM are totally decided to True, False or
        NaN, then _TM_was_computed_with_candidates should become True (not
        mandatory but recommended).

        If with_candidates is True:
        * _is_TM and _candidates_TM must be decided to True, False or NaN.
        * _TM_was_initialized and _TM_was_computed_with_candidates must become True.
        """
        # We start with _is_TM = -Inf (undecided).
        # If we find a candidate for which _candidates_TM[c] = NaN, then
        #   _is_TM becomes NaN too ("at least maybe").
        # If we find a candidate for which _candidates_TM[c] = True, then
        #   _is_TM becomes True ("surely yes").
        for c in self.losing_candidates:
            self._compute_TM_c(c)
            if not with_candidates and self._is_TM == True:
                return
            if np.isneginf(self._is_TM) and np.isnan(self._candidates_TM[c]):
                self._is_TM = np.nan
        # If we reach this point, we have decided all _candidates_TM to True,
        # False or NaN.
        self._TM_was_computed_with_candidates = True  # even if with_candidates = False
        self._is_TM = neginf_to_zero(self._is_TM)

    def _compute_TM_c(self, c):
        """Compute TM for candidate c.

        Note that this method checks if _candidates_TM[c] is already decided.
        So, it is not necessary to do this check before calling the method.

        During this method:
        * _candidates_TM[c] must be decided to True, False or NaN.
        * If it becomes True, then _is_TM must become True as well. Otherwise,
        do not update _is_TM.
        """
        self._TM_initialize_c(c)
        if np.isfinite(self._candidates_TM[c]):
            return
        self._TM_main_work_c(c)
        self._TM_conclude_c(c)

    def _compute_trivial_strategy_ordinal(self, c):
        """Compute trivial strategy for an voting system based on strict
        rankings.
        
        Arguments:
        c -- Integer. The candidate for whom we want to manipulate.

        Returns:
        preferences_test -- 2d array of integers. New Borda scores of the 
            population. For each voter preferring c to w, she now puts c on 
            top, w at bottom, and other Borda scores are modified accordingly.
        """
        preferences_test = np.copy(self.pop.preferences_borda_vtb)
        self._mylogm("Borda scores (sincere) =",
                     preferences_test, 3)
        # For manipulators: all candidates that were above c lose 1 point.
        preferences_test[np.logical_and(
            self.v_wants_to_help_c[:, c][:, np.newaxis],
            self.pop.preferences_borda_vtb >
            self.pop.preferences_borda_vtb[:, c][:, np.newaxis]
        )] -= 1
        # For manipulators: all candidates that were below w gain 1 point.
        preferences_test[np.logical_and(
            self.v_wants_to_help_c[:, c][:, np.newaxis],
            self.pop.preferences_borda_vtb <
            self.pop.preferences_borda_vtb[:, self.w][:, np.newaxis]
        )] += 1
        # For manipulators: c gets score C and w gets score 1.
        preferences_test[self.v_wants_to_help_c[:, c], c] = self.pop.C - 1
        preferences_test[self.v_wants_to_help_c[:, c], self.w] = 0
        self._mylogm("Borda scores (with trivial strategy) =",
                     preferences_test, 3)
        return preferences_test

    #%% Unison Manipulation (UM)

    @property
    def log_UM(self):
        """String. Parameters used to compute :meth:`~svvamp.Election.UM`
        and related methods.
        """
        # noinspection PyTypeChecker
        return "UM_option = " + self.UM_option
    
    def UM(self):
        """Unison manipulation.

        :returns: (``is_UM``, ``log_UM``).

        Cf. :meth:`~svvamp.Election.UM_with_candidates`.
        """
        if not self._UM_was_initialized:
            self._UM_initialize_general(with_candidates=False)
        if np.isneginf(self._is_UM):
            self._compute_UM(with_candidates=False)
        return display_pseudo_bool(self._is_UM), self.log_UM
            
    def UM_c(self, c):
        """Unison manipulation, focus on one candidate.

        :param c: Integer (candidate).

        :returns: (``candidates_UM[c]``, ``log_UM``).

        Cf. :meth:`~svvamp.Election.UM_with_candidates`.
        """
        if not self._UM_was_initialized:
            self._UM_initialize_general(with_candidates=False)
        if np.isneginf(self._candidates_UM[c]):
            self._compute_UM_c(c)
        return display_pseudo_bool(self._candidates_UM[c]), self.log_UM

    def UM_with_candidates(self):
        """Unison manipulation, full mode.

        We say that a situation is *unison-manipulable* for a candidate
        ``c`` ``!=`` :attr:`~svvamp.ElectionResult.w` iff all voters who
        prefer ``c`` to the sincere winner :attr:`~svvamp.ElectionResult.w`
        can cast the **same** ballot so that ``c`` is elected (while other
        voters still vote sincerely).

        :returns: (``is_UM``, ``log_UM``, ``candidates_UM``).

        ``is_UM``: Boolean (or ``numpy.nan``). ``True`` if UM is possible,
        ``False`` otherwise. If the algorithm cannot decide,
        then ``numpy.nan``.

        ``log_UM``: String. Parameters used to compute UM.

        ``candidates_UM``: 1d array of booleans (or ``numpy.nan``).
        ``candidates_UM[c]`` is ``True`` if UM for candidate ``c`` is
        possible, ``False`` otherwise. If the algorithm cannot decide, then
        ``numpy.nan``. For the sincere winner
        :attr:`~svvamp.ElectionResult.w`, we have by convention
        ``candidates_UM[w] = False``.

        .. seealso::

            :meth:`~svvamp.Election.UM`,
            :meth:`~svvamp.Election.UM_c`.
        """
        if not self._UM_was_initialized:
            self._UM_initialize_general(with_candidates=True)
        if not self._UM_was_computed_with_candidates:
            self._compute_UM(with_candidates=True)
        return display_pseudo_bool(self._is_UM), self.log_UM, \
               self._candidates_UM.astype(np.float)
        
    def _UM_initialize_general(self, with_candidates):
        """Initialize UM variables and do preliminary checks. Used only the
        first time UM is launched (whatever the mode).
        _UM_was_initialized --> True
        _is_UM --> True or False if we know, -inf otherwise.
        _candidates_UM[c] --> True or False if we know, -inf otherwise.

        If _candidates_UM and _is_UM are totally decided to True, False or
        NaN, then _UM_was_computed_with_candidates should become True (not
        mandatory but recommended).
        """
        self._mylog("UM: Initialize", 2)
        self._UM_was_initialized = True
        self._candidates_UM = np.full(self.pop.C, -np.inf)
        self._is_UM = -np.inf
        self._UM_preliminary_checks_general()

    def _UM_preliminary_checks_general(self):
        """Do preliminary checks for UM. Only first time UM is launched.

        Can update some _candidates_UM[c] to True or False (instead of -inf).
        * If some _candidates_UM[c] becomes True, then _is_UM must become True
        as well.
        * If _is_UM becomes True, it is not necessary to update a specific
        _candidates_UM[c].
        * If for all candidates c, _candidates_UM[c] become False,
        then _is_UM must be updated to False.
        * If _is_UM becomes False, then all _candidates_UM[c] must become
        False.
        * If _candidates_UM and _is_UM are totally decided to True or False,
        then _UM_was_computed_with_candidates should become True (not mandatory but
        recommended).
        """
        # 1) Preliminary checks that may improve _candidates_UM (all must be
        # done, except if everything is decided).
        # Majority favorite criterion
        if (self.meets_majority_favorite_c and
                self.pop.plurality_scores_novtb[self.w] > self.pop.V / 2):
            self._mylog("UM impossible (w is a majority favorite).", 2)
            self._is_UM = False
            self._candidates_UM[:] = False
            self._UM_was_computed_with_candidates = True
            return
        if (self.meets_majority_favorite_c_ctb and
                self.w == 0 and
                self.pop.plurality_scores_novtb[self.w] >= self.pop.V / 2):
            self._mylog("UM impossible (w=0 is a majority favorite with " +
                        "candidate tie-breaking).", 2)
            self._is_UM = False
            self._candidates_UM[:] = False
            self._UM_was_computed_with_candidates = True
            return
        if (self.meets_majority_favorite_c_vtb and
                self.pop.plurality_scores_vtb[self.w] > self.pop.V / 2):
            self._mylog("UM impossible (w is a majority favorite with "
                        "voter tie-breaking).", 2)
            self._is_UM = False
            self._candidates_UM[:] = False
            self._UM_was_computed_with_candidates = True
            return
        if (self.meets_majority_favorite_c_vtb_ctb and
                self.w == 0 and
                self.pop.plurality_scores_vtb[self.w] >= self.pop.V / 2):
            self._mylog("UM impossible (w=0 is a majority favorite with " +
                        "voter and candidate tie-breaking).", 2)
            self._is_UM = False
            self._candidates_UM[:] = False
            self._UM_was_computed_with_candidates = True
            return
        # Condorcet resistance
        if (self.meets_Condorcet_c and
                self.w_is_resistant_condorcet_winner):
            self._mylog("UM impossible (w is a Resistant Condorcet winner)", 2)
            self._is_UM = False
            self._candidates_UM[:] = False
            self._UM_was_computed_with_candidates = True
            return
        # Having supporters
        self._candidates_UM[np.logical_not(self.c_has_supporters)] = False
        # 2) Additional preliminary checks from the subclass.
        self._UM_preliminary_checks_general_subclass()
        if np.all(np.equal(self._candidates_UM, False)):
            self._mylog("UM: preliminary checks: UM is impossible.", 2)
            self._is_UM = False
            self._UM_was_computed_with_candidates = True
            return
        if not np.isneginf(self._is_UM):
            return
        # 3) Preliminary checks that gives only global information on _is_UM
        # (may return as soon as decision is made).
        if (self.meets_majority_favorite_c_vtb and
                self.w_is_not_condorcet_admissible):
            self._mylog("UM found (w is not Condorcet-admissible)", 2)
            self._is_UM = True
            return

    def _UM_preliminary_checks_general_subclass(self):
        """Do preliminary checks for UM. Only first time UM is launched.

        Can update some _candidates_UM[c] to True or False (instead of -inf).

        True must be propagated from specific to general, False must be 
        propagated from general to specific.
        * If some _candidates_UM[c] becomes True, then _is_UM must become True
        as well.
        * If _is_UM becomes True, it is not necessary to update a specific
        _candidates_UM[c].
        * If _is_UM becomes False, then all _candidates_UM[c] must become
        False.
        * If for all candidates c, _candidates_UM[c] becomes False,
        it is not necessary to update _is_UM.
        * If _candidates_UM and _is_UM are totally decided to True or False,
        then _UM_was_computed_with_candidates should become True (not mandatory but
        recommended).

        Put first the checks that may improve _candidates_UM (all must be
        done, except if everything is decided).
        Then the checks that gives only global information on _is_UM (which may
        return as soon as decision is made).
        """
        pass

    def _UM_initialize_c(self, c):
        """Initialize the UM loop for candidate c and may do preliminary 
        checks.

        * If _candidates_UM[c] is decided (True/False/NaN), it means that 
            all the work for c has been done before. Then get out.
        * Preliminary checks: try to decide _candidates_UM[c]. If it becomes
            True, then _is_UM must become True as well. In other cases, do not
            update _is_UM.
        """
        self._mylogv("UM: Candidate =", c, 2)
        # Check if job is done for c
        if not np.isneginf(self._candidates_UM[c]):
            self._mylog("UM: Job already done", 2)
            return
        # Preliminary checks
        self._UM_preliminary_checks_c(c)
        # Conclude what we can
        if self._candidates_UM[c] == True:
            self._mylogv("UM: Preliminary checks: UM is True for c =", c, 2)
            self._is_UM = True
        elif self._candidates_UM[c] == False:
            self._mylogv("UM: Preliminary checks: UM is False for c =", c, 2)
        else:
            self._mylogv("UM: Preliminary checks: UM is unknown for c =", c, 2)
        
    def _UM_preliminary_checks_c(self, c):
        """UM: preliminary checks for challenger c.

        Try to decide _candidates_UM[c] to True or False (instead of -inf). Do
        not update _is_UM.
        """
        n_m = self.pop.matrix_duels[c, self.w]  # Number of manipulators
        n_s = self.pop.V - n_m                  # Number of sincere voters
        # Positive pretest based on the majority favorite criterion
        if (self.meets_majority_favorite_c_vtb and
                n_m > self.pop.V / 2):
            self._mylog('UM: Preliminary checks: n_m > V / 2', 3)
            self._candidates_UM[c] = True
            return
        if (self.meets_majority_favorite_c_vtb_ctb and c == 0 and
                n_m >= self.pop.V / 2):
            self._mylog('UM: Preliminary checks: n_m >= V / 2 and c == 0', 3)
            self._candidates_UM[c] = True
            return
        # Negative pretest based on the majority favorite criterion
        # If plurality_scores_novtb[w] > (n_s + n_m) / 2, then CM impossible.
        # Necessary condition: n_m >= 2 * plurality_scores_novtb[w] - n_s.
        if self.meets_majority_favorite_c:
            if n_m < 2 * self.pop.plurality_scores_novtb[self.w] - n_s:
                self._mylog('UM: Preliminary checks: even with n_m '
                            'manipulators, w stays plurality winner (no tvb)',
                            3)
                self._candidates_UM[c] = False
                return
        if self.meets_majority_favorite_c_ctb and self.w == 0:
            if n_m < 2 * self.pop.plurality_scores_novtb[self.w] - n_s + 1:
                self._mylog('UM: Preliminary checks: even with n_m '
                            'manipulators, w stays plurality winner (ctb but '
                            'no vtb)', 3)
                self._candidates_UM[c] = False
                return
        if self.meets_majority_favorite_c_vtb:
            if n_m < 2 * self.pop.plurality_scores_vtb[self.w] - n_s:
                self._mylog('UM: Preliminary checks: even with n_m '
                            'manipulators, w stays plurality winner (with '
                            'vtb)', 3)
                self._candidates_UM[c] = False
                return
        if self.meets_majority_favorite_c_vtb_ctb and self.w == 0:
            if n_m < 2 * self.pop.plurality_scores_vtb[self.w] - n_s + 1:
                self._mylog('UM: Preliminary checks: even with n_m '
                            'manipulators, w stays plurality winner (with '
                            'vtb and ctb)', 3)
                self._candidates_UM[c] = False
                return
        # Pretest based on the same idea as Condorcet resistance
        if self.meets_Condorcet_c:
            if n_m < self.pop.threshold_c_prevents_w_Condorcet[c, self.w]:
                self._mylog('UM: Preliminary checks: c-manipulators cannot '
                            'prevent w from being a Condorcet winner', 3)
                self._candidates_UM[c] = False
                return
        # Other pretests
        self._UM_preliminary_checks_c_subclass(c)

    def _UM_preliminary_checks_c_subclass(self, c):
        """UM: preliminary checks for challenger c.

        Try to decide _candidates_UM[c] to True or False (instead of -inf). Do
        not update _is_UM.
        """
        pass

    def _UM_main_work_c(self, c):
        """ Do the main work in UM loop for candidate c.
        Must decide _candidates_UM[c] (to True, False or NaN).
        Do not update _is_UM.
        """
        # N.B.: in some subclasses, it is possible to try one method,
        # then another one if the first one fails, etc. In this general class,
        # we will simply do a switch between 'lazy' and 'exact'.
        # noinspection PyTypeChecker
        getattr(self, '_UM_main_work_c_' + self.UM_option)(
            c
        )  # Launch a sub-method like _UM_main_work_c_lazy, etc.

    def _UM_main_work_c_lazy(self, c):
        """Do the main work in UM loop for candidate c, with option 'lazy'.
        Must decide _candidates_UM[c] (to True, False or NaN).
        Do not update _is_UM.
        """
        self._candidates_UM[c] = neginf_to_nan(
            self._candidates_UM[c])

    def _UM_main_work_c_exact(self, c):
        """Do the main work in UM loop for candidate c, with option 'exact'.
        Must decide _candidates_UM[c] (to True, False or NaN).
        Do not update _is_UM.
        """
        if self.is_based_on_strict_rankings:
            self._UM_main_work_c_exact_rankings(c)
        elif self.is_based_on_utilities_minus1_1:
            self._UM_main_work_c_exact_utilities_minus1_1(c)
        else:
            raise NotImplementedError("UM: Exact manipulation is not "
                                      "implemented for this voting system.")

    def _UM_main_work_c_exact_rankings(self, c):
        """Do the main work in UM loop for candidate c, with option 'exact',
        for a voting system based only on strict rankings.
        Must decide _candidates_UM[c] (to True, False or NaN).
        Do not update _is_UM.
        """
        preferences_borda_test = np.copy(self.pop.preferences_borda_vtb)
        ballot = np.array(range(self.pop.C))
        ballot_favorite = self.pop.C - 1
        while ballot is not None:  # Loop on possible ballots
            self._mylogv("UM: Ballot =", ballot, 3)
            preferences_borda_test[self.v_wants_to_help_c[:, c], :] = ballot
            pop_test = Population(preferences_utilities=preferences_borda_test)
            result_test = self._create_result(pop_test)
            w_test = result_test.w
            if w_test == c:
                self._candidates_UM[c] = True
                return
            ballot, ballot_favorite = compute_next_borda_clever(
                ballot, ballot_favorite, self.pop.C)
        else:
            self._candidates_UM[c] = False

    def _UM_main_work_c_exact_utilities_minus1_1(self, c):
        """Do the main work in UM loop for candidate c, with option 'exact',
        for a voting system based only on utilities and where it is optimal
        for a c-manipulator to pretend that c has utility 1 and other
        candidates utility 0.
        Must decide _candidates_UM[c] (to True, False or NaN).
        Do not update _is_UM.
        """
        self._candidates_UM[c] = self.TM_c(c)[0]

    def _UM_conclude_c(self, c):
        """Conclude the UM loop for candidate c, according to the value of 
        _candidates_UM[c].
        _is_UM -->
            * If _candidates_UM[c] is True, then _is_UM = True.
            * Otherwise, do not update _is_UM.
        """
        if self._candidates_UM[c] == True:
            self._mylogv("UM: Final answer: UM is True for c =", c, 2)
            self._is_UM = True
        elif self._candidates_UM[c] == False:
            self._mylogv("UM: Final answer: UM is False for c =", c, 2)
        else:
            self._mylogv("UM: Final answer: UM is unknown for c =", c, 2)

    def _compute_UM(self, with_candidates):
        """Compute UM: is_UM.
        
        Note that this method is launched by UM only if _is_UM is not decided,
        and by UM_with_candidates only if not _UM_was_computed_with_candidates. So,
        it is not necessary to do a preliminary check on these variables.

        If with_candidates is False:
        * At the end, _is_UM must be decided to True, False or NaN.
        * _candidates_UM must be at least initialized (to an array of -inf).
        It can be partially decided to True, False or NaN (to avoid some
        computations if we come back later), but it is not mandatory.
        * Coherence is not mandatory: notably, if _is_UM is decided to True,
        it is not necessary to update a specific _candidates_UM[c].
        * _UM_was_initialized must become True.
        * If _is_UM and _candidates_UM are totally decided to True, False or
        NaN, then _UM_was_computed_with_candidates should become True (not
        mandatory but recommended).

        If with_candidates is True:
        * _is_UM and _candidates_UM must be decided to True, False or NaN.
        * _UM_was_initialized and _UM_was_computed_with_candidates must become True.
        """
        # We start with _is_UM = -Inf (undecided).
        # If we find a candidate for which _candidates_UM[c] = NaN, then
        #   _is_UM becomes NaN too ("at least maybe").
        # If we find a candidate for which _candidates_UM[c] = True, then
        #   _is_UM becomes True ("surely yes").
        for c in self.losing_candidates:
            self._compute_UM_c(c)
            if not with_candidates and self._is_UM == True:
                return
            if np.isneginf(self._is_UM) and np.isnan(self._candidates_UM[c]):
                self._is_UM = np.nan
        # If we reach this point, we have decided all _candidates_UM to True,
        # False or NaN.
        self._UM_was_computed_with_candidates = True  # even if with_candidates = False
        self._is_UM = neginf_to_zero(self._is_UM)

    def _compute_UM_c(self, c):
        """Compute UM for candidate c.

        Note that this method checks if _candidates_UM[c] is already decided.
        So, it is not necessary to do this check before calling the method.

        During this method:
        * _candidates_UM[c] must be decided to True, False or NaN.
        * If it becomes True, then _is_UM must become True as well. Otherwise,
        do not update _is_UM.
        """
        self._UM_initialize_c(c)
        if np.isfinite(self._candidates_UM[c]):
            return
        self._UM_main_work_c(c)
        self._UM_conclude_c(c)

    #%% Ignorant-Coalition Manipulation (ICM)
    # When the voting systems meets IgnMC with ctb, it is very easy, and it
    # is managed at the beginning of _compute_ICM.
    # So, for most subroutines, we can suppose that the voting system does not
    # meet IgnMC with ctb.

    @property
    def log_ICM(self):
        """String. Parameters used to compute :meth:`~svvamp.Election.ICM`
        and related methods.
        """
        # noinspection PyTypeChecker
        return "ICM_option = " + self.ICM_option
    
    def ICM(self):
        """Ignorant-Coalition Manipulation.

        :returns: (``is_ICM``, ``log_ICM``).

        Cf. :meth:`~svvamp.Election.ICM_full`.
        """
        if not self._ICM_was_initialized:
            self._ICM_initialize_general(with_candidates=False)
        if np.isneginf(self._is_ICM):
            self._compute_ICM(with_candidates=False, optimize_bounds=False)
        return display_pseudo_bool(self._is_ICM), self.log_ICM
            
    def ICM_c(self, c):
        """Ignorant-Coalition Manipulation, focus on one candidate.

        :param c: Integer (candidate).

        :returns: (``candidates_ICM[c]``, ``log_ICM``).

        Cf. :meth:`~svvamp.Election.ICM_full`.
        """
        if not self._ICM_was_initialized:
            self._ICM_initialize_general(with_candidates=False)
        if np.isneginf(self._candidates_ICM[c]):
            self._compute_ICM_c(c, optimize_bounds=False)
        return display_pseudo_bool(self._candidates_ICM[c]), self.log_ICM

    def ICM_c_with_bounds(self, c):
        """Ignorant-Coalition Manipulation, focus on one candidate, with
        bounds.

        :param c: Integer (candidate).

        :returns: (``candidates_ICM[c]``, ``log_ICM``,
                  ``necessary_coalition_size_ICM[c]``,
                  ``sufficient_coalition_size_ICM[c]``).

        Cf. :meth:`~svvamp.Election.ICM_full`.
        """
        if not self._ICM_was_initialized:
            self._ICM_initialize_general(with_candidates=False)
        if self._bounds_optimized_ICM[c] == False:
            self._compute_ICM_c(c, optimize_bounds=True)
        return display_pseudo_bool(self._candidates_ICM[c]), \
               self.log_ICM, \
               np.float(self._necessary_coalition_size_ICM[c]), \
               np.float(self._sufficient_coalition_size_ICM[c])

    def ICM_with_candidates(self):
        """Ignorant-Coalition Manipulation, with candidates.

        :returns: (``is_ICM``, ``log_ICM``, ``candidates_ICM``).

        Cf. :meth:`~svvamp.Election.ICM_full`.
        """
        if not self._ICM_was_initialized:
            self._ICM_initialize_general(with_candidates=True)
        if not self._ICM_was_computed_with_candidates:
            self._compute_ICM(with_candidates=True, optimize_bounds=False)
        return display_pseudo_bool(self._is_ICM), self.log_ICM, \
               self._candidates_ICM.astype(np.float)

    def ICM_full(self):
        """Ignorant-Coalition Manipulation, full mode.

        We say that a situation is *Ignorant-Coalition Manipulable* (ICM) for
        ``c`` ``!=`` :attr:`~svvamp.ElectionResult.w` iff voters who prefer
        ``c`` to :attr:`~svvamp.ElectionResult.w` can cast ballots so that,
        whatever the other voters do, ``c`` is elected, .

        Internally, to decide the problem, SVVAMP studies the following
        question. When considering the sub-population of voters who do not
        prefer ``c`` to :attr:`~svvamp.ElectionResult.w` (sincere voters),
        what is the minimal number :math:`x_c` of ``c``-manipulators needed to
        perform ICM? For all voting system currently implemented in SVVAMP,
        it means that ICM is possible iff there are :math:`x_c` voters or
        more who prefer ``c`` to :attr:`~svvamp.ElectionResult.w`.

        For information only, the result of SVVAMP's computations about
        :math:`x_c` is given in outputs ``necessary_coalition_size_ICM`` and
        ``sufficient_coalition_size_ICM`` (cf. below). By definition, we have
        ``necessary_coalition_size_ICM`` :math:`\leq x_c \leq`
        ``sufficient_coalition_size_ICM``.

        When :attr:`~svvamp.Election.ICM_option` = ``exact``, the exactness
        concerns the ICM decision problems (boolean results below),
        not the numerical evaluation of :math:`x_c`. It means that for all
        boolean answers below, SVVAMP will not answer ``numpy.nan`` (
        undecided). But it is possible that
        ``necessary_coalition_size_ICM[c]`` <
        ``sufficient_coalition_size_ICM[c]``.

        :returns: (``is_ICM``, ``log_ICM``, ``candidates_ICM``,
                  ``necessary_coalition_size_ICM``,
                  ``sufficient_coalition_size_ICM``).

        ``is_ICM``: Boolean (or ``numpy.nan``). ``True`` if a ICM is possible,
        ``False`` otherwise. If the algorithm cannot decide,
        then ``numpy.nan``.

        ``log_ICM`` -- String. Parameters used to compute ICM.

        ``candidates_ICM``: 1d array of booleans (or ``numpy.nan``).
        ``candidates_ICM[c]`` is ``True`` if ICM for candidate ``c`` is
        possible, ``False`` otherwise. If the algorithm cannot decide, then
        ``numpy.nan``. For the sincere winner
        :attr:`~svvamp.ElectionResult.w`, we have by convention
        ``candidates_ICM[w] = False``.

        ``necessary_coalition_size_ICM``: Integer.
        ``necessary_coalition_size_ICM[c]`` is the lower bound found by the
        algorithm for :math:`x_c`. For the sincere winner
        :attr:`~svvamp.ElectionResult.w`, we have by convention
        ``necessary_coalition_size_ICM[w] = 0``.

        ``sufficient_coalition_size_ICM``: Integer or ``numpy.inf``.
        ``sufficient_coalition_size_ICM[c]`` is the upper bound found by the
        algorithm for :math:`x_c`. For the sincere winner
        :attr:`~svvamp.ElectionResult.w`, we have by convention
        ``sufficient_coalition_size_ICM[w] = 0``.

        .. seealso::

            :meth:`~svvamp.Election.ICM`,
            :meth:`~svvamp.Election.ICM_c`,
            :meth:`~svvamp.Election.ICM_c_with_bounds`,
            :meth:`~svvamp.Election.ICM_with_candidates`.
        """
        if not self._ICM_was_initialized:
            self._ICM_initialize_general(with_candidates=True)
        if not self._ICM_was_computed_full:
            self._compute_ICM(with_candidates=True, optimize_bounds=True)
        return display_pseudo_bool(self._is_ICM), self.log_ICM, \
               self._candidates_ICM.astype(np.float), \
               self._necessary_coalition_size_ICM.astype(np.float), \
               self._sufficient_coalition_size_ICM.astype(np.float)

    def _ICM_initialize_general(self, with_candidates):
        """Initialize ICM variables an do preliminary checks. Used each time 
        ICM is launched (whatever the mode).
        _ICM_was_initialized --> True
        _is_ICM --> False or True if we know, -inf otherwise.
        _candidates_ICM[c] --> True or False if we know, -inf otherwise.

        _sufficient_coalition_size_ICM[c] --> +inf (except for w).
        _necessary_coalition_size_ICM --> 0.
        _bounds_optimized_ICM[c] --> False.
        For _sufficient_coalition_size_ICM and
        _necessary_coalition_size_ICM, it is not recommended to do better
        here.
        """
        self._mylog("ICM: Initialize", 2)
        self._ICM_was_initialized = True
        self._candidates_ICM = np.full(self.pop.C, -np.inf)
        self._candidates_ICM[self.w] = False
        self._sufficient_coalition_size_ICM = np.full(self.pop.C, np.inf)
        self._sufficient_coalition_size_ICM[self.w] = 0
        self._necessary_coalition_size_ICM = np.zeros(self.pop.C)
        self._bounds_optimized_ICM = np.zeros(self.pop.C)
        self._bounds_optimized_ICM[self.w] = True
        self._is_ICM = -np.inf
        self._ICM_preliminary_checks_general()

    def _ICM_preliminary_checks_general(self):
        """Do preliminary checks for ICM. Only first time ICM is
        launched.

        Can update some _candidates_ICM[c] to True or False (instead of
        -inf).
        * If some _candidates_ICM[c] becomes True, then _is_ICM must become
        True as well.
        * If _is_ICM becomes True, it is not necessary to update a specific
        _candidates_ICM[c].
        * If for all candidates c, _candidates_ICM[c] become False,
        then _is_ICM must be updated to False.
        * If _is_ICM becomes False, then all _candidates_ICM[c] must become
        False.
        * If _candidates_ICM and _is_ICM are totally decided to True or
        False, then _ICM_was_computed_with_candidates should become True (not
        mandatory but recommended).
        """
        if self.meets_InfMC_c and self.w_is_condorcet_winner:
            self._mylog("ICM impossible (w is a Condorcet winner)", 2)
            self._is_ICM = False
            self._candidates_ICM[:] = False
            self._ICM_was_computed_with_candidates = True
            return
        self._candidates_ICM[np.logical_not(self.c_has_supporters)] = False
        if np.all(np.equal(self._candidates_ICM, False)):
            self._mylog("ICM: preliminary checks: ICM is impossible.", 2)
            self._is_ICM = False
            self._ICM_was_computed_with_candidates = True
            return
        if self.meets_IgnMC_c and self.w_is_not_condorcet_admissible:
            self._mylog("ICM found (w is not Condorcet-admissible)", 2)
            self._is_ICM = True
            return
        # Other checks
        self._ICM_preliminary_checks_general_subclass()

    def _ICM_preliminary_checks_general_subclass(self):
        """Do preliminary checks for ICM. Only first time ICM is launched.

        Can update _is_ICM to True or False (instead of -inf).
        * If _is_ICM becomes True, it is not necessary to update a specific
        _candidates_ICM[c].
        * If _is_ICM becomes False, then all _candidates_ICM[c] must become
        False. And it is recommended that _ICM_was_computed_with_candidates becomes
        True.

        For _sufficient_coalition_size_ICM and _necessary_coalition_size_ICM, it
        is not recommended to do better here.
        """
        pass

    def _ICM_initialize_c(self, c, optimize_bounds):
        """Initialize the ICM loop for candidate c and do preliminary checks.

        * If _bounds_optimized_ICM[c] is True, it means that all the work
        for c has been done before. Then get out.
        * If _candidates_ICM[c] is decided (True/False/NaN) and
        optimize_bounds is False, then get out.
        * Preliminary checks to improve bounds
            _sufficient_coalition_size_ICM[c] and
            _necessary_coalition_size_ICM[c].
        * If the two bounds are equal, then _bounds_optimized_ICM[c] becomes
        True.
        * Update _candidates_ICM[c] to True or False if possible.
        * If we can decide _is_ICM to True, do it.

        Returns:
        job_done -- True iff we have done all the job for c (with bounds if
        optimize_bounds is True, only for _candidates_ICM[c] otherwise).
        """
        self._mylogv("ICM: Candidate =", c, 2)
        # Check if job is done for c
        if self._bounds_optimized_ICM[c] == True:
            self._mylog("ICM: Job already done", 2)
            return True
        if optimize_bounds == False and not (
                np.isneginf(self._candidates_ICM[c])):
            self._mylog("ICM: Job already done", 2)
            return True
        # Improve bounds
        self._ICM_preliminary_checks_c(c, optimize_bounds)
        # Conclude what we can
        # Some log
        n_m = self.pop.matrix_duels[c, self.w]
        self._mylogv("ICM: Preliminary checks: " +
                     "necessary_coalition_size_ICM[c] =",
                     self._necessary_coalition_size_ICM[c], 3)
        self._mylogv("ICM: Preliminary checks: " + 
                     "sufficient_coalition_size_ICM[c] =",
                     self._sufficient_coalition_size_ICM[c], 3)
        self._mylogv("ICM: Preliminary checks: " + 
                     "n_m =", n_m, 3)
        # Conclude
        if (self._sufficient_coalition_size_ICM[c] ==
                self._necessary_coalition_size_ICM[c]):
            self._mylog("ICM: Preliminary checks: Bounds are equal", 2)
            self._bounds_optimized_ICM[c] = True
        if n_m >= self._sufficient_coalition_size_ICM[c]:
            self._mylogv("ICM: Preliminary checks: ICM is True for c =",
                         c, 2)
            self._candidates_ICM[c] = True
            self._is_ICM = True
            if optimize_bounds == False or self._bounds_optimized_ICM[c]:
                return True
        elif n_m < self._necessary_coalition_size_ICM[c]:
            self._mylogv("ICM: Preliminary checks: ICM is False for c =",
                         c, 2)
            self._candidates_ICM[c] = False
            if optimize_bounds == False or self._bounds_optimized_ICM[c]:
                return True
        else:
            self._mylogv("ICM: Preliminary checks: ICM is unknown for c =",
                         c, 2)
        return False

    def _ICM_preliminary_checks_c(self, c, optimize_bounds):
        """ICM: preliminary checks for challenger c.
        
        Try to improve bounds _sufficient_coalition_size_ICM[c] and
        _necessary_coalition_size_ICM[c]. Do not update other variables.

        If optimize_bounds is False, then return as soon as
        n_m >= _sufficient_coalition_size_ICM[c], or
        _necessary_coalition_size_ICM[c] > n_m (where n_m is the number or
        manipulators).
        """
        n_m = self.pop.matrix_duels[c, self.w]  # Number of manipulators
        n_s = self.pop.V - n_m                  # Number of sincere voters
        if self.meets_InfMC_c_ctb and c != 0:
            self._update_necessary(
                self._necessary_coalition_size_ICM, c, n_s + 1,
                'ICM: InfMC_c_ctb => '
                'necessary_coalition_size_ICM[c] = n_s + 1 =')
            if not optimize_bounds and (
                    self._necessary_coalition_size_ICM[c] > n_m):
                return
        if self.meets_InfMC_c:
            self._update_necessary(
                self._necessary_coalition_size_ICM, c, n_s,
                'ICM: InfMC_c => '
                'necessary_coalition_size_ICM[c] = n_s =')
            if not optimize_bounds and (
                    self._necessary_coalition_size_ICM[c] > n_m):
                return
        if self.meets_IgnMC_c_ctb and c == 0:
            self._update_sufficient(
                self._sufficient_coalition_size_ICM, c, n_s,
                'ICM: IgnMC_c => '
                'sufficient_coalition_size_ICM[c] = n_s =')
            if not optimize_bounds and (
                    n_m >= self._sufficient_coalition_size_ICM[c]):
                return
        if self.meets_IgnMC_c:
            self._update_sufficient(
                self._sufficient_coalition_size_ICM, c, n_s + 1,
                'ICM: IgnMC_c => '
                'sufficient_coalition_size_ICM[c] = n_s + 1 =')
            if not optimize_bounds and (
                    n_m >= self._sufficient_coalition_size_ICM[c]):
                return
        # Other preliminary checks
        self._ICM_preliminary_checks_c_subclass(c, optimize_bounds)

    def _ICM_preliminary_checks_c_subclass(self, c, optimize_bounds):
        """ICM: preliminary checks for challenger c.

        Try to improve bounds _sufficient_coalition_size_ICM[c] and
        _necessary_coalition_size_ICM[c]. Do not update other variables.

        If optimize_bounds is False, then return as soon as
        n_m >= _sufficient_coalition_size_ICM[c], or
        _necessary_coalition_size_ICM[c] > n_m (where n_m is the number or
        manipulators).

        If a test is especially costly, it is recommended to test first if
        _sufficient_coalition_size_ICM[c] == _necessary_coalition_size_ICM[c]
        and to return immediately in that case.
        """
        pass

    def _ICM_main_work_c(self, c, optimize_bounds):
        """Do the main work in ICM loop for candidate c.
        * Try to improve bounds _sufficient_coalition_size_ICM[c] and
        _necessary_coalition_size_ICM[c].
        * Do not update other variables (_is_ICM, _candidates_ICM, etc.).

        If optimize_bounds is False, can return as soon as
        n_m >= _sufficient_coalition_size_ICM[c], or
        _necessary_coalition_size_ICM[c] > n_m (where n_m is the number or
        manipulators).

        Returns:
        is_quick_escape -- True if we did not improve the bound the best we
            could.
            (Allowed to be None or False otherwise).
        """
        # N.B.: in some subclasses, it is possible to try one method,
        # then another one if the first one fails, etc. In this general class,
        # we will simply do a switch between 'lazy' and 'exact'.
        # noinspection PyTypeChecker
        return getattr(self, '_ICM_main_work_c_' + self.ICM_option)(
            c, optimize_bounds
        )  # Launch a sub-method like _ICM_main_work_v_lazy, etc.

    def _ICM_main_work_c_lazy(self, c, optimize_bounds):
        """Do the main work in ICM loop for candidate c, with option 'lazy'.
        Same specifications as _ICM_main_work_c.
        """
        # With option 'lazy', there is nothing to do! And this is not a 'quick
        # escape': we did the best we could (considering laziness).
        # N.B.: for most voting system, lazy is actually quite good for ICM!
        # In fact, as soon as _meets_IgnMC_c_ctb, this lazy method is exact!
        return False

    def _ICM_main_work_c_exact(self, c, optimize_bounds):
        """Do the main work in ICM loop for candidate c, with option 'exact'.
        Same specifications as _ICM_main_work_c.
        """
        if self.meets_IgnMC_c_ctb:
            return False
        else:
            raise NotImplementedError("ICM: Exact manipulation is not "
                                      "implemented for this voting system.")

    def _ICM_conclude_c(self, c, is_quick_escape):
        """Conclude the ICM loop for candidate c.
        _bounds_optimized_ICM[c] --> if not quick_escape, becomes True.
        _candidates_ICM[c] --> True, False or NaN according to the bounds
            _sufficient_coalition_size_ICM[c] and 
            _necessary_coalition_size_ICM[c].
        _is_ICM -->
            * If _candidates_ICM[c] is True, then _is_ICM = True.
            * Otherwise, do not update _is_ICM.
        """
        if not is_quick_escape:
            self._bounds_optimized_ICM[c] = True
        n_m = self.pop.matrix_duels[c, self.w]
        if n_m >= self._sufficient_coalition_size_ICM[c]:
            self._mylogv("ICM: Final answer: ICM is True for c =", c, 2)
            self._candidates_ICM[c] = True
            self._is_ICM = True
        elif n_m < self._necessary_coalition_size_ICM[c]:
            self._mylogv("ICM: Final answer: ICM is False for c =", c, 2)
            self._candidates_ICM[c] = False
        else:
            self._mylogv("ICM: Final answer: ICM is unknown for c =", c, 2)
            self._candidates_ICM[c] = np.nan

    def _compute_ICM(self, with_candidates, optimize_bounds):
        """Compute ICM.
        
        Note that this method is launched by ICM only if not
        _ICM_was_initialized, and by ICM_with_candidates only if not
        _ICM_was_computed_with_candidates. So, it is not necessary to do a
        preliminary check on these variables.
        """
        self._mylog("Compute ICM", 1)
        # We start with _is_ICM = -Inf (undecided).
        # If we find a candidate for which _candidates_ICM[c] = NaN, then 
        #   _is_ICM becomes NaN too ("at least maybe").
        # If we find a candidate for which _candidates_ICM[c] = True, then 
        #   _is_ICM becomes True ("surely yes").
        for c in self.losing_candidates:
            self._compute_ICM_c(c, optimize_bounds)
            if not with_candidates and self._is_ICM == True:
                return
            if np.isneginf(self._is_ICM) and np.isnan(
                    self._candidates_ICM[c]):
                self._is_ICM = np.nan
        # If we reach this point, we have decided all _candidates_ICM to
        # True, False or NaN.
        self._ICM_was_computed_with_candidates = True
        self._is_ICM = neginf_to_zero(self._is_ICM)
        if optimize_bounds:
            self._ICM_was_computed_full = True

    def _compute_ICM_c(self, c, optimize_bounds):
        job_done = self._ICM_initialize_c(c, optimize_bounds)
        if job_done:
            return
        if not optimize_bounds and not np.isneginf(self._candidates_ICM[c]):
            return
        is_quick_escape = self._ICM_main_work_c(c, optimize_bounds)
        self._ICM_conclude_c(c, is_quick_escape)

    #%% Coalition Manipulation (CM)

    @property
    def log_CM(self):
        """String. Parameters used to compute :meth:`~svvamp.Election.CM`
        and related methods.
        """
        # noinspection PyTypeChecker
        if self.CM_option == 'exact':
            return "CM_option = exact"
        else:
            return ("CM_option = " + self.CM_option +
                    self._precheck_UM * (", " + self.log_UM) +
                    self._precheck_ICM * (", " + self.log_ICM) +
                    self._precheck_TM * (", " + self.log_TM))

    def CM(self):
        """Coalition Manipulation.

        :returns: (``is_CM``, ``log_CM``).

        Cf. :meth:`~svvamp.Election.CM_full`.
        """
        if not self._CM_was_initialized:
            self._CM_initialize_general(with_candidates=False)
        if np.isneginf(self._is_CM):
            self._compute_CM(with_candidates=False, optimize_bounds=False)
        return display_pseudo_bool(self._is_CM), self.log_CM

    def CM_c(self, c):
        """Coalition Manipulation, focus on one candidate.

        :param c: Integer (candidate).

        :returns: (``candidates_CM[c]``, ``log_CM``).

        Cf. :meth:`~svvamp.Election.CM_full`.
        """
        if not self._CM_was_initialized:
            self._CM_initialize_general(with_candidates=False)
        if np.isneginf(self._candidates_CM[c]):
            self._compute_CM_c(c, optimize_bounds=False)
        return display_pseudo_bool(self._candidates_CM[c]), self.log_CM

    def CM_c_with_bounds(self, c):
        """Coalition Manipulation, focus on one candidate, with bounds.

        :param c: Integer (candidate).

        :returns: (``candidates_CM[c]``, ``log_CM``,
                  ``necessary_coalition_size_CM[c]``,
                  ``sufficient_coalition_size_CM[c]``).

        Cf. :meth:`~svvamp.Election.CM_full`.
        """
        if not self._CM_was_initialized:
            self._CM_initialize_general(with_candidates=False)
        if self._bounds_optimized_CM[c] == False:
            self._compute_CM_c(c, optimize_bounds=True)
        return display_pseudo_bool(self._candidates_CM[c]), self.log_CM, \
               np.float(self._necessary_coalition_size_CM[c]), \
               np.float(self._sufficient_coalition_size_CM[c])

    def CM_with_candidates(self):
        """Coalition Manipulation, with candidates.

        :returns: (``is_CM``, ``log_CM``, ``candidates_CM``).

        Cf. :meth:`~svvamp.Election.CM_full`.
        """
        if not self._CM_was_initialized:
            self._CM_initialize_general(with_candidates=True)
        if not self._CM_was_computed_with_candidates:
            self._compute_CM(with_candidates=True, optimize_bounds=False)
        return display_pseudo_bool(self._is_CM), self.log_CM, \
               self._candidates_CM.astype(np.float)

    def CM_full(self):
        """Coalition Manipulation, full mode.

        We say that a situation is *Coalitionaly Manipulable* (CM) for
        ``c`` ``!=`` :attr:`~svvamp.ElectionResult.w` iff voters who prefer
        ``c`` to :attr:`~svvamp.ElectionResult.w` can cast ballots so that
        ``c`` is elected (while other voters still vote sincerely).

        Internally, to decide the problem, SVVAMP studies the following
        question. When considering the sub-population of voters who do not
        prefer ``c`` to :attr:`~svvamp.ElectionResult.w` (sincere voters),
        what is the minimal number :math:`x_c` of ``c``-manipulators needed to
        perform CM? For all voting system currently implemented in SVVAMP,
        it means that CM is possible iff there are :math:`x_c` voters or
        more who prefer ``c`` to :attr:`~svvamp.ElectionResult.w`. (A
        sufficient condition on the voting system is that, if a population
        elects ``c``, then an additional voter may always cast a ballot so that
        ``c`` stays elected)

        For information only, the result of SVVAMP's computations about
        :math:`x_c` is given in outputs ``necessary_coalition_size_CM`` and
        ``sufficient_coalition_size_CM`` (cf. below). By definition, we have
        ``necessary_coalition_size_CM`` :math:`\leq x_c \leq`
        ``sufficient_coalition_size_CM``.

        When :attr:`~svvamp.Election.CM_option` = ``exact``, the exactness
        concerns the CM decision problems (boolean results below),
        not the numerical evaluation of :math:`x_c`. It means that for all
        boolean answers below, SVVAMP will not answer ``numpy.nan`` (
        undecided). But it is possible that
        ``necessary_coalition_size_CM[c]`` <
        ``sufficient_coalition_size_CM[c]``.

        :returns: (``is_CM``, ``log_CM``, ``candidates_CM``,
                  ``necessary_coalition_size_CM``,
                  ``sufficient_coalition_size_CM``).

        ``is_CM``: Boolean (or ``numpy.nan``). ``True`` if a CM is possible,
        ``False`` otherwise. If the algorithm cannot decide,
        then ``numpy.nan``.

        ``log_CM`` -- String. Parameters used to compute CM.

        ``candidates_CM``: 1d array of booleans (or ``numpy.nan``).
        ``candidates_CM[c]`` is ``True`` if CM for candidate ``c`` is
        possible, ``False`` otherwise. If the algorithm cannot decide, then
        ``numpy.nan``. For the sincere winner
        :attr:`~svvamp.ElectionResult.w`, we have by convention
        ``candidates_CM[w] = False``.

        ``necessary_coalition_size_CM``: Integer.
        ``necessary_coalition_size_CM[c]`` is the lower bound found by the
        algorithm for :math:`x_c`. For the sincere winner
        :attr:`~svvamp.ElectionResult.w`, we have by convention
        ``necessary_coalition_size_CM[w] = 0``.

        ``sufficient_coalition_size_CM``: Integer or ``numpy.inf``.
        ``sufficient_coalition_size_CM[c]`` is the upper bound found by the
        algorithm for :math:`x_c`. For the sincere winner
        :attr:`~svvamp.ElectionResult.w`, we have by convention
        ``sufficient_coalition_size_CM[w] = 0``.

        .. seealso::

            :meth:`~svvamp.Election.CM`,
            :meth:`~svvamp.Election.CM_c`,
            :meth:`~svvamp.Election.CM_c_with_bounds`,
            :meth:`~svvamp.Election.CM_with_candidates`.
        """
        if not self._CM_was_initialized:
            self._CM_initialize_general(with_candidates=True)
        if not self._CM_was_computed_full:
            self._compute_CM(with_candidates=True, optimize_bounds=True)
        return display_pseudo_bool(self._is_CM), self.log_CM, \
               self._candidates_CM.astype(np.float), \
               self._necessary_coalition_size_CM.astype(np.float), \
               self._sufficient_coalition_size_CM.astype(np.float)

    def _CM_initialize_general(self, with_candidates):
        """Initialize CM variables and do preliminary checks. Used only the
        first time CM is launched (whatever the mode).
        _CM_was_initialized --> True
        _is_CM --> False or True if we know, -inf otherwise.
        _candidates_CM[c] --> True or False if we know, -inf otherwise.

        _sufficient_coalition_size_CM[c] --> +inf (except for w).
        _necessary_coalition_size_CM[c] --> 0.
        _bounds_optimized_CM[c] --> False.
        For _sufficient_coalition_size_CM and _necessary_coalition_size_CM, it
        is not recommended to do better here.
        """
        self._mylog("CM: Initialize", 2)
        self._CM_was_initialized = True
        self._candidates_CM = np.full(self.pop.C, -np.inf)
        self._candidates_CM[self.w] = False
        self._sufficient_coalition_size_CM = np.full(self.pop.C, np.inf)
        self._sufficient_coalition_size_CM[self.w] = 0
        self._necessary_coalition_size_CM = np.zeros(self.pop.C)
        self._bounds_optimized_CM = np.zeros(self.pop.C)
        self._bounds_optimized_CM[self.w] = True
        self._is_CM = -np.inf
        self._CM_preliminary_checks_general()

    def _CM_preliminary_checks_general(self):
        """Do preliminary checks for CM. Only first time CM is launched.

        Can update some _candidates_CM[c] to True or False (instead of -inf).
        * If some _candidates_CM[c] becomes True, then _is_CM must become True
        as well.
        * If _is_CM becomes True, it is not necessary to update a specific
        _candidates_CM[c].
        * If for all candidates c, _candidates_CM[c] become False,
        then _is_CM must be updated to False.
        * If _is_CM becomes False, then all _candidates_CM[c] must become
        False.
        * If _candidates_CM and _is_CM are totally decided to True or False,
        then _CM_was_computed_with_candidates should become True (not mandatory but
        recommended).
        """
        # 1) Preliminary checks that may improve _candidates_CM (all must be
        # done, except if everything is decided).
        # Majority favorite criterion
        if (self.meets_majority_favorite_c and
                self.pop.plurality_scores_novtb[self.w] > self.pop.V / 2):
            self._mylog("CM impossible (w is a majority favorite).", 2)
            self._is_CM = False
            self._candidates_CM[:] = False
            self._CM_was_computed_with_candidates = True
            return
        if (self.meets_majority_favorite_c_ctb and
                self.w == 0 and
                self.pop.plurality_scores_novtb[self.w] >= self.pop.V / 2):
            self._mylog("CM impossible (w=0 is a majority favorite with " +
                        "candidate tie-breaking).", 2)
            self._is_CM = False
            self._candidates_CM[:] = False
            self._CM_was_computed_with_candidates = True
            return
        if (self.meets_majority_favorite_c_vtb and
                self.pop.plurality_scores_vtb[self.w] > self.pop.V / 2):
            self._mylog("CM impossible (w is a majority favorite with "
                        "voter tie-breaking).", 2)
            self._is_CM = False
            self._candidates_CM[:] = False
            self._CM_was_computed_with_candidates = True
            return
        if (self.meets_majority_favorite_c_vtb_ctb and
                self.w == 0 and
                self.pop.plurality_scores_vtb[self.w] >= self.pop.V / 2):
            self._mylog("CM impossible (w=0 is a majority favorite with " +
                        "voter and candidate tie-breaking).", 2)
            self._is_CM = False
            self._candidates_CM[:] = False
            self._CM_was_computed_with_candidates = True
            return
        # Condorcet resistance
        if (self.meets_Condorcet_c and
                self.w_is_resistant_condorcet_winner):
            self._mylog("CM impossible (w is a Resistant Condorcet winner)", 2)
            self._is_CM = False
            self._candidates_CM[:] = False
            self._CM_was_computed_with_candidates = True
            return
        # Having supporters
        self._candidates_CM[np.logical_not(self.c_has_supporters)] = False
        if np.all(np.equal(self._candidates_CM, False)):
            self._mylog("CM: preliminary checks: CM is impossible.", 2)
            self._is_CM = False
            self._CM_was_computed_with_candidates = True
            return
        # 2) Preliminary checks that gives only global information on _is_CM
        # (may return as soon as decision is made).
        # InfMC
        if self.meets_InfMC_c and self.w_is_not_condorcet_admissible:
            self._mylog("CM found (w is not Condorcet-admissible)", 2)
            self._is_CM = True
            return
        # ICM
        if self._precheck_ICM:
            if self.ICM()[0] == True:
                self._mylog("CM found (thanks to ICM)", 2)
                self._is_CM = True
                return
        # TM
        if self._precheck_TM:
            if self.TM()[0] == True:
                self._mylog("CM found (thanks to TM)", 2)
                self._is_CM = True
                return
        # UM
        if self._precheck_UM:
            if self.UM()[0] == True:
                self._mylog("CM found (thanks to UM)", 2)
                self._is_CM = True
                return
        # 3) Other checks
        self._CM_preliminary_checks_general_subclass()

    def _CM_preliminary_checks_general_subclass(self):
        """Do preliminary checks for CM. Only first time CM is launched.

        Can update _is_CM to True or False (instead of -inf).
        * If _is_CM becomes True, it is not necessary to update a specific
        _candidates_CM[c].
        * If _is_CM becomes False, then all _candidates_CM[c] must become
        False. And it is recommended that _CM_was_computed_with_candidates becomes
        True.

        For _sufficient_coalition_size_CM and _necessary_coalition_size_CM, it
        is not recommended to do better here.
        """
        pass

    def _CM_initialize_c(self, c, optimize_bounds):
        """Initialize the CM loop for candidate c and do preliminary checks.

        * If _bounds_optimized_CM[c] is True, it means that all the work for c
        has been done before. Then get out.
        * If _candidates_CM[c] is decided (True/False/NaN) and
        optimize_bounds is False, then get out.
        * Preliminary checks to improve bounds
            _sufficient_coalition_size_CM[c] and
            _necessary_coalition_size_CM[c].
        * If the two bounds are equal, then _bounds_optimized_CM[c] becomes
        True.
        * Update _candidates_CM[c] to True or False if possible.
        * If we can decide _is_CM to True, do it.

        Returns:
        job_done -- True iff we have done all the job for c (with bounds if
        optimize_bounds is True, only for _candidates_CM[c] otherwise).
        """
        self._mylogv("CM: Candidate =", c, 2)
        # Check  if job is done for c
        if self._bounds_optimized_CM[c] == True:
            self._mylog("CM: Job already done", 2)
            return True
        if optimize_bounds == False and not (
                np.isneginf(self._candidates_CM[c])):
            self._mylog("CM: Job already done", 2)
            return True
        # Improve bounds
        self._CM_preliminary_checks_c(c, optimize_bounds)
        # Conclude what we can
        # Some log
        n_m = self.pop.matrix_duels[c, self.w]
        self._mylogv("CM: Preliminary checks: " + 
                     "necessary_coalition_size_CM[c] =",
                     self._necessary_coalition_size_CM[c], 3)
        self._mylogv("CM: Preliminary checks: " + 
                     "sufficient_coalition_size_CM[c] =",
                     self._sufficient_coalition_size_CM[c], 3)
        self._mylogv("CM: Preliminary checks: " + 
                     "n_m =", n_m, 3)
        # Conclude
        if (self._sufficient_coalition_size_CM[c] ==
                self._necessary_coalition_size_CM[c]):
            self._mylog("CM: Preliminary checks: Bounds are equal", 2)
            self._bounds_optimized_CM[c] = True
        if n_m >= self._sufficient_coalition_size_CM[c]:
            self._mylogv("CM: Preliminary checks: CM is True for c =", c, 2)
            self._candidates_CM[c] = True
            self._is_CM = True
            if optimize_bounds == False or self._bounds_optimized_CM[c]:
                return True
        elif n_m < self._necessary_coalition_size_CM[c]:
            self._mylogv("CM: Preliminary checks: CM is False for c =", c, 2)
            self._candidates_CM[c] = False
            if optimize_bounds == False or self._bounds_optimized_CM[c]:
                return True
        else:
            self._mylogv("CM: Preliminary checks: CM is unknown for c =", c, 2)
        return False

    def _CM_preliminary_checks_c(self, c, optimize_bounds):
        """CM: preliminary checks for challenger c.
        
        Try to improve bounds _sufficient_coalition_size_CM[c] and
        _necessary_coalition_size_CM[c]. Do not update other variables.

        If optimize_bounds is False, then return as soon as
        n_m >= _sufficient_coalition_size_CM[c], or
        _necessary_coalition_size_CM[c] > n_m (where n_m is the number or
        manipulators).
        """
        n_m = self.pop.matrix_duels[c, self.w]  # Number of manipulators
        n_s = self.pop.V - n_m                  # Number of sincere voters
        # Pretest based on Informed Majority Coalition Criterion
        if self.meets_InfMC_c_ctb and c == 0:
            self._update_sufficient(
                self._sufficient_coalition_size_CM, c, n_s,
                'CM: Preliminary checks: InfMC_c_ctb => \n    '
                'sufficient_coalition_size_CM[c] = n_s =')
            if not optimize_bounds and (
                    n_m >= self._sufficient_coalition_size_CM[c]):
                return
        if self.meets_InfMC_c:
            self._update_sufficient(
                self._sufficient_coalition_size_CM, c, n_s + 1,
                'CM: Preliminary checks: InfMC_c => \n    '
                'sufficient_coalition_size_CM[c] = n_s + 1 =')
            if not optimize_bounds and (
                    n_m >= self._sufficient_coalition_size_CM[c]):
                return
        # Pretest based on the majority favorite criterion
        # If plurality_scores_novtb[w] > (n_s + n_m) / 2, then CM impossible.
        # Necessary condition: n_m >= 2 * plurality_scores_novtb[w] - n_s.
        if self.meets_majority_favorite_c_vtb_ctb and self.w == 0:
            self._update_necessary(
                self._necessary_coalition_size_CM, c,
                2 * self.pop.plurality_scores_vtb[self.w] - n_s + 1,
                'CM: Preliminary checks: majority_favorite_c_vtb_ctb => \n    '
                'necessary_coalition_size_CM[c] = '
                '2 * plurality_scores_vtb[w] - n_s + 1 =')
            if not optimize_bounds and (
                    self._necessary_coalition_size_CM[c] > n_m):
                return
        if self.meets_majority_favorite_c_vtb:
            self._update_necessary(
                self._necessary_coalition_size_CM, c,
                2 * self.pop.plurality_scores_vtb[self.w] - n_s,
                'CM: Preliminary checks: majority_favorite_c_vtb => \n    '
                'necessary_coalition_size_CM[c] = '
                '2 * plurality_scores_vtb[w] - n_s =')
            if not optimize_bounds and (
                    self._necessary_coalition_size_CM[c] > n_m):
                return
        if self.meets_majority_favorite_c_ctb and self.w == 0:
            self._update_necessary(
                self._necessary_coalition_size_CM, c,
                2 * self.pop.plurality_scores_novtb[self.w] - n_s + 1,
                'CM: Preliminary checks: majority_favorite_c_ctb => \n    '
                'necessary_coalition_size_CM[c] ='
                '2 * plurality_scores_novtb[w] - n_s + 1 =')
            if not optimize_bounds and (
                    self._necessary_coalition_size_CM[c] > n_m):
                return
        if self.meets_majority_favorite_c:
            self._update_necessary(
                self._necessary_coalition_size_CM, c,
                2 * self.pop.plurality_scores_novtb[self.w] - n_s,
                'CM: Preliminary checks: majority_favorite_c => \n    '
                'necessary_coalition_size_CM[c] = '
                '2 * plurality_scores_novtb[w] - n_s =')
            if not optimize_bounds and (
                    self._necessary_coalition_size_CM[c] > n_m):
                return
        # Pretest based on the same idea as Condorcet resistance
        if self.meets_Condorcet_c:
            self._update_necessary(
                self._necessary_coalition_size_CM, c,
                self.pop.threshold_c_prevents_w_Condorcet[c, self.w],
                'CM: Preliminary checks: Condorcet_c => \n    '
                'necessary_coalition_size_CM[c] = '
                'threshold_c_prevents_w_Condorcet[c, w] =')
            if not optimize_bounds and (
                    self._necessary_coalition_size_CM[c] > n_m):
                return
        # Pretests based on ICM, TM and UM
        if self._precheck_ICM:
            is_ICM_c, _, nec_ICM_c, suf_ICM_c = (
                self.ICM_c_with_bounds(c))
            self._update_sufficient(
                self._sufficient_coalition_size_CM, c,
                suf_ICM_c,
                'CM: Preliminary checks: ICM => \n    '
                'sufficient_coalition_size_CM[c] = '
                'sufficient_coalition_size_ICM[c] =')
            if not optimize_bounds and (
                    n_m >= self._sufficient_coalition_size_CM[c]):
                return
        if (self._precheck_TM and
                self._necessary_coalition_size_CM[c] <= n_m <
                self._sufficient_coalition_size_CM[c]):
            if self.TM_c(c)[0] == True:
                self._update_sufficient(
                    self._sufficient_coalition_size_CM, c, n_m,
                    'CM: Preliminary checks: TM => \n    '
                    'sufficient_coalition_size_CM[c] = n_m =')
                if not optimize_bounds:
                    return
        if (self._precheck_UM and
                self._necessary_coalition_size_CM[c] <= n_m <
                self._sufficient_coalition_size_CM[c]):
            if self.UM_c(c)[0] == True:
                self._update_sufficient(
                    self._sufficient_coalition_size_CM, c, n_m,
                    'CM: Preliminary checks: UM => \n    '
                    'sufficient_coalition_size_CM[c] = n_m =')
                if not optimize_bounds:
                    return
        # Other preliminary checks
        self._CM_preliminary_checks_c_subclass(c, optimize_bounds)

    def _CM_preliminary_checks_c_subclass(self, c, optimize_bounds):
        """CM: preliminary checks for challenger c.

        Try to improve bounds _sufficient_coalition_size_CM[c] and
        _necessary_coalition_size_CM[c]. Do not update other variables.

        If optimize_bounds is False, then return as soon as
        n_m >= _sufficient_coalition_size_CM[c], or
        _necessary_coalition_size_CM[c] > n_m (where n_m is the number or
        manipulators).

        If a test is especially costly, it is recommended to test first if
        _sufficient_coalition_size_CM[c] == _necessary_coalition_size_CM[c]
        and to return immediately in that case.
        """
        pass

    def _CM_main_work_c(self, c, optimize_bounds):
        """Do the main work in CM loop for candidate c.
        * Try to improve bounds _sufficient_coalition_size_CM[c] and
        _necessary_coalition_size_CM[c].
        * Do not update other variables (_is_CM, _candidates_CM, etc.).

        If optimize_bounds is False, can return as soon as
        n_m >= _sufficient_coalition_size_CM[c], or
        _necessary_coalition_size_CM[c] > n_m (where n_m is the number or
        manipulators).

        Returns:
        is_quick_escape -- True if we did not improve the bound the best we
            could.
            (Allowed to be None or False otherwise).
        """
        # N.B.: in some subclasses, it is possible to try one method,
        # then another one if the first one fails, etc. In this general class,
        # we will simply do a switch between 'lazy' and 'exact'.
        # noinspection PyTypeChecker
        return getattr(self, '_CM_main_work_c_' + self.CM_option)(
            c, optimize_bounds
        )  # Launch a sub-method like _CM_main_work_v_lazy, etc.
    
    def _CM_main_work_c_lazy(self, c, optimize_bounds):
        """Do the main work in CM loop for candidate c, with option 'lazy'.
        Same specifications as _CM_main_work_c.
        """
        # With option 'lazy', there is nothing to do! And this is not a 'quick
        # escape': we did the best we could (considering laziness).
        return False

    def _CM_main_work_c_exact(self, c, optimize_bounds):
        """Do the main work in CM loop for candidate c, with option 'exact'.
        Same specifications as _CM_main_work_c.
        """
        if self.is_based_on_utilities_minus1_1:
            # TM was already checked during preliminary checks.
            # If TM was not True, then CM impossible.
            self._update_necessary(self._necessary_coalition_size_CM, c,
                                   self.pop.matrix_duels[c, self.w] + 1)
            return
        if not self.is_based_on_strict_rankings:
            raise NotImplementedError("CM: Exact manipulation is not "
                                      "implemented for this voting system.")
        n_m = self.pop.matrix_duels[c, self.w]
        if n_m < self._necessary_coalition_size_CM[c]:
            # This exhaustive algorithm will not do better (so, this is not a
            # quick escape).
            return
        if n_m >= self._sufficient_coalition_size_CM[c]:
            # Idem.
            return
        preferences_borda_temp = np.concatenate((
            np.tile(range(self.pop.C), (n_m, 1)),
            self.pop.preferences_borda_vtb[np.logical_not(
                self._v_wants_to_help_c[:, c]), :],
        ))
        manipulator_favorite = np.full(n_m, self.pop.C - 1)
        while preferences_borda_temp is not None:
            # self._mylogm('preferences_borda_temp =',
            #              preferences_borda_temp, 3)
            pop_test = Population(preferences_utilities=preferences_borda_temp)
            result_test = self._create_result(pop_test)
            w_test = result_test.w
            if w_test == c:
                self._update_sufficient(
                    self._sufficient_coalition_size_CM, c, n_m,
                    'CM: Manipulation found by exhaustive test =>\n'
                    '    sufficient_coalition_size_CM = n_m =')
                break
            for i_manipulator in range(n_m-1,-1,-1):
                new_ballot, new_favorite = compute_next_borda_clever(
                    preferences_borda_temp[i_manipulator, :],
                    manipulator_favorite[i_manipulator], self.pop.C
                )
                # self._mylogv('new_ballot = ', new_ballot)
                if new_ballot is None:
                    continue
                preferences_borda_temp[i_manipulator:n_m, :] = new_ballot[
                                                               np.newaxis, :]
                manipulator_favorite[i_manipulator:n_m] = new_favorite
                break
            else:
                preferences_borda_temp = None
        else:
            self._update_necessary(
                self._necessary_coalition_size_CM, c, n_m + 1,
                'CM: Manipulation proven impossible by exhaustive test =>\n'
                '    necessary_coalition_size_CM[c] = n_m + 1 =')

    def _CM_conclude_c(self, c, is_quick_escape):
        """Conclude the CM loop for candidate c.
        _bounds_optimized_CM[c] --> if not quick_escape, becomes True.
        _candidates_CM[c] --> True, False or NaN according to the bounds
            _sufficient_coalition_size_CM[c] and 
            _necessary_coalition_size_CM[c].
        _is_CM -->
            * If _candidates_CM[c] is True, then _is_CM = True.
            * Otherwise, do not update _is_CM.
        """
        if not is_quick_escape:
            self._bounds_optimized_CM[c] = True
        n_m = self.pop.matrix_duels[c, self.w]
        if n_m >= self._sufficient_coalition_size_CM[c]:
            self._mylogv("CM: Final answer: CM is True for c =", c, 2)
            self._candidates_CM[c] = True
            self._is_CM = True
        elif n_m < self._necessary_coalition_size_CM[c]:
            self._mylogv("CM: Final answer: CM is False for c =", c, 2)
            self._candidates_CM[c] = False
        else:
            self._mylogv("CM: Final answer: CM is unknown for c =", c, 2)
            self._candidates_CM[c] = np.nan

    def _compute_CM(self, with_candidates, optimize_bounds):
        """Compute CM.
        
        Note that this method is launched by CM only if not _CM_was_initialized,
        and by CM_with_candidates only if not _CM_was_computed_with_candidates. So,
        it is not necessary to do a preliminary check on these variables.
        """
        # We start with _is_CM = -Inf (undecided).
        # If we find a candidate for which _candidates_CM[c] = NaN, then 
        #   _is_CM becomes NaN too ("at least maybe").
        # If we find a candidate for which _candidates_CM[c] = True, then 
        #   _is_CM becomes True ("surely yes").
        for c in self.losing_candidates:
            self._compute_CM_c(c, optimize_bounds)
            if not with_candidates and self._is_CM == True:
                return
            if np.isneginf(self._is_CM) and np.isnan(self._candidates_CM[c]):
                self._is_CM = np.nan
        # If we reach this point, we have decided all _candidates_CM to True,
        # False or NaN.
        self._CM_was_computed_with_candidates = True  # even if with_candidates = False
        self._is_CM = neginf_to_zero(self._is_CM)
        if optimize_bounds:
            self._CM_was_computed_full = True

    def _compute_CM_c(self, c, optimize_bounds):
        job_done = self._CM_initialize_c(c, optimize_bounds)
        if job_done:
            return
        if not optimize_bounds and not np.isneginf(self._candidates_CM[c]):
            return
        is_quick_escape = self._CM_main_work_c(c, optimize_bounds)
        self._CM_conclude_c(c, is_quick_escape)

    def demo(self, log_depth=1):
        """Demonstrate the methods of Election class.

        Arguments:
        log_depth -- Integer from 0 (basic info) to 3 (verbose).
        """
        super().demo(log_depth=log_depth)
        old_log_depth = self._log_depth
        self._log_depth = log_depth

        MyLog.print_big_title("Election Class")
        
        MyLog.print_title("Basic properties of the voting system")
        print("with_two_candidates_reduces_to_plurality = ",
              self.with_two_candidates_reduces_to_plurality)
        print("is_based_on_strict_rankings = ",
              self.is_based_on_strict_rankings)
        print("is_based_on_utilities_minus1_1 = ",
              self.is_based_on_utilities_minus1_1)
        print("meets_IIA = ",
              self.meets_IIA)

        MyLog.print_title("Manipulation properties of the voting system")
        # Condorcet_c_rel_ctb (False)        ==>        Condorcet_c_rel (False)
        #  ||                                                               ||
        #  ||    Condorcet_c_vtb_ctb (False) ==> Condorcet_c_vtb (False)    ||
        #  ||           ||           ||               ||         ||         ||
        #  V            V            ||               ||         V          V
        # Condorcet_c_ctb (False)            ==>            Condorcet_c (False)
        #  ||                        ||               ||                    ||
        #  ||                        V                V                     ||
        #  ||      maj_fav_c_vtb_ctb (False) ==> maj_fav_c_vtb (False)      ||
        #  ||           ||                                       ||         ||
        #  V            V                                        V          V
        # majority_favorite_c_ctb (False)    ==>    majority_favorite_c (False)
        #  ||                                                               ||
        #  V                                                                V
        # IgnMC_c_ctb (False)                ==>                IgnMC_c (False)
        #  ||                                                               ||
        #  V                                                                V
        # InfMC_c_ctb (False)                ==>                InfMC_c (False)
        def display_bool(value):
            if value == True:
                return '(True) '
            else:
                return '(False)'
        print('Condorcet_c_rel_ctb ' +
              display_bool(self.meets_Condorcet_c_rel_ctb) +
              '        ==>        Condorcet_c_rel ' +
              display_bool(self.meets_Condorcet_c_rel))
        print(' ||                               '
              '                                ||')
        print(' ||    Condorcet_c_vtb_ctb ' +
              display_bool(self.meets_Condorcet_c_vtb_ctb) +
              ' ==> Condorcet_c_vtb ' +
              display_bool(self.meets_Condorcet_c_vtb) +
              '    ||')
        print(' ||           ||           ||      '
              '         ||         ||         ||')
        print(' V            V            ||      '
              '         ||         V          V')
        print('Condorcet_c_ctb ' +
              display_bool(self.meets_Condorcet_c_ctb) +
              '            ==>            Condorcet_c ' +
              display_bool(self.meets_Condorcet_c))
        print(' ||                        ||      '
              '         ||                    ||')
        print(' ||                        V       '
              '         V                     ||')
        print(' ||      maj_fav_c_vtb_ctb ' +
              display_bool(self.meets_majority_favorite_c_vtb_ctb) +
              ' ==> maj_fav_c_vtb ' +
              display_bool(self.meets_majority_favorite_c_vtb) +
              '      ||')
        print(' ||           ||                   '
              '                    ||         ||')
        print(' V            V                    '
              '                    V          V')
        print('majority_favorite_c_ctb ' +
              display_bool(self.meets_majority_favorite_c_ctb) +
              '    ==>    majority_favorite_c ' +
              display_bool(self.meets_majority_favorite_c))
        print(' ||                                '
              '                               ||')
        print(' V                                 '
              '                               V')
        print('IgnMC_c_ctb ' +
              display_bool(self.meets_IgnMC_c_ctb) +
              '                ==>                IgnMC_c ' +
              display_bool(self.meets_IgnMC_c))
        print(' ||                                '
              '                               ||')
        print(' V                                 '
              '                               V')
        print('InfMC_c_ctb ' +
              display_bool(self.meets_InfMC_c_ctb) +
              '                ==>                InfMC_c ' +
              display_bool(self.meets_InfMC_c))

        MyLog.print_title("Independence of Irrelevant Alternatives (IIA)")
        print("w (reminder) =", self.w)
        not_is_IIA, log_IIA, example_subset_IIA, example_winner_IIA = (
            self.not_IIA_complete())
        print("is_IIA =", not not_is_IIA)
        print("log_IIA:", log_IIA)
        print("example_winner_IIA =", example_winner_IIA)
        print("example_subset_IIA =", example_subset_IIA)

        MyLog.print_title("c-Manipulators")
        MyLog.printm("w (reminder) =", self.w)
        MyLog.printm("preferences_utilities (reminder) =",
                     self.pop.preferences_utilities)
        MyLog.printm("v_wants_to_help_c = ", self.v_wants_to_help_c)

        MyLog.print_title("Individual Manipulation (IM)")
        is_IM, log_IM = self.IM()
        print("is_IM =", is_IM)
        print("log_IM:", log_IM)
        print("")
        is_IM, log_IM, candidates_IM = self.IM_with_candidates()
        print("is_IM =", is_IM)
        print("log_IM:", log_IM)
        MyLog.printm("candidates_IM =", candidates_IM)

        MyLog.print_title("Trivial Manipulation (TM)")
        is_TM, log_TM = self.TM()
        print("is_TM =", is_TM)
        print("log_TM:", log_TM)
        print("")
        # for c in range(self.pop.C):
        #     is_TM_c, log_TM = self.TM_c(c)
        #     print("c =", c)
        #     print("is_TM_c =", is_TM_c)
        #     # print("log_TM:", log_TM)
        #     print("")
        is_TM, log_TM, candidates_TM = self.TM_with_candidates()
        print("is_TM =", is_TM)
        print("log_TM:", log_TM)
        MyLog.printm("candidates_TM =", candidates_TM)

        MyLog.print_title("Unison Manipulation (UM)")
        is_UM, log_UM = self.UM()
        print("is_UM =", is_UM)
        print("log_UM:", log_UM)
        print("")
        # for c in range(self.pop.C):
        #     is_UM_c, log_UM = self.UM_c(c)
        #     print("c =", c)
        #     print("is_UM_c =", is_UM_c)
        #     # print("log_UM:", log_UM)
        #     print("")
        is_UM, log_UM, candidates_UM = self.UM_with_candidates()
        print("is_UM =", is_UM)
        print("log_UM:", log_UM)
        MyLog.printm("candidates_UM =", candidates_UM)

        MyLog.print_title("Ignorant-Coalition Manipulation (ICM)")
        is_ICM, log_ICM = self.ICM()
        print("is_ICM =", is_ICM)
        print("log_ICM:", log_ICM)
        print("")
        # for c in range(self.pop.C):
        #     is_ICM_c, log_ICM = self.ICM_c(c)
        #     print("c =", c)
        #     print("is_ICM_c =", is_ICM_c)
        #     # print("log_ICM:", log_ICM)
        #     print("")
        # for c in range(self.pop.C):
        #     is_ICM_c, log_ICM, nec, suf = self.ICM_c_with_bounds(c)
        #     print("c =", c)
        #     print("is_ICM_c =", is_ICM_c)
        #     # print("log_ICM:", log_ICM)
        #     print("necessary_coalition_size_ICM_c =", nec)
        #     print("sufficient_coalition_size_ICM_c =", suf)
        #     print("")
        is_ICM, log_ICM, candidates_ICM = self.ICM_with_candidates()
        print("is_ICM =", is_ICM)
        print("log_ICM:", log_ICM)
        MyLog.printm("candidates_ICM =", candidates_ICM)
        print("")
        is_ICM, log_ICM, candidates_ICM, \
            necessary_coalition_size_ICM, \
            sufficient_coalition_size_ICM = self.ICM_full()
        print("is_ICM =", is_ICM)
        print("log_ICM:", log_ICM)
        MyLog.printm("candidates_ICM =", candidates_ICM)
        MyLog.printm("necessary_coalition_size_ICM =",
                 necessary_coalition_size_ICM)
        MyLog.printm("sufficient_coalition_size_ICM =",
                 sufficient_coalition_size_ICM)

        MyLog.print_title('Coalition Manipulation (CM)')
        is_CM, log_CM = self.CM()
        print("is_CM =", is_CM)
        print("log_CM:", log_CM)
        print("")
        # for c in range(self.pop.C):
        #     is_CM_c, log_CM = self.CM_c(c)
        #     print("c =", c)
        #     print("is_CM_c =", is_CM_c)
        #     # print("log_CM:", log_CM)
        #     print("")
        # for c in range(self.pop.C):
        #     is_CM_c, log_CM, nec, suf = self.CM_c_with_bounds(c)
        #     print("c =", c)
        #     print("is_CM_c =", is_CM_c)
        #     # print("log_CM:", log_CM)
        #     print("necessary_coalition_size_CM_c =", nec)
        #     print("sufficient_coalition_size_CM_c =", suf)
        #     print("")
        is_CM, log_CM, candidates_CM = self.CM_with_candidates()
        print("is_CM =", is_CM)
        print("log_CM:", log_CM)
        MyLog.printm("candidates_CM =", candidates_CM)
        print("")
        is_CM, log_CM, candidates_CM, \
            necessary_coalition_size_CM, \
            sufficient_coalition_size_CM = self.CM_full()
        print("is_CM =", is_CM)
        print("log_CM:", log_CM)
        MyLog.printm("candidates_CM =", candidates_CM)
        MyLog.printm("necessary_coalition_size_CM =",
                 necessary_coalition_size_CM)
        MyLog.printm("sufficient_coalition_size_CM =",
                 sufficient_coalition_size_CM)

        self._log_depth = old_log_depth


def neginf_to_nan(variable):
    """Convert -inf to NaN.

    Arguments:
    variable -- Number or array.

    Returns:
    variable -- Number of array.

    If variable = -inf, then variable is converted to NaN.
    If variable is a numpy array, it is converted element-wise and IN
    PLACE (the original array is modified).
    """
    if isinstance(variable, np.ndarray):
        variable[np.isneginf(variable)] = np.nan
    elif np.isneginf(variable):
        variable = np.nan
    return variable


def neginf_to_zero(variable):
    """Convert -inf to 0 / False.

    Arguments:
    variable -- Number or array.

    Returns:
    variable -- Number of array.

    If variable = -inf, then variable is converted to 0.
    If variable is a numpy array, it is converted element-wise and IN
    PLACE (the original array is modified).
    """
    if isinstance(variable, np.ndarray):
        variable[np.isneginf(variable)] = 0
    elif np.isneginf(variable):
        variable = 0
    return variable


def compute_next_subset_with_w(prev_subset, C, C_r, w):
    """Compute the next subset containing w, by lexicographic order.

    This function is internally used to compute Independence of Irrelevant
    Alternatives (IIA).

    Arguments:
    prev_subset -- 1d array of integers. prev_subset(k) is the k-th
        candidate of the subset. Candidates must be sorted by ascending
        order. Candidate w must belong to prev_subset.
    C -- Integer. Total number of candidates.
    C_r -- Integer. Number of candidates for the subset.
    w -- Integer. A candidate whose presence is required in the subset.

    Returns:
    next_subset -- 1d array of integers. next_subset(k) is the k-th
        candidate of the subset. Candidates are sorted by ascending
        order. Candidate w belongs to next_subset.

    Examples:
    next_subset = compute_next_subset_with_w(prev_subset = [0, 2, 7, 8, 9],
                                             C=10, C_r=5, w=0)
    :>>> next_subset = [0, 3, 4, 5, 6]

    next_subset = compute_next_subset_with_w(prev_subset = [0, 2, 7, 8, 9],
                                             C=10, C_r=5, w=8)
    :>>> next_subset = [0, 3, 4, 5, 8]
    """
    # TODO: this could be rewritten with an iterator.
    max_allowed_value = C
    for index_pivot in range(C_r-1, -1, -1):
        max_allowed_value -= 1
        if max_allowed_value == w:
            max_allowed_value -= 1
        if prev_subset[index_pivot] == w:
            max_allowed_value += 1
        elif prev_subset[index_pivot] < max_allowed_value:
            break  # Found the pivot
    else:
        return None
    next_subset = prev_subset
    new_member = prev_subset[index_pivot] + 1
    for i in range(index_pivot, C_r-1):
        next_subset[i] = new_member
        new_member += 1
    next_subset[C_r-1] = max(w, new_member)
    return next_subset


def compute_next_permutation(prev_permutation, C):
    """Compute next permutation by lexicographic order.

    Arguments:
    prev_permutation -- 1d array of distinct numbers.
    C -- number of elements in prev_permutation.

    Returns:
    next_permutation -- 1d array of distinct numbers. If prev_permutation
        was the last permutation in lexicographic order (i.e. all numbers
        sorted in descending order), then next_permutation = None.

    Example:
    next_permutation = compute_next_permutation([0, 2, 1, 4, 3], 5)
    :>>> next_permutation = [0, 2, 3, 1, 4]
    """
    # TODO: this could be rewritten with an iterator.
    for i in range(C-2, -1, -1):
        if prev_permutation[i] < prev_permutation[i+1]:
            index_pivot = i
            break
    else:
        return None
    index_replacement = None
    for i in range(C-1, index_pivot, -1):
        if prev_permutation[i] > prev_permutation[index_pivot]:
            index_replacement = i
            break
    return np.concatenate((
        prev_permutation[0:index_pivot],
        [prev_permutation[index_replacement]],
        prev_permutation[C-1:index_replacement:-1],
        [prev_permutation[index_pivot]],
        prev_permutation[index_replacement-1:index_pivot:-1]
    ))


def compute_next_borda_clever(prev_permutation, prev_favorite, C):
    """Compute next vector of Borda scores in 'clever' order

    Arguments:
    prev_permutation -- 1d array of integers. Must be exactly all integers
        from 0 to C - 1.
    prev_favorite -- Integer. Index of the preferred candidate. I.e.,
        prev_permutation[prev_favorite] must be equal to C - 1.
    C -- Integer. Number of elements in prev_permutation.

    Returns:
    next_permutation -- 1d array of all integers from 0 to C - 1. Next vector
        of Borda scores in 'clever' order. If prev_permutation was the
        last permutation in 'clever' order, then next_permutation = None.
    next_favorite -- Integer (or None). New preferred candidate.

    A vector of Borda scores is seen as two elements:
    (1) A permuted list of Borda scores from 0 to C-2,
    (2) The insertion of Borda score C - 1 in this list.
    The 'clever' orders sorts first by lexicographic order on (1), then by
    the position (2) (from last position to first position).

    To find next permutation, if Borda score C - 1 can be moved one step to the
    left, we do it. Otherwise, we take the next permutation of {0, ...,
    C-2} in lexicographic order, and put Borda score C - 1 at the end.

    In 'clever' order, the first vector is [0, ..., C - 1] and the last one is
    [C - 1, ..., 0].

    When looking for manipulations (IM or UM principally), here is the
    advantage of using 'clever' order instead of lexicographic order: in
    only the C first  vectors, we try all candidates as top-ranked choice.
    In many voting systems, this accelerates finding the manipulation.

    Examples:
    next_permutation, next_favorite = compute_next_borda_clever(
            prev_permutation = [0, 1, 4, 2, 3], prev_favorite = 2, C = 5)
    :>>> next_permutation = [0, 4, 1, 2, 3], next_favorite = 1.
    next_permutation, next_favorite = compute_next_borda_clever(
            prev_permutation = [4, 0, 1, 2, 3], prev_favorite = 0, C = 5)
    :>>> next_permutation = [0, 1, 3, 2, 4], next_favorite = 4.
    """
    # TODO: this could be rewritten with an iterator.
    if prev_favorite > 0:
        next_favorite = prev_favorite - 1
        next_permutation = np.copy(prev_permutation)
        next_permutation[[prev_favorite, next_favorite]] = \
            next_permutation[[next_favorite, prev_favorite]]
    else:
        try:
            next_permutation = np.concatenate((
                compute_next_permutation(
                    prev_permutation[1:C], C - 1),
                [C - 1]
            ))
            next_favorite = C - 1
        except ValueError:
            next_permutation = None
            next_favorite = None
    return next_permutation, next_favorite

def display_pseudo_bool(value):
    """Display a pseudo-boolean.

    Arguments:
    value -- True (or 1, 1., etc.), False (or 0, 0., etc.) or np.nan.

    Return:
    True, False (as booleans) or np.nan.
    """
    if np.isnan(value):
        return np.nan
    elif value == True:
        return True
    elif value == False:
        return False
    else:
        raise ValueError("Expected boolean or np.nan and got:" + format(value))