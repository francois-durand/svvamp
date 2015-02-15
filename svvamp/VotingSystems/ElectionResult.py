# -*- coding: utf-8 -*-
"""
Created on Fri Sep 26 09:51:31 2014
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

from svvamp.Utils import MyLog


class ElectionResult(MyLog.MyLog):

    # _options_parameters is a dictionary of allowed and default options.
    # Allowed is a minimal check that will be performed before launching big
    # simulations, but other checks might be performed when setting the
    # option. Allowed is either a list of values, like ['lazy', 'fast',
    # 'exact'], or a function checking if the parameter is correct.
    # Example: {'max_grade': dict(allowed=np.isfinite, default=1),
    #           'option_example': dict(allowed=[42, 51], default=42)}
    _options_parameters = {}

    @property
    def options_parameters(self):
        """Display options.

        Display a overview of available options, their default values and a
        minimal indication about what values are allowed for each option. For
        more details about a specific option, see its documentation.
        """
        return self._options_parameters

    def __init__(self, population, **kwargs):
        """Create a simple election (without manipulation).

        This is an 'abstract' class. As an end-user, you should always use its
        subclasses Approval, Plurality, etc.

        :param population: A :class:`~svvamp.Population` object.
        :param kwargs: additional keyword parameters.
        """
        super().__init__()
        self._log_identity = "ELECTION_RESULT"

        # These lines are useless and should pose problems for Exhaustive
        # ballot
        # TODO: remove these lines after some tests
        # self._pop = None
        # self._scores = None
        # self._ballots = None
        # self._w = None
        # self._candidates_by_scores_best_to_worst = None
        # self._score_w = None
        # self._scores_best_to_worst = None

        self.pop = population
        self._initialize_options(**kwargs)

    def _initialize_options(self, **kwargs):
        """Initialize options.

        Arguments: option1=value1, option2=value2, etc.
        Checks iff option1, option2 are known in self.options_parameters.
        For each option in self.options_parameters:
        * If a value is given, option is set to this value.
        * Otherwise, the default value is used.
        N.B.: validity check on the value are not performed here, but in the
        setter method for the option.
        """
        self._mylog('Initialize options', 1)
        for option in kwargs:
            if option not in self.options_parameters.keys():
                raise ValueError('Unknown option:', option)
        for option in self.options_parameters:
            if option in kwargs.keys():
                setattr(self, option, kwargs[option])
            else:
                setattr(self, option,
                        self.options_parameters[option]['default'])

    def _forget_all_computations(self):
        """Initialize / forget all computations

        Typically used when the population is modified: all results of the
        election are initialized then.
        In the subclass Election, this will also initialize all manipulation
        computations.
        """
        self._forget_results()

    def _forget_results(self):
        """Initialize / forget election results

        This concerns only the results of the election, not the manipulation
        computations.
        """
        self._scores = None
        self._ballots = None
        self._w = None
        self._candidates_by_scores_best_to_worst = None
        self._score_w = None
        self._scores_best_to_worst = None
        self._forget_results_subclass()

    def _forget_results_subclass(self):
        """Initialize / forget election results, specific to a voting system

        If a specific voting system has additional variables about the result
        of the election, they can be initialized here.
        """
        pass
        
    #%% The population

    @property
    def pop(self):
        """A :class:`~svvamp.Population object`. The population running the
        election.
        """
        return self._pop
        
    @pop.setter
    def pop(self, value):
        self._pop = value
        # We forget everything: results of an election. In the subclass
        # Election, we will also forget manipulation computations.
        self._forget_all_computations()

    #%% Counting ballots
    #   Attributes to be implemented in subclasses.

    @property
    def scores(self):
        """1d or 2d array (generally). See specific documentation for each
        voting system.
        """
        raise NotImplementedError

    #%% Counting ballots
    #   Attributes with default, but frequently redefined in subclasses

    @property
    def ballots(self):
        """2d array of integers (generally).

        Default behavior is:
        ``ballots[v, k]`` =
        :attr:`~svvamp.Population.preferences_ranking``\ `[v, k]``.

        This can be overridden by specific voting systems.
        """
        # This general method is ok only for ordinal voting systems (and
        # even in this case, it can be redefined for something more practical).
        if self._ballots is None:
            self._mylog("Compute ballots", 1)
            self._ballots = self.pop.preferences_ranking
        return self._ballots

    @property
    def w(self):
        """Integer (winning candidate).

        Default behavior: the candidate with highest score is declared the
        winner. In case of a tie, the candidate with lowest index wins. This
        can be overridden by specific voting systems.
        """
        # This general method works only if scores are scalar and the best
        # score wins.
        if self._w is None:
            self._mylog("Compute winner", 1)
            self._w = np.argmax(self.scores)
        return self._w

    @property
    def candidates_by_scores_best_to_worst(self):
        """1d array of integers (generally).

        Default behavior: ``candidates_by_scores_best_to_worst[k]`` is the
        candidate with ``k``\ :sup:`th` highest score.
        
        By definition, ``candidates_by_scores_best_to_worst[0] = w``.

        This can be overridden by specific voting systems.
        """
        # This general method works only if scores are scalar and the best
        # score wins. If the lowest score wins, then
        # candidates_by_scores_best_to_worst need to be sorted by ascending
        # score...
        if self._candidates_by_scores_best_to_worst is None:
            self._mylog("Compute candidates_by_scores_best_to_worst", 1)
            self._candidates_by_scores_best_to_worst = np.argsort(
                -self.scores, kind='mergesort')
        return self._candidates_by_scores_best_to_worst     

    @property
    def _v_might_IM_for_c(self):
        """2d array of booleans. _v_might_IM_for_c[v, c] is True unless it is
        clearly and easily excluded that v has an individual manipulation in
        favor of c. Typically, if v is not pivotal (at all), then it is
        False. Specific implementations of this function in subclasses can
        also test if v is interested in manipulating for c, but it is not
        mandatory.

        A non-trivial redefinition of this function is useful for voting
        systems where computing IM is costly. For voting systems where it is
        cheap, it it not worth the effort.
        """
        return np.ones((self.pop.V, self.pop.C), dtype=np.bool)

    #%% Counting ballots
    #   Attributes that are almost always computed here in the superclass.
        
    @property
    def score_w(self):
        """If :attr:`~svvamp.ElectionResult.scores` is a 1d array,
        then ``score_w`` is a number: it is the score of the sincere winner
        :attr:`~svvamp.ElectionResult.w`. I.e. ``score_w = scores[w]``.
        
        If :attr:`~svvamp.ElectionResult.scores` is a 2d array, then
        ``score_w`` is :attr:`~svvamp.ElectionResult.w`'s score vector.
        I.e. ``score_w = scores[:, w]``.

        This can be overridden by specific voting systems (for example,
        Ranked Pairs).
        """
        # Exception: if scores are read in rows (Schulze, Ranked pairs), this
        # needs to be redefined.
        if self._score_w is None:
            self._mylog("Compute winner's score", 1)
            if self.scores.ndim == 1:
                self._score_w = self.scores[self.w]
            elif self.scores.ndim == 2:
                self._score_w = self.scores[:, self.w]
            else:
                raise ValueError(
                    "'scores' is usually a 1d or 2d array. If you really "
                    "want it to be something else, you need to redefine "
                    "'score_w'.")
        return self._score_w

    @property 
    def scores_best_to_worst(self):
        """``scores_best_to_worst`` is the scores of the candidates, from the
        winner to the last candidate of the election.
        
        If :attr:`~svvamp.ElectionResult.scores` is a 1d array, then so is
        ``scores_best_to_worst``. For most voting systems, scalar scores are
        sorted by descending order. By definition,
        ``scores_best_to_worst[0]`` = :attr:`~svvamp.ElectionResult.score_w`.
        
        If :attr:`~svvamp.ElectionResult.scores` is a 2d array, then so is
        ``scores_best_to_worst``.
        ``scores_best_to_worst[:, k]`` is the score vector of the
        ``k``\ :sup:`th` best candidate of the election. By definition,
        ``scores_best_to_worst[:, 0]`` =
        :attr:`~svvamp.ElectionResult.score_w`.

        This can be overridden by specific voting systems (for example,
        Ranked Pairs).
        """
        # Exception: if scores are read in rows (Schulze, Ranked pairs), this
        # needs to be redefined.
        if self._scores_best_to_worst is None:
            self._mylog("Compute scores_best_to_worst", 1)
            if self.scores.ndim == 1:
                self._scores_best_to_worst = self.scores[
                    self.candidates_by_scores_best_to_worst]
            elif self.scores.ndim == 2:
                self._scores_best_to_worst = self.scores[
                    :, self.candidates_by_scores_best_to_worst]
            else:
                raise ValueError("'scores' is usually a 1d or 2d " +
                                 "array. If you really want it to be " +
                                 "something else, you need to redefine " +
                                 "'scores_best_to_worst'.")
        return self._scores_best_to_worst

    @property
    def total_utility_w(self):
        """Float. The total utility for the sincere winner
        :attr:`~svvamp.ElectionResult.w`. Be careful, this
        makes sense only if interpersonal comparison of utilities make sense.
        """
        return self.pop.total_utility_c[self.w]

    @property
    def mean_utility_w(self):
        """Float. The mean utility for the sincere winner
        :attr:`~svvamp.ElectionResult.w`. Be careful, this
        makes sense only if interpersonal comparison of utilities make sense.
        """
        return self.pop.mean_utility_c[self.w]

    #%% Condorcet efficiency and variants
        
    @property
    def w_is_condorcet_admissible(self):
        """Boolean. ``True`` iff the sincere winner
        :attr:`~svvamp.ElectionResult.w` is Condorcet-admissible.
        Cf. :attr:`svvamp.Population.condorcet_admissible_candidates`.
        """
        return self.pop.condorcet_admissible_candidates[self.w]

    @property
    def w_is_not_condorcet_admissible(self):
        """Boolean. ``True`` iff the sincere winner
        :attr:`~svvamp.ElectionResult.w` is not a Condorcet-admissible
        candidate (whether some exist or not).
        Cf. :attr:`svvamp.Population.condorcet_admissible_candidates`.
        """
        return not self.w_is_condorcet_admissible

    @property
    def w_missed_condorcet_admissible(self):
        """Boolean. True iff the sincere winner
        :attr:`~svvamp.ElectionResult.w` is not a
        Condorcet-admissible candidate, despite the fact that at least one
        exists.
        Cf. :attr:`svvamp.Population.condorcet_admissible_candidates`.
        """
        return (self.pop.nb_condorcet_admissible > 0 and
                not self.w_is_condorcet_admissible)

    @property
    def w_is_weak_condorcet_winner(self):
        """Boolean. True iff the sincere winner
        :attr:`~svvamp.ElectionResult.w` is a Weak Condorcet winner.
        Cf. :attr:`svvamp.Population.weak_condorcet_winners`.
        """
        return self.pop.weak_condorcet_winners[self.w]

    @property
    def w_is_not_weak_condorcet_winner(self):
        """Boolean. True iff the sincere winner
        :attr:`~svvamp.ElectionResult.w` is not a Weak Condorcet
        winner (whether some exist or not).
        Cf. :attr:`svvamp.Population.weak_condorcet_winners`.
        """
        return not self.w_is_weak_condorcet_winner

    @property
    def w_missed_weak_condorcet_winner(self):
        """Boolean. True iff the sincere winner
        :attr:`~svvamp.ElectionResult.w` is not a Weak Condorcet
        winner, despite the fact that at least one exists.
        Cf. :attr:`svvamp.Population.weak_condorcet_winners`.
        """
        return (self.pop.nb_weak_condorcet_winners > 0 and 
                not self.w_is_weak_condorcet_winner)

    @property
    def w_is_condorcet_winner_vtb_ctb(self):
        """Boolean. True iff the sincere winner
        :attr:`~svvamp.ElectionResult.w` is a 'Condorcet winner
        with vtb and ctb'.
        Cf. :attr:`svvamp.Population.condorcet_winner_vtb_ctb`.
        """
        return self.w == self.pop.condorcet_winner_vtb_ctb

    @property
    def w_is_not_condorcet_winner_vtb_ctb(self):
        """Boolean. True iff the sincere winner
        :attr:`~svvamp.ElectionResult.w` is not a 'Condorcet winner
        with vtb and ctb' (whether one exists or not).
        Cf. :attr:`svvamp.Population.condorcet_winner_vtb_ctb`.
        """
        return not self.w_is_condorcet_winner_vtb_ctb

    @property
    def w_missed_condorcet_winner_vtb_ctb(self):
        """Boolean. True iff the sincere winner
        :attr:`~svvamp.ElectionResult.w` is not the 'Condorcet winner
        with vtb and ctb', despite the fact that she exists.
        Cf. :attr:`svvamp.Population.condorcet_winner_vtb_ctb`.
        """
        return (not np.isnan(self.pop.condorcet_winner_vtb_ctb) and
                not self.w_is_condorcet_winner_vtb_ctb)

    @property
    def w_is_condorcet_winner_vtb(self):
        """Boolean. True iff the sincere winner
        :attr:`~svvamp.ElectionResult.w` is a 'Condorcet winner with
        vtb'.
        Cf. :attr:`svvamp.Population.condorcet_winner_vtb`.
        """
        return self.w == self.pop.condorcet_winner_vtb

    @property
    def w_is_not_condorcet_winner_vtb(self):
        """Boolean. True iff the sincere winner
        :attr:`~svvamp.ElectionResult.w` is not a 'Condorcet winner
        with vtb' (whether one exists or not).
        Cf. :attr:`svvamp.Population.condorcet_winner_vtb`.
        """
        return not self.w_is_condorcet_winner_vtb

    @property
    def w_missed_condorcet_winner_vtb(self):
        """Boolean. True iff the sincere winner
        :attr:`~svvamp.ElectionResult.w` is not the 'Condorcet winner
        with vtb', despite the fact that she exists.
        Cf. :attr:`svvamp.Population.condorcet_winner_vtb`.
        """
        return (not np.isnan(self.pop.condorcet_winner_vtb) and
                not self.w_is_condorcet_winner_vtb)

    @property
    def w_is_condorcet_winner_rel_ctb(self):
        """Boolean. True iff the sincere winner
        :attr:`~svvamp.ElectionResult.w` is a 'relative Condorcet
        winner with ties broken'.
        Cf. :attr:`svvamp.Population.condorcet_winner_ctb`.
        """
        return self.w == self.pop.condorcet_winner_rel_ctb

    @property
    def w_is_not_condorcet_winner_rel_ctb(self):
        """Boolean. True iff the sincere winner
        :attr:`~svvamp.ElectionResult.w` is not a 'relative
        Condorcet winner with ties broken' (whether one exists or not).
        Cf. :attr:`svvamp.Population.condorcet_winner_ctb`.
        """
        return not self.w_is_condorcet_winner_rel_ctb

    @property
    def w_missed_condorcet_winner_rel_ctb(self):
        """Boolean. True iff the sincere winner
        :attr:`~svvamp.ElectionResult.w` is not the 'relative
        Condorcet winner with ties broken', despite the fact that she exists.
        Cf. :attr:`svvamp.Population.condorcet_winner_ctb`.
        """
        return (not np.isnan(self.pop.condorcet_winner_rel_ctb) and
                not self.w_is_condorcet_winner_rel_ctb)

    @property
    def w_is_condorcet_winner_rel(self):
        """Boolean. True iff the sincere winner
        :attr:`~svvamp.ElectionResult.w` is a relative Condorcet
        winner.
        Cf. :attr:`svvamp.Population.condorcet_winner_rel`.
        """
        return self.w == self.pop.condorcet_winner_rel

    @property
    def w_is_not_condorcet_winner_rel(self):
        """Boolean. True iff the sincere winner
        :attr:`~svvamp.ElectionResult.w` is not a relative Condorcet
        winner (whether one exists or not).
        Cf. :attr:`svvamp.Population.condorcet_winner_rel`.
        """
        return not self.w_is_condorcet_winner_rel

    @property
    def w_missed_condorcet_winner_rel(self):
        """Boolean. True iff the sincere winner
        :attr:`~svvamp.ElectionResult.w` is not the relative
        Condorcet winner, despite the fact that she exists.
        Cf. :attr:`svvamp.Population.condorcet_winner_rel`.
        """
        return (not np.isnan(self.pop.condorcet_winner_rel) and
                not self.w_is_condorcet_winner_rel)

    @property
    def w_is_condorcet_winner_ctb(self):
        """Boolean. True iff the sincere winner
        :attr:`~svvamp.ElectionResult.w` is a 'Condorcet winner with
        ties broken'.
        Cf. :attr:`svvamp.Population.condorcet_winner_ctb`.
        """
        return self.w == self.pop.condorcet_winner_ctb

    @property
    def w_is_not_condorcet_winner_ctb(self):
        """Boolean. True iff the sincere winner
        :attr:`~svvamp.ElectionResult.w` is not a 'Condorcet winner
        with ties broken' (whether one exists or not).
        Cf. :attr:`svvamp.Population.condorcet_winner_ctb`.
        """
        return not self.w_is_condorcet_winner_ctb

    @property
    def w_missed_condorcet_winner_ctb(self):
        """Boolean. True iff the sincere winner
        :attr:`~svvamp.ElectionResult.w` is not the 'Condorcet winner
        with ties broken', despite the fact that she exists.
        Cf. :attr:`svvamp.Population.condorcet_winner_ctb`.
        """
        return (not np.isnan(self.pop.condorcet_winner_ctb) and
                not self.w_is_condorcet_winner_ctb)
                
    @property
    def w_is_condorcet_winner(self):
        """Boolean. True iff the sincere winner
        :attr:`~svvamp.ElectionResult.w` is a Condorcet winner.
        Cf. :attr:`svvamp.Population.condorcet_winner`.
        """
        return self.w == self.pop.condorcet_winner

    @property
    def w_is_not_condorcet_winner(self):
        """Boolean. True iff the sincere winner
        :attr:`~svvamp.ElectionResult.w` is not a Condorcet winner
        (whether one exists or not).
        Cf. :attr:`svvamp.Population.condorcet_winner`.
        """
        return not self.w_is_condorcet_winner

    @property
    def w_missed_condorcet_winner(self):
        """Boolean. True iff the sincere winner
        :attr:`~svvamp.ElectionResult.w` is not the Condorcet winner,
        despite the fact that she exists.
        Cf. :attr:`svvamp.Population.condorcet_winner`.
        """
        return (not np.isnan(self.pop.condorcet_winner) and
                not self.w_is_condorcet_winner)
                
    @property
    def w_is_resistant_condorcet_winner(self):
        """Boolean. True iff the sincere winner
        :attr:`~svvamp.ElectionResult.w` is a Resistant Condorcet
        winner.
        Cf. :attr:`svvamp.Population.resistant_condorcet_winner`.
        """
        return self.w == self.pop.resistant_condorcet_winner

    @property
    def w_is_not_resistant_condorcet_winner(self):
        """Boolean. True iff the sincere winner
        :attr:`~svvamp.ElectionResult.w` is not a Resistant Condorcet
        winner (whether one exists or not).
        Cf. :attr:`svvamp.Population.resistant_condorcet_winner`.
        """
        return not self.w_is_resistant_condorcet_winner

    @property
    def w_missed_resistant_condorcet_winner(self):
        """Boolean. True iff the sincere winner
        :attr:`~svvamp.ElectionResult.w` is not the Resistant
        Condorcet winner, despite the fact that she exists.
        Cf. :attr:`svvamp.Population.resistant_condorcet_winner`.
        """
        return (not np.isnan(self.pop.resistant_condorcet_winner) and
                not self.w_is_resistant_condorcet_winner)

    def demo(self, log_depth):
        """Demonstrate the methods of :class:`~svvamp.ElectionResult` class.

        :param log_depth: Integer from 0 (basic info) to 3 (verbose).
        """

        old_log_depth = self._log_depth
        self._log_depth = log_depth

        MyLog.print_big_title('Population Class')
        self.pop.demo()

        MyLog.print_big_title('ElectionResult Class')

        MyLog.print_title("Results")
        MyLog.printm("pop.preferences_utilities (reminder) =",
                     self.pop.preferences_utilities)
        # MyLog.printm("pop.preferences_borda_novtb =",
        #              self.pop.preferences_borda_novtb)
        # MyLog.printm("pop.preferences_borda_vtb =",
        #              self.pop.preferences_borda_vtb)
        MyLog.printm("pop.preferences_ranking (reminder) =",
                     self.pop.preferences_ranking)
        MyLog.printm("ballots =", self.ballots)
        MyLog.printm("scores =", self.scores)
        MyLog.printm("candidates_by_scores_best_to_worst",
                 self.candidates_by_scores_best_to_worst)
        MyLog.printm("scores_best_to_worst", self.scores_best_to_worst)
        print("w =", self.w)
        print("score_w =", self.score_w)
        print("total_utility_w =", self.total_utility_w)

        MyLog.print_title("Condorcet efficiency (vtb)")
        print("w (reminder) =", self.w)
        print("")

        print("condorcet_winner_vtb_ctb =",
              self.pop.condorcet_winner_vtb_ctb)
        print("w_is_condorcet_winner_vtb_ctb =",
              self.w_is_condorcet_winner_vtb_ctb)
        print("w_is_not_condorcet_winner_vtb_ctb =",
              self.w_is_not_condorcet_winner_vtb_ctb)
        print("w_missed_condorcet_winner_vtb_ctb =",
              self.w_missed_condorcet_winner_vtb_ctb)
        print("")

        print("condorcet_winner_vtb =",
              self.pop.condorcet_winner_vtb)
        print("w_is_condorcet_winner_vtb =",
              self.w_is_condorcet_winner_vtb)
        print("w_is_not_condorcet_winner_vtb =",
              self.w_is_not_condorcet_winner_vtb)
        print("w_missed_condorcet_winner_vtb =",
              self.w_missed_condorcet_winner_vtb)

        MyLog.print_title("Condorcet efficiency (relative)")
        print("w (reminder) =", self.w)
        print("")

        print("condorcet_winner_rel_ctb =",
              self.pop.condorcet_winner_rel_ctb)
        print("w_is_condorcet_winner_rel_ctb =",
              self.w_is_condorcet_winner_rel_ctb)
        print("w_is_not_condorcet_winner_rel_ctb =",
              self.w_is_not_condorcet_winner_rel_ctb)
        print("w_missed_condorcet_winner_rel_ctb =",
              self.w_missed_condorcet_winner_rel_ctb)
        print("")

        print("condorcet_winner_rel =",
              self.pop.condorcet_winner_rel)
        print("w_is_condorcet_winner_rel =",
              self.w_is_condorcet_winner_rel)
        print("w_is_not_condorcet_winner_rel =",
              self.w_is_not_condorcet_winner_rel)
        print("w_missed_condorcet_winner_rel =",
              self.w_missed_condorcet_winner_rel)

        MyLog.print_title("Condorcet efficiency (absolute)")
        print("w (reminder) =", self.w)
        print("")

        MyLog.printm("condorcet_admissible_candidates =",
              self.pop.condorcet_admissible_candidates)
        print("w_is_condorcet_admissible =",
              self.w_is_condorcet_admissible)
        print("w_is_not_condorcet_admissible =",
              self.w_is_not_condorcet_admissible)
        print("w_missed_condorcet_admissible =",
              self.w_missed_condorcet_admissible)
        print("")

        MyLog.printm("weak_condorcet_winners =",
                     self.pop.weak_condorcet_winners)
        print("w_is_weak_condorcet_winner =",
              self.w_is_weak_condorcet_winner)
        print("w_is_not_weak_condorcet_winner =",
              self.w_is_not_weak_condorcet_winner)
        print("w_missed_weak_condorcet_winner =",
              self.w_missed_weak_condorcet_winner)
        print("")

        print("condorcet_winner_ctb =",
              self.pop.condorcet_winner_ctb)
        print("w_is_condorcet_winner_ctb =",
              self.w_is_condorcet_winner_ctb)
        print("w_is_not_condorcet_winner_ctb =",
              self.w_is_not_condorcet_winner_ctb)
        print("w_missed_condorcet_winner_ctb =",
              self.w_missed_condorcet_winner_ctb)
        print("")

        print("condorcet_winner =",
              self.pop.condorcet_winner)
        print("w_is_condorcet_winner =",
              self.w_is_condorcet_winner)
        print("w_is_not_condorcet_winner =",
              self.w_is_not_condorcet_winner)
        print("w_missed_condorcet_winner =",
              self.w_missed_condorcet_winner)
        print("")

        print("resistant_condorcet_winner =",
              self.pop.resistant_condorcet_winner)
        print("w_is_resistant_condorcet_winner =",
              self.w_is_resistant_condorcet_winner)
        print("w_is_not_resistant_condorcet_winner =",
              self.w_is_not_resistant_condorcet_winner)
        print("w_missed_resistant_condorcet_winner =",
              self.w_missed_resistant_condorcet_winner)

        self._log_depth = old_log_depth
