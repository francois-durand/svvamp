# -*- coding: utf-8 -*-
"""
Created on 7 dec. 2018, 15:32
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
from svvamp.rules.rule_irv import RuleIRV
from svvamp.utils.util_cache import cached_property
from svvamp.preferences.profile import Profile
from svvamp.utils.pseudo_bool import equal_true
from svvamp.utils.misc import preferences_ut_to_matrix_duels_ut


class RuleIRVAverage(Rule):
    """Instant Runoff Voting based on Average Score (Nanson-like).

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
        >>> rule = RuleIRVAverage()(profile)
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
        [[0 1 2]
         [0 2 1]
         [1 0 2]
         [2 0 1]
         [2 1 0]]
        scores =
        [[ 2.  1.  2.]
         [ 3. inf  2.]
         [ 5. inf inf]]
        candidates_by_scores_best_to_worst
        [0 2 1]
        scores_best_to_worst
        [[ 2.  2.  1.]
         [ 3.  2. inf]
         [ 5. inf inf]]
        w = 0
        score_w = [2. 3. 5.]
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

    Notes
    -----
    At each round, all candidates with a Plurality score strictly lower than average are simultaneously eliminated.
    When all remaining candidates have the same Plurality score, the candidate with lowest index is declared the
    winner.

    This method meets MajFav criterion.

    * :meth:`is_cm_`:

        * :attr:`cm_option` = ``'fast'``: Rely on :class:`RuleIRV`'s fast algorithm. Polynomial heuristic. Can prove
          CM but unable to decide non-CM (except in rare obvious cases).
        * :attr:`cm_option` = ``'slow'``: Rely on :class:`RuleExhaustiveBallot`'s exact algorithm. Non-polynomial
          heuristic (:math:`2^{n_c}`). Quite efficient to prove CM or non-CM.
        * :attr:`cm_option` = ``'very_slow'``: Rely on :class:`RuleIRV`'s exact algorithm. Non-polynomial
          heuristic (:math:`n_c!`). Very efficient to prove CM or non-CM.
        * :attr:`cm_option` = ``'exact'``: Non-polynomial algorithm from superclass :class:`Rule`.

        Each algorithm above exploits the faster ones. For example, if :attr:`cm_option` = ``'very_slow'``,
        SVVAMP tries the fast algorithm first, then the slow one, then the 'very slow' one. As soon as it reaches
        a decision, computation stops.

    * :meth:`is_icm_`: Exact in polynomial time.
    * :meth:`is_im_`: Non-polynomial or non-exact algorithms from superclass :class:`Rule`.
    * :meth:`is_iia`: Non-polynomial or non-exact algorithms from superclass :class:`Rule`.
    * :meth:`is_tm_`: Exact in polynomial time.
    * :meth:`is_um_`: Non-polynomial or non-exact algorithms from superclass :class:`Rule`.
    """

    full_name = 'IRV-Average'
    abbreviation = 'IRVA'

    options_parameters = Rule.options_parameters.copy()
    options_parameters['cm_option'] = {'allowed': {'fast', 'slow', 'very_slow', 'exact'}, 'default': 'fast'}

    def __init__(self, **kwargs):
        super().__init__(
            with_two_candidates_reduces_to_plurality=True, is_based_on_rk=True,
            precheck_icm=True,
            log_identity="IRV_AVERAGE", **kwargs
        )
        # TODO The parameter precheck_icm = True was in the original SVVAMP version. Is is useful? It does not hurt
        # much anyway, but we could think about it.

    def __call__(self, profile):
        self.delete_cache(suffix='_')
        self.profile_ = profile
        # Grab the IRV ballot of the profile (or create it)
        irv_options = {}
        if self.cm_option == 'fast':
            irv_options['cm_option'] = 'fast'
        elif self.cm_option == 'slow':
            irv_options['cm_option'] = 'slow'
        else:  # self.cm_option in {'very_slow', 'exact'}:
            irv_options['cm_option'] = 'exact'
        self.irv_ = RuleIRV(**irv_options)(self.profile_)
        return self

    @cached_property
    def _count_ballots_(self):
        """
        Case where all the candidates have the same score:

            >>> profile = Profile(preferences_rk=[[0, 1], [1, 0]])
            >>> rule = RuleIRVAverage()(profile)
            >>> rule.w_
            0
        """
        self.mylog("Count ballots", 1)
        scores = []
        candidates_worst_to_best = []
        one_v_might_be_pivotal = False
        # score_temp[c] = plurality score
        # score_temp[c] will be inf when c is eliminated
        score_temp = np.copy(self.profile_.plurality_scores_rk).astype(float)
        score_average = self.profile_.n_v / self.profile_.n_c
        candidates = np.array(range(self.profile_.n_c))
        is_alive = np.ones(self.profile_.n_c, dtype=np.bool)
        while True:  # This is a round
            scores.append(np.copy(score_temp))
            least_score = np.min(score_temp)
            self.mylogv("scores =", scores, 2)
            self.mylogv("score_average =", score_average, 2)
            if least_score == score_average:
                # Then all candidates have the average. This also includes the case where only one candidate is
                # remaining. We eliminate them by decreasing order of index (highest indexes first).
                losing_candidates = np.where(np.isfinite(score_temp))[0]
                candidates_worst_to_best.extend(losing_candidates[::-1])
                if len(losing_candidates) > 1:
                    one_v_might_be_pivotal = True
                    self.mylog("One voter might be pivotal", 2)
                self.mylogv("losing_candidates[::-1] =", losing_candidates[::-1], 2)
                break
            # Remove all candidates with score < average. Lowest score first, highest indexes first.
            while least_score < score_average:
                if least_score + 1 >= score_average:
                    one_v_might_be_pivotal = True
                    self.mylog("One voter might be pivotal", 2)
                losing_candidates = np.where(score_temp == least_score)[0]
                candidates_worst_to_best.extend(losing_candidates[::-1])
                self.mylogv("losing_candidates[::-1] =", losing_candidates[::-1], 2)
                is_alive[losing_candidates] = False
                score_temp[losing_candidates] = np.inf
                least_score = np.min(score_temp)
            if least_score - 1 < score_average:
                one_v_might_be_pivotal = True
                self.mylog("One voter might be pivotal", 2)
            candidates_alive = candidates[is_alive]
            score_temp = np.bincount(
                candidates_alive[np.argmax(self.profile_.preferences_borda_rk[:, is_alive], 1)],
                minlength=self.profile_.n_c
            ).astype(float)
            score_temp[np.logical_not(is_alive)] = np.inf
            score_average = self.profile_.n_v / np.sum(is_alive)
        candidates_by_scores_best_to_worst = np.array(candidates_worst_to_best[::-1])
        w = candidates_by_scores_best_to_worst[0]
        scores = np.array(scores)
        return {'scores': scores, 'w': w, 'one_v_might_be_pivotal': one_v_might_be_pivotal,
                'candidates_by_scores_best_to_worst': candidates_by_scores_best_to_worst}

    @cached_property
    def scores_(self):
        """2d array of integers. ``scores[r, c]`` is the Plurality score of candidate ``c`` at elimination round ``r``.
        By convention, if candidate ``c`` does not participate to round ``r``, then ``scores[r, c] = numpy.inf``.
        """
        return self._count_ballots_['scores']

    @cached_property
    def w_(self):
        self.mylog("Compute w", 1)
        plurality_elimination_engine = self.profile_.plurality_elimination_engine()
        while True:
            # Result of Plurality voting
            scores = plurality_elimination_engine.scores
            self.mylogv("scores =", scores, 3)
            # Does someone win immediately?
            best_candidate = np.nanargmax(scores)
            best_score = scores[best_candidate]
            if best_score > self.profile_.n_v / 2:
                return best_candidate
            # Who gets eliminated?
            score_average = self.profile_.n_v / plurality_elimination_engine.nb_candidates_alive
            least_score = np.nanmin(scores)
            if score_average == least_score:
                # TODO: Eliminate only one, then iterate?
                return np.where(scores == score_average)[0][0]
            losers = np.where(scores < score_average)[0]
            self.mylogv("losers =", losers, 3)
            # Prepare the next round
            if plurality_elimination_engine.nb_candidates_alive - len(losers) == 1:
                for loser in losers:
                    plurality_elimination_engine.eliminate_candidate(loser)
                return plurality_elimination_engine.candidates_alive[0]
            for loser in losers:
                plurality_elimination_engine.eliminate_candidate_and_update_scores(loser)

    @cached_property
    def candidates_by_scores_best_to_worst_(self):
        """1d array of integers. Candidates are sorted according to their order of elimination. When several
        candidates are eliminated during the same round, they are sorted by Plurality score at that round (less votes
        are eliminated first) and, in case of a tie, by their index (highest indexes are eliminated first).
        """
        return self._count_ballots_['candidates_by_scores_best_to_worst']

    @cached_property
    def _one_v_might_be_pivotal_(self):
        return self._count_ballots_['one_v_might_be_pivotal']

    @cached_property
    def v_might_im_for_c_(self):
        return np.full((self.profile_.n_v, self.profile_.n_c), self._one_v_might_be_pivotal_)

    @cached_property
    def meets_majority_favorite_c_rk_ctb(self):
        return True

    # %% Unison manipulation (UM)

    def _um_preliminary_checks_c_(self, c):
        if self.um_option not in {'fast', 'lazy'} or self.cm_option not in {'fast', 'lazy'}:
            if (
                self.w_ == self.profile_.condorcet_winner_rk_ctb
                and not self.profile_.c_might_be_there_when_cw_is_eliminated_irv_style[c]
            ):
                # Impossible to manipulate with n_m manipulators
                self._candidates_um[c] = False

    # %% Coalition Manipulation (CM)

    def _cm_preliminary_checks_c_subclass_(self, c, optimize_bounds):
        """CM: preliminary checks for challenger ``c``.

        Let us consider the case where `w` is the Condorcet winner (rk ctb) (otherwise it is easy to manipulate, at
        least if utility preferences are strict). Then at some point, `w` must have a plurality score under average, at
        a moment where `c` is still present in the election.
        """
        if self.cm_option not in {'fast', 'lazy'}:
            if self.w_ == self.profile_.condorcet_winner_rk_ctb:
                self._update_necessary(
                    self._necessary_coalition_size_cm, c,
                    self.profile_.necessary_coalition_size_to_break_irv_immunity[c],
                    'CM: Preliminary checks: IRV-Immunity => \n    necessary_coalition_size_cm[c] = '
                )

    def _cm_aux_(self, c, ballots_m, preferences_rk_s):
        profile_test = Profile(preferences_rk=np.concatenate((preferences_rk_s, ballots_m)), sort_voters=False)
        if profile_test.n_v != self.profile_.n_v:
            raise AssertionError('Uh-oh!')
        winner_test = self.__class__()(profile_test).w_
        return winner_test == c

    def _cm_main_work_c_(self, c, optimize_bounds):
        n_m = self.profile_.matrix_duels_ut[c, self.w_]
        n_s = self.profile_.n_v - n_m
        candidates = np.array(range(self.profile_.n_c))
        preferences_borda_s = self.profile_.preferences_borda_rk[np.logical_not(self.v_wants_to_help_c_[:, c]), :]
        preferences_rk_s = self.profile_.preferences_rk[np.logical_not(self.v_wants_to_help_c_[:, c]), :]
        matrix_duels_vtb_temp = (preferences_ut_to_matrix_duels_ut(preferences_borda_s))
        self.mylogm("CM: matrix_duels_vtb_temp =", matrix_duels_vtb_temp, 3)
        # More preliminary checks. It's more convenient to put them in that method, because we need
        # ``preferences_borda_s`` and ``matrix_duels_vtb_temp``.
        d_neq_c = (np.array(range(self.profile_.n_c)) != c)
        # Prevent another cond. Look at the weakest duel for ``d``, she has ``matrix_duels_vtb_temp[d, e]``. We simply
        # need that:
        # ``matrix_duels_vtb_temp[d, e] <= (n_s + n_m) / 2``
        # ``2 * max_d(min_e(matrix_duels_vtb_temp[d, e])) - n_s <= n_m``
        n_manip_prevent_cond = 0
        for d in candidates[d_neq_c]:
            e_neq_d = (np.array(range(self.profile_.n_c)) != d)
            n_prevent_d = np.maximum(2 * np.min(matrix_duels_vtb_temp[d, e_neq_d]) - n_s, 0)
            n_manip_prevent_cond = max(n_manip_prevent_cond, n_prevent_d)
        self.mylogv("CM: n_manip_prevent_cond =", n_manip_prevent_cond, 3)
        self._update_necessary(self._necessary_coalition_size_cm, c, n_manip_prevent_cond,
                               'CM: Update necessary_coalition_size_cm[c] = n_manip_prevent_cond =')
        if not optimize_bounds and self._necessary_coalition_size_cm[c] > n_m:
            return True

        # Let us work
        if self.w_ == self.irv_.w_:
            self.mylog('CM: c != self.irv_.w == self.w', 3)
            if self.cm_option == "fast":
                self.irv_.cm_option = "fast"
            elif self.cm_option == "slow":
                self.irv_.cm_option = "slow"
            else:
                self.irv_.cm_option = "exact"
            irv_is_cm_c = self.irv_.is_cm_c_(c)
            if equal_true(irv_is_cm_c):
                # Use IRV without bounds
                self.mylog('CM: Use IRV without bounds')
                suggested_path_one = self.irv_.example_path_cm_c_(c)
                self.mylogv("CM: suggested_path =", suggested_path_one, 3)
                ballots_m = self.irv_.example_ballots_cm_c_(c)
                manipulation_found = self._cm_aux_(c, ballots_m, preferences_rk_s)
                self.mylogv("CM: manipulation_found =", manipulation_found, 3)
                if manipulation_found:
                    self._update_sufficient(self._sufficient_coalition_size_cm, c, n_m,
                                            'CM: Update sufficient_coalition_size_cm[c] = n_m =')
                    # We will not do better with any algorithm (even the brute force algo).
                    return False
                # Use IRV with bounds
                self.irv_.is_cm_c_with_bounds_(c)
                self.mylog('CM: Use IRV with bounds')
                suggested_path_two = self.irv_.example_path_cm_c_(c)
                self.mylogv("CM: suggested_path =", suggested_path_two, 3)
                if np.array_equal(suggested_path_one, suggested_path_two):
                    self.mylog('CM: Same suggested path as before, skip computation')
                else:
                    ballots_m = self.irv_.example_ballots_cm_c_(c)
                    manipulation_found = self._cm_aux_(c, ballots_m, preferences_rk_s)
                    self.mylogv("CM: manipulation_found =", manipulation_found, 3)
                    if manipulation_found:
                        self._update_sufficient(self._sufficient_coalition_size_cm, c, n_m,
                                                'CM: Update sufficient_coalition_size_cm[c] = n_m =')
                        # We will not do better with any algorithm (even the brute force algo).
                        return False
        else:  # self.w_ != self.irv_.w_:
            if c == self.irv_.w_:
                self.mylog('CM: c == self.irv_.w != self._w', 3)
                suggested_path = self.irv_.elimination_path_
                self.mylogv("CM: suggested_path =", suggested_path, 3)
                ballots_m = self.irv_.example_ballots_cm_w_against_(w_other_rule=self.w_)
                manipulation_found = self._cm_aux_(c, ballots_m, preferences_rk_s)
                self.mylogv("CM: manipulation_found =", manipulation_found, 3)
                if manipulation_found:
                    self._update_sufficient(self._sufficient_coalition_size_cm, c, n_m,
                                            'CM: Update sufficient_coalition_size_cm[c] = n_m =')
                    # We will not do better with any algorithm (even the brute force algo).
                    return
            else:
                self.mylog('CM: c, self.irv_.w_ and self.w_ are all distinct', 3)
        if self.cm_option == 'exact':
            return self._cm_main_work_c_exact_(c, optimize_bounds)
