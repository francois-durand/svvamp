# -*- coding: utf-8 -*-
"""
Created on Sat Oct  4 16:51:15 2014
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
import networkx as nx

from svvamp.VotingSystems.Election import compute_next_permutation
from svvamp.VotingSystems.ElectionResult import ElectionResult

# N.B.: In some distributions, 'condensation' from networkx does not meet
# its specification. Comment the adequate line below.
# import networkx as sc  # Faster but may not work
import svvamp.VotingSystems.StrongComponents as sc  # Slower but always works
from svvamp.Preferences.Population import Population


class KemenyResult(ElectionResult):
    """Results of an election using Kemeny method.

    We find the order on candidates whose total Kendall distance to the voters
    is minimal. The top element of this order is declared the winner.

    In case several orders are optimal, the first one by lexicographic
    order is given. This implies that if several winners are possible,
    the one with lowest index is declared the winner.
    """

    _options_parameters = ElectionResult._options_parameters.copy()

    def __init__(self, population, **kwargs):
        super().__init__(population, **kwargs)
        self._log_identity = "KEMENY_RESULT"

        self.G = nx.DiGraph(self.pop.matrix_victories_rk > 0)
        self.condensed = sc.condensation(self.G)
        self.sorted_condensed = nx.topological_sort(self.condensed)

    def _forget_results_subclass(self):
        self.G = nx.DiGraph(self.pop.matrix_victories_rk > 0)
        self.condensed = sc.condensation(self.G)
        self.sorted_condensed = nx.topological_sort(self.condensed)

    @property
    def score_w(self):
        """Integer. With our convention, ``scores_w[r]`` =
        :attr:`~svvamp.Population.C` - 1.
        """
        if self._score_w is None:
            self._mylog("Compute winner's score", 1)
            self._score_w = self.pop.C - 1
        return self._score_w

    @property 
    def scores_best_to_worst(self):
        """1d array of integers. With our convention, ``scores_best_to_worst``
        is the vector [:attr:`~svvamp.Population.C` - 1,
        :attr:`~svvamp.Population.C` - 2, ..., 0].
        """
        if self._scores_best_to_worst is None:
            self._mylog("Compute scores_best_to_worst", 1)
            self._scores_best_to_worst = np.array(
                range(self.pop.C - 1, -1, -1))
        return self._scores_best_to_worst        

    def _count_first_component(self):
        """Compute the optimal order for the first strongly connected 
        component of the victory matrix (ties count as a connection).
        """
        self._mylog("Count first strongly connected component", 1)
        self._candidates_by_scores_best_to_worst = []
        component = self.sorted_condensed[0]
        order = np.sort(self.condensed.node[component]['members'])
        size_component = len(order)
        best_score = -1
        best_order = None
        while order is not None:
            self._mylogv("order =", order, 3)
            score = np.sum(
                self.pop.matrix_duels_rk[:, order][order, :][np.triu_indices(
                    size_component)])
            self._mylogv("score =", score, 3)
            if score > best_score:
                best_order = order
                best_score = score
            order = compute_next_permutation(order, size_component)
        self._mylogv("best_order =", best_order, 3)
        self._w = best_order[0]
        self._candidates_by_scores_best_to_worst.extend(best_order)

    def _count_ballots(self):
        """Compute the optimal order for all the candidates.
        """
        if self._w is None:
            self._count_first_component()
        self._mylog("Count other strongly connected components", 1)
        for component in self.sorted_condensed[1:]:
            order = np.sort(self.condensed.node[component]['members'])
            size_component = len(order)
            best_score = -1
            best_order = None
            while order is not None:
                self._mylogv("order =", order, 3)
                score = np.sum(
                    self.pop.matrix_duels_rk[:, order][order, :][
                        np.triu_indices(size_component)])
                self._mylogv("score =", score, 3)
                if score > best_score:
                    best_order = order
                    best_score = score
                order = compute_next_permutation(order, size_component)
            self._mylogv("best_order =", best_order, 3)
            self._candidates_by_scores_best_to_worst.extend(best_order)
        self._candidates_by_scores_best_to_worst = np.array(
            self._candidates_by_scores_best_to_worst)
        self._scores = self.pop.C - 1 - np.argsort(
            self._candidates_by_scores_best_to_worst)

    @property
    def w(self):
        """Integer (winning candidate).
        """
        if self._w is None:
            self._count_first_component()
        return self._w
        
    @property
    def scores(self):
        """1d array of integers. By convention, scores are integers from 1 to
        :attr:`~svvamp.Population.C`, with :attr:`~svvamp.Population.C` for
        the winner and 1 for the last candidate in Kemeny optimal order.
        """
        if self._scores is None:
            self._count_ballots()
        return self._scores
                
    @property
    def candidates_by_scores_best_to_worst(self):
        """1d array of integers. This is an optimal Kemeny order.
        
        In case several orders are optimal, the first one by lexicographic
        order is given. This implies that if several winners are possible,
        the one with lowest index is declared the winner.
        
        By definition, ``candidates_by_scores_best_to_worst[0]`` =
        :attr:`~svvamp.Kemeny.w`.
        """
        if self._scores is None:  # This is not a typo
            self._count_ballots()
        return self._candidates_by_scores_best_to_worst     


if __name__ == '__main__':
    # A quick demo
    preferences_utilities = np.random.randint(-5, 5, (10, 5))
    pop = Population(preferences_utilities)
    election = KemenyResult(pop)
    election.demo(log_depth=3)