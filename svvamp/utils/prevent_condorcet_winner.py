# -*- coding: utf-8 -*-
"""
Created on 12 sep. 2026
Copyright François Durand 2014-2026
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


def kill_costs(matrix_duels_s, n_m):
    """Number of manipulators needed so that each candidate has no victory against each other candidate.

    Parameters
    ----------
    matrix_duels_s : ndarray
        Matrix of duels of the sincere voters: ``matrix_duels_s[e, d]`` is the number of sincere voters who rank
        ``e`` before ``d``.
    n_m : int
        Number of manipulators.

    Returns
    -------
    ndarray
        ``costs[e, d]`` is the minimal number of manipulators who must rank ``e`` before ``d`` so that ``d`` has no
        victory against ``e`` (i.e. so that the total number of voters ranking ``e`` before ``d`` is at least half
        of the electorate). It is 0 when this is already the case with the sincere voters alone. It is greater than
        ``n_m`` when this is impossible, even if all manipulators rank ``e`` before ``d``. By convention,
        ``costs[d, d] = n_m + 1``.

    Examples
    --------
    With 3 sincere voters and 2 manipulators, the electorate has 5 voters. To prevent a victory of ``d`` against
    ``e``, we need at least 2.5 voters (hence 3) who rank ``e`` before ``d``:

        >>> kill_costs(matrix_duels_s=[[0, 3, 1], [0, 0, 2], [2, 1, 0]], n_m=2)
        array([[3, 0, 2],
               [3, 3, 1],
               [1, 2, 3]])
    """
    matrix_duels_s = np.array(matrix_duels_s)
    n_c = matrix_duels_s.shape[0]
    n_s = int(matrix_duels_s[0, 1] + matrix_duels_s[1, 0])
    n_v = n_s + n_m
    costs = np.maximum(0, (n_v - 2 * matrix_duels_s + 1) // 2)
    costs[np.diag_indices(n_c)] = n_m + 1
    return costs


def _cheap_cycle_through(weights, start, threshold, max_length):
    """Find a simple cycle through ``start`` whose total weight is at least ``threshold``.

    Parameters
    ----------
    weights : ndarray
        ``weights[i, j]`` is the weight of arc ``i -> j``, or -1 if there is no such arc.
    start : int
        The cycle must contain this vertex.
    threshold : int
        Minimal total weight of the cycle.
    max_length : int
        Maximal number of vertices in the cycle.

    Returns
    -------
    list or None
        The vertices of the cycle, starting with ``start``, in the order of the arcs. ``None`` if no such cycle
        exists.

    Notes
    -----
    Dynamic programming over the subsets of vertices (Held-Karp style). The cost is exponential in the number of
    vertices, but this function is only used on the "core" of the candidates (cf.
    :func:`prevent_condorcet_winner`), which is small in practice.

    Examples
    --------
        >>> weights = np.array([[-1, 2, -1], [-1, -1, 2], [2, -1, -1]])
        >>> _cheap_cycle_through(weights, start=0, threshold=6, max_length=3)
        [0, 1, 2]
        >>> print(_cheap_cycle_through(weights, start=0, threshold=7, max_length=3))
        None
        >>> print(_cheap_cycle_through(weights, start=0, threshold=6, max_length=2))
        None
    """
    n = weights.shape[0]
    others = [v for v in range(n) if v != start]
    # ``best[(mask, v)]``: best weight of a path from ``start`` visiting exactly the vertices of ``mask`` (a subset
    # of ``others``, as a frozenset) and ending at ``v``. ``parent[(mask, v)]`` allows to reconstruct the path.
    best = {}
    parent = {}
    for v in others:
        if weights[start, v] >= 0:
            best[(frozenset([v]), v)] = int(weights[start, v])
            parent[(frozenset([v]), v)] = None
    frontier = [(frozenset([v]), v) for v in others if weights[start, v] >= 0]
    while frontier:
        new_frontier = []
        for mask, v in frontier:
            if weights[v, start] >= 0 and best[(mask, v)] + weights[v, start] >= threshold:
                # Reconstruct the cycle
                path = [v]
                key = (mask, v)
                while parent[key] is not None:
                    key = parent[key]
                    path.append(key[1])
                path.append(start)
                return path[::-1]
            if len(mask) >= max_length - 1:
                continue
            for u in others:
                if u in mask or weights[v, u] < 0:
                    continue
                new_mask = mask | {u}
                new_weight = best[(mask, v)] + int(weights[v, u])
                if new_weight > best.get((new_mask, u), -1):
                    if (new_mask, u) not in best:
                        new_frontier.append((new_mask, u))
                    best[(new_mask, u)] = new_weight
                    parent[(new_mask, u)] = (mask, v)
        frontier = new_frontier
    return None


def _reachable_from(is_source, is_arc):
    """Vertices reachable from a set of sources, with the parent of each newly reached vertex.

    Parameters
    ----------
    is_source : ndarray
        1d array of booleans.
    is_arc : ndarray
        2d array of booleans: ``is_arc[i, j]`` is True iff there is an arc ``i -> j``.

    Returns
    -------
    order : list
        The vertices reached (excluding the sources), in breadth-first order.
    parent : dict
        For each vertex of ``order``, the vertex from which it was reached.

    Examples
    --------
        >>> is_source = np.array([True, False, False, False])
        >>> is_arc = np.array([
        ...     [False, True, False, False],
        ...     [False, False, True, False],
        ...     [False, False, False, False],
        ...     [False, True, False, False],
        ... ])
        >>> _reachable_from(is_source, is_arc)
        ([1, 2], {1: 0, 2: 1})
    """
    reached = np.array(is_source, dtype=bool)
    order = []
    parent = {}
    frontier = list(np.where(is_source)[0])
    while frontier:
        new_frontier = []
        for v in frontier:
            for u in np.where(is_arc[v, :] & ~reached)[0]:
                reached[u] = True
                order.append(int(u))
                parent[int(u)] = int(v)
                new_frontier.append(u)
        frontier = new_frontier
    return order, parent


def prevent_condorcet_winner(matrix_duels_s, c, n_m, exact=True):
    """Decide whether manipulators can prevent any candidate other than ``c`` from being a Condorcet winner.

    Parameters
    ----------
    matrix_duels_s : ndarray
        Matrix of duels of the sincere voters: ``matrix_duels_s[e, d]`` is the number of sincere voters who rank
        ``e`` before ``d``.
    c : int
        The candidate for whom the manipulators want to manipulate. It is assumed that they all rank ``c`` first.
    n_m : int
        Number of manipulators.
    exact : bool
        If True, use an exact algorithm (whose cost may be exponential in the number of candidates). If False, use
        a polynomial algorithm which may fail to decide.

    Returns
    -------
    result : bool or nan
        True if the manipulators can cast ballots (with ``c`` on top) so that no candidate other than ``c`` is a
        Condorcet winner (in the sense of :attr:`~svvamp.Profile.matrix_victories_rk`: strict victories against all
        other candidates). False if it is impossible. ``nan`` if the algorithm cannot decide (only possible with
        ``exact=False``).
    ballots : ndarray or None
        If ``result`` is True, ``ballots[i, :]`` is the ranking cast by the ``i``-th manipulator. Otherwise, None.

    Notes
    -----
    Let ``n_v`` be the total number of voters (sincere voters and manipulators). We say that candidate ``e``
    *kills* candidate ``d`` when at least ``n_v / 2`` voters rank ``e`` before ``d``: then ``d`` has no victory
    against ``e``, hence ``d`` is not a Condorcet winner. The manipulators succeed iff each candidate other than
    ``c`` is killed by another one. Killing ``d`` with ``e`` costs a certain number of manipulators who must rank
    ``e`` before ``d`` (cf. :func:`kill_costs`). Whether ``c`` herself is a Condorcet winner is irrelevant here: in
    the voting rules using this function, ``c`` wins in that case anyway.

    * Since ``c`` is ranked first by all manipulators, she kills every candidate that she can kill. Then, by a
      "peeling" process, we place after ``c`` all candidates who are killed for free (by the sincere voters alone),
      or who can be killed by ``c`` or by an already placed candidate: this candidate is ranked before them by all
      manipulators.
    * The remaining candidates form the *core*: each of them can only be killed by another member of the core.
      Consider the digraph on the core with an arc ``e -> d`` iff ``e`` can kill ``d``. Choosing one killer for
      each member of the core gives a functional graph, whose cycles must be *realizable*: for a cycle of length
      ``L``, the manipulators must be split between ``L`` rotations of the cycle, which is possible iff the sum of
      the kill costs along the cycle is at most ``(L - 1) * n_m``. Conversely, the trees hanging on a realizable
      cycle can be placed after it in the ballots. Hence there is a solution iff every member of the core is
      reachable, in the digraph, from a realizable cycle.
    * With ``exact=True``, realizable cycles are searched by dynamic programming over the subsets of the core (cost
      exponential in the size of the core). With ``exact=False``, only cycles of length at most 3 are searched: if
      it is not enough to reach all the core, the algorithm concludes False when some member of the core is not
      reachable from any cycle at all, and ``nan`` otherwise.

    Examples
    --------
    Candidate 1 is a Condorcet winner among the 3 sincere voters. With 2 manipulators ranking 0 first, the total
    electorate has 5 voters, so that a candidate needs 3 voters against another one in order to beat her. Candidate
    1 has 3 voters against 0, so she cannot be killed by 0. But she has only 2 voters against 2: it suffices that
    both manipulators rank 2 before 1. Then candidate 2 must be killed too, and this is done by candidate 0.

        >>> matrix_duels_s = np.array([[0, 0, 1], [3, 0, 2], [2, 1, 0]])
        >>> result, ballots = prevent_condorcet_winner(matrix_duels_s, c=0, n_m=2)
        >>> result
        True
        >>> ballots
        array([[0, 2, 1],
               [0, 2, 1]])

    In the following example, it is impossible: the electorate has 5 voters and candidate 1 has 3 voters against
    every other candidate.

        >>> matrix_duels_s = np.array([[0, 0, 0], [3, 0, 3], [3, 0, 0]])
        >>> result, ballots = prevent_condorcet_winner(matrix_duels_s, c=0, n_m=2)
        >>> result
        False
        >>> print(ballots)
        None

    A case where the manipulators must split their ballots (``n_v = 6``): candidates 1 and 2 must kill each other,
    which needs 1 manipulator ranking 1 before 2 and 1 manipulator ranking 2 before 1:

        >>> matrix_duels_s = np.array([[0, 0, 0], [4, 0, 2], [4, 2, 0]])
        >>> result, ballots = prevent_condorcet_winner(matrix_duels_s, c=0, n_m=2)
        >>> result
        True
        >>> ballots
        array([[0, 1, 2],
               [0, 2, 1]])

    The same situation with an odd number of voters (``n_v = 7``): a tie is impossible, so candidates 1 and 2
    cannot kill each other:

        >>> matrix_duels_s = np.array([[0, 0, 0], [4, 0, 2], [4, 2, 0]])
        >>> result, ballots = prevent_condorcet_winner(matrix_duels_s, c=0, n_m=3)
        >>> result
        False

    With the polynomial algorithm, the result may be undecided (here, candidates 1 and 3 can only be killed by
    each other and the algorithm does not find a way to do it with cycles of length at most 3, but it does not
    prove that it is impossible either):

        >>> matrix_duels_s = np.array([
        ...     [0, 2, 4, 2],
        ...     [4, 0, 5, 3],
        ...     [2, 1, 0, 1],
        ...     [4, 3, 5, 0],
        ... ])
        >>> result, ballots = prevent_condorcet_winner(matrix_duels_s, c=0, n_m=1, exact=False)
        >>> result
        nan
        >>> result, ballots = prevent_condorcet_winner(matrix_duels_s, c=0, n_m=1, exact=True)
        >>> result
        False
    """
    matrix_duels_s = np.array(matrix_duels_s)
    n_c = matrix_duels_s.shape[0]
    costs = kill_costs(matrix_duels_s, n_m)
    is_feasible = costs <= n_m
    # Peeling: place after ``c`` all candidates who are killed for free, by ``c``, or by an already placed candidate.
    is_placed = np.zeros(n_c, dtype=bool)
    is_placed[c] = True
    placed_order = []
    changed = True
    while changed:
        changed = False
        for d in range(n_c):
            if is_placed[d]:
                continue
            if np.any(costs[:, d] == 0) or np.any(is_feasible[is_placed, d]):
                is_placed[d] = True
                placed_order.append(d)
                changed = True
    core = [d for d in range(n_c) if not is_placed[d]]
    if not core:
        ballot = [c, *placed_order]
        return True, np.tile(ballot, (n_m, 1))
    # Digraph on the core: ``weights[i, j] = n_m - cost`` if ``core[i]`` can kill ``core[j]``, -1 otherwise.
    n_core = len(core)
    costs_core = costs[np.ix_(core, core)]
    is_arc = costs_core <= n_m
    weights = np.where(is_arc, n_m - costs_core, -1)
    max_length = n_core if exact else min(n_core, 3)
    # Greedily choose vertex-disjoint realizable cycles and compute what they reach.
    is_reached = np.zeros(n_core, dtype=bool)
    cycles = []  # Each cycle is a list of core indices (in the order of the arcs).
    trees = []  # For each cycle, the list of core indices reached from it (in breadth-first order).
    parent = {}  # For each vertex of a tree, its killer.
    for i in range(n_core):
        if is_reached[i]:
            continue
        cycle = _cheap_cycle_through(weights, start=i, threshold=n_m, max_length=max_length)
        if cycle is None:
            continue
        is_source = np.zeros(n_core, dtype=bool)
        is_source[cycle] = True
        is_reached[cycle] = True
        # Vertices already reached by previous cycles cannot be reached again (cf. Notes).
        tree, tree_parent = _reachable_from(is_source, is_arc & ~is_reached[np.newaxis, :])
        is_reached[tree] = True
        cycles.append(cycle)
        trees.append(tree)
        parent.update(tree_parent)
    if not np.all(is_reached):
        if exact:
            return False, None
        # Vertices that are reachable from a cycle of any length (i.e. from a non-trivial strongly connected
        # component). If some vertex is not, then it cannot be killed at all.
        is_arc_int = is_arc.astype(int)
        reach_matrix = np.eye(n_core, dtype=int)
        for _ in range(n_core):
            reach_matrix = np.minimum(1, reach_matrix + reach_matrix @ is_arc_int)
        is_on_cycle = np.array([bool(np.any(reach_matrix[i, :] * is_arc_int[:, i])) for i in range(n_core)])
        is_reached_from_any_cycle = np.any(reach_matrix[is_on_cycle, :], axis=0) if np.any(is_on_cycle) else None
        if is_reached_from_any_cycle is None or not np.all(is_reached_from_any_cycle):
            return False, None
        return np.nan, None
    # Build the ballots: ``c``, then for each cycle, a rotation of the cycle followed by its tree, then the placed
    # candidates.
    ballots = np.zeros((n_m, n_c), dtype=int)
    ballots[:, 0] = c
    position = 1
    for cycle, tree in zip(cycles, trees, strict=True):
        length = len(cycle)
        # Rotation ``t`` starts at ``cycle[t]`` and violates the arc ``cycle[t - 1] -> cycle[t]``, whose cost is
        # ``costs_core[cycle[t - 1], cycle[t]]``. Hence at most ``n_m - cost`` manipulators can use rotation ``t``.
        remaining = n_m
        i_manipulator = 0
        for t in range(length):
            n_rotation_t = min(remaining, n_m - int(costs_core[cycle[t - 1], cycle[t]]))
            rotation = cycle[t:] + cycle[:t]
            for _ in range(n_rotation_t):
                ballots[i_manipulator, position : position + length] = [core[j] for j in rotation]
                i_manipulator += 1
            remaining -= n_rotation_t
        if remaining != 0:  # pragma: no cover
            raise AssertionError("Uh-oh!")
        position += length
        ballots[:, position : position + len(tree)] = [core[j] for j in tree]
        position += len(tree)
    ballots[:, position:] = placed_order
    return True, ballots
