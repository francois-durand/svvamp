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


def ballots_from_elimination_path(preferences_borda_s, path, c, w, n_m, block_until="reach_s"):
    """Build the ballots of the manipulators from an IRV elimination path, for Condorcet-IRV hybrid rules.

    This heuristic is designed for rules that combine IRV-style eliminations with Condorcet notions, such as
    :class:`~svvamp.RuleBenham`, :class:`~svvamp.RuleSmithIRV` and :class:`~svvamp.RuleTideman`. The ballots are
    not guaranteed to make ``c`` win: the caller must check them by computing the winner of the resulting profile.

    Parameters
    ----------
    preferences_borda_s : ndarray
        Borda scores of the sincere voters (cf. :attr:`~svvamp.Profile.preferences_borda_rk`).
    path : list or ndarray
        An IRV elimination path that makes ``c`` win with ``n_m`` manipulators: ``path[k]`` is the ``k``-th
        eliminated candidate, and ``path[-1]`` is ``c``. Typically, it is provided by the CM algorithms of
        :class:`~svvamp.RuleIRV`.
    c : int
        The candidate for whom the manipulators want to manipulate.
    w : int
        The sincere winner (of the hybrid rule). She must appear in ``path`` before ``c``.
    n_m : int
        Number of manipulators.
    block_until : str
        ``'reach_s'``: the top of the ballots is blocked so that IRV follows ``path`` until all candidates
        eliminated before ``w`` are eliminated. ``'eliminate_w'``: the top of the ballots is blocked so that IRV
        follows ``path`` until ``w`` herself is eliminated.

    Returns
    -------
    ndarray or None
        ``ballots[i, :]`` is the ranking cast by the ``i``-th manipulator. ``None`` if the path cannot be followed
        with ``n_m`` manipulators.

    Notes
    -----
    Let ``S`` be the set of candidates still alive when ``w`` is eliminated along ``path`` (including ``w``
    herself).

    1. The top of the ballots is blocked, position by position, so that IRV follows ``path`` (cf. ``block_until``).
       This is the same construction as in :class:`~svvamp.RuleIRV`: at each round, the manipulators give just
       enough first-position votes to the candidates who must survive.
    2. The rest of each ballot is filled with the candidates that are not blocked yet, in this order: ``c``, then
       the other members of ``S`` (in the order of ``path``), then ``w``, then the candidates outside ``S`` (in the
       order of ``path``).

    The filling order has three purposes: ranking ``S`` above the other candidates tends to make ``S`` the Smith
    set of the manipulated profile; ranking ``w`` at the bottom of ``S`` tends to make her eliminated inside ``S``;
    and ranking the other members of ``S`` above ``w`` tends to prevent ``w`` from remaining a Condorcet winner
    (which is the main obstacle, since the manipulators cannot change the duel between ``c`` and ``w``).

    Examples
    --------
    Three sincere voters, two manipulators for candidate 2 against the sincere winner 0. Along the path, candidate 3
    is eliminated first (so ``S = {0, 1, 2}``), then 0, then 1. In the first round, the sincere voters give one
    vote to each of 0, 1 and 3, and none to 2: one manipulator must vote for 2 so that 3 is eliminated (by
    tie-breaking, the candidate with the highest index is eliminated in case of a tie). Then the ballots are filled
    with 2 (candidate ``c``), 1 (the other member of ``S``), 0 (candidate ``w``) and 3 (outside ``S``).

        >>> preferences_borda_s = np.array([[3, 2, 1, 0], [1, 3, 2, 0], [1, 0, 2, 3]])
        >>> ballots_from_elimination_path(preferences_borda_s, path=[3, 0, 1, 2], c=2, w=0, n_m=2)
        array([[2, 1, 0, 3],
               [2, 1, 0, 3]])

    With ``block_until='eliminate_w'``, the manipulators also ensure that ``w`` is the plurality loser once ``S``
    is reached: in the second round, the votes are 1 for 0, 1 for 1 and 2 for 2 (including the manipulator), so
    the second manipulator must vote for 1.

        >>> ballots_from_elimination_path(
        ...     preferences_borda_s, path=[3, 0, 1, 2], c=2, w=0, n_m=2, block_until='eliminate_w')
        array([[2, 1, 0, 3],
               [1, 2, 0, 3]])

    When the path cannot be followed with ``n_m`` manipulators, the result is ``None``:

        >>> print(ballots_from_elimination_path(
        ...     preferences_borda_s, path=[3, 0, 1, 2], c=2, w=0, n_m=1, block_until='eliminate_w'))
        None
    """
    path = [int(d) for d in path]
    n_c = preferences_borda_s.shape[1]
    candidates = np.arange(n_c)
    index_w = path.index(w)
    if block_until == "reach_s":
        n_rounds_blocked = index_w
    elif block_until == "eliminate_w":
        n_rounds_blocked = index_w + 1
    else:
        raise ValueError(f"Unknown value for block_until: {block_until}")
    ballots = [[] for _ in range(n_m)]
    # Step 1: block the top of the ballots to ensure the beginning of the elimination path.
    scores_m_begin_r = np.zeros(n_c, dtype=int)  # Score due to manipulators at the beginning of the round
    is_candidate_alive_begin_r = np.ones(n_c, dtype=bool)
    current_top_v = -np.ones(n_m, dtype=int)  # -1 means that manipulator v is available
    is_to_place = np.ones((n_m, n_c), dtype=bool)
    for r in range(n_rounds_blocked):
        scores_tot_begin_r = np.full(n_c, np.nan)
        scores_tot_begin_r[is_candidate_alive_begin_r] = np.sum(
            np.equal(
                preferences_borda_s[:, is_candidate_alive_begin_r],
                np.max(preferences_borda_s[:, is_candidate_alive_begin_r], 1)[:, np.newaxis],
            ),
            0,
        )
        scores_tot_begin_r += scores_m_begin_r
        d = path[r]
        scores_m_new_r = np.zeros(n_c, dtype=int)
        scores_m_new_r[is_candidate_alive_begin_r] = np.maximum(
            0,
            scores_tot_begin_r[d]
            - scores_tot_begin_r[is_candidate_alive_begin_r]
            + (candidates[is_candidate_alive_begin_r] > d),
        ).astype(int)
        scores_m_begin_r = scores_m_begin_r + scores_m_new_r
        if np.sum(scores_m_begin_r) > n_m:
            return None
        scores_m_begin_r[d] = 0
        is_candidate_alive_begin_r[d] = False
        free_manipulators = np.where(current_top_v == -1)[0]
        i_manipulator = 0
        for e in range(n_c):
            for _ in range(scores_m_new_r[e]):
                manipulator = free_manipulators[i_manipulator]
                ballots[manipulator].append(e)
                current_top_v[manipulator] = e
                is_to_place[manipulator, e] = False
                i_manipulator += 1
        current_top_v[current_top_v == d] = -1
    # Step 2: fill the rest of the ballots.
    others_in_s = [d for d in path[index_w + 1 :] if d != c]
    outside_s = path[:index_w]
    filling_order = [c, *others_in_s, w, *outside_s]
    for manipulator in range(n_m):
        ballots[manipulator].extend([d for d in filling_order if is_to_place[manipulator, d]])
    return np.array(ballots, dtype=int).reshape((n_m, n_c))
