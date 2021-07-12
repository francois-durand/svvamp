import random
import numpy as np
import networkx as nx


def preferences_ut_to_preferences_rk(preferences_ut):
    """Convert utilities to rankings.

    Parameters
    ----------
    preferences_ut : list of list of numbers (or ndarray)
        ``preferences_ut[v, c]`` is the utility of candidate ``c`` as seen by voter ``v``.

    Returns
    -------
    preferences_rk : ndarray
        ``preferences_rk[v, k]`` is the candidate at rank ``k`` for voter ``v``. If
        ``preferences_ut[v,c] == preferences_ut[v,d]``, then it is drawn at random whether ``v`` prefers
        ``c`` to ``d``  or ``d`` to ``c``.

    Examples
    --------
        >>> preferences_ut_to_preferences_rk(preferences_ut=[[5, 1, 2], [4, 10, 1]])
        array([[0, 2, 1],
               [1, 0, 2]])
    """
    preferences_ut = np.array(preferences_ut)
    n_v, n_c = preferences_ut.shape
    tiebreaker = np.random.rand(n_v, n_c)
    return np.array(np.lexsort((tiebreaker, - preferences_ut), 1), dtype=int)


def preferences_rk_to_preferences_borda_rk(preferences_rk):
    """Convert rankings to Borda scores.

    Parameters
    ----------
    preferences_rk : list of list of integers (or ndarray)
        ``preferences_rk[v, k]`` is the candidate at rank ``k`` for voter ``v``.

    Returns
    -------
    preferences_borda_rk : ndarray
        ``preferences_borda_rk[v, c]`` is the Borda score (between 0 and ``C - 1``) of candidate ``c`` for voter ``v``.

    Examples
    --------
        >>> preferences_rk_to_preferences_borda_rk(preferences_rk=[[0, 2, 1], [1, 0, 2]])
        array([[2, 0, 1],
               [1, 2, 0]])
    """
    preferences_rk = np.array(preferences_rk)
    _, n_c = preferences_rk.shape
    return - np.array(np.argsort(preferences_rk, 1), dtype=int) + n_c - 1


def preferences_ut_to_preferences_borda_ut(preferences_ut):
    """Convert utilities to Borda scores (with possible equalities).

    Parameters
    ----------
    preferences_ut : list of list of numbers (or ndarray)
        ``preferences_ut[v, c]`` is the utility of candidate ``c`` as seen by voter ``v``.

    Returns
    -------
    preferences_borda_ut : ndarray
        ``preferences_borda_ut[v, c]`` gains 1 point for each ``d`` such that ``v`` prefers ``c`` to ``d``,
        and 0.5 point for each ``d`` such that ``v`` is indifferent between ``c`` and ``d``.

    Examples
    --------
        >>> preferences_ut_to_preferences_borda_ut(preferences_ut=[[5, 1, 2], [4, 10, 1]])
        array([[2., 0., 1.],
               [1., 2., 0.]])
    """
    preferences_ut = np.array(preferences_ut)
    n_v, n_c = preferences_ut.shape
    preference_borda_ut = np.zeros((n_v, n_c))
    for c in range(n_c):
        preference_borda_ut[:, c] = np.sum(
            0.5 * (preferences_ut[:, c][:, np.newaxis] >= preferences_ut)
            + 0.5 * (preferences_ut[:, c][:, np.newaxis] > preferences_ut),
            axis=1
        ) - 0.5
    return preference_borda_ut


def preferences_ut_to_matrix_duels_ut(preferences_ut):
    """Compute the matrix of duels.

    Parameters
    ----------
    preferences_ut : list of list of numbers (or ndarray)
        ``preferences_ut[v, c]`` is the utility of candidate ``c`` as seen by voter ``v``.

    Returns
    -------
    matrix_duels_ut : ndarray
        ``matrix_duels_ut[c, d]`` is the number of voters who strictly prefer candidate ``c`` to candidate ``d``.
        By convention, diagonal coefficients are set to 0.

    Examples
    --------
        >>> preferences_ut_to_matrix_duels_ut(preferences_ut=[[5, 1, 2], [4, 10, 1]])
        array([[0, 1, 2],
               [1, 0, 1],
               [0, 1, 0]])
    """
    preferences_ut = np.array(preferences_ut)
    n_v, n_c = preferences_ut.shape
    matrix_duels = np.zeros((n_c, n_c), dtype=np.int)
    for c in range(n_c):
        for d in range(c + 1, n_c):
            matrix_duels[c, d] = np.sum(preferences_ut[:, c] > preferences_ut[:, d])
            matrix_duels[d, c] = np.sum(preferences_ut[:, d] > preferences_ut[:, c])
    return matrix_duels


def matrix_victories_to_smith_set(matrix_victories):
    """
    Smith set associated to a matrix of victories.

    Parameters
    ----------
    matrix_victories : ndarray.
        ``matrix_victories[c, d]`` is 1 vs 0 in case of victory, 0.5 vs 0.5 in case of tie. Diagonal coefficients
        can use any conventional value, but they must all have the same value.

    Returns
    -------
    smith_set : list
        List of candidates in the Smith set, sorted by increasing order.

    Examples
    --------
        >>> matrix_victories = np.array([
        ...     [0, 1,  1, 1, 1,  1, 0],
        ...     [0, 0,  0, 0, 1,  0, 0],
        ...     [0, 1,  0, 0, 1, .5, 1],
        ...     [0, 1,  1, 0, 1,  1, 1],
        ...     [0, 0,  0, 0, 0,  0, 0],
        ...     [0, 1, .5, 0, 1,  0, 0],
        ...     [1, 1,  0, 0, 1,  1, 0]
        ... ])
        >>> matrix_victories_to_smith_set(matrix_victories)
        [0, 2, 3, 5, 6]
    """
    n_c = matrix_victories.shape[0]
    copeland_scores = matrix_victories.sum(axis=1)
    copeland_decreasing = sorted(copeland_scores, reverse=True)
    candidates_by_copeland_best_worst = sorted(range(n_c), key=copeland_scores.__getitem__, reverse=True)
    matrix_victories_sorted = (
        matrix_victories[candidates_by_copeland_best_worst, :][:, candidates_by_copeland_best_worst])
    i = 0
    while True:
        while i < n_c - 1 and copeland_decreasing[i] == copeland_decreasing[i + 1]:
            i += 1
        if np.all(matrix_victories_sorted[0:i + 1, i + 1:] == 1):
            return sorted(candidates_by_copeland_best_worst[0:i + 1])
        i += 1


def is_resistant_condorcet(w, preferences_ut):
    """Test for resistant Condorcet winner.

    A Condorcet winner ``w`` is *resistant* iff in any Condorcet voting system, the profile is not manipulable (cf.
    Durand et al. 2014). This is equivalent to say that for any pair ``(c, d)`` of other candidates, there is a strict
    majority of voters who simultaneously:

    * Do not prefer ``c`` to ``w``,
    * And prefer ``w`` to ``d``.

    Parameters
    ----------
    w : int (candidate) or NaN
    preferences_ut : list of list of numbers (or ndarray)
        ``preferences_ut[v, c]`` is the utility of candidate ``c`` as seen by voter ``v``.

    Returns
    -------
    bool
        If w is a resistant Condorcet winner, then True. Otherwise (or if w is NaN), then False.

    Examples
    --------
        >>> is_resistant_condorcet(w=0, preferences_ut=[[5, 1, 2], [4, 10, 1]])
        False
    """
    if np.isnan(w):
        return False
    preferences_ut = np.array(preferences_ut)
    n_v, n_c = preferences_ut.shape
    for c in range(n_c):
        if c == w:
            continue
        v_does_not_prefer_c_to_w = (preferences_ut[:, w] >= preferences_ut[:, c])
        for d in range(n_c):
            if d == w:
                continue
            v_prefers_w_to_d = (preferences_ut[:, w] > preferences_ut[:, d])
            if np.sum(np.logical_and(v_does_not_prefer_c_to_w, v_prefers_w_to_d)) <= n_v / 2:
                return False
    return True


def compute_next_subset_with_w(prev_subset, n_c, n_c_r, w):
    """Compute the next subset containing w, by lexicographic order.

    This function is internally used by :class:`Rule` to compute Independence of Irrelevant Alternatives (IIA).

    Parameters
    ----------
    prev_subset : list of integers (or 1d ndarray)
        ``prev_subset(k)`` is the ``k``-th candidate of the subset. Candidates must be sorted by ascending order.
        Candidate ``w`` must belong to ``prev_subset``.
    n_c : int
        Total number of candidates.
    n_c_r : int
        Number of candidates for the subset.
    w : int
        A candidate whose presence is required in the subset.

    Returns
    -------
    next_subset : list of integers
        ``next_subset(k)`` is the ``k``-th candidate of the subset. Candidates are sorted by ascending order.
        Candidate ``w`` belongs to next_subset.

    Examples
    --------
        >>> compute_next_subset_with_w(prev_subset=[0, 2, 7, 8, 9], n_c=10, n_c_r=5, w=0)
        [0, 3, 4, 5, 6]
        >>> compute_next_subset_with_w(prev_subset=[0, 2, 7, 8, 9], n_c=10, n_c_r=5, w=8)
        [0, 3, 4, 5, 8]

    If we have already the last subset, then the result is None:

        >>> print(compute_next_subset_with_w(prev_subset=[0, 6, 7, 8, 9], n_c=10, n_c_r=5, w=0))
        None
    """
    # TODO: this could be rewritten with an iterator.
    max_allowed_value = n_c
    for index_pivot in range(n_c_r - 1, -1, -1):
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
    for i in range(index_pivot, n_c_r - 1):
        next_subset[i] = new_member
        new_member += 1
    next_subset[n_c_r - 1] = max(w, new_member)
    return next_subset


def compute_next_permutation(prev_permutation, n_c):
    """Compute next permutation by lexicographic order.

    Parameters
    ----------
    prev_permutation : list of integers (or ndarray)
        A list of distinct numbers.
    n_c : int
        Number of elements in prev_permutation

    Returns
    -------
    next_permutation : ndarray
        If ``prev_permutation`` was the last permutation in lexicographic order (i.e. all numbers sorted in descending
        order), then ``next_permutation = None``.

    Examples
    --------
        >>> compute_next_permutation([0, 2, 1, 4, 3], 5)
        array([0, 2, 3, 1, 4])
        >>> import numpy as np
        >>> compute_next_permutation(np.array([0, 2, 1, 4, 3]), 5)
        array([0, 2, 3, 1, 4])
    """
    # TODO: this could be rewritten with an iterator.
    prev_permutation = list(prev_permutation)
    for i in range(n_c - 2, -1, -1):
        if prev_permutation[i] < prev_permutation[i + 1]:
            index_pivot = i
            break
    else:
        return None
    index_replacement = None
    for i in range(n_c - 1, index_pivot, -1):
        if prev_permutation[i] > prev_permutation[index_pivot]:
            index_replacement = i
            break
    return np.array(
        prev_permutation[0:index_pivot] + [prev_permutation[index_replacement]]
        + prev_permutation[n_c - 1:index_replacement:-1]
        + [prev_permutation[index_pivot]] + prev_permutation[index_replacement-1:index_pivot:-1],
        dtype=int
    )


def compute_next_borda_clever(prev_permutation, prev_favorite, n_c):
    """Compute next vector of Borda scores in 'clever' order.

    Parameters
    ----------
    prev_permutation : list of integers (or 1d ndarray)
        Must be exactly all integers from 0 to C - 1.
    prev_favorite : int
        Index of the preferred candidate. I.e., ``prev_permutation[prev_favorite]`` must be equal to ``C - 1``.
    n_c : int
        Number of elements in ``prev_permutation``.

    Returns
    -------
    (``next_permutation``, ``next_favorite``) : tuple
        ``next_permutation`` is a 1d array of all integers from 0 to ``C - 1``, . Next vector of Borda scores in the
        'clever' order. If ``prev_permutation`` was the last permutation in the 'clever' order, then
        ``next_permutation = None``. ``next_favorite`` is an Integer (or None); it is the new preferred candidate.

    Notes
    -----
    A vector of Borda scores is seen as two elements:

    1. A permuted list of Borda scores from 0 to ``C - 2``,
    2. The insertion of Borda score ``C - 1`` in this list.

    The 'clever' order sorts first by lexicographic order on (1), then by the position (2) (from last position to
    first position).

    To find the next permutation, if Borda score ``C - 1`` can be moved one step to the left, we do it. Otherwise,
    we take the next permutation of ``{0, ..., C - 2}`` in lexicographic order, and put Borda score ``C - 1`` at the
    end.

    In the 'clever' order, the first vector is ``[0, ..., C - 1]`` and the last one is ``[C - 1, ..., 0]``.

    When looking for manipulations (IM or UM principally), here is the advantage of using the 'clever' order instead of
    lexicographic order: in only the ``C`` first  vectors, we try all candidates as top-ranked choice. In many voting
    systems, this accelerates finding the manipulation.

    Examples
    --------
        >>> compute_next_borda_clever(prev_permutation=[0, 1, 4, 2, 3], prev_favorite=2, n_c=5)
        (array([0, 4, 1, 2, 3]), 1)
        >>> compute_next_borda_clever(prev_permutation=[4, 0, 1, 2, 3], prev_favorite=0, n_c=5)
        (array([0, 1, 3, 2, 4]), 4)

    At the end of the iteration, we return (None, None):

        >>> compute_next_borda_clever(prev_permutation=[4, 3, 2, 1, 0], prev_favorite=0, n_c=5)
        (None, None)
    """
    # TODO: this could be rewritten with an iterator.
    if prev_favorite > 0:
        next_favorite = prev_favorite - 1
        next_permutation = np.copy(prev_permutation)
        next_permutation[[prev_favorite, next_favorite]] = next_permutation[[next_favorite, prev_favorite]]
    else:
        try:
            next_permutation = np.concatenate((compute_next_permutation(prev_permutation[1:n_c], n_c - 1), [n_c - 1]))
            next_favorite = n_c - 1
        except ValueError:
            next_permutation = None
            next_favorite = None
    return next_permutation, next_favorite


def strong_connected_components(a):
    """Strong connected components of a digraph, sorted by topological order.

    Parameters
    ----------
    a : a list of lists or a numpy array
        The adjacency matrix: ``a[i, j] = True`` means that ``i`` has an edge toward ``j``.

    Returns
    -------
    list of sets
        For example, the first set is the top component of the digraph; in the case of a victory matrix, it is the
        top set (smallest subset of candidates who win against all the others).

    Examples
    --------
        >>> strong_connected_components(a=[
        ...     [False,  True, False,  True,  True],
        ...     [False, False,  True,  True,  True],
        ...     [ True, False, False,  True,  True],
        ...     [False, False, False, False,  True],
        ...     [False, False, False, False, False]
        ... ]) == [{0, 1, 2}, {3}, {4}]
        True
    """
    condensed = nx.condensation(nx.DiGraph(np.array(a)))
    return [condensed.nodes[component]['members'] for component in nx.topological_sort(condensed)]


def initialize_random_seeds(n=0):
    """Initialize the random seeds.

    Parameters
    ----------
    n : int
        The desired random seed. Default: 0.
    """
    random.seed(n)
    np.random.seed(n)
