======================
Tutorial 1: Population
======================

Create a population of 9 voters with preferences over 5 candidates,
using the Spheroid model (extending Impartial Culture to utilities)::

    import svvamp
    profile_ = svvamp.PopulationSpheroid(V=9, C=5)

You can give a label to each candidate::

    profile_.labels_candidates = ['Alice', 'Bob', 'Catherine', 'Dave', 'Ellen']

Print basic info about the population::

    print(profile_.V)
    print(profile_.C)
    print(profile_.labels_candidates)

Print voters' preferences::

    print(profile_.preferences_ut)
    print(profile_.preferences_rk)
    print(profile_.preferences_borda_ut)

Sort voters by their order of preference::

    profile_.ensure_voters_sorted_by_rk()

And check the result of the sort::

    print(profile_.preferences_ut)
    print(profile_.preferences_rk)
    print(profile_.preferences_borda_ut)

Print the Plurality score, Borda score and total utility of each candidate::

    print(profile_.plurality_scores_ut)
    print(profile_.borda_score_c_ut)
    print(profile_.total_utility_c)

Print the matrix of duels and the matrix of victories::

    print(profile_.matrix_duels_ut)
    print(profile_.matrix_victories_ut_abs)

Print the Condorcet winner::

    print(profile_.condorcet_winner_ut_abs)

If there is no Condorcet winner, then by convention,
SVVAMP returns ``numpy.nan``.
