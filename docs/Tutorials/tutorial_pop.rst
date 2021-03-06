======================
Tutorial 1: Population
======================

Create a population of 9 voters with preferences over 5 candidates,
using the Spheroid model (extending Impartial Culture to utilities)::

    import svvamp
    pop = svvamp.PopulationSpheroid(V=9, C=5)

You can give a label to each candidate::

    pop.labels_candidates = ['Alice', 'Bob', 'Catherine', 'Dave', 'Ellen']

Print basic info about the population::

    print(pop.V)
    print(pop.C)
    print(pop.labels_candidates)

Print voters' preferences::

    print(pop.preferences_ut)
    print(pop.preferences_rk)
    print(pop.preferences_borda_ut)

Sort voters by their order of preference::

    pop.ensure_voters_sorted_by_rk()

And check the result of the sort::

    print(pop.preferences_ut)
    print(pop.preferences_rk)
    print(pop.preferences_borda_ut)

Print the Plurality score, Borda score and total utility of each candidate::

    print(pop.plurality_scores_ut)
    print(pop.borda_score_c_ut)
    print(pop.total_utility_c)

Print the matrix of duels and the matrix of victories::

    print(pop.matrix_duels_ut)
    print(pop.matrix_victories_ut_abs)

Print the Condorcet winner::

    print(pop.condorcet_winner_ut_abs)

If there is no Condorcet winner, then by convention,
SVVAMP returns ``numpy.nan``.