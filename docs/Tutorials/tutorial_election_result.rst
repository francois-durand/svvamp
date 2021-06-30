====================
Tutorial 2: Election
====================

Create a population of 9 voters with preferences over 5 candidates,
using the Spheroid model (extending Impartial Culture to utilities)::

    import svvamp
    profile_ = svvamp.PopulationSpheroid(V=9, C=5)

Print the preference rankings of the population::

    profile_.ensure_voters_sorted_by_rk()
    print(profile_.preferences_rk)

Create an election, using Plurality::

    election = svvamp.Plurality(profile_)

Print the ballots_::

    print(election.ballots_)

Print the scores_ of the candidates::

    print(election.scores_)

Print the ordering of candidates according to their scores_::

    print(election.candidates_by_scores_best_to_worst_)
    print(election.scores_best_to_worst_)

Print the winner, her score and her total utility::

    print(election.w_)
    print(election.score_w_)
    print(election.total_utility_w_)

Print whether the winner of the election is a Condorcet winner::

    print(election.w_is_condorcet_winner_ut_abs_)
