========================
Tutorial 3: Manipulation
========================

Create a population of 9 voters with preferences over 5 candidates,
using the Spheroid model (extending Impartial Culture to utilities)::

    import svvamp
    pop = svvamp.PopulationSpheroid(V=9, C=5)

Create an election, using Instant-Runoff Voting::

    election = svvamp.IRV(pop)

Ask SVVAMP whether the voting system meets the Condorcet criterion (in
general, not for this particular population)::

    print(election.meets_Condorcet_c)

------------------------
Coalitional Manipulation
------------------------

Cf. :meth:`~svvamp.Election.CM_full` for a definition of this notion.

Check Coalitional Manipulation (CM)::

    print(election.CM())

SVVAMP returns a pair (``is_CM``, ``log_CM``).

For each voting system, SVVAMP uses by default its most precise algorithm
running in polynomial time. For IRV, the decision problem is
NP-complete, so this polynomial algorithm is not exact. For that reason,
``is_CM`` can be a boolean (whether the election is manipulable or not), or 
the conventional value ``numpy.nan`` meaning that the algorithm was not able
to decide.

``log_CM`` is a string representing the options used to compute CM. Check the
possible options::

    print(election.options_parameters)

The main option is ``CM_option``. Change it and compute CM again::

    election.CM_option = 'exact'
    print(election.CM())

Now, the return value ``is_CM`` is necessarily a boolean.

You could have set the option as soon as you created the election with the
following syntax::

    election = svvamp.IRV(pop, CM_option='exact')

Print more details about CM::

    print(election.CM_with_candidates())

Now, SVVAMP also return ``candidates_CM``, an array of boolean indicating
which candidates can benefit from CM.

SVVAMP is clever enough: 1. not to do obviously useless computation and 2. not
to do the same computation twice.

    1.  In this example, when calling ``CM()``, SVVAMP may decide that CM
        is impossible for candidates 0 and 1, but possible for candidate 2.
        SVVAMP stops computation and decides ``is_CM = True``.
    2.  In that case, when calling ``CM_full()``, SVVAMP does not perform the
        computation again for candidates 0, 1 and 2.

-----------------------------------------
Other notions of coalitional manipulation
-----------------------------------------

Check Ignorant-Coalition Manipulation (cf. :meth:`~svvamp.Election.ICM_full`)::

    print(election.ICM())
    print(election.ICM_with_candidates())

Check Trivial Manipulation (cf. :meth:`~svvamp.Election.TM_full`)::

    print(election.TM())
    print(election.TM_with_candidates())

Check Unison Manipulation (cf. :meth:`~svvamp.Election.UM_full`)::

    election.UM_option = 'exact'
    print(election.UM())
    print(election.UM_with_candidates())

-----------------------
Individual Manipulation
-----------------------

Cf. :meth:`~svvamp.Election.IM_full` for a definition of this notion.

Check Individual Manipulation (IM)::

    election.IM_option = 'exact'
    print(election.IM())

SVVAMP returns a boolean ``is_IM`` (whether the election is manipulable by a
single manipulator or not) and a string ``log_IM`` (the option used to compute
IM).

Print more details about IM::

    print(election.IM_full())

Now, SVVAMP returns (``is_IM``, ``log_IM``, ``candidates_IM``, ``voters_IM``
and ``v_IM_for_c``). ``candidates_IM`` indicates which candidates can benefit
from IM. ``voters_IM`` indicates which voters can and want to perform IM.
``v_IM_for_c`` indicates, for each voter ``v`` and each candidate ``c``,
whether ``v`` can and want to manipulate for ``c``.

---------------------------------------
Independence of Irrelevant Alternatives
---------------------------------------

Cf. :meth:`~svvamp.Election.not_IIA_complete` for a definition of this notion.

Modify the option in order to compute IIA with an exact (non-polynomial)
algorithm::

    import numpy
    election.IIA_subset_maximum_size = numpy.inf

Check Independence of Irrelevant Alternatives (IIA)::

    print(election.not_IIA())

SVVAMP returns a boolean (whether the election violates IIA) and a string
(the options used to do the computation).

You can ask more information about IIA::

    print(election.not_IIA_complete())

If the election violates IIA, then SVVAMP provides an example of subset of
candidates violating IIA and the corresponding winner.










