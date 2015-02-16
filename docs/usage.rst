========
Usage
========

To use SVVAMP in a project::

    import svvamp

Create a population of 10 voters with preferences over 5 candidates, 
using the Impartial Culture::

    pop1 = svvamp.PopulationSpheroid(V=10, C=5)

Demonstrate the functions of superclass :class:`~svvamp.Population`::

    pop1.demo()

Plot a population, with several projections of the utility space::

    pop2 = svvamp.PopulationSpheroid(V=100, C=5)

    pop2.plot3()

    pop2.plot4()

Define an election, using Approval voting::

    election = svvamp.Approval(pop1)

Demonstrate the functions of superclass :class:`~svvamp.Election`::

    election.demo()