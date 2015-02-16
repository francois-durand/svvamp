========
Usage
========

To use SVVAMP in a project::

    import svvamp

Create a population of 100 voters with preferences over 5 candidates, 
using the Impartial Culture::

    pop = svvamp.PopulationSpheroid(V=100, C=5)

Demonstrate the functions of superclass :class:`~svvamp.Population`::

    pop.demo()

Plot the population, with several projections of the utility space::

    pop.plot3()

    pop.plot4()

Define an election with this population, using Approval voting::

    election = svvamp.Approval(pop)

Demonstrate the functions of superclass :class:`~svvamp.Election`::

    election.demo()