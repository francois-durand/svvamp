=====
Usage
=====

**First steps**

To use SVVAMP in a project::

    import svvamp

Create a population of 10 voters with preferences over 5 candidates,
using the Spheroid model (extending Impartial Culture to utilities)::

    profile_ = svvamp.PopulationSpheroid(V=10, C=5)

Demonstrate the functions of superclass :class:`~svvamp.Population`::

    profile_.demo_()

Create an election, using Approval voting::

    election = svvamp.Approval(profile_)

Demonstrate the functions of superclass :class:`~svvamp.Election`::

    election.demo_()

**Tutorials**

.. toctree::
   :maxdepth: 3

   Tutorials/tutorial_pop
   Tutorials/tutorial_election_result
   Tutorials/tutorial_election
   Tutorials/tutorial_pop_models
