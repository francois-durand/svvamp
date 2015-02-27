=============================
Tutorial 4: Population models
=============================

.. note::

    In this tutorial, we present the probabilistic models that can be used in
    SVVAMP to generate a population. It is also possible to enter
    a population manually (:class:`~svvamp.Population`) or to import a
    population from a file (:class:`~svvamp.PopulationFromFile`).

-----------------
Plot a population
-----------------

Create a population of 100 voters with preferences over 5 candidates,
using the Von-Mises Fisher model, which represents a polarized culture::

    import svvamp
    pop = svvamp.PopulationVMFHypersphere(V=100, C=5, vmf_concentration=10)
    pop.labels_candidates = ['Alice', 'Bob', 'Catherine', 'Dave', 'Ellen']

Plot the restriction of the population to 3 candidates, for example [0, 2,
3] (Alice, Catherine and Dave), in the utility space::

    pop.plot3(indexes=[0, 2, 3])

Cf. :meth:`~svvamp.Population.plot3` for more information about this
representation.

Plot the restriction of the population to 4 candidates, for example [0, 1,
2, 4] (Alice, Bob, Catherine and Ellen), in the utility space::

    pop.plot4(indexes=[0, 1, 2, 4])

Cf. :meth:`~svvamp.Population.plot4` for more information about this
representation.

-----------------
Impartial culture
-----------------

The Spheroid model is an extension of the Impartial Culture to utilities::

    pop = svvamp.PopulationSpheroid(V=100, C=5)
    pop.plot3()
    pop.plot4()

The Cubic Uniform model is another one::

    pop = svvamp.PopulationCubicUniform(V=5000, C=3)
    pop.plot3(normalize=False)
    pop = svvamp.PopulationCubicUniform(V=5000, C=4)
    pop.plot4(normalize=False)

Cf. :class:`~svvamp.PopulationSpheroid`,
:class:`~svvamp.PopulationCubicUniform`.

--------------------------------
Neutral culture with weak orders
--------------------------------

The Ladder model is also neutral (it treats all candidates equally) and voters
are also independent, like in Impartial Culture, but weak orders are possible.

::

    pop = svvamp.PopulationLadder(V=1000, C=3, n_rungs=5)
    pop.plot3(normalize=False)

::

    pop = svvamp.PopulationLadder(V=1000, C=4, n_rungs=5)
    pop.plot4(normalize=False)

Cf. :class:`~svvamp.PopulationLadder`.

------------------
Polarized cultures
------------------

In the beginning of this tutorial, we have already met the Von-Mises Fisher
model on the hypersphere. A variant is the VMF model on the hypercircle.

Cf. :class:`~svvamp.PopulationVMFHypersphere`,
:class:`~svvamp.PopulationVMFHypercircle`.

------------------
Political spectrum
------------------

In these models, voters and candidates draw independent positions in a
Euclidean space (the 'political spectrum'). The utility of a voter ``v`` for a
candidate ``c`` is a decreasing function of the distance between their
positions. If the dimension of the political spectrum is 1,
then the population is necessarily *single-peaked* (cf. 'The theory of
committees and elections', Duncan Black, 1958).

Gaussian Well model::

    pop = svvamp.PopulationGaussianWell(V=1000, C=4, sigma=[1], shift=[0])
    pop.plot3()
    pop.plot4()

Euclidean Box model::

    pop = svvamp.PopulationEuclideanBox(V=1000, C=4, box_dimensions=[1])
    pop.plot3()
    pop.plot4()

Cf. :class:`~svvamp.PopulationEuclideanBox`,
:class:`~svvamp.PopulationGaussianWell`.









