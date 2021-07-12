.. :changelog:

=======
History
=======

0.1.0 (2021-07-12)
------------------

* Refactor the code completely. New architecture, especially for ``Rule`` (formerly ``Election``), avoiding
  diamond inheritance. Rename most classes, properties and methods.
* Cover the code with tests. Print a message when execution reaches an uncovered part of the code.
* Fix some minor bugs.
* Documentation in numpy style.
* Tutorials are now Jupyter notebooks.

0.0.4 (2015-03-10)
------------------

* Correct a minor bug in Plurality.IM (voters_IM is now updated).

0.0.3 (2015-02-28)
------------------

* Rename functions and attributes with suffix _vtb to _rk.
* Allow to define a population by both utilities and rankings.
* Add shift to Euclidean box model.
* Range voting / Majority Judgment: with a discrete set of grades, send to closest authorized grades.

0.0.2 (2015-02-16)
------------------

* 8 population models and 23 voting systems.

0.0.1 (2015-02-14)
------------------

* First release on PyPI.

