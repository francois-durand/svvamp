.. :changelog:

=======
History
=======

------------------------------------------------------
0.4.0 (2021-07-19): Black, Copeland, Split Cycle, STAR
------------------------------------------------------

* Add ``RuleBlack``.
* Add ``RuleCopeland``.
* Add ``RuleSplitCycle``.
* Add ``RuleSTAR``.
* In ``RuleRangeVoting`` and ``RuleMajorityJudgment``, add an attribute ``allowed_grades``: a list of the
  allowed grades.

--------------------------------------------------------------
0.3.0 (2021-07-16): New CM Algorithms for Smith-IRV-Like Rules
--------------------------------------------------------------

* New CM algorithms for Smith-IRV-like rules:

  * New CM algorithms for ``RuleICRV``, ``RuleSmithIRV``, ``RuleTideman``, ``RuleWoodall`` and ``RuleIRVAverage``.
  * Add ``RuleIRV.example_ballots_cm_c_`` and ``RuleIRV.example_ballots_cm_w_against_``: examples of manipulating ballots
    (used as heuristic to manipulate Smith-IRV and similar rules).
  * In ``RuleCondorcetAbsIRV`` and ``RuleCondorcetVtbIRV``, the former option ``almost_exact`` is renamed to
    ``very_slow``, for the sake of consistency with Smith-IRV and similar voting rules.

* Improve imports/exports:

  * ``ProfileFromFile`` can now import a CVR (cast vote record) file.
  * ``ProfileFromFile`` has a new parameter ``sort_candidates``: sort the candidates from strongest to weakest (in a
    Black method sense).
  * Add ``Profile.to_csv``: export the utilities to a csv file.

* Add ``Rule.options``: a dictionary with all the options of a rule.
* Bug fixes:

  * Fix a major bug in ``RuleTideman``: ballots were not counted correctly.
  * Fix a bug in ``RuleExhaustiveBallot`` and ``RuleIRV``: applying a voting rule to a profile was able to change the
    options of another (related) rule.

------------------------------------------------------
0.2.0 (2021-07-13): Smith-IRV and Similar Voting Rules
------------------------------------------------------

* A ``Profile`` object can now compute its Smith set (also called "top cycle"): ``smith_set_rk``, ``smith_set_rk_ctb``,
  ``smith_set_ut_abs``, ``smith_set_ut_abs_ctb``, ``smith_set_ut_rel``, ``smith_set_ut_rel_ctb``.
* Add Smith-IRV.
* Add Tideman's rule.
* Add Woodall's rule.
* Add constant ``RULE_CLASSES``: list of all the rule classes.
* Tools that are mostly dedicated to developers:

  * A global option allows to throw an error when an uncovered portion of code is reached.
  * Add ``Profile.to_doctest_string``.
  * Add ``Rule._set_random_options``.
  * Add ``Rule._random_instruction``.

--------------------------------------
0.1.2 (2021-07-12): Fix Release Number
--------------------------------------

* Fix release number.

----------------------------------
0.1.1 (2021-07-12): Fix Deployment
----------------------------------

* Fix deployment on PyPI.

----------------------------------------
0.1.0 (2021-07-12): Complete Refactoring
----------------------------------------

* Refactor the code completely. New architecture, especially for ``Rule`` (formerly ``Election``), avoiding
  diamond inheritance. Rename most classes, properties and methods.
* Cover the code with tests. Print a message when execution reaches an uncovered part of the code.
* Fix some minor bugs.
* Documentation in numpy style.
* Tutorials are now Jupyter notebooks.
* New rules: Kim-Roush and IRV-Average.

------------------------------------------
0.0.4 (2015-03-10): Fix a Bug in Plurality
------------------------------------------

* Correct a minor bug in Plurality.IM (voters_IM is now updated).

----------------------------------------------------
0.0.3 (2015-02-28): Miscellaneous Minor Improvements
----------------------------------------------------

* Rename functions and attributes with suffix _vtb to _rk.
* Allow to define a population by both utilities and rankings.
* Add shift to Euclidean box model.
* Range voting / Majority Judgment: with a discrete set of grades, send to closest authorized grades.

------------------------------------------
0.0.2 (2015-02-16): SVVAMP's Core Features
------------------------------------------

* 8 population models and 23 voting systems.

---------------------------------
0.0.1 (2015-02-14): First Release
---------------------------------

* First release on PyPI.
