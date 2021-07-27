.. :changelog:

=======
History
=======

---------------------------------
0.6.5 (2021-07-27): Accelerations
---------------------------------

* Accelerate ``ProfileSubsetCandidates`` (used for IIA but also for some voting rules such as ``RuleTideman``).
* Add ``ProfileUM``. This is used to speed up the generic exact algorithm for UM in the case of voting rules based
  on rankings.
* ``RuleTideman``: accelerate counting the ballot and computing the winner (especially useful for exact UM).
* Accelerate ``preferences_ut_to_matrix_duels_ut``.
* Accelerate ``matrix_victories_to_smith_set``.

-----------------------------------------------
0.6.4 (2021-07-27): Option "faster" for Maximin
-----------------------------------------------

* ``RuleMaximin``: implement ``cm_option=faster``, which is as precise as ``fast`` to compute ``is_cm_``, less precise
  to compute the bounds ``necessary_coalition_size_cm_`` and ``sufficient_coalition_size_cm_``, but a lot faster.

------------------------------------------------------
0.6.3 (2021-07-27): Accelerate ProfileSubsetCandidates
------------------------------------------------------

* ``Profile`` now has a parameter ``preferences_borda_rk``. At initialization, it can be given instead of
  ``preferences_rk``.
* ``ProfileSubsetCandidates``: accelerate the initialization method.

--------------------------------------
0.6.2 (2021-07-27): Accelerate Maximin
--------------------------------------

* Accelerate ``RuleMaximin.necessary_coalition_size_cm_`` and ``RuleMaximin.sufficient_coalition_size_cm_``.
* Add ``RuleMaximin.sufficient_coalition_size_um_c_``: number of manipulators that are sufficient (and necessary)
  for UM.

--------------------------------------
0.6.1 (2021-07-27): Accelerate Profile
--------------------------------------

* Accelerate ``Profile``: lazy evaluation of attributes ``preferences_rk``, ``preferences_ut``, ``preferences_borda_rk``
  and ``preferences_borda_ut``. This leads to a very significant acceleration for many methods (typically TM, UM and
  IIA, but also CM and IM for some voting rules).

---------------------------------------
0.6.0 (2021-07-26): ExperimentsCompiler
---------------------------------------

* Add ``ExperimentsCompiler``: draw plots and prepare tables based on the results computed by ``ExperimentAnalyzer``
  on several experiments.
* Accelerate ``RuleMajorityJudgment.necessary_coalition_size_cm_`` and
  ``RuleMajorityJudgment.sufficient_coalition_size_cm_``.

---------------------------------------
0.5.1 (2021-07-24): Fix PyPI deployment
---------------------------------------

* Fix PyPI deployment.

------------------------
0.5.0 (2021-07-24): Meta
------------------------

* This release focuses on "meta" tools that make the simulations easier.

  * Add ``StudyProfileCriteria``: a set of criteria to study for the simulator about the profiles.
  * Add ``StudyRuleCriteria``: a set of criteria to study for the simulator about one or several voting rules.
  * Add ``VotingRuleTasks``: a set of tasks for the simulator, i.e. which voting rules with which options and which
    criteria about them.
  * Add ``ExperimentAnalyzer``: a simulator designed to study small variations of a given profile.

* New features for ``Rule``:

  * Add ``check_option_allowed``: check whether a pair (option, value) is allowed.
  * Add ``cm_power_index_``: CM power index.
  * Add ``elects_condorcet_winner_rk_even_with_cm_``: True if there is a Condorcet winner, she is elected by sincere
    voting and it is not CM.
  * Add ``is_tm_or_um_``: True iff the profile is TM or UM.
  * Add ``log_``: log corresponding to a particular manipulation method.
  * Add ``nb_candidates_cm_``: number of candidates who can benefit from CM.
  * Add ``relative_social_welfare_w_``: relative social welfare of the winner.
  * Add ``worst_relative_welfare_with_cm_``: worst relative social welfare (sincere winner or candidate who can benefit
    from CM).
  * Each rule now has two class attributes ``full_name`` (name of the rule) and ``abbreviation`` (abbreviation of the
    name of the rule). For example, for ``RuleApproval``, it is ``Approval Voting`` and ``AV`` respectively.
  * ``options_parameters`` is now a class attribute.
  * Accelerate the generic brute-force algorithm for exact UM when the rule is based on rankings.

* New features for ``Profile``:

  * Add property ``relative_social_welfare_c``: relative social welfare of each candidate.
  * ``preferences_rk``, ``preferences_ut``, ``preferences_borda_rk`` and ``preferences_borda_ut`` are now properties.

* Minor changes:

  * ``GeneratorProfile`` and all its subclasses now have a parameter ``sort_voters``, which is simply passed to
    ``Profile`` when creating each profile.
  * ``ProfileGeneratorNoisedFile`` is renamed to ``GeneratorProfileNoisedFile``, for the sake of consistency with
    other profile generators.
  * ``RULE_CLASSES`` is renamed to ``ALL_RULE_CLASSES``.
  * Add utility functions ``indent`` and ``pseudo_bool_not``.

---------------------------------
0.4.3 (2021-07-21): Accelerations
---------------------------------

* ``Rule``: accelerate trivial manipulation (TM) for rules based on rankings.
* ``Profile``: compute ``preferences_borda_ut`` only when needed. In particular, if often accelerates trivial
  manipulation (which relies on examining an alternate Profile object, with trivial strategy for manipulators).
* ``RuleMajorityJudgment``, ``RuleRangeVoting`` and ``RuleSTAR``: accelerate the computation of the ballots.

--------------------------------------------------
0.4.2 (2021-07-20): Accelerate plurality_scores_ut
--------------------------------------------------

* Accelerate ``Profile.plurality_scores_ut``. As an example, for a profile with 65,000 voters and 5 candidates,
  the new version is approximately 10 times faster.

--------------------------------------------------------
0.4.1 (2021-07-20): Fix Missing Subpackage in Deployment
--------------------------------------------------------

* Fix bug: in some distributions, some subpackages of Svvamp were not included.

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
