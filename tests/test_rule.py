from svvamp import Rule, RulePlurality, Profile, RuleVeto, RuleMaximin, RuleExhaustiveBallot, RuleBaldwin, \
    RuleIRVAverage, RuleMajorityJudgment, RuleNanson, RuleRankedPairs, RuleBorda, RuleSchulze, RuleIRVDuels, \
    RuleRangeVoting, RuleKemeny, RuleICRV, RuleCondorcetSumDefeats, OPTIONS


def test_initialize_options():
    """
        >>> rule = Rule(unexpected_option=42)
        Traceback (most recent call last):
        ValueError: ('Unknown option:', 'unexpected_option')
    """
    pass


def test_mean_utility_w():
    """
        >>> profile = Profile(preferences_rk=[
        ...     [0, 1, 2],
        ...     [1, 0, 2],
        ...     [1, 2, 0],
        ...     [2, 1, 0],
        ...     [2, 1, 0],
        ... ])
        >>> rule = RulePlurality()(profile)
        >>> rule.mean_utility_w_
        1.4
    """
    pass


def test_demo_profile():
    """
        >>> profile = Profile(preferences_rk=[
        ...     [0, 1, 2],
        ...     [1, 0, 2],
        ...     [1, 2, 0],
        ...     [2, 1, 0],
        ...     [2, 1, 0],
        ... ])
        >>> rule = Rule()(profile)
        >>> rule.demo_profile_()  # doctest: +NORMALIZE_WHITESPACE
        <BLANKLINE>
        *********************
        *                   *
        *   Profile Class   *
        *                   *
        *********************
        <BLANKLINE>
        ************************
        *   Basic properties   *
        ************************
        n_v = 5
        n_c = 3
        labels_candidates = ['0', '1', '2']
        preferences_ut =
        [[2 1 0]
         [1 2 0]
         [0 2 1]
         [0 1 2]
         [0 1 2]]
        preferences_borda_ut =
        [[2. 1. 0.]
         [1. 2. 0.]
         [0. 2. 1.]
         [0. 1. 2.]
         [0. 1. 2.]]
        preferences_borda_rk =
        [[2 1 0]
         [1 2 0]
         [0 2 1]
         [0 1 2]
         [0 1 2]]
        preferences_rk =
        [[0 1 2]
         [1 0 2]
         [1 2 0]
         [2 1 0]
         [2 1 0]]
        PROFILE: Compute v_has_same_ordinal_preferences_as_previous_voter
        v_has_same_ordinal_preferences_as_previous_voter =
        [False False False False  True]
        <BLANKLINE>
        ************************
        *   Plurality scores   *
        ************************
        preferences_rk (reminder) =
        [[0 1 2]
         [1 0 2]
         [1 2 0]
         [2 1 0]
         [2 1 0]]
        PROFILE: Compute Plurality scores (rk)
        plurality_scores_rk = [1 2 2]
        PROFILE: Compute majority favorite (rk)
        majority_favorite_rk = nan
        PROFILE: Compute majority favorite (rk_ctb)
        majority_favorite_rk_ctb = nan
        <BLANKLINE>
        preferences_borda_ut (reminder) =
        [[2. 1. 0.]
         [1. 2. 0.]
         [0. 2. 1.]
         [0. 1. 2.]
         [0. 1. 2.]]
        PROFILE: Compute Plurality scores (ut)
        plurality_scores_ut = [1 2 2]
        PROFILE: Compute majority favorite (ut)
        majority_favorite_ut = nan
        PROFILE: Compute majority favorite (ut_ctb)
        majority_favorite_ut_ctb = nan
        <BLANKLINE>
        ********************
        *   Borda scores   *
        ********************
        preferences_borda_rk (reminder) =
        [[2 1 0]
         [1 2 0]
         [0 2 1]
         [0 1 2]
         [0 1 2]]
        PROFILE: Compute Borda scores of the candidates (rankings)
        PROFILE: Compute matrix of duels (with strict orders)
        borda_score_c_rk =
        [3 7 5]
        Remark: Borda scores above are computed with the matrix of duels.
        Check: np.sum(self.preferences_borda_rk, 0) =
        [3 7 5]
        PROFILE: Compute decreasing_borda_scores_rk
        decreasing_borda_scores_rk =
        [7 5 3]
        PROFILE: Compute candidates_by_decreasing_borda_score_rk
        candidates_by_decreasing_borda_score_rk =
        [1 2 0]
        <BLANKLINE>
        preferences_borda_ut (reminder) =
        [[2. 1. 0.]
         [1. 2. 0.]
         [0. 2. 1.]
         [0. 1. 2.]
         [0. 1. 2.]]
        PROFILE: Compute Borda scores of the candidates (weak orders)
        borda_score_c_ut =
        [3. 7. 5.]
        PROFILE: Compute decreasing_borda_scores_ut
        decreasing_borda_scores_ut =
        [7. 5. 3.]
        PROFILE: Compute candidates_by_decreasing_borda_score_ut
        candidates_by_decreasing_borda_score_ut =
        [1 2 0]
        <BLANKLINE>
        *****************
        *   Utilities   *
        *****************
        preferences_ut (reminder) =
        [[2 1 0]
         [1 2 0]
         [0 2 1]
         [0 1 2]
         [0 1 2]]
        PROFILE: Compute total utility of candidates
        total_utility_c =
        [3 7 5]
        PROFILE: Compute total_utility_min
        total_utility_min = 3
        PROFILE: Compute total_utility_max
        total_utility_max = 7
        PROFILE: Compute total_utility_mean
        total_utility_mean = 5.0
        PROFILE: Compute total_utility_std
        total_utility_std = 1.632993161855452
        <BLANKLINE>
        *******************************************
        *   Condorcet notions based on rankings   *
        *******************************************
        preferences_rk (reminder) =
        [[0 1 2]
         [1 0 2]
         [1 2 0]
         [2 1 0]
         [2 1 0]]
        matrix_duels_rk =
        [[0 1 2]
         [4 0 3]
         [3 2 0]]
        PROFILE: Compute matrix_victories_rk
        matrix_victories_rk =
        [[0. 0. 0.]
         [1. 0. 1.]
         [1. 0. 0.]]
        PROFILE: Compute condorcet_winner_rk
        condorcet_winner_rk = 1
        PROFILE: Compute exists_condorcet_order_rk
        exists_condorcet_order_rk = True
        PROFILE: Compute matrix_victories_rk_ctb
        matrix_victories_rk_ctb =
        [[0. 0. 0.]
         [1. 0. 1.]
         [1. 0. 0.]]
        PROFILE: Compute condorcet_winner_rk_ctb
        condorcet_winner_rk_ctb = 1
        PROFILE: Compute exists_condorcet_order_rk_ctb
        exists_condorcet_order_rk_ctb = True
        <BLANKLINE>
        ***************************************
        *   Relative Condorcet notions (ut)   *
        ***************************************
        preferences_borda_ut (reminder) =
        [[2. 1. 0.]
         [1. 2. 0.]
         [0. 2. 1.]
         [0. 1. 2.]
         [0. 1. 2.]]
        PROFILE: Compute matrix of duels
        matrix_duels_ut =
        [[0 1 2]
         [4 0 3]
         [3 2 0]]
        PROFILE: Compute matrix_victories_ut_rel
        matrix_victories_ut_rel =
        [[0. 0. 0.]
         [1. 0. 1.]
         [1. 0. 0.]]
        PROFILE: Compute condorcet_winner_ut_rel
        condorcet_winner_ut_rel = 1
        PROFILE: Compute exists_condorcet_order_ut_rel
        exists_condorcet_order_ut_rel = True
        PROFILE: Compute matrix_victories_ut_rel_ctb
        matrix_victories_ut_rel_ctb =
        [[0. 0. 0.]
         [1. 0. 1.]
         [1. 0. 0.]]
        PROFILE: Compute condorcet_winner_ut_rel_ctb
        condorcet_winner_ut_rel_ctb = 1
        PROFILE: Compute exists_condorcet_order_ut_rel_ctb
        exists_condorcet_order_ut_rel_ctb = True
        <BLANKLINE>
        ***************************************
        *   Absolute Condorcet notions (ut)   *
        ***************************************
        matrix_duels_ut (reminder) =
        [[0 1 2]
         [4 0 3]
         [3 2 0]]
        PROFILE: Compute matrix_victories_ut_abs
        matrix_victories_ut_abs =
        [[0. 0. 0.]
         [1. 0. 1.]
         [1. 0. 0.]]
        PROFILE: Compute Condorcet-admissible candidates
        condorcet_admissible_candidates =
        [False  True False]
        PROFILE: Compute number of Condorcet-admissible candidates
        nb_condorcet_admissible = 1
        PROFILE: Compute weak Condorcet winners
        weak_condorcet_winners =
        [False  True False]
        PROFILE: Compute number of weak Condorcet winners
        nb_weak_condorcet_winners = 1
        PROFILE: Compute Condorcet winner
        condorcet_winner_ut_abs = 1
        PROFILE: Compute exists_condorcet_order_ut_abs
        exists_condorcet_order_ut_abs = True
        PROFILE: Compute Resistant Condorcet winner
        resistant_condorcet_winner = nan
        PROFILE: Compute threshold_c_prevents_w_Condorcet_ut_abs
        threshold_c_prevents_w_Condorcet_ut_abs =
        [[0 0 1]
         [1 0 2]
         [0 1 0]]
        PROFILE: Compute matrix_victories_ut_abs_ctb
        matrix_victories_ut_abs_ctb =
        [[0. 0. 0.]
         [1. 0. 1.]
         [1. 0. 0.]]
        PROFILE: Compute condorcet_winner_ut_abs_ctb
        condorcet_winner_ut_abs_ctb = 1
        PROFILE: Compute exists_condorcet_order_ut_abs_ctb
        exists_condorcet_order_ut_abs_ctb = True
        <BLANKLINE>
        **********************************************
        *   Implications between Condorcet notions   *
        **********************************************
        maj_fav_ut (False)             ==>            maj_fav_ut_ctb (False)
         ||          ||                                     ||           ||
         ||          V                                      V            ||
         ||         maj_fav_rk (False) ==> maj_fav_rk_ctb (False)        ||
         V                         ||       ||                           ||
        Resistant Condorcet (False)                                      ||
         ||                        ||       ||                           ||
         V                         ||       ||                           V
        Condorcet_ut_abs (True)        ==>      Condorcet_ut_abs_ctb (True)
         ||          ||            ||       ||              ||           ||
         ||          V             V        V               V            ||
         ||       Condorcet_rk (True)  ==> Condorcet_rk_ctb (True)       ||
         V                                                               V
        Condorcet_ut_rel (True)        ==>      Condorcet_ut_rel_ctb (True)
         ||
         V
        Weak Condorcet (True)
         ||
         V
        Condorcet-admissible (True)
    """
    pass


def test_iia_subset_maximum_size_setter():
    """
        >>> rule = Rule()
        >>> rule.iia_subset_maximum_size = 'a'
        Traceback (most recent call last):
        ValueError: Unknown value for iia_subset_maximum_size: a (number or np.inf expected).
    """
    pass


def test_im_option_setter():
    """
        >>> rule = Rule()
        >>> rule.im_option = 'a'
        Traceback (most recent call last):
        ValueError: Unknown value for im_option: a
    """
    pass


def test_tm_option_setter():
    """
        >>> rule = Rule()
        >>> rule.tm_option = 'a'
        Traceback (most recent call last):
        ValueError: Unknown value for tm_option: a
    """
    pass


def test_um_option_setter():
    """
        >>> rule = Rule()
        >>> rule.um_option = 'a'
        Traceback (most recent call last):
        ValueError: Unknown value for um_option: a
    """
    pass


def test_icm_option_setter():
    """
        >>> rule = Rule()
        >>> rule.icm_option = 'a'
        Traceback (most recent call last):
        ValueError: Unknown value for icm_option: a
    """
    pass


def test_cm_option_setter():
    """
        >>> rule = Rule()
        >>> rule.cm_option = 'a'
        Traceback (most recent call last):
        ValueError: Unknown value for cm_option: a
    """
    pass


def test_is_not_iia():
    """
        >>> profile = Profile(preferences_rk=[
        ...     [0, 3, 2, 1],
        ...     [1, 0, 2, 3],
        ...     [2, 0, 1, 3],
        ...     [2, 1, 0, 3],
        ...     [3, 1, 2, 0],
        ... ])
        >>> rule = RulePlurality()(profile)
        >>> rule.is_not_iia_
        nan
    """
    pass


def test_compute_iia():
    """
        >>> profile = Profile(preferences_rk=[
        ...     [0, 1, 2],
        ...     [0, 2, 1],
        ...     [1, 0, 2],
        ...     [2, 1, 0],
        ...     [2, 1, 0],
        ... ])
        >>> rule = RuleVeto()(profile)
        >>> rule.is_iia_
        False

        >>> profile = Profile(preferences_rk=[
        ...     [0, 1, 2],
        ...     [0, 1, 2],
        ...     [1, 0, 2],
        ...     [1, 2, 0],
        ...     [1, 2, 0],
        ... ])
        >>> rule = RulePlurality()(profile)
        >>> rule.is_iia_
        True

        >>> profile = Profile(preferences_ut=[
        ...     [ 1. , -1. ,  0. ],
        ...     [ 1. ,  1. ,  0.5],
        ...     [-1. , -1. ,  0. ],
        ...     [ 1. ,  0.5,  1. ],
        ...     [ 0. , -1. ,  0. ],
        ... ], preferences_rk=[
        ...     [0, 2, 1],
        ...     [1, 0, 2],
        ...     [2, 0, 1],
        ...     [2, 0, 1],
        ...     [2, 0, 1],
        ... ])
        >>> rule = RuleMaximin()(profile)
        >>> rule.is_iia_
        True

        >>> profile = Profile(preferences_ut=[
        ...     [ 0.5,  0.5,  0. ],
        ...     [-0.5,  0.5,  0.5],
        ...     [-1. ,  0.5,  0. ],
        ...     [-1. ,  1. ,  1. ],
        ...     [-1. , -0.5, -0.5],
        ... ], preferences_rk=[
        ...     [1, 0, 2],
        ...     [1, 2, 0],
        ...     [1, 2, 0],
        ...     [2, 1, 0],
        ...     [2, 1, 0],
        ... ])
        >>> rule = RuleExhaustiveBallot()(profile)
        >>> rule.is_iia_
        True

        >>> profile = Profile(preferences_rk=[
        ...     [0, 1, 4, 2, 3],
        ...     [0, 3, 2, 4, 1],
        ...     [1, 0, 4, 3, 2],
        ...     [1, 0, 4, 3, 2],
        ...     [2, 0, 4, 3, 1],
        ...     [4, 3, 1, 2, 0],
        ... ])
        >>> rule = RuleBaldwin()(profile)
        >>> rule.is_iia_
        True

        >>> profile = Profile(preferences_rk=[
        ...     [0, 1, 2, 4, 3],
        ...     [0, 1, 3, 2, 4],
        ...     [0, 3, 2, 4, 1],
        ...     [1, 4, 3, 2, 0],
        ...     [2, 0, 1, 4, 3],
        ...     [4, 2, 3, 1, 0],
        ... ])
        >>> rule = RuleIRVAverage()(profile)
        >>> rule.is_iia_
        True

        >>> profile = Profile(preferences_ut=[
        ...     [ 0. ,  0. , -0.5],
        ...     [ 0.5, -1. ,  0. ],
        ...     [-0.5, -1. , -1. ],
        ...     [ 0. ,  0.5, -1. ],
        ...     [-1. ,  0.5, -1. ],
        ...     [-1. , -0.5,  0.5],
        ... ], preferences_rk=[
        ...     [0, 1, 2],
        ...     [0, 2, 1],
        ...     [0, 2, 1],
        ...     [1, 0, 2],
        ...     [1, 2, 0],
        ...     [2, 1, 0],
        ... ])
        >>> rule = RulePlurality()(profile)
        >>> rule.is_iia_
        True

        >>> profile = Profile(preferences_ut=[
        ...     [ 0. , -0.5,  0. ],
        ...     [ 0.5,  1. , -1. ],
        ...     [-0.5,  0. , -1. ],
        ...     [ 0.5,  0.5,  1. ],
        ...     [ 0. , -0.5,  0.5],
        ...     [-1. , -1. ,  1. ],
        ... ], preferences_rk=[
        ...     [0, 2, 1],
        ...     [1, 0, 2],
        ...     [1, 0, 2],
        ...     [2, 0, 1],
        ...     [2, 0, 1],
        ...     [2, 1, 0],
        ... ])
        >>> rule = RuleMaximin()(profile)
        >>> rule.is_iia_
        True
    """
    pass


def test_compute_iia_aux():
    """
        >>> profile = Profile(preferences_rk=[
        ...     [0, 1, 2],
        ...     [1, 0, 2],
        ...     [2, 0, 1],
        ...     [2, 0, 1],
        ...     [2, 1, 0],
        ... ])
        >>> rule = RuleMajorityJudgment()(profile)
        >>> rule.is_iia_
        True

        >>> profile = Profile(preferences_rk=[
        ...     [0, 1, 2],
        ...     [0, 2, 1],
        ...     [1, 0, 2],
        ...     [2, 1, 0],
        ...     [2, 1, 0],
        ... ])
        >>> rule = RuleMajorityJudgment()(profile)
        >>> rule.is_iia_
        False
    """
    pass


def test_voters_im():
    """
        >>> profile = Profile(preferences_rk=[
        ...     [0, 1, 2],
        ...     [1, 0, 2],
        ...     [2, 0, 1],
        ...     [2, 0, 1],
        ...     [2, 1, 0],
        ... ])
        >>> rule = RulePlurality()(profile)
        >>> rule.voters_im_
        array([0., 0., 0., 0., 0.])
    """
    pass


def test_im_main_work_v_exact_rankings():
    """
        >>> profile = Profile(preferences_rk=[
        ...     [0, 1, 2],
        ...     [0, 2, 1],
        ...     [1, 0, 2],
        ...     [1, 2, 0],
        ...     [2, 1, 0],
        ... ])
        >>> rule = RuleNanson(im_option='exact')(profile)
        >>> rule.is_im_
        True

        >>> profile = Profile(preferences_rk=[
        ...     [0, 1, 2],
        ...     [0, 2, 1],
        ...     [1, 2, 0],
        ...     [1, 2, 0],
        ...     [2, 0, 1],
        ... ])
        >>> rule = RuleNanson(im_option='exact')(profile)
        >>> rule.v_im_for_c_
        array([[0., 0., 0.],
               [0., 0., 0.],
               [0., 0., 1.],
               [0., 0., 1.],
               [0., 0., 0.]])

        >>> profile = Profile(preferences_rk=[
        ...     [0, 1, 2],
        ...     [1, 0, 2],
        ...     [1, 0, 2],
        ...     [2, 0, 1],
        ...     [2, 1, 0],
        ... ])
        >>> rule = RuleNanson(im_option='exact')(profile)
        >>> rule.v_im_for_c_
        array([[1., 0., 0.],
               [0., 0., 0.],
               [0., 0., 0.],
               [0., 0., 0.],
               [0., 0., 0.]])
        """
    pass


def test_compute_im():
    """
        >>> profile = Profile(preferences_rk=[
        ...     [0, 1, 2],
        ...     [1, 0, 2],
        ...     [2, 0, 1],
        ...     [2, 0, 1],
        ...     [2, 1, 0],
        ... ])
        >>> rule = RuleNanson()(profile)
        >>> rule.voters_im_
        array([nan, nan,  0.,  0.,  0.])

        >>> profile = Profile(preferences_ut=[
        ...     [ 1. ,  1. ,  0. , -0.5],
        ...     [ 0.5,  1. , -0.5,  0. ],
        ...     [ 0. , -1. ,  1. , -1. ],
        ...     [-0.5,  0. ,  0. , -0.5],
        ... ], preferences_rk=[
        ...     [0, 1, 2, 3],
        ...     [1, 0, 3, 2],
        ...     [2, 0, 3, 1],
        ...     [2, 1, 3, 0],
        ... ])
        >>> rule = RuleIRVAverage(im_option='exact')(profile)
        >>> rule.voters_im_
        array([1., 1., 0., 0.])

        >>> profile = Profile(preferences_ut=[
        ...     [ 0.5, -0.5, -0.5, -1. ],
        ...     [ 1. ,  0.5,  0.5, -0.5],
        ...     [ 0.5,  0. ,  1. , -1. ],
        ...     [-0.5,  0. ,  0.5, -0.5],
        ...     [ 0.5,  0.5,  0.5,  1. ],
        ...     [-1. , -1. ,  0. ,  0.5],
        ... ], preferences_rk=[
        ...     [0, 1, 2, 3],
        ...     [0, 2, 1, 3],
        ...     [2, 0, 1, 3],
        ...     [2, 1, 3, 0],
        ...     [3, 1, 0, 2],
        ...     [3, 2, 0, 1],
        ... ])
        >>> rule = RuleCondorcetSumDefeats(im_option='exact')(profile)
        >>> rule.candidates_im_
        array([1., 0., 0., 0.])
    """
    pass


def test_tm_preliminary_checks_general():
    """
        >>> profile = Profile(preferences_rk=[
        ...     [1, 2, 0],
        ...     [2, 1, 0],
        ...     [2, 1, 0],
        ...     [2, 1, 0],
        ...     [2, 1, 0],
        ... ])
        >>> rule = RuleRankedPairs()(profile)
        >>> rule.candidates_tm_
        array([0., 0., 0.])

        >>> profile = Profile(preferences_rk=[
        ...     [1, 0, 2],
        ...     [1, 0, 2],
        ...     [1, 2, 0],
        ...     [1, 2, 0],
        ...     [1, 2, 0],
        ... ])
        >>> rule = RuleBorda()(profile)
        >>> rule.candidates_tm_
        array([0., 0., 0.])

        >>> profile = Profile(preferences_ut=[
        ...     [ 0. , -0.5,  1. , -1. ],
        ...     [ 1. ,  0.5,  1. ,  0.5],
        ...     [ 0. , -0.5, -1. ,  1. ],
        ... ], preferences_rk=[
        ...     [2, 0, 1, 3],
        ...     [2, 0, 1, 3],
        ...     [3, 0, 1, 2],
        ... ])
        >>> rule = RuleSchulze()(profile)
        >>> rule.candidates_tm_
        array([0., 0., 0., 0.])

        >>> profile = Profile(preferences_ut=[
        ...     [ 1. ,  0.5, -0.5,  0. ],
        ...     [ 1. , -1. , -0.5, -0.5],
        ...     [ 0. , -1. , -1. ,  0. ],
        ...     [ 0. ,  0.5,  0.5,  0. ],
        ...     [-1. , -1. ,  0.5,  0. ],
        ...     [-0.5,  0. ,  1. ,  0.5],
        ... ], preferences_rk=[
        ...     [0, 1, 3, 2],
        ...     [0, 2, 3, 1],
        ...     [0, 3, 2, 1],
        ...     [1, 2, 0, 3],
        ...     [2, 3, 0, 1],
        ...     [2, 3, 1, 0],
        ... ])
        >>> rule = RuleMaximin()(profile)
        >>> rule.candidates_tm_
        array([0., 0., 0., 0.])

        >>> profile = Profile(preferences_ut=[
        ...     [-0.5, -1. ],
        ...     [ 0. ,  0.5],
        ... ], preferences_rk=[
        ...     [0, 1],
        ...     [1, 0],
        ... ])
        >>> rule = RuleIRVDuels()(profile)
        >>> rule.candidates_tm_
        array([0., 0.])
    """
    pass


def test_tm_main_work_c_lazy():
    """
        >>> profile = Profile(preferences_rk=[
        ...     [0, 1, 2],
        ...     [1, 0, 2],
        ...     [1, 2, 0],
        ...     [2, 0, 1],
        ...     [2, 1, 0],
        ... ])
        >>> rule = RuleNanson(tm_option='lazy')(profile)
        >>> rule.is_tm_
        nan
    """
    pass


def test_um_preliminary_checks_general():
    """
        >>> profile = Profile(preferences_rk=[
        ...     [0, 1, 2],
        ...     [0, 2, 1],
        ...     [1, 0, 2],
        ...     [1, 2, 0],
        ...     [1, 2, 0],
        ... ])
        >>> rule = RuleNanson()(profile)
        >>> rule.candidates_um_
        array([0., 0., 0.])

        >>> profile = Profile(preferences_rk=[
        ...     [0, 3, 2, 1],
        ...     [1, 2, 0, 3],
        ...     [1, 2, 3, 0],
        ...     [2, 0, 1, 3],
        ...     [3, 2, 0, 1],
        ... ])
        >>> rule = RuleSchulze()(profile)
        >>> rule.candidates_um_
        array([ 0.,  1.,  0., nan])

        >>> profile = Profile(preferences_ut=[
        ...     [ 0.5, -0.5,  0.5],
        ...     [ 0. ,  0.5, -1. ],
        ...     [ 1. ,  1. ,  0.5],
        ...     [-1. ,  0. , -0.5],
        ...     [ 0. ,  0.5,  1. ],
        ... ], preferences_rk=[
        ...     [0, 2, 1],
        ...     [1, 0, 2],
        ...     [1, 0, 2],
        ...     [1, 2, 0],
        ...     [2, 1, 0],
        ... ])
        >>> rule = RuleMaximin()(profile)
        >>> rule.candidates_um_
        array([0., 0., 0.])

        >>> profile = Profile(preferences_ut=[
        ...     [ 0. ,  0. ,  0. ],
        ...     [ 0.5,  0. , -0.5],
        ...     [ 0. , -1. ,  0. ],
        ...     [ 0. ,  0.5,  0. ],
        ...     [-0.5,  1. , -0.5],
        ...     [-0.5, -0.5,  0. ],
        ... ], preferences_rk=[
        ...     [0, 1, 2],
        ...     [0, 1, 2],
        ...     [0, 2, 1],
        ...     [1, 0, 2],
        ...     [1, 0, 2],
        ...     [2, 1, 0],
        ... ])
        >>> rule = RuleSchulze()(profile)
        >>> rule.candidates_um_
        array([0., 0., 0.])

        >>> profile = Profile(preferences_ut=[
        ...     [ 1. , -0.5, -1. ],
        ...     [ 1. , -1. , -0.5],
        ...     [ 1. ,  0. ,  0. ],
        ...     [-0.5,  0. , -0.5],
        ...     [-0.5,  0. , -0.5],
        ...     [-1. ,  0.5,  1. ],
        ... ], preferences_rk=[
        ...     [0, 1, 2],
        ...     [0, 2, 1],
        ...     [0, 2, 1],
        ...     [1, 0, 2],
        ...     [1, 0, 2],
        ...     [2, 1, 0],
        ... ])
        >>> rule = RuleExhaustiveBallot()(profile)
        >>> rule.candidates_um_
        array([0., 0., 0.])

        >>> profile = Profile(preferences_rk=[
        ...     [0, 1, 3, 2],
        ...     [0, 2, 1, 3],
        ...     [1, 0, 3, 2],
        ...     [2, 0, 3, 1],
        ...     [3, 0, 1, 2],
        ... ])
        >>> rule = RuleICRV()(profile)
        >>> rule.is_um_
        False
    """
    pass


def test_um_preliminary_checks_c():
    """
        >>> profile = Profile(preferences_rk=[
        ...     [0, 1, 3, 2],
        ...     [0, 2, 3, 1],
        ...     [1, 0, 2, 3],
        ...     [1, 3, 2, 0],
        ...     [3, 1, 2, 0],
        ... ])
        >>> rule = RuleBaldwin()(profile)
        >>> rule.candidates_um_
        array([nan,  0.,  0., nan])

        >>> profile = Profile(preferences_ut=[
        ...     [ 1. ,  0.5,  0.5],
        ...     [-0.5,  0.5, -0.5],
        ...     [ 0.5,  1. ,  0.5],
        ...     [ 0. ,  0. , -1. ],
        ...     [ 0.5, -1. ,  1. ],
        ...     [ 0.5, -1. ,  0.5],
        ... ], preferences_rk=[
        ...     [0, 2, 1],
        ...     [1, 0, 2],
        ...     [1, 0, 2],
        ...     [1, 0, 2],
        ...     [2, 0, 1],
        ...     [2, 0, 1],
        ... ])
        >>> rule = RuleExhaustiveBallot()(profile)
        >>> rule.candidates_um_
        array([ 1.,  0., nan])
    """
    pass


def test_um_main_work_c_exact():
    """
        >>> profile = Profile(preferences_rk=[
        ...     [0, 2, 1],
        ...     [0, 2, 1],
        ...     [1, 2, 0],
        ...     [1, 2, 0],
        ...     [2, 1, 0],
        ... ])
        >>> rule = RuleNanson(um_option='exact')(profile)
        >>> rule.is_um_
        True
    """
    pass


def test_icm_preliminary_checks_general():
    """
        >>> profile = Profile(preferences_rk=[
        ...     [0, 2, 1],
        ...     [1, 2, 0],
        ...     [1, 2, 0],
        ...     [2, 0, 1],
        ...     [2, 1, 0],
        ... ])
        >>> rule = RulePlurality()(profile)
        >>> rule.is_icm_c_(1)
        False

        >>> profile = Profile(preferences_rk=[
        ...     [0, 1, 2],
        ...     [0, 2, 1]
        ... ])
        >>> rule = RuleVeto()(profile)
        >>> rule.is_icm_
        False
    """
    pass


def test_icm_preliminary_checks_c():
    """
        >>> profile = Profile(preferences_rk=[
        ...     [0, 2, 1],
        ...     [0, 2, 1],
        ...     [1, 0, 2],
        ...     [2, 1, 0],
        ...     [2, 1, 0],
        ... ])
        >>> rule = RulePlurality()(profile)
        >>> rule.is_icm_c_(1)
        True

        >>> profile = Profile(preferences_rk=[
        ...     [0, 1, 2],
        ...     [1, 0, 2],
        ...     [1, 2, 0],
        ...     [2, 0, 1],
        ...     [2, 0, 1],
        ... ])
        >>> rule = RuleRangeVoting()(profile)
        >>> rule.is_icm_c_(1)
        False

        >>> profile = Profile(preferences_rk=[
        ...     [0, 1, 2, 3],
        ...     [1, 0, 3, 2],
        ...     [1, 2, 0, 3],
        ...     [1, 3, 0, 2],
        ...     [2, 0, 1, 3],
        ...     [2, 0, 3, 1],
        ... ])
        >>> rule = RuleExhaustiveBallot()(profile)
        >>> rule.is_icm_
        True
    """
    pass


def test_candidates_icm():
    """
        >>> profile = Profile(preferences_rk=[
        ...     [0, 2, 1],
        ...     [1, 0, 2],
        ...     [1, 0, 2],
        ...     [2, 0, 1],
        ...     [2, 1, 0],
        ... ])
        >>> rule = RulePlurality()(profile)
        >>> rule.candidates_icm_
        array([0., 0., 1.])
    """
    pass


def test_icm_conclude_c():
    """
        >>> profile = Profile(preferences_rk=[
        ...     [0, 1, 2],
        ...     [0, 2, 1],
        ...     [1, 0, 2],
        ...     [2, 1, 0],
        ...     [2, 1, 0],
        ... ])
        >>> rule = RuleBorda()(profile)
        >>> rule.is_icm_c_(1)
        True
    """
    pass


def test_compute_icm():
    """
        >>> profile = Profile(preferences_rk=[
        ...     [0, 1, 2],
        ...     [0, 2, 1],
        ...     [1, 0, 2],
        ...     [1, 0, 2],
        ...     [2, 1, 0],
        ... ])
        >>> rule = RuleBorda()(profile)
        >>> rule.is_icm_
        True
    """
    pass


def test_cm_preliminary_checks_general():
    """
        >>> profile = Profile(preferences_rk=[
        ...     [0, 1, 2],
        ...     [0, 1, 2],
        ...     [0, 1, 2],
        ...     [0, 2, 1],
        ...     [0, 2, 1],
        ... ])
        >>> rule = RuleMajorityJudgment()(profile)
        >>> rule.is_cm_
        False

        >>> profile = Profile(preferences_rk=[
        ...     [0, 2, 3, 1],
        ...     [1, 2, 3, 0],
        ...     [2, 0, 3, 1],
        ...     [2, 1, 3, 0],
        ...     [3, 2, 1, 0],
        ... ])
        >>> rule = RuleBaldwin()(profile)
        >>> rule.is_cm_
        False

        >>> profile = Profile(preferences_ut=[
        ...     [ 0.5, -0.5,  0. ],
        ...     [-0.5,  0.5, -1. ],
        ...     [-0.5,  0. ,  0. ],
        ...     [-0.5,  1. ,  0. ],
        ...     [-1. , -0.5,  1. ],
        ... ], preferences_rk=[
        ...     [0, 2, 1],
        ...     [1, 0, 2],
        ...     [1, 2, 0],
        ...     [1, 2, 0],
        ...     [2, 1, 0],
        ... ])
        >>> rule = RuleKemeny()(profile)
        >>> rule.is_cm_
        False

        >>> profile = Profile(preferences_rk=[
        ...     [0, 2, 1],
        ...     [0, 2, 1],
        ...     [0, 2, 1],
        ...     [1, 2, 0],
        ...     [1, 2, 0],
        ...     [2, 1, 0],
        ... ])
        >>> rule = RuleSchulze()(profile)
        >>> rule.is_cm_
        False

        >>> profile = Profile(preferences_rk=[
        ...     [0, 1, 2],
        ...     [1, 2, 0],
        ...     [1, 2, 0],
        ...     [2, 0, 1],
        ...     [2, 0, 1],
        ...     [2, 1, 0],
        ... ])
        >>> rule = RuleIRVAverage()(profile)
        >>> rule.is_cm_
        True

        >>> profile = Profile(preferences_ut=[
        ...     [ 1. , -1. , -1. ],
        ...     [ 1. ,  0.5, -1. ],
        ...     [ 0.5, -0.5,  0.5],
        ...     [-0.5,  0. , -0.5],
        ...     [ 0.5,  1. ,  0.5],
        ...     [-0.5,  0.5,  0.5],
        ... ], preferences_rk=[
        ...     [0, 1, 2],
        ...     [0, 1, 2],
        ...     [0, 2, 1],
        ...     [1, 0, 2],
        ...     [1, 0, 2],
        ...     [1, 2, 0],
        ... ])
        >>> rule = RuleIRVDuels()(profile)
        >>> rule.is_cm_
        False
    """
    pass


def test_cm_preliminary_checks_c():
    """
        >>> profile = Profile(preferences_rk=[
        ...     [1, 3, 0, 2],
        ...     [2, 1, 0, 3],
        ...     [2, 3, 1, 0],
        ...     [3, 0, 1, 2],
        ...     [3, 0, 1, 2],
        ... ])
        >>> rule = RuleNanson()(profile)
        >>> rule.is_cm_
        nan

        >>> profile = Profile(preferences_rk=[
        ...     [0, 2, 1],
        ...     [1, 0, 2],
        ...     [1, 2, 0],
        ...     [2, 0, 1],
        ...     [2, 1, 0],
        ... ])
        >>> rule = RuleBaldwin()(profile)
        >>> rule.is_cm_c_(1)
        True
    """
    pass


def test_cm_main_work_c_exact():
    """
        >>> profile = Profile(preferences_rk=[
        ...     [0, 1, 2],
        ...     [0, 2, 1],
        ...     [1, 0, 2],
        ...     [1, 0, 2],
        ...     [2, 1, 0],
        ... ])
        >>> rule = RuleNanson(cm_option='exact')(profile)
        >>> rule.is_cm_
        True
    """
    pass


# noinspection PyProtectedMember
def test_reached_uncovered_code():
    """
        >>> OPTIONS.ERROR_WHEN_UNCOVERED_CODE = True
        >>> profile = Profile(preferences_rk=[[0, 1, 2], [1, 0, 2]])
        >>> rule = RulePlurality()(profile)
        >>> rule._example_reached_uncovered_code()
        Traceback (most recent call last):
        AssertionError: Uncovered portion of code.
    """
    pass


def test_is_im_c_with_voters():
    """
        >>> profile = Profile(preferences_ut=[
        ...     [ 1. ,  0.5,  0. ],
        ...     [-1. ,  1. ,  0. ],
        ...     [-1. , -0.5,  0. ],
        ...     [-0.5, -0.5,  0. ],
        ...     [-0.5, -0.5,  1. ],
        ... ])
        >>> rule = RuleRangeVoting()(profile)
        >>> rule.is_im_v_with_candidates_(0)
        (False, array([0., 0., 0.]))
        >>> rule.is_im_c_with_voters_(0)
        (False, array([0., 0., 0., 0., 0.]))
    """
    pass


def test_temp():
    """

    """
    pass
