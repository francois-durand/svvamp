import numpy as np


class PluralityEliminationEngine:
    """
    An engine that eliminates candidates and updates the plurality scores.

    This class is used internally by SVVAMP. It is not intended for the end user.

    Parameters
    ----------
    profile: Profile
        The profile.
    """

    def __init__(self, profile):
        self.profile = profile
        self.is_alive = np.ones(profile.n_c, dtype=bool)
        """ndarray: True iff the candidate is still alive."""
        self.loser = None
        """int: The latest eliminated candidate (None at initialization)."""

    def eliminate_candidate(self, loser):
        """
        Eliminate one candidate.

        Parameters
        ----------
        loser: int
            The candidate to eliminate.
        """
        self.loser = loser
        self.is_alive[loser] = False

    def update_scores(self):
        """
        Update the plurality scores.
        """
        raise NotImplementedError

    def eliminate_candidate_and_update_scores(self, loser):
        """
        Eliminate a candidate and update the plurality scores.

        Parameters
        ----------
        loser: int
            The candidate to eliminate.
        """
        self.eliminate_candidate(loser)
        self.update_scores()

    @property
    def scores(self):
        """ndarray: The current plurality scores."""
        raise NotImplementedError

    @property
    def candidates_alive(self):
        """ndarray: List of alive candidates."""
        return np.where(self.is_alive)[0]

    @property
    def nb_candidates_alive(self):
        """int: Number of alive candidates."""
        return np.sum(self.is_alive)
