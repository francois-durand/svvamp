import numpy as np


class PluralityEliminationEngine():

    def __init__(self, profile):
        self.profile = profile
        self.is_alive = np.ones(profile.n_c, dtype=bool)
        self.loser = None

    def eliminate_candidate(self, loser):
        self.loser = loser
        self.is_alive[loser] = False

    def update_scores(self):
        raise NotImplementedError

    def eliminate_candidate_and_update_scores(self, loser):
        self.eliminate_candidate(loser)
        self.update_scores()

    @property
    def scores(self):
        raise NotImplementedError

    @property
    def candidates_alive(self):
        return np.where(self.is_alive)[0]

    @property
    def nb_candidates_alive(self):
        return np.sum(self.is_alive)
