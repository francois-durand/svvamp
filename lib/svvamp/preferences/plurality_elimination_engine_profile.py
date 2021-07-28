import numpy as np
from svvamp.preferences.plurality_elimination_engine import PluralityEliminationEngine


class PluralityEliminationEngineProfile(PluralityEliminationEngine):

    def __init__(self, profile):
        super().__init__(profile)
        self.preferences_borda_rk = profile.preferences_borda_rk.copy()
        self.ballots = profile.preferences_rk[:, 0].copy()
        self._scores = profile.plurality_scores_rk.copy().astype(float)

    def update_scores(self):
        self.preferences_borda_rk[:, self.loser] = -1
        new_ballots = np.argmax(self.preferences_borda_rk[self.ballots == self.loser, :], axis=1)
        self.ballots[self.ballots == self.loser] = new_ballots
        self._scores += np.bincount(new_ballots, minlength=self.profile.n_c)
        self._scores[self.loser] = np.nan

    @property
    def scores(self):
        return self._scores
