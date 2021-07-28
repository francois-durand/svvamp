import numpy as np
from svvamp.preferences.plurality_elimination_engine import PluralityEliminationEngine


class PluralityEliminationEngineProfileUM(PluralityEliminationEngine):

    def __init__(self, profile):
        super().__init__(profile)

        self.preferences_borda_rk_s = profile.profile_s.preferences_borda_rk.copy()
        self.preference_borda_rk_m = profile.ballot_borda_rk.copy()

        self.ballots_s = np.argmax(self.preferences_borda_rk_s, axis=1)
        self.ballot_m = profile.ballot_rk[0]

        self._scores = profile.profile_s.plurality_scores_rk.copy().astype(float)
        self._scores[self.ballot_m] += profile.n_m

    def update_scores(self):
        self.preferences_borda_rk_s[:, self.loser] = -1
        new_ballots_s = np.argmax(self.preferences_borda_rk_s[self.ballots_s == self.loser, :], axis=1)
        self.ballots_s[self.ballots_s == self.loser] = new_ballots_s
        self._scores += np.bincount(new_ballots_s, minlength=self.profile.n_c)

        self.preference_borda_rk_m[self.loser] = -1
        if self.ballot_m == self.loser:
            self.ballot_m = np.argmax(self.preference_borda_rk_m)
            self.scores[self.ballot_m] += self.profile.n_m

        self.scores[self.loser] = np.nan

    @property
    def scores(self):
        return self._scores
