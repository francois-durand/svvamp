from svvamp import ProfileGeneratorNoisedFile
import os


def test():
    file = os.path.join(os.path.dirname(__file__), 'example_ballots.t.csv')
    profile = ProfileGeneratorNoisedFile(file_name=file, relative_noise=.01, absolute_noise=.01)()
    assert profile.preferences_ut.shape == (86, 11)
    assert len(profile.labels_candidates) == 11
