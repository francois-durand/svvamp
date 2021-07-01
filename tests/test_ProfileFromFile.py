from svvamp import ProfileFromFile
import os


def test_from_csv():
    file = os.path.join(os.path.dirname(__file__), 'example_ballots.csv')
    profile = ProfileFromFile(file, sort_voters=False)
    assert profile.preferences_ut.shape == (86, 11)
    assert len(profile.labels_candidates) == 11


def test_from_csv_transposed():
    file = os.path.join(os.path.dirname(__file__), 'example_ballots.t.csv')
    profile = ProfileFromFile(file, sort_voters=False)
    assert profile.preferences_ut.shape == (86, 11)
    assert len(profile.labels_candidates) == 11


def test_from_preflib():
    file = os.path.join(os.path.dirname(__file__), 'ED-00001-00000001.soi')
    profile = ProfileFromFile(file)
    assert profile.preferences_ut.shape == (43942, 12)
    assert profile.labels_candidates == [
        'Cathal Boland F.G.', 'Clare Daly S.P.', 'Mick Davis S.F.', 'Jim Glennon F.F.', 'Ciaran Goulding Non-P',
        'Michael Kennedy F.F.', 'Nora Owen F.G.', 'Eamonn Quinn Non-P', 'Sean Ryan Lab', 'Trevor Sargent G.P.',
        'David Henry Walshe C.C. Csp', 'G.V. Wright F.F.']


def test_from_preflib_with_ties():
    file = os.path.join(os.path.dirname(__file__), 'ED-00017-00000001.toi')
    profile = ProfileFromFile(file)
    assert profile.preferences_ut.shape == (4189, 4)
