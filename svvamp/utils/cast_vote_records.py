import numpy as np
import pandas as pd
from svvamp.utils.misc import preferences_ut_to_preferences_borda_ut


def cvr_to_preferences_ut(file_name):
    df = pd.read_csv(filepath_or_buffer=file_name, sep=',', index_col=False)
    last_rank = int(df.columns[-1][4:])  # E.g. if the last column is named 'rank3' => last_rank = 3.
    # Drop useless columns
    df.drop(columns=df.columns[:-last_rank], inplace=True)
    # Replace special values with NaN
    df.replace(to_replace=r'^(WRITE-IN)|(writein)|(Write-In)|(Write-in)|(skipped)|(overvote).*',
               value=np.nan, regex=True, inplace=True)
    # Compute the set of candidates
    set_candidates = set()
    for rank in range(1, last_rank + 1):
        column_name = 'rank{}'.format(rank)
        candidates_in_column = df[column_name].unique()
        set_candidates.update(candidates_in_column)
    if np.nan in set_candidates:
        set_candidates.remove(np.nan)
    labels_candidates = list(set_candidates)
    d_candidate_index = {candidate: index for index, candidate in enumerate(labels_candidates)}
    n_c = len(labels_candidates)
    # Utilities
    df.replace(d_candidate_index, inplace=True)
    preferences_ut = []
    for index, row in df.iterrows():
        if np.all(np.isnan(row)):
            # Ignore this voter (abstains)
            continue
        voter_utilities = np.zeros(n_c, dtype=int)
        for rank in range(1, last_rank + 1):
            column_name = 'rank{}'.format(rank)
            candidate = row[column_name]
            if not np.isnan(candidate):
                voter_utilities[int(candidate)] = n_c - rank
        preferences_ut.append(voter_utilities)
    np.array(preferences_ut)
    # Convert to Borda format
    preferences_ut = preferences_ut_to_preferences_borda_ut(preferences_ut) - (n_c - 1) / 2
    # Return
    return preferences_ut, labels_candidates
