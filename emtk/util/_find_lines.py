import pandas as pd
import re


def _find_lines(aois: pd.DataFrame) -> pd.DataFrame:
    '''Return a dataframe of lines from a dataframe of AOIs.

    Parameters
    ----------
    aois : pandas.DataFrame
        Pandas dataframe of AOIs.

    Returns
    -------
    pandas.DataFrame
        Color of the background of the image. "Black" or "white".
    '''

    results = pd.DataFrame({
        'line_num': pd.Series(dtype='int'),
        'line_y': pd.Series(dtype='float'),
        'line_height': pd.Series(dtype='float')})

    for _, row in aois.iterrows():
        name, y, height = row["name"], row["y"], row["height"]
        line_num = re.search('\d+', name).group(0)

        results = results.append({
            "line_num": int(line_num),
            "line_y": y + height / 2,
            "line_height": height,
        }, ignore_index=True)

    results = results.drop_duplicates(subset="line_num")
    return results
