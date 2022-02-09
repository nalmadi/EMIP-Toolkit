import pandas as pd
import numpy as np


def _line_hit_test(lines: pd.DataFrame, fixations: pd.DataFrame,
                   line_num: str = "line_num", line_y: str = "line_y",
                   line_height: str = "line_height",
                   y0_col: str = "y0") -> pd.DataFrame:
    '''Matches fixations with their respective lines.
    A fixation is matched with a line if its coordinate is within 
    the x- and y- boundary of the line.

    Parameters
    ----------
    lines : pandas.DataFrame
        Pandas dataframe of lines.

    fixations : pandas.DataFrame
        Pandas dataframe of fixations.

    line_num : str, optional (default to "line_num")
        Name of the column in the lines dataframe that contains the lines' number.

    line_y : str, optional (default to "line_y")
        Name of the column in the lines dataframe that contains the y-coordinate of the lines.

    line_height : str, optional (default to "line_height")
        Name of the column in the lines dataframe that contains the height of the lines.

    y0_col : str, optional (default to "y0")
        Name of the column in the fixations dataframe that contains the y-coordinates of the line.

    Returns
    -------
    pandas.DataFrame
        Pandas dataframe that matches fixation with their respective lines.
    '''

    fixations_copy = fixations.copy()
    fixations_copy["line_num"] = np.nan

    for index, fixation in fixations_copy.iterrows():
        for _, line in lines.iterrows():
            if fixation[y0_col] >= line[line_y] - line[line_height] / 2 and \
               fixation[y0_col] <= line[line_y] + line[line_height] / 2:
                fixations_copy.at[index, "line_num"] = line[line_num]
                break

    fixations_copy.dropna(axis=0, inplace=True, subset=["line_num"])
    fixations_copy["line_num"] = fixations_copy["line_num"].astype(int)
    return fixations_copy
