import pandas as pd
import numpy as np

def _line_hit_test(lines: pd.DataFrame, fixations: pd.DataFrame,
                  line_num: str = "line_num", line_y: str = "line_y", 
                  line_height: str = "line_height",
                  y0_col: str = "y0") -> pd.DataFrame:

    fixations_copy = fixations.copy()
    fixations_copy["line_num"] = np.nan

    for index, fixation in fixations_copy.iterrows():
        for _, line in lines.iterrows():
            if fixation[y0_col] >= line[line_y] - line[line_height] / 2 and \
               fixation[y0_col] <= line[line_y] + line[line_height] / 2:
                fixations_copy.at[index, "line_num"] = line[line_num]
                break

    fixations_copy.dropna(axis = 0, inplace = True, subset= ["line_num"])
    fixations_copy["line_num"] = fixations_copy["line_num"].astype(int)
    return fixations_copy