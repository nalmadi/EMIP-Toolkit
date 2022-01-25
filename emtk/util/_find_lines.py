import pandas as pd
import re

def _find_lines(aois: pd.DataFrame) -> pd.DataFrame:
    ''' returns a list of line Ys '''
    
    results = pd.DataFrame({
                'line_num': pd.Series(dtype='int'),
                'line_y': pd.Series(dtype='float'),
                'line_height': pd.Series(dtype='float')})
    
    for _ , row in aois.iterrows():
        name, y, height = row["name"], row["y"], row["height"]
        line_num = re.search('\d+', name).group(0)
        
        results = results.append({
            "line_num": int(line_num),
            "line_y": y + height / 2,
            "line_height": height,
        }, ignore_index = True)

    results = results.drop_duplicates(subset = "line_num")

    # turn results["line_num"] to integer series
    return results