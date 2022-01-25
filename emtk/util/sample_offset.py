import pandas as pd

def sample_offset(eye_events: pd.DataFrame, x_offset: float, y_offset: float, 
                    x0_col: str = "x0", y0_col: str = "y0",
                    x1_col: str = "x1", y1_col: str = "y1"):
    """Returns the x and y coordinate of the fixation

    Parameters
    ----------
    x_offset : float
        offset to be applied on all fixations in the x-axis

    y_offset : float
        offset to be applied on all fixations in the y-axis
    """
    eye_events_copy = eye_events.copy()
    eye_events_copy[x0_col] = eye_events_copy[x0_col] + x_offset
    eye_events_copy[y0_col] = eye_events_copy[y0_col] + y_offset

    if x1_col in eye_events_copy.columns and y1_col in eye_events_copy:
        eye_events_copy[x1_col] = eye_events_copy[x1_col] + x_offset
        eye_events_copy[y1_col] = eye_events_copy[y1_col] + y_offset

    return eye_events_copy
