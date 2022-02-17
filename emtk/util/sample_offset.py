import pandas as pd


def sample_offset(eye_events: pd.DataFrame, x_offset: float, y_offset: float,
                  x0_col: str = "x0", y0_col: str = "y0",
                  x1_col: str = "x1", y1_col: str = "y1") -> pd.DataFrame:
    """Displace the coordinates of the eye events by specified offset distances.

    Parameters
    ----------
    x_offset : float
        Offset to be applied on all eye events in the x-axis.

    y_offset : float
        Offset to be applied on all eye events in the y-axis.

    Returns
    ----------
    x0_col : str, optional (default to "x0")
        Name of the column in the eye events dataframe that contains the x-coordinates of eye events.
        For saccades, this is the column that contains the starting x-coordinates.

    y0_col : str, optional (default to "y0")
        Name of the column in the eye events dataframe that contains the y-coordinates of eye events.
        For saccades, this is the column that contains the starting x-coordinates.

    x1_col : str, optional (default to "x1")
        Name of the column in the eye events dataframe that contains the ending x-coordinates of eye events.
        Only applicable to saccades.

    y1_col : str, optional (default to "y1")
        Name of the column in the eye events dataframe that contains the ending xy-coordinates of eye events.
        Only applicable to saccades.


    """
    eye_events_copy = eye_events.copy()
    eye_events_copy[x0_col] = eye_events_copy[x0_col] + x_offset
    eye_events_copy[y0_col] = eye_events_copy[y0_col] + y_offset

    if x1_col in eye_events_copy.columns and y1_col in eye_events_copy:
        eye_events_copy[x1_col] = eye_events_copy[x1_col] + x_offset
        eye_events_copy[y1_col] = eye_events_copy[y1_col] + y_offset

    return eye_events_copy
