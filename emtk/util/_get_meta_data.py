import pandas as pd


def _get_meta_data(eye_events: pd.DataFrame,
                   eye_tracker_col: str = "eye_tracker",
                   stimuli_module_col: str = "stimuli_module",
                   stimuli_name_col: str = "stimuli_name") -> tuple:
    '''Retrieve name of eye tracker, path to stimuli folder of the experiment,
    and name of stimuli from dataframe of eye events.

    Parameters
    ----------
    eye_events : pandas.DataFrame
        Pandas dataframe of eye events.

    eye_tracker_col : str, optional (default "eye_tracker")
        Name of the column in eye_events dataframe that contains the name of the eye tracker.

    stimuli_module_col : str, optional (default "stimuli_module")
        Name of the column in eye_events dataframe that contains the path to the stimuli module.

    stimuli_name_col : str, optional (default "stimuli_name")
        Name of the column in eye_events dataframe that contains the name of the stimuli.

    Returns
    -------
    tuple
        Name of eye tracker, path to stimuli folder of the experiment, and name of stimuli.
    '''

    col_names = [eye_tracker_col, stimuli_module_col, stimuli_name_col]

    for col in col_names:
        if len(eye_events[col].unique()) > 1:
            raise Exception("Error, there are more than " +
                            "one unique value in {col} column".format(col=col))

    return (eye_events[col].unique()[0] for col in col_names)
