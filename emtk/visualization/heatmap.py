import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from emtk.util import _get_meta_data, _get_stimuli


def heatmap(eye_events: pd.DataFrame,
            figsize: tuple[int, int] = (15, 10), color: str = 'r',
            alpha: float = .6, thresh: float = .5,
            eye_tracker_col: str = "eye_tracker",
            x0_col: str = "x0", y0_col: str = "y0",
            stimuli_module_col="stimuli_module",
            stimuli_name_col="stimuli_name", eye_event_type_col="eye_event_type") -> None:
    '''Draw a heatmap to show where the fixations focus on the stimuli image.

    Parameters
    ----------
    eye_events : pd.DataFrame
        Pandas dataframe for eye events.

    figsize : tuple[int], optional (deafault (15, 10))
        Size of the plot.

    color : str, optional (default "r")
        Color of the heatmap. This will be passed as the color argument to sns.kdeplot.

    alpha : float in [0, 1], optional (deafault .6)
        Opacity level of heatmap. This will be passed as the alpha argument to sns.kdeplot.

    thresh : float in [0, 1], optional (deafault .5)
        Lowest iso-proportion level at which to draw a contour line.

    x0_col : str, optional (default "x0")
        Name of the column in the eye events dataframe that contains the x-coordinates of the eye events.

    y0_col : str, optional (default "y0")
        Name of the column in the eye events dataframe that contains the y-coordinates of the eye events.

    stimuli_module_col : str, optional (default "stimuli_module")
        Name of the column in eye_events dataframe that contains the path to the stimuli module.

    stimuli_name_col : str, optional (default "stimuli_name")
        Name of the column in eye_events dataframe that contains the name of the stimuli.

    eye_event_type_col : str, optional (default "eye_event_type")
        Name of the column in the eye events dataframe that contains the types of the eye events.
    '''

    eye_tracker, stimuli_module, \
        stimuli_name = _get_meta_data(eye_events, eye_tracker_col,
                                      stimuli_module_col, stimuli_name_col)

    stimuli = _get_stimuli(stimuli_module, stimuli_name, eye_tracker)

    fixations = eye_events.loc[eye_events[eye_event_type_col] == "fixation"]
    x_cords = fixations[x0_col]
    y_cords = fixations[y0_col]

    _, ax = plt.subplots(figsize=figsize)

    sns.kdeplot(ax=ax, x=x_cords, y=y_cords,
                color=color, shade=True,
                thresh=thresh, alpha=alpha)

    ax.imshow(stimuli)
