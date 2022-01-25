import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from emtk.util import _get_meta_data, _get_stimuli

def heatmap(eye_events: pd.DataFrame,
            figsize: tuple[int, int] = (15, 10), color: str = 'r', 
            alpha: float = 0.6, thresh: float = 0.5,
            eye_tracker_col: str = "eye_tracker", 
            x0_col: str = "x0", y0_col: str = "y0",
            stimuli_module_col = "stimuli_module", 
            stimuli_name_col = "stimuli_name", eye_event_type_col = "eye_event_type") -> None:

    eye_tracker, stimuli_module, \
        stimuli_name = _get_meta_data(eye_events, eye_tracker_col, 
                                stimuli_module_col, stimuli_name_col)

    stimuli = _get_stimuli(stimuli_module, stimuli_name, eye_tracker)

    fixations = eye_events.loc[eye_events[eye_event_type_col] == "fixation"]
    x_cords = fixations[x0_col]
    y_cords = fixations[y0_col]

    fig, ax = plt.subplots(figsize=figsize)

    sns.kdeplot( ax=ax, x=x_cords, y=y_cords, 
                 color=color, shade=True, 
                 thresh=thresh, alpha = alpha)

    ax.imshow(stimuli)