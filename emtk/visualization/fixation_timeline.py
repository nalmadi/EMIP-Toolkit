from matplotlib import pyplot as plt
import pandas as pd

from emtk.aoi import find_aoi
from emtk.util import _find_lines, _line_hit_test, _get_meta_data, _get_stimuli

def fixation_timeline(eye_events: pd.DataFrame,
                    eye_tracker_col: str = "eye_tracker",
                    stimuli_module_col: str = "stimuli_module",
                    stimuli_name_col: str = "stimuli_name",
                    timestamp_col: str = "timestamp",
                    y0_col: str = "y0",
                    eye_event_type_col: str = "eye_event_type",
                    figsize: tuple = (15, 10)) -> None:

    fixations = eye_events.loc[eye_events[eye_event_type_col] == "fixation"]

    eye_tracker, stimuli_module, \
        stimuli_name = _get_meta_data(eye_events, eye_tracker_col, 
                                stimuli_module_col, stimuli_name_col)

    stimuli = _get_stimuli(stimuli_module, stimuli_name, eye_tracker)

    aois = find_aoi(image = stimuli)
    lines_df = _find_lines(aois)
    fixations_by_line = _line_hit_test(lines_df, fixations, y0_col = y0_col)

    fixations_by_line["start_time"] =  fixations_by_line[timestamp_col] - \
                                        fixations_by_line[timestamp_col].min() 
    
    fixations_by_line = fixations_by_line.sort_values("start_time")

    plt.figure(figsize=figsize)
    plt.plot(fixations_by_line["start_time"], fixations_by_line["line_num"])


    