import pandas as pd
from PIL import Image, ImageDraw

from emtk.aoi import find_aoi
from emtk.util import _find_lines, _line_hit_test, _get_meta_data, _get_stimuli


def fixation_duration(eye_events: pd.DataFrame, width_padding: float = 10,
                      unit_height: float = .5, horizontal_sep: float = 0,
                      image_padding: float = 10,
                      eye_tracker_col: str = "eye_tracker",
                      stimuli_module_col: str = "stimuli_module",
                      stimuli_name_col: str = "stimuli_name",
                      duration_col: str = "duration",
                      y0_col: str = "y0", eye_event_type_col: str = "eye_event_type") -> None:
    '''Draw duration of fixation on each line.
    This function draws a horizontal bar graph of fixation duration on each line in stimuli image 
    on the left of the stimuli image.

    Parameters
    ----------   
    eye_events : pd.DataFrame
        Pandas dataframe for eye events.

    width_padding : float, optional (default 10)
        Difference between the height of each line and width of bar.

    unit_height : float, optional (default .5)
        Height of bar corresponding to one unit of fixation duration.

    horizontal_sep : float, optional (default 0)
        Separation width between bar graph and stimuli image in pixels.

    image_padding : int, optional (default 10)   
        Padding expected around image in pixels.

    eye_tracker_col : str, optional (default "eye_tracker")
        Name of the column in eye_events dataframe that contains the name of the eye tracker.

    stimuli_module_col : str, optional (default "stimuli_module")
        Name of the column in eye_events dataframe that contains the path to the stimuli module.

    stimuli_name_col : str, optional (default "stimuli_name")
        Name of the column in eye_events dataframe that contains the name of the stimuli.

    duration_col : str, optional (default to "duration")
        Name of the column in the eye events dataframe that contains the duration of the eye events.

    y0_col : str, optional (default to "y0")
        Name of the column in the eye events dataframe that contains the y-coordinates of the eye events.

    eye_event_type_col : str, optional (default to "eye_event_type")
        Name of the column in the eye events dataframe that contains the types of the eye events.
    '''

    fixations = eye_events.loc[eye_events[eye_event_type_col] == "fixation"]

    eye_tracker, stimuli_module, \
        stimuli_name = _get_meta_data(fixations, eye_tracker_col,
                                      stimuli_module_col, stimuli_name_col)

    stimuli = _get_stimuli(stimuli_module, stimuli_name, eye_tracker)

    aois = find_aoi(image=stimuli)
    lines_df = _find_lines(aois)
    fixations_by_line = _line_hit_test(lines_df, fixations, y0_col=y0_col)

    # Creates fixation duration bar plots
    plot = Image.new('RGB', stimuli.size, color='white')
    draw = ImageDraw.Draw(plot)

    for _, row in lines_df.iterrows():
        width = row["line_height"] - width_padding

        fixations_on_line = fixations_by_line.loc[fixations_by_line["line_num"]
                                                  == row["line_num"]]
        height = unit_height * fixations_on_line[duration_col].sum()

        x1 = stimuli.size[0] - 10
        x0 = x1 - height
        y0 = row["line_y"] - (width / 2)
        y1 = row["line_y"] + (width / 2)

        draw.rectangle([x0, y0, x1, y1], fill="red")

    # Create combined image
    master_width = plot.size[0] + horizontal_sep + stimuli.size[0]
    master_image = Image.new("RGBA", (master_width, stimuli.size[1]),
                             (255, 255, 255, 255))

    # Paste bar plot (left) and code (right)
    master_image.paste(plot, (0, image_padding))
    master_image.paste(stimuli, (plot.size[0] + horizontal_sep, 0))

    return master_image
