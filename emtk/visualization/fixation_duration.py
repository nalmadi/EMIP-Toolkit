import pandas as pd
from PIL import Image, ImageDraw

from emtk.aoi import find_aoi
from emtk.util import _find_lines, _line_hit_test, _get_meta_data, _get_stimuli


def fixation_duration(eye_events: pd.DataFrame, width_padding: int = 10, 
                        unit_height: int = .5, horiz_sep: int = 0, 
                        image_padding: int = 10,
                        eye_tracker_col: str = "eye_tracker",
                        stimuli_module_col: str = "stimuli_module",
                        stimuli_name_col: str = "stimuli_name",
                        duration_col: str = "duration", 
                        y0_col: str = "y0", eye_event_type_col: str = "eye_event_type") -> None:

    fixations = eye_events.loc[eye_events[eye_event_type_col] == "fixation"]

    eye_tracker, stimuli_module, \
        stimuli_name = _get_meta_data(fixations, eye_tracker_col, 
                                stimuli_module_col, stimuli_name_col)

    stimuli = _get_stimuli(stimuli_module, stimuli_name, eye_tracker)

    aois = find_aoi(image = stimuli)
    lines_df = _find_lines(aois)
    fixations_by_line = _line_hit_test(lines_df, fixations, y0_col = y0_col)

    # Creates fixation duration bar plots
    plot = Image.new('RGB', stimuli.size, color='white')
    draw = ImageDraw.Draw(plot)

    for _, row in lines_df.iterrows():
        width = row["line_height"] - width_padding

        fixations_on_line = fixations_by_line.loc[fixations_by_line["line_num"] == row["line_num"]]
        height = unit_height * fixations_on_line[duration_col].sum()

        x1 = stimuli.size[0] - 10
        x0 = x1 - height
        y0 = row["line_y"] - (width / 2)
        y1 = row["line_y"] + (width / 2)

        draw.rectangle([x0, y0, x1, y1], fill = "red")

    # Create combined image
    master_width = plot.size[0] + horiz_sep + stimuli.size[0]
    master_image = Image.new("RGBA", (master_width, stimuli.size[1]),
                             (255, 255, 255, 255))

    # Paste bar plot (left) and code (right)
    master_image.paste(plot, (0, image_padding))
    master_image.paste(stimuli, (plot.size[0] + horiz_sep, 0))

    return master_image

