from PIL import Image
import pandas as pd

from emtk.util import _find_background_color, _get_meta_data, _get_stimuli


def find_aoi(eye_events: pd.DataFrame = pd.DataFrame(),
             eye_tracker_col: str = "eye_tracker",
             stimuli_module_col: str = "stimuli_module",
             stimuli_name_col: str = "stimuli_name", image: Image = None,
             level: str = "sub-line", margin_height: int = 4, margin_width: int = 7) -> pd.DataFrame:
    """Find areas of interest in the stimuli image.

    Parameters
    ----------
    eye_events : pandas.DataFrame, optional (default pandas.DataFrame())
        A pandas dataframe of eye events. It must contains columns that specify the name 
        of the eye tracker, stimuli module, and name of stimuli. If this dataframe is empty,
        image must be specified.

    eye_tracker_col : str, optional (default "eye_tracker")
        Name of the column in eye_events dataframe that contains the name of the eye tracker.

    stimuli_module_col : str, optional (default "stimuli_module")
        Name of the column in eye_events dataframe that contains the path to the stimuli module.

    stimuli_name_col : str, optional (default "stimuli_name")
        Name of the column in eye_events dataframe that contains the name of the stimuli.

    image : PIL.Image, optional (default None)
        Stimuli image. If this is empty, eye_events must be specified.

    level : str, optional (default "sub-line")
        Level of detection in AOIs. 
        "Line" for each line as an AOI or "sub-line" for each token as an AOI.

    margin_height : int, optional (default 4)
        Marginal height when finding AOIs, use smaller number for tight text layout.

    margin_width : int, optional (default 7)
        Marginal width when finding AOIs, use smaller number for tight text layout.

    Returns
    -------
    pandas.DataFrame
        a pandas DataFrame of area of interest detected by the method
    """

    if image is None:

        if eye_events.empty:
            print('Eye event dataframe is empty')
            return

        eye_tracker, stimuli_module, stimuli_name = \
            _get_meta_data(eye_events, eye_tracker_col,
                           stimuli_module_col, stimuli_name_col)

        image = _get_stimuli(stimuli_module, stimuli_name,
                             eye_tracker).convert('1')

    else:
        image = image.convert('1')
        stimuli_name = ''

    width, height = image.size

    # Detect the background color
    bg_color = _find_background_color(image=image)

    left, right = 0, width

    vertical_result, upper_bounds, lower_bounds = [], [], []

    # Move the detecting rectangle from the top to the bottom of the image
    for upper in range(height - margin_height):

        lower = upper + margin_height

        box = (left, upper, right, lower)
        minimum, maximum = image.crop(box).getextrema()

        if upper > 1:
            if bg_color == 'black':
                if vertical_result[-1][3] == 0 and maximum == 255:
                    # Rectangle detects white color for the first time in a while -> Start of one line
                    upper_bounds.append(upper)
                if vertical_result[-1][3] == 255 and maximum == 0:
                    # Rectangle detects black color for the first time in a while -> End of one line
                    lower_bounds.append(lower)
            elif bg_color == 'white':
                if vertical_result[-1][2] == 255 and minimum == 0:
                    # Rectangle detects black color for the first time in a while -> Start of one line
                    upper_bounds.append(upper)
                if vertical_result[-1][2] == 0 and minimum == 255:
                    # Rectangle detects white color for the first time in a while -> End of one line
                    lower_bounds.append(lower)

        # Storing all detection result
        vertical_result.append([upper, lower, minimum, maximum])

    final_result = []

    line_count = 1

    # Iterate through each line of code from detection
    for upper_bound, lower_bound in list(zip(upper_bounds, lower_bounds)):

        # Reset all temporary result for the next line
        horizontal_result, left_bounds, right_bounds = [], [], []

        # Move the detecting rectangle from the left to the right of the image
        for left in range(width - margin_width):

            right = left + margin_width

            box = (left, upper_bound, right, lower_bound)
            minimum, maximum = image.crop(box).getextrema()

            if left > 1:
                if bg_color == 'black':
                    if horizontal_result[-1][3] == 0 and maximum == 255:
                        # Rectangle detects black color for the first time in a while -> Start of one word
                        left_bounds.append(left)
                    if horizontal_result[-1][3] == 255 and maximum == 0:
                        # Rectangle detects white color for the first time in a while -> End of one word
                        right_bounds.append(right)
                elif bg_color == 'white':
                    if horizontal_result[-1][2] == 255 and minimum == 0:
                        # Rectangle detects black color for the first time in a while -> Start of one word
                        left_bounds.append(left)
                    if horizontal_result[-1][2] == 0 and minimum == 255:
                        # Rectangle detects white color for the first time in a while -> End of one word
                        right_bounds.append(right)

            # Storing all detection result
            horizontal_result.append([left, right, minimum, maximum])

        if level == 'sub-line':

            part_count = 1

            for left, right in list(zip(left_bounds, right_bounds)):
                final_result.append(
                    ['sub-line', f'line {line_count} part {part_count}', left, upper_bound, right, lower_bound])
                part_count += 1

        elif level == 'line':
            final_result.append(
                ['line', f'line {line_count}', left_bounds[0], upper_bound, right_bounds[-1], lower_bound])

        line_count += 1

    # Format pandas dataframe
    columns = ['kind', 'name', 'x', 'y', 'width', 'height', 'image']
    aoi = pd.DataFrame(columns=columns)

    for entry in final_result:
        kind, name, x, y, x0, y0 = entry
        width = x0 - x
        height = y0 - y
        image_name = stimuli_name

        # For better visualization
        x += margin_width / 2
        width -= margin_width

        value = [kind, name, x, y, width, height, image_name]
        dic = dict(zip(columns, value))

        aoi = aoi.append(dic, ignore_index=True)

    return aoi
