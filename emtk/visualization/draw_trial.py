import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import ImageDraw

from emtk.util import _find_background_color, _get_meta_data, _get_stimuli
from emtk.aoi import find_aoi


def __draw_aoi(draw: ImageDraw.Draw, aoi: pd.DataFrame, bg_color: str) -> None:
    """Draw areas of interest on stimuli image.

    Parameters
    ----------
    draw : PIL.ImageDraw.Draw
        Pillow Draw object imposed on stimuli image.

    aoi : pandas.DataFrame
        Pandas DataFrame of areas of interest.

    bg_color : str
        Background color of stimuli image.
    """

    outline = {'white': '#000000', 'black': '#ffffff'}

    for row in aoi[['x', 'y', 'width', 'height']].iterrows():
        y_coordinate = row[1]['y']
        x_coordinate = row[1]['x']
        height = row[1]['height']
        width = row[1]['width']
        draw.rectangle([(x_coordinate, y_coordinate),
                        (x_coordinate + width - 1, y_coordinate + height - 1)],
                       outline=outline[bg_color])

    return None


def __draw_fixation(draw: ImageDraw.Draw, fixations: pd.DataFrame, draw_number: bool = False,
                    x0_col: str = "x0", y0_col: str = "y0", duration_col: str = "duration") -> None:
    """Draw fixations with their respective orders of appearance.

    Parameters
    ----------
    draw : PIL.ImageDraw.Draw
        Draw object imposed on stimuli image.

    fixations: pandas.DataFrame
        Pandas dataframe of fixations.

    draw_number : bool
        Indicate whether user wants to draw the orders of appearance of fixations.

    x0_col : str, optional (default to "x0")
        Name of the column in the fixations dataframe that contains the x-coordinates of fixations.

    y0_col : str, optional (default to "y0")
        Name of the column in the fixations dataframe that contains the y-coordinates of fixations.

    duration_col : str, optional (default to "duration")
        Name of the column in the fixations dataframe that contains the duration of fixations.
    """

    for count, fixation in fixations.iterrows():
        _duration = fixation[duration_col]
        if 5 * (_duration / 100) < 5:
            r = 3
        else:
            r = 5 * (_duration / 100)

        x = fixation[x0_col]
        y = fixation[y0_col]

        bound = (x - r, y - r, x + r, y + r)
        outline_color = (255, 255, 0, 0)
        fill_color = (242, 255, 0, 128)
        draw.ellipse(bound, fill=fill_color, outline=outline_color)

        if draw_number:
            text_bound = (x, y - r / 2)
            text_color = (255, 0, 0, 225)
            draw.text(text_bound, str(count + 2), fill=text_color)

    return None


def __draw_saccade(draw: ImageDraw.Draw, saccades: pd.DataFrame, draw_number: bool = False,
                   x0_col: str = "x0", y0_col: str = "y0",
                   x1_col: str = "x1", y1_col: str = "y1") -> None:
    """Draw saccades with their respective orders of appearance.

    Parameters
    ----------
    draw : PIL.ImageDraw.Draw
        Draw object imposed on stimuli image.

    saccades: pandas.DataFrame
        Pandas dataframe of saccades.

    draw_number : bool
        Indicate whether user wants to draw the orders of appearance of saccades.

    x0_col : str, optional (default to "x0")
        Name of the column in the saccades dataframe that contains the starting x-coordinates of saccades.

    y0_col : str, optional (default to "y0")
        Name of the column in the saccades dataframe that contains the starting y-coordinates of saccades.

    x1_col : str, optional (default to "x1")
        Name of the column in the saccades dataframe that contains the ending x-coordinates of saccades.

    y1_col : str, optional (default to "y1")
        Name of the column in the saccades dataframe that contains the ending y-coordinates of saccades.
    """
    for count, saccade in saccades.iterrows():
        x0 = saccade[x0_col]
        y0 = saccade[y0_col]
        x1 = saccade[x1_col]
        y1 = saccade[y1_col]

        bound = (x0, y0, x1, y1)
        line_color = (122, 122, 0, 255)
        penwidth = 2
        draw.line(bound, fill=line_color, width=penwidth)

        if draw_number:
            text_bound = ((x0 + x1) / 2, (y0 + y1) / 2)
            text_color = 'darkred'
            draw.text(text_bound, str(count + 2), fill=text_color)


def __draw_raw_data(draw: ImageDraw.Draw, samples: pd.DataFrame, sample_x_col, sample_y_col) -> None:
    """Draw raw sample data.

    Parameters
    ----------
    draw : PIL.ImageDraw.Draw
        Draw object imposed on stimuli image.

    samples: pandas.DataFrame
        Pandas dataframe of raw samples.
    """

    for _, sample in samples.iterrows():
        # Invalid records
        if len(sample) > 5:
            x_cord = float(sample[sample_x_col])
            y_cord = float(sample[sample_y_col])  # - 150
        dot_size = 2

        draw.ellipse((x_cord - (dot_size / 2),
                      y_cord - (dot_size / 2),
                      x_cord + dot_size, y_cord + dot_size),
                     fill=(255, 0, 0, 100))


def draw_trial(eye_events: pd.DataFrame = pd.DataFrame(), samples: pd.DataFrame = pd.DataFrame(),
               draw_raw_data: bool = False, draw_fixation: bool = True, draw_saccade: bool = False,
               draw_number: bool = False, draw_aoi: bool = False, save_image: str = None,
               eye_tracker_col: str = "eye_tracker",
               stimuli_module_col: str = "stimuli_module",
               stimuli_name_col: str = "stimuli_name",
               x0_col: str = "x0", y0_col: str = "y0",
               x1_col: str = "x1", y1_col: str = "y1", duration_col: str = "duration",
               eye_event_type_col: str = "eye_event_type",
               sample_x_col: int = "x", sample_y_col: str = "y") -> None:
    """Draw raw data samples, fixations, and saccades over simuli images image
    Circle size indicates fixation duration.

    Parameters
    ----------   
    eye_events : pd.DataFrame
        Pandas dataframe for eye events.

    samples : pd.DataFrame
        Pandas dataframe for samples.

    draw_raw_data : bool, optional (default False)
        whether user wants raw data drawn.

    draw_fixation : bool, optional (default True)
        whether user wants fixations drawn

    draw_saccade : bool, optional (default False)
        whether user wants saccades drawn

    draw_number : bool, optional (default False)
        whether user wants to draw eye movement number

    draw_aoi : bool, optional (default False)
        whether user wants to draw eye movement number

    save_image : str, optional (default None)
        path to save the image, image is saved to this path if it parameter exists
    """

    if eye_events.empty and samples.empty:
        raise Exception('Both eye_events and samples dataframes are empty')

    metadata_df = samples if eye_events.empty else eye_events
    eye_tracker, stimuli_module, \
        stimuli_name = _get_meta_data(metadata_df, eye_tracker_col,
                                      stimuli_module_col, stimuli_name_col)

    stimuli = _get_stimuli(stimuli_module, stimuli_name, eye_tracker)

    bg_color = _find_background_color(image=stimuli)
    draw = ImageDraw.Draw(stimuli, 'RGBA')

    if draw_aoi:
        aoi = find_aoi(image=stimuli)
        __draw_aoi(draw, aoi, bg_color)

    if draw_raw_data:
        __draw_raw_data(draw, samples, sample_x_col, sample_y_col)

    if draw_fixation:
        fixations = eye_events.loc[eye_events[eye_event_type_col]
                                   == "fixation"]
        __draw_fixation(draw, fixations, draw_number,
                        x0_col, y0_col, duration_col)

    if draw_saccade:
        saccades = eye_events.loc[eye_events[eye_event_type_col] == "saccade"]
        __draw_saccade(draw, saccades, draw_number,
                       x0_col, y0_col, x1_col, y1_col)

    plt.figure(figsize=(17, 15))
    plt.imshow(np.asarray(stimuli), interpolation='nearest')

    if save_image is not None:
        plt.savefig(save_image)
        print(save_image, "saved!")
