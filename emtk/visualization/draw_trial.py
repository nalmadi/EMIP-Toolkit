import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import ImageDraw

from emtk.util import _find_background_color, _get_meta_data, _get_stimuli
from emtk.aoi import find_aoi

def __draw_aoi(draw, aoi, bg_color):
    """Private method to draw the Area of Interest on the image

    Parameters
    ----------
    draw : PIL.ImageDraw.Draw
        a Draw object imposed on the image

    aoi : pandas.DataFrame
        a DataFrame that contains the area of interest bounds

    bg_color : str
        background color
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
    

def __draw_fixation(draw, fixations, draw_number=False, 
                    x0_col = "x0", y0_col = "y0", 
                    duration_col = "duration"):
    """Private method that draws the fixation, also allow user to draw eye movement order

    Parameters
    ----------
    draw : PIL.ImageDraw.Draw
        a Draw object imposed on the image

    draw_number : bool
        whether user wants to draw the eye movement number
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


def __draw_saccade(draw, saccades, draw_number=False, 
                x0_col: str = "x0", y0_col: str = "y0", 
                x1_col: str = "x1", y1_col: str = "y1"):
    """

    Parameters
    ----------
    draw : PIL.ImageDraw.Draw
        a Draw object imposed on the image

    draw_number : bool
        whether user wants to draw the eye movement number
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


def __draw_raw_data(draw, samples: pd.DataFrame):
    """Private method that draws raw sample data

    Parameters
    ----------
    draw : PIL.ImageDraw.Draw
        a Draw object imposed on the image
    """

    for _, sample in samples.iterrows():
        # Invalid records
        if len(sample) > 5:
            x_cord = float(sample["R POR X [px]"])
            y_cord = float(sample["R POR Y [px]"])  # - 150

        dot_size = 2

        draw.ellipse((x_cord - (dot_size / 2),
                        y_cord - (dot_size / 2),
                        x_cord + dot_size, y_cord + dot_size),
                        fill=(255, 0, 0, 100))


def draw_trial(eye_events: pd.DataFrame = pd.DataFrame(), samples: pd.DataFrame = pd.DataFrame(),
                draw_raw_data: bool = False, draw_fixation=True, draw_saccade=False, 
                draw_number=False, draw_aoi=None, save_image=None,
                eye_tracker_col: str = "eye_tracker",
                stimuli_module_col: str = "stimuli_module",
                stimuli_name_col: str = "stimuli_name",
                x0_col: str = "x0", y0_col: str = "y0", 
                x1_col: str = "x1", y1_col: str = "y1", duration_col: str = "duration",
                eye_event_type_col: str = "eye_event_type") -> None:

    """Draws the trial image and raw-data/fixations over the image
        circle size indicates fixation duration

    image_path : str
        path for trial image file.

    draw_raw_data : bool, optional
        whether user wants raw data drawn.

    draw_fixation : bool, optional
        whether user wants filtered fixations drawn

    draw_saccade : bool, optional
        whether user wants saccades drawn

    draw_number : bool, optional
        whether user wants to draw eye movement number

    draw_aoi : pandas.DataFrame, optional
        Area of Interests

    save_image : str, optional
        path to save the image, image is saved to this path if it parameter exists
    """

    if eye_events.empty and samples.empty:
        raise Exception('Both eye_events and samples dataframes are empty')

    eye_tracker, stimuli_module, \
        stimuli_name = _get_meta_data(eye_events, eye_tracker_col, 
                                        stimuli_module_col, stimuli_name_col)

    stimuli = _get_stimuli(stimuli_module, stimuli_name, eye_tracker)

    bg_color = _find_background_color(image = stimuli)
    draw = ImageDraw.Draw(stimuli, 'RGBA')

    if draw_aoi:
        aoi = find_aoi(image = stimuli)
        __draw_aoi(draw, aoi, bg_color)

    if draw_raw_data:
        __draw_raw_data(draw, samples)

    if draw_fixation:
        fixations = eye_events.loc[eye_events[eye_event_type_col] == "fixation"]
        __draw_fixation(draw, fixations, draw_number, x0_col, y0_col, duration_col)

    if draw_saccade:
        saccades = eye_events.loc[eye_events[eye_event_type_col] == "saccade"]
        __draw_saccade(draw, saccades, draw_number, x0_col, y0_col, x1_col, y1_col)

    plt.figure(figsize=(17, 15))
    plt.imshow(np.asarray(stimuli), interpolation='nearest')

    if save_image is not None:
        plt.savefig(save_image)
        print(save_image, "saved!")