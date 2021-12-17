import emip_toolkit as emtk
import random

"""
Generates a synthetic set of eye movements for rectangle
"""

eye_movements = []

# fixation: x_cord, y_cord, token

def rectangle_center(x, y, width, height):
    """ finds the center of a rectangle around a token

    Parameters
    ----------
    x : float, x-coordinate of the upper left corner of a rectangle
    y : float, y-coordinate of the upper left corner of a rectangle
    width: width of a rectangle
    height: height of a rectangle

    Returns
    -------
    int tuple, a tuple of x- and y-coordinates
    """

    center_x = x + width / 2
    center_y = y + height / 2
    return center_x, center_y


def left_of_rec_center(center_x, center_y, x_offset):
    """ returns the left of the center of a rectangle by the value of x_offset
    """
    return center_x - x_offset, center_y

def is_skipped(token, threshold, probability):
    """ checks if a token should be skipped

    Parameters
    ----------
    token: string
    threshold: int, the max length of a token below (including) which the token should be skipped
    probability: float (0.0-1.0), the probability of a token (the length of which is <= threshold) being skipped

    Returns
    -------
    bool, whether a token should be skipped or not
    """
    if len(token) <= threshold:
        if random.random() <= probability:
            return True
        else: 
            return False
    else:
        return False
