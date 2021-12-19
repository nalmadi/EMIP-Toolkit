import emip_toolkit as emtk
import random

"""
Generates a synthetic set of eye movements
"""

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


def left_of_center(x, y, width, height):
    """ returns left-shifted coordinates

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

    x += width / 3 + random.randint(-10, 10)
    y += height / 2 + random.randint(-10, 10)

    return round(x, 1), y


def is_skipped(token, threshold, skip_probability):
    """ checks if a token should be skipped

    Parameters
    ----------
    token: string
    threshold: int, the max length of a token below (including) which the token should be skipped
    skip_probability: float (0.0-1.0), the probability of a token (the length of which is <= threshold) being skipped

    Returns
    -------
    bool, whether a token should be skipped or not
    """
    if len(token) <= threshold:
        if random.random() <= skip_probability:
            return True
        else: 
            return False
    else:
        return False


def generate_fixations_left(aois_with_tokens):
    """ generate left-shifted fixations

    Parameters
    ----------
    aois_with_tokens: pandas Dataframe
    
    Returns
    -------
    list, a list of fixations
    """

    fixations = []
    
    for i in range(len(aois_with_tokens)):
        x, y, width, height, token = aois_with_tokens['x'][index], aois_with_tokens['y'][index], aois_with_tokens['width'][index], aois_with_tokens['height'][index], aois_with_tokens['token'][index]
        fixation_x, fixation_y = left_of_center(x, y, width, height)
        fixations.append([fixation_x, fixation_y, token])

    return fixations


def generate_fixations_skip(aois_with_tokens, threshold, skip_probability):
    """ generate fixations with skipping

    Parameters
    ----------
    aois_with_tokens: pandas Dataframe
    threshold: int, the max length of a token below (including) which the token should be skipped
    skip_probability: float (0.0-1.0), the probability of a token (the length of which is <= threshold) being skipped

    Returns
    -------
    list, a list of fixations
    """

    fixations = []

    for i in range(len(aois_with_tokens)):
        if is_skipped(aois_with_tokens["token"][i], threshold, skip_probability)==False:
            x, y, width, height, token = aois_with_tokens['x'][i], aois_with_tokens['y'][i], aois_with_tokens['width'][i], aois_with_tokens['height'][i], aois_with_tokens['token'][i]
            fixation_x, fixation_y = left_of_center(x, y, width, height)
            fixations.append([fixation_x, fixation_y, token])
    
    return fixations


def generate_fixations_regression(aois_with_tokens, regression_probability):
    """ generate fixations with regression

    Parameters
    ----------
    aois_with_tokens: pandas Dataframe
    regression_probability: float (0.0-1.0), the probability of regression

    Returns
    -------
    list, a list of fixations
    """

    fixations = []

    index = 0

    while index < len(aois_with_tokens):
        x, y, width, height, token = aois_with_tokens['x'][index], aois_with_tokens['y'][index], aois_with_tokens['width'][index], aois_with_tokens['height'][index], aois_with_tokens['token'][index]
        fixations.append([fixation_x, fixation_y, token])
        
        if random.random() < regression_probability:
            index -= random.randint(1, 10)

            if index < 0:
                index = 0

            x, y, width, height, token = aois_with_tokens['x'][index], aois_with_tokens['y'][index], aois_with_tokens['width'][index], aois_with_tokens['height'][index], aois_with_tokens['token'][index]
            fixations.append([fixation_x, fixation_y, token]) 

        index += 1   

    return fixations


def generate_fixations_left_regression(aois_with_tokens, regression_probability):
    """ generate fixations with left-shifts and regression
    (modified from Dr. Naser Al Madi's code)

    Parameters
    ----------
    aois_with_tokens: pandas Dataframe
    regression_probability: float (0.0-1.0), the probability of regression

    Returns
    -------
    list, a list of fixations
    """

    fixations = []
    
    index = 0

    while index < len(aois_with_tokens):
        x, y, width, height, token = aois_with_tokens['x'][index], aois_with_tokens['y'][index], aois_with_tokens['width'][index], aois_with_tokens['height'][index], aois_with_tokens['token'][index]
        fixation_x, fixation_y = left_of_center(x, y, width, height)
        fixations.append([fixation_x, fixation_y, token])
        
        if random.random() < regression_probability:
            index -= random.randint(1, 10)

            if index < 0:
                index = 0

            x, y, width, height, token = aois_with_tokens['x'][index], aois_with_tokens['y'][index], aois_with_tokens['width'][index], aois_with_tokens['height'][index], aois_with_tokens['token'][index]
            fixation_x, fixation_y = left_of_center(x, y, width, height)
            fixations.append([fixation_x, fixation_y, token]) 

        index += 1   

    return fixations


def generate_fixations_left_regression_skip(aois_with_tokens, regression_probability, threshold, skip_probability):
    """ generate fixations with left-shifts, regression, and skipping

    Parameters
    ----------
    aois_with_tokens: pandas Dataframe
    regression_probability: float (0.0-1.0), the probability of regression
    threshold: int, the max length of a token below (including) which the token should be skipped
    skip_probability: float (0.0-1.0), the probability of a token (the length of which is <= threshold) being skipped

    Returns
    -------
    list, a list of fixations
    """

    fixations = []
    
    index = 0

    while index < len(aois_with_tokens):
        if is_skipped(aois_with_tokens["token"][i], threshold, skip_probability)==False:
            x, y, width, height, token = aois_with_tokens['x'][index], aois_with_tokens['y'][index], aois_with_tokens['width'][index], aois_with_tokens['height'][index], aois_with_tokens['token'][index]
            fixation_x, fixation_y = left_of_center(x, y, width, height)
            fixations.append([fixation_x, fixation_y, token])
        
        if random.random() < regression_probability:
            index -= random.randint(1, 10)

            if index < 0:
                index = 0

            if is_skipped(aois_with_tokens["token"][i], threshold, skip_probability)==False:
                x, y, width, height, token = aois_with_tokens['x'][index], aois_with_tokens['y'][index], aois_with_tokens['width'][index], aois_with_tokens['height'][index], aois_with_tokens['token'][index]
                fixation_x, fixation_y = left_of_center(x, y, width, height)
                fixations.append([fixation_x, fixation_y, token]) 

        index += 1   

    return fixations