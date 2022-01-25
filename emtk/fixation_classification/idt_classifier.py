import math
import statistics

def idt_classifier(raw_fixations, minimum_duration=50, sample_duration=4, maximum_dispersion=25):
    """I-DT classifier based on page 296 of eye tracker manual:
        https://psychologie.unibas.ch/fileadmin/user_upload/psychologie/Forschung/N-Lab/SMI_iView_X_Manual.pdf

        Notes:
            remember that some data is MSG for mouse clicks.
            some records are invalid with value -1.
            read right eye data only.

    Parameters
    ----------
    raw_fixations : list
        a list of fixations information containing timestamp, x_cord, and y_cord

    minimum_duration : int, optional
        minimum duration for a fixation in milliseconds, less than minimum is considered noise.
        set to 50 milliseconds by default

    sample_duration : int, optional
        Sample duration in milliseconds, this is 4 milliseconds based on this eye tracker

    maximum_dispersion : int, optional
        maximum distance from a group of samples to be considered a single fixation.
        Set to 25 pixels by default

    Returns
    -------
    list
        a list where each element is a list of timestamp, duration, x_cord, and y_cord
    """

    # Create moving window based on minimum_duration
    window_size = int(math.ceil(minimum_duration / sample_duration))

    window_x = []
    window_y = []

    filter_fixation = []

    # Go over all SMPs in trial data
    for timestamp, x_cord, y_cord in raw_fixations:

        # Filter (skip) coordinates outside of the screen 1920×1080 px
        if x_cord < 0 or y_cord < 0 or x_cord > 1920 or y_cord > 1080:
            continue

        # Add sample if it appears to be valid
        window_x.append(x_cord)
        window_y.append(y_cord)

        # Calculate dispersion = [max(x) - min(x)] + [max(y) - min(y)]
        dispersion = (max(window_x) - min(window_x)) + (max(window_y) - min(window_y))

        # If dispersion is above maximum_dispersion
        if dispersion > maximum_dispersion:

            # Then the window does not represent a fixation
            # Pop last item in window
            window_x.pop()
            window_y.pop()

            # Add fixation to fixations if window is not empty (size >= window_size)
            if len(window_x) == len(window_y) and len(window_x) > window_size:
                # The fixation is registered at the centroid of the window points
                filter_fixation.append(
                    [timestamp, len(window_x) * 4, statistics.mean(window_x), statistics.mean(window_y)])

            window_x = []
            window_y = []

    return filter_fixation